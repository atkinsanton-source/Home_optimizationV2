from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from home_energy_opt.config import EnergySystemConfig

_GUROBI_API: Optional[Dict[str, Any]] = None
# Treat smaller clamp deltas as pure floating-point noise.
EV_SOC_CLAMP_EPS_KWH = 1e-5
def _load_gurobi_api() -> Dict[str, Any]:
    """Lazy-load gurobipy once and cache references for fast reuse."""
    global _GUROBI_API
    if _GUROBI_API is not None:
        return _GUROBI_API

    t0 = perf_counter()
    import gurobipy as gp  # type: ignore
    from gurobipy import GRB  # type: ignore

    _GUROBI_API = {
        "gp": gp,
        "GRB": GRB,
        "import_seconds": perf_counter() - t0,
    }
    return _GUROBI_API


def _effective_ev_lb(cfg: EnergySystemConfig, reserve_kwh: float) -> float:
    """Single EV lower bound per step from provided reserve, bounded to physical limits."""
    reserve_bounded = min(max(float(reserve_kwh), 0.0), cfg.ev_cap_kwh)
    return reserve_bounded


def _scale_split_to_total(home_kwh: float, external_kwh: float, total_kwh: float) -> Tuple[float, float]:
    """Scale a split state pair to a new total while keeping the split ratio."""
    home_kwh = max(0.0, float(home_kwh))
    external_kwh = max(0.0, float(external_kwh))
    target_total = max(0.0, float(total_kwh))
    raw_total = home_kwh + external_kwh
    if raw_total <= 0.0:
        return target_total, 0.0
    if abs(raw_total - target_total) <= EV_SOC_CLAMP_EPS_KWH:
        return home_kwh, external_kwh
    scale = target_total / raw_total
    return home_kwh * scale, external_kwh * scale


@dataclass
class MPCSolveResult:
    """Container for one MPC window solve outcome."""

    first_step: Dict[str, float]
    status: str
    full_solution: Optional[Dict[str, List[float]]] = None


class WindowDataManager:
    @staticmethod
    def get_arrays(df: pd.DataFrame, i: int, H: int, cfg: EnergySystemConfig) -> Dict[str, List[float]]:
        """Extract a fixed-size MPC window and pad tail steps when needed."""
        window = df.iloc[i : i + H]
        if window.empty:
            raise ValueError("Cannot build MPC window from empty dataframe")

        pad_rows = H - len(window)
        if pad_rows > 0:
            tail = window.iloc[-1:].copy()
            # Tail padding should not force extra driving depletion.
            tail["ev_drive_kwh"] = 0.0
            # Use neutral economics for synthetic rows to reduce end-of-horizon artifacts.
            if cfg.pad_tail_neutral_prices:
                for col in (
                    "import_price_eur_per_kwh",
                    "home_grid_price_eur_per_kwh",
                    "ev_home_import_price_eur_per_kwh",
                    "ev_ext_import_price_eur_per_kwh",
                    "ev_export_price_eur_per_kwh",
                    "ev_home_export_price_eur_per_kwh",
                    "ev_external_export_price_eur_per_kwh",
                ):
                    if col in tail.columns:
                        tail[col] = float(cfg.pad_tail_price_eur_per_kwh)
            window = pd.concat([window, pd.concat([tail] * pad_rows, ignore_index=False)], axis=0)

        ev_reserve_kwh = window["ev_reserve_kwh"].astype(float).tolist()
        # Terminal reserve corresponds to state E_ev[H] (one step beyond control horizon).
        # When i+H is unavailable at dataset end, use 0.0 so terminal state is only physically bounded.
        if i + H < len(df):
            ev_reserve_kwh.append(float(df.iloc[i + H]["ev_reserve_kwh"]))
        else:
            ev_reserve_kwh.append(0.0)

        return {
            "load_kw": window["load_kw"].astype(float).tolist(),
            "home_import_price": window.get("home_grid_price_eur_per_kwh", window["import_price_eur_per_kwh"]).astype(float).tolist(),
            "ev_home_import_price": window.get("ev_home_import_price_eur_per_kwh", window["import_price_eur_per_kwh"]).astype(float).tolist(),
            "ev_drive_kwh": window["ev_drive_kwh"].astype(float).tolist(),
            "ev_reserve_kwh": ev_reserve_kwh,
            "ev_p_home_ch_max_kw": window["ev_p_home_ch_max_kw"].astype(float).tolist(),
            "ev_p_ext_ch_max_kw": window["ev_p_ext_ch_max_kw"].astype(float).tolist(),
            "ev_p_ch_total_max_kw": window["ev_p_ch_total_max_kw"].astype(float).tolist(),
            "ev_p_dis_house_max_kw": window["ev_p_dis_house_max_kw"].astype(float).tolist(),
            "ev_p_dis_grid_max_kw": window["ev_p_dis_grid_max_kw"].astype(float).tolist(),
            "ev_p_dis_total_max_kw": window["ev_p_dis_total_max_kw"].astype(float).tolist(),
            "ev_ext_import_price": window["ev_ext_import_price_eur_per_kwh"].astype(float).tolist(),
            "ev_home_export_price": window["ev_home_export_price_eur_per_kwh"].astype(float).tolist(),
            "ev_external_export_price": window["ev_external_export_price_eur_per_kwh"].astype(float).tolist(),
        }


class PersistentMPCSolver:
    """Persistent Gurobi model for v1-v3 with mutable RHS/UB updates per MPC step."""

    def __init__(self, cfg: EnergySystemConfig, H: int = 96, solver_tee: bool = False):
        self.cfg = cfg
        self.H = H
        self.build_once_time_sec = 0.0
        self.last_update_time_sec = 0.0
        self.last_optimize_time_sec = 0.0
        self.last_extract_time_sec = 0.0
        self._latest_arrays: Dict[str, List[float]] = {}

        api = _load_gurobi_api()
        self.gp = api["gp"]
        self.GRB = api["GRB"]

        t0 = perf_counter()
        self.model = self.gp.Model("home_energy_mpc_persistent")
        self.model.Params.OutputFlag = 1 if solver_tee else 0
        self.model.Params.MIPGap = cfg.gurobi_mipgap
        if cfg.gurobi_threads is not None:
            self.model.Params.Threads = cfg.gurobi_threads
        if cfg.gurobi_mipfocus is not None:
            self.model.Params.MIPFocus = cfg.gurobi_mipfocus

        self._build_model()
        self.build_once_time_sec = perf_counter() - t0

    def _add_optional_var(self, enabled: bool, name: str, ub: float) -> Any:
        if not enabled or ub <= 0.0:
            return None
        return self.model.addVars(range(self.H), lb=0.0, ub=ub, name=name)

    def _build_model(self) -> None:
        cfg = self.cfg
        H = self.H
        T = range(H)
        Te = range(H + 1)
        m = self.model

        # Version 4 remains intentionally out of scope for now.
        if cfg.enable_pv or cfg.enable_battery:
            raise NotImplementedError("PV and stationary battery modeling (Version 4) is not implemented yet.")

        # Feature toggles: only build variables/constraints for capabilities that are enabled.
        self.has_grid_export = bool(cfg.enable_grid_export and cfg.enable_ev_discharge_to_grid)
        self.has_ev_home_ch = bool(cfg.enable_ev_home_charge and cfg.max_ev_home_charge_kw > 0.0)
        self.has_ev_ext_ch = bool(cfg.enable_ev_external_charge and cfg.max_ev_external_charge_kw > 0.0)
        self.has_ev_dis_house = bool(cfg.enable_ev_discharge_to_house and cfg.max_ev_discharge_house_kw > 0.0)
        self.has_ev_dis_grid = bool(self.has_grid_export and cfg.max_ev_discharge_grid_kw > 0.0)

        self.vars: Dict[str, Any] = {
            "p_grid_import": m.addVars(T, lb=0.0, ub=cfg.p_grid_max_kw, name="p_grid_import"),
            "p_grid_export": self._add_optional_var(self.has_grid_export, "p_grid_export", cfg.p_grid_max_kw),
            "y_grid_dir": m.addVars(T, vtype=self.GRB.BINARY, name="y_grid_dir") if self.has_grid_export else None,
            "p_ev_home_ch": self._add_optional_var(self.has_ev_home_ch, "p_ev_home_ch", cfg.max_ev_home_charge_kw),
            "p_ev_ext_ch": self._add_optional_var(self.has_ev_ext_ch, "p_ev_ext_ch", cfg.max_ev_external_charge_kw),
            "p_ev_dis_house": self._add_optional_var(self.has_ev_dis_house, "p_ev_dis_house", cfg.max_ev_discharge_house_kw),
            "p_ev_dis_grid": self._add_optional_var(self.has_ev_dis_grid, "p_ev_dis_grid", cfg.max_ev_discharge_grid_kw),
            "p_ev_dis_house_home": self._add_optional_var(
                self.has_ev_dis_house,
                "p_ev_dis_house_home",
                cfg.max_ev_discharge_house_kw,
            ),
            "p_ev_dis_house_ext": self._add_optional_var(
                self.has_ev_dis_house,
                "p_ev_dis_house_ext",
                cfg.max_ev_discharge_house_kw,
            ),
            "p_ev_dis_grid_home": self._add_optional_var(
                self.has_ev_dis_grid,
                "p_ev_dis_grid_home",
                cfg.max_ev_discharge_grid_kw,
            ),
            "p_ev_dis_grid_ext": self._add_optional_var(
                self.has_ev_dis_grid,
                "p_ev_dis_grid_ext",
                cfg.max_ev_discharge_grid_kw,
            ),
            "p_ev_drive_home": m.addVars(T, lb=0.0, ub=cfg.ev_cap_kwh, name="p_ev_drive_home"),
            "p_ev_drive_ext": m.addVars(T, lb=0.0, ub=cfg.ev_cap_kwh, name="p_ev_drive_ext"),
            "y_ev_mode": None,
            "E_ev_home": m.addVars(Te, lb=0.0, ub=cfg.ev_cap_kwh, name="E_ev_home"),
            "E_ev_external": m.addVars(Te, lb=0.0, ub=cfg.ev_cap_kwh, name="E_ev_external"),
        }

        has_charge = self.has_ev_home_ch or self.has_ev_ext_ch
        has_discharge = self.has_ev_dis_house or self.has_ev_dis_grid
        if has_charge and has_discharge:
            # Binary mode variable prevents simultaneous charge and discharge at the EV.
            self.vars["y_ev_mode"] = m.addVars(T, vtype=self.GRB.BINARY, name="y_ev_mode")

        self.constrs: Dict[str, Any] = {
            "c_init_ev_home": m.addConstr(self.vars["E_ev_home"][0] == 0.0, name="c_init_ev_home"),
            "c_init_ev_external": m.addConstr(self.vars["E_ev_external"][0] == 0.0, name="c_init_ev_external"),
            "c_home_balance": {},
            "c_drive_split": {},
            "c_ev_home_dyn": {},
            "c_ev_external_dyn": {},
            "c_ev_lb": {},
            "c_ev_cap": {},
            "c_export_link": {},
            "c_ev_ch_total": {},
            "c_ev_dis_total": {},
            "c_ev_house_split": {},
            "c_ev_grid_split": {},
            "c_drive_ext_bound": {},
            "c_drive_home_bound": {},
            "c_house_home_bound": {},
            "c_house_ext_bound": {},
            "c_grid_home_bound": {},
            "c_grid_ext_bound": {},
        }

        for t in T:
            p_grid_import = self.vars["p_grid_import"][t]
            p_ev_home_ch = self.vars["p_ev_home_ch"][t] if self.vars["p_ev_home_ch"] is not None else 0.0
            p_ev_dis_house = self.vars["p_ev_dis_house"][t] if self.vars["p_ev_dis_house"] is not None else 0.0
            p_ev_ext_ch = self.vars["p_ev_ext_ch"][t] if self.vars["p_ev_ext_ch"] is not None else 0.0
            p_ev_dis_grid = self.vars["p_ev_dis_grid"][t] if self.vars["p_ev_dis_grid"] is not None else 0.0
            p_grid_export = self.vars["p_grid_export"][t] if self.vars["p_grid_export"] is not None else 0.0
            p_ev_dis_house_home = self.vars["p_ev_dis_house_home"][t] if self.vars["p_ev_dis_house_home"] is not None else 0.0
            p_ev_dis_house_ext = self.vars["p_ev_dis_house_ext"][t] if self.vars["p_ev_dis_house_ext"] is not None else 0.0
            p_ev_dis_grid_home = self.vars["p_ev_dis_grid_home"][t] if self.vars["p_ev_dis_grid_home"] is not None else 0.0
            p_ev_dis_grid_ext = self.vars["p_ev_dis_grid_ext"][t] if self.vars["p_ev_dis_grid_ext"] is not None else 0.0
            p_drive_home = self.vars["p_ev_drive_home"][t]
            p_drive_ext = self.vars["p_ev_drive_ext"][t]
            e_home = self.vars["E_ev_home"][t]
            e_ext = self.vars["E_ev_external"][t]
            e_home_next = self.vars["E_ev_home"][t + 1]
            e_ext_next = self.vars["E_ev_external"][t + 1]

            # Crucial physics: home-side power must balance at each timestep.
            self.constrs["c_home_balance"][t] = m.addConstr(
                p_grid_import + p_ev_dis_house - p_ev_home_ch == 0.0,
                name=f"c_home_balance_{t}",
            )

            # Link EV discharging-to-grid and grid export so energy accounting stays consistent.
            if self.vars["p_grid_export"] is not None and self.vars["p_ev_dis_grid"] is not None:
                self.constrs["c_export_link"][t] = m.addConstr(
                    p_grid_export == p_ev_dis_grid,
                    name=f"c_export_link_{t}",
                )
            if self.vars["p_ev_dis_house_home"] is not None and self.vars["p_ev_dis_house_ext"] is not None:
                self.constrs["c_ev_house_split"][t] = m.addConstr(
                    p_ev_dis_house == p_ev_dis_house_home + p_ev_dis_house_ext,
                    name=f"c_ev_house_split_{t}",
                )
                self.constrs["c_house_home_bound"][t] = m.addConstr(
                    p_ev_dis_house_home <= e_home * cfg.ev_eta_dis / cfg.dt_hours,
                    name=f"c_house_home_bound_{t}",
                )
                self.constrs["c_house_ext_bound"][t] = m.addConstr(
                    p_ev_dis_house_ext <= e_ext * cfg.ev_eta_dis / cfg.dt_hours,
                    name=f"c_house_ext_bound_{t}",
                )
            if self.vars["p_ev_dis_grid_home"] is not None and self.vars["p_ev_dis_grid_ext"] is not None:
                self.constrs["c_ev_grid_split"][t] = m.addConstr(
                    p_ev_dis_grid == p_ev_dis_grid_home + p_ev_dis_grid_ext,
                    name=f"c_ev_grid_split_{t}",
                )
                self.constrs["c_grid_home_bound"][t] = m.addConstr(
                    p_ev_dis_grid_home <= e_home * cfg.ev_eta_dis / cfg.dt_hours,
                    name=f"c_grid_home_bound_{t}",
                )
                self.constrs["c_grid_ext_bound"][t] = m.addConstr(
                    p_ev_dis_grid_ext <= e_ext * cfg.ev_eta_dis / cfg.dt_hours,
                    name=f"c_grid_ext_bound_{t}",
                )
            if self.vars["p_ev_home_ch"] is not None and self.vars["p_ev_ext_ch"] is not None:
                # Combined EV charging cannot exceed active charging-point capability.
                self.constrs["c_ev_ch_total"][t] = m.addConstr(
                    p_ev_home_ch + p_ev_ext_ch <= cfg.ev_cap_kwh,
                    name=f"c_ev_ch_total_{t}",
                )
            if self.vars["p_ev_dis_house"] is not None and self.vars["p_ev_dis_grid"] is not None:
                # Combined EV discharging cannot exceed active discharge capability.
                self.constrs["c_ev_dis_total"][t] = m.addConstr(
                    p_ev_dis_house + p_ev_dis_grid <= cfg.ev_cap_kwh,
                    name=f"c_ev_dis_total_{t}",
                )
            self.constrs["c_drive_split"][t] = m.addConstr(
                p_drive_home + p_drive_ext == 0.0,
                name=f"c_drive_split_{t}",
            )
            self.constrs["c_drive_ext_bound"][t] = m.addConstr(
                p_drive_ext <= e_ext,
                name=f"c_drive_ext_bound_{t}",
            )
            self.constrs["c_drive_home_bound"][t] = m.addConstr(
                p_drive_home <= e_home,
                name=f"c_drive_home_bound_{t}",
            )

            charge_term_home = cfg.ev_eta_ch * p_ev_home_ch
            charge_term_ext = cfg.ev_eta_ch * p_ev_ext_ch
            discharge_term_home = (1.0 / cfg.ev_eta_dis) * (p_ev_dis_house_home + p_ev_dis_grid_home)
            discharge_term_ext = (1.0 / cfg.ev_eta_dis) * (p_ev_dis_house_ext + p_ev_dis_grid_ext)
            self.constrs["c_ev_home_dyn"][t] = m.addConstr(
                e_home_next == e_home + (charge_term_home - discharge_term_home) * cfg.dt_hours - p_drive_home,
                name=f"c_ev_home_dyn_{t}",
            )
            self.constrs["c_ev_external_dyn"][t] = m.addConstr(
                e_ext_next == e_ext + (charge_term_ext - discharge_term_ext) * cfg.dt_hours - p_drive_ext,
                name=f"c_ev_external_dyn_{t}",
            )

            if self.vars["y_grid_dir"] is not None and self.vars["p_grid_export"] is not None:
                # Direction binary enforces mutual exclusion between importing and exporting.
                m.addConstr(
                    self.vars["p_grid_import"][t] <= cfg.p_grid_max_kw * self.vars["y_grid_dir"][t],
                    name=f"c_grid_import_mode_{t}",
                )
                m.addConstr(
                    self.vars["p_grid_export"][t] <= cfg.p_grid_max_kw * (1.0 - self.vars["y_grid_dir"][t]),
                    name=f"c_grid_export_mode_{t}",
                )

            if self.vars["y_ev_mode"] is not None:
                y = self.vars["y_ev_mode"][t]
                # EV mode binary enforces: charge OR discharge in a timestep, not both.
                if self.vars["p_ev_home_ch"] is not None:
                    m.addConstr(
                        self.vars["p_ev_home_ch"][t] <= cfg.max_ev_home_charge_kw * y,
                        name=f"c_ev_home_ch_mode_{t}",
                    )
                if self.vars["p_ev_ext_ch"] is not None:
                    m.addConstr(
                        self.vars["p_ev_ext_ch"][t] <= cfg.max_ev_external_charge_kw * y,
                        name=f"c_ev_ext_ch_mode_{t}",
                    )
                if self.vars["p_ev_dis_house"] is not None:
                    m.addConstr(
                        self.vars["p_ev_dis_house"][t] <= cfg.max_ev_discharge_house_kw * (1.0 - y),
                        name=f"c_ev_dis_house_mode_{t}",
                    )
                if self.vars["p_ev_dis_grid"] is not None:
                    m.addConstr(
                        self.vars["p_ev_dis_grid"][t] <= cfg.max_ev_discharge_grid_kw * (1.0 - y),
                        name=f"c_ev_dis_grid_mode_{t}",
                    )

        for t in Te:
            self.constrs["c_ev_lb"][t] = m.addConstr(
                self.vars["E_ev_home"][t] + self.vars["E_ev_external"][t] >= 0.0,
                name=f"c_ev_lb_{t}",
            )
            self.constrs["c_ev_cap"][t] = m.addConstr(
                self.vars["E_ev_home"][t] + self.vars["E_ev_external"][t] <= cfg.ev_cap_kwh,
                name=f"c_ev_max_{t}",
            )

        self._set_objective_coeffs(
            home_import_price=[0.0] * H,
            ev_home_import_price=[0.0] * H,
            ev_ext_import_price=[0.0] * H,
            home_export_price=[0.0] * H,
            external_export_price=[0.0] * H,
        )
        m.ModelSense = self.GRB.MINIMIZE
        m.update()

    def _set_objective_coeffs(
        self,
        home_import_price: List[float],
        ev_home_import_price: List[float],
        ev_ext_import_price: List[float],
        home_export_price: List[float],
        external_export_price: List[float],
    ) -> None:
        # Objective is linear operating cost per step.
        dt = self.cfg.dt_hours
        ev_degradation_cost = float(self.cfg.ev_degradation_eur_per_kwh_charged) * dt
        for t in range(self.H):
            self.vars["p_grid_import"][t].Obj = float(home_import_price[t]) * dt
            if self.vars["p_grid_export"] is not None:
                self.vars["p_grid_export"][t].Obj = 0.0
            if self.vars["p_ev_ext_ch"] is not None:
                self.vars["p_ev_ext_ch"][t].Obj = float(ev_ext_import_price[t]) * dt + ev_degradation_cost
            if self.vars["p_ev_home_ch"] is not None:
                self.vars["p_ev_home_ch"][t].Obj = (float(ev_home_import_price[t]) - float(home_import_price[t])) * dt + ev_degradation_cost
            if self.vars["p_ev_dis_house"] is not None:
                self.vars["p_ev_dis_house"][t].Obj = 0.0
            if self.vars["p_ev_dis_grid"] is not None:
                self.vars["p_ev_dis_grid"][t].Obj = 0.0
            if self.vars["p_ev_dis_house_home"] is not None:
                self.vars["p_ev_dis_house_home"][t].Obj = 0.0
            if self.vars["p_ev_dis_house_ext"] is not None:
                self.vars["p_ev_dis_house_ext"][t].Obj = 0.0
            if self.vars["p_ev_dis_grid_home"] is not None:
                self.vars["p_ev_dis_grid_home"][t].Obj = -float(home_export_price[t]) * dt
            if self.vars["p_ev_dis_grid_ext"] is not None:
                self.vars["p_ev_dis_grid_ext"][t].Obj = -float(external_export_price[t]) * dt
            if self.vars["p_ev_drive_home"] is not None:
                self.vars["p_ev_drive_home"][t].Obj = 0.0
            if self.vars["p_ev_drive_ext"] is not None:
                self.vars["p_ev_drive_ext"][t].Obj = 0.0
            if self.vars["y_grid_dir"] is not None:
                self.vars["y_grid_dir"][t].Obj = 0.0
            if self.vars["y_ev_mode"] is not None:
                self.vars["y_ev_mode"][t].Obj = 0.0

        for t in range(self.H + 1):
            self.vars["E_ev_home"][t].Obj = 0.0
            self.vars["E_ev_external"][t].Obj = 0.0

    def _set_ub(self, family: str, t: int, ub: float) -> None:
        var_family = self.vars.get(family)
        if var_family is not None:
            var_family[t].UB = max(0.0, float(ub))

    def update(self, window_arrays: Dict[str, List[float]], soc_ev_home_now: float, soc_ev_external_now: float) -> None:
        """Inject current state and forecast data into mutable constraint RHS/UB values."""
        t0 = perf_counter()
        self._latest_arrays = window_arrays
        # Reset initial EV state for this rolling window.
        self.constrs["c_init_ev_home"].RHS = float(soc_ev_home_now)
        self.constrs["c_init_ev_external"].RHS = float(soc_ev_external_now)

        for t in range(self.H):
            # Update forecast-dependent constraints for each look-ahead step.
            self.constrs["c_home_balance"][t].RHS = float(window_arrays["load_kw"][t])
            self.constrs["c_drive_split"][t].RHS = float(window_arrays["ev_drive_kwh"][t])

            # Update power limits from connection state (home, external, unavailable).
            self._set_ub("p_ev_home_ch", t, window_arrays["ev_p_home_ch_max_kw"][t])
            self._set_ub("p_ev_ext_ch", t, window_arrays["ev_p_ext_ch_max_kw"][t])
            self._set_ub("p_ev_dis_house", t, window_arrays["ev_p_dis_house_max_kw"][t])
            self._set_ub("p_ev_dis_grid", t, window_arrays["ev_p_dis_grid_max_kw"][t])
            self._set_ub("p_ev_dis_house_home", t, window_arrays["ev_p_dis_house_max_kw"][t])
            self._set_ub("p_ev_dis_house_ext", t, window_arrays["ev_p_dis_house_max_kw"][t])
            self._set_ub("p_ev_dis_grid_home", t, window_arrays["ev_p_dis_grid_max_kw"][t])
            self._set_ub("p_ev_dis_grid_ext", t, window_arrays["ev_p_dis_grid_max_kw"][t])
            self._set_ub("p_ev_drive_home", t, self.cfg.ev_cap_kwh)
            self._set_ub("p_ev_drive_ext", t, self.cfg.ev_cap_kwh)
            if t in self.constrs["c_ev_ch_total"]:
                self.constrs["c_ev_ch_total"][t].RHS = float(window_arrays["ev_p_ch_total_max_kw"][t])
            if t in self.constrs["c_ev_dis_total"]:
                self.constrs["c_ev_dis_total"][t].RHS = float(window_arrays["ev_p_dis_total_max_kw"][t])
        for t in range(self.H + 1):
            self.constrs["c_ev_lb"][t].RHS = max(
                0.0,
                _effective_ev_lb(self.cfg, window_arrays["ev_reserve_kwh"][t]) - EV_SOC_CLAMP_EPS_KWH,
            )
            self.constrs["c_ev_cap"][t].RHS = self.cfg.ev_cap_kwh

        self._set_objective_coeffs(
            home_import_price=window_arrays["home_import_price"],
            ev_home_import_price=window_arrays["ev_home_import_price"],
            ev_ext_import_price=window_arrays["ev_ext_import_price"],
            home_export_price=window_arrays["ev_home_export_price"],
            external_export_price=window_arrays["ev_external_export_price"],
        )
        self.model.update()
        self.last_update_time_sec = perf_counter() - t0

    def apply_mip_start(self, prev_solution: Dict[str, List[float]], shift: int = 1) -> None:
        """Warm-start current window with shifted previous variable values."""
        for family, var_by_t in self.vars.items():
            if var_by_t is None:
                continue
            prev = prev_solution.get(family)
            if prev is None:
                continue
            n = len(prev)
            for t in var_by_t.keys():
                src = min(t + shift, n - 1)
                var_by_t[t].Start = float(prev[src]) if src >= 0 else 0.0

    def optimize(self) -> str:
        t0 = perf_counter()
        self.model.optimize()
        self.last_optimize_time_sec = perf_counter() - t0
        code = self.model.Status
        if code == self.GRB.OPTIMAL:
            return "optimal"
        if self.model.SolCount > 0:
            return "feasible"
        return f"status_{code}"

    def _x(self, family: str, t: int) -> float:
        var_family = self.vars.get(family)
        if var_family is None:
            return 0.0
        return float(var_family[t].X)

    def extract_action(self, t: int = 0) -> Dict[str, float]:
        """Return one horizon action in both detailed and compatibility fields."""
        t0 = perf_counter()
        ev_home_ch = self._x("p_ev_home_ch", t)
        ev_ext_ch = self._x("p_ev_ext_ch", t)
        ev_dis_house = self._x("p_ev_dis_house", t)
        ev_dis_grid = self._x("p_ev_dis_grid", t)
        ev_dis_house_home = self._x("p_ev_dis_house_home", t)
        ev_dis_house_ext = self._x("p_ev_dis_house_ext", t)
        ev_dis_grid_home = self._x("p_ev_dis_grid_home", t)
        ev_dis_grid_ext = self._x("p_ev_dis_grid_ext", t)
        ev_drive_home = self._x("p_ev_drive_home", t)
        ev_drive_ext = self._x("p_ev_drive_ext", t)
        ev_home = self._x("E_ev_home", t)
        ev_external = self._x("E_ev_external", t)
        ev_home_next = self._x("E_ev_home", t + 1)
        ev_external_next = self._x("E_ev_external", t + 1)
        out = {
            "grid_import_kw": self._x("p_grid_import", t),
            "grid_export_kw": self._x("p_grid_export", t),
            "bat_ch_kw": 0.0,
            "bat_dis_kw": 0.0,
            "ev_home_ch_kw": ev_home_ch,
            "ev_ext_ch_kw": ev_ext_ch,
            "ev_dis_to_home_kw": ev_dis_house,
            "ev_dis_to_grid_kw": ev_dis_grid,
            "ev_dis_to_home_home_kwh": ev_dis_house_home * self.cfg.dt_hours,
            "ev_dis_to_home_external_kwh": ev_dis_house_ext * self.cfg.dt_hours,
            "ev_dis_to_grid_home_kwh": ev_dis_grid_home * self.cfg.dt_hours,
            "ev_dis_to_grid_external_kwh": ev_dis_grid_ext * self.cfg.dt_hours,
            "ev_drive_home_kwh": ev_drive_home,
            "ev_drive_external_kwh": ev_drive_ext,
            "ev_home_energy_kwh": ev_home,
            "ev_external_energy_kwh": ev_external,
            "ev_energy_kwh": ev_home + ev_external,
            "ev_home_energy_next_kwh": ev_home_next,
            "ev_external_energy_next_kwh": ev_external_next,
            "ev_energy_next_kwh": ev_home_next + ev_external_next,
            "ev_ch_kw": ev_home_ch + ev_ext_ch,
            "ev_dis_kw": ev_dis_house + ev_dis_grid,
        }
        self.last_extract_time_sec = perf_counter() - t0
        return out

    def extract_first_action(self) -> Dict[str, float]:
        """Return first-step control action in both detailed and compatibility fields."""
        return self.extract_action(0)

    def extract_solution_full_horizon(self) -> Dict[str, List[float]]:
        t0 = perf_counter()
        out: Dict[str, List[float]] = {}
        for family, var_by_t in self.vars.items():
            if var_by_t is None:
                continue
            out[family] = [float(var_by_t[t].X) for t in sorted(var_by_t.keys())]
        self.last_extract_time_sec = perf_counter() - t0
        return out


def _safe_fallback_step(row: pd.Series, cfg: EnergySystemConfig, e_ev_home: float, e_ev_external: float) -> Dict[str, float]:
    """Cheap feasible fallback that keeps the split-state accounting consistent."""
    load_kw = float(row["load_kw"])
    drive_kwh = float(row["ev_drive_kwh"])
    ev_lb = _effective_ev_lb(cfg, float(row["ev_reserve_kwh"]))
    p_ev_home_ch_max = float(row.get("ev_p_home_ch_max_kw", 0.0))
    p_ev_ext_ch_max = float(row.get("ev_p_ext_ch_max_kw", 0.0))

    total_now = e_ev_home + e_ev_external
    p_ev_home_ch = 0.0
    p_ev_ext_ch = 0.0
    p_grid_import = min(load_kw, cfg.p_grid_max_kw)

    if total_now < ev_lb:
        p_need_kwh = (ev_lb - total_now) / cfg.ev_eta_ch
        home_headroom = max(0.0, cfg.p_grid_max_kw - p_grid_import)
        p_ev_home_ch = min(p_ev_home_ch_max, home_headroom, p_need_kwh)
        p_need_kwh = max(0.0, p_need_kwh - p_ev_home_ch * cfg.dt_hours)
        p_ev_ext_ch = min(p_ev_ext_ch_max, p_need_kwh / cfg.dt_hours)
        p_grid_import = min(cfg.p_grid_max_kw, p_grid_import + p_ev_home_ch)

    drive_ext = min(drive_kwh, e_ev_external)
    drive_home = max(0.0, drive_kwh - drive_ext)
    next_home = max(0.0, e_ev_home + cfg.ev_eta_ch * p_ev_home_ch * cfg.dt_hours - drive_home)
    next_external = max(0.0, e_ev_external + cfg.ev_eta_ch * p_ev_ext_ch * cfg.dt_hours - drive_ext)
    total_next = min(cfg.ev_cap_kwh, max(ev_lb, next_home + next_external))
    if total_next > 0.0:
        scale = total_next / max(next_home + next_external, 1e-12)
        next_home *= scale
        next_external *= scale

    return {
        "grid_import_kw": p_grid_import,
        "grid_export_kw": 0.0,
        "bat_ch_kw": 0.0,
        "bat_dis_kw": 0.0,
        "ev_home_ch_kw": p_ev_home_ch,
        "ev_ext_ch_kw": p_ev_ext_ch,
        "ev_dis_to_home_kw": 0.0,
        "ev_dis_to_grid_kw": 0.0,
        "ev_dis_to_home_home_kwh": 0.0,
        "ev_dis_to_home_external_kwh": 0.0,
        "ev_dis_to_grid_home_kwh": 0.0,
        "ev_dis_to_grid_external_kwh": 0.0,
        "ev_drive_home_kwh": drive_home,
        "ev_drive_external_kwh": drive_ext,
        "ev_home_energy_kwh": e_ev_home,
        "ev_external_energy_kwh": e_ev_external,
        "ev_energy_kwh": e_ev_home + e_ev_external,
        "ev_home_energy_next_kwh": next_home,
        "ev_external_energy_next_kwh": next_external,
        "ev_energy_next_kwh": next_home + next_external,
        "ev_ch_kw": p_ev_home_ch + p_ev_ext_ch,
        "ev_dis_kw": 0.0,
    }


def _action_from_solution(solution: Dict[str, List[float]], t: int, cfg: EnergySystemConfig) -> Dict[str, float]:
    def _val(key: str) -> float:
        arr = solution.get(key)
        if arr is None or t >= len(arr):
            return 0.0
        return float(arr[t])

    ev_home_ch = _val("p_ev_home_ch")
    ev_ext_ch = _val("p_ev_ext_ch")
    ev_dis_house = _val("p_ev_dis_house")
    ev_dis_grid = _val("p_ev_dis_grid")
    ev_dis_house_home = _val("p_ev_dis_house_home")
    ev_dis_house_ext = _val("p_ev_dis_house_ext")
    ev_dis_grid_home = _val("p_ev_dis_grid_home")
    ev_dis_grid_ext = _val("p_ev_dis_grid_ext")
    ev_drive_home = _val("p_ev_drive_home")
    ev_drive_ext = _val("p_ev_drive_ext")
    ev_home = _val("E_ev_home")
    ev_external = _val("E_ev_external")
    ev_home_next_arr = solution.get("E_ev_home", [])
    ev_external_next_arr = solution.get("E_ev_external", [])
    ev_home_next = float(ev_home_next_arr[t + 1]) if t + 1 < len(ev_home_next_arr) else ev_home
    ev_external_next = float(ev_external_next_arr[t + 1]) if t + 1 < len(ev_external_next_arr) else ev_external
    return {
        "grid_import_kw": _val("p_grid_import"),
        "grid_export_kw": _val("p_grid_export"),
        "bat_ch_kw": 0.0,
        "bat_dis_kw": 0.0,
        "ev_home_ch_kw": ev_home_ch,
        "ev_ext_ch_kw": ev_ext_ch,
        "ev_dis_to_home_kw": ev_dis_house,
        "ev_dis_to_grid_kw": ev_dis_grid,
        "ev_dis_to_home_home_kwh": ev_dis_house_home * cfg.dt_hours,
        "ev_dis_to_home_external_kwh": ev_dis_house_ext * cfg.dt_hours,
        "ev_dis_to_grid_home_kwh": ev_dis_grid_home * cfg.dt_hours,
        "ev_dis_to_grid_external_kwh": ev_dis_grid_ext * cfg.dt_hours,
        "ev_drive_home_kwh": ev_drive_home,
        "ev_drive_external_kwh": ev_drive_ext,
        "ev_home_energy_kwh": ev_home,
        "ev_external_energy_kwh": ev_external,
        "ev_energy_kwh": ev_home + ev_external,
        "ev_home_energy_next_kwh": ev_home_next,
        "ev_external_energy_next_kwh": ev_external_next,
        "ev_energy_next_kwh": ev_home_next + ev_external_next,
        "ev_ch_kw": ev_home_ch + ev_ext_ch,
        "ev_dis_kw": ev_dis_house + ev_dis_grid,
    }


def _build_and_solve_window_gurobi(
    window: pd.DataFrame,
    cfg: EnergySystemConfig,
    e_ev_home0: float,
    e_ev_external0: float,
    solver_tee: bool = False,
) -> MPCSolveResult:
    """Legacy mode: rebuild one temporary model per window solve."""
    solver = PersistentMPCSolver(cfg=cfg, H=len(window), solver_tee=solver_tee)
    arrays = WindowDataManager.get_arrays(window, 0, len(window), cfg)
    solver.update(arrays, soc_ev_home_now=e_ev_home0, soc_ev_external_now=e_ev_external0)
    status = solver.optimize()
    if solver.model.SolCount == 0:
        return MPCSolveResult(first_step={}, status=status)
    return MPCSolveResult(
        first_step=solver.extract_first_action(),
        status=status,
        full_solution=solver.extract_solution_full_horizon(),
    )


def run_mpc_loop(
    df: pd.DataFrame,
    cfg: EnergySystemConfig,
    show_progress: bool = True,
    progress_every: int = 50,
    slow_step_sec: float = 2.0,
    solver_tee: bool = False,
    use_persistent_gurobi: Optional[bool] = None,
    use_mip_start: Optional[bool] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    """Run rolling-horizon MPC and return per-step actions plus solver logs."""
    # Current implementation scope: EV/grid only.
    if cfg.enable_pv or cfg.enable_battery:
        raise NotImplementedError("Versions with PV or stationary battery are prepared in config but not implemented yet.")

    # Use config defaults unless caller overrides them.
    if use_persistent_gurobi is None:
        use_persistent_gurobi = cfg.use_persistent_gurobi
    if use_mip_start is None:
        use_mip_start = cfg.use_mip_start

    # Runtime state of stored energy (kWh) propagated step-by-step.
    e_bat = cfg.bat_soc_init * cfg.bat_cap_kwh
    e_ev = cfg.ev_soc_init * cfg.ev_cap_kwh
    n_steps = len(df)
    if n_steps > 0 and str(df.iloc[0].get("ev_state", "")).strip().lower() == "home":
        e_ev_home = e_ev
        e_ev_external = 0.0
    else:
        e_ev_home = 0.0
        e_ev_external = e_ev

    # `rows`: simulation outputs, `logs`: diagnostics and timings.
    logs: List[Dict[str, object]] = []
    rows: List[Dict[str, float]] = []
    ev_reserve_kwh_by_step = pd.to_numeric(df["ev_reserve_kwh"], errors="coerce").fillna(0.0).astype(float).tolist()
    t_loop_start = perf_counter()
    slowest_solve_sec = -1.0
    slowest_solve_ts = ""

    # Force API load once so import overhead is visible in logs/progress output.
    gurobi_init_sec = _load_gurobi_api()["import_seconds"]

    persistent_solver: Optional[PersistentMPCSolver] = None
    if use_persistent_gurobi:
        # Persistent mode builds once and then only updates RHS/UB values each step.
        persistent_solver = PersistentMPCSolver(cfg=cfg, H=cfg.horizon_steps, solver_tee=solver_tee)
        logs.append(
            {
                "event": "persistent_model_built",
                "build_once_time_sec": round(persistent_solver.build_once_time_sec, 6),
                "horizon_steps": cfg.horizon_steps,
            }
        )

    if show_progress:
        print(f"[MPC] Gurobi import/init took {gurobi_init_sec:.2f}s")
        print(
            f"[MPC] Starting rolling optimization: steps={n_steps}, horizon={cfg.horizon_steps}, "
            f"apply_steps={max(1, int(cfg.mpc_apply_steps))}, "
            f"report_every={progress_every}, slow_step_sec={slow_step_sec:.2f}"
        )

    prev_solution: Optional[Dict[str, List[float]]] = None
    prev_solution_shift = 1
    # Diagnostics timing state:
    # - `prev_e_ev_raw` becomes this row's start-of-step raw EV energy.
    # - `prev_used_fallback` links current clamp diagnostics to the previous step action source.
    prev_e_ev_raw = e_ev
    prev_used_fallback = 0
    # Receding-horizon loop: solve a window, apply one or more horizon actions, then replan.
    i = 0
    while i < n_steps:
        ts = df.index[i]
        window = df.iloc[i : i + cfg.horizon_steps]
        apply_steps = min(max(1, int(cfg.mpc_apply_steps)), cfg.horizon_steps, n_steps - i)
        block_actions: List[Dict[str, float]] = []
        status = "not_run"
        t_block_start = perf_counter()
        update_sec = 0.0
        extract_sec = 0.0
        used_fallback = 0
        build_once_sec = persistent_solver.build_once_time_sec if persistent_solver is not None else 0.0
        t_solve_start = perf_counter()
        try:
            if persistent_solver is not None:
                # Build numeric arrays for the current window and inject them into the persistent model.
                arrays = WindowDataManager.get_arrays(df, i, cfg.horizon_steps, cfg)
                t_update = perf_counter()
                persistent_solver.update(arrays, soc_ev_home_now=e_ev_home, soc_ev_external_now=e_ev_external)
                if use_mip_start and prev_solution is not None:
                    # Warm-start from previous solution to reduce solve time.
                    persistent_solver.apply_mip_start(prev_solution, shift=prev_solution_shift)
                update_sec = perf_counter() - t_update

                status = persistent_solver.optimize()
                if persistent_solver.model.SolCount > 0:
                    # Apply the configured number of actions from this optimized horizon before replanning.
                    t_extract = perf_counter()
                    block_actions = [persistent_solver.extract_action(t) for t in range(apply_steps)]
                    extract_sec = perf_counter() - t_extract
                    if use_mip_start:
                        prev_solution = persistent_solver.extract_solution_full_horizon()
                        prev_solution_shift = apply_steps
                else:
                    # If optimization returns no solution, fall back to a safe heuristic.
                    used_fallback = 1
                    logs.append({"timestamp": str(ts), "step": i + 1, "status": status, "action": "fallback"})
            else:
                # Legacy mode: rebuild model per window (slower, but useful for regression comparison).
                solved = _build_and_solve_window_gurobi(window, cfg, e_ev_home, e_ev_external, solver_tee=solver_tee)
                status = solved.status
                if solved.full_solution:
                    block_actions = [_action_from_solution(solved.full_solution, t, cfg) for t in range(apply_steps)]
                    if use_mip_start:
                        prev_solution = solved.full_solution
                        prev_solution_shift = apply_steps
                else:
                    used_fallback = 1
                    logs.append({"timestamp": str(ts), "step": i + 1, "status": status, "action": "fallback"})
        except Exception as ex:
            # Defensive behavior: never stop the whole run because one step failed.
            status = f"error:{ex}"
            used_fallback = 1
            logs.append({"timestamp": str(ts), "step": i + 1, "status": status, "action": "fallback"})

        # Track pure optimization time and full step time separately.
        solve_sec = perf_counter() - t_solve_start
        block_total_sec = perf_counter() - t_block_start
        if solve_sec > slowest_solve_sec:
            slowest_solve_sec = solve_sec
            slowest_solve_ts = str(ts)

        for local_t in range(apply_steps):
            j = i + local_t
            ts_step = df.index[j]
            row = df.iloc[j]
            e_ev_start = e_ev_home + e_ev_external
            e_ev_home_start = e_ev_home
            e_ev_external_start = e_ev_external
            e_ev_pre_clamp_start = prev_e_ev_raw
            step = (
                block_actions[local_t]
                if local_t < len(block_actions)
                else _safe_fallback_step(row, cfg, e_ev_home, e_ev_external)
            )

            drive_kwh = float(row["ev_drive_kwh"])
            drive_external = min(drive_kwh, e_ev_external_start)
            drive_home = max(0.0, drive_kwh - drive_external)
            home_charge_kwh = float(step["ev_home_ch_kw"]) * cfg.dt_hours
            external_charge_kwh = float(step["ev_ext_ch_kw"]) * cfg.dt_hours
            dis_home_store_kwh = (float(step["ev_dis_to_home_home_kwh"]) + float(step["ev_dis_to_grid_home_kwh"])) / cfg.ev_eta_dis
            dis_external_store_kwh = (float(step["ev_dis_to_home_external_kwh"]) + float(step["ev_dis_to_grid_external_kwh"])) / cfg.ev_eta_dis
            e_ev_home_next = max(0.0, e_ev_home_start + cfg.ev_eta_ch * home_charge_kwh - dis_home_store_kwh - drive_home)
            e_ev_external_next = max(
                0.0,
                e_ev_external_start + cfg.ev_eta_ch * external_charge_kwh - dis_external_store_kwh - drive_external,
            )
            e_ev_raw_next = e_ev_home_next + e_ev_external_next

            # Clamp carried state against the reserve of the step it will be used in (j+1),
            # not the reserve of the action step (j), to avoid a one-step timing mismatch.
            if j + 1 < n_steps:
                ev_lb_carry = _effective_ev_lb(cfg, ev_reserve_kwh_by_step[j + 1])
            else:
                ev_lb_carry = _effective_ev_lb(cfg, ev_reserve_kwh_by_step[j])
            e_ev = min(max(e_ev_raw_next, ev_lb_carry), cfg.ev_cap_kwh)
            e_ev_home, e_ev_external = _scale_split_to_total(e_ev_home_next, e_ev_external_next, e_ev)
            # Clamp diagnostics are reported as start-of-step quantities:
            # compare current clamped start state vs previous step raw carry-over.
            ev_soc_clamp_delta_kwh = e_ev_start - e_ev_pre_clamp_start
            if abs(ev_soc_clamp_delta_kwh) <= EV_SOC_CLAMP_EPS_KWH:
                ev_soc_clamp_delta_kwh = 0.0
            ev_soc_clamped = float(abs(ev_soc_clamp_delta_kwh) > 0.0)
            ev_soc_clamped_after_fallback = float(prev_used_fallback and ev_soc_clamped > 0.5)
            e_bat = min(max(e_bat, cfg.bat_soc_min * cfg.bat_cap_kwh), cfg.bat_soc_max * cfg.bat_cap_kwh)

            step.update(
                {
                    "bat_energy_kwh": e_bat,
                    "ev_energy_kwh": e_ev_start,
                    "ev_home_energy_kwh": e_ev_home_start,
                    "ev_external_energy_kwh": e_ev_external_start,
                    "ev_energy_pre_clamp_kwh": e_ev_pre_clamp_start,
                    "ev_soc_clamped": ev_soc_clamped,
                    "ev_soc_clamp_delta_kwh": ev_soc_clamp_delta_kwh,
                    "ev_soc_clamped_after_fallback": ev_soc_clamped_after_fallback,
                    "ev_soc_clamp_after_fallback_delta_kwh": ev_soc_clamp_delta_kwh if ev_soc_clamped_after_fallback > 0.5 else 0.0,
                    "ev_consumption_kwh": drive_kwh,
                    "ev_drive_external_kwh": drive_external,
                    "ev_drive_home_kwh": drive_home,
                    "ev_dis_to_home_external_kwh": float(step["ev_dis_to_home_external_kwh"]),
                    "ev_dis_to_home_home_kwh": float(step["ev_dis_to_home_home_kwh"]),
                    "ev_dis_to_grid_external_kwh": float(step["ev_dis_to_grid_external_kwh"]),
                    "ev_dis_to_grid_home_kwh": float(step["ev_dis_to_grid_home_kwh"]),
                    "ev_home_energy_next_kwh": e_ev_home_next,
                    "ev_external_energy_next_kwh": e_ev_external_next,
                    "ev_energy_next_kwh": e_ev_raw_next,
                    "solver_status": status,
                    "used_fallback": float(used_fallback),
                }
            )
            rows.append(step)
            logs.append(
                {
                    "timestamp": str(ts_step),
                    "step": j + 1,
                    "total_steps": n_steps,
                    "window_steps": len(window),
                    "apply_steps": apply_steps,
                    "applied_step_in_block": local_t + 1,
                    "status": status,
                    "build_once_time_sec": round(build_once_sec, 6) if local_t == 0 else 0.0,
                    "update_seconds": round(update_sec, 6) if local_t == 0 else 0.0,
                    "solve_seconds": round(solve_sec, 4) if local_t == 0 else 0.0,
                    "extract_seconds": round(extract_sec, 6) if local_t == 0 else 0.0,
                    "step_total_seconds": round(block_total_sec, 4) if local_t == 0 else 0.0,
                    "used_fallback": int(used_fallback),
                    "ev_soc_clamped": int(ev_soc_clamped > 0.5),
                    "ev_soc_clamp_delta_kwh": round(ev_soc_clamp_delta_kwh, 6),
                    "ev_soc_clamped_after_fallback": int(ev_soc_clamped_after_fallback > 0.5),
                }
            )

            if show_progress:
                completed = j + 1
                should_report = (
                    completed == 1
                    or completed == n_steps
                    or completed % max(1, progress_every) == 0
                    or (local_t == 0 and solve_sec >= slow_step_sec)
                )
                if should_report:
                    elapsed = perf_counter() - t_loop_start
                    avg_step_sec = elapsed / completed
                    eta_sec = avg_step_sec * (n_steps - completed)
                    print(
                        f"[MPC] {completed}/{n_steps} ({100.0 * completed / n_steps:.1f}%) "
                        f"ts={ts_step} solve={solve_sec if local_t == 0 else 0.0:.2f}s "
                        f"step={block_total_sec if local_t == 0 else 0.0:.2f}s avg={avg_step_sec:.2f}s "
                        f"eta={eta_sec / 60.0:.1f}m status={status}"
                    )

            # Carry timing state for next row diagnostics.
            prev_e_ev_raw = e_ev_raw_next
            prev_used_fallback = used_fallback

        i += apply_steps

    if show_progress:
        total_elapsed_sec = perf_counter() - t_loop_start
        print(
            f"[MPC] Finished {n_steps} steps in {total_elapsed_sec / 60.0:.2f} min. "
            f"Slowest solve: {slowest_solve_sec:.2f}s at {slowest_solve_ts}"
        )

    out = pd.DataFrame(rows, index=df.index)
    out["grid_import_kwh"] = out["grid_import_kw"] * cfg.dt_hours
    out["grid_export_kwh"] = out["grid_export_kw"] * cfg.dt_hours
    out["home_load_kw"] = df["load_kw"]
    out["home_load_kwh"] = out["home_load_kw"] * cfg.dt_hours
    out["ev_home_ch_kwh"] = out["ev_home_ch_kw"] * cfg.dt_hours
    out["ev_ext_ch_kwh"] = out["ev_ext_ch_kw"] * cfg.dt_hours
    out["ev_dis_to_home_kwh"] = out["ev_dis_to_home_kw"] * cfg.dt_hours
    out["ev_dis_to_grid_kwh"] = out["ev_dis_to_grid_kw"] * cfg.dt_hours
    home_grid_price = pd.to_numeric(df.get("home_grid_price_eur_per_kwh", df["import_price_eur_per_kwh"]), errors="coerce").fillna(0.0)
    ev_home_import_price = pd.to_numeric(
        df.get("ev_home_import_price_eur_per_kwh", df["import_price_eur_per_kwh"]),
        errors="coerce",
    ).fillna(0.0)
    out["ext_charge_cost_eur"] = df["ev_ext_import_price_eur_per_kwh"] * out["ev_ext_ch_kwh"]
    out["ev_battery_degradation_cost_eur"] = (
        float(cfg.ev_degradation_eur_per_kwh_charged) * (out["ev_home_ch_kwh"] + out["ev_ext_ch_kwh"])
    )
    out["home_load_grid_import_kwh"] = out[["grid_import_kwh", "home_load_kwh"]].min(axis=1)
    out["ev_home_charge_cost_eur"] = ev_home_import_price * out["ev_home_ch_kwh"]
    out["home_load_cost_eur"] = home_grid_price * out["home_load_kwh"]
    out["home_grid_price_total_eur_per_kwh"] = home_grid_price
    out["ev_home_import_price_eur_per_kwh"] = ev_home_import_price
    out["ev_home_export_price_eur_per_kwh"] = df["ev_home_export_price_eur_per_kwh"]
    out["ev_external_export_price_eur_per_kwh"] = df["ev_external_export_price_eur_per_kwh"]
    out["ev_export_price_eur_per_kwh"] = df["ev_export_price_eur_per_kwh"]
    out["ev_discharge_grid_revenue_home_eur"] = df["ev_home_export_price_eur_per_kwh"] * out["ev_dis_to_grid_home_kwh"]
    out["ev_discharge_grid_revenue_external_eur"] = (
        df["ev_external_export_price_eur_per_kwh"] * out["ev_dis_to_grid_external_kwh"]
    )
    out["ev_discharge_grid_revenue_eur"] = (
        out["ev_discharge_grid_revenue_home_eur"] + out["ev_discharge_grid_revenue_external_eur"]
    )
    out["step_cost_eur"] = (
        home_grid_price * out["home_load_kwh"]
        + ev_home_import_price * out["ev_home_ch_kwh"]
        - out["ev_discharge_grid_revenue_eur"]
        + out["ext_charge_cost_eur"]
        + out["ev_battery_degradation_cost_eur"]
    )
    out["ev_reserve_kwh"] = df["ev_reserve_kwh"]
    out["ev_state"] = df["ev_state"]
    out["charging_point_effective"] = df["charging_point_effective"]
    return out, logs
