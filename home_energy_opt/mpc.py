from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from home_energy_opt.config import EnergySystemConfig

_GUROBI_API: Optional[Dict[str, Any]] = None


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
    """Single EV lower bound per step: max(fixed minimum, bounded reserve)."""
    fixed_min = cfg.ev_soc_min * cfg.ev_cap_kwh
    reserve_bounded = min(float(reserve_kwh), cfg.ev_cap_kwh)
    return max(fixed_min, reserve_bounded)


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
                for col in ("import_price_eur_per_kwh", "ev_ext_import_price_eur_per_kwh", "ev_export_price_eur_per_kwh"):
                    if col in tail.columns:
                        tail[col] = float(cfg.pad_tail_price_eur_per_kwh)
            window = pd.concat([window, pd.concat([tail] * pad_rows, ignore_index=False)], axis=0)

        return {
            "load_kw": window["load_kw"].astype(float).tolist(),
            "home_import_price": window["import_price_eur_per_kwh"].astype(float).tolist(),
            "ev_drive_kwh": window["ev_drive_kwh"].astype(float).tolist(),
            "ev_reserve_kwh": window["ev_reserve_kwh"].astype(float).tolist(),
            "ev_p_home_ch_max_kw": window["ev_p_home_ch_max_kw"].astype(float).tolist(),
            "ev_p_ext_ch_max_kw": window["ev_p_ext_ch_max_kw"].astype(float).tolist(),
            "ev_p_ch_total_max_kw": window["ev_p_ch_total_max_kw"].astype(float).tolist(),
            "ev_p_dis_house_max_kw": window["ev_p_dis_house_max_kw"].astype(float).tolist(),
            "ev_p_dis_grid_max_kw": window["ev_p_dis_grid_max_kw"].astype(float).tolist(),
            "ev_p_dis_total_max_kw": window["ev_p_dis_total_max_kw"].astype(float).tolist(),
            "ev_ext_import_price": window["ev_ext_import_price_eur_per_kwh"].astype(float).tolist(),
            "ev_export_price": window["ev_export_price_eur_per_kwh"].astype(float).tolist(),
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
        if cfg.gurobi_time_limit_sec is not None:
            self.model.Params.TimeLimit = cfg.gurobi_time_limit_sec
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

    def _terminal_reserve_lb(self, ev_reserve_kwh: List[float]) -> float:
        return _effective_ev_lb(self.cfg, ev_reserve_kwh[-1])

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
            "y_ev_mode": None,
            "E_ev": m.addVars(Te, lb=0.0, ub=cfg.ev_cap_kwh, name="E_ev"),
        }

        has_charge = self.has_ev_home_ch or self.has_ev_ext_ch
        has_discharge = self.has_ev_dis_house or self.has_ev_dis_grid
        if has_charge and has_discharge:
            # Binary mode variable prevents simultaneous charge and discharge at the EV.
            self.vars["y_ev_mode"] = m.addVars(T, vtype=self.GRB.BINARY, name="y_ev_mode")

        self.constrs: Dict[str, Any] = {
            "c_init_ev": m.addConstr(self.vars["E_ev"][0] == 0.0, name="c_init_ev"),
            "c_home_balance": {},
            "c_ev_dyn": {},
            "c_ev_lb": {},
            "c_ev_lb_terminal": m.addConstr(self.vars["E_ev"][H] >= 0.0, name="c_ev_lb_terminal"),
            "c_export_link": {},
            "c_ev_ch_total": {},
            "c_ev_dis_total": {},
        }

        for t in T:
            p_grid_import = self.vars["p_grid_import"][t]
            p_ev_home_ch = self.vars["p_ev_home_ch"][t] if self.vars["p_ev_home_ch"] is not None else 0.0
            p_ev_dis_house = self.vars["p_ev_dis_house"][t] if self.vars["p_ev_dis_house"] is not None else 0.0
            p_ev_ext_ch = self.vars["p_ev_ext_ch"][t] if self.vars["p_ev_ext_ch"] is not None else 0.0
            p_ev_dis_grid = self.vars["p_ev_dis_grid"][t] if self.vars["p_ev_dis_grid"] is not None else 0.0
            p_grid_export = self.vars["p_grid_export"][t] if self.vars["p_grid_export"] is not None else 0.0

            # Crucial physics: home-side power must balance at each timestep.
            # External charging is intentionally handled outside the home-side balance.
            #the equation is later set equal to the load kw in that timestep
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
            if self.vars["p_ev_home_ch"] is not None and self.vars["p_ev_ext_ch"] is not None:
                # Combined EV charging cannot exceed active charging-point capability.
                self.constrs["c_ev_ch_total"][t] = m.addConstr(
                    p_ev_home_ch + p_ev_ext_ch <= 0.0,
                    name=f"c_ev_ch_total_{t}",
                )
            if self.vars["p_ev_dis_house"] is not None and self.vars["p_ev_dis_grid"] is not None:
                # Combined EV discharging cannot exceed active discharge capability.
                self.constrs["c_ev_dis_total"][t] = m.addConstr(
                    p_ev_dis_house + p_ev_dis_grid <= 0.0,
                    name=f"c_ev_dis_total_{t}",
                )

            charge_term = cfg.ev_eta_ch * (p_ev_home_ch + p_ev_ext_ch)
            discharge_term = (1.0 / cfg.ev_eta_dis) * (p_ev_dis_house + p_ev_dis_grid)
            # Crucial state transition equation for EV energy.
            self.constrs["c_ev_dyn"][t] = m.addConstr(
                self.vars["E_ev"][t + 1] == self.vars["E_ev"][t] + (charge_term - discharge_term) * cfg.dt_hours,
                name=f"c_ev_dyn_{t}",
            )
            self.constrs["c_ev_lb"][t] = m.addConstr(self.vars["E_ev"][t] >= 0.0, name=f"c_ev_lb_{t}")

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
            m.addConstr(self.vars["E_ev"][t] <= cfg.ev_cap_kwh, name=f"c_ev_max_{t}")

        self._set_objective_coeffs(
            home_import_price=[0.0] * H,
            ev_ext_import_price=[0.0] * H,
            ev_export_price=[0.0] * H,
        )
        m.ModelSense = self.GRB.MINIMIZE
        m.update()

    def _set_objective_coeffs(
        self,
        home_import_price: List[float],
        ev_ext_import_price: List[float],
        ev_export_price: List[float],
    ) -> None:
        # Objective is linear operating cost per step.
        dt = self.cfg.dt_hours
        for t in range(self.H):
            self.vars["p_grid_import"][t].Obj = float(home_import_price[t]) * dt
            if self.vars["p_grid_export"] is not None:
                self.vars["p_grid_export"][t].Obj = -float(ev_export_price[t]) * dt
            if self.vars["p_ev_ext_ch"] is not None:
                self.vars["p_ev_ext_ch"][t].Obj = float(ev_ext_import_price[t]) * dt
            if self.vars["p_ev_home_ch"] is not None:
                self.vars["p_ev_home_ch"][t].Obj = 0.0
            if self.vars["p_ev_dis_house"] is not None:
                self.vars["p_ev_dis_house"][t].Obj = 0.0
            if self.vars["p_ev_dis_grid"] is not None:
                self.vars["p_ev_dis_grid"][t].Obj = 0.0
            if self.vars["y_grid_dir"] is not None:
                self.vars["y_grid_dir"][t].Obj = 0.0
            if self.vars["y_ev_mode"] is not None:
                self.vars["y_ev_mode"][t].Obj = 0.0

        for t in range(self.H + 1):
            self.vars["E_ev"][t].Obj = 0.0

    def _set_ub(self, family: str, t: int, ub: float) -> None:
        var_family = self.vars.get(family)
        if var_family is not None:
            var_family[t].UB = max(0.0, float(ub))

    def update(self, window_arrays: Dict[str, List[float]], soc_ev_now: float) -> None:
        """Inject current state and forecast data into mutable constraint RHS/UB values."""
        t0 = perf_counter()
        self._latest_arrays = window_arrays
        # Reset initial EV state for this rolling window.
        self.constrs["c_init_ev"].RHS = float(soc_ev_now)

        for t in range(self.H):
            # Update forecast-dependent constraints for each look-ahead step.
            self.constrs["c_home_balance"][t].RHS = float(window_arrays["load_kw"][t])
            self.constrs["c_ev_dyn"][t].RHS = -float(window_arrays["ev_drive_kwh"][t])
            self.constrs["c_ev_lb"][t].RHS = _effective_ev_lb(self.cfg, window_arrays["ev_reserve_kwh"][t])

            # Update power limits from connection state (home, external, unavailable).
            self._set_ub("p_ev_home_ch", t, window_arrays["ev_p_home_ch_max_kw"][t])
            self._set_ub("p_ev_ext_ch", t, window_arrays["ev_p_ext_ch_max_kw"][t])
            self._set_ub("p_ev_dis_house", t, window_arrays["ev_p_dis_house_max_kw"][t])
            self._set_ub("p_ev_dis_grid", t, window_arrays["ev_p_dis_grid_max_kw"][t])
            if t in self.constrs["c_ev_ch_total"]:
                self.constrs["c_ev_ch_total"][t].RHS = float(window_arrays["ev_p_ch_total_max_kw"][t])
            if t in self.constrs["c_ev_dis_total"]:
                self.constrs["c_ev_dis_total"][t].RHS = float(window_arrays["ev_p_dis_total_max_kw"][t])
        self.constrs["c_ev_lb_terminal"].RHS = self._terminal_reserve_lb(window_arrays["ev_reserve_kwh"])

        self._set_objective_coeffs(
            home_import_price=window_arrays["home_import_price"],
            ev_ext_import_price=window_arrays["ev_ext_import_price"],
            ev_export_price=window_arrays["ev_export_price"],
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

    def extract_first_action(self) -> Dict[str, float]:
        """Return first-step control action in both detailed and compatibility fields."""
        t0 = perf_counter()
        ev_home_ch = self._x("p_ev_home_ch", 0)
        ev_ext_ch = self._x("p_ev_ext_ch", 0)
        ev_dis_house = self._x("p_ev_dis_house", 0)
        ev_dis_grid = self._x("p_ev_dis_grid", 0)
        out = {
            "grid_import_kw": self._x("p_grid_import", 0),
            "grid_export_kw": self._x("p_grid_export", 0),
            "bat_ch_kw": 0.0,
            "bat_dis_kw": 0.0,
            "ev_home_ch_kw": ev_home_ch,
            "ev_ext_ch_kw": ev_ext_ch,
            "ev_dis_to_home_kw": ev_dis_house,
            "ev_dis_to_grid_kw": ev_dis_grid,
            "ev_ch_kw": ev_home_ch + ev_ext_ch,
            "ev_dis_kw": ev_dis_house + ev_dis_grid,
        }
        self.last_extract_time_sec = perf_counter() - t0
        return out

    def extract_solution_full_horizon(self) -> Dict[str, List[float]]:
        t0 = perf_counter()
        out: Dict[str, List[float]] = {}
        for family, var_by_t in self.vars.items():
            if var_by_t is None:
                continue
            out[family] = [float(var_by_t[t].X) for t in sorted(var_by_t.keys())]
        self.last_extract_time_sec = perf_counter() - t0
        return out


def _safe_fallback_step(row: pd.Series, cfg: EnergySystemConfig, e_ev: float) -> Dict[str, float]:
    """Cheap feasible fallback preserving home-vs-external EV channel separation."""
    load_kw = float(row["load_kw"])
    # Minimum EV energy required at this timestep (SOC floor + reserve policy).
    ev_lb = _effective_ev_lb(cfg, float(row["ev_reserve_kwh"]))
    p_ev_home_ch_max = float(row.get("ev_p_home_ch_max_kw", 0.0))
    p_ev_ext_ch_max = float(row.get("ev_p_ext_ch_max_kw", 0.0))
    p_ev_dis_house_max = float(row.get("ev_p_dis_house_max_kw", 0.0))

    p_ev_dis_to_home = 0.0
    if p_ev_dis_house_max > 0.0 and e_ev > ev_lb:
        # Only discharge the EV when we are above the lower bound.
        p_dis_e_limit = max(0.0, (e_ev - ev_lb) * cfg.ev_eta_dis / cfg.dt_hours)
        p_ev_dis_to_home = min(p_ev_dis_house_max, p_dis_e_limit, load_kw)

    p_grid_import = max(0.0, load_kw - p_ev_dis_to_home)
    p_ev_home_ch = 0.0
    p_ev_ext_ch = 0.0

    if e_ev < ev_lb:
        # If EV is below required reserve, prioritize charging back to the floor.
        p_need_kw = (ev_lb - e_ev) / (cfg.ev_eta_ch * cfg.dt_hours)
        home_headroom = max(0.0, cfg.p_grid_max_kw - p_grid_import)
        p_ev_home_ch = min(p_ev_home_ch_max, home_headroom, p_need_kw)
        p_need_kw = max(0.0, p_need_kw - p_ev_home_ch)
        p_ev_ext_ch = min(p_ev_ext_ch_max, p_need_kw)
        p_grid_import += p_ev_home_ch

    p_grid_import = min(cfg.p_grid_max_kw, p_grid_import)
    return {
        "grid_import_kw": p_grid_import,
        "grid_export_kw": 0.0,
        "bat_ch_kw": 0.0,
        "bat_dis_kw": 0.0,
        "ev_home_ch_kw": p_ev_home_ch,
        "ev_ext_ch_kw": p_ev_ext_ch,
        "ev_dis_to_home_kw": p_ev_dis_to_home,
        "ev_dis_to_grid_kw": 0.0,
        "ev_ch_kw": p_ev_home_ch + p_ev_ext_ch,
        "ev_dis_kw": p_ev_dis_to_home,
    }


def _build_and_solve_window_gurobi(
    window: pd.DataFrame,
    cfg: EnergySystemConfig,
    e_ev0: float,
    solver_tee: bool = False,
) -> MPCSolveResult:
    """Legacy mode: rebuild one temporary model per window solve."""
    solver = PersistentMPCSolver(cfg=cfg, H=len(window), solver_tee=solver_tee)
    arrays = WindowDataManager.get_arrays(window, 0, len(window), cfg)
    solver.update(arrays, soc_ev_now=e_ev0)
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
    solver_backend: str = "gurobi",
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

    # `rows`: simulation outputs, `logs`: diagnostics and timings.
    logs: List[Dict[str, object]] = []
    rows: List[Dict[str, float]] = []
    n_steps = len(df)
    t_loop_start = perf_counter()
    slowest_solve_sec = -1.0
    slowest_solve_ts = ""

    backend = solver_backend.strip().lower().replace("_", "-")
    if backend in {"native", "native-gurobipy"}:
        backend = "gurobi"
    if backend != "gurobi":
        raise ValueError(f"Unsupported solver backend: {solver_backend}. Only native gurobipy is supported.")
    # Force API load once so import overhead is visible in logs/progress output.
    backend_init_sec = _load_gurobi_api()["import_seconds"]

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
        print(f"[MPC] Backend: {backend}")
        print(f"[MPC] Backend import/init took {backend_init_sec:.2f}s")
        print(
            f"[MPC] Starting rolling optimization: steps={n_steps}, horizon={cfg.horizon_steps}, "
            f"report_every={progress_every}, slow_step_sec={slow_step_sec:.2f}"
        )

    prev_solution: Optional[Dict[str, List[float]]] = None
    # Receding-horizon loop:
    # solve a window -> apply only first action -> shift window by 1 step.
    for i, (ts, row) in enumerate(df.iterrows()):
        e_ev_start = e_ev
        window = df.iloc[i : i + cfg.horizon_steps]
        step: Dict[str, float] = {}
        status = "not_run"
        t_step_start = perf_counter()
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
                persistent_solver.update(arrays, soc_ev_now=e_ev)
                if use_mip_start and prev_solution is not None:
                    # Warm-start from previous solution to reduce solve time.
                    persistent_solver.apply_mip_start(prev_solution, shift=1)
                update_sec = perf_counter() - t_update

                status = persistent_solver.optimize()
                if persistent_solver.model.SolCount > 0:
                    # Key MPC step: apply only the first control action from the optimal horizon.
                    step = persistent_solver.extract_first_action()
                    extract_sec = persistent_solver.last_extract_time_sec
                    if use_mip_start:
                        prev_solution = persistent_solver.extract_solution_full_horizon()
                else:
                    # If optimization returns no solution, fall back to a safe heuristic.
                    step = _safe_fallback_step(row, cfg, e_ev)
                    used_fallback = 1
                    logs.append({"timestamp": str(ts), "step": i + 1, "status": status, "action": "fallback"})
            else:
                # Legacy mode: rebuild model per window (slower, but useful for regression comparison).
                solved = _build_and_solve_window_gurobi(window, cfg, e_ev, solver_tee=solver_tee)
                status = solved.status
                if solved.first_step:
                    step = solved.first_step
                    if use_mip_start:
                        prev_solution = solved.full_solution
                else:
                    step = _safe_fallback_step(row, cfg, e_ev)
                    used_fallback = 1
                    logs.append({"timestamp": str(ts), "step": i + 1, "status": status, "action": "fallback"})
        except Exception as ex:
            # Defensive behavior: never stop the whole run because one step failed.
            status = f"error:{ex}"
            step = _safe_fallback_step(row, cfg, e_ev)
            used_fallback = 1
            logs.append({"timestamp": str(ts), "step": i + 1, "status": status, "action": "fallback"})

        # Track pure optimization time and full step time separately.
        solve_sec = perf_counter() - t_solve_start
        step_total_sec = perf_counter() - t_step_start
        if solve_sec > slowest_solve_sec:
            slowest_solve_sec = solve_sec
            slowest_solve_ts = str(ts)

        drive_kwh = float(row["ev_drive_kwh"])
        # Crucial state transition: previous energy + charging - discharging - driving consumption.
        e_ev_raw = (
            e_ev
            + (cfg.ev_eta_ch * (step["ev_home_ch_kw"] + step["ev_ext_ch_kw"]))
            * cfg.dt_hours
            - ((step["ev_dis_to_home_kw"] + step["ev_dis_to_grid_kw"]) / cfg.ev_eta_dis) * cfg.dt_hours
            - drive_kwh
        )

        # Clamp to feasible range; keep delta for debugging whether clamping happened.
        ev_lb_now = _effective_ev_lb(cfg, float(row["ev_reserve_kwh"]))
        e_ev = min(max(e_ev_raw, ev_lb_now), cfg.ev_cap_kwh)
        ev_soc_clamp_delta_kwh = e_ev - e_ev_raw
        ev_soc_clamped = float(abs(ev_soc_clamp_delta_kwh) > 1e-9)
        ev_soc_clamped_after_fallback = float(used_fallback and ev_soc_clamped > 0.5)
        e_bat = min(max(e_bat, cfg.bat_soc_min * cfg.bat_cap_kwh), cfg.bat_soc_max * cfg.bat_cap_kwh)

        step.update(
            {
                "bat_energy_kwh": e_bat,
                "ev_energy_kwh": e_ev_start,
                "ev_energy_pre_clamp_kwh": e_ev_raw,
                "ev_soc_clamped": ev_soc_clamped,
                "ev_soc_clamp_delta_kwh": ev_soc_clamp_delta_kwh,
                "ev_soc_clamped_after_fallback": ev_soc_clamped_after_fallback,
                "ev_soc_clamp_after_fallback_delta_kwh": ev_soc_clamp_delta_kwh if ev_soc_clamped_after_fallback > 0.5 else 0.0,
                "ev_consumption_kwh": drive_kwh,
                "solver_status": status,
                "used_fallback": float(used_fallback),
            }
        )
        rows.append(step)
        # Keep one compact diagnostic row per solved timestep.
        logs.append(
            {
                "timestamp": str(ts),
                "step": i + 1,
                "total_steps": n_steps,
                "window_steps": len(window),
                "status": status,
                "build_once_time_sec": round(build_once_sec, 6),
                "update_seconds": round(update_sec, 6),
                "solve_seconds": round(solve_sec, 4),
                "extract_seconds": round(extract_sec, 6),
                "step_total_seconds": round(step_total_sec, 4),
                "used_fallback": int(used_fallback),
                "ev_soc_clamped": int(ev_soc_clamped > 0.5),
                "ev_soc_clamp_delta_kwh": round(ev_soc_clamp_delta_kwh, 6),
                "ev_soc_clamped_after_fallback": int(ev_soc_clamped_after_fallback > 0.5),
            }
        )

        if show_progress:
            completed = i + 1
            # Print first/last/periodic updates and any unusually slow solve.
            should_report = (
                completed == 1
                or completed == n_steps
                or completed % max(1, progress_every) == 0
                or solve_sec >= slow_step_sec
            )
            if should_report:
                elapsed = perf_counter() - t_loop_start
                avg_step_sec = elapsed / completed
                eta_sec = avg_step_sec * (n_steps - completed)
                print(
                    f"[MPC] {completed}/{n_steps} ({100.0 * completed / n_steps:.1f}%) "
                    f"ts={ts} solve={solve_sec:.2f}s step={step_total_sec:.2f}s avg={avg_step_sec:.2f}s "
                    f"eta={eta_sec / 60.0:.1f}m status={status}"
                )

    if show_progress:
        total_elapsed_sec = perf_counter() - t_loop_start
        print(
            f"[MPC] Finished {n_steps} steps in {total_elapsed_sec / 60.0:.2f} min. "
            f"Slowest solve: {slowest_solve_sec:.2f}s at {slowest_solve_ts}"
        )

    out = pd.DataFrame(rows, index=df.index)
    out["grid_import_kwh"] = out["grid_import_kw"] * cfg.dt_hours
    out["grid_export_kwh"] = out["grid_export_kw"] * cfg.dt_hours
    out["ev_home_ch_kwh"] = out["ev_home_ch_kw"] * cfg.dt_hours
    out["ev_ext_ch_kwh"] = out["ev_ext_ch_kw"] * cfg.dt_hours
    out["ev_dis_to_home_kwh"] = out["ev_dis_to_home_kw"] * cfg.dt_hours
    out["ev_dis_to_grid_kwh"] = out["ev_dis_to_grid_kw"] * cfg.dt_hours
    out["ext_charge_cost_eur"] = df["ev_ext_import_price_eur_per_kwh"] * out["ev_ext_ch_kwh"]
    out["step_cost_eur"] = (
        df["import_price_eur_per_kwh"] * out["grid_import_kwh"]
        - df["ev_export_price_eur_per_kwh"] * out["grid_export_kwh"]
        + out["ext_charge_cost_eur"]
    )
    out["ev_state"] = df["ev_state"]
    out["charging_point_effective"] = df["charging_point_effective"]
    return out, logs
