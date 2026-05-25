from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from home_energy_opt.config import EnergySystemConfig
from home_energy_opt.metrics import home_grid_import_kwh, step_cost_eur

EV_SOC_CLAMP_EPS_KWH = 1e-5


def _effective_ev_lb(cfg: EnergySystemConfig, reserve_kwh: float) -> float:
    """Single EV lower bound per step from provided reserve, bounded to physical limits."""
    return min(max(float(reserve_kwh), 0.0), cfg.ev_cap_kwh)


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
    build_seconds: float = 0.0
    solve_seconds: float = 0.0
    extract_seconds: float = 0.0


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
            tail["ev_drive_kwh"] = 0.0
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


class HiGHSMPCSolver:
    """Single-window MILP solver using SciPy's HiGHS backend."""

    def __init__(self, cfg: EnergySystemConfig, H: int = 96):
        self.cfg = cfg
        self.H = H
        self.has_grid_export = bool(cfg.enable_grid_export and cfg.enable_ev_discharge_to_grid)
        self.has_ev_home_ch = bool(cfg.enable_ev_home_charge and cfg.max_ev_home_charge_kw > 0.0)
        self.has_ev_ext_ch = bool(cfg.enable_ev_external_charge and cfg.max_ev_external_charge_kw > 0.0)
        self.has_ev_dis_house = bool(cfg.enable_ev_discharge_to_house and cfg.max_ev_discharge_house_kw > 0.0)
        self.has_ev_dis_grid = bool(self.has_grid_export and cfg.max_ev_discharge_grid_kw > 0.0)
        self.has_charge = self.has_ev_home_ch or self.has_ev_ext_ch
        self.has_discharge = self.has_ev_dis_house or self.has_ev_dis_grid
        self.var_index: Dict[str, np.ndarray] = {}
        self.integrality: np.ndarray
        self.lb: np.ndarray
        self.ub: np.ndarray
        self._build_variable_layout()

    def _add_family(self, name: str, size: int, lower: float, upper: float, integral: bool = False) -> None:
        if size <= 0:
            return
        start = int(len(self.lb_list))
        idx = np.arange(start, start + size, dtype=int)
        self.var_index[name] = idx
        self.lb_list.extend([float(lower)] * size)
        self.ub_list.extend([float(upper)] * size)
        self.integrality_list.extend([1 if integral else 0] * size)

    def _build_variable_layout(self) -> None:
        self.lb_list: List[float] = []
        self.ub_list: List[float] = []
        self.integrality_list: List[int] = []

        self._add_family("p_grid_import", self.H, 0.0, self.cfg.p_grid_max_kw)
        if self.has_grid_export:
            self._add_family("p_grid_export", self.H, 0.0, self.cfg.p_grid_max_kw)
            self._add_family("y_grid_dir", self.H, 0.0, 1.0, integral=True)
        if self.has_ev_home_ch:
            self._add_family("p_ev_home_ch", self.H, 0.0, self.cfg.max_ev_home_charge_kw)
        if self.has_ev_ext_ch:
            self._add_family("p_ev_ext_ch", self.H, 0.0, self.cfg.max_ev_external_charge_kw)
        if self.has_ev_dis_house:
            self._add_family("p_ev_dis_house", self.H, 0.0, self.cfg.max_ev_discharge_house_kw)
            self._add_family("p_ev_dis_house_home", self.H, 0.0, self.cfg.max_ev_discharge_house_kw)
            self._add_family("p_ev_dis_house_ext", self.H, 0.0, self.cfg.max_ev_discharge_house_kw)
        if self.has_ev_dis_grid:
            self._add_family("p_ev_dis_grid", self.H, 0.0, self.cfg.max_ev_discharge_grid_kw)
            self._add_family("p_ev_dis_grid_home", self.H, 0.0, self.cfg.max_ev_discharge_grid_kw)
            self._add_family("p_ev_dis_grid_ext", self.H, 0.0, self.cfg.max_ev_discharge_grid_kw)
        self._add_family("p_ev_drive_home", self.H, 0.0, self.cfg.ev_cap_kwh)
        self._add_family("p_ev_drive_ext", self.H, 0.0, self.cfg.ev_cap_kwh)
        if self.has_charge and self.has_discharge:
            self._add_family("y_ev_mode", self.H, 0.0, 1.0, integral=True)
        self._add_family("E_ev_home", self.H + 1, 0.0, self.cfg.ev_cap_kwh)
        self._add_family("E_ev_external", self.H + 1, 0.0, self.cfg.ev_cap_kwh)

        self.lb = np.asarray(self.lb_list, dtype=float)
        self.ub = np.asarray(self.ub_list, dtype=float)
        self.integrality = np.asarray(self.integrality_list, dtype=int)

    def _col(self, family: str, t: int) -> int:
        return int(self.var_index[family][t])

    def _build_problem(
        self,
        window_arrays: Dict[str, List[float]],
        soc_ev_home_now: float,
        soc_ev_external_now: float,
    ) -> Tuple[np.ndarray, LinearConstraint, float]:
        build_t0 = perf_counter()
        c = np.zeros(len(self.lb), dtype=float)
        row_idx: List[int] = []
        col_idx: List[int] = []
        data: List[float] = []
        row_lb: List[float] = []
        row_ub: List[float] = []

        def add_row(terms: List[Tuple[str, int, float]], lower: float, upper: float) -> None:
            r = len(row_lb)
            for family, t, coef in terms:
                row_idx.append(r)
                col_idx.append(self._col(family, t))
                data.append(float(coef))
            row_lb.append(float(lower))
            row_ub.append(float(upper))

        dt = self.cfg.dt_hours
        ev_degradation_cost = float(self.cfg.ev_degradation_eur_per_kwh_charged) * dt
        home_import_price = window_arrays["home_import_price"]
        ev_home_import_price = window_arrays["ev_home_import_price"]
        ev_ext_import_price = window_arrays["ev_ext_import_price"]
        home_export_price = window_arrays["ev_home_export_price"]
        external_export_price = window_arrays["ev_external_export_price"]

        c[self.var_index["E_ev_home"][0]] = 0.0
        c[self.var_index["E_ev_external"][0]] = 0.0

        add_row([("E_ev_home", 0, 1.0)], float(soc_ev_home_now), float(soc_ev_home_now))
        add_row([("E_ev_external", 0, 1.0)], float(soc_ev_external_now), float(soc_ev_external_now))

        for t in range(self.H):
            c[self._col("p_grid_import", t)] = float(home_import_price[t]) * dt
            if self.has_grid_export:
                c[self._col("p_grid_export", t)] = 0.0
            if self.has_ev_home_ch:
                c[self._col("p_ev_home_ch", t)] = (float(ev_home_import_price[t]) - float(home_import_price[t])) * dt + ev_degradation_cost
            if self.has_ev_ext_ch:
                c[self._col("p_ev_ext_ch", t)] = float(ev_ext_import_price[t]) * dt + ev_degradation_cost
            if self.has_ev_dis_grid:
                c[self._col("p_ev_dis_grid_home", t)] = -float(home_export_price[t]) * dt
                c[self._col("p_ev_dis_grid_ext", t)] = -float(external_export_price[t]) * dt
            if self.has_ev_dis_house:
                c[self._col("p_ev_dis_house_home", t)] = 0.0
                c[self._col("p_ev_dis_house_ext", t)] = 0.0
            c[self._col("p_ev_drive_home", t)] = 0.0
            c[self._col("p_ev_drive_ext", t)] = 0.0
            if self.has_grid_export:
                c[self._col("y_grid_dir", t)] = 0.0
            if self.has_charge and self.has_discharge:
                c[self._col("y_ev_mode", t)] = 0.0

            add_row(
                [("p_grid_import", t, 1.0)]
                + ([("p_ev_dis_house", t, 1.0)] if self.has_ev_dis_house else [])
                + ([("p_ev_home_ch", t, -1.0)] if self.has_ev_home_ch else []),
                float(window_arrays["load_kw"][t]),
                float(window_arrays["load_kw"][t]),
            )

            if self.has_grid_export and self.has_ev_dis_grid:
                add_row([("p_grid_export", t, 1.0), ("p_ev_dis_grid", t, -1.0)], 0.0, 0.0)

            if self.has_ev_dis_house:
                add_row(
                    [("p_ev_dis_house", t, 1.0), ("p_ev_dis_house_home", t, -1.0), ("p_ev_dis_house_ext", t, -1.0)],
                    0.0,
                    0.0,
                )
                add_row(
                    [
                        ("p_ev_dis_house_home", t, 1.0),
                        ("E_ev_home", t, -(self.cfg.ev_eta_dis / dt)),
                    ],
                    -np.inf,
                    0.0,
                )
                add_row(
                    [
                        ("p_ev_dis_house_ext", t, 1.0),
                        ("E_ev_external", t, -(self.cfg.ev_eta_dis / dt)),
                    ],
                    -np.inf,
                    0.0,
                )

            if self.has_ev_dis_grid:
                add_row(
                    [("p_ev_dis_grid", t, 1.0), ("p_ev_dis_grid_home", t, -1.0), ("p_ev_dis_grid_ext", t, -1.0)],
                    0.0,
                    0.0,
                )
                add_row(
                    [
                        ("p_ev_dis_grid_home", t, 1.0),
                        ("E_ev_home", t, -(self.cfg.ev_eta_dis / dt)),
                    ],
                    -np.inf,
                    0.0,
                )
                add_row(
                    [
                        ("p_ev_dis_grid_ext", t, 1.0),
                        ("E_ev_external", t, -(self.cfg.ev_eta_dis / dt)),
                    ],
                    -np.inf,
                    0.0,
                )

            if self.has_ev_home_ch and self.has_ev_ext_ch:
                add_row([("p_ev_home_ch", t, 1.0), ("p_ev_ext_ch", t, 1.0)], -np.inf, float(self.cfg.ev_cap_kwh))

            if self.has_ev_dis_house and self.has_ev_dis_grid:
                add_row([("p_ev_dis_house", t, 1.0), ("p_ev_dis_grid", t, 1.0)], -np.inf, float(self.cfg.ev_cap_kwh))

            add_row([("p_ev_drive_home", t, 1.0), ("p_ev_drive_ext", t, 1.0)], 0.0, 0.0)
            add_row([("p_ev_drive_ext", t, 1.0), ("E_ev_external", t, -1.0)], -np.inf, 0.0)
            add_row([("p_ev_drive_home", t, 1.0), ("E_ev_home", t, -1.0)], -np.inf, 0.0)

            home_dyn_terms: List[Tuple[str, int, float]] = [
                ("E_ev_home", t + 1, 1.0),
                ("E_ev_home", t, -1.0),
                ("p_ev_drive_home", t, 1.0),
            ]
            if self.has_ev_home_ch:
                home_dyn_terms.append(("p_ev_home_ch", t, -self.cfg.ev_eta_ch * dt))
            if self.has_ev_dis_house:
                home_dyn_terms.append(("p_ev_dis_house_home", t, dt / self.cfg.ev_eta_dis))
            if self.has_ev_dis_grid:
                home_dyn_terms.append(("p_ev_dis_grid_home", t, dt / self.cfg.ev_eta_dis))
            add_row(home_dyn_terms, 0.0, 0.0)

            external_dyn_terms: List[Tuple[str, int, float]] = [
                ("E_ev_external", t + 1, 1.0),
                ("E_ev_external", t, -1.0),
                ("p_ev_drive_ext", t, 1.0),
            ]
            if self.has_ev_ext_ch:
                external_dyn_terms.append(("p_ev_ext_ch", t, -self.cfg.ev_eta_ch * dt))
            if self.has_ev_dis_house:
                external_dyn_terms.append(("p_ev_dis_house_ext", t, dt / self.cfg.ev_eta_dis))
            if self.has_ev_dis_grid:
                external_dyn_terms.append(("p_ev_dis_grid_ext", t, dt / self.cfg.ev_eta_dis))
            add_row(external_dyn_terms, 0.0, 0.0)

            if self.has_grid_export:
                add_row([("p_grid_import", t, 1.0), ("y_grid_dir", t, -self.cfg.p_grid_max_kw)], -np.inf, 0.0)
                add_row(
                    [("p_grid_export", t, 1.0), ("y_grid_dir", t, self.cfg.p_grid_max_kw)],
                    -np.inf,
                    float(self.cfg.p_grid_max_kw),
                )

            if self.has_charge and self.has_discharge:
                y = "y_ev_mode"
                if self.has_ev_home_ch:
                    add_row([("p_ev_home_ch", t, 1.0), (y, t, -self.cfg.max_ev_home_charge_kw)], -np.inf, 0.0)
                if self.has_ev_ext_ch:
                    add_row([("p_ev_ext_ch", t, 1.0), (y, t, -self.cfg.max_ev_external_charge_kw)], -np.inf, 0.0)
                if self.has_ev_dis_house:
                    add_row([("p_ev_dis_house", t, 1.0), (y, t, self.cfg.max_ev_discharge_house_kw)], -np.inf, float(self.cfg.max_ev_discharge_house_kw))
                if self.has_ev_dis_grid:
                    add_row([("p_ev_dis_grid", t, 1.0), (y, t, self.cfg.max_ev_discharge_grid_kw)], -np.inf, float(self.cfg.max_ev_discharge_grid_kw))

        for t in range(self.H + 1):
            reserve_lb = max(0.0, _effective_ev_lb(self.cfg, window_arrays["ev_reserve_kwh"][t]) - EV_SOC_CLAMP_EPS_KWH)
            add_row([("E_ev_home", t, 1.0), ("E_ev_external", t, 1.0)], reserve_lb, np.inf)
            add_row([("E_ev_home", t, 1.0), ("E_ev_external", t, 1.0)], -np.inf, float(self.cfg.ev_cap_kwh))

        A = coo_matrix((data, (row_idx, col_idx)), shape=(len(row_lb), len(self.lb)))
        constraint = LinearConstraint(A, np.asarray(row_lb, dtype=float), np.asarray(row_ub, dtype=float))
        build_seconds = perf_counter() - build_t0
        return c, constraint, build_seconds

    def solve(self, window_arrays: Dict[str, List[float]], soc_ev_home_now: float, soc_ev_external_now: float) -> MPCSolveResult:
        t0 = perf_counter()
        c, constraint, build_seconds = self._build_problem(window_arrays, soc_ev_home_now, soc_ev_external_now)
        build_and_setup_seconds = perf_counter() - t0
        t1 = perf_counter()
        options: Dict[str, float | bool] = {"presolve": True}
        if self.cfg.milp_time_limit_sec is not None:
            options["time_limit"] = float(self.cfg.milp_time_limit_sec)
        if self.cfg.milp_rel_gap is not None:
            options["mip_rel_gap"] = float(self.cfg.milp_rel_gap)

        result = milp(
            c=c,
            integrality=self.integrality,
            bounds=Bounds(self.lb, self.ub),
            constraints=constraint,
            options=options,
        )
        solve_seconds = perf_counter() - t1
        status = self._status_from_result(result)
        if result.x is None or not np.all(np.isfinite(result.x)):
            return MPCSolveResult(first_step={}, status=status, build_seconds=build_and_setup_seconds, solve_seconds=solve_seconds)

        extract_t0 = perf_counter()
        solution = self._extract_solution(np.asarray(result.x, dtype=float))
        first_step = self.extract_action(solution, 0)
        extract_seconds = perf_counter() - extract_t0
        return MPCSolveResult(
            first_step=first_step,
            status=status,
            full_solution=solution,
            build_seconds=build_and_setup_seconds,
            solve_seconds=solve_seconds,
            extract_seconds=extract_seconds,
        )

    @staticmethod
    def _status_from_result(result) -> str:
        if getattr(result, "success", False) and getattr(result, "status", None) == 0:
            return "optimal"
        x = getattr(result, "x", None)
        if x is not None:
            return "feasible"
        return f"status_{getattr(result, 'status', 'unknown')}"

    def _extract_solution(self, x: np.ndarray) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for family, idx in self.var_index.items():
            out[family] = [float(x[i]) for i in idx]
        return out

    def extract_action(self, solution: Dict[str, List[float]], t: int = 0) -> Dict[str, float]:
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

    def extract_full_horizon(self, solution: Dict[str, List[float]]) -> Dict[str, List[float]]:
        return {family: [float(v) for v in values] for family, values in solution.items()}


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


def run_mpc_loop(
    df: pd.DataFrame,
    cfg: EnergySystemConfig,
    show_progress: bool = True,
    progress_every: int = 50,
    slow_step_sec: float = 2.0,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    """Run rolling-horizon MPC and return per-step actions plus solver logs."""
    if cfg.enable_pv or cfg.enable_battery:
        raise NotImplementedError("Versions with PV or stationary battery are prepared in config but not implemented yet.")

    e_bat = cfg.bat_soc_init * cfg.bat_cap_kwh
    e_ev = cfg.ev_soc_init * cfg.ev_cap_kwh
    n_steps = len(df)
    if n_steps > 0 and str(df.iloc[0].get("ev_state", "")).strip().lower() == "home":
        e_ev_home = e_ev
        e_ev_external = 0.0
    else:
        e_ev_home = 0.0
        e_ev_external = e_ev

    solver = HiGHSMPCSolver(cfg=cfg, H=cfg.horizon_steps)
    logs: List[Dict[str, object]] = []
    rows: List[Dict[str, float]] = []
    ev_reserve_kwh_by_step = pd.to_numeric(df["ev_reserve_kwh"], errors="coerce").fillna(0.0).astype(float).tolist()
    t_loop_start = perf_counter()
    slowest_solve_sec = -1.0
    slowest_solve_ts = ""

    if show_progress:
        print(
            f"[MPC] Starting rolling optimization (HiGHS): steps={n_steps}, horizon={cfg.horizon_steps}, "
            f"apply_steps={max(1, int(cfg.mpc_apply_steps))}, report_every={progress_every}, slow_step_sec={slow_step_sec:.2f}"
        )

    prev_e_ev_raw = e_ev
    prev_used_fallback = 0
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
        t_solve_start = perf_counter()

        try:
            arrays = WindowDataManager.get_arrays(df, i, cfg.horizon_steps, cfg)
            t_update = perf_counter()
            solved = solver.solve(arrays, soc_ev_home_now=e_ev_home, soc_ev_external_now=e_ev_external)
            update_sec = solved.build_seconds
            status = solved.status
            if solved.full_solution is not None:
                block_actions = [solver.extract_action(solved.full_solution, t) for t in range(apply_steps)]
                extract_sec = solved.extract_seconds
            else:
                used_fallback = 1
                logs.append({"timestamp": str(ts), "step": i + 1, "status": status, "action": "fallback"})
        except Exception as ex:
            status = f"error:{ex}"
            used_fallback = 1
            logs.append({"timestamp": str(ts), "step": i + 1, "status": status, "action": "fallback"})

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
            step = block_actions[local_t] if local_t < len(block_actions) else _safe_fallback_step(row, cfg, e_ev_home, e_ev_external)

            drive_kwh = float(row["ev_drive_kwh"])
            drive_external = min(drive_kwh, e_ev_external_start)
            drive_home = max(0.0, drive_kwh - drive_external)
            home_charge_kwh = float(step["ev_home_ch_kw"]) * cfg.dt_hours
            external_charge_kwh = float(step["ev_ext_ch_kw"]) * cfg.dt_hours
            dis_home_store_kwh = (float(step["ev_dis_to_home_home_kwh"]) + float(step["ev_dis_to_grid_home_kwh"])) / cfg.ev_eta_dis
            dis_external_store_kwh = (float(step["ev_dis_to_home_external_kwh"]) + float(step["ev_dis_to_grid_external_kwh"])) / cfg.ev_eta_dis
            e_ev_home_next = max(0.0, e_ev_home_start + cfg.ev_eta_ch * home_charge_kwh - dis_home_store_kwh - drive_home)
            e_ev_external_next = max(0.0, e_ev_external_start + cfg.ev_eta_ch * external_charge_kwh - dis_external_store_kwh - drive_external)
            e_ev_raw_next = e_ev_home_next + e_ev_external_next

            if j + 1 < n_steps:
                ev_lb_carry = _effective_ev_lb(cfg, ev_reserve_kwh_by_step[j + 1])
            else:
                ev_lb_carry = _effective_ev_lb(cfg, ev_reserve_kwh_by_step[j])
            e_ev = min(max(e_ev_raw_next, ev_lb_carry), cfg.ev_cap_kwh)
            e_ev_home, e_ev_external = _scale_split_to_total(e_ev_home_next, e_ev_external_next, e_ev)

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
                    "build_once_time_sec": 0.0,
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
    ev_home_import_price = pd.to_numeric(df.get("ev_home_import_price_eur_per_kwh", df["import_price_eur_per_kwh"]), errors="coerce").fillna(0.0)
    out["ext_charge_cost_eur"] = df["ev_ext_import_price_eur_per_kwh"] * out["ev_ext_ch_kwh"]
    out["ev_battery_degradation_cost_eur"] = float(cfg.ev_degradation_eur_per_kwh_charged) * (out["ev_home_ch_kwh"] + out["ev_ext_ch_kwh"])
    out["home_load_grid_import_kwh"] = home_grid_import_kwh(out["home_load_kwh"], out["ev_dis_to_home_kwh"])
    out["ev_home_charge_cost_eur"] = ev_home_import_price * out["ev_home_ch_kwh"]
    out["home_load_cost_eur"] = home_grid_price * out["home_load_grid_import_kwh"]
    out["home_grid_price_total_eur_per_kwh"] = home_grid_price
    out["ev_home_import_price_eur_per_kwh"] = ev_home_import_price
    out["ev_home_export_price_eur_per_kwh"] = df["ev_home_export_price_eur_per_kwh"]
    out["ev_external_export_price_eur_per_kwh"] = df["ev_external_export_price_eur_per_kwh"]
    out["ev_export_price_eur_per_kwh"] = df["ev_export_price_eur_per_kwh"]
    out["ev_discharge_grid_revenue_home_eur"] = df["ev_home_export_price_eur_per_kwh"] * out["ev_dis_to_grid_home_kwh"]
    out["ev_discharge_grid_revenue_external_eur"] = df["ev_external_export_price_eur_per_kwh"] * out["ev_dis_to_grid_external_kwh"]
    out["ev_discharge_grid_revenue_eur"] = out["ev_discharge_grid_revenue_home_eur"] + out["ev_discharge_grid_revenue_external_eur"]
    out["step_cost_eur"] = step_cost_eur(
        home_grid_price=home_grid_price,
        home_load_kwh=out["home_load_kwh"],
        ev_dis_to_home_kwh=out["ev_dis_to_home_kwh"],
        ev_home_import_price=ev_home_import_price,
        ev_home_ch_kwh=out["ev_home_ch_kwh"],
        ext_charge_cost_eur=out["ext_charge_cost_eur"],
        ev_battery_degradation_cost_eur=out["ev_battery_degradation_cost_eur"],
        ev_discharge_grid_revenue_eur=out["ev_discharge_grid_revenue_eur"],
    )
    out["ev_reserve_kwh"] = df["ev_reserve_kwh"]
    out["ev_state"] = df["ev_state"]
    out["charging_point_effective"] = df["charging_point_effective"]
    return out, logs
