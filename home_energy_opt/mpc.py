from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

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


def _shift_sequence(values: Sequence[float], shift: int) -> List[float]:
    """Shift one sequence left and pad the tail with the last known value."""
    arr = [float(v) for v in values]
    if not arr:
        return []
    shift = max(0, min(int(shift), len(arr)))
    if shift == 0:
        return arr
    tail = arr[-1]
    shifted = arr[shift:]
    shifted.extend([tail] * shift)
    return shifted[: len(arr)]


@dataclass
class MPCSolveResult:
    """Container for one MPC window solve outcome."""

    first_step: Dict[str, float]
    status: str
    full_solution: Optional[Dict[str, List[float]]] = None
    build_seconds: float = 0.0
    solve_seconds: float = 0.0
    extract_seconds: float = 0.0


@dataclass
class _RowSpec:
    name: str
    lower: float
    upper: float
    terms: List[Tuple[str, int, float]]


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


def _load_highs_api() -> Dict[str, Any]:
    """Lazy-load highspy once and cache references for fast reuse."""
    global _HIGHSPY_API
    try:
        return _HIGHSPY_API
    except NameError:
        pass

    t0 = perf_counter()
    try:
        import highspy  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "highspy is required for the persistent MPC solver. Install it with `pip install highspy`."
        ) from exc

    _HIGHSPY_API = {
        "highspy": highspy,
        "Highs": highspy.Highs,
        "HighsLp": highspy.HighsLp,
        "HighsSolution": highspy.HighsSolution,
        "HighsVarType": highspy.HighsVarType,
        "ObjSense": highspy.ObjSense,
        "MatrixFormat": highspy.MatrixFormat,
        "kHighsInf": getattr(highspy, "kHighsInf", float("inf")),
        "import_seconds": perf_counter() - t0,
    }
    return _HIGHSPY_API


class HighsPersistentMPCSolver:
    """Persistent HiGHS model for the current MPC formulation."""

    def __init__(self, cfg: EnergySystemConfig, H: int = 96, solver_tee: bool = False):
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
        self.row_index: Dict[str, int] = {}
        self._solution_cache: Optional[Dict[str, List[float]]] = None

        api = _load_highs_api()
        self.highspy = api["highspy"]
        self.Highs = api["Highs"]
        self.HighsSolution = api["HighsSolution"]
        self.HighsVarType = api["HighsVarType"]
        self.kHighsInf = api["kHighsInf"]

        self.build_once_time_sec = 0.0
        self.last_update_time_sec = 0.0
        self.last_optimize_time_sec = 0.0
        self.last_extract_time_sec = 0.0

        t0 = perf_counter()
        self._build_variable_layout()
        self._build_model(solver_tee=solver_tee)
        self.build_once_time_sec = perf_counter() - t0

    def _add_family(self, name: str, size: int, lower: float, upper: float, integral: bool = False) -> None:
        if size <= 0:
            return
        start = len(self._col_names)
        idx = np.arange(start, start + size, dtype=int)
        self.var_index[name] = idx
        self._col_names.extend([name] * size)
        self._col_lower.extend([float(lower)] * size)
        self._col_upper.extend([float(upper)] * size)
        self._col_integrality.extend([self.HighsVarType.kInteger if integral else self.HighsVarType.kContinuous] * size)

    def _build_variable_layout(self) -> None:
        self._col_names: List[str] = []
        self._col_lower: List[float] = []
        self._col_upper: List[float] = []
        self._col_integrality: List[Any] = []

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

        self.num_cols = len(self._col_names)

    def _col(self, family: str, t: int) -> int:
        return int(self.var_index[family][t])

    def _add_row(self, rows: List[_RowSpec], name: str, lower: float, upper: float, terms: List[Tuple[str, int, float]]) -> None:
        rows.append(_RowSpec(name=name, lower=float(lower), upper=float(upper), terms=[(v, int(t), float(c)) for v, t, c in terms]))

    def _build_row_specs(self) -> List[_RowSpec]:
        rows: List[_RowSpec] = []
        self._add_row(rows, "c_init_ev_home", 0.0, 0.0, [("E_ev_home", 0, 1.0)])
        self._add_row(rows, "c_init_ev_external", 0.0, 0.0, [("E_ev_external", 0, 1.0)])

        for t in range(self.H):
            self._add_row(
                rows,
                f"c_home_balance_{t}",
                0.0,
                0.0,
                [("p_grid_import", t, 1.0)]
                + ([("p_ev_dis_house", t, 1.0)] if self.has_ev_dis_house else [])
                + ([("p_ev_home_ch", t, -1.0)] if self.has_ev_home_ch else []),
            )
            if self.has_grid_export and self.has_ev_dis_grid:
                self._add_row(rows, f"c_export_link_{t}", 0.0, 0.0, [("p_grid_export", t, 1.0), ("p_ev_dis_grid", t, -1.0)])
            if self.has_ev_dis_house:
                self._add_row(rows, f"c_ev_dis_house_split_{t}", 0.0, 0.0, [("p_ev_dis_house", t, 1.0), ("p_ev_dis_house_home", t, -1.0), ("p_ev_dis_house_ext", t, -1.0)])
                self._add_row(rows, f"c_ev_dis_house_home_cap_{t}", -self.kHighsInf, 0.0, [("p_ev_dis_house_home", t, 1.0), ("E_ev_home", t, -(self.cfg.ev_eta_dis / self.cfg.dt_hours))])
                self._add_row(rows, f"c_ev_dis_house_ext_cap_{t}", -self.kHighsInf, 0.0, [("p_ev_dis_house_ext", t, 1.0), ("E_ev_external", t, -(self.cfg.ev_eta_dis / self.cfg.dt_hours))])
            if self.has_ev_dis_grid:
                self._add_row(rows, f"c_ev_dis_grid_split_{t}", 0.0, 0.0, [("p_ev_dis_grid", t, 1.0), ("p_ev_dis_grid_home", t, -1.0), ("p_ev_dis_grid_ext", t, -1.0)])
                self._add_row(rows, f"c_ev_dis_grid_home_cap_{t}", -self.kHighsInf, 0.0, [("p_ev_dis_grid_home", t, 1.0), ("E_ev_home", t, -(self.cfg.ev_eta_dis / self.cfg.dt_hours))])
                self._add_row(rows, f"c_ev_dis_grid_ext_cap_{t}", -self.kHighsInf, 0.0, [("p_ev_dis_grid_ext", t, 1.0), ("E_ev_external", t, -(self.cfg.ev_eta_dis / self.cfg.dt_hours))])
            if self.has_ev_home_ch and self.has_ev_ext_ch:
                self._add_row(rows, f"c_ev_charge_total_{t}", -self.kHighsInf, float(self.cfg.ev_cap_kwh), [("p_ev_home_ch", t, 1.0), ("p_ev_ext_ch", t, 1.0)])
            if self.has_ev_dis_house and self.has_ev_dis_grid:
                self._add_row(rows, f"c_ev_dis_total_{t}", -self.kHighsInf, float(self.cfg.ev_cap_kwh), [("p_ev_dis_house", t, 1.0), ("p_ev_dis_grid", t, 1.0)])
            self._add_row(rows, f"c_drive_split_{t}", 0.0, 0.0, [("p_ev_drive_home", t, 1.0), ("p_ev_drive_ext", t, 1.0)])
            self._add_row(rows, f"c_drive_external_cap_{t}", -self.kHighsInf, 0.0, [("p_ev_drive_ext", t, 1.0), ("E_ev_external", t, -1.0)])
            self._add_row(rows, f"c_drive_home_cap_{t}", -self.kHighsInf, 0.0, [("p_ev_drive_home", t, 1.0), ("E_ev_home", t, -1.0)])

            home_dyn_terms: List[Tuple[str, int, float]] = [("E_ev_home", t + 1, 1.0), ("E_ev_home", t, -1.0), ("p_ev_drive_home", t, 1.0)]
            if self.has_ev_home_ch:
                home_dyn_terms.append(("p_ev_home_ch", t, -self.cfg.ev_eta_ch * self.cfg.dt_hours))
            if self.has_ev_dis_house:
                home_dyn_terms.append(("p_ev_dis_house_home", t, self.cfg.dt_hours / self.cfg.ev_eta_dis))
            if self.has_ev_dis_grid:
                home_dyn_terms.append(("p_ev_dis_grid_home", t, self.cfg.dt_hours / self.cfg.ev_eta_dis))
            self._add_row(rows, f"c_ev_home_dyn_{t}", 0.0, 0.0, home_dyn_terms)

            external_dyn_terms: List[Tuple[str, int, float]] = [("E_ev_external", t + 1, 1.0), ("E_ev_external", t, -1.0), ("p_ev_drive_ext", t, 1.0)]
            if self.has_ev_ext_ch:
                external_dyn_terms.append(("p_ev_ext_ch", t, -self.cfg.ev_eta_ch * self.cfg.dt_hours))
            if self.has_ev_dis_house:
                external_dyn_terms.append(("p_ev_dis_house_ext", t, self.cfg.dt_hours / self.cfg.ev_eta_dis))
            if self.has_ev_dis_grid:
                external_dyn_terms.append(("p_ev_dis_grid_ext", t, self.cfg.dt_hours / self.cfg.ev_eta_dis))
            self._add_row(rows, f"c_ev_external_dyn_{t}", 0.0, 0.0, external_dyn_terms)

            if self.has_grid_export:
                self._add_row(rows, f"c_grid_dir_import_{t}", -self.kHighsInf, 0.0, [("p_grid_import", t, 1.0), ("y_grid_dir", t, -self.cfg.p_grid_max_kw)])
                self._add_row(rows, f"c_grid_dir_export_{t}", -self.kHighsInf, float(self.cfg.p_grid_max_kw), [("p_grid_export", t, 1.0), ("y_grid_dir", t, self.cfg.p_grid_max_kw)])

            if self.has_charge and self.has_discharge:
                y = "y_ev_mode"
                if self.has_ev_home_ch:
                    self._add_row(rows, f"c_ev_mode_home_ch_{t}", -self.kHighsInf, 0.0, [("p_ev_home_ch", t, 1.0), (y, t, -self.cfg.max_ev_home_charge_kw)])
                if self.has_ev_ext_ch:
                    self._add_row(rows, f"c_ev_mode_ext_ch_{t}", -self.kHighsInf, 0.0, [("p_ev_ext_ch", t, 1.0), (y, t, -self.cfg.max_ev_external_charge_kw)])
                if self.has_ev_dis_house:
                    self._add_row(rows, f"c_ev_mode_dis_house_{t}", -self.kHighsInf, float(self.cfg.max_ev_discharge_house_kw), [("p_ev_dis_house", t, 1.0), (y, t, self.cfg.max_ev_discharge_house_kw)])
                if self.has_ev_dis_grid:
                    self._add_row(rows, f"c_ev_mode_dis_grid_{t}", -self.kHighsInf, float(self.cfg.max_ev_discharge_grid_kw), [("p_ev_dis_grid", t, 1.0), (y, t, self.cfg.max_ev_discharge_grid_kw)])

        for t in range(self.H + 1):
            self._add_row(
                rows,
                f"c_ev_reserve_{t}",
                0.0,
                float(self.cfg.ev_cap_kwh),
                [("E_ev_home", t, 1.0), ("E_ev_external", t, 1.0)],
            )

        return rows

    def _build_model(self, solver_tee: bool = False) -> None:
        rows = self._build_row_specs()
        self.row_index = {row.name: int(idx) for idx, row in enumerate(rows)}
        self.num_rows = len(rows)
        self.highs = self.Highs()
        self.highs.setOptionValue("output_flag", bool(solver_tee))
        self.highs.setOptionValue("presolve", "on")
        if self.cfg.milp_time_limit_sec is not None:
            self.highs.setOptionValue("time_limit", float(self.cfg.milp_time_limit_sec))
        if self.cfg.milp_rel_gap is not None:
            self.highs.setOptionValue("mip_rel_gap", float(self.cfg.milp_rel_gap))

        for family, idxs in self.var_index.items():
            lower = np.asarray([self._col_lower[int(i)] for i in idxs], dtype=np.float64)
            upper = np.asarray([self._col_upper[int(i)] for i in idxs], dtype=np.float64)
            self.highs.addVars(len(idxs), lower, upper)
            if family in {"y_grid_dir", "y_ev_mode"}:
                for col_idx in idxs:
                    self.highs.changeColIntegrality(int(col_idx), self.HighsVarType.kInteger)

        for row in rows:
            indices: List[int] = []
            coeffs: List[float] = []
            for family, t, coef in row.terms:
                indices.append(self._col(family, t))
                coeffs.append(float(coef))
            self.highs.addRow(
                float(row.lower),
                float(row.upper),
                int(len(indices)),
                np.asarray(indices, dtype=np.int32),
                np.asarray(coeffs, dtype=np.float64),
            )

        self.highs.setOptionValue("output_flag", bool(solver_tee))

    def _set_row_bounds(self, row_name: str, lower: float, upper: float) -> None:
        idx = int(self.row_index[row_name])
        self.highs.changeRowBounds(idx, float(lower), float(upper))

    def _update_objective(self, window_arrays: Dict[str, List[float]]) -> None:
        dt = self.cfg.dt_hours
        home_import_price = window_arrays["home_import_price"]
        ev_home_import_price = window_arrays["ev_home_import_price"]
        ev_ext_import_price = window_arrays["ev_ext_import_price"]
        home_export_price = window_arrays["ev_home_export_price"]
        external_export_price = window_arrays["ev_external_export_price"]
        ev_degradation_cost = float(self.cfg.ev_degradation_eur_per_kwh_charged) * dt

        def _change(family: str, values: Sequence[float]) -> None:
            if family not in self.var_index:
                return
            for t, value in enumerate(values):
                self.highs.changeColCost(int(self.var_index[family][t]), float(value))

        _change("p_grid_import", [float(v) * dt for v in home_import_price])
        _change("p_ev_home_ch", [(float(ev_home_import_price[t]) - float(home_import_price[t])) * dt + ev_degradation_cost for t in range(self.H)])
        _change("p_ev_ext_ch", [float(v) * dt + ev_degradation_cost for v in ev_ext_import_price])
        _change("p_ev_dis_grid_home", [-float(v) * dt for v in home_export_price] if self.has_ev_dis_grid else [])
        _change("p_ev_dis_grid_ext", [-float(v) * dt for v in external_export_price] if self.has_ev_dis_grid else [])

    def _update_row_bounds(self, soc_ev_home_now: float, soc_ev_external_now: float, window_arrays: Dict[str, List[float]]) -> None:
        self._set_row_bounds("c_init_ev_home", float(soc_ev_home_now), float(soc_ev_home_now))
        self._set_row_bounds("c_init_ev_external", float(soc_ev_external_now), float(soc_ev_external_now))
        for t, load_kw in enumerate(window_arrays["load_kw"]):
            self._set_row_bounds(f"c_home_balance_{t}", float(load_kw), float(load_kw))
        for t in range(self.H + 1):
            reserve_lb = max(0.0, _effective_ev_lb(self.cfg, window_arrays["ev_reserve_kwh"][t]) - EV_SOC_CLAMP_EPS_KWH)
            self._set_row_bounds(f"c_ev_reserve_{t}", reserve_lb, float(self.cfg.ev_cap_kwh))

    def update(self, window_arrays: Dict[str, List[float]], soc_ev_home_now: float, soc_ev_external_now: float) -> None:
        """Update the persistent model for one MPC window."""
        t0 = perf_counter()
        self._update_objective(window_arrays)
        self._update_row_bounds(soc_ev_home_now, soc_ev_external_now, window_arrays)
        self.last_update_time_sec = perf_counter() - t0

    def apply_mip_start(self, prev_solution: Optional[Dict[str, List[float]]], shift: int = 1) -> None:
        """Apply a warm start from the previous horizon solution if available."""
        if not prev_solution:
            return

        try:
            start_values = np.zeros(self.num_cols, dtype=np.float64)
            for family, idxs in self.var_index.items():
                seq = prev_solution.get(family)
                if not seq:
                    continue
                shifted = _shift_sequence(seq, shift)
                for local_t, col_idx in enumerate(idxs):
                    if local_t < len(shifted):
                        start_values[int(col_idx)] = float(shifted[local_t])
            sol = self.HighsSolution()
            sol.col_value = start_values.tolist()
            if hasattr(sol, "value_valid"):
                sol.value_valid = True
            self.highs.setSolution(sol)
        except Exception:
            # Warm starts are opportunistic only.
            return

    def optimize(self) -> str:
        t0 = perf_counter()
        self.highs.run()
        self.last_optimize_time_sec = perf_counter() - t0
        status = self.highs.getModelStatus()
        status_text = str(self.highs.modelStatusToString(status)).lower()
        if self.highs.getNumCol() == 0:
            return status_text
        sol = self.highs.getSolution()
        col_value = getattr(sol, "col_value", None)
        if col_value is None:
            return status_text
        if np.asarray(col_value, dtype=float).size == 0:
            return status_text
        if "optimal" in status_text:
            return "optimal"
        if "feasible" in status_text or "time limit" in status_text or "limit" in status_text:
            return "feasible"
        return status_text

    def extract_solution_full_horizon(self) -> Dict[str, List[float]]:
        t0 = perf_counter()
        sol = self.highs.getSolution()
        values = getattr(sol, "col_value", None)
        if values is None:
            self.last_extract_time_sec = perf_counter() - t0
            return {}
        x = np.asarray(values, dtype=float)
        out: Dict[str, List[float]] = {}
        for family, idxs in self.var_index.items():
            out[family] = [float(x[int(i)]) for i in idxs]
        self.last_extract_time_sec = perf_counter() - t0
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


# Preserve the older class name used by the rest of the repo.
HiGHSMPCSolver = HighsPersistentMPCSolver


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

    prev_solution: Optional[Dict[str, List[float]]] = None
    prev_solution_shift = 1
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
            solver.update(arrays, soc_ev_home_now=e_ev_home, soc_ev_external_now=e_ev_external)
            if prev_solution is not None:
                solver.apply_mip_start(prev_solution, shift=prev_solution_shift)
            update_sec = perf_counter() - t_update

            status = solver.optimize()
            full_solution = solver.extract_solution_full_horizon()
            if full_solution:
                block_actions = [solver.extract_action(full_solution, t) for t in range(apply_steps)]
                extract_sec = solver.last_extract_time_sec
                prev_solution = full_solution
                prev_solution_shift = apply_steps
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
                    "build_once_time_sec": round(solver.build_once_time_sec, 6) if local_t == 0 else 0.0,
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
    return out, logs
