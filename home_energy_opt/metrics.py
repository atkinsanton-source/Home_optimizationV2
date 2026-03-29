from __future__ import annotations

import pandas as pd

from home_energy_opt.config import EnergySystemConfig

EV_SOC_CLAMP_EPS_KWH = 1e-5


def summarize_metrics(inp: pd.DataFrame, sim: pd.DataFrame, cfg: EnergySystemConfig) -> dict:
    """Compute aggregate cost, energy flow, and SOC-violation KPIs for one simulation."""
    def _sum_col(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns:
            return 0.0
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())

    def _max_col(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns:
            return 0.0
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).max())

    def _sum_abs_col(df: pd.DataFrame, col: str, eps: float = 0.0) -> float:
        if col not in df.columns:
            return 0.0
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).abs()
        if eps > 0.0:
            vals = vals.where(vals > eps, 0.0)
        return float(vals.sum())

    pv_energy = (inp["pv_ac_kw"] * cfg.dt_hours).sum()
    pv_export = sim["grid_export_kwh"].sum()
    pv_self_consumed = max(0.0, pv_energy - pv_export)

    return {
        "total_cost_eur": float(sim["step_cost_eur"].sum()),
        "total_system_cost_eur": float(sim["step_cost_eur"].sum()),
        "grid_import_kwh": float(sim["grid_import_kwh"].sum()),
        "grid_export_kwh": float(sim["grid_export_kwh"].sum()),
        "external_charge_kwh": _sum_col(sim, "ev_ext_ch_kwh"),
        "external_charge_cost_eur": _sum_col(sim, "ext_charge_cost_eur"),
        "ev_battery_degradation_cost_eur": _sum_col(sim, "ev_battery_degradation_cost_eur"),
        "home_ev_charge_kwh": _sum_col(sim, "ev_home_ch_kwh"),
        "ev_discharge_to_home_kwh": _sum_col(sim, "ev_dis_to_home_kwh"),
        "ev_discharge_to_grid_kwh": _sum_col(sim, "ev_dis_to_grid_kwh"),
        "reserve_clamp_events": int(_sum_col(inp, "ev_reserve_clamped")),
        "solver_fallback_steps": int(_sum_col(sim, "used_fallback")),
        "ev_soc_clamp_steps": int(_sum_col(sim, "ev_soc_clamped")),
        "ev_soc_clamp_abs_kwh": _sum_abs_col(sim, "ev_soc_clamp_delta_kwh", eps=EV_SOC_CLAMP_EPS_KWH),
        "ev_soc_clamp_after_fallback_steps": int(_sum_col(sim, "ev_soc_clamped_after_fallback")),
        "ev_soc_clamp_after_fallback_abs_kwh": _sum_abs_col(
            sim,
            "ev_soc_clamp_after_fallback_delta_kwh",
            eps=EV_SOC_CLAMP_EPS_KWH,
        ),
        "pv_self_consumption_ratio": float(pv_self_consumed / pv_energy) if pv_energy > 0 else 0.0,
        "max_grid_import_kw": _max_col(sim, "grid_import_kw"),
        "max_grid_export_kw": _max_col(sim, "grid_export_kw"),
        "bat_soc_violations": int(
            ((sim["bat_energy_kwh"] < cfg.bat_soc_min * cfg.bat_cap_kwh) | (sim["bat_energy_kwh"] > cfg.bat_soc_max * cfg.bat_cap_kwh)).sum()
        ),
        "ev_soc_violations": int(((sim["ev_energy_kwh"] < cfg.ev_soc_min * cfg.ev_cap_kwh) | (sim["ev_energy_kwh"] > cfg.ev_cap_kwh)).sum()),
    }
