from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from home_energy_opt.config import EnergySystemConfig


def _clip_charge_power(max_power_kw: float, eta_ch: float, e_now: float, e_max: float, dt_hours: float) -> float:
    """Return feasible charging power under converter and SOC limits."""
    if e_now >= e_max:
        return 0.0
    return max(0.0, min(max_power_kw, (e_max - e_now) / (eta_ch * dt_hours)))


def _clip_discharge_power(max_power_kw: float, eta_dis: float, e_now: float, e_min: float, dt_hours: float) -> float:
    """Return feasible discharging power under converter and SOC limits."""
    if e_now <= e_min:
        return 0.0
    return max(0.0, min(max_power_kw, (e_now - e_min) * eta_dis / dt_hours))


class BaselineHeuristic(ABC):
    """Common fixed-rule baseline simulation with pluggable home import price source."""

    @abstractmethod
    def _home_import_price_series(self, df: pd.DataFrame, cfg: EnergySystemConfig) -> pd.Series:
        """Return the EUR/kWh series used for home grid imports in cost accounting."""

    def simulate(self, df: pd.DataFrame, cfg: EnergySystemConfig) -> pd.DataFrame:
        """Simulate a deterministic heuristic controller over all timesteps."""
        e_bat = cfg.bat_soc_init * cfg.bat_cap_kwh
        e_ev = cfg.ev_soc_init * cfg.ev_cap_kwh
        e_bat_min = cfg.bat_soc_min * cfg.bat_cap_kwh
        e_bat_max = cfg.bat_soc_max * cfg.bat_cap_kwh

        rows = []
        # Crucial idea: this is a fixed-rule controller (no optimization), so behavior is simple and predictable.
        for _, r in df.iterrows():
            e_ev_start = e_ev
            load_kw = float(r["load_kw"])
            drive_kwh = float(r["ev_drive_kwh"])
            ev_reserve_kwh = float(r["ev_reserve_kwh"])
            ev_home_ch_max_kw = float(r.get("ev_p_home_ch_max_kw", 0.0))
            ev_ext_ch_max_kw = float(r.get("ev_p_ext_ch_max_kw", 0.0))
            ev_ext_price = float(r.get("ev_ext_import_price_eur_per_kwh", 0.0))

            e_ev_min = max(cfg.ev_soc_min * cfg.ev_cap_kwh, ev_reserve_kwh)
            e_ev_max = cfg.ev_cap_kwh

            # Priority rule: always charge EV up to currently available limits until EV is full.
            p_ev_ch_limit = _clip_charge_power(
                max_power_kw=ev_home_ch_max_kw + ev_ext_ch_max_kw,
                eta_ch=cfg.ev_eta_ch,
                e_now=e_ev,
                e_max=e_ev_max,
                dt_hours=cfg.dt_hours,
            )
            p_ev_home_ch = min(ev_home_ch_max_kw, p_ev_ch_limit)
            p_ev_ext_ch = min(ev_ext_ch_max_kw, max(0.0, p_ev_ch_limit - p_ev_home_ch))

            # Home-side power balance:
            # 1) grid first covers house load
            # 2) any remaining grid headroom can feed home EV charging
            p_grid_import = min(load_kw, cfg.p_grid_max_kw)
            home_grid_headroom = max(0.0, cfg.p_grid_max_kw - p_grid_import)
            if p_ev_home_ch > home_grid_headroom:
                p_ev_home_ch = home_grid_headroom
            p_grid_import += p_ev_home_ch
            p_grid_export = 0.0

            p_bat_ch = 0.0
            p_bat_dis = 0.0
            p_ev_dis_to_home = 0.0
            p_ev_dis_to_grid = 0.0
            p_ev_ch = p_ev_home_ch + p_ev_ext_ch

            # Crucial state transition: this is what carries today's decision into the next timestep.
            e_ev = e_ev + (cfg.ev_eta_ch * p_ev_ch) * cfg.dt_hours - drive_kwh

            violations = []
            # Flag out-of-bound EV energy before we clip, so diagnostics remain transparent.
            if e_ev < e_ev_min - 1e-6 or e_ev > e_ev_max + 1e-6:
                violations.append("ev_soc")

            # Keep battery state inside bounds.
            e_bat = min(max(e_bat, e_bat_min), e_bat_max)
            ext_charge_cost_eur = p_ev_ext_ch * cfg.dt_hours * ev_ext_price

            rows.append(
                {
                    "grid_import_kw": p_grid_import,
                    "grid_export_kw": p_grid_export,
                    "bat_ch_kw": p_bat_ch,
                    "bat_dis_kw": p_bat_dis,
                    "ev_ch_kw": p_ev_ch,
                    "ev_dis_kw": 0.0,
                    "ev_home_ch_kw": p_ev_home_ch,
                    "ev_ext_ch_kw": p_ev_ext_ch,
                    "ev_dis_to_home_kw": p_ev_dis_to_home,
                    "ev_dis_to_grid_kw": p_ev_dis_to_grid,
                    "ext_charge_cost_eur": ext_charge_cost_eur,
                    "ev_state": r.get("ev_state", ""),
                    "charging_point_effective": r.get("charging_point_effective", ""),
                    "bat_energy_kwh": e_bat,
                    "ev_energy_kwh": e_ev_start,
                    "violations": ";".join(violations),
                }
            )

        result = pd.DataFrame(rows, index=df.index)
        # Add per-step energies and monetary cost for metric aggregation.
        result["grid_import_kwh"] = result["grid_import_kw"] * cfg.dt_hours
        result["grid_export_kwh"] = result["grid_export_kw"] * cfg.dt_hours
        result["ev_home_ch_kwh"] = result["ev_home_ch_kw"] * cfg.dt_hours
        result["ev_ext_ch_kwh"] = result["ev_ext_ch_kw"] * cfg.dt_hours
        result["ev_dis_to_home_kwh"] = result["ev_dis_to_home_kw"] * cfg.dt_hours
        result["ev_dis_to_grid_kwh"] = result["ev_dis_to_grid_kw"] * cfg.dt_hours
        result["ev_battery_degradation_cost_eur"] = (
            float(cfg.ev_degradation_eur_per_kwh_charged) * (result["ev_home_ch_kwh"] + result["ev_ext_ch_kwh"])
        )
        home_import_price = self._home_import_price_series(df, cfg)
        result["home_grid_price_total_eur_per_kwh"] = home_import_price
        result["step_cost_eur"] = (
            home_import_price * result["grid_import_kwh"]
            - df["ev_export_price_eur_per_kwh"] * result["grid_export_kwh"]
            + result["ext_charge_cost_eur"]
            + result["ev_battery_degradation_cost_eur"]
        )
        return result


class BaselineDynamic(BaselineHeuristic):
    """Baseline heuristic with dynamic (preprocessed) home import price."""

    def _home_import_price_series(self, df: pd.DataFrame, cfg: EnergySystemConfig) -> pd.Series:
        return pd.to_numeric(df["import_price_eur_per_kwh"], errors="coerce").fillna(0.0).astype(float)


class BaselineStatic(BaselineHeuristic):
    """Baseline heuristic with constant home import price from config."""

    def _home_import_price_series(self, df: pd.DataFrame, cfg: EnergySystemConfig) -> pd.Series:
        return pd.Series(float(cfg.baseline_static_import_price_eur_per_kwh), index=df.index, dtype="float64")


def simulate_baseline(df: pd.DataFrame, cfg: EnergySystemConfig) -> pd.DataFrame:
    """Backward-compatible alias: run the dynamic baseline variant."""
    return BaselineDynamic().simulate(df, cfg)
