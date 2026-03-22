from __future__ import annotations

import numpy as np
import pandas as pd

from home_energy_opt.config import EnergySystemConfig

REQUIRED_COLUMNS = [
    "local_time",
    "P_PV_AC",
    "P_load",
    "day_ahead_price",
    "EV_state",
    "EV_consumption_kWh",
    "EV_required_kWh_24h",
    "charging_point",
]


def load_csv(path: str) -> pd.DataFrame:
    """Load the required raw columns and enforce a clean UTC timestamp index."""
    # Fast header scan first so we can fail early with a clear error message.
    header_cols = pd.read_csv(path, nrows=0).columns.tolist()
    missing_required = [c for c in REQUIRED_COLUMNS if c not in header_cols]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    usecols = REQUIRED_COLUMNS
    dtype_map = {
        "P_PV_AC": "float64",
        "P_load": "float64",
        "day_ahead_price": "float64",
        "EV_state": "string",
        "EV_consumption_kWh": "float64",
        "EV_required_kWh_24h": "float64",
        "charging_point": "string",
    }
    dtypes = {k: v for k, v in dtype_map.items() if k in usecols}

    # Read only required columns with concrete dtypes to reduce parse time and memory.
    df = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtypes,
    )

    # Crucial: require timezone-aware timestamps to avoid silent time shifts.
    # Example accepted format: 2024-01-01T00:00:00+01:00
    local_time_raw = df["local_time"].astype(str).str.strip()
    tz_pattern = r"(?:Z|[+-]\d{2}:?\d{2})$"
    missing_tz_mask = ~local_time_raw.str.contains(tz_pattern, regex=True, na=False)
    if missing_tz_mask.any():
        sample_values = local_time_raw[missing_tz_mask].head(3).tolist()
        raise ValueError(
            "local_time must be timezone-aware (for example, 2024-01-01T00:00:00+01:00). "
            f"Found timezone-naive values: {sample_values}"
        )

    try:
        df["local_time"] = pd.to_datetime(local_time_raw, utc=True, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Failed to parse local_time as timezone-aware datetimes. "
            "Mixed offsets are supported and normalized to UTC internally."
        ) from exc

    if getattr(df["local_time"].dt, "tz", None) is None:
        raise ValueError("local_time parsing failed to produce timezone-aware timestamps in UTC.")

    # Crucial for MPC: every row must represent exactly one unique time step.
    df = df.set_index("local_time").sort_index()
    duplicated = df.index.duplicated(keep=False)
    if duplicated.any():
        duplicate_values = df.index[duplicated][:5].tolist()
        raise ValueError(
            "Duplicate timestamps detected in local_time after UTC normalization. "
            f"Duplicate count: {int(duplicated.sum())}. Sample: {duplicate_values}"
        )
    if not df.index.is_monotonic_increasing:
        sample_values = df.index[:5].tolist()
        raise ValueError(
            "local_time index is not monotonic increasing after sorting. "
            f"First timestamps: {sample_values}"
        )
    return df


def preprocess(df: pd.DataFrame, cfg: EnergySystemConfig) -> pd.DataFrame:
    """Map raw dataset columns into model inputs and clip values to valid ranges."""
    out = df.copy()
    # Normalize raw EV state labels ("HOME", " home ", etc.) into one canonical form.
    ev_state = out["EV_state"].astype("string").str.strip().str.lower()
    out["ev_state"] = ev_state

    charging_point_effective = out["charging_point"].astype("string").str.strip().str.lower()
    invalid_mask = ~charging_point_effective.isin(cfg.supported_charging_points)
    if invalid_mask.any():
        invalid_values = charging_point_effective[invalid_mask].dropna().unique().tolist()
        allowed_values = sorted(cfg.supported_charging_points)
        raise ValueError(
            "charging_point contains unsupported values. "
            f"Unsupported: {invalid_values[:5]}. Allowed: {allowed_values}"
        )
    out["charging_point_effective"] = charging_point_effective

    # Convert raw input columns into optimization-ready units and flags.
    out["pv_ac_kw"] = out["P_PV_AC"].clip(lower=0.0) if cfg.enable_pv else 0.0
    out["load_kw"] = out["P_load"].clip(lower=0.0)
    day_ahead_eur_per_kwh = out["day_ahead_price"] / 1000.0
    out["import_price_eur_per_kwh"] = (
        (day_ahead_eur_per_kwh + cfg.import_price_adder_eur_per_kwh) * ((1.0 + cfg.import_price_adder_pct))
    )
    out["ev_connected_home"] = (charging_point_effective == "home").astype(int)
    out["ev_drive_kwh"] = out["EV_consumption_kWh"].clip(lower=0.0)

    # Crucial: reserve demand must stay inside physical battery bounds, otherwise MPC can become infeasible.
    reserve_raw = out["EV_required_kWh_24h"] * 0.0
    reserve_min = cfg.ev_soc_min * cfg.ev_cap_kwh
    out["ev_reserve_clamped_low"] = (reserve_raw < reserve_min).astype(int)
    out["ev_reserve_clamped_high"] = (reserve_raw > cfg.ev_cap_kwh).astype(int)
    out["ev_reserve_clamped"] = ((out["ev_reserve_clamped_low"] + out["ev_reserve_clamped_high"]) > 0).astype(int)
    out["ev_reserve_kwh"] = reserve_raw.clip(lower=reserve_min, upper=cfg.ev_cap_kwh)
    clamped_high = int(out["ev_reserve_clamped_high"].sum())
    if clamped_high > 0:
        print(
            f"[preprocess] Clamped EV_required_kWh_24h to EV capacity in {clamped_high} rows.",
            flush=True,
        )

    # Convert location and feature flags into direct power bounds used by constraints.
    cp_charge_max_kw = charging_point_effective.map(cfg.ev_cp_charge_power_kw).fillna(0.0).astype(float)
    cp_discharge_max_kw = charging_point_effective.map(cfg.ev_cp_discharge_power_kw).fillna(0.0).astype(float)
    is_home = charging_point_effective == "home"
    is_external = (~is_home) & (charging_point_effective != "none")

    out["ev_can_home_charge"] = (cfg.enable_ev_home_charge & is_home & (cp_charge_max_kw > 0.0)).astype(int)
    out["ev_can_external_charge"] = (cfg.enable_ev_external_charge & is_external & (cp_charge_max_kw > 0.0)).astype(int)
    out["ev_can_discharge_to_house"] = (
        cfg.enable_ev_discharge_to_house & is_home & (cp_discharge_max_kw > 0.0)
    ).astype(int)
    out["ev_can_discharge_to_grid"] = (
        cfg.enable_grid_export & cfg.enable_ev_discharge_to_grid & is_home & (cp_discharge_max_kw > 0.0)
    ).astype(int)

    out["ev_p_home_ch_max_kw"] = cp_charge_max_kw.where(out["ev_can_home_charge"] > 0, 0.0)
    out["ev_p_ext_ch_max_kw"] = cp_charge_max_kw.where(out["ev_can_external_charge"] > 0, 0.0)
    out["ev_p_ch_total_max_kw"] = cp_charge_max_kw.where(
        (out["ev_can_home_charge"] + out["ev_can_external_charge"]) > 0,
        0.0,
    )
    out["ev_p_dis_house_max_kw"] = cp_discharge_max_kw.where(out["ev_can_discharge_to_house"] > 0, 0.0)
    out["ev_p_dis_grid_max_kw"] = cp_discharge_max_kw.where(out["ev_can_discharge_to_grid"] > 0, 0.0)
    out["ev_p_dis_total_max_kw"] = cp_discharge_max_kw.where(
        (out["ev_can_discharge_to_house"] + out["ev_can_discharge_to_grid"]) > 0,
        0.0,
    )

    # Build effective import prices for external charging points.
    cp_price_fixed = charging_point_effective.map(cfg.ev_cp_import_price_fixed_eur_per_kwh).fillna(0.0).astype(float)
    cp_price_add = charging_point_effective.map(cfg.ev_cp_import_price_adder_eur_per_kwh).fillna(0.0).astype(float)
    cp_price_sub = charging_point_effective.map(cfg.ev_cp_import_price_deduction_eur_per_kwh).fillna(0.0).astype(float)
    is_home_charging_point = charging_point_effective == "home"
    cp_import_price = np.where(
        is_home_charging_point.to_numpy(),
        day_ahead_eur_per_kwh.to_numpy(),
        cp_price_fixed.to_numpy(),
    )
    cp_import_price = pd.Series(cp_import_price, index=out.index, dtype="float64") + cp_price_add - cp_price_sub

    out["ev_home_import_price_eur_per_kwh"] = out["import_price_eur_per_kwh"]
    out["ev_ext_import_price_eur_per_kwh"] = cp_import_price.where(out["ev_can_external_charge"] > 0, 0.0)

    # Build export tariff series (dynamic or fixed) and apply location-specific modifiers.
    if cfg.export_price_source == "dynamic":
        export_base = day_ahead_eur_per_kwh
    else:
        export_base = pd.Series(cfg.export_price_eur_per_kwh, index=out.index, dtype="float64")
    cp_export_add = charging_point_effective.map(cfg.ev_cp_export_price_adder_eur_per_kwh).fillna(0.0).astype(float)
    cp_export_sub = charging_point_effective.map(cfg.ev_cp_export_price_deduction_eur_per_kwh).fillna(0.0).astype(float)
    out["ev_export_price_eur_per_kwh"] = (
        export_base + cfg.export_price_adder_eur_per_kwh - cfg.export_price_deduction_eur_per_kwh + cp_export_add - cp_export_sub
    )
    if not cfg.enable_grid_export:
        out["ev_export_price_eur_per_kwh"] = 0.0

    # Compatibility hooks reused in existing result processing.
    out["ev_p_ch_max_kw"] = out["ev_p_home_ch_max_kw"] + out["ev_p_ext_ch_max_kw"]
    out["ev_p_dis_max_kw"] = out["ev_p_dis_house_max_kw"] + out["ev_p_dis_grid_max_kw"]

    return out
