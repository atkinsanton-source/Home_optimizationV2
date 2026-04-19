from dataclasses import dataclass, field
from typing import Dict, Optional

SUPPORTED_CHARGING_POINT_NAMES = ("home", "public", "workplace", "fast75", "fast150", "none")


def _default_ev_cp_charge_power_kw() -> Dict[str, float]:
    return {
        "home": 11.0,
        "public": 22.0,
        "workplace": 11.0,
        "fast75": 75.0,
        "fast150": 150.0,
        "none": 0.0,
    }


def _default_ev_cp_discharge_power_kw() -> Dict[str, float]:
    return {
        "home": 11.0,
        "public": 0.0,
        "workplace": 0.0,
        "fast75": 0.0,
        "fast150": 0.0,
        "none": 0.0,
    }


def _default_ev_cp_import_price_fixed_eur_per_kwh() -> Dict[str, float]:
    """Default fixed import prices by charging point (fresh dict per config instance)."""
    return {
        "home": 0.0,
    "public": 0.45,
    "workplace": 0.30,
    "fast75": 0.59,
    "fast150": 0.69,
    "none": 0.0,
    }


def _default_zero_by_charging_point() -> Dict[str, float]:
    """Return per-charging-point zero values (fresh dict per config instance)."""
    return {charging_point_name: 0.0 for charging_point_name in SUPPORTED_CHARGING_POINT_NAMES}


@dataclass
class EnergySystemConfig:
    """Central configuration for device limits, efficiencies, and solver options."""
    # Version preset control
    system_version: int = 1
    apply_version_preset: bool = True

    # Feature toggles
    enable_pv: bool = False
    enable_battery: bool = False
    enable_grid_export: bool = False
    enable_ev_home_charge: bool = True
    enable_ev_external_charge: bool = True
    enable_ev_discharge_to_house: bool = False
    enable_ev_discharge_to_grid: bool = False

    # Timestep
    dt_hours: float = 0.25

    # Grid
    p_grid_max_kw: float = 44.0
    import_price_adder_eur_per_kwh: float = 0.1944 # Von Octopus Energy
    import_price_adder_pct: float = 0.19  # Mehrwertsteuer 19%
    import_price_deduction_eur_per_kwh: float = 0.0
    baseline_static_import_price_eur_per_kwh: float = 0.3627
    static_mpc_import_price_eur_per_kwh: float = 0.3627

    # Export remuneration
    export_price_source: str = "dynamic"  # "fixed" or "dynamic"
    export_price_eur_per_kwh: float = 0.0778
    export_price_adder_eur_per_kwh: float = 0.0997 +0.0244 #Stromnetz Berlin dokument für 2025 Netzentgelt + Stromsteuer
    export_price_deduction_eur_per_kwh: float = 0.0

    # EV
    ev_cap_kwh: float = 79.5
    ev_soc_init: float = 1.0
    ev_soc_min: float = 0.3
    ev_eta_ch: float = 0.92
    ev_eta_dis: float = 0.92

    # Charging-point specific limits and prices
    ev_cp_charge_power_kw: Dict[str, float] = field(default_factory=_default_ev_cp_charge_power_kw)
    ev_cp_discharge_power_kw: Dict[str, float] = field(default_factory=_default_ev_cp_discharge_power_kw)
    ev_cp_import_price_fixed_eur_per_kwh: Dict[str, float] = field(default_factory=_default_ev_cp_import_price_fixed_eur_per_kwh)
    ev_cp_import_price_adder_eur_per_kwh: Dict[str, float] = field(default_factory=_default_zero_by_charging_point)
    ev_cp_import_price_deduction_eur_per_kwh: Dict[str, float] = field(default_factory=_default_zero_by_charging_point)
    ev_cp_export_price_adder_eur_per_kwh: Dict[str, float] = field(default_factory=_default_zero_by_charging_point)
    ev_cp_export_price_deduction_eur_per_kwh: Dict[str, float] = field(default_factory=_default_zero_by_charging_point)

    # Stationary battery
    bat_cap_kwh: float = 10.0
    bat_soc_init: float = 0.50
    bat_soc_min: float = 0.10
    bat_soc_max: float = 1.0
    bat_eta_ch: float = 0.95
    bat_eta_dis: float = 0.95
    bat_p_ch_max_kw: float = 6.0
    bat_p_dis_max_kw: float = 6.0

    # MPC
    horizon_steps: int = 96
    gurobi_mipgap: float = 1e-4
    gurobi_threads: Optional[int] = None
    gurobi_mipfocus: Optional[int] = None
    use_persistent_gurobi: bool = True
    use_mip_start: bool = True
    mpc_apply_steps: int = 1
    pad_tail_neutral_prices: bool = True                #maybe remove
    pad_tail_price_eur_per_kwh: float = 0.0             #maybe remove

    # Extension hooks
    enable_battery_degradation_cost: bool = False
    battery_degradation_eur_per_kwh: float = 0.0
    ev_degradation_eur_per_kwh_charged: float = 0.067

    def __post_init__(self) -> None:
        if self.apply_version_preset:
            self.apply_system_version_preset(self.system_version)
        if self.enable_ev_discharge_to_grid:
            self.enable_grid_export = True

    def apply_system_version_preset(self, version: int) -> None:
        """Apply feature/tariff defaults for one supported system version."""
        if version == 1:
            self.enable_pv = False
            self.enable_battery = False
            self.enable_grid_export = False
            self.enable_ev_home_charge = True
            self.enable_ev_external_charge = True
            self.enable_ev_discharge_to_house = False
            self.enable_ev_discharge_to_grid = False
            self.export_price_source = "fixed"
        elif version == 2:
            self.enable_pv = False
            self.enable_battery = False
            self.enable_grid_export = False
            self.enable_ev_home_charge = True
            self.enable_ev_external_charge = True
            self.enable_ev_discharge_to_house = True
            self.enable_ev_discharge_to_grid = False
            self.export_price_source = "fixed"
        elif version == 3:
            self.enable_pv = False
            self.enable_battery = False
            self.enable_grid_export = True
            self.enable_ev_home_charge = True
            self.enable_ev_external_charge = True
            self.enable_ev_discharge_to_house = True
            self.enable_ev_discharge_to_grid = True
            self.export_price_source = "dynamic"
        elif version == 4:
            # Prepared only: full physical implementation remains future work.
            self.enable_pv = True
            self.enable_battery = True
            self.enable_grid_export = True
            self.enable_ev_home_charge = True
            self.enable_ev_external_charge = True
            self.enable_ev_discharge_to_house = True
            self.enable_ev_discharge_to_grid = True
        else:
            raise ValueError(f"Unsupported system_version={version}. Expected one of [1, 2, 3, 4].")

    @property
    def supported_charging_points(self) -> set[str]:
        return set(self.ev_cp_charge_power_kw.keys())

    @property
    def max_ev_home_charge_kw(self) -> float:
        return float(max(0.0, self.ev_cp_charge_power_kw.get("home", 0.0)))

    @property
    def max_ev_external_charge_kw(self) -> float:
        return float(max(v for k, v in self.ev_cp_charge_power_kw.items() if k != "home"))

    @property
    def max_ev_discharge_house_kw(self) -> float:
        return float(max(0.0, self.ev_cp_discharge_power_kw.get("home", 0.0)))

    @property
    def max_ev_discharge_grid_kw(self) -> float:
        return float(max(0.0, self.ev_cp_discharge_power_kw.get("home", 0.0)))
