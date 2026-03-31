# Home Energy Optimization

Modular Python project for year-long 15-minute home energy simulation with three scenarios:

1. **Baseline rule-based control (dynamic home import price)**
2. **Baseline rule-based control (static home import price)**
3. **MPC optimization (native gurobipy rolling-horizon model)**

## Structure

- `home_energy_opt/config.py`: central dataclass for all parameters
- `home_energy_opt/data.py`: CSV loading, timezone-aware indexing, preprocessing and derived columns
- `home_energy_opt/baseline.py`: step-by-step non-optimizing baseline simulation
- `home_energy_opt/mpc.py`: receding-horizon MPC loop and MILP window model
- `home_energy_opt/metrics.py`: KPI calculation
- `home_energy_opt/plots.py`: interactive Plotly dashboard export
- `home_energy_opt/main.py`: CLI entry point

## Input CSV columns

Required columns:

- `local_time` (timezone-aware)
- `P_PV_AC`
- `P_load`
- `day_ahead_price` (ct/kWh)
- `EV_state`
- `EV_consumption_kWh`
- `EV_required_kWh_24h`
- `charging_point`

Other columns are allowed and ignored in v1.

## Run

```bash
python -m home_energy_opt.main data/your_timeseries.csv --steps 672 --horizon 96 --outdir outputs
```

After the run, an interactive HTML dashboard is written for the system-connection comparison,
so you can zoom/pan and toggle traces later.

To rebuild only the HTML dashboard from already existing `baseline_results.csv` and `mpc_results.csv`
without re-running MPC:

```bash
python scripts/plot_dashboard_from_outputs.py data/your_timeseries.csv --system-version 1 --outdir outputs
```

Outputs:

- `baseline_results.csv`
- `baseline_dynamic_results.csv`
- `baseline_static_results.csv`
- `mpc_results.csv`
- `mpc_solver_logs.csv`
- `metrics_comparison.csv`
- `system_connection_comparison_interactive.html` (interactive zoom/pan with dynamic axes after reopening; includes Overview + Flow Balance sections with `sources < 0`, `sinks > 0`)
- `plotly.min.js` (local Plotly runtime copied next to the HTML for offline interaction)

## Baseline Static Price Knob

Set `baseline_static_import_price_eur_per_kwh` in `home_energy_opt/config.py` to control the static home import tariff used by the static baseline.

## Extension hooks already prepared

- EV power limits currently derive from `EV_state == "home"` with fixed home limits; can be replaced by time-series columns (`P_ev_ch_max_kw[t]`, `P_ev_dis_max_kw[t]`).
- Export remuneration uses constant placeholder (`0.0778 EUR/kWh`), ready to replace with regulation-aware function.
- Objective already includes a hook for battery degradation.
- Thermal coupling (DHW/SH) can be integrated as additional buses/storages/constraints in MPC.
