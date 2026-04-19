from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import List


def parse_args() -> argparse.Namespace:
    """Parse CLI options for data path, MPC settings, and output controls."""
    p = argparse.ArgumentParser(description="Home energy optimization baseline vs MPC")
    p.add_argument("csv_path", type=str, help="Input CSV path")
    p.add_argument(
        "--system-version",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="System version preset: 1, 2, 3 implemented; 4 prepared in config only",
    )
    p.add_argument("--steps", type=int, default=7 * 24 * 4, help="Number of 15-min steps to simulate")
    p.add_argument("--horizon", type=int, default=96, help="MPC horizon steps")
    p.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    p.add_argument("--progress-every", type=int, default=50, help="Print MPC progress every N solved steps")
    p.add_argument(
        "--slow-step-sec",
        type=float,
        default=2.0,
        help="Always print MPC progress for a step if its solve time exceeds this threshold in seconds",
    )
    p.add_argument("--solver-tee", action="store_true", help="Print full Gurobi output for each MPC window solve")
    p.add_argument("--mipgap", type=float, default=1e-4, help="Gurobi MIP gap")
    p.add_argument("--threads", type=int, default=None, help="Gurobi threads")
    p.add_argument("--mipfocus", type=int, choices=[0, 1, 2, 3], default=None, help="Gurobi MIPFocus")
    p.add_argument(
        "--legacy-gurobi-rebuild",
        action="store_true",
        help="Use old gurobi mode that rebuilds the model every MPC step",
    )
    p.add_argument(
        "--use-mip-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable rolling MIP start for persistent gurobi mode",
    )
    return p.parse_args()


def _stage(msg: str) -> None:
    """Print a standardized stage marker to keep long runs readable."""
    print(f"[stage] {msg}", flush=True)


def _overlap_mask(a: pd.Series, b: pd.Series, eps: float = 1e-6) -> pd.Series:
    import pandas as pd

    a_num = pd.to_numeric(a, errors="coerce").fillna(0.0)
    b_num = pd.to_numeric(b, errors="coerce").fillna(0.0)
    return (a_num > eps) & (b_num > eps)


def _mask_windows(mask: pd.Series) -> List[tuple[pd.Timestamp, pd.Timestamp, int]]:
    import pandas as pd

    if not bool(mask.any()):
        return []
    ts = pd.DatetimeIndex(mask.index)
    pos = [i for i, v in enumerate(mask.to_numpy()) if bool(v)]
    out: List[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    start = pos[0]
    prev = pos[0]
    for p in pos[1:]:
        if p != prev + 1:
            out.append((ts[start], ts[prev], prev - start + 1))
            start = p
        prev = p
    out.append((ts[start], ts[prev], prev - start + 1))
    return out


def _adaptive_reserve_sanity_check(
    baseline_soc_kwh,
    mpc_result,
    fixed_reserve_kwh: float,
    label: str,
):
    import pandas as pd

    sanity_eps = 1e-6
    mpc_soc_kwh = pd.to_numeric(mpc_result["ev_energy_kwh"], errors="coerce").reindex(baseline_soc_kwh.index).fillna(0.0)
    sanity_mask = baseline_soc_kwh < fixed_reserve_kwh
    sanity_deficit_kwh = baseline_soc_kwh - mpc_soc_kwh
    sanity_violation_mask = sanity_mask & (sanity_deficit_kwh > sanity_eps)
    sanity_violation_steps = int(sanity_violation_mask.sum())
    sanity_max_deficit_kwh = float(sanity_deficit_kwh.loc[sanity_violation_mask].max()) if sanity_violation_steps > 0 else 0.0
    print(
        f"[sanity] {label} adaptive-reserve check: "
        f"masked_steps={int(sanity_mask.sum())}, "
        f"violations={sanity_violation_steps}, "
        f"max_deficit_kwh={sanity_max_deficit_kwh:.6f}",
        flush=True,
    )
    sanity_violations_df = pd.DataFrame(
        {
            "baseline_soc_kwh": baseline_soc_kwh,
            "mpc_soc_kwh": mpc_soc_kwh,
            "fixed_reserve_kwh": fixed_reserve_kwh,
            "deficit_kwh": sanity_deficit_kwh,
        },
        index=baseline_soc_kwh.index,
    ).loc[sanity_violation_mask]
    sanity_violations_df.index.name = "timestamp"
    return sanity_violations_df, sanity_violation_steps, sanity_max_deficit_kwh


def main() -> None:
    """Run the full workflow: load data, simulate, compare, and export artifacts."""
    t0 = perf_counter()
    args = parse_args()
    _stage("Importing core modules")
    t_stage = perf_counter()
    import pandas as pd

    from home_energy_opt.baseline import BaselineDynamic, BaselineStatic
    from home_energy_opt.config import EnergySystemConfig
    from home_energy_opt.data import load_csv, preprocess
    from home_energy_opt.metrics import summarize_metrics
    print(f"[stage] Importing core modules done in {perf_counter() - t_stage:.2f}s", flush=True)

    # Build one shared configuration object used by baseline and MPC runs.
    cfg = EnergySystemConfig(
        system_version=args.system_version,
        horizon_steps=args.horizon,
        gurobi_mipgap=args.mipgap,
        gurobi_threads=args.threads,
        gurobi_mipfocus=args.mipfocus,
    )
    if args.use_mip_start is not None:
        cfg.use_mip_start = args.use_mip_start
    comparison_apply_steps = max(1, int(round(3.0 / cfg.dt_hours)))
    cfg_mpc_dynamic_v1 = EnergySystemConfig(
        system_version=1,
        horizon_steps=args.horizon,
        gurobi_mipgap=args.mipgap,
        gurobi_threads=args.threads,
        gurobi_mipfocus=args.mipfocus,
        mpc_apply_steps=comparison_apply_steps,
    )
    cfg_mpc_static = EnergySystemConfig(
        system_version=1,
        horizon_steps=args.horizon,
        gurobi_mipgap=args.mipgap,
        gurobi_threads=args.threads,
        gurobi_mipfocus=args.mipfocus,
        mpc_apply_steps=comparison_apply_steps,
    )
    cfg_mpc_static.static_mpc_import_price_eur_per_kwh = cfg.static_mpc_import_price_eur_per_kwh
    if args.use_mip_start is not None:
        cfg_mpc_dynamic_v1.use_mip_start = args.use_mip_start
        cfg_mpc_static.use_mip_start = args.use_mip_start

    # 1) Load and validate the raw input file.
    t_stage = perf_counter()
    _stage("Loading CSV")
    raw = load_csv(args.csv_path)
    print(f"[stage] Loading CSV done in {perf_counter() - t_stage:.2f}s", flush=True)

    # 2) Normalize input columns to the model-ready schema.
    t_stage = perf_counter()
    _stage("Preprocessing data")
    data = preprocess(raw, cfg).iloc[: args.steps].copy()
    data_mpc_dynamic_v1 = preprocess(raw, cfg_mpc_dynamic_v1).iloc[: args.steps].copy()
    data_mpc_static = preprocess(raw, cfg_mpc_static).iloc[: args.steps].copy()
    static_mpc_price = float(cfg_mpc_static.static_mpc_import_price_eur_per_kwh)
    data_mpc_static["import_price_eur_per_kwh"] = static_mpc_price
    data_mpc_static["ev_home_import_price_eur_per_kwh"] = static_mpc_price
    data_mpc_static["ev_export_price_eur_per_kwh"] = 0.0
    print(f"[stage] Preprocessing data done in {perf_counter() - t_stage:.2f}s", flush=True)

    # 3) Run both rule-based baseline variants (dynamic/static home import tariff).
    t_stage = perf_counter()
    _stage("Running baseline simulations")
    baseline_dynamic = BaselineDynamic().simulate(data, cfg)
    baseline_static = BaselineStatic().simulate(data, cfg)
    print(f"[stage] Baseline simulations done in {perf_counter() - t_stage:.2f}s", flush=True)

    # Build MPC-only reserve profile: fixed reserve by default, baseline SOC when below fixed.
    data_mpc = data.copy()
    fixed_reserve_kwh = cfg.ev_soc_min * cfg.ev_cap_kwh
    baseline_soc_kwh = pd.to_numeric(baseline_dynamic["ev_energy_kwh"], errors="coerce").reindex(data_mpc.index).fillna(0.0)
    adaptive_reserve_kwh = pd.Series(fixed_reserve_kwh, index=data_mpc.index, dtype=float)
    baseline_below_fixed_mask = baseline_soc_kwh < fixed_reserve_kwh
    adaptive_reserve_kwh.loc[baseline_below_fixed_mask] = baseline_soc_kwh.loc[baseline_below_fixed_mask]
    adaptive_reserve_kwh = adaptive_reserve_kwh.clip(lower=0.0, upper=cfg.ev_cap_kwh)
    data_mpc["ev_reserve_kwh"] = adaptive_reserve_kwh
    data_mpc_dynamic_v1["ev_reserve_kwh"] = adaptive_reserve_kwh.reindex(data_mpc_dynamic_v1.index).astype(float)
    data_mpc_static["ev_reserve_kwh"] = adaptive_reserve_kwh.reindex(data_mpc_static.index).astype(float)
    # Export adaptive reserve profile for both baseline and MPC result tables.
    baseline_dynamic["ev_reserve_kwh"] = adaptive_reserve_kwh.reindex(baseline_dynamic.index).astype(float)
    baseline_static["ev_reserve_kwh"] = adaptive_reserve_kwh.reindex(baseline_static.index).astype(float)

    # 4) Import MPC lazily to avoid solver import overhead when not needed.
    t_stage = perf_counter()
    _stage("Importing MPC module")
    from home_energy_opt.mpc import run_mpc_loop
    print(f"[stage] Importing MPC module done in {perf_counter() - t_stage:.2f}s", flush=True)

    # 5) Execute rolling-horizon MPC and collect solver timing logs.
    t_stage = perf_counter()
    _stage("Running MPC loop")
    mpc, logs = run_mpc_loop(
        data_mpc,
        cfg,
        progress_every=args.progress_every,
        slow_step_sec=args.slow_step_sec,
        solver_tee=args.solver_tee,
        use_persistent_gurobi=not args.legacy_gurobi_rebuild,
        use_mip_start=args.use_mip_start,
    )
    print(f"[stage] Running MPC loop done in {perf_counter() - t_stage:.2f}s", flush=True)

    t_stage = perf_counter()
    _stage("Running dynamic V1 MPC loop")
    mpc_dynamic_v1, logs_dynamic_v1 = run_mpc_loop(
        data_mpc_dynamic_v1,
        cfg_mpc_dynamic_v1,
        progress_every=args.progress_every,
        slow_step_sec=args.slow_step_sec,
        solver_tee=args.solver_tee,
        use_persistent_gurobi=not args.legacy_gurobi_rebuild,
        use_mip_start=args.use_mip_start,
    )
    print(f"[stage] Running dynamic V1 MPC loop done in {perf_counter() - t_stage:.2f}s", flush=True)

    t_stage = perf_counter()
    _stage("Running static MPC loop")
    mpc_static, logs_static = run_mpc_loop(
        data_mpc_static,
        cfg_mpc_static,
        progress_every=args.progress_every,
        slow_step_sec=args.slow_step_sec,
        solver_tee=args.solver_tee,
        use_persistent_gurobi=not args.legacy_gurobi_rebuild,
        use_mip_start=args.use_mip_start,
    )
    print(f"[stage] Running static MPC loop done in {perf_counter() - t_stage:.2f}s", flush=True)

    # Sanity check: when baseline SOC is below fixed reserve, MPC SOC must not drop below baseline SOC.
    sanity_violations_df, sanity_violation_steps, sanity_max_deficit_kwh = _adaptive_reserve_sanity_check(
        baseline_soc_kwh,
        mpc,
        fixed_reserve_kwh,
        "mpc",
    )
    (
        sanity_violations_dynamic_v1_df,
        sanity_violation_dynamic_v1_steps,
        sanity_max_deficit_dynamic_v1_kwh,
    ) = _adaptive_reserve_sanity_check(
        baseline_soc_kwh,
        mpc_dynamic_v1,
        fixed_reserve_kwh,
        "mpc_dynamic_v1",
    )
    (
        sanity_violations_static_df,
        sanity_violation_static_steps,
        sanity_max_deficit_static_kwh,
    ) = _adaptive_reserve_sanity_check(
        baseline_soc_kwh,
        mpc_static,
        fixed_reserve_kwh,
        "mpc_static",
    )

    # 6) Persist raw simulation outputs so they can be inspected later.
    t_stage = perf_counter()
    _stage("Writing outputs")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Keep baseline_results.csv as dynamic for backward compatibility with existing scripts.
    baseline_dynamic.to_csv(outdir / "baseline_results.csv")
    baseline_dynamic.to_csv(outdir / "baseline_dynamic_results.csv")
    baseline_static.to_csv(outdir / "baseline_static_results.csv")
    mpc.to_csv(outdir / "mpc_results.csv")
    mpc_dynamic_v1.to_csv(outdir / "mpc_dynamic_v1_results.csv")
    mpc_static.to_csv(outdir / "mpc_static_results.csv")
    pd.DataFrame(logs).to_csv(outdir / "mpc_solver_logs.csv", index=False)
    pd.DataFrame(logs_dynamic_v1).to_csv(outdir / "mpc_dynamic_v1_solver_logs.csv", index=False)
    pd.DataFrame(logs_static).to_csv(outdir / "mpc_static_solver_logs.csv", index=False)
    sanity_violations_df.to_csv(outdir / "mpc_baseline_reserve_sanity_violations.csv")
    sanity_violations_dynamic_v1_df.to_csv(outdir / "mpc_dynamic_v1_baseline_reserve_sanity_violations.csv")
    sanity_violations_static_df.to_csv(outdir / "mpc_static_baseline_reserve_sanity_violations.csv")

    # Detect simultaneous opposite flows from the already-solved MPC trajectory.
    grid_mask = _overlap_mask(mpc["grid_import_kw"], mpc["grid_export_kw"])
    bat_mask = _overlap_mask(mpc["bat_ch_kw"], mpc["bat_dis_kw"])
    both_mask = grid_mask & bat_mask

    windows_rows = []
    for flow_type, mask in (
        ("grid_import_export", grid_mask),
        ("battery_charge_discharge", bat_mask),
        ("both_grid_and_battery", both_mask),
    ):
        for start, end, n_steps in _mask_windows(mask):
            windows_rows.append(
                {
                    "flow_type": flow_type,
                    "start_time": str(start),
                    "end_time": str(end),
                    "steps": int(n_steps),
                }
            )
    pd.DataFrame(windows_rows).to_csv(outdir / "simultaneous_flow_windows.csv", index=False)

    print(f"[stage] Writing outputs done in {perf_counter() - t_stage:.2f}s", flush=True)

    # 7) Aggregate KPIs for both strategies into one comparison table.
    t_stage = perf_counter()
    _stage("Computing metrics")
    metrics_baseline_dynamic = summarize_metrics(data, baseline_dynamic, cfg)
    metrics_baseline_static = summarize_metrics(data, baseline_static, cfg)
    metrics_mpc = summarize_metrics(data_mpc, mpc, cfg)
    metrics_mpc_dynamic_v1 = summarize_metrics(data_mpc_dynamic_v1, mpc_dynamic_v1, cfg_mpc_dynamic_v1)
    metrics_mpc_static = summarize_metrics(data_mpc_static, mpc_static, cfg_mpc_static)
    sanity_count_key = "mpc_baseline_reserve_sanity_violation_steps"
    sanity_max_deficit_key = "mpc_baseline_reserve_sanity_max_deficit_kwh"
    metrics_baseline_dynamic[sanity_count_key] = 0.0
    metrics_baseline_dynamic[sanity_max_deficit_key] = 0.0
    metrics_baseline_static[sanity_count_key] = 0.0
    metrics_baseline_static[sanity_max_deficit_key] = 0.0
    metrics_mpc[sanity_count_key] = float(sanity_violation_steps)
    metrics_mpc[sanity_max_deficit_key] = sanity_max_deficit_kwh
    metrics_mpc_dynamic_v1[sanity_count_key] = float(sanity_violation_dynamic_v1_steps)
    metrics_mpc_dynamic_v1[sanity_max_deficit_key] = sanity_max_deficit_dynamic_v1_kwh
    metrics_mpc_static[sanity_count_key] = float(sanity_violation_static_steps)
    metrics_mpc_static[sanity_max_deficit_key] = sanity_max_deficit_static_kwh
    metrics = pd.DataFrame(
        [metrics_baseline_static, metrics_baseline_dynamic, metrics_mpc_static, metrics_mpc_dynamic_v1, metrics_mpc],
        index=["baseline_static", "baseline_dynamic", "mpc_static", "mpc_dynamic_v1", "mpc"],
    )
    metrics.to_csv(outdir / "metrics_comparison.csv")
    print(f"[stage] Computing metrics done in {perf_counter() - t_stage:.2f}s", flush=True)

    # 8) Render interactive HTML dashboard for system-connection analysis.
    t_stage = perf_counter()
    _stage("Generating plots")
    from home_energy_opt.plots import save_system_connection_interactive_html

    save_system_connection_interactive_html(
        data.index,
        data,
        baseline_dynamic,
        mpc,
        str(outdir / "system_connection_comparison_interactive.html"),
        show_battery=cfg.enable_battery,
        show_pv=cfg.enable_pv,
    )
    print("[stage] Interactive HTML export ready", flush=True)
    print(f"[stage] Generating plots done in {perf_counter() - t_stage:.2f}s", flush=True)

    print("Baseline static metrics:")
    print(metrics.loc["baseline_static"])
    print("\nBaseline dynamic metrics:")
    print(metrics.loc["baseline_dynamic"])
    print("\nStatic MPC metrics:")
    print(metrics.loc["mpc_static"])
    print("\nDynamic V1 MPC metrics:")
    print(metrics.loc["mpc_dynamic_v1"])
    print("\nMPC metrics:")
    print(metrics.loc["mpc"])
    print("\nSimultaneous opposite-flow counts (MPC run):")
    print(f"grid_import_export_steps: {int(grid_mask.sum())}")
    print(f"battery_charge_discharge_steps: {int(bat_mask.sum())}")
    print(f"both_grid_and_battery_steps: {int(both_mask.sum())}")
    print(f"windows file: {(outdir / 'simultaneous_flow_windows.csv').resolve()}")
    print(f"\nTotal runtime: {perf_counter() - t0:.2f}s")
    print(f"\nOutputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
