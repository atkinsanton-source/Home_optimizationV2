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


def main() -> None:
    """Run the full workflow: load data, simulate, compare, and export artifacts."""
    t0 = perf_counter()
    args = parse_args()
    _stage("Importing core modules")
    t_stage = perf_counter()
    import pandas as pd

    from home_energy_opt.baseline import simulate_baseline
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

    # 1) Load and validate the raw input file.
    t_stage = perf_counter()
    _stage("Loading CSV")
    raw = load_csv(args.csv_path)
    print(f"[stage] Loading CSV done in {perf_counter() - t_stage:.2f}s", flush=True)

    # 2) Normalize input columns to the model-ready schema.
    t_stage = perf_counter()
    _stage("Preprocessing data")
    data = preprocess(raw, cfg).iloc[: args.steps].copy()
    print(f"[stage] Preprocessing data done in {perf_counter() - t_stage:.2f}s", flush=True)

    # 3) Run the rule-based baseline controller as reference.
    t_stage = perf_counter()
    _stage("Running baseline simulation")
    baseline = simulate_baseline(data, cfg)
    print(f"[stage] Baseline simulation done in {perf_counter() - t_stage:.2f}s", flush=True)

    # 4) Import MPC lazily to avoid solver import overhead when not needed.
    t_stage = perf_counter()
    _stage("Importing MPC module")
    from home_energy_opt.mpc import run_mpc_loop
    print(f"[stage] Importing MPC module done in {perf_counter() - t_stage:.2f}s", flush=True)

    # 5) Execute rolling-horizon MPC and collect solver timing logs.
    t_stage = perf_counter()
    _stage("Running MPC loop")
    mpc, logs = run_mpc_loop(
        data,
        cfg,
        progress_every=args.progress_every,
        slow_step_sec=args.slow_step_sec,
        solver_tee=args.solver_tee,
        use_persistent_gurobi=not args.legacy_gurobi_rebuild,
        use_mip_start=args.use_mip_start,
    )
    print(f"[stage] Running MPC loop done in {perf_counter() - t_stage:.2f}s", flush=True)

    # 6) Persist raw simulation outputs so they can be inspected later.
    t_stage = perf_counter()
    _stage("Writing outputs")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    baseline.to_csv(outdir / "baseline_results.csv")
    mpc.to_csv(outdir / "mpc_results.csv")
    pd.DataFrame(logs).to_csv(outdir / "mpc_solver_logs.csv", index=False)

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
    metrics_baseline = summarize_metrics(data, baseline, cfg)
    metrics_mpc = summarize_metrics(data, mpc, cfg)
    metrics = pd.DataFrame([metrics_baseline, metrics_mpc], index=["baseline", "mpc"])
    metrics.to_csv(outdir / "metrics_comparison.csv")
    print(f"[stage] Computing metrics done in {perf_counter() - t_stage:.2f}s", flush=True)

    # 8) Render interactive HTML dashboard for system-connection analysis.
    t_stage = perf_counter()
    _stage("Generating plots")
    from home_energy_opt.plots import save_system_connection_interactive_html

    save_system_connection_interactive_html(
        data.index,
        data,
        baseline,
        mpc,
        str(outdir / "system_connection_comparison_interactive.html"),
        show_battery=cfg.enable_battery,
        show_pv=cfg.enable_pv,
    )
    print("[stage] Interactive HTML export ready", flush=True)
    print(f"[stage] Generating plots done in {perf_counter() - t_stage:.2f}s", flush=True)

    print("Baseline metrics:")
    print(metrics.loc["baseline"])
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
