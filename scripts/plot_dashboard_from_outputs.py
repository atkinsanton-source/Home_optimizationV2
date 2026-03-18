from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

# Make `home_energy_opt` importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from home_energy_opt.config import EnergySystemConfig
from home_energy_opt.data import load_csv, preprocess
from home_energy_opt.plots import save_system_connection_interactive_html


def _read_results(path: Path) -> pd.DataFrame:
    """Read one simulation result CSV and enforce timestamp index."""
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    return df.sort_index()


def _stage(msg: str) -> None:
    print(f"[plot-only] {msg}", flush=True)


def main() -> None:
    """Rebuild interactive dashboard from existing baseline/MPC outputs only."""
    t0 = perf_counter()
    p = argparse.ArgumentParser(description="Rebuild interactive dashboard from existing baseline/mpc result CSVs")
    p.add_argument("csv_path", type=str, help="Original input CSV used for the simulation")
    p.add_argument(
        "--system-version",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="System version preset (controls visibility of PV/battery traces)",
    )
    p.add_argument("--outdir", type=str, default="outputs", help="Directory containing baseline/mpc result CSVs")
    p.add_argument(
        "--output-name",
        type=str,
        default="system_connection_comparison_interactive.html",
        help="Output HTML filename written into --outdir",
    )
    args = p.parse_args()

    outdir = Path(args.outdir)
    baseline_path = outdir / "baseline_results.csv"
    mpc_path = outdir / "mpc_results.csv"
    out_path = outdir / args.output_name

    _stage("Loading and preprocessing input CSV")
    cfg = EnergySystemConfig(system_version=args.system_version)
    raw = load_csv(args.csv_path)
    data = preprocess(raw, cfg)

    _stage("Reading baseline/mpc result CSVs")
    baseline = _read_results(baseline_path)
    mpc = _read_results(mpc_path)

    _stage("Aligning common timestamps")
    common_index = baseline.index.intersection(mpc.index).intersection(data.index)
    if common_index.empty:
        raise ValueError("No overlapping timestamps between data, baseline results, and MPC results.")

    baseline = baseline.loc[common_index]
    mpc = mpc.loc[common_index]
    data = data.loc[common_index]

    _stage("Rendering interactive HTML dashboard")
    save_system_connection_interactive_html(
        common_index,
        data,
        baseline,
        mpc,
        str(out_path),
        show_battery=cfg.enable_battery,
        show_pv=cfg.enable_pv,
    )
    print(f"Wrote: {out_path.resolve()}")
    print(f"[plot-only] Done in {perf_counter() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
