from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from home_energy_opt.config import EnergySystemConfig
from home_energy_opt.data import load_csv, preprocess
from home_energy_opt.mpc import run_mpc_loop


def _sum_log_field(logs: list[dict[str, object]], key: str) -> float:
    """Sum one numeric field from log rows while tolerating missing keys."""
    vals = [float(x[key]) for x in logs if key in x]
    return float(np.sum(vals)) if vals else 0.0


def _count_bad_status(df: pd.DataFrame) -> int:
    """Count MPC steps that did not end in an optimal/feasible status."""
    ok = {"optimal", "feasible"}
    return int((~df["solver_status"].isin(ok)).sum())


def main() -> None:
    """Compare legacy and persistent Gurobi MPC modes for result equivalence and timing."""
    p = argparse.ArgumentParser(description="Regression check: legacy rebuild gurobi vs persistent gurobi MPC")
    p.add_argument("csv_path", type=str)
    p.add_argument("--steps", type=int, default=2 * 24 * 4)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument("--use-mip-start", action="store_true")
    args = p.parse_args()

    cfg = EnergySystemConfig(horizon_steps=args.horizon)
    raw = load_csv(args.csv_path)
    data = preprocess(raw, cfg).iloc[: args.steps].copy()

    mpc_old, logs_old = run_mpc_loop(
        data,
        cfg,
        show_progress=False,
        use_persistent_gurobi=False,
        use_mip_start=False,
    )

    mpc_new, logs_new = run_mpc_loop(
        data,
        cfg,
        show_progress=False,
        use_persistent_gurobi=True,
        use_mip_start=args.use_mip_start,
    )

    bad_old = _count_bad_status(mpc_old)
    bad_new = _count_bad_status(mpc_new)

    # Validate that both variants achieve equivalent objective values.
    obj_old = float(mpc_old["step_cost_eur"].sum())
    obj_new = float(mpc_new["step_cost_eur"].sum())
    obj_close = np.isclose(obj_old, obj_new, rtol=args.rtol, atol=args.atol)

    n_cmp = max(1, len(data) - args.horizon + 1)
    cmp_cols = ["grid_import_kw", "grid_export_kw", "bat_ch_kw", "bat_dis_kw", "ev_ch_kw", "ev_dis_kw"]
    first_action_close = True
    # Compare only windows where both methods optimize the same first action.
    for c in cmp_cols:
        a = mpc_old[c].iloc[:n_cmp].to_numpy()
        b = mpc_new[c].iloc[:n_cmp].to_numpy()
        if not np.allclose(a, b, rtol=args.rtol, atol=args.atol):
            first_action_close = False
            break

    print("=== Regression Summary ===")
    print(f"steps={len(data)} horizon={args.horizon} compared_steps={n_cmp}")
    print(f"feasibility_old_bad_status={bad_old}")
    print(f"feasibility_new_bad_status={bad_new}")
    print(f"objective_old_eur={obj_old:.12f}")
    print(f"objective_new_eur={obj_new:.12f}")
    print(f"objective_close={obj_close} (rtol={args.rtol}, atol={args.atol})")
    print(f"first_action_close={first_action_close} (rtol={args.rtol}, atol={args.atol})")

    print("\n=== Timing (old rebuild) ===")
    print(f"total_solve_seconds={_sum_log_field(logs_old, 'solve_seconds'):.6f}")
    print(f"total_step_seconds={_sum_log_field(logs_old, 'step_total_seconds'):.6f}")

    print("\n=== Timing (persistent) ===")
    print(f"build_once_time_seconds={_sum_log_field(logs_new, 'build_once_time_sec'):.6f}")
    print(f"total_update_seconds={_sum_log_field(logs_new, 'update_seconds'):.6f}")
    print(f"total_optimize_seconds={_sum_log_field(logs_new, 'solve_seconds'):.6f}")
    print(f"total_extract_seconds={_sum_log_field(logs_new, 'extract_seconds'):.6f}")
    print(f"total_step_seconds={_sum_log_field(logs_new, 'step_total_seconds'):.6f}")

    ok = bad_old == 0 and bad_new == 0 and obj_close and first_action_close
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
