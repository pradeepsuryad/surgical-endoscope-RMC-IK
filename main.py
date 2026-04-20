"""
main.py
=======
Entry point for the Surgical Endoscope Tracking simulation.

Usage
-----
    python main.py                           # default settings, with viewer
    python main.py --no-render               # headless (CI / server)
    python main.py --laps 2 --max-iter 10    # 2 full circles, 10 NR iters
    python main.py --radius 0.12 --height 0.45 --step-mm 0.5

Pipeline
--------
1. Build CircularTrajectory  (≈ 1 mm waypoint spacing around a circle)
2. Instantiate NewtonRaphsonIK solver (configurable max iterations)
3. Run EndoscopeSimulation  (MuJoCo physics + IK loop + data logging)
4. Generate all analysis plots to results/

Requirements
------------
Place the MuJoCo Menagerie Franka Panda XML at:
    models/franka_emika_panda/panda.xml

See README.md for full setup instructions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ik_solver import NewtonRaphsonIK
from src.simulation import EndoscopeSimulation
from src.trajectory import CircularTrajectory
from src.visualizer import plot_all


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Surgical Endoscope Tracking - 7-DOF Panda SE(3) IK Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--no-render",  action="store_true",
                   help="Disable the MuJoCo viewer window (headless mode)")
    p.add_argument("--laps",       type=int,   default=1,
                   help="Number of full laps around the circle")
    p.add_argument("--max-iter",   type=int,   default=5,
                   help="Max Newton-Raphson iterations per IK call")
    p.add_argument("--radius",     type=float, default=0.15,
                   help="Circle radius [m]")
    p.add_argument("--height",     type=float, default=0.50,
                   help="Height of the circle plane above the base [m]")
    p.add_argument("--step-mm",    type=float, default=1.0,
                   help="Desired arc-length spacing between waypoints [mm]")
    p.add_argument("--model",      type=str,   default=None,
                   help="Path to Panda MuJoCo XML (overrides default)")
    p.add_argument("--record",     action="store_true",
                   help="Record a video to results/simulation.mp4 (offscreen)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------
    # 1. Trajectory
    # ------------------------------------------------------------------
    traj = CircularTrajectory(
        radius=args.radius,
        z_height=args.height,
        step_mm=args.step_mm,
    )
    print(
        f"[Trajectory]  {traj.n_waypoints} waypoints  "
        f"(R={args.radius:.3f} m, Z={args.height:.3f} m, ds={args.step_mm:.1f} mm)"
    )

    # ------------------------------------------------------------------
    # 2. IK solver
    # ------------------------------------------------------------------
    ik = NewtonRaphsonIK(max_iter=args.max_iter)
    print(f"[IK Solver]   Newton-Raphson  max_iter={args.max_iter}  (+ DLS fallback)")

    # ------------------------------------------------------------------
    # 3. Simulation
    # ------------------------------------------------------------------
    sim = EndoscopeSimulation(
        model_xml_path=args.model,
        trajectory=traj,
        ik_solver=ik,
        n_laps=args.laps,
        render=not args.no_render,
        record_video=args.record,
    )
    print(f"[Simulation]  Starting - {traj.n_waypoints * args.laps} total steps ...")
    log = sim.run()
    print(
        f"[Simulation]  Complete - {len(log.steps)} steps logged.\n"
        f"              Mean pos error : {log.pos_errors.mean()*1e3:.3f} mm\n"
        f"              Mean ori error : {log.ori_errors.mean():.5f} rad"
    )

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    plot_all(log, output_dir=results_dir)
    print(f"\n[Done]  All plots saved to: {results_dir}/")


if __name__ == "__main__":
    main()
