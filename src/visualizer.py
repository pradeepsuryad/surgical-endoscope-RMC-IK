"""
visualizer.py
=============
Matplotlib / Seaborn post-processing plots for the surgical endoscope
tracking simulation.  All figures are saved to the results/ directory as
high-resolution PNGs.

Generated plots
---------------
1. plot_3d_trajectory       — desired circle vs actual executed path (3-D)
2. plot_error_over_time     — position error (mm) and orientation error (rad)
3. plot_fk_comparison       — analytical FK vs MuJoCo FK (x, y, z vs timestep)
4. plot_joint_angles        — all 7 joint angles over time
5. plot_ik_convergence      — mean ± σ NR residual ‖e_k‖ per iteration
6. plot_computation_time    — per-step IK solve time
7. plot_all                 — calls all of the above
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")   # headless backend so plots work without a display

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from numpy.typing import NDArray

from .simulation import SimulationLog

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
_FIG_DPI = 150

_COLORS = sns.color_palette("muted")
_C_DESIRED  = _COLORS[0]
_C_ACTUAL   = _COLORS[1]
_C_ANALYTIC = _COLORS[2]
_C_MUJOCO   = _COLORS[3]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(str(path), dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path.name}")


# ---------------------------------------------------------------------------
# 1. 3D Trajectory
# ---------------------------------------------------------------------------

def plot_3d_trajectory(log: SimulationLog, output_dir: Path) -> None:
    """3D scatter/line: desired circle vs actual executed EE path."""

    desired  = log.desired_pos           # (T, 3)
    actual   = log.ee_pos_mujoco         # (T, 3)
    analytic = log.ee_pos_analytical     # (T, 3)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(desired[:, 0], desired[:, 1], desired[:, 2],
            "--", color=_C_DESIRED, lw=2.0, label="Desired trajectory", alpha=0.7)
    ax.plot(actual[:, 0], actual[:, 1], actual[:, 2],
            "-", color=_C_ACTUAL, lw=1.5, label="Actual (MuJoCo FK)")
    ax.plot(analytic[:, 0], analytic[:, 1], analytic[:, 2],
            ":", color=_C_ANALYTIC, lw=1.5, label="Analytical FK", alpha=0.8)

    # Mark start / end
    ax.scatter(*desired[0], color="green", s=60, zorder=5, label="Start")
    ax.scatter(*desired[-1], color="red",  s=60, zorder=5, label="End")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("3D End-Effector Trajectory\n(Desired vs Actual)", pad=12)
    ax.legend(loc="upper left", fontsize=9)

    _save(fig, output_dir / "01_3d_trajectory.png")


# ---------------------------------------------------------------------------
# 2. Error over time
# ---------------------------------------------------------------------------

def plot_error_over_time(log: SimulationLog, output_dir: Path) -> None:
    """Two-panel plot: position error (mm) and orientation error (rad)."""

    t       = log.sim_times
    pos_err = log.pos_errors * 1e3        # m → mm
    ori_err = log.ori_errors              # rad

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Position error
    axes[0].plot(t, pos_err, color=_C_ACTUAL, lw=1.4)
    axes[0].fill_between(t, 0, pos_err, alpha=0.15, color=_C_ACTUAL)
    axes[0].set_ylabel("Position error ||e_p|| [mm]")
    axes[0].set_title("SE(3) Tracking Error over Time")
    axes[0].axhline(1.0, ls="--", color="gray", lw=0.9, label="1 mm threshold")
    axes[0].legend(fontsize=9)

    # Orientation error
    axes[1].plot(t, ori_err, color=_C_ANALYTIC, lw=1.4)
    axes[1].fill_between(t, 0, ori_err, alpha=0.15, color=_C_ANALYTIC)
    axes[1].set_ylabel("Orientation error ||e_o|| [rad]")
    axes[1].set_xlabel("Simulation time [s]")

    fig.tight_layout()
    _save(fig, output_dir / "02_error_over_time.png")


# ---------------------------------------------------------------------------
# 3. FK comparison
# ---------------------------------------------------------------------------

def plot_fk_comparison(log: SimulationLog, output_dir: Path) -> None:
    """Compare analytical FK vs MuJoCo FK for x, y, z independently."""

    t        = log.sim_times
    analytic = log.ee_pos_analytical   # (T, 3)
    mujoco   = log.ee_pos_mujoco       # (T, 3)
    labels   = ["X [m]", "Y [m]", "Z [m]"]

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    for i, (ax, lab) in enumerate(zip(axes[:3], labels)):
        ax.plot(t, mujoco[:, i],   color=_C_MUJOCO,   lw=1.5, label="MuJoCo FK",    alpha=0.9)
        ax.plot(t, analytic[:, i], color=_C_ANALYTIC, lw=1.2, ls="--",
                label="Analytical FK", alpha=0.9)
        ax.set_ylabel(lab)
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_title("Analytical FK vs MuJoCo FK — EE Position Comparison")

    # Discrepancy norm (mm)
    diff_mm = np.linalg.norm(analytic - mujoco, axis=1) * 1e3
    axes[3].plot(t, diff_mm, color="tomato", lw=1.4)
    axes[3].fill_between(t, 0, diff_mm, alpha=0.15, color="tomato")
    axes[3].set_ylabel("‖FK discrepancy‖ [mm]")
    axes[3].set_xlabel("Simulation time [s]")
    axes[3].set_title("FK Discrepancy Norm (Analytical − MuJoCo)")

    fig.tight_layout()
    _save(fig, output_dir / "03_fk_comparison.png")


# ---------------------------------------------------------------------------
# 4. Joint angles
# ---------------------------------------------------------------------------

def plot_joint_angles(log: SimulationLog, output_dir: Path) -> None:
    """Plot all 7 joint angles over time on a shared time axis."""

    t = log.sim_times
    Q = log.q_history                  # (T, 7)

    fig, axes = plt.subplots(7, 1, figsize=(13, 14), sharex=True)
    palette = sns.color_palette("tab10", 7)

    for j, (ax, color) in enumerate(zip(axes, palette)):
        ax.plot(t, np.degrees(Q[:, j]), color=color, lw=1.4)
        ax.set_ylabel(f"q{j+1} [°]", fontsize=9)
        ax.grid(True, alpha=0.4)

    axes[0].set_title("Joint Angles over Time (smooth motion verification)")
    axes[-1].set_xlabel("Simulation time [s]")

    fig.tight_layout()
    _save(fig, output_dir / "04_joint_angles.png")


# ---------------------------------------------------------------------------
# 5. IK convergence (per-iteration residual)
# ---------------------------------------------------------------------------

def plot_ik_convergence(log: SimulationLog, output_dir: Path) -> None:
    """Mean ± σ of NR residual ‖e_k‖ at each iteration, across all timesteps."""

    # Prefer converged steps: they show actual NR decay from non-trivial initial
    # error down to tolerance. Non-converged (joint-limit) steps are nearly flat.
    overall_max = max((len(s.ik_error_norms) for s in log.steps), default=0)
    converged_norms = [
        s.ik_error_norms for s in log.steps
        if s.ik_converged and s.ik_error_norms
    ]
    if len(converged_norms) >= 10:
        all_norms = converged_norms
        step_label = f"converged steps, n = {len(converged_norms)}"
    else:
        all_norms = [
            s.ik_error_norms for s in log.steps
            if len(s.ik_error_norms) == overall_max
        ]
        if not all_norms:
            all_norms = [s.ik_error_norms for s in log.steps if s.ik_error_norms]
        step_label = f"non-converged steps, n = {len(all_norms)}"
    if not all_norms:
        return

    # Pad all sequences to overall_max so x-axis spans the full NR budget
    padded = np.array(
        [x + [x[-1]] * (overall_max - len(x)) for x in all_norms],
        dtype=np.float64,
    )                                     # (T, overall_max)

    iters  = np.arange(overall_max)
    mean_e = padded.mean(axis=0)
    std_e  = padded.std(axis=0)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.semilogy(iters, mean_e, "o-", color=_C_ACTUAL, lw=2.0, ms=6, label="Mean ||e_k||")
    ax.fill_between(
        iters,
        np.maximum(mean_e - std_e, 1e-10),
        mean_e + std_e,
        alpha=0.2,
        color=_C_ACTUAL,
        label="± 1 σ",
    )

    ax.set_xlabel("Newton-Raphson iteration k")
    ax.set_ylabel("Spatial error norm ||e_k||")
    ax.set_title(f"IK Convergence per Iteration\n"
                 f"(max_iter = {overall_max - 1}, {step_label})")
    ax.legend()

    # Annotate percentage reduction
    if len(mean_e) >= 2 and mean_e[0] > 1e-10:
        reduction_pct = (1.0 - mean_e[-1] / mean_e[0]) * 100.0
        ax.text(0.62, 0.85,
                f"Average reduction: {reduction_pct:.1f}%",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    fig.tight_layout()
    _save(fig, output_dir / "05_ik_convergence.png")


# ---------------------------------------------------------------------------
# 6. Computation time
# ---------------------------------------------------------------------------

def plot_computation_time(log: SimulationLog, output_dir: Path) -> None:
    """Per-step IK solve time with moving average overlay."""

    t          = log.sim_times
    solve_ms   = log.ik_solve_times * 1e3   # s → ms

    window = max(1, len(solve_ms) // 30)
    ma     = np.convolve(solve_ms, np.ones(window) / window, mode="valid")
    t_ma   = t[:len(ma)]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(t, solve_ms, color=_C_DESIRED, lw=0.8, alpha=0.5, label="Per-step")
    ax.plot(t_ma, ma,   color=_C_ACTUAL,   lw=2.0, label=f"Moving avg (w={window})")
    ax.axhline(solve_ms.mean(), ls="--", color="gray", lw=1.0,
               label=f"Mean = {solve_ms.mean():.2f} ms")

    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("IK solve time [ms]")
    ax.set_title("Newton-Raphson IK Computation Time per Step")
    ax.legend()

    fig.tight_layout()
    _save(fig, output_dir / "06_computation_time.png")


# ---------------------------------------------------------------------------
# 7. Summary dashboard
# ---------------------------------------------------------------------------

def plot_summary_dashboard(log: SimulationLog, output_dir: Path) -> None:
    """Single-page summary of key metrics for recruiters / quick inspection."""

    t       = log.sim_times
    pos_mm  = log.pos_errors * 1e3
    ori_rad = log.ori_errors
    Q       = log.q_history
    ms      = log.ik_solve_times * 1e3

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    fig.suptitle(
        "Surgical Endoscope Tracking — Simulation Summary\n"
        "7-DOF Franka Panda | SE(3) Newton-Raphson IK | Circular Trajectory",
        fontsize=13, fontweight="bold",
    )

    gs = gridspec.GridSpec(3, 3, figure=fig)

    # --- 3D trajectory ---
    ax3d = fig.add_subplot(gs[:2, :2], projection="3d")
    desired = log.desired_pos
    actual  = log.ee_pos_mujoco
    ax3d.plot(desired[:, 0], desired[:, 1], desired[:, 2],
              "--", color=_C_DESIRED, lw=2, label="Desired", alpha=0.7)
    ax3d.plot(actual[:, 0], actual[:, 1], actual[:, 2],
              "-", color=_C_ACTUAL, lw=1.5, label="Actual")
    ax3d.set_xlabel("X [m]"); ax3d.set_ylabel("Y [m]"); ax3d.set_zlabel("Z [m]")
    ax3d.set_title("3D Trajectory")
    ax3d.legend(fontsize=8)

    # --- Position error ---
    ax_pe = fig.add_subplot(gs[0, 2])
    ax_pe.plot(t, pos_mm, color=_C_ACTUAL, lw=1.3)
    ax_pe.set_ylabel("||e_p|| [mm]"); ax_pe.set_title("Position Error")
    ax_pe.axhline(1.0, ls="--", color="gray", lw=0.9)

    # --- Orientation error ---
    ax_oe = fig.add_subplot(gs[1, 2], sharex=ax_pe)
    ax_oe.plot(t, ori_rad, color=_C_ANALYTIC, lw=1.3)
    ax_oe.set_ylabel("||e_o|| [rad]"); ax_oe.set_title("Orientation Error")
    ax_oe.set_xlabel("Time [s]")

    # --- Joint angles (all 7 overlaid) ---
    ax_q = fig.add_subplot(gs[2, :2])
    palette = sns.color_palette("tab10", 7)
    for j, color in enumerate(palette):
        ax_q.plot(t, np.degrees(Q[:, j]), color=color, lw=0.9, label=f"q{j+1}")
    ax_q.set_ylabel("Joint angle [°]"); ax_q.set_xlabel("Time [s]")
    ax_q.set_title("Joint Angles")
    ax_q.legend(ncol=7, fontsize=7, loc="upper right")

    # --- IK solve time ---
    ax_ms = fig.add_subplot(gs[2, 2])
    ax_ms.plot(t, ms, color=_C_DESIRED, lw=0.8, alpha=0.6)
    ax_ms.axhline(ms.mean(), ls="--", color="gray", lw=1.0,
                  label=f"μ={ms.mean():.2f} ms")
    ax_ms.set_ylabel("IK time [ms]"); ax_ms.set_xlabel("Time [s]")
    ax_ms.set_title("IK Solve Time")
    ax_ms.legend(fontsize=8)

    _save(fig, output_dir / "00_summary_dashboard.png")


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def plot_all(log: SimulationLog, output_dir: Path) -> None:
    """Generate all analysis plots and save to *output_dir*.

    Parameters
    ----------
    log        : SimulationLog  collected by EndoscopeSimulation.run().
    output_dir : Path           directory to write PNGs (created if needed).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Visualizer] Generating plots ({len(log.steps)} steps) ...")

    plot_summary_dashboard(log, output_dir)
    plot_3d_trajectory(log, output_dir)
    plot_error_over_time(log, output_dir)
    plot_fk_comparison(log, output_dir)
    plot_joint_angles(log, output_dir)
    plot_ik_convergence(log, output_dir)
    plot_computation_time(log, output_dir)

    print(f"[Visualizer] {7} plots saved to {output_dir}/")
