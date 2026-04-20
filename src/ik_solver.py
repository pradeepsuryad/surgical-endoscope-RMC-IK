"""
ik_solver.py
============
Newton-Raphson Inverse Kinematics solver for a redundant (7-DOF) manipulator
operating in SE(3).

Update law
----------
    q_{k+1} = q_k + J^+ · e_k

where:
  • J  ∈ ℝ^{6×7}  — geometric Jacobian (linear rows first, angular rows last)
  • J^+ = Jᵀ(JJᵀ + λ²I)⁻¹  — damped Moore-Penrose pseudoinverse (DLS)
  • e_k ∈ ℝ⁶  — SE(3) spatial twist error (position + axis-angle)

Singularity handling
--------------------
The Levenberg-Marquardt (LM) damping coefficient λ is selected adaptively:
  • λ = lam_min  (≈ 0)  in regular configurations  → pure Moore-Penrose pinv
  • λ = lam_max         when σ_min(J) < sigma_thresh → smooth DLS fallback

The solver is intentionally capped at *max_iter* iterations per IK call
(default N = 5) to simulate real-time control latency while recording
per-iteration residuals for quantitative analysis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .kinematics import SE3, se3_error

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class IKResult:
    """Structured result returned by a single IK solve call."""

    q_final: NDArray[np.float64]        # (7,) final joint angles [rad]
    error_norms: list[float]            # ‖e_k‖ at each NR iteration
    pos_errors: list[float]             # ‖e_p‖ at each iteration [m]
    ori_errors: list[float]             # ‖e_o‖ at each iteration [rad]
    n_iters: int                        # iterations performed
    solve_time_s: float                 # wall-clock solve time [s]
    converged: bool                     # True if ‖e‖ < tol before max_iter
    damping_used: list[float] = field(default_factory=list)  # λ per iteration


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class NewtonRaphsonIK:
    """SE(3) Newton-Raphson IK with damped-least-squares singularity fallback.

    Parameters
    ----------
    max_iter     : int    Max NR iterations per solve call (default 5).
    tol          : float  Convergence threshold on ‖e‖ (default 1e-4).
    lam_min      : float  DLS damping near regular configs (≈ Moore-Penrose).
    lam_max      : float  DLS damping near singularities.
    sigma_thresh : float  Min singular value below which DLS activates.
    joint_limits : (7, 2) ndarray, optional  Lower/upper limits [rad].
    """

    # Official Franka Panda joint limits [rad]
    _PANDA_LIMITS: NDArray[np.float64] = np.array(
        [
            [-2.8973,  2.8973],
            [-1.7628,  1.7628],
            [-2.8973,  2.8973],
            [-3.0718, -0.0698],
            [-2.8973,  2.8973],
            [-0.0175,  3.7525],
            [-2.8973,  2.8973],
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        max_iter: int = 5,
        tol: float = 1e-4,
        lam_min: float = 1e-6,
        lam_max: float = 0.05,
        sigma_thresh: float = 0.05,
        null_gain: float = 0.5,
        joint_limits: NDArray[np.float64] | None = None,
    ) -> None:
        self.max_iter     = max_iter
        self.tol          = tol
        self.lam_min      = lam_min
        self.lam_max      = lam_max
        self.sigma_thresh = sigma_thresh
        self.null_gain    = null_gain   # secondary-task gain for joint-limit avoidance
        self.joint_limits = (
            self._PANDA_LIMITS if joint_limits is None
            else np.asarray(joint_limits, dtype=np.float64)
        )

    # ------------------------------------------------------------------
    # Damped pseudoinverse
    # ------------------------------------------------------------------

    def _damped_pinv(
        self,
        J: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float]:
        """Compute J^+ = Jᵀ(JJᵀ + λ²I)⁻¹ with adaptive LM damping.

        Parameters
        ----------
        J : (6, n) Jacobian.

        Returns
        -------
        J_pinv : (n, 6) damped pseudoinverse.
        lam    : float  actual damping coefficient used.
        """
        sigma_min = float(np.linalg.svd(J, compute_uv=False)[-1])
        lam = self.lam_max if sigma_min < self.sigma_thresh else self.lam_min

        m = J.shape[0]
        A = J @ J.T + (lam ** 2) * np.eye(m, dtype=np.float64)
        J_pinv = J.T @ np.linalg.inv(A)
        return J_pinv, lam

    # ------------------------------------------------------------------
    # Joint-limit clamping
    # ------------------------------------------------------------------

    def _clamp(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------

    def solve(
        self,
        q0: NDArray[np.float64],
        T_desired: SE3,
        jacobian_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        fk_fn: Callable[[NDArray[np.float64]], SE3],
    ) -> IKResult:
        """Run the Newton-Raphson IK loop.

        Parameters
        ----------
        q0          : (7,) initial joint configuration [rad].
        T_desired   : (4, 4) target SE(3) pose.
        jacobian_fn : callable  q → (6, 7) geometric Jacobian.
        fk_fn       : callable  q → (4, 4) SE(3) forward kinematics.

        Returns
        -------
        IKResult
        """
        t_start = time.perf_counter()

        q = q0.copy()
        error_norms: list[float] = []
        pos_errors:  list[float] = []
        ori_errors:  list[float] = []
        damping_used: list[float] = []
        converged = False

        for _ in range(self.max_iter):
            T_cur = fk_fn(q)
            e = se3_error(T_cur, T_desired)          # (6,)

            e_norm = float(np.linalg.norm(e))
            error_norms.append(e_norm)
            pos_errors.append(float(np.linalg.norm(e[:3])))
            ori_errors.append(float(np.linalg.norm(e[3:])))

            if e_norm < self.tol:
                converged = True
                break

            J = jacobian_fn(q)                       # (6, 7)
            J_pinv, lam = self._damped_pinv(J)
            damping_used.append(lam)

            # Primary task: SE(3) error minimisation
            dq_primary = J_pinv @ e                  # (7,)

            # Secondary task: null-space joint-limit avoidance
            # N = I - J⁺J projects into the 1-D null space of the 7-DOF arm
            n_joints = J.shape[1]
            N = np.eye(n_joints, dtype=np.float64) - J_pinv @ J
            q_mid   = (self.joint_limits[:, 0] + self.joint_limits[:, 1]) / 2.0
            q_range = self.joint_limits[:, 1] - self.joint_limits[:, 0]
            # Gradient of -(distance from midpoint)²  →  pushes joints to centre
            grad_H  = 2.0 * (q - q_mid) / (q_range / 2.0) ** 2
            dq_null = -self.null_gain * N @ grad_H

            q = self._clamp(q + dq_primary + dq_null)

        # Record final error after last update (when not converged early)
        if not converged:
            T_final = fk_fn(q)
            e_f = se3_error(T_final, T_desired)
            error_norms.append(float(np.linalg.norm(e_f)))
            pos_errors.append(float(np.linalg.norm(e_f[:3])))
            ori_errors.append(float(np.linalg.norm(e_f[3:])))

        n_iters = len(error_norms) - (0 if converged else 1)

        return IKResult(
            q_final=q,
            error_norms=error_norms,
            pos_errors=pos_errors,
            ori_errors=ori_errors,
            n_iters=n_iters,
            solve_time_s=time.perf_counter() - t_start,
            converged=converged,
            damping_used=damping_used,
        )
