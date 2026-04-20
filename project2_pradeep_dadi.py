# =============================================================================
# project2_pradeep_dadi.py
# ME5250 Project 2 — Surgical Endoscope Tracking via SE(3) Newton-Raphson IK
# 7-DOF Franka Emika Panda | MuJoCo Physics | Circular Trajectory
# Author: Dadi Pradyumna Reddy
# =============================================================================

from __future__ import annotations

# Standard library
import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

# Third-party — matplotlib backend must be set before pyplot is imported
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import seaborn as sns
from numpy.typing import NDArray


# =============================================================================
# PART 1: KINEMATICS
# Analytical FK for the Franka Panda (7-DOF) using standard DH convention,
# plus SE(3) spatial-error utilities for the Newton-Raphson IK solver.
# =============================================================================

# Type aliases
Array6 = NDArray[np.float64]   # (6,)  spatial vector
SE3    = NDArray[np.float64]   # (4,4) homogeneous transform
SO3    = NDArray[np.float64]   # (3,3) rotation matrix

# Panda DH parameter table  [a_i, d_i, alpha_i, theta_offset_i]
_PANDA_DH: NDArray[np.float64] = np.array(
    [
        [ 0.0000,  0.3330,  0.0000,      0.0],
        [ 0.0000,  0.0000, -np.pi / 2.0, 0.0],
        [ 0.0000,  0.3160,  np.pi / 2.0, 0.0],
        [ 0.0825,  0.0000,  np.pi / 2.0, 0.0],
        [-0.0825,  0.3840, -np.pi / 2.0, 0.0],
        [ 0.0000,  0.0000,  np.pi / 2.0, 0.0],
        [ 0.0880,  0.1070,  np.pi / 2.0, 0.0],
    ],
    dtype=np.float64,
)

# Fixed flange-to-EE offset (identity: DH frame 7 == EE frame)
_T_FLANGE_EE: SE3 = np.eye(4, dtype=np.float64)


def _dh_transform(a: float, d: float, alpha: float, theta: float) -> SE3:
    """Compute a single standard-DH homogeneous transform T_{i-1}^{i}."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct,      -st,      0.0,  a     ],
            [st * ca,  ct * ca, -sa,  -sa * d],
            [st * sa,  ct * sa,  ca,   ca * d],
            [0.0,      0.0,      0.0,  1.0   ],
        ],
        dtype=np.float64,
    )


def skew(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the 3x3 skew-symmetric matrix of a 3-vector."""
    return np.array(
        [
            [ 0.0,  -v[2],  v[1]],
            [ v[2],  0.0,  -v[0]],
            [-v[1],  v[0],  0.0 ],
        ],
        dtype=np.float64,
    )


def rotation_to_axis_angle(R: SO3) -> tuple[NDArray[np.float64], float]:
    """Extract axis and angle from a rotation matrix via Rodrigues formula.

    Returns
    -------
    axis  : (3,) unit vector (zero if angle ~ 0)
    angle : float in [0, pi]
    """
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if np.abs(angle) < 1e-9:
        return np.zeros(3, dtype=np.float64), 0.0
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    ) / (2.0 * np.sin(angle))
    norm = np.linalg.norm(axis)
    if norm > 1e-12:
        axis /= norm
    return axis, float(angle)


def so3_log(R: SO3) -> NDArray[np.float64]:
    """SO(3) matrix logarithm -> axis-angle 3-vector (Rodrigues inverse)."""
    axis, angle = rotation_to_axis_angle(R)
    return axis * angle


def forward_kinematics(
    q: NDArray[np.float64],
    dh: NDArray[np.float64] = _PANDA_DH,
    T_ee: SE3 = _T_FLANGE_EE,
) -> SE3:
    """Compute the full analytical forward kinematics T_0^EE in SE(3).

    Parameters
    ----------
    q   : (7,) joint angle vector [rad].
    dh  : (n, 4) DH parameter table [a, d, alpha, theta_offset].
    T_ee: (4, 4) fixed flange-to-EE transform.

    Returns
    -------
    T : (4, 4) homogeneous transform — EE pose in the base frame.
    """
    if q.shape[0] != dh.shape[0]:
        raise ValueError(
            f"Joint vector length {q.shape[0]} does not match DH rows {dh.shape[0]}."
        )
    T = np.eye(4, dtype=np.float64)
    for i in range(dh.shape[0]):
        a_i, d_i, alpha_i, theta_off_i = dh[i]
        T = T @ _dh_transform(a_i, d_i, alpha_i, q[i] + theta_off_i)
    return T @ T_ee


def fk_all_frames(
    q: NDArray[np.float64],
    dh: NDArray[np.float64] = _PANDA_DH,
) -> list[SE3]:
    """Return cumulative transforms T_0^i for i = 0 ... n.

    Returns
    -------
    transforms : list of (4, 4) ndarrays, length n+1.
    """
    transforms: list[SE3] = [np.eye(4, dtype=np.float64)]
    T = np.eye(4, dtype=np.float64)
    for i in range(dh.shape[0]):
        a_i, d_i, alpha_i, theta_off_i = dh[i]
        T = T @ _dh_transform(a_i, d_i, alpha_i, q[i] + theta_off_i)
        transforms.append(T.copy())
    return transforms


def analytical_jacobian(
    q: NDArray[np.float64],
    dh: NDArray[np.float64] = _PANDA_DH,
    T_ee: SE3 = _T_FLANGE_EE,
) -> NDArray[np.float64]:
    """Compute the 6xn geometric Jacobian analytically from DH frames.

    For revolute joint i:
        J_v_i = z_{i-1} x (p_EE - p_{i-1})
        J_w_i = z_{i-1}

    Returns
    -------
    J : (6, n) geometric Jacobian.
    """
    frames = fk_all_frames(q, dh)
    T_0_ee = frames[-1] @ T_ee
    p_ee = T_0_ee[:3, 3]
    n = dh.shape[0]
    J = np.zeros((6, n), dtype=np.float64)
    for i in range(n):
        z_i = frames[i][:3, 2]
        p_i = frames[i][:3, 3]
        J[:3, i] = np.cross(z_i, p_ee - p_i)
        J[3:, i] = z_i
    return J


def se3_error(T_current: SE3, T_desired: SE3) -> Array6:
    """Compute the 6-DOF spatial twist error e = [e_p; e_o] in R^6.

    Position error  : e_p = p_d - p_c
    Orientation error: e_o = log_SO3(R_d * R_c^T)   (axis-angle vector)

    Returns
    -------
    e : (6,) spatial error [e_p (m); e_o (rad)].
    """
    e_p = T_desired[:3, 3] - T_current[:3, 3]
    R_err = T_desired[:3, :3] @ T_current[:3, :3].T
    e_o = so3_log(R_err)
    return np.concatenate([e_p, e_o])


def se3_error_norm(T_current: SE3, T_desired: SE3) -> tuple[float, float]:
    """Return (position_error_m, orientation_error_rad) scalar norms."""
    e = se3_error(T_current, T_desired)
    return float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))


# =============================================================================
# PART 2: TRAJECTORY
# Circular trajectory generation with gaze-aligned (RCM) orientations.
# =============================================================================

def _look_at_rotation(
    position: NDArray[np.float64],
    target: NDArray[np.float64],
    up_hint: NDArray[np.float64] | None = None,
) -> SO3:
    """Build R whose local X-axis points from position toward target.

    Parameters
    ----------
    position : (3,) EE position [m].
    target   : (3,) RCM / tissue target [m].
    up_hint  : (3,) preferred up direction; defaults to world +Z.

    Returns
    -------
    R : (3, 3) rotation matrix.
    """
    if up_hint is None:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = target - position
    norm = np.linalg.norm(x_axis)
    if norm < 1e-12:
        raise ValueError("position and target are coincident.")
    x_axis = x_axis / norm
    if np.abs(np.dot(x_axis, up_hint)) > 0.999:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    z_axis = np.cross(x_axis, up_hint)
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


class CircularTrajectory:
    """Discretised circular trajectory with gaze-aligned (RCM) orientations.

    The circle lies in a horizontal plane at height z_height, centred at
    (cx, cy, z_height). The RCM target is fixed at target_point (default:
    origin). Consecutive waypoints are approximately step_mm mm apart.

    Parameters
    ----------
    radius       : float   Circle radius [m].
    z_height     : float   Height of the circle plane [m].
    cx, cy       : float   Circle centre (x, y) [m].
    step_mm      : float   Arc-length spacing between waypoints [mm].
    target_point : (3,)    Fixed RCM / tissue-target [m]; default = origin.
    """

    def __init__(
        self,
        radius: float = 0.15,
        z_height: float = 0.50,
        cx: float = 0.0,
        cy: float = 0.0,
        step_mm: float = 1.0,
        target_point: NDArray[np.float64] | None = None,
    ) -> None:
        self.radius = float(radius)
        self.z_height = float(z_height)
        self.cx = float(cx)
        self.cy = float(cy)
        self.step_mm = float(step_mm)
        self.target_point = (
            np.zeros(3, dtype=np.float64)
            if target_point is None
            else np.asarray(target_point, dtype=np.float64)
        )
        self._waypoints: list[SE3] = self._generate()

    def _generate(self) -> list[SE3]:
        """Pre-compute all SE(3) waypoints."""
        circumference = 2.0 * np.pi * self.radius
        step_m = self.step_mm * 1e-3
        n_pts = max(3, int(np.ceil(circumference / step_m)))
        thetas = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
        waypoints: list[SE3] = []
        for theta in thetas:
            pos = np.array(
                [
                    self.cx + self.radius * np.cos(theta),
                    self.cy + self.radius * np.sin(theta),
                    self.z_height,
                ],
                dtype=np.float64,
            )
            R = _look_at_rotation(pos, self.target_point)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = pos
            waypoints.append(T)
        return waypoints

    @property
    def waypoints(self) -> list[SE3]:
        """List of (4, 4) SE(3) waypoints."""
        return self._waypoints

    @property
    def n_waypoints(self) -> int:
        """Number of discrete waypoints."""
        return len(self._waypoints)

    def get_waypoint(self, idx: int) -> SE3:
        """Return SE(3) waypoint at circular index idx."""
        return self._waypoints[idx % self.n_waypoints]

    def get_positions(self) -> NDArray[np.float64]:
        """Return (N, 3) array of waypoint positions [m]."""
        return np.array([T[:3, 3] for T in self._waypoints], dtype=np.float64)

    def get_orientations(self) -> NDArray[np.float64]:
        """Return (N, 3, 3) array of waypoint rotation matrices."""
        return np.array([T[:3, :3] for T in self._waypoints], dtype=np.float64)

    def interpolate(self, t: float) -> SE3:
        """Return the SE(3) waypoint at normalised arc position t in [0, 1)."""
        idx = int(t * self.n_waypoints) % self.n_waypoints
        return self._waypoints[idx]


# =============================================================================
# PART 3: IK SOLVER
# SE(3) Newton-Raphson IK with damped-least-squares (Levenberg-Marquardt)
# singularity fallback and null-space joint-limit avoidance.
# =============================================================================

@dataclass
class IKResult:
    """Structured result returned by a single IK solve call."""

    q_final: NDArray[np.float64]       # (7,) final joint angles [rad]
    error_norms: list[float]           # ||e_k|| at each NR iteration
    pos_errors: list[float]            # ||e_p|| at each iteration [m]
    ori_errors: list[float]            # ||e_o|| at each iteration [rad]
    n_iters: int                       # iterations performed
    solve_time_s: float                # wall-clock solve time [s]
    converged: bool                    # True if ||e|| < tol before max_iter
    damping_used: list[float] = field(default_factory=list)  # lambda per iter


class NewtonRaphsonIK:
    """SE(3) Newton-Raphson IK with damped-least-squares singularity fallback.

    Update law:
        q_{k+1} = q_k + J^+ * e_k

    where J^+ = J^T (J J^T + lambda^2 I)^{-1} is the damped Moore-Penrose
    pseudoinverse (Levenberg-Marquardt).  A null-space secondary task pushes
    joints toward their midrange to avoid limits.

    Parameters
    ----------
    max_iter     : int    Max NR iterations per solve call (default 5).
    tol          : float  Convergence threshold on ||e|| (default 1e-4).
    lam_min      : float  DLS damping near regular configs (~Moore-Penrose).
    lam_max      : float  DLS damping near singularities.
    sigma_thresh : float  Min singular value below which DLS activates.
    null_gain    : float  Gain for null-space joint-limit avoidance.
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
        self.null_gain    = null_gain
        self.joint_limits = (
            self._PANDA_LIMITS if joint_limits is None
            else np.asarray(joint_limits, dtype=np.float64)
        )

    def _damped_pinv(
        self,
        J: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float]:
        """Compute J^+ = J^T (J J^T + lambda^2 I)^{-1} with adaptive LM damping.

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

    def _clamp(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

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
        jacobian_fn : callable  q -> (6, 7) geometric Jacobian.
        fk_fn       : callable  q -> (4, 4) SE(3) forward kinematics.

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
            e = se3_error(T_cur, T_desired)

            e_norm = float(np.linalg.norm(e))
            error_norms.append(e_norm)
            pos_errors.append(float(np.linalg.norm(e[:3])))
            ori_errors.append(float(np.linalg.norm(e[3:])))

            if e_norm < self.tol:
                converged = True
                break

            J = jacobian_fn(q)
            J_pinv, lam = self._damped_pinv(J)
            damping_used.append(lam)

            # Primary task: SE(3) error minimisation
            dq_primary = J_pinv @ e

            # Secondary task: null-space joint-limit avoidance
            n_joints = J.shape[1]
            N = np.eye(n_joints, dtype=np.float64) - J_pinv @ J
            q_mid   = (self.joint_limits[:, 0] + self.joint_limits[:, 1]) / 2.0
            q_range = self.joint_limits[:, 1] - self.joint_limits[:, 0]
            grad_H  = 2.0 * (q - q_mid) / (q_range / 2.0) ** 2
            dq_null = -self.null_gain * N @ grad_H

            q = self._clamp(q + dq_primary + dq_null)

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


# =============================================================================
# PART 4: SIMULATION
# MuJoCo-based environment: IK loop, physics stepping, and data logging.
# =============================================================================

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "franka_emika_panda"
_MODEL_XML = _MODEL_DIR / "panda.xml"

_EE_SITE_NAME = "attachment_site"

# Safe Panda home configuration (Franka "ready" pose)
_Q_HOME = np.array(
    [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398],
    dtype=np.float64,
)


@dataclass
class StepLog:
    """All metrics recorded at a single simulation timestep."""

    timestep: int
    sim_time: float
    wall_time: float
    q: NDArray[np.float64]
    ee_pos_mujoco: NDArray[np.float64]
    ee_rot_mujoco: NDArray[np.float64]
    ee_pos_analytical: NDArray[np.float64]
    ee_rot_analytical: NDArray[np.float64]
    desired_pos: NDArray[np.float64]
    desired_rot: NDArray[np.float64]
    pos_error_m: float
    ori_error_rad: float
    ik_solve_time_s: float
    ik_converged: bool
    ik_n_iters: int
    ik_error_norms: list[float]


@dataclass
class SimulationLog:
    """Collected log for an entire simulation run."""

    steps: list[StepLog] = field(default_factory=list)

    def append(self, log: StepLog) -> None:
        self.steps.append(log)

    @property
    def q_history(self) -> NDArray[np.float64]:
        return np.array([s.q for s in self.steps], dtype=np.float64)

    @property
    def ee_pos_mujoco(self) -> NDArray[np.float64]:
        return np.array([s.ee_pos_mujoco for s in self.steps], dtype=np.float64)

    @property
    def ee_pos_analytical(self) -> NDArray[np.float64]:
        return np.array([s.ee_pos_analytical for s in self.steps], dtype=np.float64)

    @property
    def desired_pos(self) -> NDArray[np.float64]:
        return np.array([s.desired_pos for s in self.steps], dtype=np.float64)

    @property
    def pos_errors(self) -> NDArray[np.float64]:
        return np.array([s.pos_error_m for s in self.steps], dtype=np.float64)

    @property
    def ori_errors(self) -> NDArray[np.float64]:
        return np.array([s.ori_error_rad for s in self.steps], dtype=np.float64)

    @property
    def sim_times(self) -> NDArray[np.float64]:
        return np.array([s.sim_time for s in self.steps], dtype=np.float64)

    @property
    def ik_solve_times(self) -> NDArray[np.float64]:
        return np.array([s.ik_solve_time_s for s in self.steps], dtype=np.float64)


def _make_mujoco_fk(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
):
    """Return a callable q -> (4, 4) SE(3) using MuJoCo kinematics."""
    def fk_fn(q: NDArray[np.float64]) -> NDArray[np.float64]:
        data.qpos[:7] = q
        mujoco.mj_kinematics(model, data)
        T = np.eye(4, dtype=np.float64)
        T[:3, 3]  = data.site_xpos[site_id].copy()
        T[:3, :3] = data.site_xmat[site_id].reshape(3, 3).copy()
        return T
    return fk_fn


def _make_mujoco_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
):
    """Return a callable q -> (6, 7) Jacobian using mj_jacSite."""
    def jacobian_fn(q: NDArray[np.float64]) -> NDArray[np.float64]:
        data.qpos[:7] = q
        mujoco.mj_kinematics(model, data)
        Jv = np.zeros((3, model.nv), dtype=np.float64)
        Jw = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, Jv, Jw, site_id)
        return np.vstack([Jv[:, :7], Jw[:, :7]])
    return jacobian_fn


def _resolve_site_id(model: mujoco.MjModel, name: str) -> int:
    """Return site id by name; fall back to last site if not found."""
    try:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            return sid
    except Exception:
        pass
    return model.nsite - 1


class EndoscopeSimulation:
    """Run the MuJoCo surgical endoscope tracking simulation.

    Parameters
    ----------
    model_xml_path : Path or str, optional
        Path to the Panda MuJoCo XML.
    trajectory     : CircularTrajectory, optional
    ik_solver      : NewtonRaphsonIK, optional
    n_laps         : int    Number of full laps around the circle.
    render         : bool   Whether to open a live MuJoCo viewer window.
    ee_site_name   : str    Name of the end-effector site in the XML.
    record_video   : bool   Capture offscreen frames to an MP4 file.
    video_path     : Path or str, optional  Output MP4 path.
    """

    def __init__(
        self,
        model_xml_path: Optional[Path | str] = None,
        trajectory: Optional[CircularTrajectory] = None,
        ik_solver: Optional[NewtonRaphsonIK] = None,
        n_laps: int = 1,
        render: bool = True,
        ee_site_name: str = _EE_SITE_NAME,
        record_video: bool = False,
        video_path: Optional[Path | str] = None,
    ) -> None:
        xml_path = Path(model_xml_path) if model_xml_path else _MODEL_XML
        if not xml_path.exists():
            raise FileNotFoundError(
                f"MuJoCo XML not found: {xml_path}\n"
                "Download the MuJoCo Menagerie Panda model and place it at "
                "models/franka_emika_panda/panda.xml"
            )

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data  = mujoco.MjData(self.model)

        self.site_id       = _resolve_site_id(self.model, ee_site_name)
        self.trajectory    = trajectory or CircularTrajectory()
        self.ik_solver     = ik_solver  or NewtonRaphsonIK()
        self.n_laps        = n_laps
        self.render        = render
        self.record_video  = record_video
        self.video_path    = Path(video_path) if video_path else Path("results/simulation.mp4")
        self.log           = SimulationLog()

        self._jac_fn   = _make_mujoco_jacobian(self.model, self.data, self.site_id)
        self._mj_fk_fn = _make_mujoco_fk(self.model, self.data, self.site_id)

    def _set_joints(self, q: NDArray[np.float64]) -> None:
        self.data.qpos[:7] = q
        self.data.ctrl[:7] = q
        mujoco.mj_forward(self.model, self.data)

    def _mujoco_ee_pose(self) -> NDArray[np.float64]:
        T = np.eye(4, dtype=np.float64)
        T[:3, 3]  = self.data.site_xpos[self.site_id].copy()
        T[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return T

    def run(self) -> SimulationLog:
        """Execute the full simulation loop and return the log."""
        traj = self.trajectory
        total_steps = traj.n_waypoints * self.n_laps

        q_current = _Q_HOME.copy()
        self._set_joints(q_current)

        viewer   = None
        renderer = None
        frames: list = []

        if self.render:
            try:
                viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception as exc:
                print(f"[simulation] Viewer unavailable ({exc}); running headless.")
                self.render = False

        if self.record_video:
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            _cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(_cam)
            _cam.lookat[0] = 0.0
            _cam.lookat[1] = 0.0
            _cam.lookat[2] = 0.4
            _cam.distance  = 1.5
            _cam.azimuth   = 135.0
            _cam.elevation = -20.0

        try:
            for step in range(total_steps):
                t_wall_start = time.perf_counter()

                T_desired = traj.get_waypoint(step % traj.n_waypoints)

                ik_result: IKResult = self.ik_solver.solve(
                    q0=q_current,
                    T_desired=T_desired,
                    jacobian_fn=self._jac_fn,
                    fk_fn=self._mj_fk_fn,
                )
                q_current = ik_result.q_final

                self._set_joints(q_current)
                mujoco.mj_step(self.model, self.data)

                T_mujoco     = self._mujoco_ee_pose()
                T_analytical = forward_kinematics(q_current)

                pos_err, ori_err = se3_error_norm(T_mujoco, T_desired)

                self.log.append(
                    StepLog(
                        timestep=step,
                        sim_time=float(self.data.time),
                        wall_time=time.perf_counter() - t_wall_start,
                        q=q_current.copy(),
                        ee_pos_mujoco=T_mujoco[:3, 3].copy(),
                        ee_rot_mujoco=T_mujoco[:3, :3].copy(),
                        ee_pos_analytical=T_analytical[:3, 3].copy(),
                        ee_rot_analytical=T_analytical[:3, :3].copy(),
                        desired_pos=T_desired[:3, 3].copy(),
                        desired_rot=T_desired[:3, :3].copy(),
                        pos_error_m=pos_err,
                        ori_error_rad=ori_err,
                        ik_solve_time_s=ik_result.solve_time_s,
                        ik_converged=ik_result.converged,
                        ik_n_iters=ik_result.n_iters,
                        ik_error_norms=ik_result.error_norms,
                    )
                )

                if self.render and viewer is not None and viewer.is_running():
                    viewer.sync()

                if renderer is not None:
                    renderer.update_scene(self.data, camera=_cam)
                    frames.append(renderer.render().copy())

        finally:
            if viewer is not None:
                viewer.close()
            if renderer is not None:
                renderer.close()

        if frames:
            import imageio
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(str(self.video_path), frames, fps=30)
            print(f"[simulation] Video saved -> {self.video_path}")

        return self.log


# =============================================================================
# PART 5: VISUALIZER
# Matplotlib / Seaborn post-processing plots saved to results/ as PNGs.
# =============================================================================

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
_FIG_DPI = 150

_COLORS     = sns.color_palette("muted")
_C_DESIRED  = _COLORS[0]
_C_ACTUAL   = _COLORS[1]
_C_ANALYTIC = _COLORS[2]
_C_MUJOCO   = _COLORS[3]


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(str(path), dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path.name}")


def plot_3d_trajectory(log: SimulationLog, output_dir: Path) -> None:
    """3D line plot: desired circle vs actual executed EE path."""
    desired  = log.desired_pos
    actual   = log.ee_pos_mujoco
    analytic = log.ee_pos_analytical

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(desired[:, 0], desired[:, 1], desired[:, 2],
            "--", color=_C_DESIRED, lw=2.0, label="Desired trajectory", alpha=0.7)
    ax.plot(actual[:, 0], actual[:, 1], actual[:, 2],
            "-", color=_C_ACTUAL, lw=1.5, label="Actual (MuJoCo FK)")
    ax.plot(analytic[:, 0], analytic[:, 1], analytic[:, 2],
            ":", color=_C_ANALYTIC, lw=1.5, label="Analytical FK", alpha=0.8)

    ax.scatter(*desired[0], color="green", s=60, zorder=5, label="Start")
    ax.scatter(*desired[-1], color="red",  s=60, zorder=5, label="End")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("3D End-Effector Trajectory\n(Desired vs Actual)", pad=12)
    ax.legend(loc="upper left", fontsize=9)

    _save(fig, output_dir / "01_3d_trajectory.png")


def plot_error_over_time(log: SimulationLog, output_dir: Path) -> None:
    """Two-panel plot: position error (mm) and orientation error (rad)."""
    t       = log.sim_times
    pos_err = log.pos_errors * 1e3
    ori_err = log.ori_errors

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(t, pos_err, color=_C_ACTUAL, lw=1.4)
    axes[0].fill_between(t, 0, pos_err, alpha=0.15, color=_C_ACTUAL)
    axes[0].set_ylabel("Position error ||e_p|| [mm]")
    axes[0].set_title("SE(3) Tracking Error over Time")
    axes[0].axhline(1.0, ls="--", color="gray", lw=0.9, label="1 mm threshold")
    axes[0].legend(fontsize=9)

    axes[1].plot(t, ori_err, color=_C_ANALYTIC, lw=1.4)
    axes[1].fill_between(t, 0, ori_err, alpha=0.15, color=_C_ANALYTIC)
    axes[1].set_ylabel("Orientation error ||e_o|| [rad]")
    axes[1].set_xlabel("Simulation time [s]")

    fig.tight_layout()
    _save(fig, output_dir / "02_error_over_time.png")


def plot_fk_comparison(log: SimulationLog, output_dir: Path) -> None:
    """Compare analytical FK vs MuJoCo FK for x, y, z independently."""
    t        = log.sim_times
    analytic = log.ee_pos_analytical
    mj       = log.ee_pos_mujoco
    labels   = ["X [m]", "Y [m]", "Z [m]"]

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    for i, (ax, lab) in enumerate(zip(axes[:3], labels)):
        ax.plot(t, mj[:, i],       color=_C_MUJOCO,   lw=1.5, label="MuJoCo FK",    alpha=0.9)
        ax.plot(t, analytic[:, i], color=_C_ANALYTIC, lw=1.2, ls="--",
                label="Analytical FK", alpha=0.9)
        ax.set_ylabel(lab)
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_title("Analytical FK vs MuJoCo FK - EE Position Comparison")

    diff_mm = np.linalg.norm(analytic - mj, axis=1) * 1e3
    axes[3].plot(t, diff_mm, color="tomato", lw=1.4)
    axes[3].fill_between(t, 0, diff_mm, alpha=0.15, color="tomato")
    axes[3].set_ylabel("||FK discrepancy|| [mm]")
    axes[3].set_xlabel("Simulation time [s]")
    axes[3].set_title("FK Discrepancy Norm (Analytical - MuJoCo)")

    fig.tight_layout()
    _save(fig, output_dir / "03_fk_comparison.png")


def plot_joint_angles(log: SimulationLog, output_dir: Path) -> None:
    """Plot all 7 joint angles over time on a shared time axis."""
    t = log.sim_times
    Q = log.q_history

    fig, axes = plt.subplots(7, 1, figsize=(13, 14), sharex=True)
    palette = sns.color_palette("tab10", 7)

    for j, (ax, color) in enumerate(zip(axes, palette)):
        ax.plot(t, np.degrees(Q[:, j]), color=color, lw=1.4)
        ax.set_ylabel(f"q{j + 1} [deg]", fontsize=9)
        ax.grid(True, alpha=0.4)

    axes[0].set_title("Joint Angles over Time (smooth motion verification)")
    axes[-1].set_xlabel("Simulation time [s]")

    fig.tight_layout()
    _save(fig, output_dir / "04_joint_angles.png")


def plot_ik_convergence(log: SimulationLog, output_dir: Path) -> None:
    """Mean +/- sigma of NR residual ||e_k|| at each iteration."""
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

    padded = np.array(
        [x + [x[-1]] * (overall_max - len(x)) for x in all_norms],
        dtype=np.float64,
    )

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
        label="+/- 1 sigma",
    )

    ax.set_xlabel("Newton-Raphson iteration k")
    ax.set_ylabel("Spatial error norm ||e_k||")
    ax.set_title(f"IK Convergence per Iteration\n"
                 f"(max_iter = {overall_max - 1}, {step_label})")
    ax.legend()

    if len(mean_e) >= 2 and mean_e[0] > 1e-10:
        reduction_pct = (1.0 - mean_e[-1] / mean_e[0]) * 100.0
        ax.text(0.62, 0.85,
                f"Average reduction: {reduction_pct:.1f}%",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    fig.tight_layout()
    _save(fig, output_dir / "05_ik_convergence.png")


def plot_computation_time(log: SimulationLog, output_dir: Path) -> None:
    """Per-step IK solve time with moving-average overlay."""
    t        = log.sim_times
    solve_ms = log.ik_solve_times * 1e3

    window = max(1, len(solve_ms) // 30)
    ma     = np.convolve(solve_ms, np.ones(window) / window, mode="valid")
    t_ma   = t[:len(ma)]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(t, solve_ms, color=_C_DESIRED, lw=0.8, alpha=0.5, label="Per-step")
    ax.plot(t_ma, ma, color=_C_ACTUAL, lw=2.0, label=f"Moving avg (w={window})")
    ax.axhline(solve_ms.mean(), ls="--", color="gray", lw=1.0,
               label=f"Mean = {solve_ms.mean():.2f} ms")

    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("IK solve time [ms]")
    ax.set_title("Newton-Raphson IK Computation Time per Step")
    ax.legend()

    fig.tight_layout()
    _save(fig, output_dir / "06_computation_time.png")


def plot_summary_dashboard(log: SimulationLog, output_dir: Path) -> None:
    """Single-page summary of key metrics."""
    t       = log.sim_times
    pos_mm  = log.pos_errors * 1e3
    ori_rad = log.ori_errors
    Q       = log.q_history
    ms      = log.ik_solve_times * 1e3

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    fig.suptitle(
        "Surgical Endoscope Tracking - Simulation Summary\n"
        "7-DOF Franka Panda | SE(3) Newton-Raphson IK | Circular Trajectory",
        fontsize=13, fontweight="bold",
    )

    gs = gridspec.GridSpec(3, 3, figure=fig)

    ax3d = fig.add_subplot(gs[:2, :2], projection="3d")
    desired = log.desired_pos
    actual  = log.ee_pos_mujoco
    ax3d.plot(desired[:, 0], desired[:, 1], desired[:, 2],
              "--", color=_C_DESIRED, lw=2, label="Desired", alpha=0.7)
    ax3d.plot(actual[:, 0], actual[:, 1], actual[:, 2],
              "-", color=_C_ACTUAL, lw=1.5, label="Actual")
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title("3D Trajectory")
    ax3d.legend(fontsize=8)

    ax_pe = fig.add_subplot(gs[0, 2])
    ax_pe.plot(t, pos_mm, color=_C_ACTUAL, lw=1.3)
    ax_pe.set_ylabel("||e_p|| [mm]")
    ax_pe.set_title("Position Error")
    ax_pe.axhline(1.0, ls="--", color="gray", lw=0.9)

    ax_oe = fig.add_subplot(gs[1, 2], sharex=ax_pe)
    ax_oe.plot(t, ori_rad, color=_C_ANALYTIC, lw=1.3)
    ax_oe.set_ylabel("||e_o|| [rad]")
    ax_oe.set_title("Orientation Error")
    ax_oe.set_xlabel("Time [s]")

    ax_q = fig.add_subplot(gs[2, :2])
    palette = sns.color_palette("tab10", 7)
    for j, color in enumerate(palette):
        ax_q.plot(t, np.degrees(Q[:, j]), color=color, lw=0.9, label=f"q{j + 1}")
    ax_q.set_ylabel("Joint angle [deg]")
    ax_q.set_xlabel("Time [s]")
    ax_q.set_title("Joint Angles")
    ax_q.legend(ncol=7, fontsize=7, loc="upper right")

    ax_ms = fig.add_subplot(gs[2, 2])
    ax_ms.plot(t, ms, color=_C_DESIRED, lw=0.8, alpha=0.6)
    ax_ms.axhline(ms.mean(), ls="--", color="gray", lw=1.0,
                  label=f"mean={ms.mean():.2f} ms")
    ax_ms.set_ylabel("IK time [ms]")
    ax_ms.set_xlabel("Time [s]")
    ax_ms.set_title("IK Solve Time")
    ax_ms.legend(fontsize=8)

    _save(fig, output_dir / "00_summary_dashboard.png")


def plot_all(log: SimulationLog, output_dir: Path) -> None:
    """Generate all 7 analysis plots and save to output_dir."""
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
    print(f"[Visualizer] 7 plots saved to {output_dir}/")


# =============================================================================
# ENTRY POINT
# =============================================================================

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


def main() -> None:
    args = _parse_args()

    # 1. Trajectory
    traj = CircularTrajectory(
        radius=args.radius,
        z_height=args.height,
        step_mm=args.step_mm,
    )
    print(
        f"[Trajectory]  {traj.n_waypoints} waypoints  "
        f"(R={args.radius:.3f} m, Z={args.height:.3f} m, ds={args.step_mm:.1f} mm)"
    )

    # 2. IK solver
    ik = NewtonRaphsonIK(max_iter=args.max_iter)
    print(f"[IK Solver]   Newton-Raphson  max_iter={args.max_iter}  (+ DLS fallback)")

    # 3. Simulation
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
        f"              Mean pos error : {log.pos_errors.mean() * 1e3:.3f} mm\n"
        f"              Mean ori error : {log.ori_errors.mean():.5f} rad"
    )

    # 4. Plots
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    plot_all(log, output_dir=results_dir)
    print(f"\n[Done]  All plots saved to: {results_dir}/")


if __name__ == "__main__":
    main()
