"""
kinematics.py
=============
Analytical forward kinematics for the Franka Emika Panda (7-DOF) using
standard Denavit-Hartenberg (DH) convention, plus SE(3) spatial-error
utilities required by the Newton-Raphson IK solver.

DH convention used
------------------
Each joint-frame transformation is:
    T_i = Rot_z(theta_i) · Trans_z(d_i) · Trans_x(a_i) · Rot_x(alpha_i)

Panda DH parameters (SI units: metres, radians)
-----------------------------------------------
Reference: Franka Robotics technical documentation / Denavit-Hartenberg
           parameter set widely used in the community.

 i | a_i      | d_i    | alpha_i  | theta_offset
---|----------|--------|----------|-------------
 1 |  0.0000  | 0.3330 |  0.0000  | 0
 2 |  0.0000  | 0.0000 | -pi/2    | 0
 3 |  0.0000  | 0.3160 |  pi/2    | 0
 4 |  0.0825  | 0.0000 |  pi/2    | 0
 5 | -0.0825  | 0.3840 | -pi/2    | 0
 6 |  0.0000  | 0.0000 |  pi/2    | 0
 7 |  0.0880  | 0.1070 |  pi/2    | 0

A fixed end-effector (flange-to-EE) offset is appended after joint 7.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Array6 = NDArray[np.float64]   # (6,) spatial vector
SE3    = NDArray[np.float64]   # (4,4) homogeneous transform
SO3    = NDArray[np.float64]   # (3,3) rotation matrix

# ---------------------------------------------------------------------------
# Panda DH parameter table
# ---------------------------------------------------------------------------
# Each row: [a_i, d_i, alpha_i, theta_offset_i]
_PANDA_DH: NDArray[np.float64] = np.array(
    [
        [0.0000,  0.3330,  0.0000,       0.0],
        [0.0000,  0.0000, -np.pi / 2.0,  0.0],
        [0.0000,  0.3160,  np.pi / 2.0,  0.0],
        [0.0825,  0.0000,  np.pi / 2.0,  0.0],
        [-0.0825, 0.3840, -np.pi / 2.0,  0.0],
        [0.0000,  0.0000,  np.pi / 2.0,  0.0],
        [0.0880,  0.1070,  np.pi / 2.0,  0.0],
    ],
    dtype=np.float64,
)

# Fixed flange → EE offset (Panda default: 0 m in x/y, 0 m in z, identity R)
_T_FLANGE_EE: SE3 = np.eye(4, dtype=np.float64)


# ---------------------------------------------------------------------------
# Low-level SE(3) / SO(3) helpers
# ---------------------------------------------------------------------------

def _dh_transform(a: float, d: float, alpha: float, theta: float) -> SE3:
    """Compute a single standard-DH homogeneous transform.

    Parameters
    ----------
    a, d, alpha, theta : float
        DH parameters for one joint.

    Returns
    -------
    T : (4, 4) ndarray
        Homogeneous transform from frame {i-1} to frame {i}.
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array(
        [
            [ct,       -st,       0.0,  a   ],
            [st * ca,   ct * ca, -sa,  -sa * d],
            [st * sa,   ct * sa,  ca,   ca * d],
            [0.0,       0.0,      0.0,  1.0  ],
        ],
        dtype=np.float64,
    )


def skew(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the 3×3 skew-symmetric (cross-product) matrix of a 3-vector.

    Parameters
    ----------
    v : (3,) ndarray

    Returns
    -------
    S : (3, 3) ndarray  such that  S @ u == v × u
    """
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

    Parameters
    ----------
    R : (3, 3) ndarray
        Valid rotation matrix.

    Returns
    -------
    axis  : (3,) unit vector (zero vector if angle ≈ 0)
    angle : float  in [0, π]
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
    """Matrix logarithm of SO(3) → so(3) element as a 3-vector (axis × angle).

    Uses the standard Rodrigues inverse formula.

    Parameters
    ----------
    R : (3, 3) rotation matrix

    Returns
    -------
    omega : (3,) ndarray  — rotation vector (axis-angle representation)
    """
    axis, angle = rotation_to_axis_angle(R)
    return axis * angle


# ---------------------------------------------------------------------------
# Analytical Forward Kinematics
# ---------------------------------------------------------------------------

def forward_kinematics(
    q: NDArray[np.float64],
    dh: NDArray[np.float64] = _PANDA_DH,
    T_ee: SE3 = _T_FLANGE_EE,
) -> SE3:
    """Compute the full analytical forward kinematics T_0^EE ∈ SE(3).

    Parameters
    ----------
    q   : (7,) joint angle vector [rad].
    dh  : (n, 4) DH parameter table [a, d, alpha, theta_offset].
    T_ee: (4, 4) fixed flange-to-EE transform appended after the last joint.

    Returns
    -------
    T : (4, 4) homogeneous transform — pose of the EE in the base frame.
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
    """Return a list of cumulative transforms T_0^i for i = 0 … n.

    Useful for Jacobian calculation and visualisation.

    Parameters
    ----------
    q  : (n,) joint angle vector.
    dh : (n, 4) DH parameter table.

    Returns
    -------
    transforms : list of (4, 4) ndarrays, length n+1.
                 transforms[0] == identity, transforms[n] == T_0^EE.
    """
    transforms: list[SE3] = [np.eye(4, dtype=np.float64)]
    T = np.eye(4, dtype=np.float64)
    for i in range(dh.shape[0]):
        a_i, d_i, alpha_i, theta_off_i = dh[i]
        T = T @ _dh_transform(a_i, d_i, alpha_i, q[i] + theta_off_i)
        transforms.append(T.copy())
    return transforms


# ---------------------------------------------------------------------------
# Analytical Geometric Jacobian
# ---------------------------------------------------------------------------

def analytical_jacobian(
    q: NDArray[np.float64],
    dh: NDArray[np.float64] = _PANDA_DH,
    T_ee: SE3 = _T_FLANGE_EE,
) -> NDArray[np.float64]:
    """Compute the 6×n geometric Jacobian analytically from DH frames.

    The Jacobian maps joint velocities to end-effector spatial velocity:
        [v_EE; ω_EE] = J(q) · q̇

    For a revolute joint i:
        J_v_i = z_{i-1} × (p_EE - p_{i-1})
        J_ω_i = z_{i-1}

    Parameters
    ----------
    q   : (n,) joint angles [rad].
    dh  : (n, 4) DH parameter table.
    T_ee: (4, 4) flange-to-EE offset.

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
        z_i = frames[i][:3, 2]           # z-axis of frame {i-1} in base
        p_i = frames[i][:3, 3]           # origin of frame {i-1} in base
        J[:3, i] = np.cross(z_i, p_ee - p_i)  # linear component
        J[3:, i] = z_i                         # angular component

    return J


# ---------------------------------------------------------------------------
# SE(3) Spatial Error
# ---------------------------------------------------------------------------

def se3_error(
    T_current: SE3,
    T_desired: SE3,
) -> Array6:
    """Compute the 6-DOF spatial twist error e = [e_p; e_o] ∈ ℝ⁶.

    Position error (linear part):
        e_p = p_d - p_c

    Orientation error (angular part) via SO(3) logarithm:
        R_err = R_d · R_c^T
        e_o   = log_SO3(R_err)          (axis-angle vector)

    The combined error vector drives the Newton-Raphson update:
        Δq = J⁺ · e

    Parameters
    ----------
    T_current : (4, 4) current end-effector transform.
    T_desired : (4, 4) desired end-effector transform.

    Returns
    -------
    e : (6,) spatial error vector [e_p (m); e_o (rad)].
    """
    e_p = T_desired[:3, 3] - T_current[:3, 3]

    R_c = T_current[:3, :3]
    R_d = T_desired[:3, :3]
    R_err = R_d @ R_c.T
    e_o = so3_log(R_err)

    return np.concatenate([e_p, e_o])


def se3_error_norm(
    T_current: SE3,
    T_desired: SE3,
) -> tuple[float, float]:
    """Return (position_error_norm_m, orientation_error_norm_rad).

    Convenience wrapper used for logging / plotting.
    """
    e = se3_error(T_current, T_desired)
    return float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
