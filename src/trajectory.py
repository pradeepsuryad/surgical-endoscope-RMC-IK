"""
trajectory.py
=============
Circular trajectory generation for the surgical endoscope tracking task.

The end-effector traces a 3D circle of radius R at height Z_c (base frame).
At each waypoint the desired orientation R_d ∈ SO(3) is constructed so that
the tool's local X-axis points toward a fixed Remote-Centre-of-Motion (RCM)
target — here the world origin (0, 0, 0).

Waypoints are spaced approximately *step_mm* millimetres apart along the arc,
satisfying the 1 mm assignment requirement.

Mathematical formulation
------------------------
For a point p on the circle at angle θ:
    p(θ) = [cx + R·cos(θ),  cy + R·sin(θ),  z_height]

The desired X-axis (gaze direction) is:
    x̂_d = (target − p) / ‖target − p‖

A consistent right-handed frame is completed using the world Z hint:
    ẑ_d = (x̂_d × up) / ‖…‖
    ŷ_d = ẑ_d × x̂_d

R_d = [x̂_d | ŷ_d | ẑ_d]   (columns are local axes in the base frame)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

SE3 = NDArray[np.float64]  # (4, 4) homogeneous transform
SO3 = NDArray[np.float64]  # (3, 3) rotation matrix


# ---------------------------------------------------------------------------
# Orientation helper
# ---------------------------------------------------------------------------

def _look_at_rotation(
    position: NDArray[np.float64],
    target: NDArray[np.float64],
    up_hint: NDArray[np.float64] | None = None,
) -> SO3:
    """Build R whose local X-axis points from *position* toward *target*.

    Parameters
    ----------
    position : (3,) EE position [m].
    target   : (3,) RCM / tissue target [m].
    up_hint  : (3,) preferred "up" direction; defaults to world +Z.

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

    # Avoid degenerate cross product when gaze ≈ up_hint
    if np.abs(np.dot(x_axis, up_hint)) > 0.999:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    z_axis = np.cross(x_axis, up_hint)
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    return np.column_stack([x_axis, y_axis, z_axis])


# ---------------------------------------------------------------------------
# Trajectory class
# ---------------------------------------------------------------------------

class CircularTrajectory:
    """Discretised circular trajectory with gaze-aligned orientations.

    The circle lies in a horizontal plane at height *z_height*, centred at
    (*cx*, *cy*, *z_height*).  The RCM / endoscope target is fixed at
    *target_point* (default: origin).

    Consecutive waypoints are ≈ *step_mm* mm apart along the arc.

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

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def waypoints(self) -> list[SE3]:
        """List of (4, 4) SE(3) waypoints."""
        return self._waypoints

    @property
    def n_waypoints(self) -> int:
        """Number of discrete waypoints."""
        return len(self._waypoints)

    def get_waypoint(self, idx: int) -> SE3:
        """Return SE(3) waypoint at circular index *idx*."""
        return self._waypoints[idx % self.n_waypoints]

    def get_positions(self) -> NDArray[np.float64]:
        """Return (N, 3) array of waypoint positions [m]."""
        return np.array([T[:3, 3] for T in self._waypoints], dtype=np.float64)

    def get_orientations(self) -> NDArray[np.float64]:
        """Return (N, 3, 3) array of waypoint rotation matrices."""
        return np.array([T[:3, :3] for T in self._waypoints], dtype=np.float64)

    def interpolate(self, t: float) -> SE3:
        """Return the SE(3) waypoint at normalised arc position *t* ∈ [0, 1)."""
        idx = int(t * self.n_waypoints) % self.n_waypoints
        return self._waypoints[idx]
