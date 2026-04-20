"""
simulation.py
=============
MuJoCo-based simulation environment for the surgical endoscope tracking task.

Simulation loop (per timestep)
-------------------------------
1. Retrieve the desired SE(3) waypoint from the circular trajectory.
2. Obtain the geometric Jacobian via MuJoCo's mj_jacSite.
3. Obtain the current EE pose via MuJoCo site kinematics.
4. Run the Newton-Raphson IK solver (N fixed iterations).
5. Apply updated joint angles via position control (data.ctrl).
6. Step the physics engine (mj_step).
7. Log: joint angles, MuJoCo EE pose, analytical EE pose, errors, timings.

FK Verification
---------------
At every step the analytical FK (DH-based, kinematics.py) is computed
independently of MuJoCo and stored alongside the MuJoCo FK for later
comparison in the visualiser.

MuJoCo model
------------
Place the MuJoCo Menagerie Franka Panda XML tree at:
    models/franka_emika_panda/panda.xml
(or pass a custom path to EndoscopeSimulation).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import NDArray

from .ik_solver import IKResult, NewtonRaphsonIK
from .kinematics import forward_kinematics, se3_error_norm
from .trajectory import CircularTrajectory

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "franka_emika_panda"
_MODEL_XML = _MODEL_DIR / "panda.xml"

# EE site name in the Menagerie Panda XML (hand model).
# Fallback: last site in model if this name is not found.
_EE_SITE_NAME = "attachment_site"

# Safe Panda home configuration (Franka "ready" pose)
_Q_HOME = np.array(
    [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Per-step data container
# ---------------------------------------------------------------------------


@dataclass
class StepLog:
    """All metrics recorded at a single simulation timestep."""

    timestep: int
    sim_time: float                        # MuJoCo sim time [s]
    wall_time: float                       # wall-clock duration of this step [s]
    q: NDArray[np.float64]                 # (7,) joint angles [rad]
    ee_pos_mujoco: NDArray[np.float64]     # (3,) EE position – MuJoCo FK [m]
    ee_rot_mujoco: NDArray[np.float64]     # (3, 3) EE rotation – MuJoCo FK
    ee_pos_analytical: NDArray[np.float64] # (3,) EE position – analytical FK [m]
    ee_rot_analytical: NDArray[np.float64] # (3, 3) EE rotation – analytical FK
    desired_pos: NDArray[np.float64]       # (3,) desired EE position [m]
    desired_rot: NDArray[np.float64]       # (3, 3) desired EE rotation
    pos_error_m: float                     # ‖e_p‖ [m]   (MuJoCo vs desired)
    ori_error_rad: float                   # ‖e_o‖ [rad] (MuJoCo vs desired)
    ik_solve_time_s: float                 # IK solver wall-clock time [s]
    ik_converged: bool
    ik_n_iters: int
    ik_error_norms: list[float]            # per-NR-iteration error norms


# ---------------------------------------------------------------------------
# Aggregated log
# ---------------------------------------------------------------------------


@dataclass
class SimulationLog:
    """Collected log for an entire simulation run."""

    steps: list[StepLog] = field(default_factory=list)

    def append(self, log: StepLog) -> None:
        self.steps.append(log)

    # ------------------------------------------------------------------
    # Convenience array accessors
    # ------------------------------------------------------------------

    @property
    def q_history(self) -> NDArray[np.float64]:
        """(T, 7) joint angle history [rad]."""
        return np.array([s.q for s in self.steps], dtype=np.float64)

    @property
    def ee_pos_mujoco(self) -> NDArray[np.float64]:
        """(T, 3) MuJoCo EE positions [m]."""
        return np.array([s.ee_pos_mujoco for s in self.steps], dtype=np.float64)

    @property
    def ee_pos_analytical(self) -> NDArray[np.float64]:
        """(T, 3) Analytical FK EE positions [m]."""
        return np.array([s.ee_pos_analytical for s in self.steps], dtype=np.float64)

    @property
    def desired_pos(self) -> NDArray[np.float64]:
        """(T, 3) desired EE positions [m]."""
        return np.array([s.desired_pos for s in self.steps], dtype=np.float64)

    @property
    def pos_errors(self) -> NDArray[np.float64]:
        """(T,) position error norms [m]."""
        return np.array([s.pos_error_m for s in self.steps], dtype=np.float64)

    @property
    def ori_errors(self) -> NDArray[np.float64]:
        """(T,) orientation error norms [rad]."""
        return np.array([s.ori_error_rad for s in self.steps], dtype=np.float64)

    @property
    def sim_times(self) -> NDArray[np.float64]:
        """(T,) simulation time stamps [s]."""
        return np.array([s.sim_time for s in self.steps], dtype=np.float64)

    @property
    def ik_solve_times(self) -> NDArray[np.float64]:
        """(T,) IK solve wall-clock times [s]."""
        return np.array([s.ik_solve_time_s for s in self.steps], dtype=np.float64)


# ---------------------------------------------------------------------------
# MuJoCo helper functions
# ---------------------------------------------------------------------------


def _make_mujoco_fk(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
) -> ...:
    """Return a callable q → (4, 4) SE(3) that uses MuJoCo kinematics."""

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
) -> ...:
    """Return a callable q → (6, 7) Jacobian using mj_jacSite."""

    def jacobian_fn(q: NDArray[np.float64]) -> NDArray[np.float64]:
        data.qpos[:7] = q
        mujoco.mj_kinematics(model, data)
        Jv = np.zeros((3, model.nv), dtype=np.float64)
        Jw = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, Jv, Jw, site_id)
        return np.vstack([Jv[:, :7], Jw[:, :7]])   # (6, 7)

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


# ---------------------------------------------------------------------------
# Simulation class
# ---------------------------------------------------------------------------


class EndoscopeSimulation:
    """Run the MuJoCo surgical endoscope tracking simulation.

    Parameters
    ----------
    model_xml_path : Path or str, optional
        Path to the Panda MuJoCo XML.  Defaults to models/franka_emika_panda/panda.xml.
    trajectory     : CircularTrajectory, optional
        Pre-built trajectory.  A default 15 cm / 50 cm circle is used.
    ik_solver      : NewtonRaphsonIK, optional
        IK solver instance.
    n_laps         : int
        Number of full laps around the circle.
    render         : bool
        Whether to open a live MuJoCo viewer window.
    ee_site_name   : str
        Name of the end-effector site in the XML.
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

        self.site_id    = _resolve_site_id(self.model, ee_site_name)
        self.trajectory = trajectory or CircularTrajectory()
        self.ik_solver  = ik_solver  or NewtonRaphsonIK()
        self.n_laps        = n_laps
        self.render        = render
        self.record_video  = record_video
        self.video_path    = Path(video_path) if video_path else Path("results/simulation.mp4")
        self.log           = SimulationLog()

        self._jac_fn    = _make_mujoco_jacobian(self.model, self.data, self.site_id)
        self._mj_fk_fn  = _make_mujoco_fk(self.model, self.data, self.site_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_joints(self, q: NDArray[np.float64]) -> None:
        """Write joint angles to qpos and position-control actuators."""
        self.data.qpos[:7] = q
        self.data.ctrl[:7] = q
        mujoco.mj_forward(self.model, self.data)

    def _mujoco_ee_pose(self) -> NDArray[np.float64]:
        T = np.eye(4, dtype=np.float64)
        T[:3, 3]  = self.data.site_xpos[self.site_id].copy()
        T[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return T

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> SimulationLog:
        """Execute the full simulation.

        Returns
        -------
        SimulationLog
        """
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

                # --- Desired SE(3) waypoint ---
                T_desired = traj.get_waypoint(step % traj.n_waypoints)

                # --- IK solve (MuJoCo Jacobian + MuJoCo FK) ---
                ik_result: IKResult = self.ik_solver.solve(
                    q0=q_current,
                    T_desired=T_desired,
                    jacobian_fn=self._jac_fn,
                    fk_fn=self._mj_fk_fn,
                )
                q_current = ik_result.q_final

                # --- Apply to simulation & advance physics ---
                self._set_joints(q_current)
                mujoco.mj_step(self.model, self.data)

                # --- Collect poses ---
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
