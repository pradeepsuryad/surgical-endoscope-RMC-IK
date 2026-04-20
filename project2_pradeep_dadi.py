# project2_pradeep_dadi.py
# ME5250 Project 2 — 7-DOF Panda SE(3) NR-IK, Circular Trajectory
# Dadi Pradyumna Reddy

from __future__ import annotations
import argparse, time
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mujoco, mujoco.viewer
import numpy as np
import seaborn as sns

# ── Panda DH table [a, d, alpha, theta_offset] ────────────────────────────
DH = np.array([
    [ 0.000,  0.333,  0.000,     0],
    [ 0.000,  0.000, -np.pi/2,  0],
    [ 0.000,  0.316,  np.pi/2,  0],
    [ 0.0825, 0.000,  np.pi/2,  0],
    [-0.0825, 0.384, -np.pi/2,  0],
    [ 0.000,  0.000,  np.pi/2,  0],
    [ 0.088,  0.107,  np.pi/2,  0],
], dtype=float)

# Panda joint limits [rad]
QLIM = np.array([
    [-2.8973,  2.8973],
    [-1.7628,  1.7628],
    [-2.8973,  2.8973],
    [-3.0718, -0.0698],
    [-2.8973,  2.8973],
    [-0.0175,  3.7525],
    [-2.8973,  2.8973],
])

Q_HOME = np.array([0, -0.7854, 0, -2.3562, 0, 1.5708, 0.7854])
MODEL_XML = Path(__file__).parent / "models/franka_emika_panda/panda.xml"


# ── Kinematics ─────────────────────────────────────────────────────────────

def dh_mat(a, d, al, th):
    ct, st, ca, sa = np.cos(th), np.sin(th), np.cos(al), np.sin(al)
    return np.array([
        [ct,    -st,    0,   a   ],
        [st*ca,  ct*ca, -sa, -sa*d],
        [st*sa,  ct*sa,  ca,  ca*d],
        [0,      0,      0,   1  ],
    ])

def so3_log(R):
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if abs(th) < 1e-9:
        return np.zeros(3)
    ax = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2*np.sin(th))
    return ax / np.linalg.norm(ax) * th

def fk(q):
    T = np.eye(4)
    for i, (a, d, al, th0) in enumerate(DH):
        T = T @ dh_mat(a, d, al, q[i] + th0)
    return T

def fk_frames(q):
    Ts, T = [np.eye(4)], np.eye(4)
    for i, (a, d, al, th0) in enumerate(DH):
        T = T @ dh_mat(a, d, al, q[i] + th0)
        Ts.append(T.copy())
    return Ts

def jac_analytic(q):
    frames = fk_frames(q)
    p_ee = frames[-1][:3, 3]
    J = np.zeros((6, 7))
    for i in range(7):
        z = frames[i][:3, 2]
        J[:3, i] = np.cross(z, p_ee - frames[i][:3, 3])
        J[3:, i] = z
    return J

def se3_err(T_c, T_d):
    ep = T_d[:3, 3] - T_c[:3, 3]
    eo = so3_log(T_d[:3, :3] @ T_c[:3, :3].T)
    return np.r_[ep, eo]

def se3_err_nrm(T_c, T_d):
    e = se3_err(T_c, T_d)
    return np.linalg.norm(e[:3]), np.linalg.norm(e[3:])


# ── Trajectory ─────────────────────────────────────────────────────────────

def look_at_R(pos, tgt):
    up = np.array([0., 0., 1.])
    xax = tgt - pos; xax /= np.linalg.norm(xax)
    if abs(np.dot(xax, up)) > 0.999: up = np.array([0., 1., 0.])
    zax = np.cross(xax, up); zax /= np.linalg.norm(zax)
    yax = np.cross(zax, xax); yax /= np.linalg.norm(yax)
    return np.column_stack([xax, yax, zax])

def make_traj(R=0.15, z=0.5, cx=0., cy=0., step_mm=1.0, tgt=None):
    if tgt is None: tgt = np.zeros(3)
    n = max(3, int(np.ceil(2*np.pi*R / (step_mm * 1e-3))))
    wpts = []
    for th in np.linspace(0, 2*np.pi, n, endpoint=False):
        pos = np.array([cx + R*np.cos(th), cy + R*np.sin(th), z])
        T = np.eye(4)
        T[:3, :3] = look_at_R(pos, tgt)
        T[:3,  3] = pos
        wpts.append(T)
    return wpts


# ── NR-IK solver ───────────────────────────────────────────────────────────

def nr_ik(q0, T_d, jac_fn, fk_fn, max_iter=5, tol=1e-4,
          lam0=1e-6, lam1=0.05, sig_thr=0.05, kg=0.5):
    q = q0.copy()
    e_hist, ep_hist, eo_hist, lam_hist = [], [], [], []
    conv = False
    t0 = time.perf_counter()

    for _ in range(max_iter):
        T_c = fk_fn(q)
        e = se3_err(T_c, T_d)
        e_hist.append(np.linalg.norm(e))
        ep_hist.append(np.linalg.norm(e[:3]))
        eo_hist.append(np.linalg.norm(e[3:]))
        if e_hist[-1] < tol:
            conv = True
            break

        J = jac_fn(q)
        lam = lam1 if np.linalg.svd(J, compute_uv=False)[-1] < sig_thr else lam0
        lam_hist.append(lam)
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(6))

        # primary task + null-space joint-limit avoidance
        dq = J_pinv @ e
        N = np.eye(7) - J_pinv @ J
        q_mid = (QLIM[:, 0] + QLIM[:, 1]) / 2
        grad = 2*(q - q_mid) / ((QLIM[:, 1] - QLIM[:, 0]) / 2)**2
        dq += -kg * N @ grad

        q = np.clip(q + dq, QLIM[:, 0], QLIM[:, 1])

    if not conv:  # record final error after last update
        T_c = fk_fn(q)
        e = se3_err(T_c, T_d)
        e_hist.append(np.linalg.norm(e))
        ep_hist.append(np.linalg.norm(e[:3]))
        eo_hist.append(np.linalg.norm(e[3:]))

    return {
        'q':      q,
        'e_hist': e_hist,
        'ep_hist':ep_hist,
        'eo_hist':eo_hist,
        'lam_hist':lam_hist,
        'n_iter': len(e_hist) - (0 if conv else 1),
        'dt':     time.perf_counter() - t0,
        'conv':   conv,
    }


# ── Simulation ─────────────────────────────────────────────────────────────

def run_sim(wpts, n_laps=1, max_iter=5, render=True, record=False, model_path=None):
    xml = Path(model_path) if model_path else MODEL_XML
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)

    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if sid < 0: sid = m.nsite - 1

    def mj_fk(q):
        d.qpos[:7] = q; mujoco.mj_kinematics(m, d)
        T = np.eye(4)
        T[:3, 3]  = d.site_xpos[sid].copy()
        T[:3, :3] = d.site_xmat[sid].reshape(3, 3).copy()
        return T

    def mj_jac(q):
        d.qpos[:7] = q; mujoco.mj_kinematics(m, d)
        Jv = np.zeros((3, m.nv)); Jw = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, Jv, Jw, sid)
        return np.vstack([Jv[:, :7], Jw[:, :7]])

    q = Q_HOME.copy()
    d.qpos[:7] = q; d.ctrl[:7] = q
    mujoco.mj_forward(m, d)

    log = {k: [] for k in ['t','q','p_mj','R_mj','p_an','R_an',
                            'p_d','R_d','ep','eo','dt_ik','conv','n_iter','e_hist']}

    viewer = mujoco.viewer.launch_passive(m, d) if render else None
    renderer, frames = None, []
    if record:
        renderer = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = [0, 0, 0.4]; cam.distance = 1.5
        cam.azimuth = 135; cam.elevation = -20

    N = len(wpts) * n_laps
    for step in range(N):
        T_d = wpts[step % len(wpts)]
        res = nr_ik(q, T_d, mj_jac, mj_fk, max_iter=max_iter)
        q = res['q']

        d.qpos[:7] = q; d.ctrl[:7] = q
        mujoco.mj_forward(m, d)
        mujoco.mj_step(m, d)

        T_mj = mj_fk(q)
        T_an = fk(q)
        ep, eo = se3_err_nrm(T_mj, T_d)

        log['t'].append(float(d.time))
        log['q'].append(q.copy())
        log['p_mj'].append(T_mj[:3, 3].copy())
        log['R_mj'].append(T_mj[:3, :3].copy())
        log['p_an'].append(T_an[:3, 3].copy())
        log['R_an'].append(T_an[:3, :3].copy())
        log['p_d'].append(T_d[:3, 3].copy())
        log['R_d'].append(T_d[:3, :3].copy())
        log['ep'].append(ep); log['eo'].append(eo)
        log['dt_ik'].append(res['dt'])
        log['conv'].append(res['conv'])
        log['n_iter'].append(res['n_iter'])
        log['e_hist'].append(res['e_hist'])

        if viewer and viewer.is_running(): viewer.sync()
        if renderer:
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render().copy())

    if viewer:   viewer.close()
    if renderer: renderer.close()

    if frames:
        import imageio
        Path('results').mkdir(exist_ok=True)
        imageio.mimwrite('results/simulation.mp4', frames, fps=30)
        print('video saved -> results/simulation.mp4')

    for k in ['t', 'ep', 'eo', 'dt_ik', 'conv', 'n_iter']:
        log[k] = np.array(log[k])
    for k in ['q', 'p_mj', 'p_an', 'p_d']:
        log[k] = np.array(log[k])

    return log


# ── Plots ──────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
C = sns.color_palette("muted")

def _save(fig, path):
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {Path(path).name}")

def plot_3d(log, out):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*log['p_d'].T,  "--", c=C[0], lw=2,   label="Desired",       alpha=0.7)
    ax.plot(*log['p_mj'].T, "-",  c=C[1], lw=1.5, label="MuJoCo FK")
    ax.plot(*log['p_an'].T, ":",  c=C[2], lw=1.5, label="Analytical FK", alpha=0.8)
    ax.scatter(*log['p_d'][0],  c="g", s=60, zorder=5, label="Start")
    ax.scatter(*log['p_d'][-1], c="r", s=60, zorder=5, label="End")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_title("3D EE Trajectory (Desired vs Actual)")
    ax.legend(fontsize=9)
    _save(fig, out / "01_3d_trajectory.png")

def plot_errors(log, out):
    t = log['t']
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(t, log['ep']*1e3, c=C[1], lw=1.4)
    ax[0].fill_between(t, 0, log['ep']*1e3, alpha=0.15, color=C[1])
    ax[0].axhline(1, ls="--", c="gray", lw=0.9, label="1 mm")
    ax[0].set_ylabel("||ep|| [mm]"); ax[0].set_title("SE(3) Tracking Error")
    ax[0].legend(fontsize=9)
    ax[1].plot(t, log['eo'], c=C[2], lw=1.4)
    ax[1].fill_between(t, 0, log['eo'], alpha=0.15, color=C[2])
    ax[1].set_ylabel("||eo|| [rad]"); ax[1].set_xlabel("time [s]")
    fig.tight_layout()
    _save(fig, out / "02_error_over_time.png")

def plot_fk_cmp(log, out):
    t = log['t']; p_an = log['p_an']; p_mj = log['p_mj']
    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for i, lbl in enumerate(["X [m]", "Y [m]", "Z [m]"]):
        ax[i].plot(t, p_mj[:, i], c=C[3], lw=1.5, label="MuJoCo FK")
        ax[i].plot(t, p_an[:, i], c=C[2], lw=1.2, ls="--", label="Analytical FK")
        ax[i].set_ylabel(lbl); ax[i].legend(fontsize=8)
    ax[0].set_title("Analytical FK vs MuJoCo FK")
    diff = np.linalg.norm(p_an - p_mj, axis=1) * 1e3
    ax[3].plot(t, diff, c="tomato", lw=1.4)
    ax[3].fill_between(t, 0, diff, alpha=0.15, color="tomato")
    ax[3].set_ylabel("||FK diff|| [mm]"); ax[3].set_xlabel("time [s]")
    fig.tight_layout()
    _save(fig, out / "03_fk_comparison.png")

def plot_joints(log, out):
    t = log['t']; Q = log['q']
    fig, ax = plt.subplots(7, 1, figsize=(13, 14), sharex=True)
    for j, c in enumerate(sns.color_palette("tab10", 7)):
        ax[j].plot(t, np.degrees(Q[:, j]), c=c, lw=1.4)
        ax[j].set_ylabel(f"q{j+1} [deg]", fontsize=9)
    ax[0].set_title("Joint Angles"); ax[-1].set_xlabel("time [s]")
    fig.tight_layout()
    _save(fig, out / "04_joint_angles.png")

def plot_convergence(log, out):
    e_all = log['e_hist']
    Nmax  = max(len(e) for e in e_all)
    good  = [e for e, c in zip(e_all, log['conv']) if c]
    use   = good if len(good) >= 10 else [e for e in e_all if len(e) == Nmax]
    lbl   = f"converged, n={len(use)}" if good else f"non-conv, n={len(use)}"
    if not use: return

    pad = np.array([e + [e[-1]] * (Nmax - len(e)) for e in use])
    mu  = pad.mean(0); sig = pad.std(0)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogy(mu, "o-", c=C[1], lw=2, ms=6, label="Mean ||e_k||")
    ax.fill_between(range(Nmax), np.maximum(mu - sig, 1e-10), mu + sig,
                    alpha=0.2, color=C[1], label="+/-1 sig")
    ax.set_xlabel("NR iteration k"); ax.set_ylabel("||e_k||")
    ax.set_title(f"IK Convergence (max_iter={Nmax-1}, {lbl})")
    ax.legend()
    if mu[0] > 1e-10:
        ax.text(0.62, 0.85, f"Reduction: {(1 - mu[-1]/mu[0])*100:.1f}%",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    fig.tight_layout()
    _save(fig, out / "05_ik_convergence.png")

def plot_timing(log, out):
    t = log['t']; ms = log['dt_ik'] * 1e3
    w  = max(1, len(ms) // 30)
    ma = np.convolve(ms, np.ones(w)/w, mode="valid")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, ms, c=C[0], lw=0.8, alpha=0.5, label="per step")
    ax.plot(t[:len(ma)], ma, c=C[1], lw=2, label=f"MA (w={w})")
    ax.axhline(ms.mean(), ls="--", c="gray", lw=1, label=f"mean={ms.mean():.2f} ms")
    ax.set_xlabel("time [s]"); ax.set_ylabel("IK time [ms]")
    ax.set_title("IK Solve Time"); ax.legend()
    fig.tight_layout()
    _save(fig, out / "06_computation_time.png")

def plot_dashboard(log, out):
    t  = log['t']; Q = log['q']; ms = log['dt_ik'] * 1e3
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    fig.suptitle("Surgical Endoscope Tracking — Summary\n"
                 "7-DOF Franka Panda | SE(3) NR-IK | Circular Trajectory",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig)

    ax3d = fig.add_subplot(gs[:2, :2], projection="3d")
    ax3d.plot(*log['p_d'].T,  "--", c=C[0], lw=2,   label="Desired", alpha=0.7)
    ax3d.plot(*log['p_mj'].T, "-",  c=C[1], lw=1.5, label="Actual")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title("3D Trajectory"); ax3d.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, log['ep']*1e3, c=C[1], lw=1.3)
    ax.axhline(1, ls="--", c="gray", lw=0.9)
    ax.set_ylabel("||ep|| [mm]"); ax.set_title("Pos Error")

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, log['eo'], c=C[2], lw=1.3)
    ax.set_ylabel("||eo|| [rad]"); ax.set_title("Ori Error"); ax.set_xlabel("t [s]")

    ax = fig.add_subplot(gs[2, :2])
    for j, c in enumerate(sns.color_palette("tab10", 7)):
        ax.plot(t, np.degrees(Q[:, j]), c=c, lw=0.9, label=f"q{j+1}")
    ax.set_ylabel("[deg]"); ax.set_xlabel("t [s]"); ax.set_title("Joint Angles")
    ax.legend(ncol=7, fontsize=7, loc="upper right")

    ax = fig.add_subplot(gs[2, 2])
    ax.plot(t, ms, c=C[0], lw=0.8, alpha=0.6)
    ax.axhline(ms.mean(), ls="--", c="gray", lw=1, label=f"mu={ms.mean():.2f}ms")
    ax.set_ylabel("IK [ms]"); ax.set_xlabel("t [s]"); ax.set_title("Solve Time")
    ax.legend(fontsize=8)

    _save(fig, out / "00_summary_dashboard.png")

def plot_all(log, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    print(f"generating plots ({len(log['t'])} steps)...")
    plot_dashboard(log, out)
    plot_3d(log, out)
    plot_errors(log, out)
    plot_fk_cmp(log, out)
    plot_joints(log, out)
    plot_convergence(log, out)
    plot_timing(log, out)
    print(f"done — {out}/")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-render",  action="store_true")
    ap.add_argument("--record",     action="store_true")
    ap.add_argument("--laps",       type=int,   default=1)
    ap.add_argument("--max-iter",   type=int,   default=5)
    ap.add_argument("--radius",     type=float, default=0.15)
    ap.add_argument("--height",     type=float, default=0.50)
    ap.add_argument("--step-mm",    type=float, default=1.0)
    ap.add_argument("--model",      type=str,   default=None)
    args = ap.parse_args()

    wpts = make_traj(R=args.radius, z=args.height, step_mm=args.step_mm)
    print(f"trajectory: {len(wpts)} waypoints  R={args.radius}m  z={args.height}m  step={args.step_mm}mm")
    print(f"IK: NR max_iter={args.max_iter}, DLS fallback enabled")

    log = run_sim(wpts, n_laps=args.laps, max_iter=args.max_iter,
                  render=not args.no_render, record=args.record,
                  model_path=args.model)

    print(f"complete: {len(log['t'])} steps  "
          f"mean ep={log['ep'].mean()*1e3:.3f}mm  "
          f"mean eo={log['eo'].mean():.5f}rad")

    plot_all(log, Path(__file__).parent / "results")
