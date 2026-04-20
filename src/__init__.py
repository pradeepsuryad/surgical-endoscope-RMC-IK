"""
surgical_endoscope_tracking.src
================================
Package for the surgical endoscope tracking simulation.

Modules
-------
kinematics  : Analytical FK (DH params) and SE(3) spatial-error math.
ik_solver   : Newton-Raphson / damped-least-squares IK solver.
trajectory  : Circular trajectory generation with RCM orientation targeting.
simulation  : MuJoCo environment setup and main simulation step loop.
visualizer  : Matplotlib / Seaborn post-processing plots.
"""
