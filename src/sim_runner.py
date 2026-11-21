"""Simulation runner for Firefly Dual-Arm with Mink IK and Bumpless Gravity Control."""

from __future__ import annotations

import pathlib
import time
import numpy as np
import mujoco
import mink

from src.controller import BumplessGravityController
from src.ik_solver import FireflyDualIK, quintic_blend

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSET_XML_PATH = PROJECT_ROOT / "model" / "firefly_dual.xml"
SCENE_XML_PATH = PROJECT_ROOT / "model" / "scene.xml"


def interpolate_se3(t_start: mink.SE3, t_goal: mink.SE3, u: float) -> mink.SE3:
    """Smooth C^2 quintic interpolation on SE(3)."""
    s, _ = quintic_blend(u)
    return t_start.interpolate(t_goal, s)


def run_closed_loop_demo(duration: float = 4.0, launch_viewer: bool = False) -> dict[str, np.ndarray]:
    """Run a dual-arm closed-loop simulation from home pose with bumpless engage."""
    sim_model = mujoco.MjModel.from_xml_path(str(SCENE_XML_PATH))
    dyn_model = mujoco.MjModel.from_xml_path(str(ASSET_XML_PATH))
    data = mujoco.MjData(sim_model)

    mujoco.mj_resetDataKeyframe(sim_model, data, sim_model.key("home").id)
    mujoco.mj_forward(sim_model, data)

    ik = FireflyDualIK(dyn_model)
    ctrl = BumplessGravityController(dyn_model, engage_duration=0.35)

    # Simulate bumpless handoff from current measured state
    q_init = data.qpos[: dyn_model.nq].copy()
    dq_init = data.qvel[: dyn_model.nv].copy()
    t_left_0, t_right_0 = ik.reset_to_measured(q_init)
    ctrl.engage(q_init, dq_init)

    # Smooth target: move left EE forward +0.08 m, down -0.06 m; right EE forward +0.08 m, up +0.05 m
    t_left_goal = mink.SE3.from_rotation_and_translation(
        t_left_0.rotation(), t_left_0.translation() + np.array([0.08, -0.04, -0.06])
    )
    t_right_goal = mink.SE3.from_rotation_and_translation(
        t_right_0.rotation(), t_right_0.translation() + np.array([0.05, 0.04, -0.05])
    )

    dt = sim_model.opt.timestep
    steps = int(duration / dt)
    move_start, move_dur = 0.5, 2.0

    history = {
        "t": np.zeros(steps),
        "left_err_m": np.zeros(steps),
        "right_err_m": np.zeros(steps),
        "max_tau": np.zeros(steps),
    }

    viewer_ctx = None
    if launch_viewer:
        from mujoco import viewer as mj_viewer
        viewer_ctx = mj_viewer.launch_passive(sim_model, data)

    try:
        for k in range(steps):
            t = k * dt
            if t >= move_start:
                u = min(1.0, (t - move_start) / move_dur)
                ik.set_targets(
                    interpolate_se3(t_left_0, t_left_goal, u),
                    interpolate_se3(t_right_0, t_right_goal, u),
                )

            q_meas = data.qpos[: dyn_model.nq].copy()
            dq_meas = data.qvel[: dyn_model.nv].copy()

            q_ref, dq_ref = ik.step(dt, q_meas=q_meas, blend_meas=0.02)
            tau = ctrl.step(dt, q_meas, dq_meas, q_ref, dq_ref)
            data.ctrl[: dyn_model.nu] = tau

            mujoco.mj_step(sim_model, data)

            history["t"][k] = t
            history["left_err_m"][k] = np.linalg.norm(
                data.site("left_ee_site").xpos - ik.left_task.transform_target_to_world.translation()
            )
            history["right_err_m"][k] = np.linalg.norm(
                data.site("right_ee_site").xpos - ik.right_task.transform_target_to_world.translation()
            )
            history["max_tau"][k] = np.max(np.abs(tau))

            if viewer_ctx is not None:
                viewer_ctx.sync()
                time.sleep(dt * 0.5)
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()

    return history


if __name__ == "__main__":
    hist = run_closed_loop_demo(duration=4.0, launch_viewer=False)
    print(f"Final left EE error : {hist['left_err_m'][-1]*1000:.2f} mm")
    print(f"Final right EE error: {hist['right_err_m'][-1]*1000:.2f} mm")
    print(f"Peak torque         : {np.max(hist['max_tau']):.2f} Nm")
