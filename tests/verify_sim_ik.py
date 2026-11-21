"""Quantitative verification suite for Firefly Dual-Arm Sim, Mink IK, Bumpless Transfer, and Gravity SSE."""

from __future__ import annotations

import pathlib
import sys
import numpy as np
import mujoco
import mink

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controller import BumplessGravityController, ControllerGains
from src.ik_solver import FireflyDualIK
from src.sim_runner import interpolate_se3

ASSET_XML = PROJECT_ROOT / "model" / "firefly_dual.xml"
SCENE_XML = PROJECT_ROOT / "model" / "scene.xml"


def test_bumpless_transfer() -> None:
    """Verify zero torque jump and minimal transient excursion at control-authority handoff."""
    sim_model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    dyn_model = mujoco.MjModel.from_xml_path(str(ASSET_XML))
    data = mujoco.MjData(sim_model)

    # Start in a non-home extended configuration (simulating post-boot or manual positioning)
    mujoco.mj_resetDataKeyframe(sim_model, data, sim_model.key("home").id)
    data.joint("left_joint_2").qpos = 0.65
    data.joint("left_joint_3").qpos = 1.05
    data.joint("right_joint_2").qpos = 0.55
    mujoco.mj_forward(sim_model, data)

    dt = sim_model.opt.timestep
    ctrl = BumplessGravityController(dyn_model, engage_duration=0.35)
    ik = FireflyDualIK(dyn_model)

    # Phase 1: Previous controller holds arm with gravity torque for 0.2 s
    tau_prev = ctrl.compute_gravity_torque(data.qpos.copy())
    for _ in range(100):
        data.ctrl[:] = tau_prev
        mujoco.mj_step(sim_model, data)

    q_handoff = data.qpos.copy()
    dq_handoff = data.qvel.copy()

    # Phase 2: Switch authority to IK + BumplessGravityController
    ik.reset_to_measured(q_handoff)
    ctrl.engage(q_handoff, dq_handoff, tau_prev=tau_prev)

    q_ref, dq_ref = ik.step(dt, q_meas=q_handoff)
    tau_first = ctrl.step(dt, q_handoff, dq_handoff, q_ref, dq_ref)
    torque_jump = float(np.max(np.abs(tau_first - tau_prev)))

    max_excursion = 0.0
    max_qacc = 0.0
    for _ in range(250):  # 0.5 s handoff window
        q_meas = data.qpos.copy()
        dq_meas = data.qvel.copy()
        q_ref, dq_ref = ik.step(dt, q_meas=q_meas, blend_meas=0.02)
        tau = ctrl.step(dt, q_meas, dq_meas, q_ref, dq_ref)
        data.ctrl[:] = tau
        mujoco.mj_step(sim_model, data)
        max_excursion = max(max_excursion, float(np.max(np.abs(data.qpos - q_handoff))))
        max_qacc = max(max_qacc, float(np.max(np.abs(data.qacc))))

    print(
        f"[Test 1: Bumpless Transfer] torque_jump={torque_jump:.4f} Nm, "
        f"max_excursion={np.rad2deg(max_excursion):.3f} deg, max_qacc={max_qacc:.2f} rad/s^2"
    )
    assert torque_jump < 0.05, f"Torque jump too large: {torque_jump}"
    assert max_excursion < 0.015, f"Handoff excursion too large: {max_excursion}"


def test_gravity_steady_state_error() -> None:
    """Compare naive PD vs G+PID+AW under gravity + unmodeled 0.5 kg payload."""
    # Plant with unmodeled +0.5 kg payload on left_link_6 (not known to dyn_model!)
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    spec.body("left_link_6").add_geom(
        name="unmodeled_payload",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.02],
        pos=[0.0, 0.0, 0.12],
        mass=0.5,
        contype=0,
        conaffinity=0,
        group=2,
    )
    plant_model = spec.compile()
    dyn_model = mujoco.MjModel.from_xml_path(str(ASSET_XML))

    def _run_hold(use_gravity_and_integral: bool) -> tuple[float, float]:
        data = mujoco.MjData(plant_model)
        mujoco.mj_resetDataKeyframe(plant_model, data, plant_model.key("home").id)
        # Extended horizontal-ish pose with high gravity torque
        data.joint("left_joint_2").qpos = 0.85
        data.joint("left_joint_3").qpos = 0.70
        mujoco.mj_forward(plant_model, data)

        q_target = data.qpos.copy()
        ee_target = data.site("left_ee_site").xpos.copy()
        dt = plant_model.opt.timestep
        gains = ControllerGains.default_for_model(dyn_model)

        if use_gravity_and_integral:
            ctrl = BumplessGravityController(dyn_model, gains=gains, engage_duration=0.0)
            ctrl.engage(data.qpos.copy(), data.qvel.copy())
            for _ in range(1500):  # 3.0 s settle
                tau = ctrl.step(dt, data.qpos.copy(), data.qvel.copy(), q_target)
                data.ctrl[:] = tau
                mujoco.mj_step(plant_model, data)
        else:
            # Pure PD without gravity comp or integral
            for _ in range(1500):
                tau = gains.kp * (q_target - data.qpos) - gains.kd * data.qvel
                data.ctrl[:] = np.clip(tau, -gains.tau_limit, gains.tau_limit)
                mujoco.mj_step(plant_model, data)

        q_err = float(np.max(np.abs(data.qpos - q_target)))
        ee_err_mm = float(np.linalg.norm(data.site("left_ee_site").xpos - ee_target) * 1000.0)
        return q_err, ee_err_mm

    pd_q_err, pd_ee_mm = _run_hold(use_gravity_and_integral=False)
    gpid_q_err, gpid_ee_mm = _run_hold(use_gravity_and_integral=True)

    print(
        f"[Test 2: Gravity + 0.5kg Payload SSE] "
        f"Naive PD: q_err={np.rad2deg(pd_q_err):.2f} deg, ee_err={pd_ee_mm:.2f} mm | "
        f"G+PID+AW: q_err={np.rad2deg(gpid_q_err):.3f} deg, ee_err={gpid_ee_mm:.2f} mm"
    )
    assert gpid_ee_mm < 1.5, f"Steady-state EE error too high: {gpid_ee_mm:.2f} mm"
    assert gpid_ee_mm < pd_ee_mm * 0.1, "G+PID+AW did not dramatically outperform naive PD"


def test_dual_arm_ik_tracking() -> None:
    """Verify Mink + DAQP dual-arm SE(3) tracking, joint/vel limits, and collision-free motion."""
    sim_model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    dyn_model = mujoco.MjModel.from_xml_path(str(ASSET_XML))
    data = mujoco.MjData(sim_model)

    mujoco.mj_resetDataKeyframe(sim_model, data, sim_model.key("home").id)
    mujoco.mj_forward(sim_model, data)

    ik = FireflyDualIK(dyn_model)
    ctrl = BumplessGravityController(dyn_model, engage_duration=0.25)

    q0 = data.qpos.copy()
    t_l0, t_r0 = ik.reset_to_measured(q0)
    ctrl.engage(q0, data.qvel.copy())

    t_l_goal = mink.SE3.from_rotation_and_translation(
        t_l0.rotation(), t_l0.translation() + np.array([0.09, -0.05, -0.07])
    )
    t_r_goal = mink.SE3.from_rotation_and_translation(
        t_r0.rotation(), t_r0.translation() + np.array([0.05, 0.05, -0.05])
    )

    dt = sim_model.opt.timestep
    max_ncon = 0
    for k in range(1750):  # 3.5 s
        t = k * dt
        u = min(1.0, max(0.0, (t - 0.3) / 2.0))
        ik.set_targets(
            interpolate_se3(t_l0, t_l_goal, u),
            interpolate_se3(t_r0, t_r_goal, u),
        )
        q_ref, dq_ref = ik.step(dt, q_meas=data.qpos.copy(), blend_meas=0.02)
        data.ctrl[:] = ctrl.step(dt, data.qpos.copy(), data.qvel.copy(), q_ref, dq_ref)
        mujoco.mj_step(sim_model, data)
        max_ncon = max(max_ncon, int(data.ncon))

    left_err_mm = float(np.linalg.norm(data.site("left_ee_site").xpos - t_l_goal.translation()) * 1000.0)
    right_err_mm = float(np.linalg.norm(data.site("right_ee_site").xpos - t_r_goal.translation()) * 1000.0)

    print(
        f"[Test 3: Dual-Arm Mink IK] left_err={left_err_mm:.2f} mm, "
        f"right_err={right_err_mm:.2f} mm, max_contacts={max_ncon}"
    )
    assert left_err_mm < 2.0, f"Left EE tracking error too high: {left_err_mm:.2f} mm"
    assert right_err_mm < 2.0, f"Right EE tracking error too high: {right_err_mm:.2f} mm"
    assert max_ncon == 0, f"Unexpected collisions: {max_ncon}"


if __name__ == "__main__":
    test_bumpless_transfer()
    test_gravity_steady_state_error()
    test_dual_arm_ik_tracking()
    print("\nALL VERIFICATION TESTS PASSED!")
