"""Step 3: Left-arm gravity-compensated MIT pick-and-place for tissue pack.

Reuses Step 2's GuardedHexDeviceApi (t=0 send gate blocking both arm & secondary
gripper commands when un-armed or stopped, plus frame-0 E-Stop tracker),
zero-write passive dual-arm state reading, and full-frame k=0..N MuJoCo safety check
with GR100 finger DOFs, stationary/carried/placed tissue-pack colliders, and
fail-closed gripper/arm/thermal/torque verification.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable

import mink
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ik_solver import FireflyDualIK
from real_calibration import LEFT_ARM_WS_URL
from step2_left_arm_air_jog import (
    Q_LEFT_READY,
    SafeArmSendGate,
    _check_arm_fault,
    _check_over_temperature,
    _install_parking_stop_tracker,
    _install_safe_send_gate,
    _is_parking_stop_active,
    _safe_read_q,
    _send_arm_command,
    build_quintic_segment,
    compute_left_arm_gravity_torque,
    create_guarded_hex_device_api,
    detect_target_from_top_d435,
    read_dual_arms_passive_q,
)

# GR100 gripper joint convention in model/firefly_dual.xml & assets/firefly_dual.urdf:
#   0.00 rad = fully open (~97.5 mm fingertip gap)
#   0.60 rad = fully closed (~10.1 mm fingertip gap)
GRIP_OPEN_RAD = 0.02
GRIP_MAX_OPEN_ALLOWED_RAD = 0.08  # Both MuJoCo verification & runtime open check use this exact bound
GRIP_CLOSE_RAD = 0.55


def _solve_left_ik(
    ik: FireflyDualIK,
    q_seed: np.ndarray,
    pos_wc: np.ndarray,
    R_wc: np.ndarray,
    left_q_adrs: list[int],
    label: str,
) -> np.ndarray:
    target = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R_wc), pos_wc)
    q_sol, info = ik.solve_pose_global(
        q_seed=q_seed,
        left_target=target,
        right_target=None,
        pos_tol_m=0.002,
        ori_tol_rad=0.02,
    )
    if not info.get("converged", False):
        raise RuntimeError(f"Step 3 IK ({label}) did not converge: {info}")
    return np.array([q_sol[a] for a in left_q_adrs], dtype=np.float64)


def check_perception_unoccluded(perception: dict[str, Any], min_length_m: float = 0.13) -> list[str]:
    """Detect clipped/occluded tissue-pack segmentation (e.g. arm blocking top D435)."""
    issues: list[str] = []
    fp = perception.get("object_footprint_m", [0.0, 0.0])
    length_m, width_m = float(fp[0]), float(fp[1])
    if length_m < min_length_m:
        issues.append(
            f"Detected object length {length_m:.3f} m < {min_length_m:.3f} m "
            "(likely occluded by robot arm in top D435 view)"
        )
    if width_m < 0.05 or width_m > 0.14:
        issues.append(f"Detected object width {width_m:.3f} m outside [0.05, 0.14] m")
    return issues


def build_step3_verification_model(
    z_table_wc: float,
    perception: dict[str, Any],
    obj_xy_margin_m: float = 0.002,
    obj_z_margin_m: float = 0.002,
) -> tuple[mujoco.MjModel, dict[str, Any]]:
    """Compile firefly_dual.xml with tabletop + mocap stationary/placed & carried object colliders."""
    spec = mujoco.MjSpec.from_file(str(ROOT / "model" / "firefly_dual.xml"))

    table_half_z = 0.025
    spec.worldbody.add_geom(
        name="table_surface_col",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.35, 0.0, float(z_table_wc) - table_half_z],
        size=[0.60, 0.65, table_half_z],
        group=3,
        contype=1,
        conaffinity=1,
        rgba=[0.6, 0.45, 0.3, 0.3],
    )

    cx, cy = float(perception["object_center_wc_m"][0]), float(perception["object_center_wc_m"][1])
    z_top = float(perception["object_z_top_wc_m"])
    length_m, width_m = perception.get("object_footprint_m", [0.18, 0.09])
    major_xy = perception.get("major_axis_wc_xy", [0.0, 1.0])
    yaw = math.atan2(float(major_xy[1]), float(major_xy[0]))
    half_z = max(0.005, 0.5 * (z_top - z_table_wc + obj_z_margin_m))
    center_z = z_table_wc + half_z
    box_size = [
        0.5 * float(length_m) + obj_xy_margin_m,
        0.5 * float(width_m) + obj_xy_margin_m,
        half_z,
    ]
    yaw_quat = [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]

    # Mocap body 1: tissue pack on table (at pick pose before lift, at place pose after release)
    table_obj_body = spec.worldbody.add_body(
        name="table_object_mocap",
        mocap=True,
        pos=[cx, cy, center_z],
        quat=yaw_quat,
    )
    table_obj_body.add_geom(
        name="target_object_col",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, 0.0],
        size=box_size,
        group=3,
        contype=1,
        conaffinity=1,
        rgba=[0.9, 0.2, 0.2, 0.35],
    )

    # Mocap body 2: tissue pack carried by left gripper during lift -> to_place_hover -> descend_place
    carried_body = spec.worldbody.add_body(
        name="carried_object_mocap",
        mocap=True,
        pos=[cx, cy, -10.0],
        quat=[1.0, 0.0, 0.0, 0.0],
    )
    carried_body.add_geom(
        name="carried_object_col",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, 0.0],
        size=box_size,
        group=3,
        contype=1,
        conaffinity=1,
        rgba=[0.2, 0.8, 0.2, 0.35],
    )

    model = spec.compile()
    meta = {
        "pick_pos": np.array([cx, cy, center_z], dtype=np.float64),
        "yaw_quat": np.array(yaw_quat, dtype=np.float64),
        "half_z": half_z,
    }
    return model, meta


def plan_and_verify_step3(
    q_meas: np.ndarray,
    right_q_meas: np.ndarray,
    perception: dict[str, Any],
    place_offset_y_m: float = 0.14,
    hover_clearance_m: float = 0.08,
    max_vel: float = 0.25,
    dt: float = 0.01,
) -> dict[str, Any]:
    """Plan all 9 pick-and-place segments and verify EVERY frame (stationary, carried, placed)."""
    q_meas = np.asarray(q_meas, dtype=np.float64)
    right_q_meas = np.asarray(right_q_meas, dtype=np.float64)
    z_table = float(perception["z_table_wc_m"])
    c = np.asarray(perception["object_center_wc_m"], dtype=np.float64)
    z_top = float(perception["object_z_top_wc_m"])

    # Safe grasp Z: middle of tissue pack, always >= 4.5 cm above table surface
    z_grasp = max(z_table + 0.045, z_top - 0.035)
    z_hover = z_top + hover_clearance_m

    p_hover = np.array([c[0], c[1], z_hover], dtype=np.float64)
    p_grasp = np.array([c[0], c[1], z_grasp], dtype=np.float64)
    p_place_hover = np.array([c[0], c[1] + place_offset_y_m, z_hover], dtype=np.float64)
    p_place = np.array([c[0], c[1] + place_offset_y_m, z_grasp + 0.008], dtype=np.float64)

    # Align gripper jaws across the tissue pack's minor (short ~8 cm) axis
    major_xy = np.asarray(perception.get("major_axis_wc_xy", [0.0, 1.0]), dtype=np.float64)
    major_xy = major_xy / max(1e-9, float(np.linalg.norm(major_xy)))
    z_tcp = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    y_tcp = np.array([-major_xy[1], major_xy[0], 0.0], dtype=np.float64)
    if y_tcp[0] < 0:
        y_tcp = -y_tcp
    x_tcp = np.cross(y_tcp, z_tcp)
    R_grasp = np.column_stack([x_tcp, y_tcp, z_tcp])

    base_model = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "firefly_dual.xml"))
    verify_model, obj_meta = build_step3_verification_model(z_table_wc=z_table, perception=perception)
    verify_data = mujoco.MjData(verify_model)

    left_q_adrs = [
        int(verify_model.jnt_qposadr[mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_JOINT, f"left_joint_{i}")])
        for i in range(1, 7)
    ]
    right_q_adrs = [
        int(verify_model.jnt_qposadr[mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_JOINT, f"right_joint_{i}")])
        for i in range(1, 7)
    ]
    left_grip_adrs = [
        int(verify_model.jnt_qposadr[mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_JOINT, "left_grip_joint_1")]),
        int(verify_model.jnt_qposadr[mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_JOINT, "left_grip_joint_2")]),
    ]
    left_ranges = np.array(
        [
            verify_model.jnt_range[mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_JOINT, f"left_joint_{i}")]
            for i in range(1, 7)
        ],
        dtype=np.float64,
    )
    left_tcp_sid = mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_SITE, "left_tcp_site")
    table_gid = mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_GEOM, "table_surface_col")
    obj_gid = mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_col")
    carried_gid = mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_GEOM, "carried_object_col")
    table_mocap_id = int(verify_model.body_mocapid[mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_BODY, "table_object_mocap")])
    carried_mocap_id = int(verify_model.body_mocapid[mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_BODY, "carried_object_mocap")])

    left_col_gids = [
        gid
        for gid in range(verify_model.ngeom)
        if (verify_model.geom_contype[gid] != 0 or verify_model.geom_conaffinity[gid] != 0)
        and verify_model.body(verify_model.geom_bodyid[gid]).name.startswith("left_")
    ]
    env_and_right_gids = [
        gid
        for gid in range(verify_model.ngeom)
        if (verify_model.geom_contype[gid] != 0 or verify_model.geom_conaffinity[gid] != 0)
        and not verify_model.body(verify_model.geom_bodyid[gid]).name.startswith("left_")
        and gid not in (obj_gid, carried_gid)
    ]

    ik = FireflyDualIK(base_model, site_suffix="tcp_site")
    q_seed = np.zeros(base_model.nq, dtype=np.float64)
    for i, a in enumerate(left_q_adrs):
        q_seed[a] = Q_LEFT_READY[i]
    for i, a in enumerate(right_q_adrs):
        q_seed[a] = right_q_meas[i]

    q_hover = _solve_left_ik(ik, q_seed, p_hover, R_grasp, left_q_adrs, "hover")
    for i, a in enumerate(left_q_adrs):
        q_seed[a] = q_hover[i]
    q_grasp = _solve_left_ik(ik, q_seed, p_grasp, R_grasp, left_q_adrs, "grasp")
    q_place_hover = _solve_left_ik(ik, q_seed, p_place_hover, R_grasp, left_q_adrs, "place_hover")
    for i, a in enumerate(left_q_adrs):
        q_seed[a] = q_place_hover[i]
    q_place = _solve_left_ik(ik, q_seed, p_place, R_grasp, left_q_adrs, "place")

    segments = {
        "unfold": build_quintic_segment(q_meas, Q_LEFT_READY, max_vel=max_vel, dt=dt),
        "to_hover": build_quintic_segment(Q_LEFT_READY, q_hover, max_vel=max_vel, dt=dt),
        "descend_grasp": build_quintic_segment(q_hover, q_grasp, max_vel=0.15, dt=dt),
        "lift": build_quintic_segment(q_grasp, q_hover, max_vel=0.18, dt=dt),
        "to_place_hover": build_quintic_segment(q_hover, q_place_hover, max_vel=0.22, dt=dt),
        "descend_place": build_quintic_segment(q_place_hover, q_place, max_vel=0.15, dt=dt),
        "retreat_place": build_quintic_segment(q_place, q_place_hover, max_vel=0.20, dt=dt),
        "return_ready": build_quintic_segment(q_place_hover, Q_LEFT_READY, max_vel=max_vel, dt=dt),
        "return_rest": build_quintic_segment(Q_LEFT_READY, q_meas, max_vel=max_vel, dt=dt),
    }
    seg_grip_rad = {
        "unfold": GRIP_OPEN_RAD,
        "to_hover": GRIP_OPEN_RAD,
        "descend_grasp": GRIP_OPEN_RAD,
        "lift": GRIP_CLOSE_RAD,
        "to_place_hover": GRIP_CLOSE_RAD,
        "descend_place": GRIP_CLOSE_RAD,
        "retreat_place": GRIP_OPEN_RAD,
        "return_ready": GRIP_OPEN_RAD,
        "return_rest": GRIP_OPEN_RAD,
    }
    pick_stationary_segs = {"unfold", "to_hover", "descend_grasp"}
    carried_segs = {"lift", "to_place_hover", "descend_place"}
    placed_stationary_segs = {"retreat_place", "return_ready", "return_rest"}

    pick_pos = obj_meta["pick_pos"]
    place_pos = np.array([p_place[0], p_place[1], pick_pos[2]], dtype=np.float64)
    yaw_quat = obj_meta["yaw_quat"]
    offscreen_pos = np.array([0.0, 0.0, -10.0], dtype=np.float64)

    # Compute TCP -> object offset at q_grasp so carried_object_col tracks the physical grasp pose
    mujoco.mj_resetData(verify_model, verify_data)
    for i, a in enumerate(left_q_adrs):
        verify_data.qpos[a] = q_grasp[i]
    mujoco.mj_forward(verify_model, verify_data)
    R_tcp_grasp = verify_data.site_xmat[left_tcp_sid].reshape(3, 3).copy()
    p_tcp_grasp = verify_data.site_xpos[left_tcp_sid].copy()
    tcp_to_obj_local = R_tcp_grasp.T @ (pick_pos - p_tcp_grasp)
    R_yaw = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(R_yaw, yaw_quat)
    R_obj_in_tcp = R_tcp_grasp.T @ R_yaw.reshape(3, 3)

    total_frames = 0
    min_table_dist = 1e9
    min_obj_dist = 1e9
    min_carried_env_dist = 1e9
    limit_violations = 0
    collisions: list[tuple[str, int, str, str, float]] = []

    for seg_name, seg in segments.items():
        g_rad = seg_grip_rad[seg_name]
        # Verify open-jaw segments at BOTH nominal GRIP_OPEN_RAD and worst-case GRIP_MAX_OPEN_ALLOWED_RAD
        g_check_list = (
            [GRIP_OPEN_RAD, GRIP_MAX_OPEN_ALLOWED_RAD]
            if g_rad == GRIP_OPEN_RAD
            else [g_rad]
        )
        for k in range(len(seg)):
            total_frames += 1
            q_k = seg[k]
            if np.any(q_k < left_ranges[:, 0] - 1e-6) or np.any(q_k > left_ranges[:, 1] + 1e-6):
                limit_violations += 1

            for g_test in g_check_list:
                mujoco.mj_resetData(verify_model, verify_data)
                for i, a in enumerate(left_q_adrs):
                    verify_data.qpos[a] = q_k[i]
                for i, a in enumerate(right_q_adrs):
                    verify_data.qpos[a] = right_q_meas[i]
                for ga in left_grip_adrs:
                    verify_data.qpos[ga] = g_test

                # Place mocap object colliders according to stage
                mujoco.mj_kinematics(verify_model, verify_data)
                if seg_name in pick_stationary_segs:
                    verify_data.mocap_pos[table_mocap_id] = pick_pos
                    verify_data.mocap_quat[table_mocap_id] = yaw_quat
                    verify_data.mocap_pos[carried_mocap_id] = offscreen_pos
                elif seg_name in carried_segs:
                    verify_data.mocap_pos[table_mocap_id] = offscreen_pos
                    R_tcp_k = verify_data.site_xmat[left_tcp_sid].reshape(3, 3)
                    p_tcp_k = verify_data.site_xpos[left_tcp_sid]
                    verify_data.mocap_pos[carried_mocap_id] = p_tcp_k + R_tcp_k @ tcp_to_obj_local
                    q_car = np.zeros(4, dtype=np.float64)
                    mujoco.mju_mat2Quat(q_car, (R_tcp_k @ R_obj_in_tcp).ravel())
                    verify_data.mocap_quat[carried_mocap_id] = q_car
                else:  # placed_stationary_segs
                    verify_data.mocap_pos[table_mocap_id] = place_pos
                    verify_data.mocap_quat[table_mocap_id] = yaw_quat
                    verify_data.mocap_pos[carried_mocap_id] = offscreen_pos

                mujoco.mj_forward(verify_model, verify_data)

                for ci in range(verify_data.ncon):
                    con = verify_data.contact[ci]
                    if con.dist < -1e-4:
                        g1 = verify_model.geom(con.geom1).name or f"g{con.geom1}"
                        g2 = verify_model.geom(con.geom2).name or f"g{con.geom2}"
                        b1 = verify_model.body(verify_model.geom_bodyid[con.geom1]).name
                        b2 = verify_model.body(verify_model.geom_bodyid[con.geom2]).name
                        # Ignore contact between carried_object_col and the left gripper jaws holding it,
                        # but DO catch collisions with left_link_1..5, wrist D405 bracket, right arm, or table!
                        carried_gripper_bodies = {
                            "left_grip_link_1",
                            "left_grip_link_2",
                            "left_gr100_base_link",
                            "left_link_6",
                        }
                        if (g1 == "carried_object_col" and b2 in carried_gripper_bodies) or (
                            g2 == "carried_object_col" and b1 in carried_gripper_bodies
                        ):
                            continue
                        if b1.startswith("left_") or b2.startswith("left_") or g1 == "carried_object_col" or g2 == "carried_object_col":
                            collisions.append((seg_name, k, g1, g2, round(float(-con.dist) * 1000.0, 2)))

                for gid in left_col_gids:
                    d_tab = float(mujoco.mj_geomDistance(verify_model, verify_data, gid, table_gid, 1.0, None))
                    min_table_dist = min(min_table_dist, d_tab)
                    if seg_name in pick_stationary_segs or seg_name in placed_stationary_segs:
                        d_obj = float(mujoco.mj_geomDistance(verify_model, verify_data, gid, obj_gid, 1.0, None))
                        min_obj_dist = min(min_obj_dist, d_obj)

                if seg_name in carried_segs:
                    # Non-table obstacles (right arm, workcell): check >= 5 mm clearance on EVERY frame!
                    for egid in env_and_right_gids:
                        if egid == table_gid:
                            d_tab_car = float(mujoco.mj_geomDistance(verify_model, verify_data, carried_gid, table_gid, 1.0, None))
                            if d_tab_car < -0.0005:
                                min_carried_env_dist = min(min_carried_env_dist, d_tab_car)
                            elif (seg_name == "to_place_hover") or (seg_name == "lift" and k >= int(0.4 * len(seg))) or (seg_name == "descend_place" and k <= int(0.6 * len(seg))):
                                min_carried_env_dist = min(min_carried_env_dist, d_tab_car)
                        else:
                            d_car = float(mujoco.mj_geomDistance(verify_model, verify_data, carried_gid, egid, 1.0, None))
                            min_carried_env_dist = min(min_carried_env_dist, d_car)

    blocking = check_perception_unoccluded(perception)
    if limit_violations > 0:
        blocking.append(f"Joint limit violations on {limit_violations} frames")
    if collisions:
        blocking.append(f"Collisions detected: {collisions[0]}")
    if min_table_dist < 0.015:
        blocking.append(f"Min finger/arm clearance above table {min_table_dist:.4f} m < 0.0150 m")
    if min_carried_env_dist < 0.005:
        blocking.append(f"Min carried-object clearance to table/right-arm/workcell {min_carried_env_dist:.4f} m < 0.0050 m")

    return {
        "safe_to_execute": len(blocking) == 0,
        "blocking_reasons": blocking,
        "total_frames_checked": total_frames,
        "grip_open_rad": GRIP_OPEN_RAD,
        "grip_close_rad": GRIP_CLOSE_RAD,
        "min_geom_clearance_table_m": round(min_table_dist, 4),
        "min_geom_clearance_object_m": round(min_obj_dist, 4) if min_obj_dist < 1e8 else None,
        "min_carried_object_clearance_m": round(min_carried_env_dist, 4) if min_carried_env_dist < 1e8 else None,
        "p_hover_wc_m": [round(float(x), 4) for x in p_hover],
        "p_grasp_wc_m": [round(float(x), 4) for x in p_grasp],
        "p_place_wc_m": [round(float(x), 4) for x in p_place],
        "total_motion_duration_s": round(sum(len(s) - 1 for s in segments.values()) * dt, 2),
        "_segments": segments,
    }


def _find_gripper_device(api: Any) -> Any:
    """Locate SdtHandGr100 (left gripper type=6) or fallback hand device."""
    from hex_device import Hands, public_api_types_pb2

    for dev in api.device_list:
        if isinstance(dev, Hands):
            return dev
    finder = getattr(api, "find_optional_device_by_robot_type", None)
    if finder is None:
        return None
    for sdt in (
        public_api_types_pb2.SecondaryDeviceType.SdtHandGr100,
        public_api_types_pb2.SecondaryDeviceType.SdtHandGp100,
        public_api_types_pb2.SecondaryDeviceType.SdtHandGp80G1,
    ):
        opt = finder(sdt)
        if opt:
            return opt[0] if isinstance(opt, list) else opt
    return None


def _read_fresh_gripper_q(hand_dev: Any) -> tuple[float | None, str | None]:
    """Pop one fresh gripper position sample if available; returns (val, error_msg)."""
    if not hand_dev.has_new_data():
        return None, None
    pos = hand_dev.get_motor_positions(True)
    if pos is None or len(pos) == 0:
        return None, "Empty gripper position telemetry"
    try:
        val = float(pos[0])
    except (TypeError, ValueError):
        return None, f"Non-numeric gripper position telemetry: {pos[0]!r}"
    if not math.isfinite(val) or val < -0.1 or val > 0.7:
        return None, f"Invalid/out-of-range gripper telemetry: {val}"
    return val, None


def execute_step3_pick_place(
    segments: dict[str, np.ndarray],
    dt: float = 0.01,
    max_tracking_err_rad: float = 0.25,
    stale_timeout_s: float = 0.15,
    goal_tol_rad: float = 0.02,
    sustained_torque_limit_nm: float = 14.0,
    sustained_torque_max_steps: int = 50,
    api_factory: Callable[[], Any] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Execute full gravity-compensated MIT pick-and-place with fail-closed checks at every stage."""
    from hex_device import Arm
    from hex_device.motor_base import CommandType

    stop_requested = False

    def _sigint(_s, _f):
        nonlocal stop_requested
        stop_requested = True

    old_handler = signal.signal(signal.SIGINT, _sigint)

    dyn_model = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "firefly_dual.xml"))
    dyn_data = mujoco.MjData(dyn_model)
    dyn_q_adrs = [
        int(dyn_model.jnt_qposadr[mujoco.mj_name2id(dyn_model, mujoco.mjtObj.mjOBJ_JOINT, f"left_joint_{i}")])
        for i in range(1, 7)
    ]
    dyn_dof_adrs = [
        int(dyn_model.jnt_dofadr[mujoco.mj_name2id(dyn_model, mujoco.mjtObj.mjOBJ_JOINT, f"left_joint_{i}")])
        for i in range(1, 7)
    ]

    if api_factory is None:
        api = create_guarded_hex_device_api(
            ws_url=LEFT_ARM_WS_URL, control_hz=200, enable_kcp=True, local_port=0
        )
        arm_cls = Arm
    else:
        api = api_factory()
        arm_cls = None

    kp_mit = [60.0, 80.0, 80.0, 40.0, 25.0, 20.0]
    kd_mit = [3.0, 4.0, 4.0, 2.0, 1.2, 1.0]
    vel_lim = [0.6] * 6
    acc_lim = [1.2] * 6

    arm_dev = None
    hand_dev = None
    send_gate: SafeArmSendGate | None = None
    aborted_reason: str | None = None
    max_err_seen = 0.0
    max_tau_seen = 0.0
    high_tau_streak = 0
    grasp_verified = False
    release_verified = False
    q_grip_before_close: float | None = None
    q_grip_after_close: float | None = None
    q_grip_after_open: float | None = None
    final_q: np.ndarray | None = None
    final_feedback_fresh = False

    try:
        t0 = clock_fn()
        while clock_fn() - t0 < 5.0:
            for dev in api.device_list:
                is_arm = (
                    isinstance(dev, arm_cls)
                    if arm_cls is not None
                    else (hasattr(dev, "joint_position_control") or hasattr(dev, "compensated_mit_control"))
                )
                if is_arm and dev.has_new_data():
                    arm_dev = dev
            if hand_dev is None:
                hand_dev = _find_gripper_device(api)
            if arm_dev is not None and hand_dev is not None and hand_dev.has_new_data():
                break
            sleep_fn(0.05)

        if arm_dev is None:
            raise TimeoutError("Could not connect to left arm.")

        send_gate = getattr(api, "send_gate", None) or _install_safe_send_gate(arm_dev, start_blocked=True)
        _install_parking_stop_tracker(arm_dev, send_gate=send_gate)

        # Pre-start fail-closed gates BEFORE allow_execution() / arm_dev.start() / hand commands!
        if hand_dev is None or not hand_dev.has_new_data():
            aborted_reason = "Left gripper (SdtHandGr100) not detected or no telemetry; refusing pick-and-place."

        ps_present = bool(getattr(arm_dev, "_parking_stop_present", False))
        ps_detail = arm_dev.get_parking_stop_detail() if hasattr(arm_dev, "get_parking_stop_detail") else None
        if aborted_reason is None and (ps_present or _is_parking_stop_active(ps_detail)):
            aborted_reason = f"Pre-start parking stop active: {ps_detail}"

        if aborted_reason is None:
            init_errs = arm_dev.get_motor_error_codes()
            if init_errs and any(e is not None for e in init_errs):
                aborted_reason = f"Pre-start motor error codes: {init_errs}"

        if aborted_reason is None:
            temp_err = _check_over_temperature(arm_dev, max_temp_c=65.0)
            if temp_err is not None:
                aborted_reason = f"Pre-start {temp_err}"

        q_live = _safe_read_q(arm_dev, True)
        if q_live is None:
            if aborted_reason is None:
                raise RuntimeError("Arm queue empty at startup.")
            last_q = segments["unfold"][0].copy()
        else:
            last_q = q_live.copy()
            q0_plan = segments["unfold"][0]
            if aborted_reason is None and float(np.max(np.abs(q_live - q0_plan))) > 0.05:
                aborted_reason = "Live arm moved since planning."

        last_fresh_t = clock_fn()
        hand_last_fresh_t = clock_fn()
        hand_last_fresh_q: float | None = None
        hand_fresh_in_stage = 0

        def _hold_and_stream(
            traj: np.ndarray,
            grip_pos: float | None = None,
            check_carried_grasp: bool = False,
        ) -> None:
            nonlocal aborted_reason, last_q, last_fresh_t, max_err_seen, max_tau_seen
            nonlocal high_tau_streak, hand_last_fresh_t, hand_last_fresh_q, hand_fresh_in_stage
            nonlocal grasp_verified
            if aborted_reason is not None:
                return
            hand_fresh_in_stage = 0
            t_seg = clock_fn()
            for k in range(len(traj)):
                now_t = clock_fn()
                q_cmd = traj[k]
                dq_cmd = (traj[min(k + 1, len(traj) - 1)] - traj[max(k - 1, 0)]) / (
                    dt if k == 0 or k == len(traj) - 1 else 2.0 * dt
                )
                aborted_reason, last_q, err, max_tau, last_fresh_t = _check_arm_fault(
                    api, arm_dev, q_cmd, last_q, now_t, last_fresh_t, stale_timeout_s, max_tracking_err_rad, stop_requested
                )
                max_err_seen = max(max_err_seen, err)
                max_tau_seen = max(max_tau_seen, max_tau)
                if max_tau >= sustained_torque_limit_nm:
                    high_tau_streak += 1
                    if aborted_reason is None and high_tau_streak >= sustained_torque_max_steps:
                        aborted_reason = f"Sustained high torque >= {sustained_torque_limit_nm:.1f} Nm for {high_tau_streak} steps"
                else:
                    high_tau_streak = 0
                if aborted_reason is not None:
                    return

                tau_g = compute_left_arm_gravity_torque(dyn_model, dyn_data, dyn_q_adrs, dyn_dof_adrs, last_q)
                _send_arm_command(arm_dev, q_cmd, dq_cmd, tau_g, "mit_grav", kp_mit, kd_mit, vel_lim, acc_lim)
                if hand_dev is not None and grip_pos is not None:
                    hand_dev.motor_command(CommandType.POSITION, [grip_pos])
                    fresh_gq, gq_err = _read_fresh_gripper_q(hand_dev)
                    if gq_err is not None:
                        aborted_reason = gq_err
                        return
                    if fresh_gq is not None:
                        hand_last_fresh_q = fresh_gq
                        hand_last_fresh_t = now_t
                        hand_fresh_in_stage += 1
                        if check_carried_grasp and (fresh_gq < 0.08 or fresh_gq > 0.52):
                            grasp_verified = False
                            aborted_reason = (
                                f"Carried tissue pack lost/empty grasp mid-transport "
                                f"(q_grip={fresh_gq:.3f} rad outside [0.08, 0.52])"
                            )
                            return
                    elif (now_t - hand_last_fresh_t) > stale_timeout_s:
                        aborted_reason = (
                            f"Gripper feedback frozen/stale for {now_t - hand_last_fresh_t:.3f}s "
                            f"(> {stale_timeout_s:.3f}s)"
                        )
                        return

                sleep_s = t_seg + (k + 1) * dt - clock_fn()
                if sleep_s > 0:
                    sleep_fn(sleep_s)

        if aborted_reason is None:
            # Consume initial open gripper reading and reject non-finite startup telemetry
            _, init_gq_err = _read_fresh_gripper_q(hand_dev)
            if init_gq_err is not None:
                aborted_reason = init_gq_err
            hand_last_fresh_t = clock_fn()
            if aborted_reason is None:
                if send_gate is not None:
                    send_gate.allow_execution()
                arm_dev.start()
                if hasattr(hand_dev, "set_pos_torque"):
                    hand_dev.set_pos_torque(2.5)
                hand_dev.motor_command(CommandType.POSITION, [GRIP_OPEN_RAD])

            # 1) Unfold -> Hover (verify gripper is within MuJoCo-verified GRIP_MAX_OPEN_ALLOWED_RAD before descend!)
            _hold_and_stream(np.repeat(q_live[None, :], 25, axis=0), GRIP_OPEN_RAD)
            _hold_and_stream(segments["unfold"], GRIP_OPEN_RAD)
            _hold_and_stream(segments["to_hover"], GRIP_OPEN_RAD)

            if aborted_reason is None:
                if hand_fresh_in_stage == 0 or hand_last_fresh_q is None:
                    aborted_reason = "No fresh gripper telemetry at hover before descend; refusing descend."
                elif hand_last_fresh_q > GRIP_MAX_OPEN_ALLOWED_RAD:
                    aborted_reason = (
                        f"Gripper not open within verified bound at hover before descend "
                        f"(q_grip={hand_last_fresh_q:.3f} rad > {GRIP_MAX_OPEN_ALLOWED_RAD:.2f}); refusing descend."
                    )

            # 2) Descend to grasp pose (open jaws)
            if aborted_reason is None:
                _hold_and_stream(segments["descend_grasp"], GRIP_OPEN_RAD)

            if aborted_reason is None:
                if hand_fresh_in_stage == 0 or hand_last_fresh_q is None:
                    aborted_reason = "No fresh gripper telemetry before close; aborting."
                else:
                    q_grip_before_close = hand_last_fresh_q
                    if q_grip_before_close > GRIP_MAX_OPEN_ALLOWED_RAD:
                        aborted_reason = (
                            f"Gripper not open within verified bound before close "
                            f"(q_grip={q_grip_before_close:.3f} rad > {GRIP_MAX_OPEN_ALLOWED_RAD:.2f})."
                        )

            # 3) Close gripper while holding grasp pose (1.2 s) & require continuous FRESH closure confirmation
            if aborted_reason is None:
                q_grasp_hold = np.repeat(segments["descend_grasp"][-1][None, :], 120, axis=0)
                _hold_and_stream(q_grasp_hold, GRIP_CLOSE_RAD)

            if aborted_reason is None:
                if (
                    hand_fresh_in_stage == 0
                    or hand_last_fresh_q is None
                    or (clock_fn() - hand_last_fresh_t) > stale_timeout_s
                ):
                    aborted_reason = "Gripper feedback lost during close (stale telemetry); aborting lift."
                else:
                    q_grip_after_close = hand_last_fresh_q
                    if (q_grip_after_close - q_grip_before_close) < 0.05:
                        aborted_reason = (
                            f"Gripper did not close from open pose ({q_grip_before_close:.3f} -> {q_grip_after_close:.3f} rad); aborting lift."
                        )
                    elif q_grip_after_close < 0.08:
                        aborted_reason = f"Gripper failed to close (q_grip={q_grip_after_close:.3f} rad < 0.08); aborting lift."
                    elif q_grip_after_close > 0.52:
                        aborted_reason = f"Gripper closed empty (q_grip={q_grip_after_close:.3f} rad > 0.52); aborting lift."
                    else:
                        grasp_verified = True

            # 4) Lift -> Move to Place (continuously verify carried grasp!) -> Open -> Retreat -> Return to Rest
            if aborted_reason is None and grasp_verified:
                _hold_and_stream(segments["lift"], GRIP_CLOSE_RAD, check_carried_grasp=True)
                _hold_and_stream(segments["to_place_hover"], GRIP_CLOSE_RAD, check_carried_grasp=True)
                _hold_and_stream(segments["descend_place"], GRIP_CLOSE_RAD, check_carried_grasp=True)

                q_grip_before_open = hand_last_fresh_q
                if aborted_reason is None:
                    q_place_hold = np.repeat(segments["descend_place"][-1][None, :], 100, axis=0)
                    _hold_and_stream(q_place_hold, GRIP_OPEN_RAD)

                if aborted_reason is None:
                    if (
                        hand_fresh_in_stage == 0
                        or hand_last_fresh_q is None
                        or (clock_fn() - hand_last_fresh_t) > stale_timeout_s
                    ):
                        aborted_reason = "Gripper feedback lost during open at place pose; refusing retreat."
                    else:
                        q_grip_after_open = hand_last_fresh_q
                        open_delta = (q_grip_before_open - q_grip_after_open) if q_grip_before_open is not None else 0.0
                        if q_grip_after_open > GRIP_MAX_OPEN_ALLOWED_RAD or open_delta < 0.03:
                            aborted_reason = (
                                f"Gripper stuck/did not open within verified bound at place pose "
                                f"({q_grip_before_open:.3f} -> {q_grip_after_open:.3f} rad); refusing retreat."
                            )
                        else:
                            release_verified = True

                if aborted_reason is None and release_verified:
                    _hold_and_stream(segments["retreat_place"], GRIP_OPEN_RAD)
                    _hold_and_stream(segments["return_ready"], GRIP_OPEN_RAD)
                    _hold_and_stream(segments["return_rest"], GRIP_OPEN_RAD)

        q_end = _safe_read_q(arm_dev, False)
        final_q = q_end if q_end is not None else last_q
        final_feedback_fresh = (clock_fn() - last_fresh_t) <= stale_timeout_s
    finally:
        signal.signal(signal.SIGINT, old_handler)
        if send_gate is not None:
            send_gate.latch_stop()
        if arm_dev is not None:
            try:
                arm_dev.stop()
            except Exception:
                pass
        try:
            api.close()
        except Exception:
            pass

    end_seg = segments.get("return_rest", segments["unfold"])
    expected_end = end_seg[-1]
    final_err = float(np.max(np.abs(final_q - expected_end))) if final_q is not None else None
    if aborted_reason is None and not final_feedback_fresh:
        aborted_reason = "Final telemetry stale at completion"
    if aborted_reason is None and (final_err is None or final_err > goal_tol_rad):
        aborted_reason = f"Return-to-rest goal not reached: final_err={final_err:.4f} rad > goal_tol={goal_tol_rad:.4f} rad"

    completed = (
        aborted_reason is None
        and grasp_verified
        and release_verified
        and final_feedback_fresh
        and final_err is not None
        and final_err <= goal_tol_rad
    )

    return {
        "completed": bool(completed),
        "aborted_reason": aborted_reason,
        "grasp_verified": grasp_verified,
        "release_verified": release_verified,
        "q_grip_before_close_rad": round(q_grip_before_close, 4) if q_grip_before_close is not None else None,
        "q_grip_after_close_rad": round(q_grip_after_close, 4) if q_grip_after_close is not None else None,
        "q_grip_after_open_rad": round(q_grip_after_open, 4) if q_grip_after_open is not None else None,
        "final_feedback_fresh": bool(final_feedback_fresh),
        "final_err_to_goal_rad": round(final_err, 4) if final_err is not None else None,
        "max_tracking_err_rad": round(max_err_seen, 4),
        "max_torque_nm": round(max_tau_seen, 3),
        "gripper_detected": hand_dev is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: Left-arm MIT+gravity tissue-pack pick-and-place.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Plan and verify all 9 segments in MuJoCo only.")
    mode.add_argument("--execute", action="store_true", help="Execute on real left arm (port 8439).")
    parser.add_argument("--offline-step1", type=Path, default=None, help="Offline step1 report (dry-run only).")
    parser.add_argument("--offline-perception", type=Path, default=None, help="Offline perception report (dry-run only).")
    args = parser.parse_args()

    if args.execute and (args.offline_step1 is not None or args.offline_perception is not None):
        raise RuntimeError("Refusing --execute with offline inputs; live dual-arm and top-D435 perception are required.")

    out_dir = ROOT / "docs" / "reports" / "step3_snapshots"
    report_path = ROOT / "docs" / "reports" / "step3_pick_place_report.json"

    if args.offline_perception is not None:
        perception = json.loads(args.offline_perception.read_text(encoding="utf-8"))["perception"]
    else:
        print("1) Detecting tissue pack & tabletop from live top D435...")
        perception = detect_target_from_top_d435(out_dir)

    if args.offline_step1 is not None:
        s1 = json.loads(args.offline_step1.read_text(encoding="utf-8"))
        q_meas = np.asarray(s1["left_arm"]["arm"]["positions_rad"], dtype=np.float64)
        right_q_meas = np.asarray(s1["right_arm"]["arm"]["positions_rad"], dtype=np.float64)
    else:
        print("2) Reading live dual-arm joint angles via zero-write passive WS...")
        q_meas, right_q_meas = read_dual_arms_passive_q()

    print("3) Planning & verifying all 9 pick-and-place segments in MuJoCo...")
    plan = plan_and_verify_step3(q_meas=q_meas, right_q_meas=right_q_meas, perception=perception)
    segments = plan.pop("_segments")
    print(
        f"   safe_to_execute={plan['safe_to_execute']}, frames={plan['total_frames_checked']}, "
        f"min_table_clearance={plan['min_geom_clearance_table_m']} m, "
        f"min_obj_clearance={plan['min_geom_clearance_object_m']} m, "
        f"min_carried_clearance={plan['min_carried_object_clearance_m']} m, blocking={plan['blocking_reasons']}"
    )

    exec_res = None
    if args.execute:
        if not plan["safe_to_execute"]:
            raise RuntimeError(f"Refusing to execute unsafe Step 3 plan: {plan['blocking_reasons']}")
        print("4) EXECUTING Step 3 pick-and-place on real left arm (port 8439)...")
        exec_res = execute_step3_pick_place(segments)
        print(f"   exec_res={exec_res}")

    report = {
        "mode": "execute" if args.execute else ("offline_dry_run" if args.offline_step1 else "dry_run"),
        "perception": perception,
        "plan": plan,
        "execution": exec_res,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
