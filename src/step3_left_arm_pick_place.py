"""Step 3: Left-arm gravity-compensated MIT pick-and-place for tissue pack.

Reuses Step 2's GuardedHexDeviceApi (t=0 send gate + frame-0 E-Stop tracker),
zero-write passive dual-arm state reading, and full-frame k=0..N MuJoCo safety check.

Stages:
  1. q_meas -> q_ready (safe unfold)
  2. q_ready -> q_hover (10 cm above tissue pack top, jaws open & aligned to short edge)
  3. q_hover -> q_grasp (descend to middle of tissue pack, >= 4.5 cm above table)
  4. Gripper close (SdtHandGp100 -> 0.02 rad)
  5. q_grasp -> q_lift (+10 cm vertical lift)
  6. q_lift -> q_place_hover (offset in +Y) -> q_place -> Gripper open (0.55 rad) -> q_place_hover
  7. q_place_hover -> q_ready -> q_meas (return to safe rest pose before arm.stop())
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

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
    _install_parking_stop_tracker,
    _install_safe_send_gate,
    _is_parking_stop_active,
    _safe_read_q,
    _send_arm_command,
    build_quintic_segment,
    build_verification_model,
    capture_verification_snapshots,
    compute_left_arm_gravity_torque,
    create_guarded_hex_device_api,
    detect_target_from_top_d435,
    read_dual_arms_passive_q,
)

GRIP_OPEN_RAD = 0.55
GRIP_CLOSE_RAD = 0.02


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


def plan_and_verify_step3(
    q_meas: np.ndarray,
    right_q_meas: np.ndarray,
    perception: dict[str, Any],
    place_offset_y_m: float = 0.14,
    hover_clearance_m: float = 0.08,
    max_vel: float = 0.25,
    dt: float = 0.01,
) -> dict[str, Any]:
    """Plan all 7 pick-and-place segments and verify EVERY frame in MuJoCo."""
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
    # TCP +Y (jaw opening axis) along minor axis (perpendicular to major_xy), TCP +Z straight down
    z_tcp = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    y_tcp = np.array([-major_xy[1], major_xy[0], 0.0], dtype=np.float64)
    if y_tcp[0] < 0:
        y_tcp = -y_tcp  # keep wrist cable twist minimal
    x_tcp = np.cross(y_tcp, z_tcp)
    R_grasp = np.column_stack([x_tcp, y_tcp, z_tcp])

    base_model = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "firefly_dual.xml"))
    verify_model = build_verification_model(z_table_wc=z_table, perception=None)  # table collider; object is grasped
    verify_data = mujoco.MjData(verify_model)

    left_q_adrs = [
        int(verify_model.jnt_qposadr[mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_JOINT, f"left_joint_{i}")])
        for i in range(1, 7)
    ]
    right_q_adrs = [
        int(verify_model.jnt_qposadr[mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_JOINT, f"right_joint_{i}")])
        for i in range(1, 7)
    ]
    left_ranges = np.array(
        [
            verify_model.jnt_range[mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_JOINT, f"left_joint_{i}")]
            for i in range(1, 7)
        ],
        dtype=np.float64,
    )
    table_gid = mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_GEOM, "table_surface_col")
    left_col_gids = [
        gid
        for gid in range(verify_model.ngeom)
        if (verify_model.geom_contype[gid] != 0 or verify_model.geom_conaffinity[gid] != 0)
        and verify_model.body(verify_model.geom_bodyid[gid]).name.startswith("left_")
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

    # Verify EVERY frame across ALL 9 segments
    total_frames = 0
    min_table_dist = 1e9
    limit_violations = 0
    collisions = []

    for seg_name, seg in segments.items():
        for k in range(len(seg)):
            total_frames += 1
            q_k = seg[k]
            mujoco.mj_resetData(verify_model, verify_data)
            for i, a in enumerate(left_q_adrs):
                verify_data.qpos[a] = q_k[i]
            for i, a in enumerate(right_q_adrs):
                verify_data.qpos[a] = right_q_meas[i]
            mujoco.mj_forward(verify_model, verify_data)

            if np.any(q_k < left_ranges[:, 0] - 1e-6) or np.any(q_k > left_ranges[:, 1] + 1e-6):
                limit_violations += 1
            for ci in range(verify_data.ncon):
                con = verify_data.contact[ci]
                if con.dist < -1e-4:
                    b1 = verify_model.body(verify_model.geom_bodyid[con.geom1]).name
                    b2 = verify_model.body(verify_model.geom_bodyid[con.geom2]).name
                    if b1.startswith("left_") or b2.startswith("left_"):
                        collisions.append((seg_name, k, b1, b2, round(float(-con.dist) * 1000.0, 2)))
            for gid in left_col_gids:
                d_tab = float(mujoco.mj_geomDistance(verify_model, verify_data, gid, table_gid, 1.0, None))
                min_table_dist = min(min_table_dist, d_tab)

    blocking: list[str] = []
    if limit_violations > 0:
        blocking.append(f"Joint limit violations on {limit_violations} frames")
    if collisions:
        blocking.append(f"Collisions detected: {collisions[0]}")
    if min_table_dist < 0.015:
        blocking.append(f"Min finger/arm clearance above table {min_table_dist:.4f} m < 0.0150 m")

    return {
        "safe_to_execute": len(blocking) == 0,
        "blocking_reasons": blocking,
        "total_frames_checked": total_frames,
        "min_geom_clearance_table_m": round(min_table_dist, 4),
        "p_hover_wc_m": [round(float(x), 4) for x in p_hover],
        "p_grasp_wc_m": [round(float(x), 4) for x in p_grasp],
        "p_place_wc_m": [round(float(x), 4) for x in p_place],
        "total_motion_duration_s": round(sum(len(s) - 1 for s in segments.values()) * dt, 2),
        "_segments": segments,
    }


def execute_step3_pick_place(
    segments: dict[str, np.ndarray],
    dt: float = 0.01,
    max_tracking_err_rad: float = 0.25,
    stale_timeout_s: float = 0.15,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute full gravity-compensated MIT pick-and-place with SdtHandGp100 gripper."""
    from hex_device import Arm, Hands, public_api_types_pb2
    from hex_device.motor_base import CommandType

    stop_requested = False

    def _sigint(_s, _f):
        nonlocal stop_requested
        stop_requested = True

    old_handler = signal.signal(signal.SIGINT, _sigint)
    api = create_guarded_hex_device_api(
        ws_url=LEFT_ARM_WS_URL, control_hz=200, enable_kcp=True, local_port=0
    )

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

    kp_mit = [60.0, 80.0, 80.0, 40.0, 25.0, 20.0]
    kd_mit = [3.0, 4.0, 4.0, 2.0, 1.2, 1.0]
    vel_lim = [0.6] * 6
    acc_lim = [1.2] * 6
    zeros6 = np.zeros(6, dtype=np.float64)

    arm_dev = None
    hand_dev = None
    send_gate: SafeArmSendGate | None = None
    aborted_reason: str | None = None
    max_err_seen = 0.0
    max_tau_seen = 0.0

    try:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 5.0:
            for dev in api.device_list:
                if isinstance(dev, Arm) and dev.has_new_data():
                    arm_dev = dev
                elif isinstance(dev, Hands) and dev.has_new_data():
                    hand_dev = dev
            if hand_dev is None:
                opt = api.find_optional_device_by_robot_type(
                    public_api_types_pb2.SecondaryDeviceType.SdtHandGp100
                )
                if opt:
                    hand_dev = opt[0] if isinstance(opt, list) else opt
            if arm_dev is not None:
                break
            time.sleep(0.05)

        if arm_dev is None:
            raise TimeoutError("Could not connect to left arm.")

        send_gate = getattr(api, "send_gate", None) or _install_safe_send_gate(arm_dev, start_blocked=True)
        _install_parking_stop_tracker(arm_dev, send_gate=send_gate)

        ps_present = bool(getattr(arm_dev, "_parking_stop_present", False))
        ps_detail = arm_dev.get_parking_stop_detail()
        if ps_present or _is_parking_stop_active(ps_detail):
            aborted_reason = f"Pre-start parking stop active: {ps_detail}"

        q_live = _safe_read_q(arm_dev, True)
        if q_live is None:
            raise RuntimeError("Arm queue empty at startup.")
        last_q = q_live.copy()
        q0_plan = segments["unfold"][0]
        if aborted_reason is None and float(np.max(np.abs(q_live - q0_plan))) > 0.05:
            aborted_reason = "Live arm moved since planning."

        last_fresh_t = time.perf_counter()

        def _hold_and_stream(traj: np.ndarray, grip_pos: float | None = None) -> None:
            nonlocal aborted_reason, last_q, last_fresh_t, max_err_seen, max_tau_seen
            if aborted_reason is not None:
                return
            t_seg = time.perf_counter()
            for k in range(len(traj)):
                now_t = time.perf_counter()
                q_cmd = traj[k]
                dq_cmd = (traj[min(k + 1, len(traj) - 1)] - traj[max(k - 1, 0)]) / (
                    dt if k == 0 or k == len(traj) - 1 else 2.0 * dt
                )
                aborted_reason, last_q, err, max_tau, last_fresh_t = _check_arm_fault(
                    api, arm_dev, q_cmd, last_q, now_t, last_fresh_t, stale_timeout_s, max_tracking_err_rad, stop_requested
                )
                max_err_seen = max(max_err_seen, err)
                max_tau_seen = max(max_tau_seen, max_tau)
                if aborted_reason is not None:
                    return
                tau_g = compute_left_arm_gravity_torque(dyn_model, dyn_data, dyn_q_adrs, dyn_dof_adrs, last_q)
                _send_arm_command(arm_dev, q_cmd, dq_cmd, tau_g, "mit_grav", kp_mit, kd_mit, vel_lim, acc_lim)
                if hand_dev is not None and grip_pos is not None:
                    hand_dev.motor_command(CommandType.POSITION, [grip_pos])
                sleep_s = t_seg + (k + 1) * dt - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)

        if aborted_reason is None:
            send_gate.allow_execution()
            arm_dev.start()
            if hand_dev is not None:
                hand_dev.set_pos_torque(2.5)
                hand_dev.motor_command(CommandType.POSITION, [GRIP_OPEN_RAD])

            # 1) Unfold -> Hover -> Descend (open jaws)
            _hold_and_stream(np.repeat(q_live[None, :], 25, axis=0), GRIP_OPEN_RAD)
            _hold_and_stream(segments["unfold"], GRIP_OPEN_RAD)
            _hold_and_stream(segments["to_hover"], GRIP_OPEN_RAD)
            _hold_and_stream(segments["descend_grasp"], GRIP_OPEN_RAD)

            # 2) Close gripper while holding grasp pose (1.2 s)
            q_grasp_hold = np.repeat(segments["descend_grasp"][-1][None, :], 120, axis=0)
            _hold_and_stream(q_grasp_hold, GRIP_CLOSE_RAD)

            # 3) Lift -> Move to Place -> Open -> Retreat -> Safe Return to Rest
            _hold_and_stream(segments["lift"], GRIP_CLOSE_RAD)
            _hold_and_stream(segments["to_place_hover"], GRIP_CLOSE_RAD)
            _hold_and_stream(segments["descend_place"], GRIP_CLOSE_RAD)
            q_place_hold = np.repeat(segments["descend_place"][-1][None, :], 100, axis=0)
            _hold_and_stream(q_place_hold, GRIP_OPEN_RAD)
            _hold_and_stream(segments["retreat_place"], GRIP_OPEN_RAD)
            _hold_and_stream(segments["return_ready"], GRIP_OPEN_RAD)
            _hold_and_stream(segments["return_rest"], GRIP_OPEN_RAD)
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

    return {
        "completed": aborted_reason is None,
        "aborted_reason": aborted_reason,
        "max_tracking_err_rad": round(max_err_seen, 4),
        "max_torque_nm": round(max_tau_seen, 3),
        "gripper_detected": hand_dev is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: Left-arm MIT+gravity tissue-pack pick-and-place.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Plan and verify all 9 segments in MuJoCo only.")
    mode.add_argument("--execute", action="store_true", help="Execute on real left arm (port 8439).")
    args = parser.parse_args()

    out_dir = ROOT / "docs" / "reports" / "step3_snapshots"
    report_path = ROOT / "docs" / "reports" / "step3_pick_place_report.json"

    print("1) Detecting tissue pack & tabletop from live top D435...")
    perception = detect_target_from_top_d435(out_dir)
    print("2) Reading live dual-arm joint angles via zero-write passive WS...")
    q_meas, right_q_meas = read_dual_arms_passive_q()
    print("3) Planning & verifying all 9 pick-and-place segments in MuJoCo...")
    plan = plan_and_verify_step3(q_meas=q_meas, right_q_meas=right_q_meas, perception=perception)
    segments = plan.pop("_segments")
    print(
        f"   safe_to_execute={plan['safe_to_execute']}, frames={plan['total_frames_checked']}, "
        f"min_table_clearance={plan['min_geom_clearance_table_m']} m, blocking={plan['blocking_reasons']}"
    )

    exec_res = None
    if args.execute:
        if not plan["safe_to_execute"]:
            raise RuntimeError(f"Refusing to execute unsafe Step 3 plan: {plan['blocking_reasons']}")
        print("4) EXECUTING Step 3 pick-and-place on real left arm (port 8439)...")
        exec_res = execute_step3_pick_place(segments, out_dir=out_dir)
        print(f"   exec_res={exec_res}")

    report = {
        "mode": "execute" if args.execute else "dry_run",
        "perception": perception,
        "plan": plan,
        "execution": exec_res,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
