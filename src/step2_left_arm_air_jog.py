"""Step 2: Left-arm safe unfold + collision-checked low-speed air hover above target.

Usage:
  1) Dry-run (perception + IK + collision verification, NO robot motion):
     /home/lakeside/.venvs/mujoco/bin/python src/step2_left_arm_air_jog.py --dry-run

  2) Execute on real left arm (port 8439, keep E-stop ready!):
     /home/lakeside/.venvs/mujoco/bin/python src/step2_left_arm_air_jog.py --execute
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import cv2
import hex_device
from hex_device import Arm, HexDeviceApi
import mink
import mujoco
import numpy as np
import pyrealsense2 as rs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ik_solver import FireflyDualIK, quintic_blend
from real_calibration import (
    CAMERA_INTRINSICS,
    CAMERA_SERIALS,
    LEFT_ARM_WS_URL,
    T_ARM1_BASE_FROM_TOP_D435,
    URDF_WORKCELL_TO_LEFT_BASE,
    transform_points,
)

# Safe unfolded ready posture for left arm (well inside joint limits, zero self-collision)
Q_LEFT_READY = np.array([0.0, -0.75, 1.75, 0.0, 0.57, 0.0], dtype=np.float64)


def detect_target_from_top_d435(out_dir: Path) -> dict:
    """Capture top D435 RGB-D and locate the highest tabletop object (tissue pack) in workcell_base."""
    out_dir.mkdir(parents=True, exist_ok=True)
    serial = CAMERA_SERIALS["top_d435"]
    intr = CAMERA_INTRINSICS["top_d435"]

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

    try:
        profile = pipe.start(cfg)
        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        align = rs.align(rs.stream.color)
        frames = None
        for _ in range(12):
            frames = pipe.wait_for_frames(5000)
        aligned = align.process(frames)
        color_img = np.asanyarray(aligned.get_color_frame().get_data())
        depth_m = np.asanyarray(aligned.get_depth_frame().get_data()).astype(np.float64) * depth_scale
    finally:
        try:
            pipe.stop()
        except Exception:
            pass

    h, w = depth_m.shape
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    valid = (depth_m > 0.30) & (depth_m < 1.20)
    u_v = u_grid[valid].astype(np.float64)
    v_v = v_grid[valid].astype(np.float64)
    z_v = depth_m[valid]

    x_c = (u_v - intr["cx"]) * z_v / intr["fx"]
    y_c = (v_v - intr["cy"]) * z_v / intr["fy"]
    pts_cam = np.stack([x_c, y_c, z_v], axis=1)
    pts_arm1 = transform_points(T_ARM1_BASE_FROM_TOP_D435, pts_cam)
    pts_wc = pts_arm1 + URDF_WORKCELL_TO_LEFT_BASE[None, :]

    # Map workcell Z back to full image for clean 2D segmentation
    z_wc_img = np.full((h, w), np.nan, dtype=np.float64)
    z_wc_img[valid] = pts_wc[:, 2]

    # Estimate table surface z_table from 15th percentile of central region (below objects)
    roi_z = z_wc_img[60:420, 60:580]
    roi_valid = roi_z[np.isfinite(roi_z)]
    z_table = float(np.percentile(roi_valid, 15))

    # Segment tabletop object: pixels 2.5 cm to 18 cm above table inside table ROI
    obj_mask = np.zeros((h, w), dtype=np.uint8)
    cond = (
        np.isfinite(z_wc_img)
        & (z_wc_img > z_table + 0.025)
        & (z_wc_img < z_table + 0.18)
    )
    obj_mask[cond] = 255
    # Restrict to workspace ROI (avoid top edge cables)
    obj_mask[:70, :] = 0
    obj_mask[440:, :] = 0
    obj_mask[:, :40] = 0
    obj_mask[:, 600:] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, kernel)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)
    if num_labels <= 1:
        raise RuntimeError("No tabletop object detected above z_table + 2.5 cm in top D435 image.")

    # Pick largest connected component
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    comp_mask = labels == best_idx

    # Gather 3D points of the detected object
    ys, xs = np.where(comp_mask)
    z_obj = depth_m[ys, xs]
    ok = (z_obj > 0.30) & (z_obj < 1.20)
    xs, ys, z_obj = xs[ok].astype(np.float64), ys[ok].astype(np.float64), z_obj[ok]
    pts_obj_cam = np.stack(
        [(xs - intr["cx"]) * z_obj / intr["fx"], (ys - intr["cy"]) * z_obj / intr["fy"], z_obj],
        axis=1,
    )
    pts_obj_arm1 = transform_points(T_ARM1_BASE_FROM_TOP_D435, pts_obj_cam)
    pts_obj_wc = pts_obj_arm1 + URDF_WORKCELL_TO_LEFT_BASE[None, :]

    center_wc = np.median(pts_obj_wc, axis=0)
    z_top_wc = float(np.percentile(pts_obj_wc[:, 2], 90))
    center_arm1 = center_wc - URDF_WORKCELL_TO_LEFT_BASE

    # Estimate principal axis in workcell XY via PCA
    xy_centered = pts_obj_wc[:, :2] - center_wc[:2]
    cov = np.cov(xy_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_xy = eigvecs[:, 1]
    minor_xy = eigvecs[:, 0]
    length_m = float(np.percentile(xy_centered @ major_xy, 95) - np.percentile(xy_centered @ major_xy, 5))
    width_m = float(np.percentile(xy_centered @ minor_xy, 95) - np.percentile(xy_centered @ minor_xy, 5))

    # Draw overlay
    vis = color_img.copy()
    x0, y0, bw, bh, area = stats[best_idx]
    cv2.rectangle(vis, (x0, y0), (x0 + bw, y0 + bh), (0, 255, 0), 2)
    cu, cv_px = int(np.median(xs)), int(np.median(ys))
    cv2.drawMarker(vis, (cu, cv_px), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    label = f"wc=({center_wc[0]:.3f},{center_wc[1]:.3f}) z_top={z_top_wc:.3f}m"
    cv2.putText(vis, label, (max(10, x0 - 40), max(25, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    overlay_path = out_dir / "step2_target_overlay.png"
    cv2.imwrite(str(overlay_path), vis)

    return {
        "z_table_wc_m": round(z_table, 4),
        "z_table_arm1_m": round(z_table - float(URDF_WORKCELL_TO_LEFT_BASE[2]), 4),
        "object_center_wc_m": [round(float(x), 4) for x in center_wc],
        "object_center_arm1_m": [round(float(x), 4) for x in center_arm1],
        "object_z_top_wc_m": round(z_top_wc, 4),
        "object_footprint_m": [round(length_m, 4), round(width_m, 4)],
        "major_axis_wc_xy": [round(float(major_xy[0]), 4), round(float(major_xy[1]), 4)],
        "minor_axis_wc_xy": [round(float(minor_xy[0]), 4), round(float(minor_xy[1]), 4)],
        "pixel_area": int(area),
        "overlay_path": str(overlay_path),
    }


def read_left_arm_q(timeout_s: float = 5.0) -> np.ndarray:
    """Read current 6-DOF left arm joint positions in read-only mode."""
    hex_device.set_log_level("WARNING")
    api = HexDeviceApi(ws_url=LEFT_ARM_WS_URL, control_hz=50, enable_kcp=True, local_port=0)
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            for dev in api.device_list:
                if isinstance(dev, Arm) and dev.has_new_data():
                    return np.asarray(dev.get_motor_positions(False), dtype=np.float64)
            time.sleep(0.05)
        raise TimeoutError("Timed out reading left arm joint positions.")
    finally:
        try:
            api.close()
        except Exception:
            pass


def build_quintic_segment(q0: np.ndarray, q1: np.ndarray, max_vel: float, dt: float) -> np.ndarray:
    """Build C^2 quintic joint-space trajectory respecting max_vel (rad/s)."""
    dq_max = float(np.max(np.abs(q1 - q0)))
    # Quintic peak derivative ds/du = 1.875
    duration = max(1.5, 1.875 * dq_max / max_vel)
    steps = max(2, int(np.ceil(duration / dt)))
    traj = np.zeros((steps + 1, len(q0)), dtype=np.float64)
    for k in range(steps + 1):
        s, _ = quintic_blend(k / steps)
        traj[k] = q0 + s * (q1 - q0)
    return traj


def plan_and_verify_step2(
    q_meas: np.ndarray,
    target_hover_wc: np.ndarray,
    z_table_wc: float,
    max_vel: float = 0.28,
    dt: float = 0.01,
) -> dict:
    """Plan Segment 1 (safe unfold) + Segment 2 (IK air hover) and verify in MuJoCo."""
    model = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "firefly_dual.xml"))
    data = mujoco.MjData(model)

    left_q_adrs = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"left_joint_{i}")])
        for i in range(1, 7)
    ]
    left_ranges = np.array(
        [
            model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"left_joint_{i}")]
            for i in range(1, 7)
        ],
        dtype=np.float64,
    )
    left_tcp_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_tcp_site")

    # Solve top-down hover IK for left_tcp_site using FireflyDualIK
    ik = FireflyDualIK(model, site_suffix="tcp_site")
    q_init = np.zeros(model.nq, dtype=np.float64)
    for i, adr in enumerate(left_q_adrs):
        q_init[adr] = Q_LEFT_READY[i]

    # Top-down orientation (yaw=+90 deg): TCP +Z straight down [0,0,-1], TCP +Y along +X
    # (aligns parallel gripper jaws across the 8.4 cm short edge & keeps J6 cable twist small ~0.86 rad)
    R_topdown = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    left_target = mink.SE3.from_rotation_and_translation(
        mink.SO3.from_matrix(R_topdown), target_hover_wc
    )
    q_sol, info = ik.solve_pose_global(
        q_seed=q_init,
        left_target=left_target,
        right_target=None,
        pos_tol_m=0.002,
        ori_tol_rad=0.02,
    )
    if not info.get("converged", False):
        raise RuntimeError(f"Left-arm hover IK did not converge: {info}")

    q_hover = np.array([q_sol[adr] for adr in left_q_adrs], dtype=np.float64)

    # Build smooth low-speed quintic trajectories
    seg1 = build_quintic_segment(q_meas, Q_LEFT_READY, max_vel=max_vel, dt=dt)
    seg2 = build_quintic_segment(Q_LEFT_READY, q_hover, max_vel=max_vel, dt=dt)
    full_traj = np.vstack([seg1, seg2[1:]])

    # Verify trajectory in MuJoCo (clearance above z_table, joint limits, collisions, velocity)
    min_tcp_z = 1e9
    max_seg_vel = float(np.max(np.abs(np.diff(full_traj, axis=0)) / dt))
    collisions_after_unfold = []
    out_of_limits_after_unfold = 0

    for k in range(0, len(full_traj), 2):
        q_k = full_traj[k]
        mujoco.mj_resetData(model, data)
        for i, adr in enumerate(left_q_adrs):
            data.qpos[adr] = q_k[i]
        mujoco.mj_forward(model, data)

        tcp_z = float(data.site_xpos[left_tcp_sid][2])
        min_tcp_z = min(min_tcp_z, tcp_z)

        # Skip the first 15% of Segment 1 while smoothly lifting off the folded mechanical stop
        if k > int(0.15 * len(seg1)):
            if np.any(q_k < left_ranges[:, 0] - 1e-4) or np.any(q_k > left_ranges[:, 1] + 1e-4):
                out_of_limits_after_unfold += 1
            for ci in range(data.ncon):
                con = data.contact[ci]
                if con.dist < -1e-3:
                    g1 = model.geom(con.geom1).name or f"g{con.geom1}"
                    g2 = model.geom(con.geom2).name or f"g{con.geom2}"
                    if g1.startswith("left_") or g2.startswith("left_"):
                        collisions_after_unfold.append((k, g1, g2, round(float(con.dist) * 1000.0, 2)))

    # Final FK at hover
    mujoco.mj_resetData(model, data)
    for i, adr in enumerate(left_q_adrs):
        data.qpos[adr] = q_hover[i]
    mujoco.mj_forward(model, data)
    hover_tcp_pos = data.site_xpos[left_tcp_sid].copy()
    hover_tcp_z_axis = data.site_xmat[left_tcp_sid].reshape(3, 3)[:, 2].copy()

    safe = (
        len(collisions_after_unfold) == 0
        and out_of_limits_after_unfold == 0
        and (min_tcp_z - z_table_wc) >= 0.08
        and max_seg_vel <= max_vel + 0.05
    )

    return {
        "safe_to_execute": bool(safe),
        "q_meas": [round(float(x), 5) for x in q_meas],
        "q_ready": [round(float(x), 5) for x in Q_LEFT_READY],
        "q_hover": [round(float(x), 5) for x in q_hover],
        "ik_left_pos_err_mm": round(float(info.get("left_pos_err_m", 0.0)) * 1000.0, 3),
        "ik_left_ori_err_rad": round(float(info.get("left_ori_err_rad", 0.0)), 4),
        "hover_tcp_wc_m": [round(float(x), 4) for x in hover_tcp_pos],
        "hover_tcp_z_axis": [round(float(x), 4) for x in hover_tcp_z_axis],
        "min_tcp_z_wc_m": round(min_tcp_z, 4),
        "min_clearance_above_table_m": round(min_tcp_z - z_table_wc, 4),
        "max_joint_vel_rad_s": round(max_seg_vel, 4),
        "seg1_duration_s": round((len(seg1) - 1) * dt, 2),
        "seg2_duration_s": round((len(seg2) - 1) * dt, 2),
        "total_duration_s": round((len(full_traj) - 1) * dt, 2),
        "collisions_after_unfold": collisions_after_unfold[:10],
        "out_of_limits_after_unfold": int(out_of_limits_after_unfold),
        "_traj": full_traj,
    }


def execute_left_arm_trajectory(
    traj: np.ndarray,
    dt: float = 0.01,
    max_tracking_err_rad: float = 0.25,
) -> dict:
    """Stream verified trajectory to real left arm (port 8439) with safety watchdog."""
    stop_requested = False

    def _sigint_handler(_sig, _frame):
        nonlocal stop_requested
        stop_requested = True

    old_handler = signal.signal(signal.SIGINT, _sigint_handler)
    hex_device.set_log_level("WARNING")
    api = HexDeviceApi(ws_url=LEFT_ARM_WS_URL, control_hz=200, enable_kcp=True, local_port=0)

    max_err_seen = 0.0
    max_torque_seen = 0.0
    aborted_reason = None
    completed_steps = 0
    final_q = None

    try:
        t0 = time.monotonic()
        arm_dev = None
        while time.monotonic() - t0 < 5.0:
            for dev in api.device_list:
                if isinstance(dev, Arm) and dev.has_new_data():
                    arm_dev = dev
                    break
            if arm_dev is not None:
                break
            time.sleep(0.05)

        if arm_dev is None:
            raise TimeoutError("Could not connect to left arm for execution.")

        q_live = np.asarray(arm_dev.get_motor_positions(False), dtype=np.float64)
        start_diff = float(np.max(np.abs(q_live - traj[0])))
        if start_diff > 0.05:
            raise RuntimeError(f"Live arm moved since planning (diff={start_diff:.4f} rad > 0.05). Aborting.")

        # Acquire control and hold current position briefly before moving
        arm_dev.start()
        vel_lim = [0.6] * 6
        acc_lim = [1.2] * 6
        for _ in range(30):
            arm_dev.joint_position_control(q_live.tolist(), vel_lim, acc_lim)
            time.sleep(0.01)

        t_start = time.perf_counter()
        for k in range(len(traj)):
            if stop_requested:
                aborted_reason = "User SIGINT"
                break
            if api.is_api_exit():
                aborted_reason = "HexDeviceApi exited"
                break

            q_cmd = traj[k]
            arm_dev.joint_position_control(q_cmd.tolist(), vel_lim, acc_lim)

            q_now = np.asarray(arm_dev.get_motor_positions(False), dtype=np.float64)
            tau_now = np.asarray(arm_dev.get_motor_torques(False), dtype=np.float64)
            err = float(np.max(np.abs(q_cmd - q_now)))
            max_err_seen = max(max_err_seen, err)
            max_torque_seen = max(max_torque_seen, float(np.max(np.abs(tau_now))))

            errs = arm_dev.get_motor_error_codes()
            if errs and any(e is not None for e in errs):
                aborted_reason = f"Motor error codes: {errs}"
                break
            if err > max_tracking_err_rad:
                aborted_reason = f"Tracking error {err:.4f} rad > {max_tracking_err_rad} rad"
                break

            completed_steps = k + 1
            target_t = t_start + (k + 1) * dt
            sleep_s = target_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

        # Hold final position for 1.0 s to let controller settle cleanly
        q_hold = (
            np.asarray(arm_dev.get_motor_positions(False), dtype=np.float64)
            if aborted_reason
            else traj[-1]
        )
        for _ in range(100):
            arm_dev.joint_position_control(q_hold.tolist(), vel_lim, acc_lim)
            time.sleep(0.01)
        final_q = np.asarray(arm_dev.get_motor_positions(False), dtype=np.float64)
        arm_dev.stop()
    finally:
        signal.signal(signal.SIGINT, old_handler)
        try:
            api.close()
        except Exception:
            pass

    return {
        "completed": aborted_reason is None,
        "aborted_reason": aborted_reason,
        "completed_steps": completed_steps,
        "total_steps": int(len(traj)),
        "max_tracking_err_rad": round(max_err_seen, 4),
        "max_torque_nm": round(max_torque_seen, 3),
        "final_q_rad": [round(float(x), 5) for x in final_q] if final_q is not None else None,
        "final_err_to_goal_rad": round(float(np.max(np.abs(final_q - traj[-1]))), 4) if final_q is not None else None,
    }


def capture_verification_snapshots(out_dir: Path) -> dict[str, str]:
    """Capture post-hover RGB snapshots from top_d435 and left_d405."""
    saved = {}
    for cam_name in ("top_d435", "left_d405"):
        serial = CAMERA_SERIALS[cam_name]
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        try:
            pipe.start(cfg)
            frames = None
            for _ in range(10):
                frames = pipe.wait_for_frames(5000)
            img = np.asanyarray(frames.get_color_frame().get_data())
            p = out_dir / f"step2_after_hover_{cam_name}.png"
            cv2.imwrite(str(p), img)
            saved[cam_name] = str(p)
        finally:
            try:
                pipe.stop()
            except Exception:
                pass
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: Left-arm safe unfold + low-speed air hover.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Plan and verify in MuJoCo only (no arm motion).")
    mode.add_argument("--execute", action="store_true", help="Execute verified trajectory on real left arm.")
    parser.add_argument("--hover-clearance", type=float, default=0.10, help="Hover height above object top (m).")
    parser.add_argument("--max-vel", type=float, default=0.28, help="Max joint speed (rad/s).")
    parser.add_argument("--target-xyz", type=float, nargs=3, default=None, help="Optional override hover XYZ in workcell_base (m).")
    args = parser.parse_args()

    out_dir = ROOT / "docs" / "reports" / "step2_snapshots"
    report_path = ROOT / "docs" / "reports" / "step2_air_jog_report.json"

    print("1) Detecting tabletop & target from top D435...")
    perception = detect_target_from_top_d435(out_dir)
    print(f"   z_table_wc={perception['z_table_wc_m']} m, obj_wc={perception['object_center_wc_m']}, z_top={perception['object_z_top_wc_m']} m")

    if args.target_xyz is not None:
        hover_wc = np.asarray(args.target_xyz, dtype=np.float64)
    else:
        c = perception["object_center_wc_m"]
        hover_wc = np.array([c[0], c[1], perception["object_z_top_wc_m"] + args.hover_clearance], dtype=np.float64)

    print(f"2) Reading live left arm q_meas (read-only)...")
    q_meas = read_left_arm_q()

    print(f"3) Planning & verifying trajectory to hover_wc={hover_wc.round(4).tolist()}...")
    plan = plan_and_verify_step2(
        q_meas=q_meas,
        target_hover_wc=hover_wc,
        z_table_wc=perception["z_table_wc_m"],
        max_vel=args.max_vel,
    )
    traj = plan.pop("_traj")
    print(f"   safe_to_execute={plan['safe_to_execute']}, total_duration={plan['total_duration_s']} s, min_clearance={plan['min_clearance_above_table_m']} m")

    exec_res = None
    post_snaps = None
    if args.execute:
        if not plan["safe_to_execute"]:
            raise RuntimeError("Plan failed safety verification — refusing to execute!")
        print("4) EXECUTING verified low-speed trajectory on real left arm (port 8439)...")
        exec_res = execute_left_arm_trajectory(traj)
        print(f"   completed={exec_res['completed']}, max_tracking_err={exec_res['max_tracking_err_rad']} rad, max_torque={exec_res['max_torque_nm']} N·m")
        print("5) Capturing post-hover verification snapshots...")
        post_snaps = capture_verification_snapshots(out_dir)

    report = {
        "mode": "execute" if args.execute else "dry_run",
        "target_hover_wc_m": [round(float(x), 4) for x in hover_wc],
        "perception": perception,
        "plan": plan,
        "execution": exec_res,
        "post_hover_snapshots": post_snaps,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved report: {report_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
