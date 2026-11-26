"""Step 2: Left-arm safe unfold + full-frame dual-arm & obstacle-checked air hover.

Safety & audit guarantees:
1. Pure offline dry-run (--offline-step1 / --offline-perception) never opens a network
   socket or camera. Live read-only mode uses src/readonly_ws_client.py (zero uplink
   WebSocket sends — never instantiates HexDeviceApi background tasks).
2. Trajectory verification checks EVERY frame k = 0 .. N (zero skipped frames) with:
   - Live/measured right-arm joint angles written into MuJoCo (never assumed zero),
   - Dynamic tabletop collider at z_table_wc,
   - Dynamic 3D oriented target-object (tissue pack) collider with safety margins,
   - Full left-arm collision-geom clearance (all links, fingers, wrist D405 bracket)
     against self, workcell, right arm, tabletop, and target object.
   Any limit violation or collision on ANY frame (including k = 0) sets
   safe_to_execute = False with explicit blocking_reasons.
3. Execution loop enforces immediate stop on fault/SIGINT/parking-stop/stale feedback
   (0 commands sent after fault detection, immediate arm.stop()), plus strict
   goal-reached & fresh-telemetry gating for completed = True.
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

import cv2
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
    RIGHT_ARM_WS_URL,
    T_ARM1_BASE_FROM_TOP_D435,
    URDF_WORKCELL_TO_LEFT_BASE,
    transform_points,
)
from readonly_ws_client import read_arm_passive_readonly

# Safe unfolded ready posture for left arm (well inside joint limits, zero self-collision)
Q_LEFT_READY = np.array([0.0, -0.75, 1.75, 0.0, 0.57, 0.0], dtype=np.float64)


def detect_target_from_top_d435(out_dir: Path) -> dict[str, Any]:
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

    z_wc_img = np.full((h, w), np.nan, dtype=np.float64)
    z_wc_img[valid] = pts_wc[:, 2]

    roi_z = z_wc_img[60:420, 60:580]
    roi_valid = roi_z[np.isfinite(roi_z)]
    z_table = float(np.percentile(roi_valid, 15))

    obj_mask = np.zeros((h, w), dtype=np.uint8)
    cond = (
        np.isfinite(z_wc_img)
        & (z_wc_img > z_table + 0.025)
        & (z_wc_img < z_table + 0.18)
    )
    obj_mask[cond] = 255
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

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    comp_mask = labels == best_idx

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

    xy_centered = pts_obj_wc[:, :2] - center_wc[:2]
    cov = np.cov(xy_centered.T)
    _, eigvecs = np.linalg.eigh(cov)
    major_xy = eigvecs[:, 1]
    minor_xy = eigvecs[:, 0]
    length_m = float(np.percentile(xy_centered @ major_xy, 95) - np.percentile(xy_centered @ major_xy, 5))
    width_m = float(np.percentile(xy_centered @ minor_xy, 95) - np.percentile(xy_centered @ minor_xy, 5))

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


def read_dual_arms_passive_q(timeout_s: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Read both left and right 6-DOF joint positions via strictly zero-write passive WS."""
    left_res = read_arm_passive_readonly("left_arm", LEFT_ARM_WS_URL, timeout_s=timeout_s)
    if not left_res["connected"] or not left_res["arm"]:
        raise RuntimeError(f"Failed passive read of left arm: {left_res['error']}")
    right_res = read_arm_passive_readonly("right_arm", RIGHT_ARM_WS_URL, timeout_s=timeout_s)
    if not right_res["connected"] or not right_res["arm"]:
        raise RuntimeError(f"Failed passive read of right arm: {right_res['error']}")
    return (
        np.asarray(left_res["arm"]["positions_rad"], dtype=np.float64),
        np.asarray(right_res["arm"]["positions_rad"], dtype=np.float64),
    )


def build_quintic_segment(q0: np.ndarray, q1: np.ndarray, max_vel: float, dt: float) -> np.ndarray:
    """Build C^2 quintic joint-space trajectory respecting max_vel (rad/s)."""
    dq_max = float(np.max(np.abs(q1 - q0)))
    duration = max(1.5, 1.875 * dq_max / max_vel)
    steps = max(2, int(np.ceil(duration / dt)))
    traj = np.zeros((steps + 1, len(q0)), dtype=np.float64)
    for k in range(steps + 1):
        s, _ = quintic_blend(k / steps)
        traj[k] = q0 + s * (q1 - q0)
    return traj


def build_verification_model(
    z_table_wc: float,
    perception: dict[str, Any] | None = None,
    obj_xy_margin_m: float = 0.010,
    obj_z_margin_m: float = 0.005,
) -> mujoco.MjModel:
    """Compile firefly_dual.xml with dynamic tabletop and detected target-object colliders."""
    spec = mujoco.MjSpec.from_file(str(ROOT / "model" / "firefly_dual.xml"))

    # Add tabletop surface collider (top face at z_table_wc)
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

    # Add detected target object (tissue pack) as an oriented 3D box obstacle
    if perception is not None and "object_center_wc_m" in perception:
        cx, cy = float(perception["object_center_wc_m"][0]), float(perception["object_center_wc_m"][1])
        z_top = float(perception["object_z_top_wc_m"])
        length_m, width_m = perception.get("object_footprint_m", [0.18, 0.09])
        major_xy = perception.get("major_axis_wc_xy", [0.0, 1.0])
        yaw = math.atan2(float(major_xy[1]), float(major_xy[0]))
        half_z = max(0.005, 0.5 * (z_top - z_table_wc + obj_z_margin_m))
        center_z = z_table_wc + half_z
        spec.worldbody.add_geom(
            name="target_object_col",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[cx, cy, center_z],
            quat=[math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
            size=[0.5 * float(length_m) + obj_xy_margin_m, 0.5 * float(width_m) + obj_xy_margin_m, half_z],
            group=3,
            contype=1,
            conaffinity=1,
            rgba=[0.9, 0.2, 0.2, 0.35],
        )

    return spec.compile()


def plan_and_verify_step2(
    q_meas: np.ndarray,
    right_q_meas: np.ndarray,
    target_hover_wc: np.ndarray,
    z_table_wc: float,
    perception: dict[str, Any] | None = None,
    max_vel: float = 0.28,
    dt: float = 0.01,
    min_table_clearance_m: float = 0.02,
    min_object_clearance_m: float = 0.02,
) -> dict[str, Any]:
    """Plan Segment 1 (unfold) + Segment 2 (IK air hover) and verify EVERY frame k=0..N."""
    q_meas = np.asarray(q_meas, dtype=np.float64)
    right_q_meas = np.asarray(right_q_meas, dtype=np.float64)

    base_model = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "firefly_dual.xml"))
    verify_model = build_verification_model(z_table_wc=z_table_wc, perception=perception)
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
    left_tcp_sid = mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_SITE, "left_tcp_site")
    table_gid = mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_GEOM, "table_surface_col")
    obj_gid = mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_col")

    left_col_gids = [
        gid
        for gid in range(verify_model.ngeom)
        if (verify_model.geom_contype[gid] != 0 or verify_model.geom_conaffinity[gid] != 0)
        and verify_model.body(verify_model.geom_bodyid[gid]).name.startswith("left_")
    ]

    # Solve top-down hover IK for left_tcp_site using FireflyDualIK
    ik = FireflyDualIK(base_model, site_suffix="tcp_site")
    q_init = np.zeros(base_model.nq, dtype=np.float64)
    for i, adr in enumerate(left_q_adrs):
        q_init[adr] = Q_LEFT_READY[i]
    for i, adr in enumerate(right_q_adrs):
        q_init[adr] = right_q_meas[i]

    # Top-down orientation (yaw=+90 deg): TCP +Z straight down [0,0,-1], TCP +Y along +X
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

    # Build smooth low-speed quintic trajectories (skip unfold segment if already at Q_LEFT_READY)
    if float(np.max(np.abs(q_meas - Q_LEFT_READY))) < 1e-3:
        seg1 = q_meas[None, :].copy()
        seg2 = build_quintic_segment(q_meas, q_hover, max_vel=max_vel, dt=dt)
        full_traj = seg2
    else:
        seg1 = build_quintic_segment(q_meas, Q_LEFT_READY, max_vel=max_vel, dt=dt)
        seg2 = build_quintic_segment(Q_LEFT_READY, q_hover, max_vel=max_vel, dt=dt)
        full_traj = np.vstack([seg1, seg2[1:]])

    # Verify EVERY SINGLE FRAME k = 0 .. len(full_traj) - 1 (stride 1, ZERO skipped frames!)
    min_tcp_z = 1e9
    min_geom_table_dist = 1e9
    min_geom_obj_dist = 1e9
    max_seg_vel = float(np.max(np.abs(np.diff(full_traj, axis=0)) / dt))
    limit_violations: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    max_penetration_mm = 0.0

    for k in range(len(full_traj)):
        q_k = full_traj[k]
        mujoco.mj_resetData(verify_model, verify_data)
        for i, adr in enumerate(left_q_adrs):
            verify_data.qpos[adr] = q_k[i]
        for i, adr in enumerate(right_q_adrs):
            verify_data.qpos[adr] = right_q_meas[i]
        mujoco.mj_forward(verify_model, verify_data)

        tcp_z = float(verify_data.site_xpos[left_tcp_sid][2])
        min_tcp_z = min(min_tcp_z, tcp_z)

        # Joint limit check on every frame including k=0
        for j in range(6):
            lo, hi = float(left_ranges[j, 0]), float(left_ranges[j, 1])
            if q_k[j] < lo - 1e-6 or q_k[j] > hi + 1e-6:
                over = float(lo - q_k[j]) if q_k[j] < lo else float(q_k[j] - hi)
                limit_violations.append(
                    {"frame": k, "t_s": round(k * dt, 3), "joint": f"left_joint_{j + 1}", "exceed_rad": round(over, 6)}
                )

        # Contact check on every frame including k=0
        for ci in range(verify_data.ncon):
            con = verify_data.contact[ci]
            if con.dist < -1e-4:
                g1 = verify_model.geom(con.geom1).name or f"g{con.geom1}"
                g2 = verify_model.geom(con.geom2).name or f"g{con.geom2}"
                b1 = verify_model.body(verify_model.geom_bodyid[con.geom1]).name
                b2 = verify_model.body(verify_model.geom_bodyid[con.geom2]).name
                if b1.startswith("left_") or b2.startswith("left_"):
                    pen_mm = round(float(-con.dist) * 1000.0, 3)
                    max_penetration_mm = max(max_penetration_mm, pen_mm)
                    if len(collisions) < 50:
                        collisions.append(
                            {"frame": k, "t_s": round(k * dt, 3), "geom1": g1, "geom2": g2, "penetration_mm": pen_mm}
                        )

        # Full left-arm geom distance to tabletop & target object
        for gid in left_col_gids:
            d_tab = float(mujoco.mj_geomDistance(verify_model, verify_data, gid, table_gid, 1.0, None))
            if d_tab < min_geom_table_dist:
                min_geom_table_dist = d_tab
            if obj_gid >= 0:
                d_obj = float(mujoco.mj_geomDistance(verify_model, verify_data, gid, obj_gid, 1.0, None))
                if d_obj < min_geom_obj_dist:
                    min_geom_obj_dist = d_obj

    # Final FK at hover
    mujoco.mj_resetData(verify_model, verify_data)
    for i, adr in enumerate(left_q_adrs):
        verify_data.qpos[adr] = q_hover[i]
    for i, adr in enumerate(right_q_adrs):
        verify_data.qpos[adr] = right_q_meas[i]
    mujoco.mj_forward(verify_model, verify_data)
    hover_tcp_pos = verify_data.site_xpos[left_tcp_sid].copy()
    hover_tcp_z_axis = verify_data.site_xmat[left_tcp_sid].reshape(3, 3)[:, 2].copy()

    blocking_reasons: list[str] = []
    if limit_violations:
        frames_viol = sorted({v["frame"] for v in limit_violations})
        blocking_reasons.append(
            f"Joint limit violation on {len(frames_viol)} frame(s) (first={frames_viol[0]}, last={frames_viol[-1]}, "
            f"max_exceed={max(v['exceed_rad'] for v in limit_violations):.5f} rad)"
        )
    if collisions:
        blocking_reasons.append(
            f"Collision/penetration detected on trajectory (max_penetration={max_penetration_mm:.3f} mm, "
            f"first_frame={collisions[0]['frame']}, pair={collisions[0]['geom1']} vs {collisions[0]['geom2']})"
        )
    if min_geom_table_dist < min_table_clearance_m:
        blocking_reasons.append(
            f"Min left-arm geom clearance to table ({min_geom_table_dist:.4f} m) < required {min_table_clearance_m:.4f} m"
        )
    if obj_gid >= 0 and min_geom_obj_dist < min_object_clearance_m:
        blocking_reasons.append(
            f"Min left-arm geom clearance to target object ({min_geom_obj_dist:.4f} m) < required {min_object_clearance_m:.4f} m"
        )
    if max_seg_vel > max_vel + 0.05:
        blocking_reasons.append(f"Max joint velocity {max_seg_vel:.4f} rad/s > limit {max_vel + 0.05:.4f} rad/s")

    return {
        "safe_to_execute": len(blocking_reasons) == 0,
        "blocking_reasons": blocking_reasons,
        "frames_checked": int(len(full_traj)),
        "frames_skipped": 0,
        "q_meas": [round(float(x), 5) for x in q_meas],
        "right_q_used": [round(float(x), 5) for x in right_q_meas],
        "q_ready": [round(float(x), 5) for x in Q_LEFT_READY],
        "q_hover": [round(float(x), 5) for x in q_hover],
        "ik_left_pos_err_mm": round(float(info.get("pos_err_mm", 0.0)), 3),
        "ik_left_ori_err_rad": round(float(info.get("ori_err_rad", 0.0)), 4),
        "hover_tcp_wc_m": [round(float(x), 4) for x in hover_tcp_pos],
        "hover_tcp_z_axis": [round(float(x), 4) for x in hover_tcp_z_axis],
        "min_tcp_z_wc_m": round(min_tcp_z, 4),
        "min_geom_clearance_table_m": round(min_geom_table_dist, 4),
        "min_geom_clearance_object_m": round(min_geom_obj_dist, 4) if obj_gid >= 0 else None,
        "max_penetration_mm": round(max_penetration_mm, 3),
        "max_joint_vel_rad_s": round(max_seg_vel, 4),
        "seg1_duration_s": round((len(seg1) - 1) * dt, 2),
        "seg2_duration_s": round((len(seg2) - 1) * dt, 2),
        "total_duration_s": round((len(full_traj) - 1) * dt, 2),
        "limit_violation_count": len(limit_violations),
        "limit_violations_sample": limit_violations[:10],
        "collision_count": len(collisions),
        "collisions_sample": collisions[:10],
        "_traj": full_traj,
    }


def _is_parking_stop_active(ps_detail: Any) -> bool:
    """Detect ANY active parking stop, including PscEmergencyStopButton (enum value 0!)."""
    if ps_detail is None:
        return False
    if isinstance(ps_detail, str):
        return bool(ps_detail.strip())
    # Protobuf ParkingStopDetail: PscEmergencyStopButton has enum value 0, so we MUST
    # check serialized presence rather than `category != 0`.
    if hasattr(ps_detail, "SerializeToString"):
        return len(ps_detail.SerializeToString()) > 0
    cat = getattr(ps_detail, "category", None)
    reason = getattr(ps_detail, "reason", "")
    return bool(cat is not None or reason)


class SafeArmSendGate:
    """Blocks clear_parking_stop unconditionally and blocks control/motion until explicitly armed.

    Starts in blocked state (`execution_allowed=False`) so ZERO `clear_parking_stop`,
    `api_control_initialize=True`, or motion commands can be sent during SDK startup
    or device discovery, and once `latch_stop()` is called it permanently blocks again.
    """

    def __init__(self, orig_send: Callable[[Any], Any], start_blocked: bool = True) -> None:
        self._orig_send = orig_send
        self.execution_allowed = not start_blocked
        self.latched_stopped = False
        self.blocked_messages: list[str] = []

    def allow_execution(self) -> None:
        if not self.latched_stopped:
            self.execution_allowed = True

    def latch_stop(self) -> None:
        self.latched_stopped = True
        self.execution_allowed = False

    def _classify_blocked(self, msg: Any) -> str | None:
        # Non-actuator APIDown messages (enable_kcp, placeholder_message/kcp_start,
        # set_report_frequency) must ALWAYS pass through so KCP & telemetry work.
        has_arm = hasattr(msg, "HasField") and msg.HasField("arm_command")
        has_sec = hasattr(msg, "HasField") and msg.HasField("secondary_device_command")
        if hasattr(msg, "HasField") and not has_arm and not has_sec:
            return None

        if has_sec and (self.latched_stopped or not self.execution_allowed):
            return (
                "post_stop_secondary_device_command"
                if self.latched_stopped
                else "pre_arm_secondary_device_command"
            )

        if has_arm:
            arm_cmd = getattr(msg, "arm_command", None)
            if arm_cmd is not None:
                shared = getattr(arm_cmd, "arm_shared_command", None)
                if shared is not None and getattr(shared, "clear_parking_stop", False):
                    self.latch_stop()
                    return "clear_parking_stop"
                if self.latched_stopped or not self.execution_allowed:
                    excl = getattr(arm_cmd, "arm_exclusive_command", None)
                    if excl is not None and hasattr(excl, "WhichOneof"):
                        which = excl.WhichOneof("exclusive_command")
                        if which == "api_control_initialize":
                            if excl.api_control_initialize:
                                return (
                                    "post_stop_api_control_initialize_true"
                                    if self.latched_stopped
                                    else "pre_arm_api_control_initialize_true"
                                )
                            return None  # Explicit stop (api_control_initialize=False) is allowed
                    return "post_stop_arm_command" if self.latched_stopped else "pre_arm_arm_command"
        return None

    async def __call__(self, msg: Any) -> Any:
        reason = self._classify_blocked(msg)
        if reason is not None:
            self.blocked_messages.append(reason)
            return None
        return await self._orig_send(msg)


def _install_safe_send_gate(arm_dev: Any, start_blocked: bool = False) -> SafeArmSendGate | None:
    orig = getattr(arm_dev, "_send_message", None)
    if orig is None or not callable(orig):
        return None
    gate = SafeArmSendGate(orig, start_blocked=start_blocked)
    arm_dev._send_message = gate
    return gate


def _install_parking_stop_tracker(
    arm_dev: Any,
    initial_api_up: Any = None,
    send_gate: SafeArmSendGate | None = None,
) -> None:
    """Wrap arm_dev._update to preserve proto3 HasField('parking_stop_detail') from frame 0
    and synchronously latch SafeArmSendGate the instant a parking stop arrives.
    """
    if not hasattr(arm_dev, "_parking_stop_present"):
        arm_dev._parking_stop_present = False
    if send_gate is not None:
        arm_dev._linked_send_gate = send_gate
    if initial_api_up is not None:
        try:
            if hasattr(initial_api_up, "HasField") and initial_api_up.HasField("arm_status"):
                has_ps = bool(initial_api_up.arm_status.HasField("parking_stop_detail"))
                arm_dev._parking_stop_present = has_ps
                if has_ps and getattr(arm_dev, "_linked_send_gate", None) is not None:
                    arm_dev._linked_send_gate.latch_stop()
        except Exception:
            pass
    if getattr(arm_dev, "_parking_stop_tracker_installed", False):
        return
    orig_update = getattr(arm_dev, "_update", None)
    if orig_update is None or not callable(orig_update):
        return

    def _wrapped_update(api_up_data, timestamp):
        try:
            if hasattr(api_up_data, "HasField") and api_up_data.HasField("arm_status"):
                has_ps = bool(api_up_data.arm_status.HasField("parking_stop_detail"))
                arm_dev._parking_stop_present = has_ps
                if has_ps and getattr(arm_dev, "_linked_send_gate", None) is not None:
                    arm_dev._linked_send_gate.latch_stop()
        except Exception:
            pass
        return orig_update(api_up_data, timestamp)

    arm_dev._update = _wrapped_update
    arm_dev._parking_stop_tracker_installed = True


def create_guarded_hex_device_api(**kwargs: Any) -> Any:
    """Create a HexDeviceApi subclass whose send gate and frame-0 E-Stop tracker are active from t=0."""
    import hex_device
    from hex_device import HexDeviceApi

    hex_device.set_log_level("WARNING")

    class GuardedHexDeviceApi(HexDeviceApi):
        def __init__(self, **inner_kwargs: Any) -> None:
            self.send_gate = SafeArmSendGate(super()._send_down_message, start_blocked=True)
            super().__init__(**inner_kwargs)

        async def _send_down_message(self, data: Any) -> Any:
            return await self.send_gate(data)

        def _create_and_register_device(self, robot_type: Any, api_up: Any) -> Any:
            device = super()._create_and_register_device(robot_type, api_up)
            if device is not None:
                _install_parking_stop_tracker(
                    device, initial_api_up=api_up, send_gate=self.send_gate
                )
            return device

    return GuardedHexDeviceApi(**kwargs)


def _safe_read_q(arm_dev: Any, pop: bool = False) -> np.ndarray | None:
    raw = arm_dev.get_motor_positions(pop)
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    return arr if arr.ndim == 1 and arr.size > 0 else None


def _safe_read_tau(arm_dev: Any) -> np.ndarray | None:
    raw = arm_dev.get_motor_torques(False)
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    return arr if arr.ndim == 1 and arr.size > 0 else None


def _check_over_temperature(arm_dev: Any, max_temp_c: float = 65.0) -> str | None:
    """Check each motor temperature independently so one None/NaN channel never masks another's over-temp."""
    if not hasattr(arm_dev, "get_motor_temperatures"):
        return None
    temps = arm_dev.get_motor_temperatures()
    if temps is None:
        return None
    for idx, raw_t in enumerate(temps):
        if raw_t is None:
            continue
        try:
            t_val = float(raw_t)
        except (TypeError, ValueError):
            continue
        if math.isfinite(t_val) and t_val >= max_temp_c:
            return f"Motor J{idx + 1} over-temperature {t_val:.1f}C >= {max_temp_c:.1f}C"
    return None


def _check_arm_fault(
    api: Any,
    arm_dev: Any,
    q_cmd: np.ndarray,
    last_q: np.ndarray,
    now_t: float,
    last_fresh_t: float,
    stale_timeout_s: float,
    max_tracking_err_rad: float,
    stop_requested: bool,
    max_temp_c: float = 65.0,
) -> tuple[str | None, np.ndarray, float, float, float]:
    """Check all fault conditions safely even on single-frame SDK queues."""
    if stop_requested:
        q_now = _safe_read_q(arm_dev, False)
        return "User SIGINT", (q_now if q_now is not None else last_q), 0.0, 0.0, last_fresh_t
    if api.is_api_exit():
        q_now = _safe_read_q(arm_dev, False)
        return "HexDeviceApi exited", (q_now if q_now is not None else last_q), 0.0, 0.0, last_fresh_t

    # Read torque BEFORE popping position so a single-frame FIFO queue never empties before torque read!
    tau_now = _safe_read_tau(arm_dev)
    if arm_dev.has_new_data():
        q_popped = _safe_read_q(arm_dev, True)
        if q_popped is not None:
            last_fresh_t = now_t
            q_now = q_popped
        else:
            q_now = last_q
    else:
        q_peek = _safe_read_q(arm_dev, False)
        q_now = q_peek if q_peek is not None else last_q

    err = float(np.max(np.abs(q_cmd - q_now)))
    max_tau = float(np.max(np.abs(tau_now))) if tau_now is not None else 0.0

    if now_t - last_fresh_t > stale_timeout_s:
        return (
            f"Feedback frozen/stale for {now_t - last_fresh_t:.3f}s (> {stale_timeout_s:.3f}s)",
            q_now,
            err,
            max_tau,
            last_fresh_t,
        )

    ps_present = bool(getattr(arm_dev, "_parking_stop_present", False))
    ps = arm_dev.get_parking_stop_detail() if hasattr(arm_dev, "get_parking_stop_detail") else None
    if ps_present or _is_parking_stop_active(ps):
        return f"Parking stop active (present={ps_present}): {ps}", q_now, err, max_tau, last_fresh_t

    errs = arm_dev.get_motor_error_codes()
    if errs and any(e is not None for e in errs):
        return f"Motor error codes: {errs}", q_now, err, max_tau, last_fresh_t

    temp_err = _check_over_temperature(arm_dev, max_temp_c=max_temp_c)
    if temp_err is not None:
        return temp_err, q_now, err, max_tau, last_fresh_t

    if err > max_tracking_err_rad:
        return f"Tracking error {err:.4f} rad > {max_tracking_err_rad} rad", q_now, err, max_tau, last_fresh_t

    return None, q_now, err, max_tau, last_fresh_t


def compute_left_arm_gravity_torque(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_q_adrs: list[int],
    left_dof_adrs: list[int],
    q_rad: np.ndarray,
) -> np.ndarray:
    """Compute exact MuJoCo gravity feedforward torque tau_g(q) for left arm (N*m)."""
    mujoco.mj_resetData(model, data)
    for i, adr in enumerate(left_q_adrs):
        data.qpos[adr] = float(q_rad[i])
    mujoco.mj_forward(model, data)
    tau = np.array([float(data.qfrc_bias[d]) for d in left_dof_adrs], dtype=np.float64)
    limits = np.array([15.0, 15.0, 15.0, 15.0, 4.0, 4.0], dtype=np.float64)
    return np.clip(tau, -limits, limits)


def _send_arm_command(
    arm_dev: Any,
    q_cmd: np.ndarray,
    dq_cmd: np.ndarray,
    tau_ff: np.ndarray,
    control_mode: str,
    kp: list[float],
    kd: list[float],
    vel_lim: list[float],
    acc_lim: list[float],
) -> None:
    if (
        control_mode == "mit_grav"
        and hasattr(arm_dev, "motor_command")
        and hasattr(arm_dev, "construct_mit_command")
    ):
        from hex_device.motor_base import CommandType

        cmds = arm_dev.construct_mit_command(
            pos=q_cmd.tolist(),
            speed=dq_cmd.tolist(),
            torque=tau_ff.tolist(),
            kp=kp,
            kd=kd,
        )
        # MUST use motor_command(CommandType.MIT, ...) so _last_command_time is updated!
        arm_dev.motor_command(CommandType.MIT, cmds)
    else:
        arm_dev.joint_position_control(q_cmd.tolist(), vel_lim, acc_lim)


def execute_left_arm_trajectory(
    traj: np.ndarray,
    dt: float = 0.01,
    max_tracking_err_rad: float = 0.25,
    stale_timeout_s: float = 0.15,
    goal_tol_rad: float = 0.02,
    settle_steps: int = 80,
    control_mode: str = "mit_grav",
    return_to_start: bool = True,
    on_goal_settled: Callable[[], None] | None = None,
    api_factory: Callable[[], Any] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.perf_counter,
    stop_flag: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Stream verified trajectory to left arm with gravity-compensated MIT & immediate-stop watchdog."""
    stop_requested = False

    def _sigint_handler(_sig, _frame):
        nonlocal stop_requested
        stop_requested = True

    old_handler = signal.signal(signal.SIGINT, _sigint_handler)

    # Load MuJoCo dynamics model BEFORE opening controller connection to avoid startup delay
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
        from hex_device import Arm

        api = create_guarded_hex_device_api(
            ws_url=LEFT_ARM_WS_URL, control_hz=200, enable_kcp=True, local_port=0
        )
        arm_cls = Arm
    else:
        api = api_factory()
        arm_cls = None

    max_err_seen = 0.0
    max_torque_seen = 0.0
    aborted_reason: str | None = None
    completed_steps = 0
    commands_sent = 0
    final_q: np.ndarray | None = None
    hover_q: np.ndarray | None = None
    hover_err: float | None = None
    final_feedback_fresh = False
    arm_dev = None
    send_gate: SafeArmSendGate | None = None

    kp_mit = [60.0, 80.0, 80.0, 40.0, 25.0, 20.0]
    kd_mit = [3.0, 4.0, 4.0, 2.0, 1.2, 1.0]
    vel_lim = [0.6] * 6
    acc_lim = [1.2] * 6
    zeros6 = np.zeros(6, dtype=np.float64)

    def _is_stop() -> bool:
        return stop_requested or (stop_flag is not None and stop_flag())

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
                    break
            if arm_dev is not None:
                break
            sleep_fn(0.05)

        if arm_dev is None:
            raise TimeoutError("Could not connect to left arm for execution.")

        # Reuse API-level gate (active from t=0) or install arm-level gate
        send_gate = getattr(api, "send_gate", None)
        if send_gate is None:
            send_gate = _install_safe_send_gate(arm_dev, start_blocked=True)
        _install_parking_stop_tracker(arm_dev, send_gate=send_gate)

        # Pre-start health gate on frame 0 BEFORE unlocking send_gate or calling arm_dev.start()!
        ps_present = bool(getattr(arm_dev, "_parking_stop_present", False))
        ps_detail = arm_dev.get_parking_stop_detail() if hasattr(arm_dev, "get_parking_stop_detail") else None
        if ps_present or _is_parking_stop_active(ps_detail):
            aborted_reason = f"Pre-start parking stop active (present={ps_present}): {ps_detail}"
        else:
            init_errs = arm_dev.get_motor_error_codes()
            if init_errs and any(e is not None for e in init_errs):
                aborted_reason = f"Pre-start motor error codes: {init_errs}"

        q_live = _safe_read_q(arm_dev, True)
        if q_live is None:
            if aborted_reason is None:
                raise RuntimeError("Arm telemetry queue empty after initial has_new_data().")
            last_q = traj[0].copy()
        else:
            last_q = q_live.copy()
            if aborted_reason is None:
                start_diff = float(np.max(np.abs(q_live - traj[0])))
                if start_diff > 0.05:
                    aborted_reason = f"Live arm moved since planning (diff={start_diff:.4f} rad > 0.05)."

        last_fresh_t = clock_fn()

        if aborted_reason is None:
            if send_gate is not None:
                send_gate.allow_execution()
            arm_dev.start()

            # Brief bumpless hold at start with gravity feedforward active
            for _ in range(25):
                now_t = clock_fn()
                aborted_reason, last_q, err, max_tau, last_fresh_t = _check_arm_fault(
                    api, arm_dev, q_live, last_q, now_t, last_fresh_t, stale_timeout_s, max_tracking_err_rad, _is_stop()
                )
                if aborted_reason is not None:
                    break
                tau_g = compute_left_arm_gravity_torque(dyn_model, dyn_data, dyn_q_adrs, dyn_dof_adrs, last_q)
                _send_arm_command(arm_dev, q_live, zeros6, tau_g, control_mode, kp_mit, kd_mit, vel_lim, acc_lim)
                commands_sent += 1
                sleep_fn(dt)

        # Forward trajectory streaming loop: check faults BEFORE every command
        t_start = clock_fn()
        if aborted_reason is None:
            for k in range(len(traj)):
                now_t = clock_fn()
                q_cmd = traj[k]
                dq_cmd = (traj[min(k + 1, len(traj) - 1)] - traj[max(k - 1, 0)]) / (
                    dt if k == 0 or k == len(traj) - 1 else 2.0 * dt
                )
                aborted_reason, last_q, err, max_tau, last_fresh_t = _check_arm_fault(
                    api, arm_dev, q_cmd, last_q, now_t, last_fresh_t, stale_timeout_s, max_tracking_err_rad, _is_stop()
                )
                max_err_seen = max(max_err_seen, err)
                max_torque_seen = max(max_torque_seen, max_tau)
                if aborted_reason is not None:
                    break

                tau_g = compute_left_arm_gravity_torque(dyn_model, dyn_data, dyn_q_adrs, dyn_dof_adrs, last_q)
                _send_arm_command(arm_dev, q_cmd, dq_cmd, tau_g, control_mode, kp_mit, kd_mit, vel_lim, acc_lim)
                commands_sent += 1
                completed_steps = k + 1

                target_t = t_start + (k + 1) * dt
                sleep_s = target_t - clock_fn()
                if sleep_s > 0:
                    sleep_fn(sleep_s)

        # Settle at hover goal ONLY if no abort occurred
        if aborted_reason is None:
            for _ in range(settle_steps):
                now_t = clock_fn()
                aborted_reason, last_q, err, max_tau, last_fresh_t = _check_arm_fault(
                    api, arm_dev, traj[-1], last_q, now_t, last_fresh_t, stale_timeout_s, max_tracking_err_rad, _is_stop()
                )
                max_err_seen = max(max_err_seen, err)
                max_torque_seen = max(max_torque_seen, max_tau)
                if aborted_reason is not None:
                    break
                tau_g = compute_left_arm_gravity_torque(dyn_model, dyn_data, dyn_q_adrs, dyn_dof_adrs, last_q)
                _send_arm_command(arm_dev, traj[-1], zeros6, tau_g, control_mode, kp_mit, kd_mit, vel_lim, acc_lim)
                commands_sent += 1
                sleep_fn(dt)

        q_hover_end = _safe_read_q(arm_dev, False)
        hover_q = q_hover_end if q_hover_end is not None else last_q
        hover_err = float(np.max(np.abs(hover_q - traj[-1])))

        if aborted_reason is None and on_goal_settled is not None:
            # Capture snapshots while arm is actively holding hover pose!
            on_goal_settled()

        # Smoothly return to safe start/rest pose before releasing API control (avoids mid-air drop!)
        if aborted_reason is None and return_to_start:
            ret_traj = traj[::-1]
            t_ret = clock_fn()
            for k in range(len(ret_traj)):
                now_t = clock_fn()
                q_cmd = ret_traj[k]
                dq_cmd = (ret_traj[min(k + 1, len(ret_traj) - 1)] - ret_traj[max(k - 1, 0)]) / (
                    dt if k == 0 or k == len(ret_traj) - 1 else 2.0 * dt
                )
                aborted_reason, last_q, err, max_tau, last_fresh_t = _check_arm_fault(
                    api, arm_dev, q_cmd, last_q, now_t, last_fresh_t, stale_timeout_s, max_tracking_err_rad, _is_stop()
                )
                max_err_seen = max(max_err_seen, err)
                max_torque_seen = max(max_torque_seen, max_tau)
                if aborted_reason is not None:
                    break
                tau_g = compute_left_arm_gravity_torque(dyn_model, dyn_data, dyn_q_adrs, dyn_dof_adrs, last_q)
                _send_arm_command(arm_dev, q_cmd, dq_cmd, tau_g, control_mode, kp_mit, kd_mit, vel_lim, acc_lim)
                commands_sent += 1

                target_t = t_ret + (k + 1) * dt
                sleep_s = target_t - clock_fn()
                if sleep_s > 0:
                    sleep_fn(sleep_s)

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

    expected_end = traj[0] if return_to_start else traj[-1]
    final_err = float(np.max(np.abs(final_q - expected_end))) if final_q is not None else None
    if aborted_reason is None and (not final_feedback_fresh):
        aborted_reason = "Final telemetry stale at completion"
    if aborted_reason is None and (hover_err is None or hover_err > goal_tol_rad):
        aborted_reason = f"Hover goal not reached: hover_err={hover_err:.4f} rad > goal_tol={goal_tol_rad:.4f} rad"
    if aborted_reason is None and (final_err is None or final_err > goal_tol_rad):
        aborted_reason = f"Goal not reached: final_err={final_err:.4f} rad > goal_tol={goal_tol_rad:.4f} rad"

    completed = (
        aborted_reason is None
        and completed_steps == int(len(traj))
        and final_feedback_fresh
        and hover_err is not None
        and hover_err <= goal_tol_rad
        and final_err is not None
        and final_err <= goal_tol_rad
    )

    return {
        "completed": bool(completed),
        "control_mode": control_mode,
        "return_to_start": bool(return_to_start),
        "aborted_reason": aborted_reason,
        "completed_steps": completed_steps,
        "total_steps": int(len(traj)),
        "commands_sent": commands_sent,
        "final_feedback_fresh": bool(final_feedback_fresh),
        "max_tracking_err_rad": round(max_err_seen, 4),
        "max_torque_nm": round(max_torque_seen, 3),
        "hover_q_rad": [round(float(x), 5) for x in hover_q] if hover_q is not None else None,
        "hover_err_to_goal_rad": round(hover_err, 4) if hover_err is not None else None,
        "final_q_rad": [round(float(x), 5) for x in final_q] if final_q is not None else None,
        "final_err_to_goal_rad": round(final_err, 4) if final_err is not None else None,
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
    parser = argparse.ArgumentParser(description="Step 2: Left-arm unfold + full-frame verified air hover.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Plan and verify in MuJoCo only (zero robot motion).")
    mode.add_argument("--execute", action="store_true", help="Execute verified trajectory on real left arm.")
    parser.add_argument(
        "--offline-step1",
        type=Path,
        default=None,
        help="Path to saved step1_readonly_report.json for 100%% offline dry-run (no WS connection).",
    )
    parser.add_argument(
        "--offline-perception",
        type=Path,
        default=None,
        help="Path to saved step2_air_jog_report.json to reuse saved perception offline (no camera access).",
    )
    parser.add_argument("--hover-clearance", type=float, default=0.10, help="Hover height above object top (m).")
    parser.add_argument("--max-vel", type=float, default=0.28, help="Max joint speed (rad/s).")
    parser.add_argument("--control-mode", choices=["mit_grav", "pos"], default="mit_grav", help="Controller mode.")
    parser.add_argument("--no-return", action="store_true", help="Do not return to start pose after hover.")
    parser.add_argument("--target-xyz", type=float, nargs=3, default=None, help="Optional override hover XYZ in workcell_base (m).")
    args = parser.parse_args()

    out_dir = ROOT / "docs" / "reports" / "step2_snapshots"
    report_path = ROOT / "docs" / "reports" / "step2_air_jog_report.json"

    if args.execute and (args.offline_step1 is not None or args.offline_perception is not None):
        raise RuntimeError(
            "Refusing --execute with --offline-step1 or --offline-perception; "
            "live dual-arm state and live top-D435 perception are mandatory for execution."
        )

    if args.offline_perception is not None:
        print(f"1) Loading saved perception offline from {args.offline_perception}...")
        perception = json.loads(args.offline_perception.read_text(encoding="utf-8"))["perception"]
    else:
        print("1) Detecting tabletop & target from top D435...")
        perception = detect_target_from_top_d435(out_dir)

    if args.target_xyz is not None:
        hover_wc = np.asarray(args.target_xyz, dtype=np.float64)
    else:
        c = perception["object_center_wc_m"]
        hover_wc = np.array([c[0], c[1], perception["object_z_top_wc_m"] + args.hover_clearance], dtype=np.float64)

    if args.offline_step1 is not None:
        print(f"2) Loading saved dual-arm joint readings offline from {args.offline_step1}...")
        s1 = json.loads(args.offline_step1.read_text(encoding="utf-8"))
        q_meas = np.asarray(s1["left_arm"]["arm"]["positions_rad"], dtype=np.float64)
        right_q_meas = np.asarray(s1["right_arm"]["arm"]["positions_rad"], dtype=np.float64)
    else:
        print("2) Reading live dual-arm joint angles via zero-write passive WS...")
        q_meas, right_q_meas = read_dual_arms_passive_q()

    print(f"3) Planning & verifying EVERY frame k=0..N to hover_wc={hover_wc.round(4).tolist()}...")
    plan = plan_and_verify_step2(
        q_meas=q_meas,
        right_q_meas=right_q_meas,
        target_hover_wc=hover_wc,
        z_table_wc=perception["z_table_wc_m"],
        perception=perception,
        max_vel=args.max_vel,
    )
    traj = plan.pop("_traj")
    print(
        f"   safe_to_execute={plan['safe_to_execute']}, frames_checked={plan['frames_checked']} "
        f"(skipped={plan['frames_skipped']}), blocking_reasons={plan['blocking_reasons']}"
    )

    exec_res = None
    post_snaps = None
    if args.execute:
        if args.offline_step1 is not None:
            raise RuntimeError("Refusing --execute with --offline-step1; live dual-arm state is required.")
        if not plan["safe_to_execute"]:
            raise RuntimeError(f"Plan failed safety verification — refusing to execute! Reasons: {plan['blocking_reasons']}")
        print(f"4) EXECUTING verified low-speed trajectory ({args.control_mode}) on real left arm (port 8439)...")
        snap_holder: dict[str, str] = {}
        exec_res = execute_left_arm_trajectory(
            traj,
            control_mode=args.control_mode,
            return_to_start=not args.no_return,
            on_goal_settled=lambda: snap_holder.update(capture_verification_snapshots(out_dir)),
        )
        print(
            f"   completed={exec_res['completed']}, aborted_reason={exec_res['aborted_reason']}, "
            f"hover_err={exec_res['hover_err_to_goal_rad']} rad, max_tracking_err={exec_res['max_tracking_err_rad']} rad"
        )
        post_snaps = snap_holder if snap_holder else capture_verification_snapshots(out_dir)

    report = {
        "mode": "execute" if args.execute else ("offline_dry_run" if args.offline_step1 else "dry_run"),
        "target_hover_wc_m": [round(float(x), 4) for x in hover_wc],
        "perception": perception,
        "plan": plan,
        "execution": exec_res,
        "post_hover_snapshots": post_snaps,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
