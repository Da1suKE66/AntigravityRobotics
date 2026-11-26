"""Step 1: Strictly READ-ONLY real-robot & RealSense camera bring-up verification.

Safety guarantee:
- NEVER calls device.start(), motor_command(), enable_mit(), or any actuator command.
- Only reads telemetry from left arm (port 8439) and right arm (port 9439), checks
  joint limits & FK against model/firefly_dual.xml, captures snapshots from all 3
  RealSense cameras, and estimates tabletop height z_table from top D435 depth.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pyrealsense2 as rs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from readonly_ws_client import read_arm_passive_readonly

from real_calibration import (
    CAMERA_INTRINSICS,
    CAMERA_SERIALS,
    LEFT_ARM_WS_URL,
    RIGHT_ARM_WS_URL,
    T_ARM1_BASE_FROM_ARM2_BASE,
    T_ARM1_BASE_FROM_TOP_D435,
    T_ARM1_LINK6_FROM_LEFT_D405,
    T_ARM2_BASE_FROM_TOP_D435,
    T_ARM2_LINK6_FROM_RIGHT_D405,
    URDF_WORKCELL_TO_LEFT_BASE,
    check_se3_orthogonality,
    transform_points,
)


def _to_list(val):
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return [float(x) for x in val.tolist()]
    if isinstance(val, (list, tuple)):
        return [float(x) if isinstance(x, (int, float, np.floating, np.integer)) else str(x) for x in val]
    return val


def read_arm_readonly(name: str, ws_url: str, timeout_s: float = 5.0) -> dict:
    """Connect to one Firefly Y6 arm controller in strictly zero-write passive WS mode."""
    return read_arm_passive_readonly(name=name, ws_url=ws_url, timeout_s=timeout_s)


def evaluate_fk_in_mujoco(left_q: list[float] | None, right_q: list[float] | None) -> dict:
    """Validate live joint angles against model/firefly_dual.xml limits and compute FK."""
    model_path = ROOT / "model" / "firefly_dual.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    limit_checks: dict[str, list[dict]] = {"left": [], "right": []}
    for side, q_vals in (("left", left_q), ("right", right_q)):
        if not q_vals:
            continue
        for idx in range(min(6, len(q_vals))):
            jname = f"{side}_joint_{idx + 1}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            qadr = model.jnt_qposadr[jid]
            val = float(q_vals[idx])
            data.qpos[qadr] = val
            lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
            in_bounds = lo <= val <= hi
            margin = min(val - lo, hi - val)
            limit_checks[side].append(
                {
                    "joint": jname,
                    "q_rad": round(val, 6),
                    "range_rad": [round(lo, 6), round(hi, 6)],
                    "in_bounds": bool(in_bounds),
                    "margin_rad": round(margin, 6),
                }
            )

    mujoco.mj_forward(model, data)

    def _pose_in_frame(site_name: str, ref_body: str | None = None) -> dict:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        pos_w = data.site_xpos[sid].copy()
        rot_w = data.site_xmat[sid].reshape(3, 3).copy()
        if ref_body is not None:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ref_body)
            bpos = data.xpos[bid]
            brot = data.xmat[bid].reshape(3, 3)
            pos = brot.T @ (pos_w - bpos)
            rot = brot.T @ rot_w
        else:
            pos = pos_w
            rot = rot_w
        # Z-axis of site (gripper approach direction)
        return {
            "pos_m": [round(float(x), 5) for x in pos],
            "z_axis": [round(float(x), 5) for x in rot[:, 2]],
        }

    # Count active contacts in the live configuration
    contacts = []
    for ci in range(data.ncon):
        con = data.contact[ci]
        if con.dist < -1e-4:
            g1 = model.geom(con.geom1).name or f"geom_{con.geom1}"
            g2 = model.geom(con.geom2).name or f"geom_{con.geom2}"
            contacts.append({"geom1": g1, "geom2": g2, "dist_mm": round(float(con.dist) * 1000.0, 3)})

    return {
        "model_path": str(model_path),
        "limit_checks": limit_checks,
        "all_in_bounds": all(c["in_bounds"] for side in limit_checks.values() for c in side),
        "fk": {
            "left_tcp_in_workcell": _pose_in_frame("left_tcp_site", None),
            "left_tcp_in_arm1_base": _pose_in_frame("left_tcp_site", "left_base_link"),
            "right_tcp_in_workcell": _pose_in_frame("right_tcp_site", None),
            "right_tcp_in_arm2_base": _pose_in_frame("right_tcp_site", "right_base_link"),
        },
        "penetrations_gt_0p1mm": contacts,
    }


def check_realsense_cameras(out_dir: Path) -> dict:
    """Sequentially open each RealSense camera, save RGB snapshot, and estimate z_table on top_d435."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = rs.context()
    detected = {}
    for dev in ctx.query_devices():
        sn = dev.get_info(rs.camera_info.serial_number)
        detected[sn] = {
            "name": dev.get_info(rs.camera_info.name),
            "usb_type": dev.get_info(rs.camera_info.usb_type_descriptor)
            if dev.supports(rs.camera_info.usb_type_descriptor)
            else "unknown",
            "firmware": dev.get_info(rs.camera_info.firmware_version),
        }

    results: dict = {"detected_devices": detected, "cameras": {}, "table_height_estimate": None}

    for cam_name, serial in CAMERA_SERIALS.items():
        cam_res: dict = {"serial": serial, "ok": False, "error": None}
        if serial not in detected:
            cam_res["error"] = "Serial not detected by librealsense"
            results["cameras"][cam_name] = cam_res
            continue

        cam_res["usb_type"] = detected[serial]["usb_type"]
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        # Conservative 640x480@15fps works reliably on both USB 2.0 and USB 3.0
        fps = 15
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)

        try:
            profile = pipe.start(cfg)
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = float(depth_sensor.get_depth_scale())
            align = rs.align(rs.stream.color)

            # Warm up auto-exposure
            frames = None
            for _ in range(12):
                frames = pipe.wait_for_frames(5000)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            color_img = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float64) * depth_scale

            img_path = out_dir / f"{cam_name}_{serial}_rgb.png"
            cv2.imwrite(str(img_path), color_img)

            # SDK reported intrinsics for cross-check
            c_intr = color_frame.profile.as_video_stream_profile().intrinsics
            valid_mask = (depth_m > 0.05) & (depth_m < 2.0)
            valid_ratio = float(np.mean(valid_mask))

            cam_res.update(
                {
                    "ok": True,
                    "shape": list(color_img.shape),
                    "depth_scale_m": depth_scale,
                    "valid_depth_ratio": round(valid_ratio, 4),
                    "median_depth_m": round(float(np.median(depth_m[valid_mask])), 4) if valid_ratio > 0 else None,
                    "sdk_intrinsics": {
                        "fx": round(c_intr.fx, 3),
                        "fy": round(c_intr.fy, 3),
                        "cx": round(c_intr.ppx, 3),
                        "cy": round(c_intr.ppy, 3),
                    },
                    "snapshot_path": str(img_path),
                }
            )

            # Estimate tabletop height z_table using top D435 central ROI + calibrated extrinsic
            if cam_name == "top_d435":
                intr = CAMERA_INTRINSICS["top_d435"]
                h, w = depth_m.shape
                # Central tabletop patch (avoid edges/brackets)
                u_grid, v_grid = np.meshgrid(np.arange(160, 480, 2), np.arange(140, 380, 2))
                z_roi = depth_m[v_grid, u_grid]
                m = (z_roi > 0.30) & (z_roi < 1.20)
                if np.count_nonzero(m) > 200:
                    u_v = u_grid[m].astype(np.float64)
                    v_v = v_grid[m].astype(np.float64)
                    z_v = z_roi[m]
                    x_c = (u_v - intr["cx"]) * z_v / intr["fx"]
                    y_c = (v_v - intr["cy"]) * z_v / intr["fy"]
                    pts_cam = np.stack([x_c, y_c, z_v], axis=1)
                    pts_arm1 = transform_points(T_ARM1_BASE_FROM_TOP_D435, pts_cam)
                    pts_wc = pts_arm1 + URDF_WORKCELL_TO_LEFT_BASE[None, :]

                    # Filter near-horizontal table points (within +/- 8 cm of lower-mode z)
                    z_a1 = pts_arm1[:, 2]
                    # Use 30th-70th percentile band to reject tall objects on table
                    z_low = float(np.percentile(z_a1, 15))
                    z_med = float(np.percentile(z_a1, 50))
                    z_high = float(np.percentile(z_a1, 85))
                    results["table_height_estimate"] = {
                        "sample_count": int(len(z_a1)),
                        "z_arm1_base_p15_p50_p85_m": [round(z_low, 4), round(z_med, 4), round(z_high, 4)],
                        "z_workcell_p15_p50_p85_m": [
                            round(float(np.percentile(pts_wc[:, 2], 15)), 4),
                            round(float(np.percentile(pts_wc[:, 2], 50)), 4),
                            round(float(np.percentile(pts_wc[:, 2], 85)), 4),
                        ],
                        "note": "Approximate tabletop plane from central D435 ROI (includes any objects currently on table).",
                    }
        except Exception as exc:
            cam_res["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                pipe.stop()
            except Exception:
                pass

        results["cameras"][cam_name] = cam_res

    return results


def main() -> None:
    report_dir = ROOT / "docs" / "reports"
    snap_dir = report_dir / "step1_snapshots"
    report_path = report_dir / "step1_readonly_report.json"

    print("=== Step 1: Strictly Read-Only Hardware Bring-up Check ===")
    print("1) Checking calibration SE(3) matrices...")
    se3_checks = {
        "T_arm1_link6_from_left_d405": check_se3_orthogonality(T_ARM1_LINK6_FROM_LEFT_D405),
        "T_arm2_link6_from_right_d405": check_se3_orthogonality(T_ARM2_LINK6_FROM_RIGHT_D405),
        "T_arm1_base_from_top_d435": check_se3_orthogonality(T_ARM1_BASE_FROM_TOP_D435),
        "T_arm2_base_from_top_d435": check_se3_orthogonality(T_ARM2_BASE_FROM_TOP_D435),
        "T_arm1_base_from_arm2_base": check_se3_orthogonality(T_ARM1_BASE_FROM_ARM2_BASE),
    }

    print(f"2) Reading Left Arm (read-only): {LEFT_ARM_WS_URL}")
    left_res = read_arm_readonly("left_arm", LEFT_ARM_WS_URL)
    print(f"   Left connected={left_res['connected']} err={left_res['error']}")

    print(f"3) Reading Right Arm (read-only): {RIGHT_ARM_WS_URL}")
    right_res = read_arm_readonly("right_arm", RIGHT_ARM_WS_URL)
    print(f"   Right connected={right_res['connected']} err={right_res['error']}")

    left_q = left_res["arm"]["positions_rad"] if left_res["arm"] else None
    right_q = right_res["arm"]["positions_rad"] if right_res["arm"] else None

    print("4) Evaluating live joint angles & FK in MuJoCo (model/firefly_dual.xml)...")
    fk_res = evaluate_fk_in_mujoco(left_q, right_q)

    print("5) Capturing RealSense snapshots & estimating tabletop z_table...")
    cam_res = check_realsense_cameras(snap_dir)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "se3_checks": se3_checks,
        "left_arm": left_res,
        "right_arm": right_res,
        "mujoco_fk_check": fk_res,
        "realsense": cam_res,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved full report to: {report_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
