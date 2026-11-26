"""Render an offline MuJoCo simulation video + keyframe sheet for Step 3 tissue-pack pick-and-place."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from step3_left_arm_pick_place import (
    GRIP_CLOSE_RAD,
    GRIP_OPEN_RAD,
    build_step3_verification_model,
    plan_and_verify_step3,
)


def main() -> None:
    s1 = json.loads((ROOT / "docs" / "reports" / "step1_readonly_report.json").read_text(encoding="utf-8"))
    s3 = json.loads((ROOT / "docs" / "reports" / "step3_pick_place_report.json").read_text(encoding="utf-8"))

    q_meas = np.asarray(s1["left_arm"]["arm"]["positions_rad"], dtype=np.float64)
    right_q_meas = np.asarray(s1["right_arm"]["arm"]["positions_rad"], dtype=np.float64)
    perception = s3["perception"]

    plan = plan_and_verify_step3(q_meas=q_meas, right_q_meas=right_q_meas, perception=perception)
    segments = plan.pop("_segments")

    model, obj_meta = build_step3_verification_model(
        z_table_wc=float(perception["z_table_wc_m"]),
        perception=perception,
    )
    # Show group 3 (table & tissue pack colliders) in render
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    data = mujoco.MjData(model)

    left_q_adrs = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"left_joint_{i}")])
        for i in range(1, 7)
    ]
    right_q_adrs = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"right_joint_{i}")])
        for i in range(1, 7)
    ]
    left_grip_adrs = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_grip_joint_1")]),
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_grip_joint_2")]),
    ]
    left_tcp_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_tcp_site")
    table_mocap_id = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table_object_mocap")])
    carried_mocap_id = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "carried_object_mocap")])

    pick_pos = obj_meta["pick_pos"]
    p_place = np.asarray(plan["p_place_wc_m"], dtype=np.float64)
    place_pos = np.array([p_place[0], p_place[1], pick_pos[2]], dtype=np.float64)
    yaw_quat = obj_meta["yaw_quat"]
    offscreen_pos = np.array([0.0, 0.0, -10.0], dtype=np.float64)

    # Compute TCP -> object offset at q_grasp
    q_grasp = segments["descend_grasp"][-1]
    mujoco.mj_resetData(model, data)
    for i, a in enumerate(left_q_adrs):
        data.qpos[a] = q_grasp[i]
    mujoco.mj_forward(model, data)
    R_tcp_grasp = data.site_xmat[left_tcp_sid].reshape(3, 3).copy()
    p_tcp_grasp = data.site_xpos[left_tcp_sid].copy()
    tcp_to_obj_local = R_tcp_grasp.T @ (pick_pos - p_tcp_grasp)
    R_yaw = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(R_yaw, yaw_quat)
    R_obj_in_tcp = R_tcp_grasp.T @ R_yaw.reshape(3, 3)

    # Physical jaw angle when clamping an ~80 mm wide soft tissue pack (~0.18 rad -> ~78 mm fingertip gap)
    grip_hold_rad = 0.18
    seg_grip = {
        "unfold": GRIP_OPEN_RAD,
        "to_hover": GRIP_OPEN_RAD,
        "descend_grasp": GRIP_OPEN_RAD,
        "lift": grip_hold_rad,
        "to_place_hover": grip_hold_rad,
        "descend_place": grip_hold_rad,
        "retreat_place": GRIP_OPEN_RAD,
        "return_ready": GRIP_OPEN_RAD,
        "return_rest": GRIP_OPEN_RAD,
    }
    pick_segs = {"unfold", "to_hover", "descend_grasp"}
    carried_segs = {"lift", "to_place_hover", "descend_place"}

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.35, 0.05, 0.18]
    cam.distance = 1.15
    cam.azimuth = 140.0
    cam.elevation = -28.0

    opt = mujoco.MjvOption()
    opt.geomgroup[2] = 1  # visual meshes
    opt.geomgroup[3] = 1  # table & tissue-pack boxes

    renderer = mujoco.Renderer(model, height=720, width=1280)
    out_mp4 = ROOT / "docs" / "media" / "step3_pick_place_sim.mp4"
    out_png = ROOT / "docs" / "media" / "step3_pick_place_keyframes.png"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 720))
    keyframes: list[np.ndarray] = []
    key_stages = ["unfold", "to_hover", "descend_grasp", "lift", "to_place_hover", "retreat_place"]

    try:
        for seg_name, seg in segments.items():
            g_rad = seg_grip[seg_name]
            # Render at ~30 fps (stride=3 on 100 Hz trajectory -> 1x real-time)
            indices = list(range(0, len(seg), 3))
            if indices[-1] != len(seg) - 1:
                indices.append(len(seg) - 1)
            for idx_pos, k in enumerate(indices):
                q_k = seg[k]
                mujoco.mj_resetData(model, data)
                for i, a in enumerate(left_q_adrs):
                    data.qpos[a] = q_k[i]
                for i, a in enumerate(right_q_adrs):
                    data.qpos[a] = right_q_meas[i]
                for ga in left_grip_adrs:
                    data.qpos[ga] = g_rad

                mujoco.mj_kinematics(model, data)
                if seg_name in pick_segs:
                    data.mocap_pos[table_mocap_id] = pick_pos
                    data.mocap_quat[table_mocap_id] = yaw_quat
                    data.mocap_pos[carried_mocap_id] = offscreen_pos
                elif seg_name in carried_segs:
                    data.mocap_pos[table_mocap_id] = offscreen_pos
                    R_tcp_k = data.site_xmat[left_tcp_sid].reshape(3, 3)
                    p_tcp_k = data.site_xpos[left_tcp_sid]
                    data.mocap_pos[carried_mocap_id] = p_tcp_k + R_tcp_k @ tcp_to_obj_local
                    q_car = np.zeros(4, dtype=np.float64)
                    mujoco.mju_mat2Quat(q_car, (R_tcp_k @ R_obj_in_tcp).ravel())
                    data.mocap_quat[carried_mocap_id] = q_car
                else:
                    data.mocap_pos[table_mocap_id] = place_pos
                    data.mocap_quat[table_mocap_id] = yaw_quat
                    data.mocap_pos[carried_mocap_id] = offscreen_pos

                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam, scene_option=opt)
                bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                fp = perception["object_footprint_m"]
                h_obj = float(perception["object_z_top_wc_m"]) - float(perception["z_table_wc_m"])
                cv2.putText(
                    bgr,
                    f"Step 3 Sim | {seg_name} | grip={g_rad:.2f} rad | obj={fp[0]*100:.1f}x{fp[1]*100:.1f}x{h_obj*100:.1f} cm",
                    (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )
                writer.write(bgr)
                if seg_name in key_stages and idx_pos == len(indices) - 1:
                    keyframes.append(cv2.resize(bgr, (640, 360)))
    finally:
        writer.release()
        renderer.close()

    if len(keyframes) == 6:
        top = np.hstack(keyframes[:3])
        bot = np.hstack(keyframes[3:])
        cv2.imwrite(str(out_png), np.vstack([top, bot]))

    # Transcode to H.264 (yuv420p) for universal browser/player support
    tmp_h264 = out_mp4.with_name(f"{out_mp4.stem}_h264.mp4")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg_exe, "-y", "-i", str(out_mp4), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp_h264)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.replace(tmp_h264, out_mp4)
    print(f"Saved H.264 video: {out_mp4}")
    print(f"Saved keyframes: {out_png}")


if __name__ == "__main__":
    main()
