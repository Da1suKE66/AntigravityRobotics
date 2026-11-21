"""Render an MP4 video and 2x2 keyframe sheet of the Firefly Dual-Arm simulation."""

from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import pathlib
import sys
import numpy as np
import mujoco
import mink
from PIL import Image, ImageDraw
import imageio.v2 as imageio

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controller import BumplessGravityController
from src.ik_solver import FireflyDualIK
from src.sim_runner import interpolate_se3

ASSET_XML = PROJECT_ROOT / "model" / "firefly_dual.xml"
SCENE_XML = PROJECT_ROOT / "model" / "scene.xml"
OUT_DIR = pathlib.Path("/home/lakeside/.gemini/antigravity/brain/1cf4bbf1-98fd-4fe6-955e-be1b1890bb60")


def render_demo_video() -> tuple[pathlib.Path, pathlib.Path]:
    """Simulate 5.0 s: Bumpless handoff -> Dual-arm circle + reach -> Gravity hold, and render MP4."""
    sim_model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    dyn_model = mujoco.MjModel.from_xml_path(str(ASSET_XML))
    data = mujoco.MjData(sim_model)

    # Ensure offscreen framebuffer supports 1280x720
    sim_model.vis.global_.offwidth = 1280
    sim_model.vis.global_.offheight = 720

    mujoco.mj_resetDataKeyframe(sim_model, data, sim_model.key("home").id)
    mujoco.mj_forward(sim_model, data)

    ik = FireflyDualIK(dyn_model)
    ctrl = BumplessGravityController(dyn_model, engage_duration=0.40)

    q_init = data.qpos.copy()
    t_l0, t_r0 = ik.reset_to_measured(q_init)
    ctrl.engage(q_init, data.qvel.copy())

    dt = sim_model.opt.timestep  # 0.002 s (500 Hz)
    duration = 5.0
    steps = int(duration / dt)
    fps = 30
    render_every = max(1, int(round(1.0 / (fps * dt))))

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.35, 0.0, 0.48]
    cam.distance = 1.45
    cam.azimuth = 140.0
    cam.elevation = -22.0

    renderer = mujoco.Renderer(sim_model, height=720, width=1280)
    frames: list[np.ndarray] = []

    try:
        for k in range(steps):
            t = k * dt
            # Phase 1 (0.0 - 0.5s): Bumpless hold at initial pose
            # Phase 2 (0.5 - 2.0s): Smooth SE(3) approach forward & down
            # Phase 3 (2.0 - 4.0s): Coordinated dual-arm circular motion (r=40mm)
            # Phase 4 (4.0 - 5.0s): Hold under gravity
            if t < 0.5:
                phase_lbl = "Phase 1: Bumpless Engage & Gravity Hold"
                t_l_des, t_r_des = t_l0, t_r0
            elif t < 2.0:
                phase_lbl = "Phase 2: C2 Quintic SE(3) Reach"
                u = (t - 0.5) / 1.5
                t_l_goal = mink.SE3.from_rotation_and_translation(
                    t_l0.rotation(), t_l0.translation() + np.array([0.06, -0.05, -0.06])
                )
                t_r_goal = mink.SE3.from_rotation_and_translation(
                    t_r0.rotation(), t_r0.translation() + np.array([0.06, 0.05, -0.06])
                )
                t_l_des = interpolate_se3(t_l0, t_l_goal, u)
                t_r_des = interpolate_se3(t_r0, t_r_goal, u)
            elif t < 4.0:
                phase_lbl = "Phase 3: Dual-Arm Mink+DAQP Circle (r=40mm)"
                ang = 2.0 * np.pi * ((t - 2.0) / 2.0)
                r = 0.04 * min(1.0, (t - 2.0) / 0.3)
                off_l = np.array([0.06, -0.05 + r * np.cos(ang) - r, -0.06 + r * np.sin(ang)])
                off_r = np.array([0.06, 0.05 - r * np.cos(ang) + r, -0.06 + r * np.sin(ang)])
                t_l_des = mink.SE3.from_rotation_and_translation(t_l0.rotation(), t_l0.translation() + off_l)
                t_r_des = mink.SE3.from_rotation_and_translation(t_r0.rotation(), t_r0.translation() + off_r)
            else:
                phase_lbl = "Phase 4: Steady-State Gravity Hold"
                off_l = np.array([0.06, -0.05, -0.06])
                off_r = np.array([0.06, 0.05, -0.06])
                t_l_des = mink.SE3.from_rotation_and_translation(t_l0.rotation(), t_l0.translation() + off_l)
                t_r_des = mink.SE3.from_rotation_and_translation(t_r0.rotation(), t_r0.translation() + off_r)

            ik.set_targets(t_l_des, t_r_des)
            q_meas = data.qpos.copy()
            dq_meas = data.qvel.copy()
            q_ref, dq_ref = ik.step(dt, q_meas=q_meas, blend_meas=0.02)
            tau = ctrl.step(dt, q_meas, dq_meas, q_ref, dq_ref)
            data.ctrl[:] = tau
            mujoco.mj_step(sim_model, data)

            if k % render_every == 0:
                renderer.update_scene(data, camera=cam)
                rgb = renderer.render()
                l_err = np.linalg.norm(data.site("left_ee_site").xpos - t_l_des.translation()) * 1000.0
                r_err = np.linalg.norm(data.site("right_ee_site").xpos - t_r_des.translation()) * 1000.0
                img = Image.fromarray(rgb)
                draw = ImageDraw.Draw(img)
                hud = (
                    f"t={t:4.2f}s | {phase_lbl}\n"
                    f"Left EE err: {l_err:4.2f} mm | Right EE err: {r_err:4.2f} mm | "
                    f"Peak tau: {np.max(np.abs(tau)):5.2f} Nm | ncon={data.ncon}"
                )
                draw.rectangle([16, 16, 760, 68], fill=(15, 20, 30))
                draw.text((26, 24), hud, fill=(240, 248, 255))
                frames.append(np.asarray(img))
    finally:
        renderer.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mp4_path = OUT_DIR / "firefly_sim_demo.mp4"
    imageio.mimsave(str(mp4_path), frames, fps=fps, macro_block_size=16)

    # Save 2x2 keyframe sheet
    idxs = [0, len(frames) // 3, (2 * len(frames)) // 3, len(frames) - 1]
    tiles = [Image.fromarray(frames[i]).resize((640, 360)) for i in idxs]
    sheet = Image.new("RGB", (1280, 720))
    sheet.paste(tiles[0], (0, 0))
    sheet.paste(tiles[1], (640, 0))
    sheet.paste(tiles[2], (0, 360))
    sheet.paste(tiles[3], (640, 360))
    sheet_path = OUT_DIR / "firefly_sim_keyframes.png"
    sheet.save(str(sheet_path))

    print(f"Saved video: {mp4_path} ({len(frames)} frames)")
    print(f"Saved sheet: {sheet_path}")
    return mp4_path, sheet_path


if __name__ == "__main__":
    render_demo_video()
