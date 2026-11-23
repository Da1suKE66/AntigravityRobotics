"""Closed-loop tabletop pick-and-place pipeline for Firefly Dual-Arm in MuJoCo."""

from __future__ import annotations

import csv
import json
import pathlib
import sys
from dataclasses import dataclass
import numpy as np
import mujoco
import mink
from PIL import Image, ImageDraw
import imageio.v2 as imageio

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.controller import BumplessGravityController
from src.ik_solver import FireflyDualIK, quintic_blend
from src.planner import FireflyPlanner, PlannedTrajectory
ASSET_XML = PROJECT_ROOT / "model" / "firefly_dual.xml"
GRASP_SCENE_XML = PROJECT_ROOT / "model" / "grasp_scene.xml"
PATHS_DIR = PROJECT_ROOT / "docs" / "paths"
REPORTS_DIR = PROJECT_ROOT / "docs" / "reports"
MEDIA_DIR = PROJECT_ROOT / "docs" / "media"

# Top-down grasp orientation: +Z_tcp points world -Z; fingers separate along world Y
R_TOP_DOWN = mink.SO3.from_matrix(
    np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
)


@dataclass
class PickPlaceSpec:
    """Configuration for one single-arm tabletop pick-and-place task."""

    side: str
    obj_body: str
    place_xy: tuple[float, float]
    obj_half_height: float
    grasp_z_offset: float = 0.010
    place_surface_z: float = 0.082
    pre_height: float = 0.085
    lift_height: float = 0.095
    grip_open: float = 0.02
    grip_close: float = 0.52


class GraspExecutor:
    """Executes staged closed-loop pick-and-place with path recording & optional video rendering."""

    def __init__(self, record_video: bool = False) -> None:
        self.sim_model = mujoco.MjModel.from_xml_path(str(GRASP_SCENE_XML))
        self.dyn_model = mujoco.MjModel.from_xml_path(str(ASSET_XML))
        self.data = mujoco.MjData(self.sim_model)
        self.dt = self.sim_model.opt.timestep  # 0.002 s

        mujoco.mj_resetDataKeyframe(self.sim_model, self.data, self.sim_model.key("home").id)
        mujoco.mj_forward(self.sim_model, self.data)

        self.ik = FireflyDualIK(
            self.dyn_model,
            solver="daqp",
            damping=1e-4,
            pos_cost=4.0,
            ori_cost=1.2,
            posture_cost=1e-3,
            lm_damping=0.05,
            site_suffix="tcp_site",
            freeze_grippers=True,
        )
        # Note: planner collision check runs on asset-only model during planning,
        # or on grasp_scene while ignoring tabletop/prop contacts at grasp contact.
        self.planner = FireflyPlanner(self.dyn_model, self.ik)
        self.ctrl = BumplessGravityController(self.dyn_model, engage_duration=0.25)

        q0 = self.data.qpos[: self.dyn_model.nq].copy()
        dq0 = self.data.qvel[: self.dyn_model.nv].copy()
        self.ik.reset_to_measured(q0)
        self.ctrl.engage(q0, dq0)

        self.q_cmd = q0.copy()
        self.sim_time = 0.0
        self.peak_tau = 0.0
        self.stage_logs: list[dict[str, object]] = []
        self.exec_trace: list[dict[str, object]] = []

        self.record_video = record_video
        self.frames: list[np.ndarray] = []
        self._renderer = None
        self._render_every = max(1, int(round(1.0 / (30.0 * self.dt))))
        self._step_count = 0
        self._current_label = "Init"

        if record_video:
            self.sim_model.vis.global_.offwidth = 1280
            self.sim_model.vis.global_.offheight = 720
            self._renderer = mujoco.Renderer(self.sim_model, height=720, width=1280)
            self._cam = mujoco.MjvCamera()
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._cam.lookat[:] = [0.38, 0.0, 0.28]
            self._cam.distance = 1.25
            self._cam.azimuth = 135.0
            self._cam.elevation = -26.0

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _grip_indices(self, side: str) -> list[int]:
        return [6, 7] if side == "left" else [14, 15]

    def _step_control(self, q_ref: np.ndarray, dq_ref: np.ndarray) -> None:
        q_meas = self.data.qpos[: self.dyn_model.nq].copy()
        dq_meas = self.data.qvel[: self.dyn_model.nv].copy()
        tau = self.ctrl.step(self.dt, q_meas, dq_meas, q_ref, dq_ref)
        self.data.ctrl[: self.dyn_model.nu] = tau
        mujoco.mj_step(self.sim_model, self.data)

        self.sim_time += self.dt
        self._step_count += 1
        self.peak_tau = max(self.peak_tau, float(np.max(np.abs(tau))))

        if self._step_count % 10 == 0:
            self.exec_trace.append(
                {
                    "t_s": round(self.sim_time, 4),
                    "stage": self._current_label,
                    "q_meas": np.round(q_meas, 5).tolist(),
                    "left_tcp": np.round(self.data.site("left_tcp_site").xpos, 5).tolist(),
                    "right_tcp": np.round(self.data.site("right_tcp_site").xpos, 5).tolist(),
                    "red_block": np.round(self.data.body("red_block").xpos, 5).tolist(),
                    "blue_cylinder": np.round(self.data.body("blue_cylinder").xpos, 5).tolist(),
                }
            )

        if self._renderer is not None and (self._step_count % self._render_every == 0):
            self._renderer.update_scene(self.data, camera=self._cam)
            rgb = self._renderer.render()
            img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(img)
            rb = self.data.body("red_block").xpos
            bc = self.data.body("blue_cylinder").xpos
            hud = (
                f"t={self.sim_time:5.2f}s | {self._current_label}\n"
                f"red_block=({rb[0]:.3f}, {rb[1]:.3f}, {rb[2]:.3f}) | "
                f"blue_cyl=({bc[0]:.3f}, {bc[1]:.3f}, {bc[2]:.3f}) | "
                f"peak_tau={self.peak_tau:4.1f} Nm"
            )
            draw.rectangle([16, 16, 860, 68], fill=(15, 20, 30))
            draw.text((26, 24), hud, fill=(240, 248, 255))
            self.frames.append(np.asarray(img))

    def _run_trajectory(self, traj: PlannedTrajectory, label: str) -> None:
        self._current_label = label
        steps = int(round(traj.t[-1] / self.dt))
        for k in range(steps):
            q_ref, dq_ref = traj.sample(k * self.dt)
            # Preserve current commanded gripper targets
            q_ref[6:8] = self.q_cmd[6:8]
            q_ref[14:16] = self.q_cmd[14:16]
            dq_ref[6:8] = 0.0
            dq_ref[14:16] = 0.0
            self._step_control(q_ref, dq_ref)
        self.q_cmd = traj.q[-1].copy()
        self.q_cmd[6:8] = q_ref[6:8]
        self.q_cmd[14:16] = q_ref[14:16]

    def _move_pose(
        self,
        label: str,
        side: str,
        target_pose: mink.SE3,
        duration: float,
        planned_segments: list[PlannedTrajectory],
    ) -> dict[str, object]:
        q_start = self.q_cmd.copy()
        traj, ik_info = self.planner.plan_pose_p2p(
            q_start,
            left_target=target_pose if side == "left" else None,
            right_target=target_pose if side == "right" else None,
            duration=duration,
            dt=self.dt,
        )
        planned_segments.append(traj)
        self._run_trajectory(traj, label)
        err_mm = float(np.linalg.norm(self.data.site(f"{side}_tcp_site").xpos - target_pose.translation()) * 1000.0)
        info = {
            "stage": label,
            "ik_pos_err_mm": round(float(ik_info["pos_err_mm"]), 4),
            "exec_pos_err_mm": round(err_mm, 3),
            "collision_free": traj.collision_free,
        }
        self.stage_logs.append(info)
        return info

    def _move_joint_home(
        self,
        label: str,
        side: str,
        duration: float,
        planned_segments: list[PlannedTrajectory],
    ) -> None:
        q_home = self.dyn_model.key("home").qpos.copy()
        q_goal = self.q_cmd.copy()
        dofs = self.ik.left_arm_dofs if side == "left" else self.ik.right_arm_dofs
        q_goal[dofs] = q_home[dofs]
        traj = self.planner.plan_joint_p2p(self.q_cmd.copy(), q_goal, duration=duration, dt=self.dt)
        planned_segments.append(traj)
        self._run_trajectory(traj, label)
        self.stage_logs.append({"stage": label, "collision_free": traj.collision_free})

    def _set_gripper(self, label: str, side: str, q_target: float, ramp_s: float = 0.45, hold_s: float = 0.25) -> None:
        self._current_label = label
        g_idx = self._grip_indices(side)
        g0 = self.q_cmd[g_idx].copy()
        g1 = np.array([q_target, q_target], dtype=float)
        ramp_steps = max(1, int(round(ramp_s / self.dt)))
        for k in range(ramp_steps):
            u = (k + 1) / ramp_steps
            s, ds = quintic_blend(u)
            q_ref = self.q_cmd.copy()
            dq_ref = np.zeros(self.dyn_model.nv)
            q_ref[g_idx] = g0 + s * (g1 - g0)
            dq_ref[g_idx] = (ds / max(ramp_s, 1e-6)) * (g1 - g0)
            self._step_control(q_ref, dq_ref)
        self.q_cmd[g_idx] = g1
        hold_steps = max(0, int(round(hold_s / self.dt)))
        for _ in range(hold_steps):
            self._step_control(self.q_cmd.copy(), np.zeros(self.dyn_model.nv))
        self.stage_logs.append(
            {
                "stage": label,
                "grip_cmd_rad": q_target,
                "grip_meas_rad": [round(float(self.data.qpos[i]), 4) for i in g_idx],
            }
        )

    def _hold(self, label: str, duration: float) -> None:
        self._current_label = label
        for _ in range(int(round(duration / self.dt))):
            self._step_control(self.q_cmd.copy(), np.zeros(self.dyn_model.nv))

    def execute_pick_and_place(self, spec: PickPlaceSpec) -> dict[str, object]:
        """Execute a complete single-arm pick-and-place task and measure physical completion."""
        side = spec.side
        obj_pos0 = self.data.body(spec.obj_body).xpos.copy()
        segments: list[PlannedTrajectory] = []

        p_pre_grasp = mink.SE3.from_rotation_and_translation(
            R_TOP_DOWN, np.array([obj_pos0[0], obj_pos0[1], obj_pos0[2] + spec.pre_height])
        )
        p_grasp = mink.SE3.from_rotation_and_translation(
            R_TOP_DOWN, np.array([obj_pos0[0], obj_pos0[1], obj_pos0[2] + spec.grasp_z_offset])
        )
        p_lift = mink.SE3.from_rotation_and_translation(
            R_TOP_DOWN, np.array([obj_pos0[0], obj_pos0[1], obj_pos0[2] + spec.lift_height])
        )

        place_z_obj = spec.place_surface_z + spec.obj_half_height
        p_pre_place = mink.SE3.from_rotation_and_translation(
            R_TOP_DOWN, np.array([spec.place_xy[0], spec.place_xy[1], place_z_obj + spec.lift_height])
        )
        p_place = mink.SE3.from_rotation_and_translation(
            R_TOP_DOWN, np.array([spec.place_xy[0], spec.place_xy[1], place_z_obj + spec.grasp_z_offset + 0.002])
        )
        p_retract = mink.SE3.from_rotation_and_translation(
            R_TOP_DOWN, np.array([spec.place_xy[0], spec.place_xy[1], place_z_obj + spec.pre_height])
        )

        # Stage 1: Open gripper
        self._set_gripper(f"{side}:open_gripper", side, spec.grip_open, ramp_s=0.35, hold_s=0.10)
        # Stage 2: Move to pre-grasp
        self._move_pose(f"{side}:pre_grasp", side, p_pre_grasp, duration=1.6, planned_segments=segments)
        # Stage 3: Descend to grasp
        self._move_pose(f"{side}:descend_grasp", side, p_grasp, duration=0.9, planned_segments=segments)
        # Stage 4: Close gripper onto object
        self._set_gripper(f"{side}:close_grasp", side, spec.grip_close, ramp_s=0.45, hold_s=0.35)
        # Stage 5: Lift and verify mid-air retention
        self._move_pose(f"{side}:lift", side, p_lift, duration=0.9, planned_segments=segments)
        lifted_pos = self.data.body(spec.obj_body).xpos.copy()
        lift_gain_mm = float((lifted_pos[2] - obj_pos0[2]) * 1000.0)
        # Stage 6: Transport over place zone
        self._move_pose(f"{side}:transport", side, p_pre_place, duration=1.5, planned_segments=segments)
        # Stage 7: Lower onto place zone
        self._move_pose(f"{side}:lower_place", side, p_place, duration=0.9, planned_segments=segments)
        # Stage 8: Open gripper to release
        self._set_gripper(f"{side}:release", side, spec.grip_open, ramp_s=0.35, hold_s=0.15)
        # Stage 9: Retract upward & return arm to home
        self._move_pose(f"{side}:retract", side, p_retract, duration=0.8, planned_segments=segments)
        self._move_joint_home(f"{side}:return_home", side, duration=1.3, planned_segments=segments)
        # Stage 10: Physical settle window after release
        self._hold(f"{side}:settle", duration=0.6)

        # Measure post-release, settled physical state
        final_pos = self.data.body(spec.obj_body).xpos.copy()
        final_mat = self.data.body(spec.obj_body).xmat.reshape(3, 3)
        tilt_deg = float(np.rad2deg(np.arccos(np.clip(final_mat[2, 2], -1.0, 1.0))))
        jnt_id = self.sim_model.joint(f"{spec.obj_body}_freejoint").id
        v_adr = int(self.sim_model.jnt_dofadr[jnt_id])
        lin_speed = float(np.linalg.norm(self.data.qvel[v_adr : v_adr + 3]))
        ang_speed = float(np.linalg.norm(self.data.qvel[v_adr + 3 : v_adr + 6]))
        place_xy_err_mm = float(np.linalg.norm(final_pos[:2] - np.array(spec.place_xy)) * 1000.0)
        z_err_mm = float((final_pos[2] - place_z_obj) * 1000.0)

        # Save concatenated planned path to docs/paths/
        self._save_combined_path(segments, f"grasp_{side}_{spec.obj_body}")

        passed = (
            lift_gain_mm > 80.0
            and place_xy_err_mm < 15.0
            and abs(z_err_mm) < 6.0
            and tilt_deg < 10.0
            and lin_speed < 0.02
        )
        return {
            "side": side,
            "object": spec.obj_body,
            "start_xyz_m": np.round(obj_pos0, 5).tolist(),
            "target_place_xy_m": list(spec.place_xy),
            "lifted_z_gain_mm": round(lift_gain_mm, 2),
            "settled_xyz_m": np.round(final_pos, 5).tolist(),
            "place_xy_err_mm": round(place_xy_err_mm, 2),
            "place_z_err_mm": round(z_err_mm, 2),
            "settled_tilt_deg": round(tilt_deg, 2),
            "settled_lin_speed_m_s": round(lin_speed, 5),
            "settled_ang_speed_rad_s": round(ang_speed, 5),
            "passed": passed,
        }

    def _save_combined_path(self, segments: list[PlannedTrajectory], stem: str) -> None:
        PATHS_DIR.mkdir(parents=True, exist_ok=True)
        t_list, q_list, dq_list, lx_list, rx_list = [], [], [], [], []
        t_offset = 0.0
        for seg in segments:
            t_list.append(seg.t + t_offset)
            q_list.append(seg.q)
            dq_list.append(seg.dq)
            lx_list.append(seg.left_tcp_xyz)
            rx_list.append(seg.right_tcp_xyz)
            t_offset += float(seg.t[-1]) + self.dt

        t_all = np.concatenate(t_list)
        q_all = np.vstack(q_list)
        dq_all = np.vstack(dq_list)
        lx_all = np.vstack(lx_list)
        rx_all = np.vstack(rx_list)

        combined = PlannedTrajectory(
            mode="staged_pick_place",
            dt=self.dt,
            t=t_all,
            q=q_all,
            dq=dq_all,
            left_tcp_xyz=lx_all,
            right_tcp_xyz=rx_all,
            min_dist_mm=min(s.min_dist_mm for s in segments),
            max_ncon=max(s.max_ncon for s in segments),
            collision_free=all(s.collision_free for s in segments),
        )
        combined.save(PATHS_DIR, stem)

    def save_outputs(self, task_results: list[dict[str, object]]) -> dict[str, object]:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        PATHS_DIR.mkdir(parents=True, exist_ok=True)

        # Save closed-loop executed trace CSV
        exec_csv = PATHS_DIR / "grasp_executed_trace.csv"
        with exec_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["t_s", "stage", "rb_x", "rb_y", "rb_z", "bc_x", "bc_y", "bc_z", "l_x", "l_y", "l_z", "r_x", "r_y", "r_z"])
            for row in self.exec_trace:
                writer.writerow(
                    [
                        row["t_s"],
                        row["stage"],
                        *row["red_block"],
                        *row["blue_cylinder"],
                        *row["left_tcp"],
                        *row["right_tcp"],
                    ]
                )

        report = {
            "total_sim_time_s": round(self.sim_time, 3),
            "peak_torque_nm": round(self.peak_tau, 3),
            "all_passed": all(bool(r["passed"]) for r in task_results),
            "tasks": task_results,
            "stages": self.stage_logs,
        }
        report_path = REPORTS_DIR / "grasp_verification_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        if self.record_video and self.frames:
            MEDIA_DIR.mkdir(parents=True, exist_ok=True)
            mp4_path = MEDIA_DIR / "firefly_grasp_demo.mp4"
            imageio.mimsave(str(mp4_path), self.frames, fps=30, macro_block_size=16)
            idxs = [0, len(self.frames) // 4, len(self.frames) // 2, (3 * len(self.frames)) // 4, len(self.frames) - 1]
            # 2x2 sheet covering key moments (lift red, place red, lift blue, settled)
            pick_idxs = [
                int(len(self.frames) * 0.22),
                int(len(self.frames) * 0.42),
                int(len(self.frames) * 0.72),
                len(self.frames) - 1,
            ]
            tiles = [Image.fromarray(self.frames[i]).resize((640, 360)) for i in pick_idxs]
            sheet = Image.new("RGB", (1280, 720))
            sheet.paste(tiles[0], (0, 0))
            sheet.paste(tiles[1], (640, 0))
            sheet.paste(tiles[2], (0, 360))
            sheet.paste(tiles[3], (640, 360))
            sheet_path = MEDIA_DIR / "firefly_grasp_keyframes.png"
            sheet.save(str(sheet_path))
            print(f"Saved grasp video: {mp4_path} ({len(self.frames)} frames)")
            print(f"Saved grasp sheet: {sheet_path}")

        return report


def run_dual_arm_pick_and_place(record_video: bool = True) -> dict[str, object]:
    """Run sequential left + right pick-and-place verification in MuJoCo."""
    executor = GraspExecutor(record_video=record_video)
    try:
        res_left = executor.execute_pick_and_place(
            PickPlaceSpec(
                side="left",
                obj_body="red_block",
                place_xy=(0.41, 0.07),
                obj_half_height=0.022,
                grasp_z_offset=0.010,
            )
        )
        res_right = executor.execute_pick_and_place(
            PickPlaceSpec(
                side="right",
                obj_body="blue_cylinder",
                place_xy=(0.41, -0.07),
                obj_half_height=0.030,
                grasp_z_offset=0.012,
            )
        )
        report = executor.save_outputs([res_left, res_right])
    finally:
        executor.close()

    print("\n=== DUAL-ARM PICK & PLACE VERIFICATION ===")
    for t in report["tasks"]:
        tag = "PASS" if t["passed"] else "FAIL"
        print(
            f"  [{tag}] {t['side']:5s} ({t['object']:13s}): "
            f"lift={t['lifted_z_gain_mm']:.1f}mm | "
            f"xy_err={t['place_xy_err_mm']:.2f}mm | "
            f"z_err={t['place_z_err_mm']:.2f}mm | "
            f"tilt={t['settled_tilt_deg']:.2f}deg | "
            f"v={t['settled_lin_speed_m_s']:.4f}m/s"
        )
    print(f"Peak torque: {report['peak_torque_nm']:.2f} Nm | Total sim time: {report['total_sim_time_s']:.2f} s")
    return report


if __name__ == "__main__":
    run_dual_arm_pick_and_place(record_video=True)
