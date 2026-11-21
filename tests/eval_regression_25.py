"""Evaluate Mink+DAQP IK and BumplessGravityController on the 25-case ik_regression_25 suite."""

from __future__ import annotations

import json
import pathlib
import sys
import numpy as np
import mujoco
import mink

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controller import BumplessGravityController
from src.ik_solver import FireflyDualIK, quintic_blend

ASSET_XML = PROJECT_ROOT / "model" / "firefly_dual.xml"
SCENE_XML = PROJECT_ROOT / "model" / "scene.xml"
REG_DIR = PROJECT_ROOT / "ik_regression_25"


def _xyzw_to_se3(xyz: list[float], xyzw: list[float]) -> mink.SE3:
    x, y, z, w = xyzw
    return mink.SE3(wxyz_xyz=np.array([w, x, y, z, xyz[0], xyz[1], xyz[2]], dtype=float))


def _q12_to_q16(q: list[float], side: str | None = None) -> np.ndarray:
    q16 = np.zeros(16, dtype=float)
    q16[6:8] = 0.25
    q16[14:16] = 0.25
    if len(q) == 12:
        q16[0:6] = q[0:6]
        q16[8:14] = q[6:12]
    elif len(q) == 6:
        if side == "right":
            q16[8:14] = q
        else:
            q16[0:6] = q
    else:
        raise ValueError(f"Unexpected start_q_rad length: {len(q)}")
    return q16


def _pose_err(actual: mink.SE3, target: mink.SE3) -> tuple[float, float]:
    pos_mm = float(np.linalg.norm(actual.translation() - target.translation()) * 1000.0)
    rot_err = float(np.linalg.norm((actual.rotation().inverse() @ target.rotation()).log()))
    return pos_mm, rot_err


class _TCPDualIK(FireflyDualIK):
    """FireflyDualIK configured to track the 170 mm TCP sites (left_tcp_site / right_tcp_site)."""

    def __init__(self, model: mujoco.MjModel) -> None:
        super().__init__(model, solver="daqp", damping=1e-4, pos_cost=2.0, ori_cost=0.6, posture_cost=5e-3)
        self.left_task = mink.FrameTask(
            frame_name="left_tcp_site", frame_type="site", position_cost=2.0, orientation_cost=0.6, lm_damping=1.0
        )
        self.right_task = mink.FrameTask(
            frame_name="right_tcp_site", frame_type="site", position_cost=2.0, orientation_cost=0.6, lm_damping=1.0
        )

    def get_ee_pose(self, side: str) -> mink.SE3:
        return self.configuration.get_transform_frame_to_world(f"{side}_tcp_site", "site")


def eval_all() -> None:
    manifest = json.loads((REG_DIR / "manifest.json").read_text())
    sim_model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    dyn_model = mujoco.MjModel.from_xml_path(str(ASSET_XML))
    dt = sim_model.opt.timestep  # 0.002s

    results = []
    for entry in manifest["cases"]:
        cid = entry["id"]
        kind = entry["kind"]
        case = json.loads((REG_DIR / entry["path"]).read_text())
        inp = case["input"]

        if kind == "contact_at_reference":
            # Check static self/scene collisions at the reference pose
            ref = case.get("reference", {})
            q12 = ref.get("q_rad") or ref.get("joint_angles_rad") or inp.get("start_q_rad")
            data = mujoco.MjData(sim_model)
            data.qpos[:] = _q12_to_q16(q12)
            mujoco.mj_forward(sim_model, data)
            pairs = [(sim_model.geom(c.geom1).name or f"g{c.geom1}", sim_model.geom(c.geom2).name or f"g{c.geom2}") for c in data.contact]
            results.append((cid, kind, f"ncon={data.ncon} pairs={pairs[:3]}"))
            continue

        if kind == "sequence":
            # H02: 12-frame sequence on one side
            side = inp["side"]
            q16 = _q12_to_q16(inp["start_q_rad"], side=side)
            ik = _TCPDualIK(dyn_model)
            ik.reset_to_measured(q16)
            max_step_deg = 0.0
            max_pos_mm = 0.0
            max_ori_rad = 0.0
            prev_q = q16.copy()
            for tgt in inp["targets_world"]:
                t_se3 = _xyzw_to_se3(tgt["xyz_m"], tgt["quat_xyzw"])
                if side == "left":
                    ik.set_targets(left_target=t_se3)
                else:
                    ik.set_targets(right_target=t_se3)
                for _ in range(40):
                    q_sol, _ = ik.step(0.01)
                step_deg = float(np.rad2deg(np.max(np.abs(q_sol - prev_q))))
                max_step_deg = max(max_step_deg, step_deg)
                prev_q = q_sol.copy()
                p_mm, o_rad = _pose_err(ik.get_ee_pose(side), t_se3)
                max_pos_mm = max(max_pos_mm, p_mm)
                max_ori_rad = max(max_ori_rad, o_rad)
            results.append((cid, kind, f"max_pos={max_pos_mm:.3f}mm max_ori={max_ori_rad:.4f}rad max_step={max_step_deg:.3f}deg"))
            continue

        if kind == "single_pose":
            # M01-M05: single-arm pose from start_q_rad
            side = inp["side"]
            q16 = _q12_to_q16(inp["start_q_rad"], side=side)
            ik = _TCPDualIK(dyn_model)
            ik.reset_to_measured(q16)
            t_se3 = _xyzw_to_se3(inp["target_world"]["xyz_m"], inp["target_world"]["quat_xyzw"])
            if side == "left":
                ik.set_targets(left_target=t_se3)
            else:
                ik.set_targets(right_target=t_se3)
            for _ in range(400):
                ik.step(0.01)
            p_mm, o_rad = _pose_err(ik.get_ee_pose(side), t_se3)
            results.append((cid, kind, f"pos_err={p_mm:.4f}mm ori_err={o_rad:.5f}rad"))
            continue

        if kind == "cartesian_path":
            # E01-E05: timed waypoints + hold_s in full closed-loop physics
            q16 = _q12_to_q16(inp["start_q_rad"])
            data = mujoco.MjData(sim_model)
            data.qpos[:] = q16
            mujoco.mj_forward(sim_model, data)

            ik = _TCPDualIK(dyn_model)
            ctrl = BumplessGravityController(dyn_model, engage_duration=0.0)
            ik.reset_to_measured(q16)
            ctrl.engage(q16, np.zeros(16))

            times = np.array(inp["waypoint_times_s"], dtype=float)
            tgts = inp["targets_world"]
            total_s = float(inp["seconds"]) + float(inp.get("hold_s", 1.0))
            steps = int(total_s / dt)
            peak_mm = 0.0
            max_ncon = 0
            for k in range(steps):
                t = k * dt
                idx = int(np.searchsorted(times, t, side="right")) - 1
                idx = int(np.clip(idx, 0, len(times) - 1))
                t_l = _xyzw_to_se3(tgts["left"][idx]["xyz_m"], tgts["left"][idx]["quat_xyzw"])
                t_r = _xyzw_to_se3(tgts["right"][idx]["xyz_m"], tgts["right"][idx]["quat_xyzw"])
                if idx < len(times) - 1 and times[idx + 1] > times[idx]:
                    u = (t - times[idx]) / (times[idx + 1] - times[idx])
                    t_l2 = _xyzw_to_se3(tgts["left"][idx + 1]["xyz_m"], tgts["left"][idx + 1]["quat_xyzw"])
                    t_r2 = _xyzw_to_se3(tgts["right"][idx + 1]["xyz_m"], tgts["right"][idx + 1]["quat_xyzw"])
                    t_l = t_l.interpolate(t_l2, float(np.clip(u, 0.0, 1.0)))
                    t_r = t_r.interpolate(t_r2, float(np.clip(u, 0.0, 1.0)))
                ik.set_targets(t_l, t_r)
                q_ref, dq_ref = ik.step(dt, q_meas=data.qpos.copy(), blend_meas=0.02)
                data.ctrl[:] = ctrl.step(dt, data.qpos.copy(), data.qvel.copy(), q_ref, dq_ref)
                mujoco.mj_step(sim_model, data)
                max_ncon = max(max_ncon, int(data.ncon))
                if t <= float(inp["seconds"]) and k % 5 == 0:
                    el = np.linalg.norm(data.site("left_tcp_site").xpos - t_l.translation()) * 1000.0
                    er = np.linalg.norm(data.site("right_tcp_site").xpos - t_r.translation()) * 1000.0
                    peak_mm = max(peak_mm, float(el), float(er))
            results.append((cid, kind, f"moving_peak={peak_mm:.2f}mm ncon={max_ncon}"))
            continue

        if kind == "point_to_point":
            # H01, H03-H05, P01-P05: dual-arm P2P in full physics
            q16 = _q12_to_q16(inp["start_q_rad"])
            data = mujoco.MjData(sim_model)
            data.qpos[:] = q16
            mujoco.mj_forward(sim_model, data)

            ik = _TCPDualIK(dyn_model)
            ctrl = BumplessGravityController(dyn_model, engage_duration=0.25)
            t_l0, t_r0 = ik.reset_to_measured(q16)
            ctrl.engage(q16, np.zeros(16))

            tw = inp["target_world"]
            t_l_goal = _xyzw_to_se3(tw["left"]["xyz_m"], tw["left"]["quat_xyzw"])
            t_r_goal = _xyzw_to_se3(tw["right"]["xyz_m"], tw["right"]["quat_xyzw"])

            move_s = 3.5
            hold_s = 1.5
            max_ncon = 0
            for k in range(int((move_s + hold_s) / dt)):
                u = min(1.0, (k * dt) / move_s)
                s, _ = quintic_blend(u)
                ik.set_targets(t_l0.interpolate(t_l_goal, s), t_r0.interpolate(t_r_goal, s))
                q_ref, dq_ref = ik.step(dt, q_meas=data.qpos.copy(), blend_meas=0.02)
                data.ctrl[:] = ctrl.step(dt, data.qpos.copy(), data.qvel.copy(), q_ref, dq_ref)
                mujoco.mj_step(sim_model, data)
                max_ncon = max(max_ncon, int(data.ncon))

            el = float(np.linalg.norm(data.site("left_tcp_site").xpos - t_l_goal.translation()) * 1000.0)
            er = float(np.linalg.norm(data.site("right_tcp_site").xpos - t_r_goal.translation()) * 1000.0)
            results.append((cid, kind, f"left={el:.2f}mm right={er:.2f}mm ncon={max_ncon}"))

    print("\n=== IK_REGRESSION_25 EVALUATION ===")
    for cid, kind, summary in results:
        print(f"  {cid:4s} ({kind:20s}): {summary}")


if __name__ == "__main__":
    eval_all()
