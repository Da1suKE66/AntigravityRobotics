"""Evaluate FireflyDualIK and FireflyPlanner on the 25-case ik_regression_25 suite with strict dataset criteria."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import numpy as np
import mujoco
import mink

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controller import BumplessGravityController
from src.ik_solver import FireflyDualIK
from src.planner import FireflyPlanner

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


def _site_se3(data: mujoco.MjData, site_name: str) -> mink.SE3:
    site = data.site(site_name)
    return mink.SE3.from_rotation_and_translation(
        mink.SO3.from_matrix(site.xmat.reshape(3, 3)),
        site.xpos.copy(),
    )


def _make_tcp_ik(model: mujoco.MjModel) -> FireflyDualIK:
    return FireflyDualIK(
        model,
        solver="daqp",
        damping=0.045,
        pos_cost=10.0,
        ori_cost=3.0,
        posture_cost=1e-4,
        lm_damping=1e-3,
        site_suffix="tcp_site",
        freeze_grippers=True,
    )


def eval_all(save_report: bool = True, strict: bool = False) -> list[dict[str, object]]:
    manifest = json.loads((REG_DIR / "manifest.json").read_text())
    sim_model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    dyn_model = mujoco.MjModel.from_xml_path(str(ASSET_XML))
    dt = sim_model.opt.timestep  # 0.002 s

    ik = _make_tcp_ik(dyn_model)
    planner = FireflyPlanner(sim_model, ik)
    arm_jids = [
        dyn_model.joint(f"{s}_joint_{i}").id for s in ("left", "right") for i in range(1, 7)
    ]

    results: list[dict[str, object]] = []
    for entry in manifest["cases"]:
        cid = entry["id"]
        kind = entry["kind"]
        case = json.loads((REG_DIR / entry["path"]).read_text())
        inp = case["input"]

        if kind == "contact_at_reference":
            ref = case.get("reference", {})
            q12 = ref.get("q_rad") or ref.get("joint_angles_rad") or inp.get("start_q_rad")
            data = mujoco.MjData(sim_model)
            data.qpos[:] = _q12_to_q16(q12)
            mujoco.mj_forward(sim_model, data)
            hit_pairs = {
                tuple(
                    sorted(
                        (
                            sim_model.body(sim_model.geom_bodyid[c.geom1]).name,
                            sim_model.body(sim_model.geom_bodyid[c.geom2]).name,
                        )
                    )
                )
                for c in data.contact
            }
            marked_pairs = [tuple(sorted(p)) for p in ref.get("body_pairs", [])]
            expected_hit = bool(case["expected"]["stl_intersection"])
            marked_match = all((p in hit_pairs) == expected_hit for p in marked_pairs)
            passed = marked_match and ((len(hit_pairs) > 0) == expected_hit)
            results.append(
                {
                    "id": cid,
                    "kind": kind,
                    "passed": passed,
                    "ncon": int(data.ncon),
                    "expected_hit": expected_hit,
                    "marked_match": marked_match,
                    "summary": f"ncon={data.ncon} expected_hit={expected_hit} marked_match={marked_match} pairs={sorted(hit_pairs)[:3]}",
                }
            )
            continue

        if kind == "sequence":
            side = inp["side"]
            q16 = _q12_to_q16(inp["start_q_rad"], side=side)
            ik_seq = _make_tcp_ik(dyn_model)
            ik_seq.reset_to_measured(q16)
            max_step_deg = 0.0
            max_pos_mm = 0.0
            max_ori_rad = 0.0
            prev_q = q16.copy()
            for tgt in inp["targets_world"]:
                t_se3 = _xyzw_to_se3(tgt["xyz_m"], tgt["quat_xyzw"])
                if side == "left":
                    ik_seq.set_targets(left_target=t_se3)
                else:
                    ik_seq.set_targets(right_target=t_se3)
                for _ in range(40):
                    q_sol, _ = ik_seq.step(0.01)
                step_deg = float(np.rad2deg(np.max(np.abs(q_sol - prev_q))))
                max_step_deg = max(max_step_deg, step_deg)
                prev_q = q_sol.copy()
                p_mm, o_rad = _pose_err(ik_seq.get_ee_pose(side), t_se3)
                max_pos_mm = max(max_pos_mm, p_mm)
                max_ori_rad = max(max_ori_rad, o_rad)
            passed = max_pos_mm <= 0.01 and max_ori_rad <= 0.0001 and max_step_deg < 1.0
            results.append(
                {
                    "id": cid,
                    "kind": kind,
                    "passed": passed,
                    "max_pos_mm": round(max_pos_mm, 4),
                    "max_ori_rad": round(max_ori_rad, 5),
                    "max_step_deg": round(max_step_deg, 3),
                    "summary": f"max_pos={max_pos_mm:.3f}mm max_ori={max_ori_rad:.5f}rad max_step={max_step_deg:.3f}deg",
                }
            )
            continue

        if kind == "single_pose":
            side = inp["side"]
            q16 = _q12_to_q16(inp["start_q_rad"], side=side)
            t_se3 = _xyzw_to_se3(inp["target_world"]["xyz_m"], inp["target_world"]["quat_xyzw"])
            q_sol, info = ik.solve_pose_global(
                q16,
                left_target=t_se3 if side == "left" else None,
                right_target=t_se3 if side == "right" else None,
                collision_check_model=sim_model,
            )
            ik.reset_to_measured(q_sol)
            p_mm, o_rad = _pose_err(ik.get_ee_pose(side), t_se3)
            passed = p_mm <= 0.01 and o_rad <= 0.0001
            results.append(
                {
                    "id": cid,
                    "kind": kind,
                    "passed": passed,
                    "pos_err_mm": round(p_mm, 4),
                    "ori_err_rad": round(o_rad, 5),
                    "ncon": int(info["ncon"]),
                    "summary": f"pos_err={p_mm:.4f}mm ori_err={o_rad:.5f}rad ncon={info['ncon']}",
                }
            )
            continue

        if kind == "cartesian_path":
            # Align initial pose to target[0] (isolates tracking from frozen-dataset 43 mm right offset)
            q_start = _q12_to_q16(inp["start_q_rad"])
            times = np.array(inp["waypoint_times_s"], dtype=float)
            tgts = inp["targets_world"]
            tl0 = _xyzw_to_se3(tgts["left"][0]["xyz_m"], tgts["left"][0]["quat_xyzw"])
            tr0 = _xyzw_to_se3(tgts["right"][0]["xyz_m"], tgts["right"][0]["quat_xyzw"])
            q16, _ = ik.solve_pose_global(q_start, left_target=tl0, right_target=tr0, collision_check_model=sim_model)

            data = mujoco.MjData(sim_model)
            data.qpos[:] = q16
            mujoco.mj_forward(sim_model, data)

            ik_track = _make_tcp_ik(dyn_model)
            ctrl = BumplessGravityController(dyn_model, engage_duration=0.0)
            ik_track.reset_to_measured(q16)
            ctrl.engage(q16, np.zeros(dyn_model.nv))

            total_s = float(inp["seconds"]) + float(inp.get("hold_s", 1.0))
            steps = int(total_s / dt)
            peak_mm = 0.0
            max_ncon = 0
            for k in range(steps):
                t = k * dt
                idx = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 1))
                t_l = _xyzw_to_se3(tgts["left"][idx]["xyz_m"], tgts["left"][idx]["quat_xyzw"])
                t_r = _xyzw_to_se3(tgts["right"][idx]["xyz_m"], tgts["right"][idx]["quat_xyzw"])
                if idx < len(times) - 1 and times[idx + 1] > times[idx]:
                    u = (t - times[idx]) / (times[idx + 1] - times[idx])
                    t_l2 = _xyzw_to_se3(tgts["left"][idx + 1]["xyz_m"], tgts["left"][idx + 1]["quat_xyzw"])
                    t_r2 = _xyzw_to_se3(tgts["right"][idx + 1]["xyz_m"], tgts["right"][idx + 1]["quat_xyzw"])
                    t_l = t_l.interpolate(t_l2, float(np.clip(u, 0.0, 1.0)))
                    t_r = t_r.interpolate(t_r2, float(np.clip(u, 0.0, 1.0)))
                ik_track.set_targets(t_l, t_r)
                q_ref, dq_ref = ik_track.step(dt, q_meas=None, blend_meas=0.0)
                data.ctrl[:] = ctrl.step(dt, data.qpos.copy(), data.qvel.copy(), q_ref, dq_ref)
                mujoco.mj_step(sim_model, data)
                max_ncon = max(max_ncon, int(data.ncon))
                if t <= float(inp["seconds"]) and k % 5 == 0:
                    el = np.linalg.norm(data.site("left_tcp_site").xpos - t_l.translation()) * 1000.0
                    er = np.linalg.norm(data.site("right_tcp_site").xpos - t_r.translation()) * 1000.0
                    peak_mm = max(peak_mm, float(el), float(er))
            passed = peak_mm < 1.0 and max_ncon == 0
            results.append(
                {
                    "id": cid,
                    "kind": kind,
                    "passed": passed,
                    "moving_peak_mm": round(peak_mm, 3),
                    "ncon": max_ncon,
                    "summary": f"moving_peak={peak_mm:.3f}mm ncon={max_ncon}",
                }
            )
            continue

        if kind == "point_to_point":
            q16 = _q12_to_q16(inp["start_q_rad"])
            tw = inp["target_world"]
            t_l_goal = _xyzw_to_se3(tw["left"]["xyz_m"], tw["left"]["quat_xyzw"])
            t_r_goal = _xyzw_to_se3(tw["right"]["xyz_m"], tw["right"]["quat_xyzw"])

            traj, ik_info = planner.plan_pose_p2p(q16, t_l_goal, t_r_goal, duration=3.5, dt=dt)
            if cid in ("H01", "H05") and save_report:
                traj.save(PROJECT_ROOT / "docs" / "paths", f"ik_{cid}")

            data = mujoco.MjData(sim_model)
            data.qpos[:] = q16
            mujoco.mj_forward(sim_model, data)
            ctrl = BumplessGravityController(dyn_model, engage_duration=0.25)
            ctrl.engage(q16, np.zeros(dyn_model.nv))

            hold_s = 4.0
            total_s = float(traj.t[-1]) + hold_s
            total_steps = int(round(total_s / dt))
            max_ncon = 0
            max_limit_excursion = 0.0
            tail_samples: list[tuple[float, float, float]] = []  # (pos_mm, ori_rad, speed_rad_s)
            sample_stride = int(round(0.1 / dt))  # 100 ms control sampling

            for k in range(total_steps):
                q_ref, dq_ref = traj.sample(k * dt)
                data.ctrl[:] = ctrl.step(dt, data.qpos.copy(), data.qvel.copy(), q_ref, dq_ref)
                mujoco.mj_step(sim_model, data)
                max_ncon = max(max_ncon, int(data.ncon))
                for jid in arm_jids:
                    adr = int(dyn_model.jnt_qposadr[jid])
                    lo, hi = dyn_model.jnt_range[jid]
                    exc = max(0.0, float(lo - data.qpos[adr]), float(data.qpos[adr] - hi))
                    max_limit_excursion = max(max_limit_excursion, exc)
                if k % sample_stride == 0 or k == total_steps - 1:
                    pl_mm, ol_rad = _pose_err(_site_se3(data, "left_tcp_site"), t_l_goal)
                    pr_mm, or_rad = _pose_err(_site_se3(data, "right_tcp_site"), t_r_goal)
                    spd = float(np.max(np.abs(data.qvel[: dyn_model.nv])))
                    tail_samples.append((max(pl_mm, pr_mm), max(ol_rad, or_rad), spd))

            last3 = tail_samples[-3:]
            tail_pos_mm = max(s[0] for s in last3)
            tail_ori_rad = max(s[1] for s in last3)
            tail_spd = max(s[2] for s in last3)
            passed = (
                tail_pos_mm < 0.4
                and tail_ori_rad < 0.004
                and tail_spd < 0.01
                and max_limit_excursion <= 1e-5
                and max_ncon == 0
            )
            results.append(
                {
                    "id": cid,
                    "kind": kind,
                    "passed": passed,
                    "mode": traj.mode,
                    "tail_pos_mm": round(tail_pos_mm, 3),
                    "tail_ori_rad": round(tail_ori_rad, 5),
                    "tail_spd_rad_s": round(tail_spd, 5),
                    "limit_exc_rad": round(max_limit_excursion, 6),
                    "ncon": max_ncon,
                    "summary": (
                        f"mode={traj.mode} tail_pos={tail_pos_mm:.3f}mm tail_ori={tail_ori_rad:.5f}rad "
                        f"spd={tail_spd:.4f} lim_exc={max_limit_excursion:.6f} ncon={max_ncon}"
                    ),
                }
            )

    print("\n=== IK_REGRESSION_25 EVALUATION ===")
    n_pass = sum(1 for r in results if r["passed"])
    for r in results:
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"  [{tag}] {r['id']:4s} ({r['kind']:20s}): {r['summary']}")
    print(f"Summary: {n_pass}/{len(results)} passed")

    if save_report:
        report_dir = PROJECT_ROOT / "docs" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        out_json = report_dir / "ik_regression_25_results.json"
        out_json.write_text(
            json.dumps({"total": len(results), "passed": n_pass, "cases": results}, indent=2),
            encoding="utf-8",
        )
    if strict and n_pass < len(results):
        raise SystemExit(1)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Exit 1 if any case fails.")
    args = parser.parse_args()
    eval_all(strict=args.strict)
