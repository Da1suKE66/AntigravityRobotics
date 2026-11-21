"""Replay frozen Firefly IK fixtures with the project's calibrated MuJoCo environment."""

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np


DATA = Path(__file__).resolve().parent
SIDES = ("left", "right")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def native_pose(pose):
    return dict(xyz_m=pose["xyz_m"], quat_wxyz=np.roll(pose["quat_xyzw"], 1).tolist())


def arm_fk_ok(sim, side, joints, target, tol):
    from check_bundled_ik import measure_pose, sim_fk

    low, high = sim.limits[sim.arm_drives[side]].T
    error = measure_pose(*sim_fk(sim, side, joints), native_pose(target))
    return bool(np.all((low <= joints) & (joints <= high)) and
                error["position_error_mm"] <= tol["branch_fk_mm"] and
                np.deg2rad(error["orientation_error_deg"]) <= tol["branch_fk_rad"])


def replay_plan(sim, plan, goals, tol):
    from check_ik_corners import joint_telemetry

    if plan is None:
        return None
    run = joint_telemetry(sim, np.array(plan["goal_q_rad"]), goals, plan["ramp_s"], via=plan["via_q_rad"])
    metrics = {k: run[k] for k in ("contacts", "warnings", "tail_position_mm", "tail_orientation_deg",
                                   "tail_joint_speed_rad_s", "worst_limit_excursion_deg", "max_speed_to_spec_ratio")}
    metrics["verified"] = bool(run["passed"] and
                               np.deg2rad(run["worst_limit_excursion_deg"]) <= tol["max_joint_excursion_rad"] and
                               run["max_speed_to_spec_ratio"] <= tol["max_speed_to_spec_ratio"])
    return metrics


def endpoint(sim, case, temp, tol):
    from check_bundled_ik import select_simulated_branch, target_diagnostics

    inputs = case["input"]
    path = temp / "target.json"
    path.write_text(json.dumps(inputs["target_world"]))
    goals = {s: native_pose(inputs["target_world"][s]) for s in SIDES}
    gate = select_simulated_branch(sim, path, seed=inputs["start_q_rad"])
    q = case["reference"]["q_rad"]
    drives = np.r_[sim.arm_drives["left"], sim.arm_drives["right"]]
    command = sim.model.key("home").ctrl.copy()
    command[drives] = q
    ref = target_diagnostics(sim, command, goals)
    witnessed = all(arm_fk_ok(sim, s, q[6*i:6*i+6], inputs["target_world"][s], tol)
                    for i, s in enumerate(SIDES))
    plan = gate["selected_trajectory"]
    return dict(gate_status=gate["status"], candidate_pairs=gate["candidate_pair_count"],
                rejection_reasons=sorted({r for p in gate["pairs"] for r in p["reasons"]}),
                reference_fk_valid=witnessed, reference_contacts=ref["target_overlaps"],
                plan=plan, replay=replay_plan(sim, plan, goals, tol))


def mesh_checks(sim, case, root):
    from check_bundled_ik import mujoco

    ref = case["reference"]
    model = sim.model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.qpos[sim.qadr[np.r_[sim.arm_drives["left"], sim.arm_drives["right"]]]] = ref["q_rad"]
    mujoco.mj_forward(model, data)
    # Compare body pairs so the original triangles can still be scored if the
    # production scene later uses several convex pieces per body.
    active = {}
    for contact in data.contact:
        if contact.dist < 0:
            pair = tuple(sorted(model.body(model.geom_bodyid[g]).name for g in (contact.geom1, contact.geom2)))
            active[pair] = min(active.get(pair, 0.), float(contact.dist * 1000))
    objects = {}
    for name, frame in ref["source_meshes"].items():
        transform = np.empty(9)
        mujoco.mju_quat2Mat(transform, np.roll(frame["body_quat_xyzw"], 1))
        body = data.body(name)
        rotation = body.xmat.reshape(3, 3)
        objects[name] = dict(body=name, mesh=frame["mesh"], scale=frame["scale"],
                             world_xyz_m=(body.xpos + rotation @ frame["body_xyz_m"]).tolist(),
                             world_rotation=(rotation @ transform.reshape(3, 3)).tolist())
    pairs = []
    for bodies in ref["body_pairs"]:
        depth = active.get(tuple(sorted(bodies)))
        pairs.append(dict(bodies=bodies, mujoco_contact=depth is not None, mujoco_signed_mm=depth,
                          objects=[objects[b] for b in bodies]))
    proc = subprocess.run([root / ".venv/bin/python", root / "examples/check_sim_collision_meshes.py", "--stl"],
                          input=json.dumps({"pairs": pairs}), capture_output=True, text=True, check=True)
    return [{k: v for k, v in p.items() if k != "objects"} for p in json.loads(proc.stdout)["pairs"]]


def point_case(sim, case, root, temp, tol):
    metrics = endpoint(sim, case, temp, tol)
    plan_ok = bool(metrics["replay"] and metrics["replay"]["verified"])
    if case["kind"] == "point_to_point":
        return ("pass" if plan_ok and metrics["reference_fk_valid"] else "regression"), metrics
    pairs = metrics["mesh_pairs"] = mesh_checks(sim, case, root)
    hit = case["expected"]["stl_intersection"]
    if (not metrics["reference_fk_valid"] or metrics["candidate_pairs"] == 0 or
            any(p["stl_surfaces_intersect"] != hit for p in pairs) or (metrics["plan"] and not plan_ok)):
        return "regression", metrics
    if hit:
        marked = all(p["mujoco_contact"] for p in pairs)
        safe_decision = plan_ok or "target_collision" in metrics["rejection_reasons"]
        return ("pass" if marked and safe_decision else "regression"), metrics
    return ("known_issue" if any(p["mujoco_contact"] for p in pairs) else "improved"), metrics


def numeric_motion(sim, side, solved, goals, tol):
    from check_bundled_ik import plan_pair, target_diagnostics

    initial = sim.model.key("home").ctrl.copy()
    command = initial.copy()
    command[sim.arm_drives[side]] = solved
    static = target_diagnostics(sim, command, goals)
    if static["target_overlaps"] or static["static_torques_outside_limits"]:
        return dict(verified=False, static_contacts=static["target_overlaps"],
                    static_torque_violations=static["static_torques_outside_limits"])
    plan, _, _, reasons = plan_pair(sim, initial, command, goals, None, None)
    run = replay_plan(sim, plan, goals, tol)
    return dict(verified=bool(run and run["verified"]), plan=plan, replay=run, reasons=reasons)


def pose_case(sim, case, temp, tol):
    from check_bundled_ik import select_simulated_branch
    from evaluate_ik_motion import solve_and_score

    inp = case["input"]
    row = dict(side=inp["side"], target=inp["target_world"], seed=inp["start_q_rad"], q_ref=case["reference"]["q_rad"])
    solve_and_score(sim, [row], [])
    res = row["result"]
    num = res["numerical"]
    valid_witness = arm_fk_ok(sim, row["side"], row["q_ref"], row["target"], tol)
    metrics = dict(branch_count=res["count"], complete=res["complete"], invalid_candidates=res["invalid_candidates"],
                   branch_q_rad=res["nearest_home_q"], worst_branch_fk_mm=res["worst_fk_mm"],
                   worst_branch_fk_deg=res["worst_fk_deg"], reference_fk_valid=valid_witness, numerical=num,
                   branch_rejection_reasons=res["rejected_reasons"])
    if not (valid_witness and num["success"] and num["independently_in_tolerance"]) or res["invalid_candidates"]:
        return "regression", metrics
    if case["expected"].get("verify_numeric_trajectory"):
        other = "right" if row["side"] == "left" else "left"
        targets = {row["side"]: row["target"], other: inp["other_arm_at_home_world"]}
        goals = {s: native_pose(targets[s]) for s in SIDES}
        path = temp / "target.json"
        path.write_text(json.dumps(targets))
        gate = select_simulated_branch(sim, path)
        run = replay_plan(sim, gate["selected_trajectory"], goals, tol)
        metrics["public_planner"] = dict(status=gate["status"], plan=gate["selected_trajectory"], replay=run)
        metrics["numeric_motion"] = numeric_motion(sim, row["side"], num["q_rad"], goals, tol)
        if run and not run["verified"]:
            return "regression", metrics
        if not run:
            return ("known_issue" if metrics["numeric_motion"]["verified"] else "regression"), metrics
    if res["count"]:
        return "improved", metrics
    if res["complete"] and not case["recorded"]["complete"]:
        return "regression", metrics
    return "known_issue", metrics


def sequence_case(sim, case, tol):
    from evaluate_ik_motion import solve_and_score

    inp = case["input"]
    path = dict(side=inp["side"], initial=inp["start_q_rad"], targets=inp["targets_world"])
    solve_and_score(sim, [], [path])
    valid = arm_fk_ok(sim, path["side"], path["initial"], inp["start_target_world"], tol)
    metrics = dict(missing_frames=path["missing_frames"], max_fk_mm=path["geometric_max_mm"],
                   max_fk_deg=path["geometric_max_deg"], chosen_q_rad=path["chosen"], start_fk_valid=valid)
    if path["missing_frames"] or not valid:
        return "regression", metrics
    steps = np.abs(np.rad2deg(np.diff([path["initial"], *path["chosen"]], axis=0)))
    metrics["joint_step_deg_by_frame"] = steps.max(axis=1).tolist()
    ok = (steps.max() < case["expected"]["max_joint_step_deg"] and metrics["max_fk_mm"] <= tol["branch_fk_mm"] and
          np.deg2rad(metrics["max_fk_deg"]) <= tol["branch_fk_rad"])
    return ("pass" if ok else "regression"), metrics


def tracking_case(sim, case, tol):
    from check_bundled_ik import measure_pose
    from evaluate_ik_motion import curve_pose, moving_replay, solve_and_score

    inp = case["input"]
    seconds = inp["seconds"]
    times = inp["waypoint_times_s"]
    dt = sim.model.opt.timestep * sim.substeps
    if inp["hold_s"] != 4 or not np.allclose(times, np.arange(round(seconds/dt)+1) * dt, rtol=0, atol=1e-12):
        raise ValueError("Timing in the project replay differs from the fixture")
    paths = []
    for i, side in enumerate(SIDES):
        targets = inp["targets_world"][side]
        initial = inp["start_q_rad"][6*i:6*i+6]
        if not arm_fk_ok(sim, side, initial, targets[0], tol):
            raise ValueError(f"Start TCP differs from the fixture for {side}")
        base = native_pose(targets[0])
        for t, target in zip(times, targets, strict=True):
            analytic = curve_pose(base, side, inp["curve"], t / seconds)
            err = measure_pose(np.asarray(analytic["xyz_m"]), np.asarray(analytic["quat_wxyz"]), native_pose(target))
            if err["position_error_mm"] > 1e-8 or err["orientation_error_deg"] > 1e-8:
                raise ValueError("Continuous reference curve disagrees with frozen waypoints")
        paths.append(dict(side=side, targets=targets, initial=initial))
    solve_and_score(sim, [], paths)
    run = moving_replay(sim, inp["curve"], seconds, *paths)
    metrics = {k: v for k, v in run.items() if k != "trace_50ms"}
    if not run["executed"]:
        return "regression", metrics
    healthy = (run["settled"] and not run["warnings"] and run["peak_speed_ratio"] <= tol["max_speed_to_spec_ratio"]
               and np.deg2rad(run["max_limit_excursion_deg"]) <= tol["max_joint_excursion_rad"])
    if not healthy:
        return "regression", metrics
    peak = run["moving_position_mm"]["max"]
    if not run["contacts"] and peak < case["expected"]["moving_peak_position_mm_below"]:
        return "improved", metrics
    saved = case["recorded"]
    new_contacts = set(map(tuple, run["contacts"])) - set(map(tuple, saved["contacts"]))
    worse = (new_contacts or peak > saved["moving_position_mm"]["max"] + tol["tracking_baseline_roundoff_mm"] or
             run["moving_angle_deg"]["max"] > saved["moving_angle_deg"]["max"] + tol["tracking_baseline_roundoff_deg"])
    return ("regression" if worse else "known_issue"), metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, help="RoboControl-IK project root (auto-detected in the checkout)")
    parser.add_argument("--case", action="append", help="one case ID, repeatable")
    parser.add_argument("--group", choices=("legacy", "miss", "collision", "tracking", "normal"))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--strict", action="store_true", help="also exit 1 for known issues")
    parser.add_argument("--output", type=Path, help="write JSON measurements")
    args = parser.parse_args()
    manifest = read(DATA / "manifest.json")
    entries = manifest["cases"]
    wanted = set(args.case or [])
    if wanted - {e["id"] for e in entries}:
        parser.error(f"Unknown case IDs: {sorted(wanted - {e['id'] for e in entries})}")
    entries = [e for e in entries if (not wanted or e["id"] in wanted) and (not args.group or e["group"] == args.group)]
    if not entries:
        parser.error("No matching cases")
    if args.list:
        for e in entries:
            print(e["id"], e["group"], e["title"])
        return 0
    roots = [args.project] if args.project else [Path.cwd(), *DATA.parents]
    root = next((p.resolve() for p in roots if (p / "examples/check_bundled_ik.py").is_file()), None)
    if root is None:
        parser.error("Provide the current RoboControl-IK checkout with --project")
    sys.path.insert(0, str(root / "examples"))
    from check_bundled_ik import Firefly, mujoco

    observed_hashes = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in manifest["model_files_sha256"]}
    report = dict(format=manifest["format"], version=manifest["version"], simulation_only=True, mujoco=mujoco.__version__,
                  model_files_sha256=observed_hashes, changed_model_files=[f for f, digest in observed_hashes.items()
                  if digest != manifest["model_files_sha256"][f]], cases=[])
    tol = manifest["tolerances"]
    with closing(Firefly()) as sim, TemporaryDirectory() as tmp:
        for e in entries:
            case = read(DATA / e["path"])
            inp = case["input"]
            try:
                if case["kind"] != "sequence":
                    sides = (inp["side"],) if case["kind"] == "single_pose" else SIDES
                    home = sim.model.key("home").ctrl[np.concatenate([sim.arm_drives[s] for s in sides])]
                    if not np.allclose(home, inp["start_q_rad"], atol=1e-12, rtol=0):
                        raise ValueError("Model home differs from the frozen start")
                if case["kind"] in ("point_to_point", "contact_at_reference"):
                    status, measurements = point_case(sim, case, root, Path(tmp), tol)
                elif case["kind"] == "single_pose":
                    status, measurements = pose_case(sim, case, Path(tmp), tol)
                elif case["kind"] == "sequence":
                    status, measurements = sequence_case(sim, case, tol)
                elif case["kind"] == "cartesian_path":
                    status, measurements = tracking_case(sim, case, tol)
                else:
                    raise ValueError(f"Unknown case kind {case['kind']}")
            except Exception as exc:
                status, measurements = "error", dict(error=f"{type(exc).__name__}: {exc}")
                if isinstance(exc, subprocess.CalledProcessError):
                    measurements["stderr"] = exc.stderr
            print(case["id"], status, case["title"], flush=True)
            report["cases"].append(dict(id=case["id"], group=case["group"], status=status, measurements=measurements))
    counts = report["counts"] = dict(Counter(r["status"] for r in report["cases"]))
    print(counts, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output.parent) as tmp:
            staged = Path(tmp) / "results.json"
            staged.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=1) + "\n")
            staged.replace(args.output)
    return int(bool(counts.get("error") or counts.get("regression") or (args.strict and counts.get("known_issue"))))


if __name__ == "__main__":
    raise SystemExit(main())
