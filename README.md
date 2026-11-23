# Firefly Dual-Arm MuJoCo Simulation, Constrained Mink IK & Pick-and-Place

Clean MuJoCo 3.x (`MjSpec`) simulation, multi-hull collision modeling, constrained + multi-seed global IK (`mink` + `daqp`), **Bumpless Transfer + Gravity-Compensated (G + PID + Anti-Windup)** control, and closed-loop **tabletop pick-and-place** for the `firefly_dual` dual-arm robot.

![Grasp Keyframes](docs/media/firefly_grasp_keyframes.png)

## Project Structure

- `assets/firefly_dual.urdf`, `assets/meshes/` & `assets/meshes/collision/` — Source URDF, visual meshes, and decomposed `<=60`-vertex convex collision hulls.
- `model/firefly_dual.xml`, `model/scene.xml` & `model/grasp_scene.xml` — Normalized MuJoCo MJCF asset, workcell scene, and tabletop grasping scene.
- `src/build_mjcf.py` & `src/grasp_scene.py` — Programmatic `mujoco.MjSpec` builders (visual group 2 / collision group 3 separation, decomposed arm/finger hulls, workcell/camera colliders, `implicitfast` 500 Hz, torque `<motor>` actuators, `125 mm` EE / `170 mm` TCP sites, calibrated cameras, and graspable props).
- `src/ik_solver.py` & `src/planner.py` — `FireflyDualIK` (gripper-decoupled Mink + DAQP differential IK + `solve_pose_global` multi-branch warm-start) and `FireflyPlanner` (collision-checked P2P/sequential planning with JSON/CSV trajectory export).
- `src/controller.py` — `BumplessGravityController` (`PASSIVE -> ENGAGING -> ACTIVE` bumpless handoff, dynamics-twin gravity feedforward $\tau_g(q_{\text{meas}})$, and anti-windup PID).
- `src/grasp_pipeline.py` — Closed-loop 10-stage dual-arm tabletop pick-and-place executor, path logger, and MP4 renderer.
- `src/sim_runner.py` & `src/render_demo.py` — Closed-loop trajectory demo and 720p MP4 renderer (`docs/media/firefly_sim_demo.mp4`).
- `tests/verify_sim_ik.py`, `tests/audit_sim_ik.py`, `tests/probe_grasp_execution.py`, `tests/eval_regression_25.py` & `tests/test_grasp_pick_place.py` — Core verification, 9-item audit suite (`9/9`), finger-speed & contact-abort probe, strict 25-case IK regression (`21/25`), and pick-and-place verification.
- `docs/paths/`, `docs/reports/` & `docs/media/` — Saved planned/executed trajectories, JSON/Markdown evaluation reports (`docs/reports/SIMULATION_IK_GRASP_REPORT.md`), and demo videos.
- `CHANGELOG.md` — Project changelog, verified metrics, and remaining notes.

## Quick Start

```bash
# 1. Rebuild MJCF + grasping scene from URDF
/home/lakeside/.venvs/mujoco/bin/python src/build_mjcf.py
/home/lakeside/.venvs/mujoco/bin/python src/grasp_scene.py

# 2. Run core, audit & grasp-probe verification suites
/home/lakeside/.venvs/mujoco/bin/python tests/verify_sim_ik.py
/home/lakeside/.venvs/mujoco/bin/python tests/audit_sim_ik.py
/home/lakeside/.venvs/mujoco/bin/python tests/probe_grasp_execution.py

# 3. Run strict 25-case IK regression benchmark (writes docs/reports/ik_regression_25_results.json)
/home/lakeside/.venvs/mujoco/bin/python tests/eval_regression_25.py

# 4. Run dual-arm pick-and-place verification & render video
/home/lakeside/.venvs/mujoco/bin/python tests/test_grasp_pick_place.py
/home/lakeside/.venvs/mujoco/bin/python src/grasp_pipeline.py
```
