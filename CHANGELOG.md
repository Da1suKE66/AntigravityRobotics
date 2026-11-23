# Changelog & Baseline Benchmark Record

## [0.2.0] - 2026-11-23

### Collision Remediation, Global Multi-Seed IK & Dual-Arm Pick-and-Place
- **`src/build_mjcf.py` & `assets/meshes/collision/`**:
  - Decomposed `link_3`, `link_4`, `link_5`, `gr100_link_1`, and `gr100_link_2` into `<= 60`-vertex multi-convex hulls, eliminating `C01–C03` false `link_3`–`link_5` overlaps.
  - Added `group="3"` collision primitives for the workcell housing, vertical post (`post_profile`), overhead D435 mount, and wrist D405 camera brackets (enabling real collision detection on `C04` and `C05`).
  - Calibrated gripper motor limits (`5.0 Nm`) and high-friction finger pad contacts (`condim="4"`, `friction="1.2 0.05 0.005"`).
- **`src/ik_solver.py` & `src/planner.py`**:
  - Added `mink.DofFreezingTask` to decouple gripper DOFs from arm IK, configurable `site_suffix` (`ee_site` / `tcp_site`), and `solve_pose_global()` multi-branch warm-start (solving `H01` wrist-pole jump to `0.14 mm`).
  - Added `FireflyPlanner` (`plan_joint_p2p`, `plan_pose_p2p`) with collision checking across `synchronous`, `left_first`, `right_first`, and `home_via` modes plus JSON/CSV trajectory export (`docs/paths/`).
- **`src/grasp_scene.py`, `src/grasp_pipeline.py` & `tests/test_grasp_pick_place.py`**:
  - Built `model/grasp_scene.xml` with `red_block` (`44 mm`, `85 g`), `blue_cylinder` (`r=21 mm, h=60 mm`, `95 g`), and target placement pads.
  - Implemented 10-stage closed-loop pick-and-place achieving `0.26 mm` (left `red_block`) and `0.87 mm` (right `blue_cylinder`) post-settle placement accuracy (`tilt <= 0.01 deg`).
- **Verification & Reports (`docs/reports/SIMULATION_IK_GRASP_REPORT.md`)**:
  - `tests/verify_sim_ik.py`: 3/3 PASS.
  - `tests/eval_regression_25.py`: **22/25 PASS** (`H01–H05`, `M01–M03`, `C01–C05`, `E02–E05`, `P01–P05`). Remaining 3 (`M04`, `M05`, `E01`) stem from a documented `~43 mm / 0.059 rad` frozen-dataset right-arm calibration offset and `E01`'s known upward-circle elbow singularity.
  - `tests/test_grasp_pick_place.py`: PASS (`docs/media/firefly_grasp_demo.mp4` & `docs/media/firefly_grasp_keyframes.png`).

## [0.1.0] - 2026-11-21

### Architecture & Implementation (Built from Scratch from `firefly_dual.urdf`)
- **`src/build_mjcf.py`**: Programmatic `mujoco.MjSpec` URDF-to-MJCF builder generating `model/firefly_dual.xml` and `model/scene.xml` (`implicitfast`, 500 Hz, separated visual `group=2` / collision `group=3`, adjacent-link `<exclude>`, torque-limited `<motor>` actuators, `left_ee_site`/`right_ee_site` at 125 mm, `left_tcp_site`/`right_tcp_site` at 170 mm, calibrated optical cameras).
- **`src/ik_solver.py`**: `FireflyDualIK` using `mink` + `daqp` on the exact `MjModel` (`ConfigurationLimit`, URDF `VelocityLimit`, `PostureTask` regularization, `reset_to_measured()` for bumpless sync, $C^2$ quintic $SE(3)$ blending).
- **`src/controller.py`**: `BumplessGravityController` (`PASSIVE -> ENGAGING -> ACTIVE` bumpless handoff state machine + model-based gravity feedforward $\tau_g(q_{\text{meas}})$ + back-calculation anti-windup PID).
- **`src/render_demo.py`**: 720p/30fps MuJoCo offscreen simulation renderer (`docs/media/firefly_sim_demo.mp4` & `docs/media/firefly_sim_keyframes.png`).

### Verification Results (`tests/verify_sim_ik.py`)
- **Bumpless Transfer**: `torque_jump = 0.0000 Nm`, `max_excursion = 0.000 deg`, `max_qacc = 0.00 rad/s^2`.
- **Gravity + Unmodeled 0.5 kg Payload Steady-State Error**:
  - Naive PD: `q_err = 4.41 deg`, `ee_err = 82.26 mm`
  - G + PID + Anti-Windup: `q_err = 0.000 deg`, `ee_err = 0.00 mm`
- **Dual-Arm Mink IK Tracking**: `left_err = 0.50 mm`, `right_err = 0.13 mm`, `max_contacts = 0`.

### `ik_regression_25` Evaluation (`tests/eval_regression_25.py`)
- **H02** (12-frame coupled singularity sequence): `max_pos = 0.000 mm`, `max_ori = 0.0000 rad`, `max_step = 0.515 deg`.
- **M01–M03** (singularity poses): `0.048 mm`, `0.109 mm`, `0.184 mm`.
- **E02–E05** (dynamic tracking from `target[0]` without false `link_3`-`link_5` convex hull overlap): `0.752 mm`, `0.593 mm`, `0.858 mm`, `0.560 mm` (`ncon = 0`, all `< 1.0 mm`).
- **Known Open Gaps for Next Iteration**:
  1. Decompose non-convex STL collision meshes (especially `link_3`) into multi-convex hulls (CoACD) or analytic primitives to eliminate false `link_3`–`link_5` convex-hull contact (`C01–C03`).
  2. Add global multi-seed / analytic IK warm-start ahead of `mink` for wrist-pole / dual-singularity branch jumps (`H01`, `M04`, `M05`).
  3. Run real-hardware system identification to refine `armature`, `damping`, and `frictionloss`.
