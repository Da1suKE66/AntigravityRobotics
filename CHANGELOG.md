# Changelog & Baseline Benchmark Record

## [0.2.3] - 2026-11-23

### Active Safe-Hold Braking, Anti-Windup Unit Fix, Library Singularity Damping & H01 Soft Margin
- **`src/controller.py` & `src/grasp_pipeline.py`**:
  - Added `ControlMode.SAFE_HOLD`, `enter_safe_hold(q_meas)`, `compute_brake_ctrl(q_meas, dq_meas)`, `compute_bias_torque(q_meas, dq_meas)`, and `GraspExecutor.step_safe_hold()` / `hold_safe()`: unexpected contact abort now engages active damped braking at the contact pose, reducing 0.6 s post-abort TCP drift from `16.89 mm` to **`1.963 mm`**.
  - Fixed back-calculation anti-windup units from `kb * sat_diff` to `(kb / ki) * sat_diff` (`anti_windup_kb = 5.0 1/s`) and added viscous damping feedforward (`model.dof_damping * dq_target`). Under dataset-rated **`9/3 N·m`** torque limits, `E04` tracking error dropped from `3.980 mm` to **`0.936 mm`** (`< 1.0 mm`) and `E01` dropped from `34.125 mm` to **`5.170 mm`**, achieving the same **`22/25`** strict pass count as `20/5 N·m`.
- **`src/ik_solver.py` & `src/planner.py`**:
  - Raised library default `FireflyDualIK(damping=0.025)` for singularity-safe default behavior out of the box.
  - Added dynamic `limit_margin_rad = 6e-4` in `FireflyPlanner.plan_pose_p2p` (clamping converged `q_goal` inside `[lo + 6e-4, hi - 6e-4]`), resolving `H01` (`lim_exc = 0.000000 rad`, `tail_pos = 0.129 mm`, PASS).
- **`tests/eval_regression_25.py`**:
  - Added opt-in `--align-right-frame` diagnostic flag (default `False` to preserve raw frozen-dataset comparisons) isolating the `a534fa06...` vs `eafe1dab...` URDF right-arm calibration offset (`43.022 mm / 0.05929 rad`).
- **Verification**:
  - `tests/eval_regression_25.py` (raw frozen targets): **`22/25` PASS** (`H01–H05` 5/5, `M01–M03` 3/3, `C01–C05` 5/5, `E02–E05` 4/4, `P01–P05` 5/5; identical **`22/25`** under rated `9/3 N·m`).
  - `tests/eval_regression_25.py --align-right-frame --strict`: **`25/25` PASS** (`E01 = 0.567 mm`, `M01–M05 <= 0.0054 mm`).
  - `tests/audit_sim_ik.py`: **9/9 PASS** (`12.365 s`).
  - `tests/probe_grasp_execution.py`: **PASS** (exit 0; finger speeds `<= 1.5 rad/s`, abort at step `100 / 100`).
  - `tests/verify_sim_ik.py`: **3/3 PASS** (`0.05 mm` under `+0.5 kg` payload).
  - `tests/test_grasp_pick_place.py`: **PASS** (`0.19 / 0.15 mm`, `peak_torque = 11.89 Nm`, `forbidden_contacts = 0`).

## [0.2.2] - 2026-11-23

### Runtime Contact Abort, Gripper Speed Enforcement, Singularity Damping & Diagnostics
- **`src/grasp_pipeline.py`**:
  - Enforced URDF `1.5 rad/s` gripper velocity limits in `_set_gripper` by latching measured `data.qpos` (`g0`) and scaling quintic ramp duration ($T \ge 1.875 |\Delta q| / 1.38$, min `0.65 s`), reducing release/close speeds below `1.5 rad/s` (`tests/probe_grasp_execution.py` PASS) and improving pick-place XY accuracy to `0.19 / 0.17 mm`.
  - Added immediate runtime abort (`abort_on_forbidden_contact=True`) in `_step_control`: on any forbidden contact, commands safe gravity-hold torque and raises `RuntimeError` on the exact contact tick (`0` extra steps after contact).
- **`src/ik_solver.py` & `src/build_mjcf.py`**:
  - Fixed `solve_pose_global` fallback scoring initialization (`best_score = (2, 10**9, 1e12, 1e12)`) so failed/colliding candidates report true FK residuals instead of `1e9`.
  - Set `solref_limit = [0.004, 1.0]` (`2 * dt`) on joints, reducing `H01` limit excursion 13× (`0.000324 -> 0.000025 rad`).
- **`tests/eval_regression_25.py`**:
  - Enabled Tikhonov velocity damping (`damping = 0.045`) in `_make_tcp_ik`, eliminating 500 Hz elbow-singularity chattering on `E01` (`451.493 mm -> 5.419 mm`, `ncon = 0`) while keeping `E02–E05` all `< 1.0 mm` (`0.843 / 0.553 / 0.998 / 0.490 mm`).
- **Verification**:
  - `tests/audit_sim_ik.py`: **9/9 PASS** (`12.006 s`).
  - `tests/probe_grasp_execution.py`: **PASS** (exit 0; finger speeds `<= 1.5 rad/s`, abort at step `100 / 100`).
  - `tests/verify_sim_ik.py`: **3/3 PASS**.
  - `tests/test_grasp_pick_place.py`: **PASS** (`0.19 / 0.17 mm`, `forbidden_contacts = 0`).
  - `tests/eval_regression_25.py`: **21/25 PASS** (`E01` down to `5.419 mm`, `H01` `lim_exc = 0.000025 rad`).

## [0.2.1] - 2026-11-23

### Audit Remediation: Grasp Safety Gates, Per-Arm Multi-Seed IK, Gripper Mimic & Payload PID
- **`src/build_mjcf.py`, `src/ik_solver.py`, `src/controller.py`**:
  - Fixed `"_grip_joint_"` vs `"_joint_"` substring bug across MJCF builder, IK DOF classification, and controller gains; removed independent motor actuator on mimic follower `*_grip_joint_2` (`effort = 0 Nm` per URDF, `nu = 14`).
  - Enabled `mink.DofFreezingTask` + zeroed gripper velocity outputs during arm IK (`<= 1e-6 rad/s` on measured gripper changes).
  - Tuned G+PID+AW gains (`kp=240/180/65`, `ki=320/260/95`, `kd=18/14/4.0`, `i_clamp=0.12`) and fixed `tests/verify_sim_ik.py` to attach the `0.5 kg` payload as a child body (`0.05 mm` SSE at 3.0 s).
- **`src/ik_solver.py` & `src/planner.py`**:
  - Upgraded `solve_pose_global` to independent per-arm 6-DOF multi-branch search + collision-free combination (`40/40` on wide-workspace poses in `tests/audit_sim_ik.py`).
  - Added strict position limit validation, quintic duration stretching to respect URDF velocity limits, dense `stride=1` collision checking, and `scene_qpos_tail` / `allowed_body_pairs` support in `FireflyPlanner`.
- **`src/grasp_pipeline.py` & `tests/test_grasp_pick_place.py`**:
  - Connected `GraspExecutor` to full `grasp_scene.xml` for collision checking, added pre-execution IK convergence & collision rejection gates, and added runtime forbidden-contact monitoring (`forbidden_contacts = 0`, peak torque `11.92 Nm`, placement XY error `0.26 / 0.25 mm`).
- **Verification**:
  - `tests/audit_sim_ik.py`: **9/9 PASS** (`11.58 s`).
  - `tests/verify_sim_ik.py`: **3/3 PASS**.
  - `tests/test_grasp_pick_place.py`: **PASS**.
  - `tests/eval_regression_25.py` (strict dataset thresholds): **21/25 PASS** (`E02–E05 < 1.0 mm`, `H03–H05` & `P01–P05 < 0.05 mm` at 4.0 s hold, `C01–C05` 5/5 marked pair match).

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
