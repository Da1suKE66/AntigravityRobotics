# Firefly Dual-Arm Simulation: IK & Pick-and-Place Engineering Report

![Dual-Arm Pick-and-Place Keyframes](../media/firefly_grasp_keyframes.png)

## 1. Summary of Fixes & Upgrades

1. **Gripper Substring Classification & URDF Mimic Fidelity (`src/build_mjcf.py`, `src/ik_solver.py`, `src/controller.py`)**:
   - Fixed the `"_joint_"` substring check that previously matched `"_grip_joint_"` before gripper branches.
   - Removed the independent actuator on the passive mimic follower (`left_grip_joint_2` / `right_grip_joint_2`, `effort = 0 Nm` per URDF) and tightened `<equality>` `solref="0.004 1.0"` so finger 2 tracks finger 1 passively without independent drive torque.
   - Applied `GRIPPER_DYNAMICS` (`effort=5.0`, `armature=0.002`, `damping=0.08`, `frictionloss=0.03`) to gripper joints and activated `mink.DofFreezingTask` + zeroed gripper velocity outputs during arm IK.

2. **Multi-Hull Collision Decomposition & Workcell Colliders (`src/build_mjcf.py`, `assets/meshes/collision/`)**:
   - Decomposed concave arm and gripper links (`link_3` -> 4 hulls, `link_4` -> 2 hulls, `link_5` -> 3 hulls, `gr100_link_1`/`gr100_link_2` -> 3 hulls each, `<= 60` vertices per hull), eliminating false `link_3`–`link_5` overlaps (`C01–C03`).
   - Added `group="3"` box colliders for `post_adapter_plate`, `post_socket`, `post_profile`, `top_post_u_piece`, `top_camera_mount_plate`, `top_d435_visual_mesh_frame`, `e3_housing_cad_frame`, and wrist camera brackets (`A_fixed`, `B_camera_plate`, `d405`).

3. **Independent Per-Arm Multi-Seed Global IK & Limit-Safe Planner (`src/ik_solver.py`, `src/planner.py`)**:
   - Upgraded `FireflyDualIK.solve_pose_global(...)` to solve independent 6-DOF per-arm multi-seed branches (`40/40` on wide-workspace reachable dual-arm poses) and select the closest collision-free combination.
   - Upgraded `FireflyPlanner.plan_joint_p2p` to validate joint position limits (raising `ValueError` on out-of-range inputs), automatically stretch segment duration $T_{\min} = \max_i 1.875 |\Delta q_i| / v_{\max,i}$ to obey URDF velocity limits, check every sample (`stride=1`) against the full simulation scene, and respect stage-specific `allowed_body_pairs`.

4. **Grasp Executor Safety Gates, Active Safe-Hold Braking & Finger Speed Compliance (`src/grasp_pipeline.py`, `src/controller.py`, `tests/verify_sim_ik.py`)**:
   - Connected `GraspExecutor` to the full `grasp_scene.xml` (`self.sim_model`) with live `scene_qpos_tail`, stage-specific allowed object-grasp pairs, pre-execution IK convergence & collision rejection gates (`RuntimeError` before motion), and **immediate runtime abort on forbidden contact** (0 extra steps continued after contact; transitions immediately into `ControlMode.SAFE_HOLD`).
   - Added active damped braking (`ControlMode.SAFE_HOLD`, `enter_safe_hold`, `compute_brake_ctrl`, `step_safe_hold`, `hold_safe`) using gravity+Coriolis RNE bias plus high-damping PD clamping at the contact configuration, reducing 0.6 s post-abort TCP coasting from `16.89 mm` to **`1.963 mm`**.
   - Updated `_set_gripper` to latch actual measured finger positions (`data.qpos`) and enforce quintic ramp duration $T \ge 1.875 |\Delta q| / 1.38$ (keeping both commanded and measured finger speeds strictly below the `1.5 rad/s` URDF limit).
   - Fixed `solve_pose_global` fallback scoring (`best_score` initialization), raised library-default `FireflyDualIK(damping=0.025)` for singularity-safe default behavior, and added a dynamic `limit_margin_rad=6e-4` in `FireflyPlanner.plan_pose_p2p` (eliminating `H01`'s `0.000025 rad` physical stop excursion -> `0.000000 rad`).
   - Fixed back-calculation anti-windup units in `BumplessGravityController` (`(kb / ki) * sat_diff`, `kb = 5.0 1/s`) and added viscous damping feedforward (`dof_damping * dq_target`), preserving `0.05 mm` `+0.5 kg` payload steady-state accuracy and cutting rated `9/3 N·m` `E04` tracking error from `3.980 mm` to **`0.936 mm`** (`< 1.0 mm`, matching `22/25` pass count under rated torque limits).

---

## 2. Quantitative Verification Results

### 2.1 Directed Simulation & IK Audit (`tests/audit_sim_ik.py`) — **9/9 PASS**

| Check | Result |
| --- | --- |
| `test_near_home_reachable_dual_arm_poses` (12/12) | **PASS** |
| `test_wide_workspace_reachable_dual_arm_poses` (40/40) | **PASS** |
| `test_arm_ik_leaves_measured_grippers_stationary` | **PASS** |
| `test_mimic_finger_has_no_independent_drive_torque` (`<= 5.0 / 0.0 Nm`) | **PASS** |
| `test_position_hold_with_physical_half_kg_payload` (`+0.500 kg`, `< 1.5 mm` @ 3 s) | **PASS** |
| `test_joint_plan_respects_position_limit` (rejects `3.0 rad > 2.862 rad`) | **PASS** |
| `test_joint_plan_respects_velocity_limit` (`<= 5.585 rad/s`) | **PASS** |
| `test_grasp_stage_declared_clear_has_no_table_contact` (`z=0.04 m` rejected, `0` contact steps) | **PASS** |
| `test_grasp_executor_rejects_unreachable_goal_before_moving` (`<= 1e-5 rad` displacement) | **PASS** |

### 2.2 Core Control & IK Suite (`tests/verify_sim_ik.py`) — **3/3 PASS**

- **Bumpless Transfer**: `torque_jump = 0.0000 Nm`, `max_excursion = 0.010 deg`, `max_qacc = 0.00 rad/s^2`
- **Gravity + Physical 0.5 kg Payload (3.0 s)**: Naive PD `71.10 mm` -> G+PID+AW **`0.05 mm`** (`0.006 deg`)
- **Dual-Arm Mink IK**: `left_err = 0.49 mm`, `right_err = 0.12 mm`, `max_contacts = 0`

### 2.3 Strict 25-Case IK Regression (`tests/eval_regression_25.py` -> `ik_regression_25_results.json`) — **22/25 PASS** (Raw Frozen Targets) / **25/25 PASS** (`--align-right-frame`)

Evaluated under original dataset thresholds (`M`: `<= 0.01 mm / 0.0001 rad`; `H02`: `< 1°`; `E`: `< 1.0 mm` from `target[0]`; `H/P`: `4.0 s` hold `< 0.4 mm / 0.004 rad / 0.01 rad/s`, `lim_exc <= 1e-5 rad`; `C`: `marked_match = True`):

- **H01–H05**: **5/5 PASS** (`H01` `tail_pos = 0.129 mm`, `tail_ori = 0.00060 rad`, `lim_exc = 0.000000 rad`; `H02` `max_step = 0.516 deg`; `H03–H05` `tail_pos = 0.028–0.045 mm`, `lim_exc = 0.000000`, `ncon = 0`).
- **M01–M03**: **3/3 PASS** (`0.0011 mm`, `0.0004 mm`, `0.0029 mm`, `ncon = 0`).
- **M04–M05**: Known frozen-dataset right-arm URDF mismatch (`assets/firefly_dual.urdf` SHA-256 `a534fa06...` nominal symmetric vs frozen dataset `eafe1dab...` calibrated right arm offset `43.022 mm / 0.05929 rad`): `14.3611 mm` / `7.3624 mm` on raw targets; **`<= 0.0054 mm`** with `--align-right-frame`.
- **C01–C05**: **5/5 PASS** (`C01–C03` `ncon = 0`; `C04–C05` `marked_match = True` — all marked collision pairs detected along with additional workcell/camera-bracket contacts).
- **E02–E05**: **4/4 PASS** (`0.967 mm`, `0.544 mm`, `0.930 mm`, `0.501 mm`, all `< 1.0 mm`, `ncon = 0`). Under dataset-rated **`9/3 N·m`** torque limits, `E02–E05` remain **4/4 PASS** (`E04 = 0.936 mm < 1.0 mm`) and overall strict score remains **`22/25`**.
- **E01**: Hits the right-arm straight-elbow singularity due to the right-arm frame mismatch (`5.170 mm`, `ncon = 0` on both `20/5` and `9/3 N·m`); drops to **`0.567 mm`** (`< 1.0 mm`, PASS) with `--align-right-frame --strict` (`25/25 PASS`).
- **P01–P05**: **5/5 PASS** (`tail_pos = 0.021–0.028 mm`, `lim_exc = 0.000000`, `ncon = 0`).

### 2.4 Dual-Arm Pick-and-Place & Probe Verification (`tests/test_grasp_pick_place.py` & `tests/probe_grasp_execution.py`) — **PASS**

Measured after release and `0.6 s` settle (`forbidden_contacts = 0`, `peak_torque = 11.89 Nm`, all finger speeds `<= 1.5 rad/s`, unexpected contact abort at step `100 / 100`, `0.6 s` `SAFE_HOLD` post-abort drift `1.963 mm`):

| Arm | Object | Lifted Z Gain | Settled XY Error | Settled Z Error | Settled Tilt | Settled Speed | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `left` | `red_block` | `85.2 mm` | `0.19 mm` | `-0.03 mm` | `0.00 deg` | `0.0000 m/s` | **PASS** |
| `right` | `blue_cylinder` | `83.1 mm` | `0.15 mm` | `-0.03 mm` | `0.01 deg` | `0.0001 m/s` | **PASS** |
