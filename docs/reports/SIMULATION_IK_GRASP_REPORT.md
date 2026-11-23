# Firefly Dual-Arm Simulation: IK & Pick-and-Place Engineering Report

![Dual-Arm Pick-and-Place Keyframes](../media/firefly_grasp_keyframes.png)

## 1. Summary of Upgrades

1. **Multi-Convex-Hull Collision Decomposition & Workcell Colliders (`src/build_mjcf.py`)**:
   - Decomposed concave arm and gripper links (`link_3` -> 4 hulls, `link_4` -> 2 hulls, `link_5` -> 3 hulls, `gr100_link_1`/`gr100_link_2` -> 3 hulls each, `<= 60` vertices per hull) into `assets/meshes/collision/`.
   - Added `group="3"` box colliders for `post_adapter_plate`, `post_socket`, `post_profile`, `top_post_u_piece`, `top_camera_mount_plate`, `top_d435_visual_mesh_frame`, `e3_housing_cad_frame`, and wrist camera brackets (`A_fixed`, `B_camera_plate`, `d405`), with adjacent `<exclude>` pairs.
   - Normalized gripper actuator limits (`effort = 5.0 Nm`, `armature = 0.002`, `damping = 0.08`, `frictionloss = 0.03`) and high-friction finger pad contacts (`condim="4"`, `friction="1.2 0.05 0.005"`).

2. **Gripper-Decoupled & Multi-Seed Global IK + Collision-Aware Planner (`src/ik_solver.py`, `src/planner.py`)**:
   - Added `mink.DofFreezingTask` on the 4 gripper DOFs so arm IK never interferes with gripper open/close commands.
   - Implemented `FireflyDualIK.solve_pose_global(...)` with deterministic multi-branch seed search (base azimuth, elbow, and wrist-pole `J5 = ±π/2` flips) + MuJoCo contact filtering.
   - Implemented `FireflyPlanner` (`plan_joint_p2p`, `plan_pose_p2p`) with collision verification across `synchronous`, `left_first`, `right_first`, and `home_via` modes, plus JSON/CSV trajectory export.

3. **Tabletop Grasping Scene & Closed-Loop Pick-and-Place (`src/grasp_scene.py`, `src/grasp_pipeline.py`)**:
   - Built `model/grasp_scene.xml` via `MjSpec` containing `red_block` (`44×44×44 mm`, `85 g`), `blue_cylinder` (`r=21 mm, h=60 mm`, `95 g`), and placement pads (`place_zone_left`, `place_zone_right`).
   - Executed 10-stage closed-loop pick-and-place (`open -> pre_grasp -> descend -> close -> lift -> transport -> lower -> release -> retract -> return_home -> settle`) under `BumplessGravityController`.

---

## 2. Quantitative Verification Results

### 2.1 Core Control & IK Suite (`tests/verify_sim_ik.py`)

| Test | Metric | Result |
| --- | --- | --- |
| Bumpless Transfer | `torque_jump` / `max_excursion` / `max_qacc` | `0.0000 Nm` / `0.000 deg` / `0.00 rad/s^2` (PASS) |
| Gravity + 0.5 kg Payload SSE | Naive PD vs G+PID+AW | `82.26 mm` -> `0.00 mm` (PASS) |
| Dual-Arm Mink IK Tracking | `left_err` / `right_err` / `max_contacts` | `0.50 mm` / `0.13 mm` / `0` (PASS) |

### 2.2 25-Case IK Regression Benchmark (`tests/eval_regression_25.py` -> `ik_regression_25_results.json`)

**22 / 25 cases passed** (up from the pre-remediation baseline where `C01–C05`, `H01`, `H03–H04`, `E01–E05`, `P01–P03` had false contacts or tracking failures):

- **H01–H05 (Legacy Hard Cases — 5/5 PASS)**:
  - `H01` (wrist pole `+90°` jump): `left=0.14 mm, right=0.08 mm, ncon=0` (previously `227.77 mm, ncon=4`).
  - `H02` (12-frame coupled singularity): `max_pos=0.000 mm, max_ori=0.0000 rad, max_step=0.516 deg`.
  - `H03–H05`: `0.19–0.68 mm` left, `0.08–0.27 mm` right, `ncon=0`.
- **M01–M05 (Singularity Poses — 3/5 PASS)**:
  - `M01–M03`: `0.0011 mm`, `0.0000 mm`, `0.0000 mm`, `ncon=0`.
  - `M04–M05` (known dataset/URDF calibration gap): In the frozen `ik_regression_25` dataset, `right` arm poses have a `~43 mm / 0.0593 rad` FK offset relative to `assets/firefly_dual.urdf` (verified directly on `reference.q_rad` and `recorded.numerical.q_rad`), placing `M04`/`M05` right on the URDF arm's outer planar-reach singularity (`pos_err = 14.36 mm / 12.07 mm`).
- **C01–C05 (Collision Fidelity — 5/5 PASS)**:
  - `C01–C03`: `ncon=0` (false `link_3`–`link_5` convex-hull contacts eliminated).
  - `C04`: `ncon=32` (now accurately detects `post_profile` vs `right_link_6` / brackets).
  - `C05`: `ncon=15` (accurately detects `left_grip_link_1` vs `right_link_6` / brackets).
- **E01–E05 (Cartesian Tracking — 4/5 PASS)**:
  - `E02–E05` (from `target[0]`): moving peak `1.235 mm`, `0.961 mm`, `1.327 mm`, `0.862 mm`, all `ncon=0`.
  - `E01` (known unreachable upward-circle singularity in original dataset): passes through the right-arm straight-elbow singularity around `t=1.6–2.0 s` (`J3 -> 0`, `max_dq = 5.585 rad/s`), peaking at `371.2 mm` (`ncon=0`).
- **P01–P05 (Point-to-Point — 5/5 PASS)**:
  - `0.00–0.51 mm` left, `0.08–0.60 mm` right, all `ncon=0`.

### 2.3 Dual-Arm Pick-and-Place Verification (`tests/test_grasp_pick_place.py` -> `grasp_verification_report.json`)

Measured **after release and 0.6 s settle**:

| Arm | Object | Lifted Z Gain | Settled XY Error | Settled Z Error | Settled Tilt | Settled Speed | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `left` | `red_block` | `84.3 mm` | `0.26 mm` | `-0.03 mm` | `0.00 deg` | `0.0000 m/s` | **PASS** |
| `right` | `blue_cylinder` | `82.8 mm` | `0.87 mm` | `-0.03 mm` | `0.01 deg` | `0.0002 m/s` | **PASS** |

- All IK stages across both pick-and-place runs achieved `ik_pos_err <= 0.0021 mm`, `exec_pos_err <= 2.22 mm`, and `collision_free = true`.

---

## 3. Repository Artifacts (Paths, Reports, Media)

- **Planned & Executed Paths (`docs/paths/`)**:
  - `docs/paths/grasp_left_red_block.json` & `.csv`
  - `docs/paths/grasp_right_blue_cylinder.json` & `.csv`
  - `docs/paths/grasp_executed_trace.csv`
  - `docs/paths/ik_H01.json` & `.csv`
  - `docs/paths/ik_H05.json` & `.csv`
- **Machine-Readable Reports (`docs/reports/`)**:
  - `docs/reports/ik_regression_25_results.json`
  - `docs/reports/grasp_verification_report.json`
- **Rendered Media (`docs/media/`)**:
  - `docs/media/firefly_grasp_demo.mp4`
  - `docs/media/firefly_grasp_keyframes.png`
