# Changelog & Baseline Benchmark Record

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
