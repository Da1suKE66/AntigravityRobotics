"""Constrained differential & global multi-seed IK solver for Firefly Dual-Arm using Mink + DAQP."""

from __future__ import annotations

import numpy as np
import mujoco
import mink


def quintic_blend(u: float) -> tuple[float, float]:
    """Return C^2 minimum-jerk position scale s(u) and derivative ds/du for u in [0, 1]."""
    u = float(np.clip(u, 0.0, 1.0))
    u2 = u * u
    u3 = u2 * u
    s = 10.0 * u3 - 15.0 * u3 * u + 6.0 * u3 * u2
    ds = 30.0 * u2 - 60.0 * u3 + 30.0 * u3 * u
    return s, ds


def build_arm_collision_pairs(model: mujoco.MjModel) -> list[tuple[list[str], list[str]]]:
    """Build non-adjacent geom group pairs for mink.CollisionAvoidanceLimit."""
    left_arm, right_arm, env_geoms = [], [], []
    for gid in range(model.ngeom):
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            continue
        name = model.geom(gid).name
        if not name:
            continue
        bname = model.body(model.geom_bodyid[gid]).name
        if bname.startswith("left_"):
            left_arm.append(name)
        elif bname.startswith("right_"):
            right_arm.append(name)
        else:
            env_geoms.append(name)
    pairs: list[tuple[list[str], list[str]]] = []
    if left_arm and right_arm:
        pairs.append((left_arm, right_arm))
    if left_arm and env_geoms:
        pairs.append((left_arm, env_geoms))
    if right_arm and env_geoms:
        pairs.append((right_arm, env_geoms))
    return pairs


class FireflyDualIK:
    """Dual-arm constrained differential IK (Mink + DAQP) using the exact MuJoCo MjModel.

    Using the identical MjModel for both IK and dynamics eliminates URDF-vs-IK
    frame/convention mismatches (Sim2Real kinematic gap).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        solver: str = "daqp",
        damping: float = 1e-3,
        pos_cost: float = 1.0,
        ori_cost: float = 0.4,
        posture_cost: float = 1e-2,
        lm_damping: float = 1.0,
        site_suffix: str = "ee_site",
        freeze_grippers: bool = True,
        collision_model: mujoco.MjModel | None = None,
        min_collision_dist: float = 0.005,
    ) -> None:
        self.model = model
        self.solver = solver
        self.damping = damping
        self.site_suffix = site_suffix
        self.freeze_grippers = freeze_grippers
        self.configuration = mink.Configuration(model)

        self.left_frame = f"left_{site_suffix}"
        self.right_frame = f"right_{site_suffix}"

        self.left_task = mink.FrameTask(
            frame_name=self.left_frame,
            frame_type="site",
            position_cost=pos_cost,
            orientation_cost=ori_cost,
            lm_damping=lm_damping,
        )
        self.right_task = mink.FrameTask(
            frame_name=self.right_frame,
            frame_type="site",
            position_cost=pos_cost,
            orientation_cost=ori_cost,
            lm_damping=lm_damping,
        )
        self.posture_task = mink.PostureTask(model, cost=posture_cost)

        # Build velocity limits from URDF specs (rad/s) and separate arm vs gripper DOFs.
        # Check _grip_joint_ BEFORE _joint_ because _grip_joint_ contains _joint_!
        vel_limits = {}
        self.left_arm_dofs: list[int] = []
        self.right_arm_dofs: list[int] = []
        self.grip_dofs: list[int] = []

        for jid in range(model.njnt):
            j = model.joint(jid)
            jname = j.name
            dof = int(j.dofadr[0])
            if "_grip_joint_" in jname:
                vel_limits[jname] = 1.5
                self.grip_dofs.append(dof)
            elif "_joint_" in jname:
                idx = int(jname.rsplit("_", 1)[1])
                vel_limits[jname] = 5.585053606381854 if idx <= 4 else 12.566370614359172
                if jname.startswith("left_"):
                    self.left_arm_dofs.append(dof)
                elif jname.startswith("right_"):
                    self.right_arm_dofs.append(dof)

        self.dof_freezing_task = (
            mink.DofFreezingTask(model, self.grip_dofs, gain=1.0)
            if (freeze_grippers and self.grip_dofs)
            else None
        )

        self.limits: list[mink.Limit] = [
            mink.ConfigurationLimit(model),
            mink.VelocityLimit(model, vel_limits),
        ]
        if collision_model is not None:
            pairs = build_arm_collision_pairs(collision_model)
            if pairs:
                self.limits.append(
                    mink.CollisionAvoidanceLimit(
                        collision_model,
                        geom_pairs=pairs,
                        minimum_distance_from_collisions=min_collision_dist,
                        collision_detection_distance=0.025,
                    )
                )

        # Initialize targets to current configuration
        self.reset_to_measured(model.key("home").qpos if model.nkey > 0 else np.zeros(model.nq))

    def reset_to_measured(self, q_meas: np.ndarray) -> tuple[mink.SE3, mink.SE3]:
        """Synchronize IK internal state to actual measured joint positions (for Bumpless Transfer)."""
        self.configuration.update(q_meas)
        self.posture_task.set_target(q_meas.copy())
        t_left = self.get_ee_pose("left")
        t_right = self.get_ee_pose("right")
        self.left_task.set_target(t_left)
        self.right_task.set_target(t_right)
        return t_left, t_right

    def get_ee_pose(self, side: str) -> mink.SE3:
        """Return current end-effector SE3 pose from Mink's forward kinematics."""
        return self.configuration.get_transform_frame_to_world(f"{side}_{self.site_suffix}", "site")

    def set_targets(
        self,
        left_target: mink.SE3 | None = None,
        right_target: mink.SE3 | None = None,
    ) -> None:
        """Update desired end-effector SE3 targets."""
        if left_target is not None:
            self.left_task.set_target(left_target)
        if right_target is not None:
            self.right_task.set_target(right_target)

    def step(
        self,
        dt: float,
        q_meas: np.ndarray | None = None,
        blend_meas: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve one QP differential IK step and return (q_ref, dq_ref).

        Args:
            dt: Control timestep (s).
            q_meas: Optional measured joint positions for closed-loop drift feedback.
            blend_meas: Weight in [0, 1] for blending internal IK state toward q_meas.
        """
        if q_meas is not None and blend_meas > 0.0:
            q_sync = (1.0 - blend_meas) * self.configuration.q + blend_meas * q_meas
            self.configuration.update(q_sync)

        # Keep posture target on gripper DOFs aligned with current configuration when frozen
        if self.freeze_grippers and self.grip_dofs:
            tgt = self.posture_task.target_q.copy()
            tgt[self.grip_dofs] = self.configuration.q[self.grip_dofs]
            self.posture_task.set_target(tgt)

        tasks: list[mink.Task] = [self.left_task, self.right_task, self.posture_task]
        if self.dof_freezing_task is not None:
            tasks.append(self.dof_freezing_task)

        vel = mink.solve_ik(
            self.configuration,
            tasks,
            dt,
            solver=self.solver,
            damping=self.damping,
            limits=self.limits,
        )
        if self.freeze_grippers and self.grip_dofs:
            vel[self.grip_dofs] = 0.0
        self.configuration.integrate_inplace(vel, dt)
        return self.configuration.q.copy(), vel.copy()

    @staticmethod
    def _pose_error(actual: mink.SE3, target: mink.SE3) -> tuple[float, float]:
        pos_m = float(np.linalg.norm(actual.translation() - target.translation()))
        ori_rad = float(np.linalg.norm((actual.rotation().inverse() @ target.rotation()).log()))
        return pos_m, ori_rad

    def _clamp_to_limits(self, q: np.ndarray) -> np.ndarray:
        q_out = q.copy()
        for jid in range(self.model.njnt):
            if self.model.jnt_limited[jid]:
                adr = int(self.model.jnt_qposadr[jid])
                lo, hi = self.model.jnt_range[jid]
                q_out[adr] = np.clip(q_out[adr], lo + 1e-4, hi - 1e-4)
        return q_out

    def _build_arm_seeds(self, q_base: np.ndarray, side: str) -> list[np.ndarray]:
        """Build deterministic + stratified 6-DOF seed bank for one arm."""
        dofs = self.left_arm_dofs if side == "left" else self.right_arm_dofs
        seeds: list[np.ndarray] = [q_base[dofs].copy()]

        if self.model.nkey > 0:
            seeds.append(self.model.key("home").qpos[dofs].copy())

        # Structured branch seeds: base azimuth, elbow, wrist-pole flips
        patterns = [
            (0.0, False, 0.35, 1.05, 0.74, 0.0, 0.0),
            (np.pi * 0.5, False, 0.15, 0.35, 0.30, np.pi * 0.5, 0.0),
            (-np.pi * 0.5, False, 0.15, 0.35, 0.30, -np.pi * 0.5, 0.0),
            (np.pi, False, 0.15, 0.25, 0.10, np.pi * 0.5, -1.5),
            (-np.pi, False, 0.15, 0.25, 0.10, -np.pi * 0.5, 1.5),
            (-2.15, True, 0.15, 0.25, 0.10, np.pi * 0.5, -1.8),
            (-2.45, True, 0.25, 0.30, 1.25, np.pi * 0.5, -0.2),
            (1.15, True, 0.35, 0.85, 0.80, -np.pi * 0.5, -2.1),
            (2.15, True, 0.15, 0.25, 0.10, -np.pi * 0.5, 1.5),
            (-2.0, True, -0.80, 0.30, 1.00, np.pi * 0.5, -1.4),
            (0.0, False, -0.35, 0.30, 0.90, np.pi * 0.5, 0.0),
            (0.0, False, -0.35, 0.30, 0.90, -np.pi * 0.5, 0.0),
            (0.0, True, -0.40, 1.80, -0.80, 0.0, 0.0),
            (0.0, True, 0.60, 1.60, -0.60, 0.0, 0.0),
        ]
        q_arm_0 = q_base[dofs]
        for j1_v, abs_j1, j2, j3, j4, j5, j6 in patterns:
            cand = np.array(
                [j1_v if abs_j1 else (q_arm_0[0] + j1_v), j2, j3, j4, j5, j6],
                dtype=float,
            )
            q_full = q_base.copy()
            q_full[dofs] = cand
            seeds.append(self._clamp_to_limits(q_full)[dofs])

        # Add deterministic uniform random arm restarts covering full joint range
        low = np.array([self.model.jnt_range[self.model.jnt_qposadr == d][0, 0] if False else 0.0 for d in dofs])
        jids = [jid for jid in range(self.model.njnt) if int(self.model.jnt_qposadr[jid]) in dofs]
        jids.sort(key=lambda jid: int(self.model.jnt_qposadr[jid]))
        lo = np.array([self.model.jnt_range[jid, 0] + 1e-3 for jid in jids])
        hi = np.array([self.model.jnt_range[jid, 1] - 1e-3 for jid in jids])
        rng = np.random.default_rng(17)
        for _ in range(18):
            seeds.append(rng.uniform(lo, hi))

        return seeds

    @staticmethod
    def count_robot_collisions(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        allowed_body_pairs: set[tuple[str, str]] | None = None,
    ) -> int:
        """Count forbidden contacts involving at least one robot body (ignoring non-robot resting contacts)."""
        count = 0
        for c in data.contact:
            b1 = model.body(model.geom_bodyid[c.geom1]).name
            b2 = model.body(model.geom_bodyid[c.geom2]).name
            if not (b1.startswith(("left_", "right_")) or b2.startswith(("left_", "right_"))):
                continue
            if allowed_body_pairs is not None:
                pair = tuple(sorted((b1, b2)))
                if pair in allowed_body_pairs:
                    continue
            count += 1
        return count

    def _solve_single_arm_branches(
        self,
        q_base: np.ndarray,
        side: str,
        target: mink.SE3,
        pos_tol_m: float,
        ori_tol_rad: float,
        max_iters: int,
        max_branches: int = 6,
        seeds_override: list[np.ndarray] | None = None,
    ) -> list[tuple[np.ndarray, float, float, float]]:
        """Solve independent 6-DOF IK for one side across seeds, returning distinct branches."""
        dofs = self.left_arm_dofs if side == "left" else self.right_arm_dofs
        other_dofs = self.right_arm_dofs if side == "left" else self.left_arm_dofs
        other_freeze = mink.DofFreezingTask(self.model, other_dofs + self.grip_dofs, gain=1.0)
        task = self.left_task if side == "left" else self.right_task
        task.set_target(target)

        seeds = seeds_override if seeds_override is not None else self._build_arm_seeds(q_base, side)
        converged: list[tuple[np.ndarray, float, float, float]] = []
        best_fallback: tuple[np.ndarray, float, float, float] | None = None
        best_err_score = 1e9

        dt_solve = 0.1
        for arm_seed in seeds:
            q_s = q_base.copy()
            q_s[dofs] = arm_seed
            self.configuration.update(q_s)
            self.posture_task.set_target(q_s)

            for _ in range(max_iters):
                vel = mink.solve_ik(
                    self.configuration,
                    [task, self.posture_task, other_freeze],
                    dt_solve,
                    solver=self.solver,
                    damping=1e-6,
                    limits=[self.limits[0]],
                )
                vel[other_dofs + self.grip_dofs] = 0.0
                self.configuration.integrate_inplace(vel, dt_solve)
                p_m, o_r = self._pose_error(self.get_ee_pose(side), target)
                if p_m < 1e-5 and o_r < 1e-4:
                    break

            q_arm = self.configuration.q[dofs].copy()
            p_m, o_r = self._pose_error(self.get_ee_pose(side), target)
            dist = float(np.linalg.norm(q_arm - q_base[dofs]))
            err_score = p_m * 1000.0 + o_r * 50.0
            if err_score < best_err_score:
                best_err_score = err_score
                best_fallback = (q_arm, p_m, o_r, dist)

            if p_m <= pos_tol_m and o_r <= ori_tol_rad:
                if all(np.linalg.norm(q_arm - prev[0]) > 0.12 for prev in converged):
                    converged.append((q_arm, p_m, o_r, dist))
                if len(converged) >= max_branches:
                    break

        if converged:
            converged.sort(key=lambda item: item[3])
            return converged
        return [best_fallback] if best_fallback is not None else [(q_base[dofs].copy(), 1e9, 1e9, 0.0)]

    def solve_pose_global(
        self,
        q_seed: np.ndarray,
        left_target: mink.SE3 | None = None,
        right_target: mink.SE3 | None = None,
        collision_check_model: mujoco.MjModel | None = None,
        pos_tol_m: float = 2e-4,
        ori_tol_rad: float = 2e-3,
        max_iters: int = 120,
        scene_qpos_tail: np.ndarray | None = None,
        allowed_body_pairs: set[tuple[str, str]] | None = None,
    ) -> tuple[np.ndarray, dict[str, float | int | bool]]:
        """Solve global IK independently per arm and pick the closest collision-free combination."""
        saved_q = self.configuration.q.copy()
        saved_left = self.left_task.transform_target_to_world
        saved_right = self.right_task.transform_target_to_world
        saved_posture = self.posture_task.target_q.copy()
        saved_posture_cost = self.posture_task.cost.copy()
        saved_left_cost = self.left_task.cost.copy()
        saved_left_lm = self.left_task.lm_damping
        saved_right_cost = self.right_task.cost.copy()
        saved_right_lm = self.right_task.lm_damping

        col_data = mujoco.MjData(collision_check_model) if collision_check_model is not None else None

        def _check_ncon(q_cand: np.ndarray) -> int:
            if col_data is None or collision_check_model is None:
                return 0
            n_robot = len(q_cand)
            col_data.qpos[:n_robot] = q_cand
            if scene_qpos_tail is not None and collision_check_model.nq > n_robot:
                n_tail = min(len(scene_qpos_tail), collision_check_model.nq - n_robot)
                col_data.qpos[n_robot : n_robot + n_tail] = scene_qpos_tail[:n_tail]
            mujoco.mj_forward(collision_check_model, col_data)
            return self.count_robot_collisions(collision_check_model, col_data, allowed_body_pairs)

        self.posture_task.cost = np.full_like(saved_posture_cost, 1e-6)
        self.left_task.set_position_cost(10.0 if left_target is not None else 0.0)
        self.left_task.set_orientation_cost(3.0 if left_target is not None else 0.0)
        self.left_task.lm_damping = 1e-4
        self.right_task.set_position_cost(10.0 if right_target is not None else 0.0)
        self.right_task.set_orientation_cost(3.0 if right_target is not None else 0.0)
        self.right_task.lm_damping = 1e-4

        best_q = q_seed.copy()
        best_info: dict[str, float | int | bool] = {
            "pos_err_mm": 1e9,
            "ori_err_rad": 1e9,
            "ncon": 0,
            "converged": False,
        }

        try:
            # Fast path: try current q_seed directly first
            l0 = (
                self._solve_single_arm_branches(
                    q_seed, "left", left_target, pos_tol_m, ori_tol_rad, max_iters,
                    seeds_override=[q_seed[self.left_arm_dofs].copy()],
                )[0]
                if left_target is not None
                else (q_seed[self.left_arm_dofs].copy(), 0.0, 0.0, 0.0)
            )
            r0 = (
                self._solve_single_arm_branches(
                    q_seed, "right", right_target, pos_tol_m, ori_tol_rad, max_iters,
                    seeds_override=[q_seed[self.right_arm_dofs].copy()],
                )[0]
                if right_target is not None
                else (q_seed[self.right_arm_dofs].copy(), 0.0, 0.0, 0.0)
            )
            q_fast = q_seed.copy()
            q_fast[self.left_arm_dofs] = l0[0]
            q_fast[self.right_arm_dofs] = r0[0]
            fast_pos = max(l0[1], r0[1])
            fast_ori = max(l0[2], r0[2])
            if fast_pos <= pos_tol_m and fast_ori <= ori_tol_rad:
                fast_ncon = _check_ncon(q_fast)
                if fast_ncon == 0:
                    return q_fast, {
                        "pos_err_mm": fast_pos * 1000.0,
                        "ori_err_rad": fast_ori,
                        "ncon": 0,
                        "converged": True,
                        "branch_combo_index": 0,
                    }

            left_branches = (
                self._solve_single_arm_branches(q_seed, "left", left_target, pos_tol_m, ori_tol_rad, max_iters)
                if left_target is not None
                else [(q_seed[self.left_arm_dofs].copy(), 0.0, 0.0, 0.0)]
            )
            right_branches = (
                self._solve_single_arm_branches(q_seed, "right", right_target, pos_tol_m, ori_tol_rad, max_iters)
                if right_target is not None
                else [(q_seed[self.right_arm_dofs].copy(), 0.0, 0.0, 0.0)]
            )

            # Evaluate all (left_branch, right_branch) combinations sorted by (not_tol, err, joint_dist)
            candidates: list[tuple[tuple[int, float, float], np.ndarray, float, float]] = []
            for ql, pl, ol, dl in left_branches:
                for qr, pr, orr, dr in right_branches:
                    q_cand = q_seed.copy()
                    q_cand[self.left_arm_dofs] = ql
                    q_cand[self.right_arm_dofs] = qr
                    max_pos = max(pl, pr)
                    max_ori = max(ol, orr)
                    in_tol = (max_pos <= pos_tol_m) and (max_ori <= ori_tol_rad)
                    sort_key = (0 if in_tol else 1, dl + dr, max_pos * 1000.0 + max_ori * 50.0)
                    candidates.append((sort_key, q_cand, max_pos, max_ori))
            candidates.sort(key=lambda c: c[0])

            best_score = (1, 1, 1e9, 1e9)
            for idx_c, (_, q_cand, max_pos, max_ori) in enumerate(candidates):
                ncon = _check_ncon(q_cand)
                in_tol = (max_pos <= pos_tol_m) and (max_ori <= ori_tol_rad)
                arm_dofs = self.left_arm_dofs + self.right_arm_dofs
                j_dist = float(np.linalg.norm(q_cand[arm_dofs] - q_seed[arm_dofs]))
                score = (0 if in_tol else 1, ncon, max_pos * 1000.0 + max_ori * 50.0, j_dist)
                if score < best_score:
                    best_score = score
                    best_q = q_cand
                    best_info = {
                        "pos_err_mm": max_pos * 1000.0,
                        "ori_err_rad": max_ori,
                        "ncon": ncon,
                        "converged": bool(in_tol and ncon == 0),
                        "branch_combo_index": idx_c,
                    }
                    if in_tol and ncon == 0:
                        break
        finally:
            self.configuration.update(saved_q)
            self.left_task.set_target(saved_left)
            self.right_task.set_target(saved_right)
            self.posture_task.set_target(saved_posture)
            self.posture_task.cost = saved_posture_cost
            self.left_task.cost = saved_left_cost
            self.left_task.lm_damping = saved_left_lm
            self.right_task.cost = saved_right_cost
            self.right_task.lm_damping = saved_right_lm

        return best_q, best_info
