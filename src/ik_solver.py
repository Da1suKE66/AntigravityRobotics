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

        # Build velocity limits from URDF specs (rad/s) and identify gripper DOFs
        vel_limits = {}
        grip_dofs: list[int] = []
        self.left_arm_dofs: list[int] = []
        self.right_arm_dofs: list[int] = []
        self.grip_dofs: list[int] = []

        for jid in range(model.njnt):
            j = model.joint(jid)
            jname = j.name
            dof = int(j.dofadr[0])
            if "_joint_" in jname:
                idx = int(jname.rsplit("_", 1)[1])
                vel_limits[jname] = 5.585053606381854 if idx <= 4 else 12.566370614359172
                if jname.startswith("left_"):
                    self.left_arm_dofs.append(dof)
                elif jname.startswith("right_"):
                    self.right_arm_dofs.append(dof)
            elif "_grip_joint_" in jname:
                vel_limits[jname] = 1.5
                grip_dofs.append(dof)
                self.grip_dofs.append(dof)

        self.dof_freezing_task = (
            mink.DofFreezingTask(model, grip_dofs, gain=1.0)
            if (freeze_grippers and grip_dofs)
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

    def _build_seed_bank(self, q_base: np.ndarray, side: str | None = None) -> list[np.ndarray]:
        """Build deterministic multi-branch seed bank covering wrist-pole and elbow flips."""
        seeds = [q_base.copy()]
        if self.model.nkey > 0:
            seeds.append(self.model.key("home").qpos.copy())

        sides = [side] if side in ("left", "right") else ["left", "right"]
        # Branch perturbations: (J1_val_or_delta, is_abs_j1, J2, J3, J4, J5)
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
        ]
        for j1_v, abs_j1, j2, j3, j4, j5, j6 in patterns:
            q_s = q_base.copy()
            for s in sides:
                dofs = self.left_arm_dofs if s == "left" else self.right_arm_dofs
                q_s[dofs[0]] = j1_v if abs_j1 else (q_base[dofs[0]] + j1_v)
                q_s[dofs[1]] = j2
                q_s[dofs[2]] = j3
                q_s[dofs[3]] = j4
                q_s[dofs[4]] = j5
                q_s[dofs[5]] = j6
            seeds.append(self._clamp_to_limits(q_s))
        return seeds

    def solve_pose_global(
        self,
        q_seed: np.ndarray,
        left_target: mink.SE3 | None = None,
        right_target: mink.SE3 | None = None,
        collision_check_model: mujoco.MjModel | None = None,
        pos_tol_m: float = 2e-4,
        ori_tol_rad: float = 2e-3,
        max_iters: int = 120,
    ) -> tuple[np.ndarray, dict[str, float | int | bool]]:
        """Solve global IK across multi-branch seeds, returning lowest-excursion valid solution."""
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
        active_side = "left" if (left_target is not None and right_target is None) else (
            "right" if (right_target is not None and left_target is None) else None
        )
        seeds = self._build_seed_bank(q_seed, side=active_side)

        # Temporarily favor fast convergence with low posture regularization
        self.posture_task.cost = np.full_like(saved_posture_cost, 1e-6)
        self.left_task.set_position_cost(10.0 if left_target is not None else 0.0)
        self.left_task.set_orientation_cost(3.0 if left_target is not None else 0.0)
        self.left_task.lm_damping = 1e-4
        self.right_task.set_position_cost(10.0 if right_target is not None else 0.0)
        self.right_task.set_orientation_cost(3.0 if right_target is not None else 0.0)
        self.right_task.lm_damping = 1e-4

        best_q = q_seed.copy()
        best_score = (1, 1, 1e9, 1e9)  # (not_tol, ncon, err, joint_dist)
        best_info: dict[str, float | int | bool] = {
            "pos_err_mm": 1e9,
            "ori_err_rad": 1e9,
            "ncon": 0,
            "converged": False,
        }

        try:
            dt_solve = 0.1
            for s_idx, seed in enumerate(seeds):
                self.configuration.update(seed)
                self.posture_task.set_target(seed)
                t_l = left_target if left_target is not None else self.get_ee_pose("left")
                t_r = right_target if right_target is not None else self.get_ee_pose("right")
                self.set_targets(t_l, t_r)

                for _ in range(max_iters):
                    tasks: list[mink.Task] = [self.posture_task]
                    if left_target is not None:
                        tasks.append(self.left_task)
                    if right_target is not None:
                        tasks.append(self.right_task)
                    if self.dof_freezing_task is not None:
                        tasks.append(self.dof_freezing_task)
                    vel = mink.solve_ik(
                        self.configuration,
                        tasks,
                        dt_solve,
                        solver=self.solver,
                        damping=1e-6,
                        limits=[self.limits[0]],
                    )
                    self.configuration.integrate_inplace(vel, dt_solve)
                    err_l = self._pose_error(self.get_ee_pose("left"), t_l) if left_target is not None else (0.0, 0.0)
                    err_r = self._pose_error(self.get_ee_pose("right"), t_r) if right_target is not None else (0.0, 0.0)
                    if max(err_l[0], err_r[0]) < 1e-5 and max(err_l[1], err_r[1]) < 1e-4:
                        break

                q_cand = self.configuration.q.copy()
                err_l = self._pose_error(self.get_ee_pose("left"), t_l) if left_target is not None else (0.0, 0.0)
                err_r = self._pose_error(self.get_ee_pose("right"), t_r) if right_target is not None else (0.0, 0.0)
                max_pos = max(err_l[0], err_r[0])
                max_ori = max(err_l[1], err_r[1])
                in_tol = (max_pos <= pos_tol_m) and (max_ori <= ori_tol_rad)

                ncon = 0
                if col_data is not None and collision_check_model is not None:
                    col_data.qpos[: len(q_cand)] = q_cand
                    mujoco.mj_forward(collision_check_model, col_data)
                    ncon = int(col_data.ncon)

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
                        "seed_index": s_idx,
                    }
                    # If initial seed already converged cleanly without collision, return immediately
                    if s_idx == 0 and in_tol and ncon == 0:
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
