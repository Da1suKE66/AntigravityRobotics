"""Constrained differential IK solver for Firefly Dual-Arm using Mink + DAQP."""

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
    ) -> None:
        self.model = model
        self.solver = solver
        self.damping = damping
        self.configuration = mink.Configuration(model)

        self.left_task = mink.FrameTask(
            frame_name="left_ee_site",
            frame_type="site",
            position_cost=pos_cost,
            orientation_cost=ori_cost,
            lm_damping=1.0,
        )
        self.right_task = mink.FrameTask(
            frame_name="right_ee_site",
            frame_type="site",
            position_cost=pos_cost,
            orientation_cost=ori_cost,
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(model, cost=posture_cost)

        # Build velocity limits from URDF specs (rad/s)
        vel_limits = {}
        for jid in range(model.njnt):
            jname = model.joint(jid).name
            if "_joint_" in jname:
                idx = int(jname.rsplit("_", 1)[1])
                vel_limits[jname] = 5.585053606381854 if idx <= 4 else 12.566370614359172
            elif "_grip_joint_" in jname:
                vel_limits[jname] = 1.5

        self.limits = [
            mink.ConfigurationLimit(model),
            mink.VelocityLimit(model, vel_limits),
        ]

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
        return self.configuration.get_transform_frame_to_world(f"{side}_ee_site", "site")

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

        tasks = [self.left_task, self.right_task, self.posture_task]
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
