"""Targeted simulation and IK checks. Run directly with the repository's MuJoCo Python."""

from __future__ import annotations

import pathlib
import sys
import unittest

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.controller import BumplessGravityController
from src.grasp_pipeline import GraspExecutor, R_TOP_DOWN
from src.ik_solver import FireflyDualIK
from src.planner import FireflyPlanner


class SimulationIKAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.robot = mujoco.MjModel.from_xml_path(str(ROOT / "model/firefly_dual.xml"))
        cls.scene = mujoco.MjModel.from_xml_path(str(ROOT / "model/scene.xml"))
        cls.home = cls.robot.key("home").qpos.copy()
        cls.arms = [
            cls.robot.joint(f"{side}_joint_{i}").qposadr[0]
            for side in ("left", "right") for i in range(1, 7)
        ]

    def _check_pose_from_home(self, ik: FireflyDualIK, data: mujoco.MjData) -> str | None:
        """Use the current collision-free FK pose as the target, then independently check the result."""
        sites = [data.site(f"{side}_tcp_site") for side in ("left", "right")]
        targets = [
            mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(site.xmat.reshape(3, 3)), site.xpos.copy()
            )
            for site in sites
        ]
        q, info = ik.solve_pose_global(self.home, *targets, collision_check_model=self.scene)
        data.qpos[:] = q
        mujoco.mj_forward(self.scene, data)
        pos_mm, ori_rad = 0.0, 0.0
        for target, site in zip(targets, sites):
            pos_mm = max(pos_mm, np.linalg.norm(site.xpos - target.translation()) * 1000)
            relative = target.rotation().as_matrix().T @ site.xmat.reshape(3, 3)
            ori_rad = max(ori_rad, Rotation.from_matrix(relative).magnitude())
        if info["converged"] and data.ncon == 0 and pos_mm <= 0.2 and ori_rad <= 0.002:
            return None
        return f"{pos_mm:.3f} mm, {ori_rad:.5f} rad, ncon={data.ncon}, converged={info['converged']}"

    def test_near_home_reachable_dual_arm_poses(self) -> None:
        ik = FireflyDualIK(self.robot, site_suffix="tcp_site")
        data = mujoco.MjData(self.scene)
        rng = np.random.default_rng(23)
        for sample in range(12):
            with self.subTest(sample=sample):
                data.qpos[:] = self.home
                data.qpos[self.arms] += rng.uniform(-0.2, 0.2, len(self.arms))
                mujoco.mj_forward(self.scene, data)
                self.assertEqual(data.ncon, 0, "Reference pose for this check must be contact-free")
                self.assertIsNone(self._check_pose_from_home(ik, data))

    def test_wide_workspace_reachable_dual_arm_poses(self) -> None:
        ik = FireflyDualIK(self.robot, site_suffix="tcp_site")
        data = mujoco.MjData(self.scene)
        rng = np.random.default_rng(23)
        low, high = self.robot.jnt_range[self.arms].T
        margin = 0.05 * (high - low)
        checked, misses = 0, []
        for draw in range(1, 1001):
            data.qpos[:] = self.home
            data.qpos[self.arms] = rng.uniform(low + margin, high - margin)
            mujoco.mj_forward(self.scene, data)
            if data.ncon:
                continue
            checked += 1
            problem = self._check_pose_from_home(ik, data)
            if problem:
                misses.append(f"#{checked} (draw {draw}): {problem}")
            if checked == 40:
                break
        self.assertEqual(checked, 40, "Could not generate 40 collision-free reference poses")
        self.assertEqual(len(misses), 0, f"Missed {len(misses)}/{checked} known reachable poses:\n" + "\n".join(misses))

    def test_arm_ik_leaves_measured_grippers_stationary(self) -> None:
        ik = FireflyDualIK(self.robot, freeze_grippers=True)
        ik.reset_to_measured(self.home)
        measured = self.home.copy()
        fingers = [
            self.robot.joint(f"{side}_grip_joint_{i}") for side in ("left", "right") for i in (1, 2)
        ]
        measured[[joint.qposadr[0] for joint in fingers]] = 0.5
        _, velocity = ik.step(0.002, q_meas=measured, blend_meas=1.0)
        finger_speed = np.abs(velocity[[joint.dofadr[0] for joint in fingers]])
        self.assertLessEqual(float(np.max(finger_speed)), 1e-6, "Arm IK commanded finger motion (rad/s)")

    def test_mimic_finger_has_no_independent_drive_torque(self) -> None:
        data = mujoco.MjData(self.robot)
        data.qpos[:] = self.home
        for act_id in range(self.robot.nu):
            joint_id = self.robot.actuator_trnid[act_id, 0]
            if "_grip_joint_" in self.robot.joint(joint_id).name:
                data.ctrl[act_id] = self.robot.actuator_ctrlrange[act_id, 1]
        mujoco.mj_forward(self.robot, data)
        # The URDF gives the driver 5 Nm and the coupled follower 0 Nm.
        for finger, limit in ((1, 5.0), (2, 0.0)):
            torques = [
                data.joint(f"{side}_grip_joint_{finger}").qfrc_actuator[0] for side in ("left", "right")
            ]
            self.assertLessEqual(
                max(abs(torque) for torque in torques), limit + 1e-9,
                f"Finger {finger} direct actuator torques (Nm): {torques}",
            )

    def test_position_hold_with_physical_half_kg_payload(self) -> None:
        spec = mujoco.MjSpec.from_file(str(ROOT / "model/scene.xml"))
        payload = spec.body("left_link_6").add_body(name="audit_payload", pos=[0, 0, 0.12])
        payload.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02], mass=0.5, contype=0, conaffinity=0
        )
        plant = spec.compile()
        self.assertAlmostEqual(mujoco.mj_getTotalmass(plant) - mujoco.mj_getTotalmass(self.scene), 0.5)
        data = mujoco.MjData(plant)
        data.qpos[:] = self.home
        data.joint("left_joint_2").qpos = 0.85
        data.joint("left_joint_3").qpos = 0.70
        mujoco.mj_forward(plant, data)
        q_goal = data.qpos.copy()
        site = data.site("left_ee_site")
        target = site.xpos.copy()
        ctrl = BumplessGravityController(self.robot, engage_duration=0.0)
        ctrl.engage(q_goal, data.qvel.copy())
        for _ in range(round(3.0 / plant.opt.timestep)):
            data.ctrl[:] = ctrl.step(plant.opt.timestep, data.qpos, data.qvel, q_goal)
            mujoco.mj_step(plant, data)
        # Use the same readout and 3 s acceptance threshold as verify_sim_ik.py.
        self.assertLess(np.linalg.norm(site.xpos - target) * 1000, 1.5)

    def test_joint_plan_respects_position_limit(self) -> None:
        planner = FireflyPlanner(self.scene, FireflyDualIK(self.robot))
        goal = self.home.copy()
        joint = self.robot.joint("left_joint_1")
        dof = joint.qposadr[0]
        goal[dof] = 3.0
        try:
            path = planner.plan_joint_p2p(self.home, goal)
        except ValueError:
            return  # Explicit rejection is valid.
        self.assertLessEqual(np.max(path.q[:, dof]), joint.range[1])

    def test_joint_plan_respects_velocity_limit(self) -> None:
        planner = FireflyPlanner(self.scene, FireflyDualIK(self.robot))
        goal = self.home.copy()
        dof = self.robot.joint("left_joint_1").dofadr[0]
        goal[dof] = 0.5
        try:
            path = planner.plan_joint_p2p(self.home, goal, duration=0.1)
        except ValueError:
            return
        self.assertLessEqual(np.max(np.abs(path.dq[:, dof])), 5.585053606381854 + 1e-8)

    def test_grasp_stage_declared_clear_has_no_table_contact(self) -> None:
        executor = GraspExecutor(record_video=False)
        self.addCleanup(executor.close)
        advance = executor._step_control
        table = executor.sim_model.geom("tabletop").id
        model = executor.sim_model
        robot_geoms = {
            i for i in range(model.ngeom)
            if model.body(model.geom_bodyid[i]).name.startswith(("left_", "right_"))
        }
        contact_steps = 0

        def observe(q_ref: np.ndarray, dq_ref: np.ndarray) -> None:
            nonlocal contact_steps
            advance(q_ref, dq_ref)
            contact_steps += any(
                (c.geom1 == table and c.geom2 in robot_geoms) or
                (c.geom2 == table and c.geom1 in robot_geoms)
                for c in executor.data.contact
            )

        executor._step_control = observe
        target = mink.SE3.from_rotation_and_translation(R_TOP_DOWN, np.array([0.40, 0.16, 0.04]))
        try:
            result = executor._move_pose("audit:table", "left", target, 1.6, [])
        except (ValueError, RuntimeError):
            return  # The stage explicitly rejected or aborted the move.
        if result["collision_free"]:
            self.assertEqual(contact_steps, 0, f"Stage reported clear with {contact_steps} robot/table contact steps")

    def test_grasp_executor_rejects_unreachable_goal_before_moving(self) -> None:
        executor = GraspExecutor(record_video=False)
        self.addCleanup(executor.close)
        target = mink.SE3.from_rotation_and_translation(R_TOP_DOWN, np.array([2.0, 0.16, 0.25]))
        initial = executor.data.qpos[:self.robot.nq].copy()
        try:
            executor._move_pose("audit:unreachable", "left", target, 1.6, [])
        except (ValueError, RuntimeError):
            pass
        displacement = np.abs(executor.data.qpos[:self.robot.nq] - initial)
        self.assertLessEqual(np.max(displacement), 1e-5, "Robot moved toward an infeasible pose (rad)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
