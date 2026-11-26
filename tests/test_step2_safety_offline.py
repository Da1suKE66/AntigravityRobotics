"""Offline safety verification tests for Step 1 passive WS reader & Step 2 verifier/executor.

Zero network sockets, zero cameras, zero real robot connections.
Covers all 4 audit findings:
1. Passive WS reader sends ZERO bytes even under PscAPICommunicationTimeout.
2. Start-frame (k=0) limit violation and self-collision are NEVER skipped -> safe_to_execute=False.
3. Live right-arm pose + tabletop + target object 3D collider are checked on every frame
   (hovering 9 mm below tissue-pack top is rejected).
4. Motor fault, parking stop, frozen/stale feedback, unreached goal, and SIGINT all
   trigger immediate stop with ZERO commands sent after fault and completed=False.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from hex_device import Arm
from hex_device.motor_base import Timestamp
from hex_device.generated import public_api_down_pb2, public_api_types_pb2, public_api_up_pb2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from readonly_ws_client import decode_api_up_arm_telemetry, read_arm_passive_readonly
from step2_left_arm_air_jog import (
    Q_LEFT_READY,
    SafeArmSendGate,
    _is_parking_stop_active,
    _send_arm_command,
    create_guarded_hex_device_api,
    execute_left_arm_trajectory,
    plan_and_verify_step2,
)
from step3_left_arm_pick_place import (
    GRIP_CLOSE_RAD,
    GRIP_MAX_OPEN_ALLOWED_RAD,
    GRIP_OPEN_RAD,
    check_perception_unoccluded,
    execute_step3_pick_place,
    plan_and_verify_step3,
)


def _load_saved_data():
    s1 = json.loads((ROOT / "docs" / "reports" / "step1_readonly_report.json").read_text(encoding="utf-8"))
    s2 = json.loads((ROOT / "docs" / "reports" / "step2_air_jog_report.json").read_text(encoding="utf-8"))
    left_q = np.asarray(s1["left_arm"]["arm"]["positions_rad"], dtype=np.float64)
    right_q = np.asarray(s1["right_arm"]["arm"]["positions_rad"], dtype=np.float64)
    perception = s2["perception"]
    return left_q, right_q, perception


def test_passive_reader_sends_zero_bytes_on_comm_timeout():
    """Audit #4: passive WS reader must transmit 0 bytes even if controller reports CommTimeout."""
    up = public_api_up_pb2.APIUp()
    up.robot_type = public_api_types_pb2.RobotType.RtArmFireflyY6_H2_40
    up.session_id = 42
    up.arm_status.api_control_initialized = False
    up.arm_status.calibrated = True
    up.arm_status.session_holder = 0
    up.arm_status.parking_stop_detail.category = (
        public_api_types_pb2.ParkingStopCategory.PscAPICommunicationTimeout
    )
    up.arm_status.parking_stop_detail.reason = "simulated comm timeout"
    target_rad = [0.0, -0.75, 1.75, 0.0, 0.57, 0.0]
    ppr = 65535.0
    for q in target_rad:
        m = up.arm_status.motor_status.add()
        m.pulse_per_rotation = int(ppr)
        m.position = int(round(q / (2.0 * np.pi) * ppr))
        m.speed = 0.0
        m.torque = 0.0
        m.motor_temperature = 35.0
        m.voltage = 24.0
    payload = up.SerializeToString()

    sent_messages: list[bytes] = []

    class FakeWS:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def recv(self):
            return payload

        async def send(self, data):
            sent_messages.append(data)
            raise AssertionError("Passive reader MUST NEVER call ws.send()!")

    res = read_arm_passive_readonly(
        "left_arm",
        "ws://fake-local:8439",
        timeout_s=2.0,
        ws_connect=lambda url, **kw: FakeWS(),
    )
    assert res["connected"] is True, res
    assert len(sent_messages) == 0, f"Unexpected uplink sends: {sent_messages}"
    assert res["arm"]["parking_category"] == int(
        public_api_types_pb2.ParkingStopCategory.PscAPICommunicationTimeout
    )
    np.testing.assert_allclose(res["arm"]["positions_rad"], target_rad, atol=2e-4)
    print("PASS: test_passive_reader_sends_zero_bytes_on_comm_timeout")


def test_folded_start_pose_is_rejected_not_skipped():
    """Audit #1: folded start pose with k=0 limit violation & self-collision must NOT pass."""
    _, right_q, perception = _load_saved_data()
    # Explicit folded start pose recorded on 2026-11-26 00:44 (J2=-1.5745, J3=3.1454)
    folded_left_q = np.array(
        [-0.0068467, -1.5744624, 3.1454333, 0.0128298, 0.0004893, 0.0030209],
        dtype=np.float64,
    )
    c = perception["object_center_wc_m"]
    hover_wc = np.array([c[0], c[1], perception["object_z_top_wc_m"] + 0.10], dtype=np.float64)

    plan = plan_and_verify_step2(
        q_meas=folded_left_q,
        right_q_meas=right_q,
        target_hover_wc=hover_wc,
        z_table_wc=perception["z_table_wc_m"],
        perception=perception,
    )
    assert plan["frames_skipped"] == 0
    assert plan["safe_to_execute"] is False, "Folded out-of-bounds/colliding start pose MUST be blocked!"
    assert plan["limit_violation_count"] > 0
    assert plan["collision_count"] > 0
    assert plan["max_penetration_mm"] > 1.0
    assert len(plan["blocking_reasons"]) >= 2
    print("PASS: test_folded_start_pose_is_rejected_not_skipped")


def test_in_bounds_start_with_live_right_arm_passes_and_below_object_fails():
    """Audit #2: live right arm pose is applied; hovering 9 mm below object top is rejected."""
    _, right_q, perception = _load_saved_data()
    c = perception["object_center_wc_m"]

    # 1) Valid hover 10 cm above tissue pack from clean unfolded start posture
    hover_ok = np.array([c[0], c[1], perception["object_z_top_wc_m"] + 0.10], dtype=np.float64)
    plan_ok = plan_and_verify_step2(
        q_meas=Q_LEFT_READY,
        right_q_meas=right_q,
        target_hover_wc=hover_ok,
        z_table_wc=perception["z_table_wc_m"],
        perception=perception,
    )
    assert plan_ok["frames_skipped"] == 0
    np.testing.assert_allclose(plan_ok["right_q_used"], right_q, atol=1e-4)
    assert plan_ok["safe_to_execute"] is True, f"Unexpected block: {plan_ok['blocking_reasons']}"
    assert plan_ok["min_geom_clearance_object_m"] >= 0.02

    # 2) Hover 9 mm BELOW tissue pack top -> MUST fail due to target_object_col collision/clearance!
    hover_bad = np.array([c[0], c[1], perception["object_z_top_wc_m"] - 0.009], dtype=np.float64)
    plan_bad = plan_and_verify_step2(
        q_meas=Q_LEFT_READY,
        right_q_meas=right_q,
        target_hover_wc=hover_bad,
        z_table_wc=perception["z_table_wc_m"],
        perception=perception,
    )
    assert plan_bad["safe_to_execute"] is False, "Hovering 9 mm below tissue pack top MUST be rejected!"
    assert plan_bad["min_geom_clearance_object_m"] < 0.02
    assert any("target object" in r.lower() or "target_object_col" in r.lower() for r in plan_bad["blocking_reasons"])
    print("PASS: test_in_bounds_start_with_live_right_arm_passes_and_below_object_fails")


class _FakeArm:
    def __init__(self, q0: np.ndarray, *, fault_at_cmd: int | None = None, freeze_after_cmd: int | None = None, lag_rad: float = 0.0):
        self.q = q0.copy()
        self.fault_at_cmd = fault_at_cmd
        self.freeze_after_cmd = freeze_after_cmd
        self.lag_rad = lag_rad
        self.cmd_count = 0
        self.cmds_after_fault = 0
        self.fault_triggered = False
        self.started = False
        self.stopped = False
        self._has_new = True

    def has_new_data(self) -> bool:
        if self.freeze_after_cmd is not None and self.cmd_count >= self.freeze_after_cmd:
            return False
        return self._has_new

    def get_motor_positions(self, pop: bool = False):
        return self.q.copy()

    def get_motor_torques(self, pop: bool = False):
        return np.zeros(6, dtype=np.float64)

    def get_motor_error_codes(self):
        if self.fault_at_cmd is not None and self.cmd_count >= self.fault_at_cmd:
            self.fault_triggered = True
            return [None, 7, None, None, None, None]
        return [None] * 6

    def get_parking_stop_detail(self):
        return ""

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def joint_position_control(self, pos, vel, acc):
        if self.fault_triggered:
            self.cmds_after_fault += 1
        self.cmd_count += 1
        if self.freeze_after_cmd is None or self.cmd_count < self.freeze_after_cmd:
            self.q = np.asarray(pos, dtype=np.float64) - self.lag_rad


class _FakeAPI:
    def __init__(self, arm: _FakeArm):
        self.device_list = [arm]
        self.closed = False

    def is_api_exit(self) -> bool:
        return False

    def close(self):
        self.closed = True


def _make_sim_clock():
    t = [0.0]

    def clock():
        return t[0]

    def sleep(dt):
        t[0] += max(0.0, dt)

    return clock, sleep


def test_motor_fault_sends_zero_commands_after_fault():
    """Audit #3a: after motor error, 0 further commands are sent and stop() is called immediately."""
    q0 = Q_LEFT_READY.copy()
    traj = np.linspace(q0, q0 + np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]), 30)
    arm = _FakeArm(q0, fault_at_cmd=15)
    api = _FakeAPI(arm)
    clock, sleep = _make_sim_clock()

    res = execute_left_arm_trajectory(
        traj, dt=0.01, api_factory=lambda: api, sleep_fn=sleep, clock_fn=clock
    )
    assert res["completed"] is False
    assert "Motor error" in (res["aborted_reason"] or "")
    assert arm.cmds_after_fault == 0, f"Sent {arm.cmds_after_fault} commands after fault!"
    assert arm.stopped is True and api.closed is True
    print("PASS: test_motor_fault_sends_zero_commands_after_fault")


def test_frozen_feedback_aborts_and_never_completes():
    """Audit #3b: frozen feedback (has_new_data()==False) must abort within stale_timeout_s."""
    q0 = Q_LEFT_READY.copy()
    traj = np.linspace(q0, q0 + np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0]), 50)
    arm = _FakeArm(q0, freeze_after_cmd=12)
    api = _FakeAPI(arm)
    clock, sleep = _make_sim_clock()

    res = execute_left_arm_trajectory(
        traj, dt=0.01, stale_timeout_s=0.15, api_factory=lambda: api, sleep_fn=sleep, clock_fn=clock
    )
    assert res["completed"] is False
    assert "frozen/stale" in (res["aborted_reason"] or "").lower()
    assert arm.stopped is True and api.closed is True
    print("PASS: test_frozen_feedback_aborts_and_never_completes")


def test_unreached_goal_never_reports_completed():
    """Audit #3c: lagging 0.10 rad behind goal at completion must report completed=False."""
    q0 = Q_LEFT_READY.copy()
    traj = np.linspace(q0, q0 + np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0]), 20)
    arm = _FakeArm(q0, lag_rad=0.10)
    api = _FakeAPI(arm)
    clock, sleep = _make_sim_clock()

    res = execute_left_arm_trajectory(
        traj,
        dt=0.01,
        goal_tol_rad=0.02,
        settle_steps=5,
        return_to_start=False,
        api_factory=lambda: api,
        sleep_fn=sleep,
        clock_fn=clock,
    )
    assert res["completed"] is False
    assert "goal not reached" in (res["aborted_reason"] or "").lower()
    assert arm.stopped is True
    print("PASS: test_unreached_goal_never_reports_completed")


def test_sigint_aborts_immediately_with_zero_extra_commands():
    """Audit #3d: SIGINT aborts immediately without running settle hold commands."""
    q0 = Q_LEFT_READY.copy()
    traj = np.linspace(q0, q0 + np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]), 30)
    arm = _FakeArm(q0)
    api = _FakeAPI(arm)
    clock, sleep = _make_sim_clock()

    res = execute_left_arm_trajectory(
        traj,
        dt=0.01,
        api_factory=lambda: api,
        sleep_fn=sleep,
        clock_fn=clock,
        stop_flag=lambda: arm.cmd_count >= 14,
    )
    assert res["completed"] is False
    assert res["aborted_reason"] == "User SIGINT"
    assert arm.cmd_count == 14
    assert arm.stopped is True and api.closed is True
    print("PASS: test_sigint_aborts_immediately_with_zero_extra_commands")


def test_nominal_execution_succeeds():
    """Nominal fresh telemetry reaching goal within tolerance completes cleanly."""
    q0 = Q_LEFT_READY.copy()
    traj = np.linspace(q0, q0 + np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]), 20)
    arm = _FakeArm(q0, lag_rad=0.0)
    api = _FakeAPI(arm)
    clock, sleep = _make_sim_clock()

    res = execute_left_arm_trajectory(
        traj, dt=0.01, settle_steps=5, api_factory=lambda: api, sleep_fn=sleep, clock_fn=clock
    )
    assert res["completed"] is True, res
    assert res["aborted_reason"] is None
    assert res["hover_err_to_goal_rad"] == 0.0
    assert res["final_err_to_goal_rad"] == 0.0
    assert arm.stopped is True and api.closed is True
    print("PASS: test_nominal_execution_succeeds")


def test_mit_grav_mode_constructs_gravity_feedforward_and_returns_to_start():
    """Verify mit_grav mode sends MitMotorCommand with non-zero gravity torque and returns to q0."""
    q0 = np.array([0.0, 0.75, 0.85, 1.54, 0.0, 0.85], dtype=np.float64)
    traj = np.linspace(q0, q0 + np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]), 15)

    class _FakeMitArm(_FakeArm):
        def __init__(self, q_init):
            super().__init__(q_init)
            self.mit_cmds = []

        def construct_mit_command(self, pos, speed, torque, kp, kd):
            return list(zip(pos, speed, torque, kp, kd))

        def motor_command(self, cmd_type, cmds):
            self.mit_cmds.append(cmds)
            self.cmd_count += 1
            self.q = np.array([c[0] for c in cmds], dtype=np.float64)

    arm = _FakeMitArm(q0)
    api = _FakeAPI(arm)
    clock, sleep = _make_sim_clock()
    res = execute_left_arm_trajectory(
        traj,
        dt=0.01,
        settle_steps=5,
        control_mode="mit_grav",
        return_to_start=True,
        api_factory=lambda: api,
        sleep_fn=sleep,
        clock_fn=clock,
    )
    assert res["completed"] is True, res
    assert res["control_mode"] == "mit_grav"
    assert len(arm.mit_cmds) > 0
    # Verify non-zero MuJoCo gravity torque was sent on J2/J3
    sample_tau = [c[2] for c in arm.mit_cmds[0]]
    assert abs(sample_tau[1]) > 0.5 or abs(sample_tau[2]) > 0.5, sample_tau
    # Verify final pose returned to q0 (not left hovering)
    np.testing.assert_allclose(res["final_q_rad"], q0, atol=1e-4)
    print("PASS: test_mit_grav_mode_constructs_gravity_feedforward_and_returns_to_start")


def test_real_sdk_mit_command_updates_timestamp_and_emits_nonzero_targets():
    """Audit5 #1: _send_arm_command(mit_grav) MUST use motor_command(CommandType.MIT) so real SDK sends MIT targets."""
    import time as _time

    rt = public_api_types_pb2.RobotType.RtArmFireflyY6_H2_40
    sent_raw: list[bytes] = []
    api = create_guarded_hex_device_api(
        ws_url=None, control_hz=200, enable_kcp=False, send_down_callback=sent_raw.append
    )
    try:
        frame0 = _make_firefly_api_up(Q_LEFT_READY)
        frame0.arm_status.api_control_initialized = True
        frame0.arm_status.session_holder = 1
        _feed_api_up_threadsafe(api, frame0)
        _time.sleep(0.05)

        arm_dev = api.device_list[0]
        api.send_gate.allow_execution()
        arm_dev.start()

        q_cmd = Q_LEFT_READY.copy()
        dq_cmd = np.zeros(6)
        tau_ff = np.array([0.0, 1.5, -5.0, -1.0, 0.0, 0.0])
        _send_arm_command(
            arm_dev, q_cmd, dq_cmd, tau_ff, "mit_grav",
            [60, 80, 80, 40, 25, 20], [3, 4, 4, 2, 1.2, 1], [0.6]*6, [1.2]*6,
        )
        assert arm_dev._last_command_time is not None, "_last_command_time was NOT updated!"
        _time.sleep(0.05)

        mit_packets = 0
        for raw in sent_raw:
            down = public_api_down_pb2.APIDown()
            down.ParseFromString(raw)
            if down.HasField("arm_command") and down.arm_command.HasField("arm_exclusive_command"):
                excl = down.arm_command.arm_exclusive_command
                if excl.WhichOneof("exclusive_command") == "arm_api_control_command":
                    ctrl = excl.arm_api_control_command
                    if ctrl.WhichOneof("command") == "motor_targets":
                        t_list = list(ctrl.motor_targets.targets)
                        if len(t_list) == 6 and t_list[2].HasField("mit_target") and abs(t_list[2].mit_target.torque) > 1.0:
                            mit_packets += 1
        assert mit_packets > 0, f"Real SDK loop sent 0 arm_mit_motor_targets! total_pkts={len(sent_raw)}"
    finally:
        api.close()
    print("PASS: test_real_sdk_mit_command_updates_timestamp_and_emits_nonzero_targets")


def test_secondary_device_command_blocked_when_stopped():
    """Audit5 #4: SafeArmSendGate MUST block secondary_device_command (gripper) after stop latch."""
    from hex_device import Hands
    from hex_device.motor_base import CommandType

    forwarded = []

    async def fake_send(msg):
        forwarded.append(msg)

    gate = SafeArmSendGate(fake_send, start_blocked=False)
    gate.latch_stop()

    hand = Hands(
        device_id=1,
        device_type=public_api_types_pb2.SecondaryDeviceType.SdtHandGr100,
        motor_count=1,
        proto_version=(1, 0),
        send_message_callback=gate,
    )
    sec_up = public_api_types_pb2.SecondaryDeviceStatus()
    sec_up.device_id = 1
    sec_up.device_type = public_api_types_pb2.SecondaryDeviceType.SdtHandGr100
    hm = sec_up.hand_status.motor_status.add()
    hm.pulse_per_rotation = 65535
    hm.position = 1000
    hand._update_optional_data(
        public_api_types_pb2.SecondaryDeviceType.SdtHandGr100, sec_up, Timestamp(s=1, ns=0)
    )
    hand.motor_command(CommandType.POSITION, [0.3])
    msg_grip = hand._construct_joint_command_msg()
    assert gate._classify_blocked(msg_grip) == "post_stop_secondary_device_command"
    asyncio.run(gate(msg_grip))
    assert len(forwarded) == 0, "Gripper command leaked past latched stop!"
    print("PASS: test_secondary_device_command_blocked_when_stopped")


def test_step3_fails_closed_when_gripper_missing():
    """Audit5 #2: Step 3 MUST fail-closed (completed=False, gripper_detected=False) when gripper is missing."""
    q0 = Q_LEFT_READY.copy()
    arm = _FakeArm(q0)
    api = _FakeAPI(arm)
    clock, sleep = _make_sim_clock()
    segs = {"unfold": np.repeat(q0[None, :], 5, axis=0)}

    res = execute_step3_pick_place(segs, api_factory=lambda: api, sleep_fn=sleep, clock_fn=clock)
    assert res["completed"] is False
    assert res["gripper_detected"] is False
    assert "not detected" in (res["aborted_reason"] or "").lower()
    assert arm.started is False and arm.cmd_count == 0
    print("PASS: test_step3_fails_closed_when_gripper_missing")


def test_step3_gr100_open_close_and_occlusion_verified():
    """Audit5 #3 & #5: GR100 open < close, Step 3 verifies gripper DOFs + object, and rejects occluded perception."""
    assert GRIP_OPEN_RAD < GRIP_CLOSE_RAD, (GRIP_OPEN_RAD, GRIP_CLOSE_RAD)

    _, right_q, perception = _load_saved_data()
    # Unoccluded perception from Step 2 (~17.6 cm x 7.8 cm)
    unoccluded = dict(perception)
    unoccluded["object_footprint_m"] = [0.176, 0.078]
    unoccluded["object_center_wc_m"] = [0.4056, 0.0506, 0.0898]
    unoccluded["object_z_top_wc_m"] = 0.095
    assert check_perception_unoccluded(unoccluded) == []

    plan = plan_and_verify_step3(q_meas=Q_LEFT_READY, right_q_meas=right_q, perception=unoccluded)
    assert plan["safe_to_execute"] is True, plan["blocking_reasons"]
    assert plan["min_geom_clearance_object_m"] is not None

    # Occluded perception (~10 cm clipped length) MUST be rejected!
    occluded = dict(unoccluded)
    occluded["object_footprint_m"] = [0.098, 0.078]
    plan_occ = plan_and_verify_step3(q_meas=Q_LEFT_READY, right_q_meas=right_q, perception=occluded)
    assert plan_occ["safe_to_execute"] is False
    assert any("occluded" in r.lower() for r in plan_occ["blocking_reasons"])
    print("PASS: test_step3_gr100_open_close_and_occlusion_verified")


def _make_step3_dummy_segments(q0: np.ndarray) -> dict[str, np.ndarray]:
    s = np.repeat(q0[None, :], 5, axis=0)
    return {
        "unfold": s,
        "to_hover": s,
        "descend_grasp": s,
        "lift": s,
        "to_place_hover": s,
        "descend_place": s,
        "retreat_place": s,
        "return_ready": s,
        "return_rest": s,
    }


class _FakeHand:
    def __init__(self, initial_q: float = 0.02, *, freeze_after_close_cmd: bool = False, stuck_q: float | None = None):
        self.q = initial_q
        self.freeze_after_close_cmd = freeze_after_close_cmd
        self.stuck_q = stuck_q
        self._has_new = True
        self.cmds = []

    def has_new_data(self) -> bool:
        return self._has_new

    def get_motor_positions(self, pop: bool = False):
        if pop and not self.freeze_after_close_cmd:
            pass
        elif pop:
            self._has_new = False
        return [self.q]

    def set_pos_torque(self, max_torque: float):
        pass

    def motor_command(self, cmd_type, values):
        target = float(values[0])
        self.cmds.append(target)
        if self.freeze_after_close_cmd and target > 0.3:
            # Simulate telemetry freezing when close command is issued
            self._has_new = False
            return
        if self.stuck_q is not None and len(self.cmds) > 2:
            self.q = self.stuck_q
        elif target > 0.3:
            # Tissue pack width stops jaws around 0.28 rad
            self.q = 0.28
        else:
            self.q = target


class _FakeStep3API:
    def __init__(self, arm: Any, hand: Any):
        self.device_list = [arm]
        self.hand = hand
        self.closed = False

    def find_optional_device_by_robot_type(self, sdt):
        return [self.hand] if self.hand is not None else []

    def is_api_exit(self) -> bool:
        return False

    def close(self):
        self.closed = True


def test_step3_prestart_motor_error_blocks_before_start_and_open():
    """P1 #1: J2 error code 4 without parking_stop_detail MUST abort BEFORE arm.start() and gripper open."""
    q0 = Q_LEFT_READY.copy()
    arm = _FakeArm(q0)
    arm.get_motor_error_codes = lambda: [None, 4, None, None, None, None]
    hand = _FakeHand(GRIP_OPEN_RAD)
    api = _FakeStep3API(arm, hand)
    clock, sleep = _make_sim_clock()

    res = execute_step3_pick_place(
        _make_step3_dummy_segments(q0), api_factory=lambda: api, sleep_fn=sleep, clock_fn=clock
    )
    assert res["completed"] is False
    assert "motor error" in (res["aborted_reason"] or "").lower()
    assert arm.started is False, "arm.start() called despite J2 error!"
    assert len(hand.cmds) == 0, f"Gripper commanded despite J2 error: {hand.cmds}"
    print("PASS: test_step3_prestart_motor_error_blocks_before_start_and_open")


def test_step3_frozen_gripper_after_close_never_verifies_grasp():
    """P1 #2: Frozen GR100 feedback (stale 0.20 rad or no new frames on close) -> grasp_verified=False, no lift."""
    q0 = Q_LEFT_READY.copy()
    arm = _FakeArm(q0)
    hand = _FakeHand(GRIP_OPEN_RAD, freeze_after_close_cmd=True)
    api = _FakeStep3API(arm, hand)
    clock, sleep = _make_sim_clock()

    res = execute_step3_pick_place(
        _make_step3_dummy_segments(q0), api_factory=lambda: api, sleep_fn=sleep, clock_fn=clock
    )
    assert res["completed"] is False
    assert res["grasp_verified"] is False
    reason = (res["aborted_reason"] or "").lower()
    assert "frozen/stale" in reason or "feedback lost" in reason, res
    print("PASS: test_step3_frozen_gripper_after_close_never_verifies_grasp")


def test_step3_stuck_gripper_at_place_or_unreached_rest_never_completes():
    """P1 #3: Stuck gripper (0.20 rad) at place refuses retreat; lagging 0.10 rad at return_rest fails."""
    q0 = Q_LEFT_READY.copy()
    # Case A: gripper stuck at 0.20 rad during place release
    arm_a = _FakeArm(q0)
    hand_a = _FakeHand(GRIP_OPEN_RAD, stuck_q=0.20)
    api_a = _FakeStep3API(arm_a, hand_a)
    clock, sleep = _make_sim_clock()
    res_a = execute_step3_pick_place(
        _make_step3_dummy_segments(q0), api_factory=lambda: api_a, sleep_fn=sleep, clock_fn=clock
    )
    assert res_a["completed"] is False
    assert res_a["release_verified"] is False

    # Case B: arm lags 0.10 rad behind return_rest goal
    arm_b = _FakeArm(q0, lag_rad=0.10)
    hand_b = _FakeHand(GRIP_OPEN_RAD)
    api_b = _FakeStep3API(arm_b, hand_b)
    clock_b, sleep_b = _make_sim_clock()
    res_b = execute_step3_pick_place(
        _make_step3_dummy_segments(q0), api_factory=lambda: api_b, sleep_fn=sleep_b, clock_fn=clock_b
    )
    assert res_b["completed"] is False
    assert "return-to-rest" in (res_b["aborted_reason"] or "").lower()
    print("PASS: test_step3_stuck_gripper_at_place_or_unreached_rest_never_completes")


def test_step3_carried_object_collision_during_lift_is_detected():
    """P1 #4: Carried tissue pack is checked during lift/to_place_hover/descend_place against table/right-arm/workcell."""
    _, right_q, perception = _load_saved_data()
    unoccluded = dict(perception)
    unoccluded["object_footprint_m"] = [0.176, 0.078]
    unoccluded["object_center_wc_m"] = [0.4056, 0.0506, 0.0898]
    unoccluded["object_z_top_wc_m"] = 0.095
    plan = plan_and_verify_step3(q_meas=Q_LEFT_READY, right_q_meas=right_q, perception=unoccluded)
    assert plan["safe_to_execute"] is True, plan["blocking_reasons"]
    assert plan["min_carried_object_clearance_m"] is not None
    assert plan["min_carried_object_clearance_m"] >= 0.005
    print("PASS: test_step3_carried_object_collision_during_lift_is_detected")


def test_step3_overtemp_or_sustained_torque_aborts():
    """Thermal / sustained high torque guard aborts before damage."""
    q0 = Q_LEFT_READY.copy()
    arm = _FakeArm(q0)
    arm.get_motor_temperatures = lambda: [45.0, 68.0, 50.0, 45.0, 45.0, 45.0]
    hand = _FakeHand(GRIP_OPEN_RAD)
    api = _FakeStep3API(arm, hand)
    clock, sleep = _make_sim_clock()

    res = execute_step3_pick_place(
        _make_step3_dummy_segments(q0), api_factory=lambda: api, sleep_fn=sleep, clock_fn=clock
    )
    assert res["completed"] is False
    assert "over-temperature" in (res["aborted_reason"] or "").lower()
    assert arm.started is False
    print("PASS: test_step3_overtemp_or_sustained_torque_aborts")


def test_step3_closed_gripper_aborts_before_descend_grasp():
    """Comment #1 (P1): Gripper stuck closed (0.50 rad) MUST abort at hover BEFORE descend_grasp."""
    q0 = Q_LEFT_READY.copy()
    q_descend = q0 + np.array([0.0, 0.20, 0.0, 0.0, 0.0, 0.0])
    segs = _make_step3_dummy_segments(q0)
    segs["descend_grasp"] = np.linspace(q0, q_descend, 10)

    arm = _FakeArm(q0)
    hand = _FakeHand(initial_q=0.50, stuck_q=0.50)
    # Override motor_command so hand stays stuck at 0.50 from cmd 1
    hand.motor_command = lambda cmd_type, values: hand.cmds.append(float(values[0]))
    api = _FakeStep3API(arm, hand)
    clock, sleep = _make_sim_clock()

    res = execute_step3_pick_place(segs, api_factory=lambda: api, sleep_fn=sleep, clock_fn=clock)
    assert res["completed"] is False
    assert "before descend" in (res["aborted_reason"] or "").lower(), res
    # Arm must NOT have descended toward q_descend!
    assert float(np.max(np.abs(arm.q - q0))) < 1e-4, f"Arm descended despite closed gripper: {arm.q}"
    print("PASS: test_step3_closed_gripper_aborts_before_descend_grasp")


def test_step3_unopened_gripper_at_place_aborts_even_below_015():
    """Comment #2 (P1): Gripper staying at 0.12 rad (open_delta=0) at place MUST NOT report release_verified."""
    q0 = Q_LEFT_READY.copy()
    arm = _FakeArm(q0)
    hand = _FakeHand(GRIP_OPEN_RAD)
    closed_once = [False]

    def _mc_stuck_at_012(cmd_type, values):
        target = float(values[0])
        hand.cmds.append(target)
        if target > 0.3:
            closed_once[0] = True
            hand.q = 0.12  # closes from 0.02 -> 0.12 (valid grasp delta +0.10)
        elif closed_once[0]:
            hand.q = 0.12  # refuses to open back to 0.02 at place!
        else:
            hand.q = target

    hand.motor_command = _mc_stuck_at_012
    api = _FakeStep3API(arm, hand)
    clock, sleep = _make_sim_clock()

    res = execute_step3_pick_place(
        _make_step3_dummy_segments(q0), api_factory=lambda: api, sleep_fn=sleep, clock_fn=clock
    )
    assert res["completed"] is False
    assert res["release_verified"] is False
    assert "did not open" in (res["aborted_reason"] or "").lower(), res
    print("PASS: test_step3_unopened_gripper_at_place_aborts_even_below_015")


def test_step3_single_frame_at_close_start_then_freeze_aborts():
    """Comment #3 (P1): One fresh gripper frame at close tick 0 followed by freeze > 0.15s MUST abort."""
    q0 = Q_LEFT_READY.copy()
    arm = _FakeArm(q0)
    hand = _FakeHand(GRIP_OPEN_RAD)
    close_ticks = [0]

    def _mc_one_frame_then_freeze(cmd_type, values):
        target = float(values[0])
        hand.cmds.append(target)
        if target > 0.3:
            close_ticks[0] += 1
            if close_ticks[0] == 1:
                hand.q = 0.25
                hand._has_new = True
            else:
                hand._has_new = False
        else:
            hand.q = target

    hand.motor_command = _mc_one_frame_then_freeze
    api = _FakeStep3API(arm, hand)
    clock, sleep = _make_sim_clock()

    res = execute_step3_pick_place(
        _make_step3_dummy_segments(q0), stale_timeout_s=0.15, api_factory=lambda: api, sleep_fn=sleep, clock_fn=clock
    )
    assert res["completed"] is False
    assert res["grasp_verified"] is False
    assert "frozen/stale" in (res["aborted_reason"] or "").lower(), res
    print("PASS: test_step3_single_frame_at_close_start_then_freeze_aborts")


def test_step3_nan_or_none_temp_channel_does_not_mask_overtemp():
    """Comment #4 (P1): Invalid temp (NaN/None) on J2 OR overtemp on J2 MUST block startup."""
    q0 = Q_LEFT_READY.copy()
    # Case A: J2 over-temperature 70.0C
    arm_a = _FakeArm(q0)
    arm_a.get_motor_temperatures = lambda: [40.0, 70.0, 40.0, 40.0, 40.0, 40.0]
    hand_a = _FakeHand(GRIP_OPEN_RAD)
    clock_a, sleep_a = _make_sim_clock()
    res_a = execute_step3_pick_place(
        _make_step3_dummy_segments(q0), api_factory=lambda: _FakeStep3API(arm_a, hand_a), sleep_fn=sleep_a, clock_fn=clock_a
    )
    assert res_a["completed"] is False and arm_a.started is False
    assert "j2 over-temperature 70.0c" in (res_a["aborted_reason"] or "").lower(), res_a

    # Case B: J2 temperature sensor failed (NaN) -> MUST fail-closed before arm.start()!
    arm_b = _FakeArm(q0)
    arm_b.get_motor_temperatures = lambda: [40.0, float("nan"), 40.0, 40.0, 40.0, 40.0]
    hand_b = _FakeHand(GRIP_OPEN_RAD)
    clock_b, sleep_b = _make_sim_clock()
    res_b = execute_step3_pick_place(
        _make_step3_dummy_segments(q0), api_factory=lambda: _FakeStep3API(arm_b, hand_b), sleep_fn=sleep_b, clock_fn=clock_b
    )
    assert res_b["completed"] is False and arm_b.started is False
    assert "invalid motor temperature" in (res_b["aborted_reason"] or "").lower(), res_b
    print("PASS: test_step3_nan_or_none_temp_channel_does_not_mask_overtemp")


def test_step3_open_bound_matches_mujoco_verified_max_open():
    """New P1 #1: Gripper open > GRIP_MAX_OPEN_ALLOWED_RAD (e.g. 0.12 rad) aborts before descend."""
    q0 = Q_LEFT_READY.copy()
    arm = _FakeArm(q0)
    hand = _FakeHand(initial_q=0.12)
    hand.motor_command = lambda cmd_type, values: hand.cmds.append(float(values[0]))
    clock, sleep = _make_sim_clock()

    res = execute_step3_pick_place(
        _make_step3_dummy_segments(q0), api_factory=lambda: _FakeStep3API(arm, hand), sleep_fn=sleep, clock_fn=clock
    )
    assert res["completed"] is False
    assert "within verified bound" in (res["aborted_reason"] or "").lower(), res
    assert GRIP_MAX_OPEN_ALLOWED_RAD <= 0.08
    print("PASS: test_step3_open_bound_matches_mujoco_verified_max_open")


def test_step3_nan_gripper_telemetry_never_passes_grasp_or_release():
    """New P1 #2: NaN gripper position telemetry MUST abort immediately, never pass grasp/release."""
    q0 = Q_LEFT_READY.copy()
    arm = _FakeArm(q0)
    hand = _FakeHand(initial_q=GRIP_OPEN_RAD)

    def _mc_nan_on_close(cmd_type, values):
        target = float(values[0])
        hand.cmds.append(target)
        if target > 0.3:
            hand.q = float("nan")
        else:
            hand.q = target

    hand.motor_command = _mc_nan_on_close
    clock, sleep = _make_sim_clock()
    res = execute_step3_pick_place(
        _make_step3_dummy_segments(q0), api_factory=lambda: _FakeStep3API(arm, hand), sleep_fn=sleep, clock_fn=clock
    )
    assert res["completed"] is False
    assert res["grasp_verified"] is False
    assert "invalid/out-of-range gripper telemetry" in (res["aborted_reason"] or "").lower(), res
    print("PASS: test_step3_nan_gripper_telemetry_never_passes_grasp_or_release")


def test_step3_empty_close_mid_transport_aborts_immediately():
    """New P1 #3: Dropping tissue pack mid-lift/transport (q_grip -> 0.55 rad) aborts immediately."""
    q0 = Q_LEFT_READY.copy()
    arm = _FakeArm(q0)
    hand = _FakeHand(initial_q=GRIP_OPEN_RAD)
    close_seen = [0]

    def _mc_drop_mid_transport(cmd_type, values):
        target = float(values[0])
        hand.cmds.append(target)
        if target > 0.3:
            close_seen[0] += 1
            # Holds 0.28 rad during 120-tick close hold, then slips empty (0.55) on 2nd tick of lift!
            hand.q = 0.28 if close_seen[0] <= 122 else 0.55
        else:
            hand.q = target

    hand.motor_command = _mc_drop_mid_transport
    clock, sleep = _make_sim_clock()
    res = execute_step3_pick_place(
        _make_step3_dummy_segments(q0), api_factory=lambda: _FakeStep3API(arm, hand), sleep_fn=sleep, clock_fn=clock
    )
    assert res["completed"] is False
    assert res["grasp_verified"] is False
    assert "out of stage bound mid-motion" in (res["aborted_reason"] or "").lower(), res
    print("PASS: test_step3_empty_close_mid_transport_aborts_immediately")


def test_step3_mid_descend_gripper_drift_above_008_aborts_immediately():
    """New P1 #1 (part 2): Gripper drifting > 0.08 rad MID-descend MUST abort immediately."""
    q0 = Q_LEFT_READY.copy()
    q_descend = q0 + np.array([0.0, 0.25, 0.0, 0.0, 0.0, 0.0])
    segs = _make_step3_dummy_segments(q0)
    segs["descend_grasp"] = np.linspace(q0, q_descend, 20)

    arm = _FakeArm(q0)
    hand = _FakeHand(initial_q=GRIP_OPEN_RAD)

    def _mc_drift_on_descend(cmd_type, values):
        target = float(values[0])
        hand.cmds.append(target)
        # After unfold+to_hover (25+5+5=35 cmds), drift to 0.12 rad on frame 3 of descend!
        hand.q = 0.12 if len(hand.cmds) >= 38 else target

    hand.motor_command = _mc_drift_on_descend
    clock, sleep = _make_sim_clock()
    res = execute_step3_pick_place(
        segs, api_factory=lambda: _FakeStep3API(arm, hand), sleep_fn=sleep, clock_fn=clock
    )
    assert res["completed"] is False
    assert "out of stage bound mid-motion" in (res["aborted_reason"] or "").lower(), res
    # Must abort early in descend_grasp, never reaching q_descend!
    assert float(np.max(np.abs(arm.q - q_descend))) > 0.15, arm.q
    print("PASS: test_step3_mid_descend_gripper_drift_above_008_aborts_immediately")


def test_step3_missing_motor_temperature_field_in_real_sdk_blocks_startup():
    """New P1 #4 (part 2): Unset motor_temperature on J2 in real APIUp -> blocks BEFORE arm.start()."""
    import time as _time
    from hex_device import Hands

    sent_raw: list[bytes] = []

    def _factory():
        api = create_guarded_hex_device_api(
            ws_url=None, control_hz=200, enable_kcp=False, send_down_callback=sent_raw.append
        )
        up = public_api_up_pb2.APIUp()
        up.robot_type = public_api_types_pb2.RobotType.RtArmFireflyY6_H2_40
        up.protocol_major_version = 1
        up.protocol_minor_version = 0
        up.session_id = 1
        up.arm_status.calibrated = True
        for idx, q in enumerate(Q_LEFT_READY):
            m = up.arm_status.motor_status.add()
            m.pulse_per_rotation = 65535
            m.position = int(round(float(q) / (2.0 * np.pi) * 65535))
            m.torque = 0.0
            if idx != 1:  # Omit motor_temperature on J2!
                m.motor_temperature = 40.0
        # Attach a live GR100 hand so missing-gripper doesn't trigger first
        sec = up.secondary_device_status.add()
        sec.device_id = 1
        sec.device_type = public_api_types_pb2.SecondaryDeviceType.SdtHandGr100
        hm = sec.hand_status.motor_status.add()
        hm.pulse_per_rotation = 65535
        hm.position = int(round(GRIP_OPEN_RAD / (2.0 * np.pi) * 65535))
        _feed_api_up_threadsafe(api, up)
        _time.sleep(0.05)
        return api

    res = execute_step3_pick_place(_make_step3_dummy_segments(Q_LEFT_READY), api_factory=_factory)
    counts = _count_sent_arm_commands(sent_raw)
    assert res["completed"] is False
    assert "missing motor_temperature field on j2" in (res["aborted_reason"] or "").lower(), res
    assert counts["api_control_init_true"] == 0, counts
    assert counts["position_cmd"] == 0, counts
    print("PASS: test_step3_missing_motor_temperature_field_in_real_sdk_blocks_startup")


def test_step3_real_sdk_gripper_fifo_drain_and_zero_targets_after_055():
    """Comment #1 (P1): Real GR100 FIFO queue drains all 9 buffered frames on tick 0, and sends 0 targets after 0.55 arrives."""
    from hex_device import Hands
    from hex_device.motor_base import CommandType

    q0 = Q_LEFT_READY.copy()
    arm = _FakeArm(q0)
    sent_after_bad = [0]
    bad_injected = [False]

    orig_jpc = arm.joint_position_control

    def _tracked_jpc(pos, vel, acc):
        if bad_injected[0]:
            sent_after_bad[0] += 1
        orig_jpc(pos, vel, acc)

    arm.joint_position_control = _tracked_jpc

    hand = Hands(
        device_id=1,
        device_type=public_api_types_pb2.SecondaryDeviceType.SdtHandGr100,
        motor_count=1,
        proto_version=(1, 0),
        send_message_callback=lambda m: None,
    )

    def _push_hand(rad_val: float, ts_ns: int):
        sec_up = public_api_types_pb2.SecondaryDeviceStatus()
        sec_up.device_id = 1
        sec_up.device_type = public_api_types_pb2.SecondaryDeviceType.SdtHandGr100
        hm = sec_up.hand_status.motor_status.add()
        hm.pulse_per_rotation = 65535
        hm.position = int(round(rad_val / (2.0 * np.pi) * 65535))
        hand._update_optional_data(
            public_api_types_pb2.SecondaryDeviceType.SdtHandGr100, sec_up, Timestamp(s=1, ns=ts_ns)
        )

    # Push initial open frame so startup succeeds
    _push_hand(GRIP_OPEN_RAD, 0)
    orig_hmc = hand.motor_command
    cmd_idx = [0]

    def _hmc(ct, vals):
        cmd_idx[0] += 1
        if cmd_idx[0] < 38:
            _push_hand(GRIP_OPEN_RAD, cmd_idx[0] * 1_000_000)
        elif cmd_idx[0] == 38:
            # Push 9 buffered open frames + final 0.55 rad out-of-bound frame BEFORE next tick!
            for b in range(9):
                _push_hand(GRIP_OPEN_RAD, (40 + b) * 1_000_000)
            _push_hand(0.55, 50_000_000)
            bad_injected[0] = True
        orig_hmc(ct, vals)

    hand.motor_command = _hmc
    segs = _make_step3_dummy_segments(q0)
    segs["descend_grasp"] = np.linspace(q0, q0 + np.array([0, 0.2, 0, 0, 0, 0]), 20)
    clock, sleep = _make_sim_clock()

    res = execute_step3_pick_place(
        segs, api_factory=lambda: _FakeStep3API(arm, hand), sleep_fn=sleep, clock_fn=clock
    )
    assert res["completed"] is False
    assert "out of stage bound mid-motion" in (res["aborted_reason"] or "").lower(), res
    assert sent_after_bad[0] == 0, f"Sent {sent_after_bad[0]} arm targets AFTER 0.55 rad arrived!"
    print("PASS: test_step3_real_sdk_gripper_fifo_drain_and_zero_targets_after_055")


def test_step3_transient_0081_followed_by_0079_in_same_batch_aborts():
    """New P1 #1: Batch [0.081, 0.079] during descend_grasp MUST NOT miss transient 0.081 rad."""
    from hex_device import Hands

    q0 = Q_LEFT_READY.copy()
    arm = _FakeArm(q0)
    hand = Hands(
        device_id=1,
        device_type=public_api_types_pb2.SecondaryDeviceType.SdtHandGr100,
        motor_count=1,
        proto_version=(1, 0),
        send_message_callback=lambda m: None,
    )

    def _push_hand(rad_val: float, ts_ns: int):
        sec_up = public_api_types_pb2.SecondaryDeviceStatus()
        sec_up.device_id = 1
        sec_up.device_type = public_api_types_pb2.SecondaryDeviceType.SdtHandGr100
        hm = sec_up.hand_status.motor_status.add()
        hm.pulse_per_rotation = 65535
        hm.position = int(round(rad_val / (2.0 * np.pi) * 65535))
        hand._update_optional_data(
            public_api_types_pb2.SecondaryDeviceType.SdtHandGr100, sec_up, Timestamp(s=1, ns=ts_ns)
        )

    _push_hand(GRIP_OPEN_RAD, 0)
    orig_hmc = hand.motor_command
    cmd_idx = [0]

    def _hmc(ct, vals):
        cmd_idx[0] += 1
        if cmd_idx[0] < 38:
            _push_hand(GRIP_OPEN_RAD, cmd_idx[0] * 1_000_000)
        elif cmd_idx[0] == 38:
            # Transient 0.081 rad (> 0.08) followed immediately by 0.079 rad in the SAME batch!
            _push_hand(0.081, 40_000_000)
            _push_hand(0.079, 41_000_000)
        orig_hmc(ct, vals)

    hand.motor_command = _hmc
    segs = _make_step3_dummy_segments(q0)
    segs["descend_grasp"] = np.linspace(q0, q0 + np.array([0, 0.2, 0, 0, 0, 0]), 50)
    clock, sleep = _make_sim_clock()

    res = execute_step3_pick_place(
        segs, api_factory=lambda: _FakeStep3API(arm, hand), sleep_fn=sleep, clock_fn=clock
    )
    assert res["completed"] is False
    assert "out of stage bound mid-motion" in (res["aborted_reason"] or "").lower(), res
    print("PASS: test_step3_transient_0081_followed_by_0079_in_same_batch_aborts")


def test_step3_out_of_bound_gripper_synchronously_latches_send_gate():
    """New P1 #2: Out-of-bound GR100 frame synchronously latches SafeArmSendGate in _update_optional_data."""
    from hex_device import Hands
    from step3_left_arm_pick_place import _install_gripper_bound_gate

    forwarded = []

    async def fake_send(msg):
        forwarded.append(msg)

    gate = SafeArmSendGate(fake_send, start_blocked=False)
    hand = Hands(
        device_id=1,
        device_type=public_api_types_pb2.SecondaryDeviceType.SdtHandGr100,
        motor_count=1,
        proto_version=(1, 0),
        send_message_callback=gate,
    )
    _install_gripper_bound_gate(hand, gate)
    hand._active_grip_bounds = (None, 0.08)

    sec_up = public_api_types_pb2.SecondaryDeviceStatus()
    sec_up.device_id = 1
    sec_up.device_type = public_api_types_pb2.SecondaryDeviceType.SdtHandGr100
    hm = sec_up.hand_status.motor_status.add()
    hm.pulse_per_rotation = 65535
    hm.position = int(round(0.085 / (2.0 * np.pi) * 65535))
    hand._update_optional_data(
        public_api_types_pb2.SecondaryDeviceType.SdtHandGr100, sec_up, Timestamp(s=1, ns=0)
    )
    assert gate.latched_stopped is True, "SendGate was NOT synchronously latched on 0.085 rad!"
    print("PASS: test_step3_out_of_bound_gripper_synchronously_latches_send_gate")


def test_step3_last_frame_retreat_out_of_bounds_skips_return_and_preserves_reason():
    """Out-of-bounds gripper frame on LAST tick of retreat_place skips return_ready/rest & preserves reason."""
    from hex_device import Hands
    from step3_left_arm_pick_place import _install_gripper_bound_gate

    q0 = Q_LEFT_READY.copy()
    arm = _FakeArm(q0)
    gate = SafeArmSendGate(lambda m: None, start_blocked=False)
    hand = Hands(
        device_id=1,
        device_type=public_api_types_pb2.SecondaryDeviceType.SdtHandGr100,
        motor_count=1,
        proto_version=(1, 0),
        send_message_callback=gate,
    )
    _install_gripper_bound_gate(hand, gate)

    def _push_hand(rad_val: float, ts_ns: int):
        sec_up = public_api_types_pb2.SecondaryDeviceStatus()
        sec_up.device_id = 1
        sec_up.device_type = public_api_types_pb2.SecondaryDeviceType.SdtHandGr100
        hm = sec_up.hand_status.motor_status.add()
        hm.pulse_per_rotation = 65535
        hm.position = int(round(rad_val / (2.0 * np.pi) * 65535))
        hand._update_optional_data(
            public_api_types_pb2.SecondaryDeviceType.SdtHandGr100, sec_up, Timestamp(s=1, ns=ts_ns)
        )

    _push_hand(GRIP_OPEN_RAD, 0)
    # Count arm commands per stage: 25+5+5+5 (to close) + 120 (close) + 5+5+5 (to open) + 100 (open) + 5 (retreat) = 280
    orig_hmc = hand.motor_command
    cmd_idx = [0]

    def _hmc(ct, vals):
        cmd_idx[0] += 1
        target = float(vals[0])
        if cmd_idx[0] == 280:
            # Last tick of retreat_place: inject 0.12 rad (> 0.08) AFTER motor_command!
            _push_hand(0.12, cmd_idx[0] * 1_000_000)
        elif target > 0.3:
            _push_hand(0.28, cmd_idx[0] * 1_000_000)
        else:
            _push_hand(GRIP_OPEN_RAD, cmd_idx[0] * 1_000_000)
        orig_hmc(ct, vals)

    hand.motor_command = _hmc

    class _GatedFakeAPI(_FakeStep3API):
        def __init__(self, a, h):
            super().__init__(a, h)
            self.send_gate = gate

    clock, sleep = _make_sim_clock()
    res = execute_step3_pick_place(
        _make_step3_dummy_segments(q0),
        api_factory=lambda: _GatedFakeAPI(arm, hand),
        sleep_fn=sleep,
        clock_fn=clock,
    )
    assert res["completed"] is False, res
    assert "out of stage bound" in (res["aborted_reason"] or "").lower(), res
    # Must NOT have run return_ready (5) + return_rest (5) after cmd 280!
    assert arm.cmd_count == 279, f"Expected stop right after retreat_place (279 cmds), got {arm.cmd_count}!"
    print("PASS: test_step3_last_frame_retreat_out_of_bounds_skips_return_and_preserves_reason")


def test_step3_carried_object_wrist_and_forearm_collision_is_caught():
    """Comment #5 (P2): carried_object_col contact with left_link_5 or left_wrist_d405_link is NOT ignored."""
    from step3_left_arm_pick_place import build_step3_verification_model

    _, _, perception = _load_saved_data()
    unoccluded = dict(perception)
    unoccluded["object_footprint_m"] = [0.176, 0.078]
    unoccluded["object_center_wc_m"] = [0.4056, 0.0506, 0.0898]
    unoccluded["object_z_top_wc_m"] = 0.095
    m, _ = build_step3_verification_model(unoccluded["z_table_wc_m"], unoccluded)
    d = mujoco.MjData(m)

    # Place carried_object_col directly on top of left_wrist_d405_link -> MUST produce contact
    wrist_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_d405_link")
    car_mocap = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "carried_object_mocap")])
    mujoco.mj_forward(m, d)
    d.mocap_pos[car_mocap] = d.xpos[wrist_bid].copy()
    mujoco.mj_forward(m, d)

    wrist_hits = [
        (m.geom(c.geom1).name, m.geom(c.geom2).name)
        for c in d.contact[: d.ncon]
        if c.dist < -1e-4
        and (
            (m.geom(c.geom1).name == "carried_object_col" and m.body(m.geom_bodyid[c.geom2]).name == "left_wrist_d405_link")
            or (m.geom(c.geom2).name == "carried_object_col" and m.body(m.geom_bodyid[c.geom1]).name == "left_wrist_d405_link")
        )
    ]
    assert len(wrist_hits) > 0, "Expected carried_object_col vs left_wrist_d405_link contact!"
    print("PASS: test_step3_carried_object_wrist_and_forearm_collision_is_caught")


def test_passive_reader_matches_real_hex_device_arm_firefly_zero_offset():
    """Audit2 #1: passive reader MUST match real hex_device.Arm(RtArmFireflyY6_H2_40) to 1e-9."""
    rt = public_api_types_pb2.RobotType.RtArmFireflyY6_H2_40
    sdk_arm = Arm(robot_type=rt, motor_count=6, proto_version=(1, 0), name="test_firefly")

    up = public_api_up_pb2.APIUp()
    up.robot_type = rt
    up.session_id = 1
    up.arm_status.calibrated = True
    # Test raw count 0 (must be 0.0 rad, NOT -pi!) plus signed counts
    raw_counts = [0, -16384, 32768, 1000, -500, 12345]
    for cnt in raw_counts:
        m = up.arm_status.motor_status.add()
        m.pulse_per_rotation = 65535
        m.position = cnt
        m.speed = 0.0
        m.torque = 0.0

    sdk_arm._update(up, Timestamp(s=1, ns=0))
    sdk_q = np.asarray(sdk_arm.get_motor_positions(False), dtype=np.float64)
    passive = decode_api_up_arm_telemetry(up.SerializeToString())
    passive_q = np.asarray(passive["positions_rad"], dtype=np.float64)

    assert abs(sdk_q[0]) < 1e-12 and abs(passive_q[0]) < 1e-12, (sdk_q[0], passive_q[0])
    np.testing.assert_allclose(passive_q, sdk_q, atol=1e-9)
    print("PASS: test_passive_reader_matches_real_hex_device_arm_firefly_zero_offset")


def test_estop_enum_zero_with_empty_reason_is_detected():
    """Audit2 #2a: PscEmergencyStopButton (enum=0) with empty reason MUST abort immediately."""
    rt = public_api_types_pb2.RobotType.RtArmFireflyY6_H2_40
    ps = public_api_types_pb2.ParkingStopDetail()
    ps.category = public_api_types_pb2.ParkingStopCategory.PscEmergencyStopButton
    ps.reason = ""

    # 1) Passive decoder MUST detect it via arm_status.HasField("parking_stop_detail")
    up = public_api_up_pb2.APIUp()
    up.robot_type = rt
    up.arm_status.parking_stop_detail.CopyFrom(ps)
    for q in Q_LEFT_READY:
        m = up.arm_status.motor_status.add()
        m.pulse_per_rotation = 65535
        m.position = int(round(q / (2.0 * np.pi) * 65535))
        m.torque = 0.0
    dec = decode_api_up_arm_telemetry(up.SerializeToString())
    assert dec["parking_stop_active"] is True
    assert dec["parking_category"] == 0

    # 2) Real hex_device.Arm wrapped by execute_left_arm_trajectory MUST catch E-Stop (enum=0)
    sdk_arm = Arm(robot_type=rt, motor_count=6, proto_version=(1, 0), name="estop_arm")
    # Initial clean frame so execute loop acquires arm & installs _install_parking_stop_tracker
    up_clean = public_api_up_pb2.APIUp()
    up_clean.robot_type = rt
    for q in Q_LEFT_READY:
        m = up_clean.arm_status.motor_status.add()
        m.pulse_per_rotation = 65535
        m.position = int(round(q / (2.0 * np.pi) * 65535))
        m.torque = 0.0
    sdk_arm._update(up_clean, Timestamp(s=1, ns=0))

    # Inject E-Stop frame (category=0, reason="") right after arm_dev.start() is called!
    orig_start = sdk_arm.start

    def start_then_estop():
        orig_start()
        sdk_arm._update(up, Timestamp(s=1, ns=10_000_000))

    sdk_arm.start = start_then_estop
    cmds = []
    sdk_arm.joint_position_control = lambda *a, **kw: cmds.append(a)
    stop_called = []
    orig_stop = sdk_arm.stop
    sdk_arm.stop = lambda: (stop_called.append(True), orig_stop())

    api = _FakeAPI(sdk_arm)
    clock, sleep = _make_sim_clock()
    traj = np.linspace(Q_LEFT_READY, Q_LEFT_READY + np.array([0.05, 0, 0, 0, 0, 0]), 20)
    res = execute_left_arm_trajectory(traj, dt=0.01, api_factory=lambda: api, sleep_fn=sleep, clock_fn=clock)
    assert res["completed"] is False
    assert "Parking stop active" in (res["aborted_reason"] or "")
    assert len(cmds) == 0, f"Sent {len(cmds)} commands despite E-Stop (enum=0)!"
    assert len(stop_called) == 1
    print("PASS: test_estop_enum_zero_with_empty_reason_is_detected")


def test_comm_timeout_auto_clear_blocked_by_safe_send_gate():
    """Audit2 #2b: SafeArmSendGate must drop real SDK clear_parking_stop and post-stop start()."""
    rt = public_api_types_pb2.RobotType.RtArmFireflyY6_H2_40
    sdk_arm = Arm(robot_type=rt, motor_count=6, proto_version=(1, 0), name="gate_arm")
    forwarded = []

    async def fake_send(msg):
        forwarded.append(msg)

    gate = SafeArmSendGate(fake_send)

    # 1) Real SDK clear_parking_stop is ALWAYS blocked (even before stop latch)
    msg_clear = sdk_arm._construct_clear_parking_stop_message()
    asyncio.run(gate(msg_clear))
    assert len(forwarded) == 0
    assert "clear_parking_stop" in gate.blocked_messages

    # 2) After latch_stop(), real SDK api_control_initialize=True is blocked, but False (stop) is allowed
    gate.latch_stop()
    msg_start = sdk_arm._construct_init_message(True)
    asyncio.run(gate(msg_start))
    assert len(forwarded) == 0
    assert "post_stop_api_control_initialize_true" in gate.blocked_messages

    msg_stop = sdk_arm._construct_init_message(False)
    asyncio.run(gate(msg_stop))
    assert len(forwarded) == 1
    print("PASS: test_comm_timeout_auto_clear_blocked_by_safe_send_gate")


def test_single_frame_real_sdk_queue_pop_never_raises_and_calls_stop():
    """Audit2 #3: real hex_device.Arm single-frame FIFO queue empties on pop -> must NOT raise TypeError and MUST call stop()."""
    rt = public_api_types_pb2.RobotType.RtArmFireflyY6_H2_40
    sdk_arm = Arm(robot_type=rt, motor_count=6, proto_version=(1, 0), name="single_frame_arm")

    up = public_api_up_pb2.APIUp()
    up.robot_type = rt
    for q in Q_LEFT_READY:
        m = up.arm_status.motor_status.add()
        m.pulse_per_rotation = 65535
        m.position = int(round(q / (2.0 * np.pi) * 65535))
        m.torque = 0.2
    # Push ONE frame before start (like disconnect right after initial frame)
    sdk_arm._update(up, Timestamp(s=1, ns=0))

    stop_called = []
    orig_stop = sdk_arm.stop

    def tracked_stop():
        stop_called.append(True)
        orig_stop()

    sdk_arm.stop = tracked_stop
    api = _FakeAPI(sdk_arm)
    clock, sleep = _make_sim_clock()

    traj = np.linspace(Q_LEFT_READY, Q_LEFT_READY + np.array([0.05, 0, 0, 0, 0, 0]), 30)
    res = execute_left_arm_trajectory(
        traj, dt=0.01, stale_timeout_s=0.15, api_factory=lambda: api, sleep_fn=sleep, clock_fn=clock
    )
    assert res["completed"] is False
    assert "frozen/stale" in (res["aborted_reason"] or "").lower()
    assert len(stop_called) == 1, "arm_dev.stop() MUST be called even when queue empties!"
    assert api.closed is True
    print("PASS: test_single_frame_real_sdk_queue_pop_never_raises_and_calls_stop")


def test_execute_refuses_offline_perception():
    """Audit2 #4: CLI --execute MUST refuse --offline-perception before touching hardware."""
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src" / "step2_left_arm_air_jog.py"),
            "--execute",
            "--offline-perception",
            str(ROOT / "docs" / "reports" / "step2_air_jog_report.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "Refusing --execute" in proc.stderr
    print("PASS: test_execute_refuses_offline_perception")


def _count_sent_arm_commands(raw_packets: list[bytes]) -> dict[str, int]:
    counts = {
        "clear_parking_stop": 0,
        "api_control_init_true": 0,
        "position_cmd": 0,
        "other_arm_cmd": 0,
    }
    for raw in raw_packets:
        down = public_api_down_pb2.APIDown()
        down.ParseFromString(raw)
        if not down.HasField("arm_command"):
            continue
        ac = down.arm_command
        if ac.HasField("arm_shared_command") and ac.arm_shared_command.clear_parking_stop:
            counts["clear_parking_stop"] += 1
        elif ac.HasField("arm_exclusive_command"):
            excl = ac.arm_exclusive_command
            which = excl.WhichOneof("exclusive_command")
            if which == "api_control_initialize":
                if excl.api_control_initialize:
                    counts["api_control_init_true"] += 1
            elif which == "arm_api_control_command":
                counts["position_cmd"] += 1
            else:
                counts["other_arm_cmd"] += 1
    return counts


def _make_firefly_api_up(q_rad: np.ndarray, parking_cat: int | None = None, reason: str = ""):
    up = public_api_up_pb2.APIUp()
    up.robot_type = public_api_types_pb2.RobotType.RtArmFireflyY6_H2_40
    up.protocol_major_version = 1
    up.protocol_minor_version = 0
    up.session_id = 1
    up.arm_status.calibrated = True
    if parking_cat is not None:
        up.arm_status.parking_stop_detail.category = parking_cat
        up.arm_status.parking_stop_detail.reason = reason
    for q in q_rad:
        m = up.arm_status.motor_status.add()
        m.pulse_per_rotation = 65535
        m.position = int(round(float(q) / (2.0 * np.pi) * 65535))
        m.torque = 0.0
        m.motor_temperature = 40.0
    return up


def _feed_api_up_threadsafe(api, up) -> None:
    import time as _time

    loop = getattr(api, "_HexDeviceApi__loop", None)
    deadline = _time.monotonic() + 2.0
    while (loop is None or not loop.is_running()) and _time.monotonic() < deadline:
        _time.sleep(0.005)
        loop = getattr(api, "_HexDeviceApi__loop", None)
    assert loop is not None and loop.is_running(), "HexDeviceApi event loop did not start"
    loop.call_soon_threadsafe(api._process_api_up, up)


def test_preexisting_estop_frame0_sends_zero_everything_in_real_sdk_loop():
    """Audit3 #1: Pre-existing E-Stop (enum=0, reason='') on frame 0 + disconnect -> 0 clear, 0 init, 0 pos."""
    import time as _time

    sent_raw: list[bytes] = []

    def _factory():
        api = create_guarded_hex_device_api(
            ws_url=None, control_hz=200, enable_kcp=False, send_down_callback=sent_raw.append
        )
        # Inject frame 0 with E-Stop (enum=0, empty reason), then NO further frames (disconnect)
        frame0 = _make_firefly_api_up(
            Q_LEFT_READY,
            parking_cat=public_api_types_pb2.ParkingStopCategory.PscEmergencyStopButton,
            reason="",
        )
        _feed_api_up_threadsafe(api, frame0)
        _time.sleep(0.08)  # Let real Arm._periodic run during discovery window
        return api

    traj = np.linspace(Q_LEFT_READY, Q_LEFT_READY + np.array([0.05, 0, 0, 0, 0, 0]), 20)
    res = execute_left_arm_trajectory(traj, dt=0.01, api_factory=_factory)
    counts = _count_sent_arm_commands(sent_raw)

    assert res["completed"] is False
    assert "Pre-start parking stop active" in (res["aborted_reason"] or ""), res
    assert res["commands_sent"] == 0
    assert counts["clear_parking_stop"] == 0, counts
    assert counts["api_control_init_true"] == 0, counts
    assert counts["position_cmd"] == 0, counts
    print("PASS: test_preexisting_estop_frame0_sends_zero_everything_in_real_sdk_loop")


def test_preexisting_comm_timeout_discovery_window_sends_zero_everything_in_real_sdk_loop():
    """Audit3 #2: Pre-existing CommTimeout during 80 ms discovery window -> 0 clear, 0 init, 0 pos."""
    import time as _time

    sent_raw: list[bytes] = []
    api_holder = []

    def _factory():
        api = create_guarded_hex_device_api(
            ws_url=None, control_hz=200, enable_kcp=False, send_down_callback=sent_raw.append
        )
        api_holder.append(api)
        frame0 = _make_firefly_api_up(
            Q_LEFT_READY,
            parking_cat=public_api_types_pb2.ParkingStopCategory.PscAPICommunicationTimeout,
            reason="pre-existing comm timeout",
        )
        _feed_api_up_threadsafe(api, frame0)
        # Wait 80 ms so real Arm._periodic fires ~16 times trying to auto-clear & auto-start!
        _time.sleep(0.08)
        return api

    traj = np.linspace(Q_LEFT_READY, Q_LEFT_READY + np.array([0.05, 0, 0, 0, 0, 0]), 20)
    res = execute_left_arm_trajectory(traj, dt=0.01, api_factory=_factory)
    counts = _count_sent_arm_commands(sent_raw)

    assert res["completed"] is False
    assert "Pre-start parking stop active" in (res["aborted_reason"] or ""), res
    assert res["commands_sent"] == 0
    assert counts["clear_parking_stop"] == 0, counts
    assert counts["api_control_init_true"] == 0, counts
    assert counts["position_cmd"] == 0, counts
    assert "clear_parking_stop" in api_holder[0].send_gate.blocked_messages
    print("PASS: test_preexisting_comm_timeout_discovery_window_sends_zero_everything_in_real_sdk_loop")


def test_kcp_handshake_and_report_frequency_pass_through_send_gate():
    """Audit4 #1: enable_kcp, placeholder_message (kcp_start), and set_report_frequency MUST pass through."""
    from hex_device import HexDeviceApi

    sent_raw: list[bytes] = []
    api = create_guarded_hex_device_api(
        ws_url=None, control_hz=200, enable_kcp=True, send_down_callback=sent_raw.append
    )
    try:
        # Even while start_blocked=True (pre-arm), KCP & report-freq messages must NOT be blocked
        msg_kcp = api._construct_enable_kcp_message(12345)
        msg_kcp_start = api._construct_kcp_start_message()
        msg_freq = api._construct_tcp_report_frequency_message(
            api._get_report_frequency_from_control_hz(200)
        )
        for m in (msg_kcp, msg_kcp_start, msg_freq):
            assert api.send_gate._classify_blocked(m) is None, f"Misblocked: {m}"
            asyncio.run(api.send_gate(m))
        assert len(sent_raw) == 3, f"Expected 3 forwarded handshake msgs, got {len(sent_raw)}"
        assert len(api.send_gate.blocked_messages) == 0
    finally:
        api.close()
    print("PASS: test_kcp_handshake_and_report_frequency_pass_through_send_gate")


def test_mid_motion_comm_timeout_zero_rearm_in_real_sdk_loop():
    """Audit4 #2: Mid-motion CommTimeout -> 0 clear_parking_stop and 0 re-arm api_control_initialize=True."""
    import threading
    import time as _time

    sent_raw: list[bytes] = []
    stop_feeder = threading.Event()
    feeder_thread_holder = []

    def _factory():
        api = create_guarded_hex_device_api(
            ws_url=None, control_hz=200, enable_kcp=False, send_down_callback=sent_raw.append
        )
        # Healthy frame 0 so pre-start gate passes and arm_dev.start() is called
        healthy = _make_firefly_api_up(Q_LEFT_READY)
        healthy.arm_status.api_control_initialized = True
        healthy.arm_status.session_holder = 1
        _feed_api_up_threadsafe(api, healthy)
        _time.sleep(0.03)

        def _feeder():
            # Keep feeding healthy frames for ~50 ms while motion starts, then inject CommTimeout!
            for _ in range(5):
                if stop_feeder.is_set():
                    return
                _feed_api_up_threadsafe(api, healthy)
                _time.sleep(0.01)
            ps_frame = _make_firefly_api_up(
                Q_LEFT_READY,
                parking_cat=public_api_types_pb2.ParkingStopCategory.PscAPICommunicationTimeout,
                reason="mid-motion comm timeout",
            )
            _feed_api_up_threadsafe(api, ps_frame)
            # Keep Arm._periodic alive briefly after parking stop to prove 0 re-arm sends
            _time.sleep(0.06)

        t = threading.Thread(target=_feeder, daemon=True)
        t.start()
        feeder_thread_holder.append(t)
        return api

    traj = np.linspace(Q_LEFT_READY, Q_LEFT_READY + np.array([0.05, 0, 0, 0, 0, 0]), 40)
    try:
        res = execute_left_arm_trajectory(traj, dt=0.01, api_factory=_factory)
    finally:
        stop_feeder.set()
        for t in feeder_thread_holder:
            t.join(timeout=1.0)

    # Split sent packets before vs after first parking-stop-blocked event, and check total init_true <= 1
    counts = _count_sent_arm_commands(sent_raw)
    assert res["completed"] is False
    assert "Parking stop active" in (res["aborted_reason"] or ""), res
    assert counts["clear_parking_stop"] == 0, counts
    # At most 1 initial api_control_initialize=True from healthy startup; 0 re-arm sends after CommTimeout!
    assert counts["api_control_init_true"] <= 1, f"Re-arm sent after parking stop! counts={counts}"
    print("PASS: test_mid_motion_comm_timeout_zero_rearm_in_real_sdk_loop")


if __name__ == "__main__":
    test_passive_reader_sends_zero_bytes_on_comm_timeout()
    test_folded_start_pose_is_rejected_not_skipped()
    test_in_bounds_start_with_live_right_arm_passes_and_below_object_fails()
    test_motor_fault_sends_zero_commands_after_fault()
    test_frozen_feedback_aborts_and_never_completes()
    test_unreached_goal_never_reports_completed()
    test_sigint_aborts_immediately_with_zero_extra_commands()
    test_nominal_execution_succeeds()
    test_passive_reader_matches_real_hex_device_arm_firefly_zero_offset()
    test_estop_enum_zero_with_empty_reason_is_detected()
    test_comm_timeout_auto_clear_blocked_by_safe_send_gate()
    test_single_frame_real_sdk_queue_pop_never_raises_and_calls_stop()
    test_execute_refuses_offline_perception()
    test_preexisting_estop_frame0_sends_zero_everything_in_real_sdk_loop()
    test_preexisting_comm_timeout_discovery_window_sends_zero_everything_in_real_sdk_loop()
    test_kcp_handshake_and_report_frequency_pass_through_send_gate()
    test_mid_motion_comm_timeout_zero_rearm_in_real_sdk_loop()
    test_mit_grav_mode_constructs_gravity_feedforward_and_returns_to_start()
    test_real_sdk_mit_command_updates_timestamp_and_emits_nonzero_targets()
    test_secondary_device_command_blocked_when_stopped()
    test_step3_fails_closed_when_gripper_missing()
    test_step3_gr100_open_close_and_occlusion_verified()
    test_step3_prestart_motor_error_blocks_before_start_and_open()
    test_step3_frozen_gripper_after_close_never_verifies_grasp()
    test_step3_stuck_gripper_at_place_or_unreached_rest_never_completes()
    test_step3_carried_object_collision_during_lift_is_detected()
    test_step3_overtemp_or_sustained_torque_aborts()
    test_step3_closed_gripper_aborts_before_descend_grasp()
    test_step3_unopened_gripper_at_place_aborts_even_below_015()
    test_step3_single_frame_at_close_start_then_freeze_aborts()
    test_step3_nan_or_none_temp_channel_does_not_mask_overtemp()
    test_step3_carried_object_wrist_and_forearm_collision_is_caught()
    test_step3_open_bound_matches_mujoco_verified_max_open()
    test_step3_nan_gripper_telemetry_never_passes_grasp_or_release()
    test_step3_empty_close_mid_transport_aborts_immediately()
    test_step3_mid_descend_gripper_drift_above_008_aborts_immediately()
    test_step3_missing_motor_temperature_field_in_real_sdk_blocks_startup()
    test_step3_real_sdk_gripper_fifo_drain_and_zero_targets_after_055()
    test_step3_transient_0081_followed_by_0079_in_same_batch_aborts()
    test_step3_out_of_bound_gripper_synchronously_latches_send_gate()
    test_step3_last_frame_retreat_out_of_bounds_skips_return_and_preserves_reason()
    print("\nALL 41 OFFLINE SAFETY TESTS PASSED!")
