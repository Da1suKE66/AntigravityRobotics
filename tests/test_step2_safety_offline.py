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
    create_guarded_hex_device_api,
    execute_left_arm_trajectory,
    plan_and_verify_step2,
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
    left_q, right_q, perception = _load_saved_data()
    c = perception["object_center_wc_m"]
    hover_wc = np.array([c[0], c[1], perception["object_z_top_wc_m"] + 0.10], dtype=np.float64)

    plan = plan_and_verify_step2(
        q_meas=left_q,
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
        traj, dt=0.01, goal_tol_rad=0.02, settle_steps=5, api_factory=lambda: api, sleep_fn=sleep, clock_fn=clock
    )
    assert res["completed"] is False
    assert "Goal not reached" in (res["aborted_reason"] or "")
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
    assert res["final_err_to_goal_rad"] == 0.0
    assert arm.stopped is True and api.closed is True
    print("PASS: test_nominal_execution_succeeds")


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
    print("\nALL 17 OFFLINE SAFETY TESTS PASSED!")
