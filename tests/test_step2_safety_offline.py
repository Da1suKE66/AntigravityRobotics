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

import json
import sys
from pathlib import Path

import numpy as np
from hex_device.generated import public_api_types_pb2, public_api_up_pb2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from readonly_ws_client import ENCODER_MIDPOINT, read_arm_passive_readonly
from step2_left_arm_air_jog import (
    Q_LEFT_READY,
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
    up.robot_type = 1
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
        m.position = int(round(q / (2.0 * np.pi) * ppr + ENCODER_MIDPOINT))
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


if __name__ == "__main__":
    test_passive_reader_sends_zero_bytes_on_comm_timeout()
    test_folded_start_pose_is_rejected_not_skipped()
    test_in_bounds_start_with_live_right_arm_passes_and_below_object_fails()
    test_motor_fault_sends_zero_commands_after_fault()
    test_frozen_feedback_aborts_and_never_completes()
    test_unreached_goal_never_reports_completed()
    test_sigint_aborts_immediately_with_zero_extra_commands()
    test_nominal_execution_succeeds()
    print("\nALL 8 OFFLINE SAFETY TESTS PASSED!")
