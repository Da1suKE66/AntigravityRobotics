# Firefly 双臂仿真与 IK 复测

2026-11-23 · 当前提交 `73febbd`（v0.2.1）· 对照提交 `34f8e71`

**上次失败的 8 项定向检查现已全部通过。** 默认抓放、实际 0.5 kg 负载和扩大范围后的静态 IK 都有实测支持。仍需优先处理的是 E01 的动态跟踪、正常抓放中的夹爪超速，以及意外接触后的停止策略。[上次检查报告](SIMULATION_IK_AUDIT_2026-11-23.md)

## 相同条件下的进展

| 项目 | 上版 | 本版 |
| --- | ---: | ---: |
| 上次的 9 项定向检查 | 1/9 | **9/9** |
| 从 home 求解 40 组已知可达、终点无碰撞的双臂目标 | 34/40 | **40/40** |
| 腕部实际增加 0.5 kg，保持 3 s 后的位置误差 | 13.60 mm | **0.05 mm**，门槛 1.5 mm |
| 默认抓放的左、右落位 XY 误差 | 0.26 / 0.87 mm | **0.26 / 0.25 mm**，全程禁止接触数为 0 |
| 两个提交都使用本版的 25 例评测脚本 | 17/25 | **21/25**，新增通过 H05、P05、E02、E04 |

上版发布时的 22/25 使用了较宽的验收门槛。表中的 17/25 来自在上版代码和模型副本中运行本版评测脚本。同一个脚本在 `--strict` 下对这两次运行都返回退出码 1。[当前评测脚本](../../tests/eval_regression_25.py) · [本版逐例结果](ik_regression_25_results.json)

本版核心控制测试 **3/3** 通过，双臂跟踪终点误差为左 0.48 mm、右 0.11 mm。抓放时红块和蓝柱分别提升 85.6 mm、83.5 mm。从上次相同的起点请求桌面以下目标和不可达目标，两次均在仿真时间 **0 s** 被拒绝，关节位移为 0。[核心测试](../../tests/verify_sim_ik.py) · [抓放记录](grasp_verification_report.json)

另换随机种子 **42、73、2026**，分别按上次的方法从关节行程中央 90% 取样，每组保留 40 个无碰撞终点。120 个新目标均从 home 求解成功，独立 FK 位置误差 ≤0.2 mm、姿态误差 ≤0.002 rad。这里检查静态终点，目标之间的运动路径另行评估。[取样与独立复核方法](../../tests/audit_sim_ik.py#L34)

## 原始目标仍失败的四例

| 用例 | 上版，同一评测脚本 | 本版 | 判据 |
| --- | ---: | ---: | --- |
| H01 左腕边界姿态 | 关节越限 0.000263 rad | **0.000324 rad**，尾段位置误差已降到 0.037 mm | 全程关节越限 ≤0.000010 rad |
| M04 右臂静态位姿 | 14.361 mm | **14.361 mm / 0.03471 rad** | 数值 IK ≤0.01 mm / 0.0001 rad |
| M05 右臂静态位姿 | 12.068 mm | **7.362 mm / 0.02801 rad** | 同上 |
| E01 向上画圆 | 动态峰值 369.705 mm | **451.493 mm** | 运动峰值 <1 mm |

E01 上次按当时的脚本测到 371.211 mm，用本版评测器重跑旧代码为 369.705 mm。本版在 **2.11 s** 出现右臂实测峰值。同一时刻，IK 参考关节角对应的 TCP 离目标仅 **3.894 mm**。整个回放中手臂实测关节速度最高达规格的 **1.203 倍**，4 s 保持段末三个控制采样点的最大位置误差仍有 **14.20 mm**。E02–E05 的运动峰值依次为 **0.846、0.552、0.993、0.487 mm**，额外检查其保持段均小于 0.03 mm。[跟踪回放与判据](../../tests/eval_regression_25.py#L191)

**右臂坐标有可测的系统性差异。** 冻结用例 E01 的起始右 TCP 与当前模型 FK 相差 43.022 mm、0.05929 rad。在临时副本中，用这个首帧估计刚体变换并对齐右臂目标，E01 峰值降到 **0.856 mm**，M05 位置误差小于 0.0001 mm。M04 对齐后按默认求解容差得到 0.0350 mm，再把求解器容差设为该用例要求的 0.01 mm / 0.0001 rad，实测得到 **0.00537 mm / 0.0000117 rad**。这三组是诊断对照，21/25 采用冻结文件中的原始目标。[求解器默认容差](../../src/ik_solver.py#L351)

**力矩上限会改变结论。** 仓库仿真中 J1–J4、J5–J6 的控制上限分别为 20、5 N·m。只将手臂控制器限幅改为冻结包的额定 **9、3 N·m**，重新运行同一评测，得到 **20/25**。E04 峰值升到 **20.772 mm**，E01 升到 **621.505 mm** 且出现接触。这个对照保留了当前模型和控制器。仓库 E 组还会先用 IK 对齐首帧，正式判据只检查运动峰值和接触，所以 21/25 的适用范围是本仓库的测试协议。[冻结包额定力矩](../../ik_regression_25/manifest.json#L243) · [控制器限幅](../../src/controller.py#L55) · [E 组判据](../../tests/eval_regression_25.py#L229)

## 抓放补测发现的两个问题

1. **默认抓放让主动夹指超速。** URDF 规定上限 1.5 rad/s。闭合时指令峰值为 2.083 rad/s，左右指实测均约 2.055 rad/s。释放时指令为 2.678 rad/s，左右指实测分别达到 **2.130、2.135 rad/s**。抓放脚本自行给夹指生成时间曲线，绕开了点到点规划器的速度检查。在临时仿真中只将闭合与释放时长改为 **0.625 s**，实测最高 **1.491 rad/s**，两次抓放均成功，落位误差为 0.25 / 0.24 mm。[URDF 速度上限](../../assets/firefly_dual.urdf#L639) · [夹指指令生成](../../src/grasp_pipeline.py#L295)
2. **运行中发现禁止接触后，当前动作仍会继续。** 先让执行器接受一段 0.8 s 的无碰撞运动，再在第 100 步将场景中的蓝柱移入左夹指。0.200 s 检出接触后，执行器继续了 **300 步、0.600 s**，到第 400 步返回 `collision_free=false`，逐步累计 38 个禁止接触点。若执行要求遇碰即停，停止判断需要放在控制步内。[运行时计数](../../src/grasp_pipeline.py#L145) · [阶段结果生成](../../src/grasp_pipeline.py#L232) · [补测脚本](../../tests/probe_grasp_execution.py)

另有一项错误诊断信息需要修正。在左臂目标 `(2.0, 0.16, 0.25) m` 上，执行器正确拒绝运动，但全局 IK 报告的位置和姿态误差均为初始化值 `1e9`。对实际返回的关节角独立计算，误差为 **1405.97 mm / 2.092 rad**。这会干扰失败原因分析。[候选评分和返回值](../../src/ik_solver.py#L462)

## 复跑与验证范围

在当前提交的干净副本中测试，Python 3.10.12、MuJoCo 3.17.0、Mink 1.3.0、DAQP 0.11.2，物理步长 2 ms。由 URDF 重建的三个 MJCF 和碰撞网格与提交版逐字节一致，25 例和抓放重新生成的 JSON 也与本版留档一致。本次结果均来自仿真。

```bash
sim_python=/home/lakeside/.venvs/mujoco/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$sim_python" tests/audit_sim_ik.py
"$sim_python" tests/verify_sim_ik.py
"$sim_python" -m src.sim_runner
"$sim_python" tests/test_grasp_pick_place.py
"$sim_python" tests/eval_regression_25.py --strict
"$sim_python" tests/probe_grasp_execution.py
```

原有回归与抓放脚本会写入其所在副本的 `docs/` 目录。上面最后两条命令在本版分别因四个冻结用例失败和实测夹指超速返回 1。要复现同口径的上版结果，可把本版的 `tests/eval_regression_25.py` 复制到 `34f8e71` 的临时副本，再从该副本运行。

在仓库根目录用同一解释器运行以下对照，可以复现 9/3 N·m 下的 20/25。限幅值直接取自冻结清单，运行结果输出到终端。

```python
import json
import runpy
from pathlib import Path
from unittest.mock import patch
from src.controller import ControllerGains

ratings = json.loads(Path("ik_regression_25/manifest.json").read_text())["physics"]["rated_nm_per_arm"]
defaults = ControllerGains.default_for_model

def rated_gains(model):
    gains = defaults(model)
    for side in ("left", "right"):
        for index, torque in enumerate(ratings, start=1):
            joint = model.joint(f"{side}_joint_{index}")
            gains.tau_limit[joint.dofadr[0]] = torque
    return gains

with patch.object(ControllerGains, "default_for_model", staticmethod(rated_gains)):
    runpy.run_path("tests/eval_regression_25.py")["eval_all"](save_report=False)
```
