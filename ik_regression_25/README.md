# Firefly Y640 双臂 IK 回归测试集 v1

25 个用例分为 5 组，每组 5 个。所有输入取自已经做过的仿真实验。封包时使用同一标定场景重新运行，12 个达到各自验收条件，13 个复现已知问题。这个比例反映本测试集的定向选样。可在改动求解器、碰撞模型或伺服后，用同一批起点和目标复测。[机器可读索引](manifest.json) · [封包时实测结果](baseline.json)

## 用例

| 编号 | 原始样本或动作 | 封包时的结果 |
| --- | --- | --- |
| [H01](cases/H01.json) | `witness_left_wrist_family_025`，左腕正极点 | 旧版漏解，当前从 home 用 4 秒无接触到位 |
| [H02](cases/H02.json) | `left_coupled_singularities_+1`，12 帧连续目标 | 旧版在第 6 步 J1 跳转约 180°。当前该步所有关节中最大变化 0.00000693°，整段最大 0.516° |
| [H03](cases/H03.json) | `same_pose_alternate_elbow`，同一完整位姿的两个肘形 | 原数值首解触发左 3、5 连杆凸包接触，当前选出另一肘形并通过回放 |
| [H04](cases/H04.json) | `load_left_j2_8.99`，左 J2 静态负载约 8.99 N·m | 旧 4 秒动作在 5.114 秒触地，包含保持阶段。当前 9 秒运动并保持 4 秒通过 |
| [H05](cases/H05.json) | `crossing_fingers_cross`，左右夹指交叉 | 同步直达在 1.932 秒接触，当前先左后右，各运动 4 秒通过 |
| [M01](cases/M01.json) | `left_elbow_and_wrist_035` | 分支 0 个，误报已穷尽。数值解另经同一调度器验证，6 秒运动无接触，保持误差 0.002 mm |
| [M02](cases/M02.json) | `left_elbow_and_wrist_003` | 分支 0 个，误报已穷尽，默认数值解通过独立 FK |
| [M03](cases/M03.json) | `left_elbow_and_wrist_111` | 分支 0 个，返回奇异解族未穷尽，默认数值解通过独立 FK |
| [M04](cases/M04.json) | `right_elbow_and_wrist_009` | 分支 0 个，误报已穷尽，默认数值解通过独立 FK |
| [M05](cases/M05.json) | `right_elbow_and_wrist_118` | 分支 0 个，返回奇异解族未穷尽，默认数值解通过独立 FK |
| [C01](cases/C01.json) | `near_home_02`，右 3、5 连杆 | 凸包重叠 0.065 mm，原始 STL 表面间隙 8.914 mm |
| [C02](cases/C02.json) | `near_home_04`，左 3、5 连杆 | 凸包重叠 1.188 mm，原始 STL 表面间隙 8.511 mm |
| [C03](cases/C03.json) | `near_home_07`，左右各自的 3、5 连杆 | 两侧均报凸包接触，STL 间隙分别为 9.311 和 8.680 mm |
| [C04](cases/C04.json) | `medium_00`，立柱型材与右腕第 6 连杆 | 固定参考姿态下两者 STL 三角面相交，规划器拦截 |
| [C05](cases/C05.json) | `medium_01`，左夹指与右腕第 6 连杆 | 固定参考姿态下两者 STL 三角面相交，规划器拦截。文件中另有同姿态其余接触记录 |
| [E01](cases/E01.json) | 向上画直径 40 mm 的圆，4 秒 | 运动峰值 15.279 mm，1.410 秒首次接触 |
| [E02](cases/E02.json) | 向下画直径 40 mm 的圆，4 秒 | 运动峰值 3.449 mm，全程无接触 |
| [E03](cases/E03.json) | 同一向下圆，8 秒 | 运动峰值 1.547 mm，全程无接触 |
| [E04](cases/E04.json) | 50 × 30 mm 的 8 字并转腕 ±12°，4 秒 | 运动峰值 5.422 mm，全程无接触 |
| [E05](cases/E05.json) | 同一带转腕 8 字，12 秒 | 运动峰值 1.915 mm，全程无接触 |
| [P01](cases/P01.json) | `both_world_x_10mm`，两侧 TCP 沿世界 X 平移 10 mm | 通过 |
| [P02](cases/P02.json) | `tool_roll`，双侧 J6 转至约 1.1 rad | 通过 |
| [P03](cases/P03.json) | 历史普通组 `near_home_08` | 通过 |
| [P04](cases/P04.json) | 历史普通组 `near_home_11` | 通过 |
| [P05](cases/P05.json) | 历史普通组 `medium_01`，中等幅度点到点 | 通过 |

C05 和 P05 原报告恰好使用相同样本名。两次采样的种子分别是 5347921 和 20261118，目标位置、姿态及参考关节角均可在各自用例中查看。

## 复跑

在 RoboControl-IK 项目根目录使用包内 MuJoCo 解释器。求解器和原始三角网格复核通过项目的 Pinocchio / Coal 环境运行。测试数据固定在每个用例中，复跑直接读取这些文件。项目中需要已有新版仿真模型、原始网格、两个 Python 环境及 `examples` 下的验证脚本。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
firefly-sim-ik/.venv/bin/python tests/data/ik_regression_25/run.py --output /tmp/ik25-current.json
firefly-sim-ik/.venv/bin/python tests/data/ik_regression_25/run.py --case M01
firefly-sim-ik/.venv/bin/python tests/data/ik_regression_25/run.py --group tracking --strict
```

若将 ZIP 解压到其他位置，在工程根目录将脚本路径换为解压目录中的 `run.py`。也可用 `--project` 显式指定工程。用 `--list` 查看编号，`--case` 可重复。解压后可在数据集目录运行 `sha256sum -c SHA256SUMS` 核验包内文件。

默认模式会执行所有选中的案例，遇到 `regression` 或 `error` 返回退出码 1。`pass` 表示达到原验收条件，`known_issue` 表示仍有已记录的问题，`improved` 表示已达到该问题的修复条件。严格模式 `--strict` 遇到仍有已知问题也会返回 1。每次输出都保留逐案实测值，可跟踪尚未跨过验收线的变化。

## 输入、证据与判定

- 每个用例的 `input` 含起点及完整世界坐标 TCP 位姿，`reference` 含可行的几何参考角、曾返回的轨迹或待核验的固定姿态，`expected` 给出判定条件，`recorded` 保留原实验观测。求解仅使用目标和种子。来源报告、种子和文件 SHA-256 可由 `sources` 与索引中的 `source_reports` 定位。
- 世界系为 `workcell_base`，位置单位 m，位姿四元数顺序为 `[x,y,z,w]`，TCP 在两侧 `link_6` 局部 +Z 方向 170 mm。单臂角度依次为 J1–J6，双臂数组先左后右，单位 rad。H02 由给定首帧开始，后续 11 帧以上一解为种子。其他用例由仿真 home 开始。
- 几何分支用 MuJoCo 独立正解检查，位置和方向阈值分别为 0.000011 mm 和 0.00000011 rad。默认数值 IK 使用 0.01 mm 和 0.0001 rad。M01 另测同一世界目标的双臂公开规划入口和数值解轨迹。
- 点到点轨迹按求解器实际选取的运动顺序、中间点和时长独立积分，终点保持 4 秒，末 3 个控制周期的位置误差须小于 0.4 mm、姿态误差小于 0.004 rad、关节速度小于 0.01 rad/s。碰撞按 2 ms 检查。动力学沿用 J1–4 的 9 N·m 和 J5–6 的 3 N·m 额定上限，位置越限容差及速度判据见索引。
- C 组在固定关节姿态对齐同一对连杆，比较原始 STL 三角面和场景接触。C01–C03 的间隙只适用于标记的姿态与连杆对。C04–C05 也可接受经独立回放通过的其他肘形，标记参考姿态的真实相交仍须检出。
- E 组以“轨迹 × 时长”为一个用例，输入保留左右臂每 100 ms 的时间戳和 TCP 目标。连续参考采用五次时间律。XZ 圆为半径 20 mm，YZ 八字采用一次与二次正弦，左右镜像，工具绕自身 Z 轴转动。运动中每 10 ms 比较同一时刻的实际和期望 TCP，取双臂中较大的误差。原报告每 50 ms 的实际轨迹、期望轨迹和最差帧随包保存。五例均能在终点保持 4 秒。本集合用运动段峰值小于 1 mm 且全程无接触作为跟踪目标。相对已知基线的误差回退比较留 0.01 mm 和 0.01° 的数值余量。
