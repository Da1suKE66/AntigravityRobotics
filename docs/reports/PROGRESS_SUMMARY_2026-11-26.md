# Firefly 双臂仿真与控制项目进度总结

**日期**：2026-11-26  
**当前基线提交**：`70bf66d`（`v0.2.3`，`origin/main`）  
**执行环境**：Python 3.10.12 · MuJoCo 3.17.0 · Mink 1.3.0 · DAQP 0.11.2（仿真步长 `2 ms` / `500 Hz`）

---

## 1. 当前总体状态

1. **核心仿真、IK 与双臂抓放已跑通并完成多轮加固**：
   - **默认双臂抓放**（`20/5 N·m`）：左右臂分别稳定抓起并放置 `red_block`（提升 `85.2 mm`，落位 XY 误差 **`0.19 mm`**）与 `blue_cylinder`（提升 `83.1 mm`，落位 XY 误差 **`0.15 mm`**），全程禁止接触为 **`0`**，主动夹指实测峰值 **`1.3764 rad/s`**（低于 URDF 上限 `1.5 rad/s`），峰值力矩 **`11.893 N·m`**。
   - **定向审计与核心控制测试**：`tests/audit_sim_ik.py` **9/9 PASS**，`tests/verify_sim_ik.py` **3/3 PASS**（实际挂载 `+0.5 kg` 子刚体负载保持 3 s 稳态误差 **`0.05 mm`**），额外 3 组随机种子共 **120/120** 可达双臂位姿全部求解成功。
   - **严格 25 例 IK 回归**（`tests/eval_regression_25.py --strict`）：原始冻结目标 **22/25 PASS**（`H01–H05` 5/5、`M01–M03` 3/3、`C01–C05` 5/5、`E02–E05` 4/4、`P01–P05` 5/5）；启用 `--align-right-frame` 对齐冻结数据集右臂外参差异后达到 **25/25 PASS**。
2. **力矩上限策略（保持现状）**：
   - 模型（`assets/firefly_dual.urdf`、`src/build_mjcf.py`）与控制器（`src/controller.py`）保持 **`20/5 N·m`**（J1–J4 上限 `20.0 N·m`，J5–J6 及夹爪上限 `5.0 N·m`）。
   - 冻结回归清单 `ik_regression_25/manifest.json` 中的 `rated_nm_per_arm: [9, 9, 9, 9, 3, 3]` 仅作为历史基准字段保留，不反向压低仓库默认控制器限幅。

---

## 2. 迭代演进里程碑

| 版本 / 提交 | 核心工作内容 | 关键量化变化 |
| --- | --- | --- |
| `dc7f52b` (`v0.1.0`) | 从 URDF 程序化构建 MJCF（`implicitfast` 500 Hz）、Mink+DAQP IK、无扰切换 + 重力补偿 PID 控制器 | `verify_sim_ik.py` 3/3；`H02`、`M01–M03`、`E02–E05` 通过 |
| `34f8e71` (`v0.2.0`) | 多凸包网格分解（`link_3/4/5` 与夹指 `<=60` 顶点）、工作站/相机碰撞体、`solve_pose_global`、`FireflyPlanner`、桌面抓放场景与流水线 | 消除 `C01–C03` 假自碰撞；首次打通双臂 10 阶段抓放闭环 |
| `73febbd` (`v0.2.1`) | 修复 `_grip_joint_` 与 `_joint_` 子串匹配 bug、移除从指独立电机（`nu=14`）、分臂 6-DOF 多分支全局 IK、规划器关节/速度限位与执行器前置门控、修复 0.5 kg 负载测试刚体挂载 | 定向审计 **9/9 PASS**；宽工作空间 IK `34/40 -> 40/40`；0.5 kg 负载 3 s 误差 `13.60 -> 0.05 mm` |
| `4940140` (`v0.2.2`) | `_set_gripper` 实测锁存与五次多项式限速（`<= 1.5 rad/s`）、运行期禁止接触当步中止、修复 fallback 评分初始化、`solref_limit` 硬化、`E01` 奇异阻尼 | 夹指峰值 `2.135 -> 1.371 rad/s`；碰撞中止从多跑 300 步降至 **0 步**；`E01` `451.493 -> 5.419 mm` |
| `70bf66d` (`v0.2.3`) | `ControlMode.SAFE_HOLD` 主动阻尼制动、反算抗饱和量纲修正 `(kb/ki)*sat_diff` + 粘性阻尼前馈、库默认 `damping=0.025`、P2P 动态限位软裕度 `6e-4 rad`、`--align-right-frame` | `H01` 越限 `0.000025 -> 0.000000 rad`（**22/25**）；0.6 s 碰撞后滑移 `16.890 -> 1.559 mm`；`9/3 N·m` 下 `E04` `3.980 -> 0.936 mm` |

---

## 3. 已明确根因的外部基准差异（`M04`、`M05`、`E01`）

- 仓库名义对称 `assets/firefly_dual.urdf`（SHA-256 `a534fa06...`）与冻结数据集使用的标定版 URDF（`eafe1dab...`）在右臂安装外参上存在一致的 **`43.022 mm / 0.05929 rad`** 偏移（用其余 340 条右臂记录做 FK 校验，对齐后最大位置差仅 `0.000024 mm`）。
- 因此在原始冻结目标下，`M04`、`M05` 因右臂臂展缺口 `0.362 mm / 1.400 mm` 无法在名义 URDF 上满足 `0.01 mm` 静态容差；`E01` 圆弧上段在 `1.58–2.29 s` 臂展不足导致 `5.170 mm` 误差。使用 `--align-right-frame` 后三者全部通过（`M04 = 0.0054 mm`、`M05 < 0.0001 mm`、`E01 = 0.567 mm`，**25/25 PASS**）。

---

## 4. 当前已知待改进项（来自 `SIMULATION_IK_RECHECK_70bf66d.md`）

1. **限位附近规划的 `limit_margin_rad = 6e-4` 裁剪副作用**（`src/planner.py`）：
   - 当某一关节本就位于合法限位（如左 J1 = `2.86234 rad`）且只命令保持原位或只移动另一条手臂时，双臂统一向内裁剪会引入 `0.361 mm / 0.000600 rad` 未报告偏移，超出全局 IK 默认 `0.2 mm` 容差。
   - 后续建议：仅对“规划起点/目标确实逼近限位且存在伺服超调风险的运动关节”施加局部裕度，并在实际裁剪后重新用正运动学更新返回给调用方的位姿误差。
2. **`SAFE_HOLD` 制动期间的接触与力矩统计漏记**（`src/grasp_pipeline.py`）：
   - `step_safe_hold()` 闭环制动本身有效（0.6 s 最大位移 `1.559 mm`），但函数内部未继续累加禁止接触计数和真实施加力矩峰值（导致 `8.779 N·m` vs 实际 `11.137 N·m` 的统计差值）。
   - 后续建议：在 `step_safe_hold()` 中同步执行与 `_step_control()` 一致的接触计数与峰值力矩记录。
3. **极端低力矩（`9/3 N·m`）下的完整抓放**：
   - 若将控制器强行压低到数据集额定 `9/3 N·m`，左臂释放红块阶段（`7.652 s`）因左 J2 长时间饱和会造成 `left_grip_link_1` 轻触 `place_zone_left`（`0.091 mm`）触发中止。保持当前 `20/5 N·m` 运行则全程无此问题。

---

## 5. 关键文件与验证入口

- **核心代码**：`src/build_mjcf.py` · `src/ik_solver.py` · `src/planner.py` · `src/controller.py` · `src/grasp_scene.py` · `src/grasp_pipeline.py`
- **测试套件**：`tests/verify_sim_ik.py` · `tests/audit_sim_ik.py` · `tests/test_grasp_pick_place.py` · `tests/probe_grasp_execution.py` · `tests/eval_regression_25.py` · `tests/probe_sim_ik_edges.py`
- **报告与媒体**：`docs/reports/SIMULATION_IK_GRASP_REPORT.md` · `docs/reports/SIMULATION_IK_RECHECK_70bf66d.md` · `docs/reports/grasp_verification_report.json` · `docs/reports/ik_regression_25_results.json` · `docs/media/firefly_grasp_demo.mp4`
