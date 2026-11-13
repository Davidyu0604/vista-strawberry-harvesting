# 安装、采集与训练

[返回项目首页](../README.md) · [系统架构](architecture.md) · [实验结果](results.md)

所有命令均从仓库根目录执行。

## 安装与启动

当前开发环境为 Linux、Python 3.12，窗口运行需要可用的 OpenGL 桌面环境。以下命令从新克隆的仓库根目录执行，直接调用 Python 入口，不依赖开发者本机的父目录环境。

```bash
git clone https://github.com/Davidyu0604/vista-strawberry-harvesting.git
cd vista-strawberry-harvesting
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/requirements-vista.txt
cp .env.example .env
```

在本地 `.env` 中填写：

```dotenv
LLM_BASE_URL=https://YOUR_PROVIDER/v1
LLM_MODEL=YOUR_VISION_MODEL
LLM_API_KEY=YOUR_API_KEY
OPENAI_MAX_TOKENS=4096
```

服务须支持 Chat Completions、图像输入和工具调用。客户端还会发送 `thinking` 参数，其他服务可能需要适配该参数。`.env` 已被 Git 忽略；模板只包含占位配置。

先检查配置（不请求模型），再检查相机渲染（不请求模型）：

```bash
python src/vista.py --check-config
MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx python src/vista.py --check-render --viewer
```

执行一回合采摘，保存双路图像：

```bash
MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx python -u src/vista.py \
  --peduncle-test --viewer --record-images --allow-row-search \
  --mobile-layout --mobile-layout-level moderate \
  --centered-grasp --visual-proposals \
  --seed 12 --max-calls 32 \
  --out "reports/harvest-seed12-$(date +%Y%m%d-%H%M%S)"
```

该命令会调用你配置的模型服务。固定 seed 只固定场景，不保证模型决策或每次采摘结果相同。场景与材质会自动生成；同一工作目录下请串行运行仿真，避免多个进程同时改写生成文件。

## 批量采集

```bash
MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx python -u src/collect_fresh50.py \
  --start-seed 1 --episodes 10 \
  --out "reports/collection-$(date +%Y%m%d-%H%M%S)"
```

`--episodes` 表示尝试次数，不保证成功条数。正常完成但未采到果的回合会保留；运行中断或数据完整性异常会停止批处理。追加 `--dry-run` 可查看计划而不运行仿真或请求模型。

采摘成功应查看 `report.json` 的 `success`、`quality.json` 的 `harvest_success` 及对应验收字段；`episode_completed` 仅表示回合完成。

## 数据处理与 ACT 训练

仓库包含处理和训练代码，**不包含原始采集数据、导出数据集或训练权重**。

| 入口 | 作用 |
|---|---|
| `training_preparation.py` | 压缩稳定的模型等待与决策等待，保留原始记录和帧映射 |
| `export_act_dataset.py` | 导出 LeRobot 数据，JPEG 字节嵌入 Parquet，并核验来源 |
| `smoke_act_dataset.py` | 检查数据加载与 ACT 短程训练链路 |
| `train_act.py` | ACT 训练、验证、保存检查点与续训 |

本地实验已整理 60 条轨迹，按 seed 隔离为 50 条训练、10 条验证。等待压缩将 100211 帧缩减至 76526 帧，保留所有非等待动作阶段。这些数字描述本地数据处理结果，并非公开基准成绩；对应原始记录和报告未包含在本仓库。

ACT 训练环境与仿真环境分开。本地验证组合为 LeRobot 0.6.1、PyTorch 2.10.0 + CUDA 12.6、torchvision 0.25.0；`requirements/requirements-vista.txt` 只安装仿真与模型客户端依赖，`requirements/requirements-act-export.txt` 也不是完整训练环境锁文件。训练环境的独立安装流程尚待整理。

在已配置 LeRobot/PyTorch 的环境并准备好数据后，可直接调用：

```bash
python src/train_act.py \
  datasets/strawberry-act-60-v1 training/act-fresh60-v1 \
  --steps 20000 --batch-size 4
```

当前训练器专用于 60 回合及 50/10 划分，需要导出器生成的 `export_manifest.json` 和 `export_verification.json`；不能直接传入任意 LeRobot 数据集。默认使用预训练 ResNet18，首次运行需要下载权重或提前准备缓存。同目录续训追加 `--resume`。

原 50 条训练已完成 20,000 步；seed 46 单条诊断已测到累计 1,500 步，仍未完成夹持。下一轮训练需先解决单条末段误差，离线 loss 不代表采摘成功。现有 `run*.sh` 脚本保留开发机的父目录虚拟环境约定；新安装请使用上面的 Python 命令。


## ACT 闭环采摘

训练完成后，使用 `best.pt` 启动实际策略控制。开发机入口如下，从项目父目录执行：

```bash
MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx strawberry-sim/scripts/run-act-eval.sh \
  --checkpoint strawberry-sim/training/act-fresh60-v1/best.pt \
  --seed 61 --viewer --record-images --max-steps 3000 \
  --out "strawberry-sim/reports/04_训练验证/act-seed61-$(date +%Y%m%d-%H%M%S)"
```

该入口通过两个本地进程复用仿真与 ACT 环境：父目录 `.venv-vista` 执行 MuJoCo，`.venv-act-train` 执行策略推理。可直接用仿真 Python 运行 `evaluate_act.py`，通过 `--policy-python` 指定自己的 ACT Python。

默认执行每次预测动作块的第一个动作，与训练配置 `n_action_steps=1` 一致；严格加载权重，沿用训练时的状态归一化、图像预处理并反归一化动作。图像经 JPEG quality 92 编码，与示教存储方式一致。动作限制为机械臂关节范围、既有每控制步 0.04 rad 速率上限，以及底盘接口速度范围；限幅次数会记录。无视觉模型服务调用，也无脚本采摘轨迹接管。

`--max-steps 3000` 对应最多 150 秒仿真时间，实际耗时依赖渲染与推理速度。窗口关闭或 Ctrl+C 会保存已完成轨迹及停止原因。结果位于新输出目录的 `report.json`；`inference.log` 保存推理日志。`--record-images` 会额外保存输入图像。

进度日志中的 `base_offset` 为底盘距起点的位移，`arm_change` 为关节相对初始姿态的最大变化；`cmd_v` / `cmd_w` 为下发的底盘线速度／角速度，`joint_target_delta` 为预测关节目标与当前状态的最大差值。如果这些动作量持续接近零，检查 `trajectory.jsonl` 中的预测与执行动作，区分策略保持原位和执行限幅。离线 L1 较低仍可能出现闭环停滞，不能据此认定策略已学会采摘。

验收由仿真状态独立计算：恰好一颗成熟果摘下并稳定入筐、机械臂回到观察就绪位姿、底盘停止，持续 10 个控制步。ACT 没有教师控制器的阶段标签，所以不伪造这些标签参与验收。报告同时保留入筐、摘果数和返回等分项。seed 61 不在当前 60 条示教的训练或验证 seed 中；单回合结果不代表总体成功率。

### 单条示教拟合诊断

当策略在训练见过的场景中仍然停滞，可用 `run-act-overfit.sh` 从现有权重继续训练一条示教，检查模型能否记住这条轨迹。它只选原训练集中的唯一 seed，继承初始权重和归一化统计，在新目录写结果，不修改原来的 50/10 数据划分。该实验并非从零训练，也不测泛化能力。

```bash
strawberry-sim/scripts/run-act-overfit.sh \
  strawberry-sim/datasets/strawberry-act-60-v1 \
  strawberry-sim/training/act-overfit-seed46-v1 \
  --init-checkpoint strawberry-sim/training/act-fresh60-v1/best.pt \
  --episode-seed 46 --steps 500 --batch-size 4 --eval-every 100
```

`same_episode_chunk_l1` 和 `same_episode_first_action_l1` 来自同一条训练轨迹，不能当作验证集成绩。实验权重带有 `diagnostic` 标记，闭环测试时需显式传入 `--allow-diagnostic`。评估可用 `--n-action-steps N` 指定每次预测后连续执行的动作数（1 到 checkpoint 的 chunk_size）；默认沿用 checkpoint 配置，结果记录实际值。此参数不放大动作、不插入脚本指令。比较不同执行长度时应使用相同 seed 和权重，并分别保存结果。

继续同条诊断权重时，指定它为 `--init-checkpoint` 并添加 `--allow-diagnostic-init`，输出仍须使用新目录。这里只继承权重和归一化统计，AdamW 优化器重新创建；不是恢复完整训练状态。程序拒绝跨 seed 的诊断初始化，并保存累计单条训练步数。建议每次增加有限步数后复测闭环接近与夹持，再决定下一轮。

闭环轨迹还记录每步输入 JPEG 的 SHA-256，以及执行后的双指同柄接触果实编号。接触字段仅供离线验收，不传给策略；连续接触、拉摘事件及完整入筐验收应分别核对，不能用 TCP 接近或闭爪角度代替夹持成功。

## 测试

在仿真环境中运行，部分 ACT 测试另需训练依赖：

```bash
MUJOCO_GL=egl python -m unittest discover -s tests
```

场景生成会写入共享资产，测试与其他仿真进程需串行执行。
