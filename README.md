# VISTA Strawberry Harvesting

基于 VISTA 风格视觉与工具交互的草莓采摘仿真项目，使用 MuJoCo、移动底盘和 SO101 机械臂，覆盖视觉选果、果柄定位、夹持拉摘、车载投筐，以及示教数据处理和 ACT 模仿学习训练。

本仓库是一套实验实现。视觉模型负责观察与调用工具，运动由几何计算和控制模块执行；ACT 是另一个利用采集轨迹训练的策略分支。

## 功能

- 随机化大棚布局与沿行搜索；moderate 场景在起点两侧布置成熟果。
- 双路近景与腕部图像，结合 RGB-D 候选区域进行成熟度判断与果柄定位。
- SO101 果柄居中夹持、拉摘、投放到车载筐并返回观察姿态。
- 20 Hz 状态、动作和图像记录，以及独立的采摘结果与数据完整性检查。
- 稳定等待片段压缩、LeRobot 数据导出、按 seed 隔离的训练／验证划分和 ACT 训练。

## 安装与启动

当前开发环境为 Linux、Python 3.12，窗口运行需要可用的 OpenGL 桌面环境。以下命令从新克隆的仓库根目录执行，直接调用 Python 入口，不依赖开发者本机的父目录环境。

```bash
git clone https://github.com/Davidyu0604/vista-strawberry-harvesting.git
cd vista-strawberry-harvesting
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-vista.txt
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
python vista.py --check-config
MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx python vista.py --check-render --viewer
```

执行一回合采摘，保存双路图像：

```bash
MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx python -u vista.py \
  --peduncle-test --viewer --record-images --allow-row-search \
  --mobile-layout --mobile-layout-level moderate \
  --centered-grasp --visual-proposals \
  --seed 12 --max-calls 32 \
  --out "reports/harvest-seed12-$(date +%Y%m%d-%H%M%S)"
```

该命令会调用你配置的模型服务。固定 seed 只固定场景，不保证模型决策或每次采摘结果相同。场景与材质会自动生成；同一工作目录下请串行运行仿真，避免多个进程同时改写生成文件。

## 批量采集

```bash
MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx python -u collect_fresh50.py \
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

ACT 训练环境与仿真环境分开。本地验证组合为 LeRobot 0.6.1、PyTorch 2.10.0 + CUDA 12.6、torchvision 0.25.0；`requirements-vista.txt` 只安装仿真与模型客户端依赖，`requirements-act-export.txt` 也不是完整训练环境锁文件。训练环境的独立安装流程尚待整理。

在已配置 LeRobot/PyTorch 的环境并准备好数据后，可直接调用：

```bash
python train_act.py \
  datasets/strawberry-act-60-v1 training/act-fresh60-v1 \
  --steps 20000 --batch-size 4
```

当前训练器专用于 60 回合及 50/10 划分，需要导出器生成的 `export_manifest.json` 和 `export_verification.json`；不能直接传入任意 LeRobot 数据集。默认使用预训练 ResNet18，首次运行需要下载权重或提前准备缓存。同目录续训追加 `--resume`。

GPU 短程训练与续训已验证，完整训练及策略在仿真中的闭环采摘效果仍需评估。现有 `run*.sh` 脚本保留开发机的父目录虚拟环境约定；新安装请使用上面的 Python 命令。

## 代码与开发记录

| 路径 | 内容 |
|---|---|
| `vista.py`、`*_prompt.txt` | 视觉模型交互与工具调度 |
| `scene.py`、`mobile_scene.py`、`natural_assets.py` | 场景与生成资产 |
| `peduncle.py`、`centered_grasp.py` | 果柄物理与运动控制 |
| `assets/so101/` | SO101 模型、网格、上游许可证与来源 |
| `tests/` | 自动测试 |
| `tools/validation/`、`tools/diagnostics/` | 验证、审计与诊断脚本 |

[GUIDE.md](GUIDE.md) 记录控制与采集经验；[文件说明](docs/FILES.md) 介绍代码组织；[本机开发记录](docs/local-development-notes.md) 保留发布前的实验笔记。历史文档中的绝对路径、数据链接和状态描述仅适用于当时的开发环境，部分引用文件未随仓库发布。

在仿真环境中执行测试（部分 ACT 测试另需训练依赖）：

```bash
MUJOCO_GL=egl python -m unittest discover -s tests
```

## 许可证与来源

项目原创代码和文档采用 [Apache License 2.0](LICENSE)，见 [NOTICE](NOTICE)。SO101 第三方资产保留其[原许可证](assets/so101/LICENSE)和[固定版本来源及哈希](assets/so101/SOURCE.json)。
