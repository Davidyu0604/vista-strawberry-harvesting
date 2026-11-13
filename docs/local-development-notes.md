# 本机开发记录（发布前快照）

以下保留发布前的 README，包含本机绝对路径、历史状态和未随源码发布的数据链接；首次安装请阅读[项目首页](../README.md)。文中的“当前”指记录当时，不代表最新训练进度。

# 草莓大棚采摘仿真

完整 MuJoCo 大棚中的移动底盘与 SO101 机械臂。当前目标是用视觉区分成熟果，必要时移车观察，每回合采一颗成熟草莓、送入车载固定筐并返回。当前目标已调整为50条训练、10条验证，共60条；现已完成60条验收和导出，按seed隔离为50条训练、10条验证，GPU试训通过。2026-11-12已按用户要求清空旧实验原始数据和旧训练样本，旧成功记录也已删除。

## 日常只需要看这几个入口

| 入口 | 用途 |
|---|---|
| [recordings/by-seed/](../recordings/by-seed/) | 按种子浏览：一个 seed 一个文件夹，里面按日期区分每次采摘回合 |
| [recordings/index.html](../recordings/index.html) | 双击查看采摘记录列表，进入回合、图片和已有回放 |
| [reports/README.md](../reports/README.md) | 验证结果、失败原因和实验结论 |
| [datasets/](../datasets/) | 已导出的50训练＋10验证LeRobot数据集及等待压缩副本 |
| [GUIDE.md](../GUIDE.md) | 控制经验、已遇到的问题和处理依据 |
| [docs/FILES.md](../docs/FILES.md) | 每类目录、核心程序和回合内部文件的说明 |

## 带窗口运行

在 `/home/yu/LeRobot` 下运行。每次输出到新的目录；不要点击 Reset 或 Backspace。

```bash
cd /home/yu/LeRobot
MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx PYTHONPATH= \
/home/yu/LeRobot/.venv-vista/bin/python -u strawberry-sim/vista.py \
  --peduncle-test --viewer --record-images --allow-row-search \
  --mobile-layout --mobile-layout-level moderate \
  --centered-grasp --visual-proposals \
  --seed 169 --max-calls 32 \
  --out "strawberry-sim/reports/02_批量评估/seed169-$(date +%Y%m%d-%H%M%S)"
```

使用 `--out` 时原始数据就在指定目录，结束后加入记录列表。不指定时默认保存到 `recordings/episodes/`。当前采集两路图像：近景 `harvest_close`、腕部 `wrist`。

成功看 `report.json` 的 `success`（`quality.json` 中对应 `harvest_success`），并核对单成熟果入筐及返回字段，不能只看 `episode_completed`。seed 固定场景，不固定模型决策；一次成功不代表重复稳定。最新控制改动和验证状态以 [报告首页](../reports/README.md) 为准。

## 回放与训练数据

当前可直接开始首轮ACT训练：[完整训练说明](../reports/04_训练验证/fresh60-training-v1/README.md)。RTX4060与新CUDA训练环境已验证，采用预训练ResNet18、batch4、20k步；仅训练侧参与归一化。

```bash
cd /home/yu/LeRobot
strawberry-sim/run-act-train.sh \
  strawberry-sim/datasets/strawberry-act-60-v1 \
  strawberry-sim/training/act-fresh60-v1 \
  --steps 20000 --batch-size 4
```

同输出目录续训时追加`--resume`。训练结果仍需仿真闭环验收。

2026-11-13按用户要求处理等待卡顿：[60条处理副本](../datasets/fresh60-trimmed-v1/README.md)将100211帧缩短至76526帧，去掉23685帧稳定的模型／决策等待（23.64%，约19.74分钟）。全部非等待动作及物理保持保留，原始记录不变。新训练时间轴20Hz连续编号，并保存原始帧／时间映射；图片引用原目录。后续导出器可用`--prepared`读取处理目录。全量数据检查与导出夹具验证见[处理报告](../reports/04_训练验证/wait-trimming-v1/README.md)，现已完成训练数据复核、50/10分组与完整ACT导出，见下方训练入口。

- 已有回放从记录列表打开；不自动生成每回合视频。
- 在 MuJoCo 窗口重放：`replay_viewer.py`。
- 需要视频时使用 `export_replay_video.py`，默认导出 WebM；原始数据不裁剪。
- `export_act_dataset.py` 和 `smoke_act_dataset.py` 是导出、加载与短程训练验证工具；当前仍是样例链路。

导出已改为直接将原JPEG字节嵌入LeRobot Parquet，避免先转PNG造成体积膨胀；分辨率、帧率和全部阶段保持不变。每张导出图像逐字节核查，训练加载不依赖源图片目录。[存储对比与ACT验证](../reports/04_训练验证/jpeg-export-v1/README.md)：大棚32张图片的小样本缩小约77%，实际50条大小仍需按新数据测量。

此前5回合ACT导出与12步短程训练曾通过链路检查；该样本及旧原始回合现已删除，不再可加载或回放。历史文字结论见 `docs/experimental-text-history-20261112.tar.gz`；它不含图片、轨迹或权重，不能恢复原实验数据。清理清单见 [重采前清理记录](../recordings/20261112-reset-for-50.json)。

## 按种子整理与最终交付

日常从 `recordings/by-seed/seed-001/`、`seed-002/` 等目录查看。每个 `episode-日期-时间-标识` 是一次独立运行的快捷入口，同一种子可有多条轨迹；原名、源路径、场景版本及验收标签见 `by-seed/manifest.json` 和记录首页。新回合注册后自动归类。新采集的原始文件保留实际输出路径，不额外复制图片。`recordings/episodes/` 是原始记录和注册入口；旧32条当前格式回合及53条旧版实验已全部清空。

最终训练数据沿用 [LeRobot v3 官方格式](https://huggingface.co/docs/lerobot/main/lerobot-dataset-v3)：`data/` 保存状态、动作及时间序列，`meta/` 描述字段、任务与回合边界；视频形式的图像另存 `videos/`。现有导出工具采用图像嵌入Parquet的方式，相机类型为 `image`，无需额外生成回放视频；新数据导出时重新检查格式与加载结果。

正式交付应是一个可独立加载的数据集目录，包含数据说明、字段/单位/FPS、回合与seed/版本映射、按seed隔离的训练验证划分、训练集归一化参数、来源与质检清单。回合是采样单位，seed是分组与划分依据；同一seed的多个回合不能分散到训练和验证两边。交付时选择验收合格轨迹并验证完整包，不直接打包 `by-seed/` 快捷入口或整个 `reports/`。旧 `greenhouse-act-pilot-v1` 已删除，新训练集已生成于datasets/strawberry-act-60-v1。

以上脚本仍在项目根目录，旧的常用运行路径不变。参数可通过 `--help` 查看。

## 采集与训练／验证标准（50＋10）

当前温和地图为`peduncle-v23-mobile-bilateral`：起点沿行−Y与+Y两个方向26–38cm内各至少一颗成熟果，近处仍有未熟果和遮挡。旧版只保证+Y，导致36/42往−Y搜索72cm内没有成熟果；现按用户要求改为双侧。1000个布局和36/38/42实际移车后的相机检查通过，当前新v23已有21条在线采摘及完整性通过，48、53已向−Y搜索后成功。[布局修改与验证](../reports/03_场景调试/bilateral-ripe-v1/README.md)。原命令`--mobile-layout-level moderate`直接使用新地图。

当前控制修订为`uncertainty-deferral-v1`：不确定小片段或无深度编号不再挡住其他明确成熟候选；没有明确候选时允许复核后移车换视角；深度变化超过4mm只更新该候选为待确认，不驳回整批复核。复核用完后仍刷新候选图。原30/31/32阻断状态重放误差0，分别通过复核放行P2、两次17.59cm观察移车；157项相关测试通过，30、31、32完整在线复测均成功（44次调用），候选暂缓与观察性搜索已触发；4mm分支仍仅离线覆盖。详见[修复验证](../reports/01_控制验证/uncertainty-deferral-v1/README.md)。此前禁止未完成提前finish与拉摘预检恢复保留，seed27夹爪后退替代路径已有在线成功证据。

计划见 [collection-plan-50.json](../recordings/collection-plan-50.json)（保留历史文件名）。**60条已完成首轮训练数据验收和导出，50条训练＋10条验证按seed隔离。** 累计84次在线尝试、报告调用计数1225、61条采摘及完整性候选，旧seed12行为阻断1条仍排除，接受60条。等待压缩后的76526帧及153052张JPEG已完整导出并核验，GPU试训及续训通过；正式20000步训练尚未启动。所有原始记录保留。

- 固定当前 `peduncle-v23-mobile-bilateral` 温和场景和控制代码；使用上方完整运行参数，不添加旧固定移车参数。
- 每回合一颗成熟果，稳定夹持、摘下、车载固定筐投放并返回；检查独立采摘结果与双路图像/动作完整性，不能只看 `episode_completed`。
- 多个不同seed，保留全部搜索、等待、恢复与阶段标签；同seed的回合只能属于同一数据划分。具体seed列表和划分尚待新批次计划确定，不把旧成功seed当作唯一采集范围。
- 带窗口串行运行，不点击Reset或Backspace；每回合最多32次模型调用，每批最多6次尝试/192次调用，批次之间复核。
- 不自动生成视频。新数据存到独立的 `reports/02_批量评估/fresh50-v1/` 批次目录，通过 `recordings/by-seed/` 查看。

当前专用批处理入口为 `collect_fresh50.py`：完整启用上述采集参数，带窗口串行运行，每组最多5回合/160次调用，组间保存原始数据检查摘要。默认新seed 16–25，共10次尝试；失败保留、不自动重试。原始图片全量解码、时间/动作/阶段、采摘结果及源码哈希自动核对，运行中断或完整性异常会停止后续回合。最终行为审核与正式训练批准仍单独进行。`--dry-run`只打印计划，不启动仿真或模型。

```bash
cd /home/yu/LeRobot
MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx PYTHONPATH= \
.venv-vista/bin/python -u strawberry-sim/collect_fresh50.py \
  --start-seed 16 --episodes 10 \
  --out "strawberry-sim/reports/02_批量评估/fresh50-v1/seeds-16-25-$(date +%Y%m%d-%H%M%S)"
```

两组可在同一次启动中连续执行，全任务最多320次调用，每组均低于192次上限。采集回合数不保证成功条数；正常结束但未采到果会保留并继续下一个seed，关闭窗口/Ctrl+C或运行错误会停止。只运行一个采集进程，期间不要启动其他场景构建，不点击Reset或Backspace。

`collect_pilot.py` 与 `evaluate_seed_batch.py` 属于旧实验入口，其默认参数不覆盖当前完整标准。

## 开发文件已分类

```text
strawberry-sim/
├── vista.py、其他核心模块       在线控制、场景、记录与常用入口
├── assets/                    模型、网格、材质
├── recordings/                记录列表、默认回合目录、回放快捷入口
├── reports/                   按实验类别存放验证结果及显式指定的回合
├── datasets/                  导出的训练格式数据
├── tests/                     自动测试文件
├── tools/
│   ├── validation/            38 个离线验证、场景检查脚本
│   └── diagnostics/           9 个审计、路径诊断、对照与种子筛选工具
├── docs/                      文件说明、原 README 技术历史、迁移清单
├── versions/                  历史源码版本
└── GUIDE.md                   控制与数据记录经验
```

测试（串行运行，勿与另一个场景构建进程同时执行）：

```bash
cd /home/yu/LeRobot
MUJOCO_GL=egl PYTHONPATH= .venv-vista/bin/python -m unittest discover -s strawberry-sim/tests
```

此前开发文件整理未改在线控制或场景逻辑；2026-11-12的旧实验数据清空是后续独立操作。辅助脚本旧位置到新位置见 [迁移清单](../docs/file-organization.json)。[原 README 完整技术历史](../docs/technical-history.md) 保留供追溯，其中旧命令与“最新”描述不代表当前状态。

整理验证：253项测试全部通过；核心Python源码哈希保持一致；迁移工具在其他工作目录下启动和读取历史RGB-D检查通过。详见 [迁移与验证清单](../docs/file-organization.json)。
