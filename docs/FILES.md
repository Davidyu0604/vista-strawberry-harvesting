# 仓库目录

[返回首页](../README.md) · [运行指南](getting-started.md)

根目录保留首页、经验指南、许可证和配置示例；代码、启动脚本、提示词及依赖分开存放。

```text
strawberry-sim/
├── README.md                 项目首页与 seed 60 演示
├── GUIDE.md                  实验结论和历史经验
├── src/                      仿真、控制、记录、ACT 源码
├── scripts/                  本机快捷启动脚本
├── prompts/                  视觉模型提示词
├── requirements/             仿真、导出及历史依赖清单
├── assets/                   机器人模型、网格与纹理
├── docs/                     架构、使用说明、演示与迁移清单
├── tests/                    自动回归测试
├── tools/                    验证、诊断和演示制作工具
├── datasets/                 本地原始数据的训练导出副本（不提交）
├── recordings/               本地示教与记录索引（不提交）
├── reports/                  本地实验记录和验收结果（不提交）
├── training/                 本地模型权重与训练日志（不提交）
└── versions/                 本地历史源码版本（不提交）
```

## 常用入口

从仓库根目录执行；新安装使用自己配置的 Python 环境。

| 用途 | 入口 |
|---|---|
| 视觉／工具采摘 | `python src/vista.py`，完整参数见运行指南 |
| 批量采集 | `python src/collect_fresh50.py` |
| ACT 正式训练 | `python src/train_act.py` |
| ACT 单条拟合诊断 | `python src/train_act_overfit.py` |
| ACT 闭环评估 | `python src/evaluate_act.py` |
| 示教回放 | `python src/replay_viewer.py` |
| 数据压缩与导出 | `src/training_preparation.py`、`src/export_act_dataset.py` |
| 本机快捷运行 | `scripts/run-vista.sh`、`scripts/run-act-train.sh`、`scripts/run-act-eval.sh` |
| 回归测试 | `MUJOCO_GL=egl python -m unittest discover -s tests` |

快捷脚本保留开发机父目录的 `.venv-vista`、`.venv-act-train` 环境约定，自动按脚本位置寻找仓库；不是新安装环境的依赖安装器。ACT 评估通过 `--policy-python` 指定独立推理环境。

## 源码分工

| 任务 | `src/` 中的文件 |
|---|---|
| 在线视觉决策 | `vista.py`；基础提示词在 `prompts/` |
| 采摘动作与路径 | `centered_grasp.py`、`grasp_pose.py`、`target_parking.py` |
| 感知与候选恢复 | `visual_proposals.py`、`stalk_proposals.py`、`visual_maturity.py`、`local_targets.py`、`perception_recovery.py`、`proposal_identity.py`、`unlisted_fruit_search.py` |
| 仿真与物理 | `scene.py`、`env.py`、`peduncle.py`、`mobile_scene.py`、`chassis.py`、`mounted_basket.py`、`natural_assets.py`、`scene_details.py` |
| 记录与回放 | `run.py`、`recording_quality.py`、`record_catalog.py`、`replay_source.py`、`replay_viewer.py`、`export_replay_video.py` |
| ACT | `train_act.py`、`train_act_overfit.py`、`act_inference_worker.py`、`evaluate_act.py` |
| 路径与源码快照 | `project_paths.py` |

保留现有模块导入方式。直接运行 `src/文件.py` 即可；交互式导入时设置 `PYTHONPATH=src`。测试和开发工具自行加入源码路径。场景资产、`.env`、数据和报告仍相对仓库根目录定位。

## 清理与历史复现

2026-11-13 清理了约 5.13 GiB 可重新生成的 Hugging Face Arrow 转换缓存和 Python 字节码。原始 Parquet、示教、诊断权重、失败记录与演示均保留；下次完整读取训练数据时可能重新生成转换缓存。模型下载缓存和训练依赖没有删除。

本次迁移映射：[repository-layout-20261113.json](repository-layout-20261113.json)。更早的开发工具迁移：[file-organization.json](file-organization.json)。旧报告、原始回合 `source/` 快照和数据 manifest 不改写；历史命令中的根目录 Python／shell 文件分别对应现在的 `src/`／`scripts/`。回放会按哈希选择匹配的旧源码。

整理前的源码备份、删除清单和核验结果在本地 `reports/99_历史归档/repository-organization-20261113/`。没有删除原训练权重，也没有重新开始训练。
