# 文件与目录说明

[返回项目首页](../README.md)。日常看记录时先打开 [recordings/index.html](../recordings/index.html)，不需要逐个运行 Python 文件。

## 截图中的目录

| 目录 | 内容与用途 | 日常是否需要打开 |
|---|---|---|
| `assets/` | SO101 模型、草莓和大棚网格、纹理；运行场景需要 | 通常不用 |
| `datasets/` | LeRobot/ACT 格式的数据样例；与原始仿真日志不同 | 做训练时看 |
| `recordings/` | 记录列表、默认保存的回合，以及指向其他位置的快捷入口 | **最常用** |
| `reports/` | 控制验证、在线批量评估、场景调试、训练验证、历史报告 | 查看成败原因时看 |
| `versions/` | 历史代码版本，供旧实验对照；不是日常运行入口 | 通常不用 |
| `tests/` | 自动回归测试，检查代码修改是否破坏既有行为 | 开发时用 |
| `tools/validation/` | `validate_*.py`、`verify*.py`：离线回放验收、场景和控制验证 | 开发时用 |
| `tools/diagnostics/` | `audit_*`、`compare_*`、`diagnose_*`、种子筛选、性能检查 | 排错时用 |
| `docs/` | 本说明、历史技术说明、文件迁移清单 | 不清楚文件作用时看 |
| `__pycache__/` | Python 自动生成的导入缓存，子目录也可能出现；不是采摘数据，清除后会再生成 | 不用看 |

`recordings/` 和 `reports/` 并非两份同样的数据：默认回合在前者；你命令明确指定 `--out reports/...` 时原始回合在后者。列表和快捷链接负责统一查看，不复制图片。主要占用磁盘的是记录中的图像、旧实验和数据集，代码脚本本身很小；这次没有删数据来腾空间。

## 运行和数据工具（保留在根目录）

| 文件 | 作用 |
|---|---|
| `vista.py` | **当前主要入口**：请求视觉模型、调用工具、控制采摘、保存和验收回合 |
| `run.py` | 基础仿真/录制入口；也提供在线程序使用的 Recorder |
| `evaluate_seed_batch.py` | 批量 seed 评估与结果汇总；运行前核对参数，不等于正式采集脚本 |
| `collect_pilot.py` | 较早的固定批次试采入口；不是当前温和移动场景的推荐日常命令 |
| `record_catalog.py` | 更新 `recordings/index.html` 和记录索引 |
| `recording_quality.py` | 轨迹、图片、时间同步和记录完整性质检；质检通过不等于采摘成功 |
| `replay_viewer.py` | 在 MuJoCo 窗口物理重放已有回合，不调用模型 |
| `export_replay_video.py` | 根据记录按需导出回放视频 |
| `replay.py` | 较早的离线重放/抽帧工具，非日常首选播放器 |
| `replay_source.py` | 按源码哈希选择匹配旧回合的物理版本，供回放程序调用 |
| `export_act_dataset.py` | 原始回合转换为 LeRobot/ACT 格式的样例导出工具 |
| `smoke_act_dataset.py` | 检查数据能加载并完成少量 ACT 优化步骤，非完整训练 |
| `run-*.sh`、`run.sh` | 启动便捷脚本；具体开关以文件内容为准，当前完整命令在项目首页 |

## 控制代码（由 vista.py 调用，一般不用单独运行）

| 文件 | 作用 |
|---|---|
| `centered_grasp.py` | 居中抓取、观察姿态、接近、拉取与跟踪误差保护 |
| `grasp_pose.py` | 夹爪姿态、果柄轴线及路径相关几何计算 |
| `local_targets.py` | 当前停车点的候选队列、近果优先、失败次数与移动条件 |
| `visual_proposals.py` | 从 RGB-D 生成果体候选区域；编号不是仿真果实 ID |
| `stalk_proposals.py` | 果柄拟合及图像内果柄候选 |
| `visual_maturity.py` | 果体颜色/深度证据的成熟度检查 |
| `proposal_identity.py` | 多个图像碎片是否对应同一果体的证据检查 |
| `perception_recovery.py` | 观察调整、候选复核、定位失败等有限恢复流程 |
| `unlisted_fruit_search.py` | 对模型报告的未编号红果测距，满足条件时允许观察性短移 |
| `preclose_vision.py` | 闭爪前视觉复核相关逻辑，属于可选实验功能 |
| `detachment_review.py` | 根据拉取前后视觉证据审核是否分离 |
| `target_parking.py` | 根据测得目标计算停车候选的运动学评分，属于可选停车模式 |

## 场景与资源代码

| 文件 | 作用 |
|---|---|
| `scene.py` | 组装大棚、机器人、果实等 MuJoCo 场景 |
| `env.py` | 基础仿真环境、步进、相机和机器人操作 |
| `peduncle.py` | 果柄采摘环境、连接/摘取物理机制及验收所需状态 |
| `mobile_scene.py` | 移动场景的布局、成熟度、初始位姿随机化和搜索范围 |
| `chassis.py` | 移动底盘结构 |
| `mounted_basket.py` | 车载固定筐的结构和安装位置 |
| `natural_assets.py` | 植株、果实等自然外观资源 |
| `scene_details.py` | 场景细节构造 |
| `scene.xml`、`scene.json` | 已生成的场景文件及元数据；不代表每个新回合的当前场景，回合以自身配置和源码快照为准 |
| `project_paths.py` | 整理后开发工具的路径辅助，不参与在线控制 |

在线策略只能使用图像、RGB-D、标定和机器人状态；环境中的果实真值仅能用于场景生成及离线验收，不能因为代码放在同一目录就供策略直接读取。

## 配置和文档

- `README.md`：当前日常使用首页。
- `GUIDE.md`：控制经验、失败原因和改动依据；含历史记录，按时间阅读。
- `docs/technical-history.md`：原来较长的 README 全文，保留旧技术说明。
- `vista_prompt.txt`、`peduncle_prompt.txt`、`peduncle_mobile_prompt.txt`：不同模式的基础模型提示词；实际回合提示词还会由程序追加，查看回合内 `system_prompt.txt` 最准确。
- `.env`：本地模型连接配置，包含敏感信息；`.env.example` 是配置示例。
- `requirements*.txt`：依赖说明。`.act-deps/`、`.hf-cache/` 是训练依赖或缓存目录，不是采摘轨迹。

## 一条采摘记录里面是什么

| 文件/目录 | 含义 |
|---|---|
| `report.json` | 最终验收结果、是否采摘成功、调用次数、停止原因 |
| `quality.json` | 记录完整性检查结果 |
| `recording.json` | 记录格式、相机、频率等说明 |
| `trajectory.jsonl` | 每个控制步的状态、动作、时间和阶段；一行一步 |
| `harvest_close/`、`wrist/` | 同步记录的两路相机图片 |
| `observations/` | 发给模型的观察图、对应深度及诊断裁剪等 |
| `vista-events.jsonl` | 模型请求、回答、工具执行结果与恢复过程 |
| `run-config.json` | seed、启动配置和源码哈希 |
| `system_prompt.txt` | 此回合实际使用的系统提示 |
| `source/` | 此回合源码快照，供复现旧物理和代码 |
| `terminal.png` | 终止时图像（如有） |
| `replay*.webm` / `replay*.mp4` | 按需生成的视频，并非每个回合都有 |

正式交付的一条数据应是一段通过验收的完整单果轨迹，不是单张图片，也不是只有 `report.json`。目前保留原始等待、搜索与恢复阶段；导出训练前再按明确规则处理。

## 开发脚本搬到哪里了

文件名不变，分类路径如下。历史报告和 `source/` 快照不改写；旧文档中的辅助脚本命令请按本表改路径。常用 `vista.py`、回放、导出入口仍在根目录。

### 自动测试

- [test_body_semantics_review.py](../tests/test_body_semantics_review.py)
- [test_centered_cycle.py](../tests/test_centered_cycle.py)
- [test_centered_session.py](../tests/test_centered_session.py)
- [test_contact_hold.py](../tests/test_contact_hold.py)
- [test_detachment_review.py](../tests/test_detachment_review.py)
- [test_edge_viewpoint_search.py](../tests/test_edge_viewpoint_search.py)
- [test_fixed_base.py](../tests/test_fixed_base.py)
- [test_grasp_pose.py](../tests/test_grasp_pose.py)
- [test_local_targets.py](../tests/test_local_targets.py)
- [test_mobile_layout.py](../tests/test_mobile_layout.py)
- [test_model_retry.py](../tests/test_model_retry.py)
- [test_mounted_basket.py](../tests/test_mounted_basket.py)
- [test_optimized_parking.py](../tests/test_optimized_parking.py)
- [test_parking_trial.py](../tests/test_parking_trial.py)
- [test_peduncle.py](../tests/test_peduncle.py)
- [test_perception_recovery.py](../tests/test_perception_recovery.py)
- [test_preclose_vision.py](../tests/test_preclose_vision.py)
- [test_proposal_identity.py](../tests/test_proposal_identity.py)
- [test_proposal_review.py](../tests/test_proposal_review.py)
- [test_recording.py](../tests/test_recording.py)
- [test_recovery_exhaustion.py](../tests/test_recovery_exhaustion.py)
- [test_row_search.py](../tests/test_row_search.py)
- [test_segmented_approach.py](../tests/test_segmented_approach.py)
- [test_staged_pull.py](../tests/test_staged_pull.py)
- [test_stalk_fit.py](../tests/test_stalk_fit.py)
- [test_stalk_proposals.py](../tests/test_stalk_proposals.py)
- [test_station_budget.py](../tests/test_station_budget.py)
- [test_target_parking.py](../tests/test_target_parking.py)
- [test_uncertain_observation.py](../tests/test_uncertain_observation.py)
- [test_unlisted_fruit_search.py](../tests/test_unlisted_fruit_search.py)
- [test_vista.py](../tests/test_vista.py)
- [test_visual_maturity.py](../tests/test_visual_maturity.py)
- [test_visual_proposals.py](../tests/test_visual_proposals.py)

### 离线验证与场景检查

- [validate_blended_pull.py](../tools/validation/validate_blended_pull.py)
- [validate_body_context.py](../tools/validation/validate_body_context.py)
- [validate_body_semantics_review.py](../tools/validation/validate_body_semantics_review.py)
- [validate_centered_cycle.py](../tools/validation/validate_centered_cycle.py)
- [validate_centered_session.py](../tools/validation/validate_centered_session.py)
- [validate_compact_layout.py](../tools/validation/validate_compact_layout.py)
- [validate_coverage_identity.py](../tools/validation/validate_coverage_identity.py)
- [validate_delivery_stop.py](../tools/validation/validate_delivery_stop.py)
- [validate_duplicate_identity.py](../tools/validation/validate_duplicate_identity.py)
- [validate_edge_viewpoint_search.py](../tools/validation/validate_edge_viewpoint_search.py)
- [validate_maturity_layout.py](../tools/validation/validate_maturity_layout.py)
- [validate_missing_stalk_observation.py](../tools/validation/validate_missing_stalk_observation.py)
- [validate_mobile_short_search.py](../tools/validation/validate_mobile_short_search.py)
- [validate_mounted_basket.py](../tools/validation/validate_mounted_basket.py)
- [validate_perception_recovery.py](../tools/validation/validate_perception_recovery.py)
- [validate_recovery_exhaustion.py](../tools/validation/validate_recovery_exhaustion.py)
- [validate_semantic_recovery_search.py](../tools/validation/validate_semantic_recovery_search.py)
- [validate_staged_delivery.py](../tools/validation/validate_staged_delivery.py)
- [validate_staged_pull.py](../tools/validation/validate_staged_pull.py)
- [validate_stalk_proposals_offline.py](../tools/validation/validate_stalk_proposals_offline.py)
- [validate_target_parking_offline.py](../tools/validation/validate_target_parking_offline.py)
- [validate_uncertain_observation.py](../tools/validation/validate_uncertain_observation.py)
- [validate_unlisted_fruit_search.py](../tools/validation/validate_unlisted_fruit_search.py)
- [validate_visibility_review.py](../tools/validation/validate_visibility_review.py)
- [validate_visual_proposals_offline.py](../tools/validation/validate_visual_proposals_offline.py)
- [verify.py](../tools/validation/verify.py)
- [verify_compact.py](../tools/validation/verify_compact.py)
- [verify_details.py](../tools/validation/verify_details.py)
- [verify_diversity.py](../tools/validation/verify_diversity.py)
- [verify_ground_basket.py](../tools/validation/verify_ground_basket.py)
- [verify_interactive.py](../tools/validation/verify_interactive.py)
- [verify_localization_offline.py](../tools/validation/verify_localization_offline.py)
- [verify_maturity_offline.py](../tools/validation/verify_maturity_offline.py)
- [verify_mobile.py](../tools/validation/verify_mobile.py)
- [verify_natural.py](../tools/validation/verify_natural.py)
- [verify_peduncle.py](../tools/validation/verify_peduncle.py)
- [verify_segmented_approach.py](../tools/validation/verify_segmented_approach.py)
- [verify_vista_delivery.py](../tools/validation/verify_vista_delivery.py)

### 诊断、审计与筛选

- [audit_localization.py](../tools/diagnostics/audit_localization.py)
- [audit_recording.py](../tools/diagnostics/audit_recording.py)
- [audit_wrist_views.py](../tools/diagnostics/audit_wrist_views.py)
- [benchmark.py](../tools/diagnostics/benchmark.py)
- [compare_approach_offline.py](../tools/diagnostics/compare_approach_offline.py)
- [compare_parking_offline.py](../tools/diagnostics/compare_parking_offline.py)
- [diagnose_approach_path.py](../tools/diagnostics/diagnose_approach_path.py)
- [replay_approach_contacts.py](../tools/diagnostics/replay_approach_contacts.py)
- [screen_one_move_seeds.py](../tools/diagnostics/screen_one_move_seeds.py)
