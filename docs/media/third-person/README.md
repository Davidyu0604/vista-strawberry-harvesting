# 大棚全景与第三视角

[首页](../../../README.md) · [双路相机原始记录](../README.md)

| 文件 | 内容 |
|---|---|
| `greenhouse-overview.png` | Seed 60 初始场景全景，展示棚架、种植区与行间通道 |
| `robot-third-person.png` | 初始位姿近景，展示底盘、SO101 机械臂和车载筐 |
| `seed-060-third-person.mp4` | 新增第三视角的物理回放，约 32.8 秒 |
| `seed-060-third-person.gif` | 首页自动播放副本 |
| `third-person-bend_pull_hold.png` | 拉摘保持阶段的第三视角 |
| `third-person-delivery_release.png` | 投筐释放阶段的第三视角 |
| `third-person-replay.json` | 源文件哈希、回放偏差和独立验收结果 |

## 生成方法

原始采集只保存近景和腕部相机。第三视角是在核对物理源码哈希后，重新执行原始关节和底盘指令得到的展示画面，不是当时录制的第三路视频。所有物理步都执行，不通过设置状态直接跳到目标位置。

外部相机只跟随底盘沿行位置，不追踪果实真值。逐行检查关节、底盘与 TCP 是否和原轨迹一致，并在结束时核查单果摘下、入筐和返回。原始轨迹中的目标果位置字段为空，因此不宣称进行了逐帧果实位置对照；拉摘事件及最终入筐结果另行核验。

选帧沿用双路相机预览的已验证等待压缩索引，以 2 倍速播放。初始静态图开启完整展示质量，视频使用快速渲染；只改变显示质量与相机，不改变物理场景、控制命令或策略观测。

生成过程使用临时源码与资产目录、CPU Mesa 软件渲染，不调用模型 API，不写入训练集和检查点。脚本见 [`render_third_person.py`](../../../tools/presentation/render_third_person.py)。

具备原始记录时，在仓库根目录执行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/presentation/render_third_person.py \
  reports/02_批量评估/fresh50-v1/bilateral-seed-60-20261113-095816/batch-1/seed-60 \
  --out docs/media/third-person --video
```

脚本需要仿真依赖、`imageio-ffmpeg` 及 Mesa EGL，默认采用软件渲染。已存在 MP4 时拒绝覆盖；原始记录未随公开仓库分发。
