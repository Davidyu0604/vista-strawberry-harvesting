# 演示媒体与来源

[返回首页](../../README.md) · [实验口径](../results.md) · [来源清单](../evidence/demo.json)

本页列出的双路相机媒体来自本地 seed 60 成功回合的原始图像记录，没有重新生成画面。新增的大棚全景与第三视角采用独立物理回放，方法见[第三视角说明](third-person/README.md)。两种演示均未使用 ACT 权重。

| 文件 | 内容 |
|---|---|
| `seed-060-preview.gif` | README 自动播放预览，640 像素宽 |
| `seed-060-preview.mp4` | 相同预览内容，960 × 464 |
| `seed-060-full.mp4` | 全部 2,112 个记录帧，20 fps，原速 105.6 秒 |
| `seed-060-poster.png` | 拉摘保持阶段的双相机关键帧 |

预览沿用已验证的数据处理索引，将 802 帧稳定等待移除，得到 1,310 帧；每 4 帧采样一帧并以 10 fps 输出，约 32.8 秒，相当于对压缩后的时间轴以 2 倍速播放。字幕显示源记录时间与阶段；时间跳跃来自等待压缩。GIF 为预览 MP4 的缩放、调色板压缩版本。

完整回放没有删帧或改变播放速度，仍包含模型等待和决策稳定阶段。原始图像内容不变，仅增加相机名称、阶段和时间信息。记录时间是仿真时间，不应当作模型服务的精确墙钟耗时。

媒体文件 SHA-256、源记录关键文件哈希、控制源码哈希和预览选帧列表见 [`demo.json`](../evidence/demo.json)。原始采集目录未随仓库分发，这些哈希用于以后对照，不意味着目前可仅从仓库重建原始数据集。

生成脚本是 [`build_portfolio_assets.py`](../../tools/presentation/build_portfolio_assets.py)，需要本地原始回合、处理 manifest、导出验收和 GPU 诊断记录。具备这些原始资料时，从仓库根目录执行：

```bash
python -m pip install imageio-ffmpeg
python tools/presentation/build_portfolio_assets.py
```

脚本只读取源数据并生成公开媒体／汇总，不启动模型服务、仿真或训练。它拒绝覆盖已有 MP4；重新生成前需明确移走旧的派生媒体。
