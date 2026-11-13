# 开发工具

- [validation/](validation/)：离线控制验收、历史回合重放检查、场景验证。
- [diagnostics/](diagnostics/)：记录审计、路径/接触分析、停车对照、种子筛选、性能检查。
- 完整文件列表和作用见 [文件说明](../docs/FILES.md)。

这些工具不是日常采摘入口，部分脚本针对特定历史版本或回合。先读脚本参数和说明；不要一次运行所有工具。场景构建和物理验证串行执行，避免共享资源写入冲突。需要视频的诊断脚本可能生成视频，日常不自动调用。

例如查看新恢复分支离线验收工具的参数：

```bash
/home/yu/LeRobot/.venv-vista/bin/python /home/yu/LeRobot/strawberry-sim/tools/validation/validate_unlisted_fruit_search.py --help
```
