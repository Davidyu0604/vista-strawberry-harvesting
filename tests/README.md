# 自动测试

33个测试文件集中于此。推荐从LeRobot目录串行发现并运行全部测试（无显示器时用EGL）：

```bash
cd /home/yu/LeRobot
MUJOCO_GL=egl PYTHONPATH= .venv-vista/bin/python -m unittest discover -s strawberry-sim/tests
```

运行单组也可以直接指定文件：

```bash
.venv-vista/bin/python strawberry-sim/tests/test_unlisted_fruit_search.py
```

部分测试创建完整场景、写临时图片或使用离线模型fixture；不要与另一个仿真构建进程并行执行。测试不调用付费模型服务。旧文档的根目录test_*.py路径已迁入本目录。
