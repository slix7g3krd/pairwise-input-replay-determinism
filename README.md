# pairwise-input-replay-determinism

## 领域

游戏、图形与多媒体

## 主题

输入录制与确定性回放

## 说明

事件归档、固定随机与时间步夹紧。

初始状态下 `gameplay/input_replay_determinism.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
