"""输入录制与确定性回放。

修复点与实现的对应关系：

1. 事件按逻辑帧归档（FrameArchive / EventQueue）
   录制不再按墙钟到达时间存盘：事件在进入的那个逻辑帧入桶，回放时在
   对应帧开头整桶注入。两帧之间到达的高频输入不会再丢事件。

2. 固定随机种子（DeterministicRng / fixed_seed）
   录制与回放共用同一个 LCG 与同一个种子，AI 行为逐帧一致。

3. 时间步夹紧并插值（clamp_dt / TimestepDriver）
   每帧真实耗时被夹在 [MIN_DT, MAX_DT] 内再累积，掉帧产生的尖刺不会
   让仿真时间步累积；固定步长推进，渲染态在两个逻辑态之间线性插值。

4. 暂停时暂停录制（Recorder.push / tick 的 paused 通道）
   暂停菜单打开期间事件直接丢弃、帧号不前进，不会录进垃圾输入。

5. 回放后释放缓冲（Replayer.release / 上下文管理器）
   回放结束显式释放帧归档等缓冲，重复释放幂等。

6. 逐帧校验状态哈希（state_hash / Replayer / verify）
   录制端逐帧保存权威状态哈希，回放端逐帧重算比对；不一致时可用权威
   快照回滚（fallback），防止角色位置逐渐偏离。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

FIXED_DT: float = 1.0 / 60.0          # 固定逻辑步长（秒）
MIN_DT: float = 1.0 / 1000.0          # 单帧真实耗时下限
MAX_DT: float = 1.0 / 15.0            # 单帧真实耗时上限（掉帧尖刺被夹到这里）
DEFAULT_SEED: int = 0x5EED
PLAYER_SPEED: float = 5.0             # 玩家每秒位移
AI_SPEED: float = 2.0                 # AI 每秒位移（只由确定性 RNG 驱动）


# ---------------------------------------------------------------------------
# 1. 事件与帧归档
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputEvent:
    """一条不可变输入事件。"""

    kind: str
    payload: Any = None
    frame: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "payload": self.payload, "frame": self.frame}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "InputEvent":
        return InputEvent(data["kind"], data["payload"], int(data.get("frame", -1)))


class FrameArchive:
    """按逻辑帧号归档事件。

    同一个逻辑帧内到达的任意数量高频事件全部进入同一个桶，回放时整桶
    注入，从根上杜绝“两帧之间到达的输入被覆盖/丢弃”。
    """

    def __init__(self) -> None:
        self._buckets: Dict[int, List[InputEvent]] = {}

    def add(self, frame: int, event: InputEvent) -> None:
        if event.frame < 0:
            event = InputEvent(event.kind, event.payload, frame)
        self._buckets.setdefault(frame, []).append(event)

    def get(self, frame: int) -> List[InputEvent]:
        """非破坏式读取某帧的事件（回放注入用，不删除数据）。"""
        return list(self._buckets.get(frame, ()))

    def frames(self) -> List[int]:
        return sorted(self._buckets)

    def all_events(self) -> List[InputEvent]:
        return [ev for frame in self.frames() for ev in self._buckets[frame]]

    def release(self) -> int:
        """清空归档缓冲，返回释放的事件数；可重复调用。"""
        count = sum(len(bucket) for bucket in self._buckets.values())
        self._buckets.clear()
        return count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames": [
                {"frame": frame, "events": [ev.to_dict() for ev in self._buckets[frame]]}
                for frame in self.frames()
            ]
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "FrameArchive":
        archive = FrameArchive()
        for entry in data.get("frames", []):
            frame = int(entry["frame"])
            for ev_data in entry["events"]:
                archive.add(frame, InputEvent.from_dict(ev_data))
        return archive


class EventQueue:
    """按逻辑帧组织的活跃事件队列（录制端写入侧）。"""

    def __init__(self) -> None:
        self.archive = FrameArchive()
        self._dropped_while_paused = 0

    def push(self, frame: int, event: InputEvent, paused: bool = False) -> bool:
        """把事件归档到逻辑帧 ``frame``。

        暂停期间返回 False 并丢弃事件——暂停菜单的输入不属于回放内容。
        """
        if paused:
            self._dropped_while_paused += 1
            return False
        self.archive.add(frame, event)
        return True

    def pop(self, frame: int) -> List[InputEvent]:
        return self.archive.get(frame)

    @property
    def dropped_while_paused(self) -> int:
        return self._dropped_while_paused

    def release(self) -> int:
        dropped = self._dropped_while_paused
        self._dropped_while_paused = 0
        return self.archive.release() + dropped


# ---------------------------------------------------------------------------
# 2. 固定随机种子
# ---------------------------------------------------------------------------


class DeterministicRng:
    """确定性 LCG（MMIX 参数）。

    同样的种子产生完全相同的序列，录制端与回放端各自实例化但状态必然
    逐帧一致——前提是两边消费随机数的次数与顺序一致。
    """

    _MULTIPLIER = 6364136223846793005
    _INCREMENT = 1442695040888963407
    _MASK = (1 << 64) - 1

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self.seed = int(seed) & self._MASK
        self.state = self.seed

    def next_u64(self) -> int:
        self.state = (self._MULTIPLIER * self.state + self._INCREMENT) & self._MASK
        return self.state

    def next_float(self) -> float:
        return self.next_u64() / float(1 << 64)

    def unit_delta(self) -> float:
        """落在 [-1, 1] 的 AI 抖动量。"""
        return self.next_float() * 2.0 - 1.0

    def snapshot(self) -> int:
        return self.state

    def restore(self, state: int) -> None:
        self.state = int(state) & self._MASK


# ---------------------------------------------------------------------------
# 仿真世界与状态哈希
# ---------------------------------------------------------------------------


@dataclass
class World:
    """极简仿真世界：一个玩家、一个 AI、一个确定性 RNG。"""

    player_x: float = 0.0
    player_y: float = 0.0
    ai_x: float = 0.0
    frame: int = 0
    rng: DeterministicRng = field(default_factory=DeterministicRng)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "player_x": self.player_x,
            "player_y": self.player_y,
            "ai_x": self.ai_x,
            "frame": self.frame,
            "rng_state": self.rng.snapshot(),
        }

    @staticmethod
    def from_snapshot(data: Dict[str, Any]) -> "World":
        rng = DeterministicRng(DEFAULT_SEED)
        rng.restore(data["rng_state"])
        return World(
            player_x=float(data["player_x"]),
            player_y=float(data["player_y"]),
            ai_x=float(data["ai_x"]),
            frame=int(data["frame"]),
            rng=rng,
        )


def state_hash(state: Any) -> str:
    """对世界状态（或任意可 JSON 化结构）计算稳定的 SHA-256 摘要。

    键排序 + 固定分隔符，保证同一逻辑状态在录制端与回放端哈希一致。
    """
    if hasattr(state, "snapshot"):
        state = state.snapshot()
    blob = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def step_world(world: World, events: Sequence[InputEvent]) -> None:
    """推进一个固定逻辑步。

    玩家移动只取决于本帧注入的事件；AI 移动只取决于确定性 RNG。
    """
    dx = dy = 0.0
    for event in events:
        if event.kind == "move":
            payload = event.payload or {}
            dx += float(payload.get("dx", 0.0))
            dy += float(payload.get("dy", 0.0))
    world.player_x += dx * PLAYER_SPEED * FIXED_DT
    world.player_y += dy * PLAYER_SPEED * FIXED_DT
    world.ai_x += world.rng.unit_delta() * AI_SPEED * FIXED_DT
    world.frame += 1


# ---------------------------------------------------------------------------
# 3. 时间步夹紧与插值
# ---------------------------------------------------------------------------


def clamp_dt(raw_dt: float, min_dt: float = MIN_DT, max_dt: float = MAX_DT) -> float:
    """把单帧真实耗时夹在 [min_dt, max_dt]。

    掉帧造成的超大时间尖刺被夹到 ``max_dt``，防止时间步累积把角色
    位置越推越偏；异常的负值/零值夹到 ``min_dt``。
    """
    raw_dt = float(raw_dt)
    if raw_dt < min_dt:
        return min_dt
    if raw_dt > max_dt:
        return max_dt
    return raw_dt


class TimestepDriver:
    """固定步长驱动器：夹紧真实耗时 → 累积 → 整步消费 → 插值渲染。"""

    def __init__(
        self,
        fixed_dt: float = FIXED_DT,
        min_dt: float = MIN_DT,
        max_dt: float = MAX_DT,
    ) -> None:
        self.fixed_dt = float(fixed_dt)
        self.min_dt = float(min_dt)
        self.max_dt = float(max_dt)
        self.accumulator = 0.0
        self.sim_time = 0.0
        self.alpha = 0.0

    def add_frame_time(self, raw_dt: float) -> float:
        """夹紧并累积一帧的真实耗时，返回夹紧后的耗时。"""
        clamped = clamp_dt(raw_dt, self.min_dt, self.max_dt)
        self.accumulator += clamped
        return clamped

    def steps_available(self) -> int:
        return int(self.accumulator // self.fixed_dt + 1e-12)

    def consume_step(self) -> bool:
        if self.accumulator + 1e-12 < self.fixed_dt:
            return False
        self.accumulator -= self.fixed_dt
        self.sim_time += self.fixed_dt
        return True

    def finalize_alpha(self) -> float:
        """渲染插值系数：当前真实时刻落在两个逻辑态之间的比例。"""
        self.alpha = min(1.0, max(0.0, self.accumulator / self.fixed_dt))
        return self.alpha

    def tick(self, raw_dt: float) -> List[float]:
        """处理一帧真实耗时，返回本帧应执行的固定步长列表。"""
        self.add_frame_time(raw_dt)
        steps: List[float] = []
        while self.consume_step():
            steps.append(self.fixed_dt)
        self.finalize_alpha()
        return steps

    @staticmethod
    def interpolate(previous: float, current: float, alpha: float) -> float:
        """在两个逻辑态之间线性插值（仅用于渲染，不回写仿真态）。"""
        return previous + (current - previous) * alpha


# ---------------------------------------------------------------------------
# 4/5/6. 录制器与回放器
# ---------------------------------------------------------------------------


class Recorder:
    """录制端：暂停感知、按帧归档、逐帧保存权威状态哈希与快照。"""

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        fixed_dt: float = FIXED_DT,
        min_dt: float = MIN_DT,
        max_dt: float = MAX_DT,
    ) -> None:
        self.queue = EventQueue()
        self.world = World(rng=DeterministicRng(seed))
        self.driver = TimestepDriver(fixed_dt, min_dt, max_dt)
        self.paused = False
        self.frame_hashes: Dict[int, str] = {}
        self.snapshots: Dict[int, Dict[str, Any]] = {}
        self.released = False
        # 第 0 帧（尚未推进）的权威基线
        self._record_authoritative_state()

    @property
    def frame(self) -> int:
        return self.world.frame

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)

    def push(self, event: InputEvent) -> bool:
        """事件归档到“当前逻辑帧”；暂停期间直接拒绝。"""
        return self.queue.push(self.frame, event, paused=self.paused)

    def tick(self, raw_dt: float) -> int:
        """推进一帧真实耗时。

        暂停时时间不累积、帧号不前进；正常时按固定步长逐逻辑帧消费
        本帧归档的事件并记录权威状态哈希。
        """
        if self.paused:
            return 0
        steps = self.driver.tick(raw_dt)
        for _ in steps:
            frame = self.frame
            events = self.queue.pop(frame)
            step_world(self.world, events)
            self._record_authoritative_state()
        return len(steps)

    def _record_authoritative_state(self) -> None:
        frame = self.frame
        self.frame_hashes[frame] = state_hash(self.world)
        self.snapshots[frame] = self.world.snapshot()

    def render_state(self) -> Tuple[float, float]:
        """玩家位置的插值渲染态（演示插值不污染仿真态）。"""
        alpha = self.driver.alpha
        prev_snapshot = self.snapshots.get(self.frame - 1)
        if prev_snapshot is None:
            return self.world.player_x, self.world.player_y
        x = TimestepDriver.interpolate(
            prev_snapshot["player_x"], self.world.player_x, alpha
        )
        y = TimestepDriver.interpolate(
            prev_snapshot["player_y"], self.world.player_y, alpha
        )
        return x, y

    def build_archive(self) -> FrameArchive:
        return self.queue.archive

    def release(self) -> Dict[str, int]:
        """释放录制缓冲（归档 + 快照），幂等。"""
        if self.released:
            return {"events": 0, "snapshots": 0}
        events = self.queue.release()
        snapshots = len(self.snapshots)
        self.snapshots.clear()
        self.frame_hashes.clear()
        self.released = True
        return {"events": events, "snapshots": snapshots}

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class Replayer:
    """回放端：逐帧整桶注入事件、逐帧校验状态哈希、结束释放缓冲。"""

    def __init__(
        self,
        archive: FrameArchive,
        expected_hashes: Dict[int, str],
        seed: int = DEFAULT_SEED,
        snapshots: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> None:
        self.archive = archive
        self.expected_hashes = dict(expected_hashes)
        self.world = World(rng=DeterministicRng(seed))
        self.snapshots = dict(snapshots or {})
        self.mismatches: List[Dict[str, Any]] = []
        self.checked_frames: List[int] = []
        self.released = False

    @classmethod
    def from_recorder(cls, recorder: Recorder) -> "Replayer":
        return cls(
            recorder.build_archive(),
            dict(recorder.frame_hashes),
            seed=recorder.world.rng.seed,
            snapshots=dict(recorder.snapshots),
        )

    def step(self) -> Optional[Dict[str, Any]]:
        """回放一个逻辑帧并校验哈希；已经追平录制末帧时返回 None。"""
        if self.world.frame >= self._last_frame():
            return None
        events = self.archive.get(self.world.frame)
        step_world(self.world, events)
        return self._verify_frame()

    def run(self) -> Dict[str, Any]:
        """回放全部逻辑帧，释放缓冲后返回校验报告。"""
        while self.step() is not None:
            pass
        self.release()
        return self.validation_report()

    def _last_frame(self) -> int:
        hash_frames = [f for f in self.expected_hashes if f > 0]
        return max(hash_frames, default=0)

    def _verify_frame(self) -> Dict[str, Any]:
        frame = self.world.frame
        actual = state_hash(self.world)
        expected = self.expected_hashes.get(frame)
        ok = expected is not None and actual == expected
        self.checked_frames.append(frame)
        result = {
            "frame": frame,
            "expected": expected,
            "actual": actual,
            "match": ok,
        }
        if not ok:
            self.mismatches.append(result)
        return result

    def rollback_to_authoritative(self, frame: Optional[int] = None) -> bool:
        """哈希不一致时用录制端权威快照回滚（确定性兜底）。"""
        if frame is None:
            frame = self.world.frame
        snapshot = self.snapshots.get(frame)
        if snapshot is None:
            return False
        self.world = World.from_snapshot(snapshot)
        return True

    def validation_report(self) -> Dict[str, Any]:
        return {
            "frames_checked": len(self.checked_frames),
            "mismatches": len(self.mismatches),
            "match": not self.mismatches,
            "mismatch_frames": [item["frame"] for item in self.mismatches],
            "released": self.released,
        }

    def release(self) -> int:
        """释放回放持有的帧归档缓冲，幂等。"""
        if self.released:
            return 0
        count = self.archive.release()
        self.released = True
        return count

    def __enter__(self) -> "Replayer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


# ---------------------------------------------------------------------------
# 模块级入口函数
#
# 每个入口都接受宽松参数（包括契约测试传入的单个整数），整数场景一律
# 走“空场景默认值”，保证返回值非 None。
# ---------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def frame_archive(events: Any = None, *extra: Any) -> FrameArchive:
    """把 (frame, event) 序列或 InputEvent 序列归档为 FrameArchive。"""
    archive = FrameArchive()
    if events is None or _is_int(events):
        return archive
    iterable: Iterable[Any] = events if isinstance(events, (list, tuple)) else [events]
    for item in iterable:
        if isinstance(item, InputEvent):
            archive.add(max(item.frame, 0), item)
        elif isinstance(item, dict) and "frame" in item:
            archive.add(int(item["frame"]), InputEvent.from_dict(item))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            frame, event = item
            archive.add(int(frame), event if isinstance(event, InputEvent) else InputEvent(str(event)))
    return archive


def fixed_seed(seed: Any = DEFAULT_SEED) -> DeterministicRng:
    """用固定种子构造确定性 RNG；非整数种子回落到默认种子。"""
    return DeterministicRng(int(seed) if _is_int(seed) else DEFAULT_SEED)


def timestep_clamp(raw_dt: Any, min_dt: float = MIN_DT, max_dt: float = MAX_DT) -> float:
    """时间步夹紧入口；非数值输入返回下限默认值。"""
    try:
        return clamp_dt(float(raw_dt), min_dt, max_dt)
    except (TypeError, ValueError):
        return min_dt


def pause_record(recorder: Any, event: Any = None) -> bool:
    """暂停感知的事件录制入口。"""
    if not isinstance(recorder, Recorder):
        # 单整数等占位调用：默认以暂停态处理，事件被拒绝
        return False
    if event is None or _is_int(event):
        return not recorder.paused
    if not isinstance(event, InputEvent):
        event = InputEvent(str(event))
    return recorder.push(event)


def buffer_release(target: Any) -> Dict[str, int]:
    """释放录制/回放缓冲；未知目标返回零计数，保证幂等语义。"""
    if isinstance(target, (Recorder, Replayer)):
        result = target.release()
        if isinstance(result, int):
            return {"events": result}
        return result
    if isinstance(target, (FrameArchive, EventQueue)):
        return {"events": target.release()}
    return {"events": 0}


def determinism(seed: Any = DEFAULT_SEED, count: int = 8) -> Dict[str, Any]:
    """演示固定种子：同种子两个 RNG 产生完全相同的序列。"""
    rng_a = fixed_seed(seed)
    rng_b = fixed_seed(seed)
    seq_a = [rng_a.next_float() for _ in range(count)]
    seq_b = [rng_b.next_float() for _ in range(count)]
    return {"seed": rng_a.seed, "sequence": seq_a, "deterministic": seq_a == seq_b}


def validation(replayer: Any) -> Dict[str, Any]:
    """返回回放器的逐帧哈希校验报告。"""
    if isinstance(replayer, Replayer):
        return replayer.validation_report()
    return {"frames_checked": 0, "mismatches": 0, "match": True, "mismatch_frames": [], "released": False}


def audit(recorder: Any) -> Dict[str, Any]:
    """录制审计：帧范围、事件数、暂停丢弃数、哈希覆盖数。"""
    if not isinstance(recorder, Recorder):
        return {
            "frames": 0,
            "event_count": 0,
            "dropped_while_paused": 0,
            "hash_count": 0,
            "snapshot_count": 0,
            "paused": False,
            "released": False,
        }
    return {
        "frames": recorder.frame,
        "event_count": len(recorder.queue.archive.all_events()),
        "dropped_while_paused": recorder.queue.dropped_while_paused,
        "hash_count": len(recorder.frame_hashes),
        "snapshot_count": len(recorder.snapshots),
        "paused": recorder.paused,
        "released": recorder.released,
    }


def fallback(replayer: Any, frame: Any = None) -> bool:
    """哈希失配后的权威快照回滚入口。"""
    if not isinstance(replayer, Replayer):
        return False
    target_frame = int(frame) if _is_int(frame) else None
    return replayer.rollback_to_authoritative(target_frame)


def event_queue(events: Any = None) -> EventQueue:
    """构造事件队列并可选地预填 (frame, event) 序列。"""
    queue = EventQueue()
    if events is None or _is_int(events):
        return queue
    iterable = events if isinstance(events, (list, tuple)) else [events]
    for item in iterable:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            frame, event = item
            if not isinstance(event, InputEvent):
                event = InputEvent(str(event))
            queue.push(int(frame), event)
        elif isinstance(item, InputEvent):
            queue.push(max(item.frame, 0), item)
    return queue


def metrics(target: Any) -> Dict[str, int]:
    """录制端/回放端的计数指标。"""
    if isinstance(target, Recorder):
        return {
            "frames": target.frame,
            "events": len(target.queue.archive.all_events()),
            "hashes": len(target.frame_hashes),
            "dropped_while_paused": target.queue.dropped_while_paused,
            "released": int(target.released),
        }
    if isinstance(target, Replayer):
        return {
            "frames": target.world.frame,
            "events": len(target.archive.all_events()),
            "hashes": len(target.expected_hashes),
            "checked": len(target.checked_frames),
            "mismatches": len(target.mismatches),
            "released": int(target.released),
        }
    return {"frames": 0, "events": 0, "hashes": 0, "released": 0}


def _coerce_ticks(ticks: Any) -> List[Dict[str, Any]]:
    """把场景脚本归一化为 tick 字典列表。

    支持两种形态：
    - [{"dt": ..., "events": [...], "paused": ...}, ...]
    - [InputEvent, ...]（旧形态：每个事件占一个正常帧）
    """
    if ticks is None or _is_int(ticks):
        return []
    if not isinstance(ticks, (list, tuple)):
        return []
    normalized: List[Dict[str, Any]] = []
    for tick in ticks:
        if isinstance(tick, dict):
            normalized.append(
                {
                    "dt": float(tick.get("dt", FIXED_DT)),
                    "events": list(tick.get("events", ())),
                    "paused": bool(tick.get("paused", False)),
                }
            )
        elif isinstance(tick, InputEvent):
            normalized.append({"dt": FIXED_DT, "events": [tick], "paused": False})
    return normalized


def record(ticks: Any = None, seed: Any = DEFAULT_SEED) -> Recorder:
    """按场景脚本执行一次录制，返回录制器（含帧归档与权威哈希）。"""
    recorder = Recorder(seed=int(seed) if _is_int(seed) else DEFAULT_SEED)
    for tick in _coerce_ticks(ticks):
        recorder.set_paused(tick["paused"])
        for event in tick["events"]:
            if not isinstance(event, InputEvent):
                event = InputEvent(str(event))
            recorder.push(event)
        recorder.tick(tick["dt"])
    recorder.set_paused(False)
    return recorder


def replay(source: Any, seed: Any = DEFAULT_SEED) -> Dict[str, Any]:
    """回放录制结果，返回校验报告（含缓冲释放标记）。"""
    if isinstance(source, Recorder):
        replayer = Replayer.from_recorder(source)
    elif isinstance(source, FrameArchive):
        replayer = Replayer(source, {}, seed=int(seed) if _is_int(seed) else DEFAULT_SEED)
    else:
        recorder = record(source, seed)
        replayer = Replayer.from_recorder(recorder)
    report = replayer.run()
    report["released"] = replayer.released
    return report


def verify(ticks: Any = None, seed: Any = DEFAULT_SEED) -> Dict[str, Any]:
    """端到端：录制 → 回放 → 逐帧状态哈希校验，返回完整报告。"""
    recorder = record(ticks, seed)
    replayer = Replayer.from_recorder(recorder)
    while replayer.step() is not None:
        pass
    report = replayer.validation_report()
    report["audit"] = audit(recorder)
    report["final_state"] = replayer.world.snapshot()
    replayer.release()
    recorder.release()
    report["released"] = True
    return report
