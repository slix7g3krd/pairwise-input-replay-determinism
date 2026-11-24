"""六个修复点的端到端回归测试。"""

import pytest

from gameplay.input_replay_determinism import (
    DEFAULT_SEED,
    DeterministicRng,
    FIXED_DT,
    FrameArchive,
    InputEvent,
    MAX_DT,
    MIN_DT,
    Recorder,
    Replayer,
    TimestepDriver,
    World,
    audit,
    buffer_release,
    determinism,
    fallback,
    frame_archive,
    metrics,
    pause_record,
    record,
    replay,
    state_hash,
    timestep_clamp,
    verify,
)


def move(dx=0.0, dy=0.0):
    return InputEvent("move", {"dx": dx, "dy": dy})


# --- 修复 1：事件按逻辑帧归档，高频输入不丢事件 -----------------------------


def test_high_frequency_events_between_frames_are_all_archived():
    recorder = Recorder()
    # 同一逻辑帧内涌入 5 条高频输入（两帧之间到达的极端情况）
    for _ in range(5):
        assert recorder.push(move(dx=1.0))
    recorder.tick(FIXED_DT)

    bucket = recorder.build_archive().get(0)
    assert len(bucket) == 5
    assert all(ev.frame == 0 for ev in bucket)


def test_replay_reproduces_recorded_world_state_exactly():
    ticks = [
        {"dt": FIXED_DT, "events": [move(dx=1.0), move(dy=1.0)]},
        {"dt": FIXED_DT, "events": [move(dx=-1.0)]},
        {"dt": FIXED_DT, "events": []},
        {"dt": FIXED_DT, "events": [move(dy=-1.0), move(dx=1.0), move(dx=1.0)]},
    ]
    report = verify(ticks)
    assert report["match"] is True
    assert report["mismatches"] == 0
    assert report["frames_checked"] == 4


def test_frame_archive_round_trip():
    archive = frame_archive([(0, move(1.0)), (0, move(0.0, 1.0)), (2, move(-1.0))])
    assert archive.frames() == [0, 2]
    assert len(archive.get(0)) == 2

    restored = FrameArchive.from_dict(archive.to_dict())
    assert restored.get(0) == archive.get(0)
    assert restored.get(2) == archive.get(2)


# --- 修复 2：固定随机种子，AI 行为可复现 ------------------------------------


def test_fixed_seed_rng_sequence_is_deterministic():
    info = determinism(DEFAULT_SEED, count=16)
    assert info["deterministic"] is True
    assert DeterministicRng(DEFAULT_SEED).next_float() == DeterministicRng(DEFAULT_SEED).next_float()


def test_ai_trajectory_identical_across_record_and_replay():
    ticks = [{"dt": FIXED_DT, "events": [move(dx=1.0)]} for _ in range(10)]
    recorder = record(ticks, seed=42)
    replayer = Replayer.from_recorder(recorder)
    replayer.run()
    assert replayer.world.ai_x == recorder.world.ai_x
    assert replayer.world.rng.snapshot() == recorder.world.rng.snapshot()


# --- 修复 3：时间步夹紧并插值，掉帧不累积偏差 -------------------------------


def test_timestep_clamp_caps_spikes_and_floors_bad_values():
    assert timestep_clamp(10.0) == MAX_DT          # 掉帧尖刺被夹
    assert timestep_clamp(-1.0) == MIN_DT          # 异常负值被夹
    inside = FIXED_DT
    assert timestep_clamp(inside) == inside


def test_spiky_frame_times_consume_same_fixed_steps_as_smooth():
    total = 1.0
    # 平滑：60 个均匀帧；抖动：分组求和严格等于 1.0 的不均帧
    smooth = TimestepDriver()
    for _ in range(60):
        smooth.tick(total / 60.0)
    assert smooth.steps_available() == 0
    assert smooth.sim_time == pytest.approx(total)

    janky = TimestepDriver()
    # 每帧耗时均落在 [MIN_DT, MAX_DT] 内，但帧间不均匀；分组求和凑成 total
    groups = [(0.05, 8), (0.04, 5), (0.02, 10), (0.01, 20)]
    for chunk, count in groups:
        for _ in range(count):
            janky.tick(chunk)
    assert sum(chunk * count for chunk, count in groups) == pytest.approx(total)
    assert janky.sim_time == pytest.approx(total)
    assert janky.sim_time == smooth.sim_time


def test_render_interpolation_does_not_mutate_simulation_state():
    recorder = Recorder()
    recorder.tick(FIXED_DT)
    sim_x = recorder.world.player_x
    render_x, _ = recorder.render_state()
    assert recorder.world.player_x == sim_x          # 仿真态未被插值污染
    assert isinstance(render_x, float)


# --- 修复 4：暂停时暂停录制，暂停期间事件不入档、帧号不前进 -----------------


def test_paused_recorder_drops_events_and_does_not_advance_frame():
    recorder = Recorder()
    recorder.tick(FIXED_DT)
    frame_before = recorder.frame

    recorder.set_paused(True)
    assert pause_record(recorder, move(1.0)) is False
    assert recorder.push(move(1.0)) is False
    steps = recorder.tick(10.0 * FIXED_DT)           # 暂停多久都不推进
    assert steps == 0
    assert recorder.frame == frame_before
    assert audit(recorder)["dropped_while_paused"] == 2
    assert recorder.build_archive().all_events() == []

    recorder.set_paused(False)
    assert recorder.push(move(1.0)) is True
    recorder.tick(FIXED_DT)
    assert recorder.frame == frame_before + 1


# --- 修复 5：回放结束释放录制缓冲，重复释放幂等 -----------------------------


def test_replay_releases_archive_buffer():
    recorder = record([{"dt": FIXED_DT, "events": [move(1.0)]}])
    replayer = Replayer.from_recorder(recorder)
    report = replayer.run()
    assert report["released"] is True
    assert replayer.archive.all_events() == []
    assert replayer.release() == 0                   # 幂等
    assert buffer_release(replayer) == {"events": 0}


def test_context_managers_release_buffers():
    with Recorder() as recorder:
        recorder.push(move(1.0))
        recorder.tick(FIXED_DT)
    assert recorder.released is True

    archive = FrameArchive()
    with Replayer(archive, {}) as replayer:
        pass
    assert replayer.released is True


# --- 修复 6：逐帧校验状态哈希，失配可回滚到权威快照 -------------------------


def test_state_hash_stable_and_sensitive_to_drift():
    world_a = World(rng=DeterministicRng(1))
    world_b = World(rng=DeterministicRng(1))
    assert state_hash(world_a) == state_hash(world_b)
    world_b.player_x += 0.001
    assert state_hash(world_a) != state_hash(world_b)


def test_per_frame_hash_mismatch_is_detected_and_can_rollback():
    recorder = record([{"dt": FIXED_DT, "events": [move(1.0)]} for _ in range(3)], seed=7)
    replayer = Replayer.from_recorder(recorder)

    first = replayer.step()
    assert first["match"] is True

    # 人为制造漂移：篡改回放世界位置
    replayer.world.player_x += 5.0
    result = replayer.step()
    assert result["match"] is False
    assert replayer.mismatches[-1]["frame"] == 2

    # 兜底：用录制端权威快照回滚后继续回放，后续帧重新对齐
    assert fallback(replayer) is True
    assert state_hash(replayer.world) == replayer.expected_hashes[replayer.world.frame]
    assert replayer.step()["match"] is True


def test_verify_end_to_end_report_and_metrics():
    ticks = [
        {"dt": FIXED_DT, "events": [move(1.0)]},
        {"dt": FIXED_DT, "events": [], "paused": True},
        {"dt": FIXED_DT, "events": [move(0.0, 1.0), move(0.0, 1.0)]},
    ]
    report = verify(ticks, seed=99)
    assert report["match"] is True
    assert report["released"] is True
    assert report["audit"]["dropped_while_paused"] == 0  # 暂停帧本身无事件
    assert metrics(record(ticks))["frames"] == 2         # 暂停帧不推进帧号


def test_replay_entrypoint_accepts_recorder_and_scripts():
    ticks = [{"dt": FIXED_DT, "events": [move(1.0)]}]
    assert replay(ticks)["match"] is True
    assert replay(record(ticks))["match"] is True
