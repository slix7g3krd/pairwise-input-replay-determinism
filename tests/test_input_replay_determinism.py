import pytest
import gameplay.input_replay_determinism as mod

CASES = [
    ("frame_archive", (1,)),
    ("fixed_seed", (1,)),
    ("timestep_clamp", (1,)),
    ("pause_record", (1,)),
    ("buffer_release", (1,)),
    ("determinism", (1,)),
    ("state_hash", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("event_queue", (1,)),
    ("metrics", (1,)),
    ("record", (1,)),
    ("replay", (1,)),
    ("verify", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
