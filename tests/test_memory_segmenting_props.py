"""Property 9: gripper-event segmentation covers each close/open event exactly once (Req 3.1)."""
import numpy as np
from hypothesis import given, settings, strategies as st

from motormem.memory.segmenting import DEBOUNCE, gripper_events, event_windows, parse_relation


@st.composite
def command_sequences(draw):
    """Alternating open/closed runs (each >= DEBOUNCE long) plus optional short flickers inside runs."""
    start_closed = draw(st.booleans())
    runs = draw(st.lists(st.integers(DEBOUNCE + 2, 40), min_size=1, max_size=8))
    cmd, truth, state, t = [], [], start_closed, 0
    for i, n in enumerate(runs):
        if i > 0:
            truth.append((t, "close" if state else "open"))
        vals = np.full(n, 1.0 if state else -1.0) * draw(st.floats(0.2, 1.0))
        if draw(st.booleans()) and n >= 3 * DEBOUNCE:
            # flicker shorter than DEBOUNCE, at least DEBOUNCE steps away from both run edges
            L = draw(st.integers(1, DEBOUNCE - 1))
            k = draw(st.integers(DEBOUNCE, n - DEBOUNCE - L))
            vals[k:k + L] = -vals[k:k + L]
        cmd.extend(vals.tolist())
        t += n
        state = not state
    return np.array(cmd), truth


@settings(max_examples=300, deadline=None)
@given(command_sequences())
def test_events_match_generated_transitions(seq):
    cmd, truth = seq
    ev = gripper_events(cmd)
    assert [(e.t, e.kind) for e in ev] == truth


@settings(max_examples=300, deadline=None)
@given(command_sequences())
def test_each_event_covered_by_exactly_one_window(seq):
    cmd, _ = seq
    T = len(cmd)
    ev = gripper_events(cmd)
    wins = event_windows(ev, T)
    for e in ev:
        keyed = [w for w in wins if w.event == e]
        assert len(keyed) == 1
        w = keyed[0]
        assert 0 <= w.start <= e.t < w.end <= T
        assert w.kind == ("grasp" if e.kind == "close" else "place")
    # windows are ordered by their key events and never start before the previous key event
    for a, b in zip(wins, wins[1:]):
        assert a.event.t < b.event.t
        assert b.start >= a.event.t


def test_relation_parsing():
    assert parse_relation("put the bowl on the plate") == "on"
    assert parse_relation("put the bowl on top of the cabinet") == "top_of"
    assert parse_relation("put the cream cheese in the bowl") == "in"
    assert parse_relation("open the top drawer and put the bowl inside") == "in"
    assert parse_relation("put the chocolate pudding to the left of the plate") == "left_of"
    assert parse_relation("push the plate to the front of the stove") == "front_of"
    assert parse_relation("pick up the book and place it in the back compartment of the caddy") == "in"
