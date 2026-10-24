"""Property 6: the program parser accepts only the grammar and never executes anything (Req 5.2)."""
from __future__ import annotations

import builtins

import pytest
from hypothesis import given, settings, strategies as st

from motormem.brain.program import (
    ACTUATE_ACTIONS, MAX_CALLS, PLACE_RELATIONS, PUSH_LIMIT, Call, ProgramError,
    format_program, is_valid_call, parse_program,
)

marks = st.integers(1, 99)
offsets = st.floats(-PUSH_LIMIT, PUSH_LIMIT, allow_nan=False, allow_infinity=False)
parts = st.lists(st.sampled_from(["handle", "rim", "knob", "lid", "top", "left", "side"]),
                 min_size=1, max_size=3).map(" ".join)
calls = st.one_of(
    st.builds(lambda m: Call("pick", (m,)), marks),
    st.builds(lambda m, p: Call("pick", (m,), p), marks, parts),
    st.builds(lambda m: Call("look", (m,)), marks),
    st.just(Call("home", ())),
    st.builds(lambda m, r: Call("place", (m, r)), marks, st.sampled_from(PLACE_RELATIONS)),
    st.builds(lambda m, a: Call("actuate", (m, a)), marks, st.sampled_from(ACTUATE_ACTIONS)),
    st.builds(lambda m, x, y: Call("push", (m, float(x), float(y))), marks, offsets, offsets),
)
programs = st.lists(calls, min_size=1, max_size=MAX_CALLS)

# Grammar-shaped fuzz: tokens that look like programs, so the fuzzer reaches deep parser paths.
TOKENS = ["pick", "place", "push", "actuate", "look", "home", "eval", "exec", "__import__", "os",
          "(", ")", ",", " ", "\n", "#", "=", "dx", "dy", ".", "*", "**", "-", "+", "[", "]",
          "lambda", ":", ";", "'", '"', "0", "1", "3", "42", "99", "100", "0.1", "-0.5", "1e999",
          "True", "None", "on", "in", "open", "turn_on", '"on"', '"in"', '"open"', "x", "`"]
grammarish = st.lists(st.sampled_from(TOKENS), max_size=40).map("".join)


def _check(text: str) -> None:
    try:
        out = parse_program(text)
    except ProgramError as e:
        assert str(e)          # always a reason
        return
    assert 1 <= len(out) <= MAX_CALLS
    assert all(is_valid_call(c) for c in out), out


@settings(max_examples=400, deadline=None)
@given(st.text(max_size=300))
def test_arbitrary_text_parses_to_grammar_or_raises(text):
    _check(text)


@settings(max_examples=600, deadline=None)
@given(grammarish)
def test_grammar_shaped_text_parses_to_grammar_or_raises(text):
    _check(text)


@settings(max_examples=300, deadline=None)
@given(programs)
def test_valid_programs_round_trip(prog):
    assert parse_program(format_program(prog)) == prog


@settings(max_examples=200, deadline=None)
@given(programs, st.lists(st.sampled_from(["", "   ", "# a comment", "  # x"]), max_size=5), st.data())
def test_round_trip_with_comments_and_blank_lines(prog, extras, data):
    lines = format_program(prog).splitlines()
    for e in extras:
        lines.insert(data.draw(st.integers(0, len(lines))), e)
    text = "\n".join(l + ("  # why" if i % 2 else "") for i, l in enumerate(lines))
    assert parse_program(text) == prog


def test_never_evaluates(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("parser must not eval/exec/compile-exec")
    monkeypatch.setattr(builtins, "eval", boom)
    monkeypatch.setattr(builtins, "exec", boom)
    flag = tmp_path / "pwned"
    evil = [
        f"__import__('os').system('touch {flag}')",
        f"pick(__import__('os').system('touch {flag}'))",
        "pick(*[1])", "pick(**{'m': 1})", "pick(1 if True else 2)", "pick((lambda: 1)())",
        "os.system('ls')", "pick(3).x", "pick(3)[0]", "pick(3); home()", "pick(3)\nexec('1')",
    ]
    for p in evil:
        with pytest.raises(ProgramError):
            parse_program(p)
    assert not flag.exists()


@pytest.mark.parametrize("prog,expected", [
    ("pick(3)", [Call("pick", (3,))]),
    ('pick(3, part="handle")', [Call("pick", (3,), "handle")]),
    ("pick(3, part='drawer handle')", [Call("pick", (3,), "drawer handle")]),
    ('place(2, "on")', [Call("place", (2, "on"))]),
    ("place(2, 'in')", [Call("place", (2, "in"))]),
    ("push(4, dx=0.1, dy=-0.05)", [Call("push", (4, 0.1, -0.05))]),
    ("push(4, dy=0.2)", [Call("push", (4, 0.0, 0.2))]),
    ('actuate(1, "turn_on")\nhome()', [Call("actuate", (1, "turn_on")), Call("home", ())]),
    ("```\nlook(7)\n```", [Call("look", (7,))]),
])
def test_examples(prog, expected):
    assert parse_program(prog) == expected


@pytest.mark.parametrize("prog", [
    "", "# only a comment", "pick(0)", "pick(100)", "pick(True)", "pick(3.0)", 'pick("3")',
    'place(2, "under")', "place(2)", "push(1, dx=0.31)", "push(1, dx=1e999)", "push(1)",
    "push(1, dz=0.1)", "push(1, 0.1, dx=0.1)", 'actuate(1, "spin")', "home(1)", "grasp(1)",
    'pick(3, part="")', 'pick(3, part="a b c d")', 'pick(3, part="handle2")', 'pick(3, part=5)',
    'look(3, part="handle")', 'place(3, "on", part="rim")',
    "\n".join(["home()"] * (MAX_CALLS + 1)), "pick(3) pick(4)",
])
def test_rejections(prog):
    with pytest.raises(ProgramError):
        parse_program(prog)
