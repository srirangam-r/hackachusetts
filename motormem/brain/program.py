"""Restricted skill-program grammar for the robot brain (Req 5.2).

One call per line, literal arguments only:

    pick(M)                          # or pick(M, part="handle") to grasp a named part
    place(M, "on"|"in"|"top_of"|"left_of"|"right_of"|"front_of"|"back_of")
    push(M, dx=<m>, dy=<m>)          # metres, |dx|, |dy| <= 0.3
    actuate(M, "open"|"close"|"turn_on"|"turn_off")
    look(M)
    home()

M is an integer mark 1..99 from the last observation. Blank lines and ``#`` comments are
allowed; at most ``MAX_CALLS`` calls. Lines are parsed with ``ast.parse(mode="eval")`` and only
``Call(Name, literal args)`` nodes are accepted. Nothing is ever evaluated or executed.
"""
from __future__ import annotations

import ast
import math
import warnings
from dataclasses import dataclass

PLACE_RELATIONS = ("on", "in", "top_of", "left_of", "right_of", "front_of", "back_of")
ACTUATE_ACTIONS = ("open", "close", "turn_on", "turn_off")
SKILLS = ("pick", "place", "push", "actuate", "look", "home")
MARK_MIN, MARK_MAX = 1, 99
PUSH_LIMIT = 0.3          # metres
MAX_CALLS = 12
MAX_LINE_CHARS = 200
MAX_PROGRAM_CHARS = 4000


class ProgramError(ValueError):
    """A program outside the grammar. The message is short and model-readable."""


PART_MAX_WORDS = 3


@dataclass(frozen=True)
class Call:
    """One validated skill call. `args` is normalised per skill:

    pick/look: (mark,)   place: (mark, relation)   push: (mark, dx, dy)
    actuate: (mark, action)   home: ()

    `part` is an optional short noun phrase for `pick` (e.g. "handle"); None otherwise.
    """
    name: str
    args: tuple = ()
    part: str | None = None

    @property
    def mark(self) -> int | None:
        return self.args[0] if self.args else None

    def __str__(self) -> str:
        return format_call(self)


# ---------------------------------------------------------------------------- validation
def _is_mark(v) -> bool:
    return type(v) is int and MARK_MIN <= v <= MARK_MAX


def _is_offset(v) -> bool:
    return type(v) in (int, float) and math.isfinite(v) and abs(v) <= PUSH_LIMIT


def _is_part(v) -> bool:
    """A part is a short noun phrase: letters and spaces only, 1..PART_MAX_WORDS words."""
    if type(v) is not str:
        return False
    words = v.split()
    return (0 < len(words) <= PART_MAX_WORDS
            and all(w.isalpha() for w in words) and v == " ".join(words))


def is_valid_call(call: Call) -> bool:
    """Independent grammar membership check (used by tests and as a final guard)."""
    if not isinstance(call, Call) or call.name not in SKILLS or not isinstance(call.args, tuple):
        return False
    if call.part is not None and (call.name != "pick" or not _is_part(call.part)):
        return False
    a = call.args
    if call.name in ("pick", "look"):
        return len(a) == 1 and _is_mark(a[0])
    if call.name == "place":
        return len(a) == 2 and _is_mark(a[0]) and type(a[1]) is str and a[1] in PLACE_RELATIONS
    if call.name == "actuate":
        return len(a) == 2 and _is_mark(a[0]) and type(a[1]) is str and a[1] in ACTUATE_ACTIONS
    if call.name == "push":
        return (len(a) == 3 and _is_mark(a[0]) and type(a[1]) is float and type(a[2]) is float
                and _is_offset(a[1]) and _is_offset(a[2]))
    if call.name == "home":
        return a == ()
    return False


# ---------------------------------------------------------------------------- parsing
_LITERAL_TYPES = (int, float, str)


def _literal(node: ast.AST):
    """Return the value of a plain literal (number, string, or signed number); else raise."""
    if isinstance(node, ast.Constant) and type(node.value) in _LITERAL_TYPES:
        return node.value
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float)):
        v = node.operand.value
        return -v if isinstance(node.op, ast.USub) else v
    raise ProgramError("arguments must be plain literals (numbers or quoted strings)")


def _mark(v) -> int:
    if not _is_mark(v):
        raise ProgramError(f"mark must be an integer {MARK_MIN}..{MARK_MAX}, got {v!r}")
    return v


def _offset(name: str, v) -> float:
    if type(v) not in (int, float) or not math.isfinite(v):
        raise ProgramError(f"{name} must be a number in metres")
    if abs(v) > PUSH_LIMIT:
        raise ProgramError(f"{name}={v} out of range (|{name}| <= {PUSH_LIMIT} m)")
    return float(v)


def _choice(what: str, v, options: tuple[str, ...]) -> str:
    if type(v) is not str or v not in options:
        raise ProgramError(f"{what} must be one of {', '.join(options)}; got {v!r}")
    return v


def _part(v) -> str:
    if type(v) is not str:
        raise ProgramError("part must be a short quoted noun phrase, e.g. part=\"handle\"")
    norm = " ".join(v.split())
    if not _is_part(norm):
        raise ProgramError(f'part must be <= {PART_MAX_WORDS} words of letters/spaces; got {v!r}')
    return norm


def _build(name: str, pos: list, kw: dict) -> Call:
    if name == "pick":
        if set(kw) - {"part"} or len(pos) != 1:
            raise ProgramError('usage: pick(M) or pick(M, part="handle")')
        part = _part(kw["part"]) if "part" in kw else None
        return Call("pick", (_mark(pos[0]),), part)
    if name == "look":
        if kw or len(pos) != 1:
            raise ProgramError("usage: look(M)")
        return Call(name, (_mark(pos[0]),))
    if name == "home":
        if kw or pos:
            raise ProgramError("usage: home()")
        return Call(name, ())
    if name == "place":
        if kw or len(pos) != 2:
            raise ProgramError('usage: place(M, "on"|"in"|"top_of"|"left_of"|"right_of"|"front_of"|"back_of")')
        return Call(name, (_mark(pos[0]), _choice("relation", pos[1], PLACE_RELATIONS)))
    if name == "actuate":
        if kw or len(pos) != 2:
            raise ProgramError('usage: actuate(M, "open"|"close"|"turn_on"|"turn_off")')
        return Call(name, (_mark(pos[0]), _choice("action", pos[1], ACTUATE_ACTIONS)))
    if name == "push":
        usage = "usage: push(M, dx=<m>, dy=<m>)"
        if set(kw) - {"dx", "dy"} or not 1 <= len(pos) <= 3:
            raise ProgramError(usage)
        vals = dict(zip(("dx", "dy"), pos[1:]))
        for k, v in kw.items():
            if k in vals:
                raise ProgramError(f"{k} given twice; {usage}")
            vals[k] = v
        if not vals:
            raise ProgramError(usage)
        return Call(name, (_mark(pos[0]), _offset("dx", vals.get("dx", 0.0)), _offset("dy", vals.get("dy", 0.0))))
    raise ProgramError(f"unknown skill {name!r}; allowed: {', '.join(SKILLS)}")


def parse_line(line: str) -> Call:
    """Parse one non-empty, comment-free line into a `Call`."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # e.g. SyntaxWarning for "1x"
            tree = ast.parse(line.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
        raise ProgramError("not a single skill call") from None
    node = tree.body
    if not isinstance(node, ast.Call):
        raise ProgramError("each line must be one skill call like pick(3)")
    if not isinstance(node.func, ast.Name):
        raise ProgramError("call a skill by name, e.g. pick(3)")
    name = node.func.id
    if name not in SKILLS:
        raise ProgramError(f"unknown skill {name!r}; allowed: {', '.join(SKILLS)}")
    pos = []
    for a in node.args:
        if isinstance(a, ast.Starred):
            raise ProgramError("starred arguments are not allowed")
        pos.append(_literal(a))
    kw = {}
    for k in node.keywords:
        if k.arg is None:
            raise ProgramError("** arguments are not allowed")
        if k.arg in kw:
            raise ProgramError(f"{k.arg} given twice")
        kw[k.arg] = _literal(k.value)
    call = _build(name, pos, kw)
    assert is_valid_call(call), call   # parser and checker must agree
    return call


def parse_program(text: str) -> list[Call]:
    """Parse a whole program. Raises `ProgramError("line N: reason")` on the first bad line."""
    if not isinstance(text, str):
        raise ProgramError("program must be a string")
    if len(text) > MAX_PROGRAM_CHARS:
        raise ProgramError(f"program too long (> {MAX_PROGRAM_CHARS} characters)")
    calls: list[Call] = []
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("```"):   # blank, comment, or a markdown fence line
            continue
        if len(line) > MAX_LINE_CHARS:
            raise ProgramError(f"line {n}: line too long")
        try:
            calls.append(parse_line(line))
        except ProgramError as e:
            raise ProgramError(f"line {n}: {e}") from None
        if len(calls) > MAX_CALLS:
            raise ProgramError(f"too many calls (max {MAX_CALLS})")
    if not calls:
        raise ProgramError("empty program: write one skill call per line, e.g. pick(3)")
    return calls


# ---------------------------------------------------------------------------- formatting
def format_call(call: Call) -> str:
    a = call.args
    if call.name == "pick" and call.part is not None:
        return f'pick({a[0]}, part="{call.part}")'
    if call.name in ("pick", "look"):
        return f"{call.name}({a[0]})"
    if call.name in ("place", "actuate"):
        return f'{call.name}({a[0]}, "{a[1]}")'
    if call.name == "push":
        return f"push({a[0]}, dx={a[1]!r}, dy={a[2]!r})"
    return "home()"


def format_program(calls: list[Call]) -> str:
    return "\n".join(format_call(c) for c in calls)
