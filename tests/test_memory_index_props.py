"""Property 5: removed demos/objects never come back from a query (Req 3.4, 3.5) + MotorMemory basics."""
import numpy as np
from hypothesis import given, settings, strategies as st

from motormem.memory.index import MotorMemory
from motormem.types import Segment, SegmentKind

NAMES = ["akita_black_bowl", "plate", "flat_stove", "wooden_cabinet", "wine_bottle", "cream_cheese"]
RELS = [None, "on", "in", "top_of", "open", "turn_on"]
KINDS = list(SegmentKind)


def make_seg(i, kind, obj, rel, held, demo):
    return Segment(segment_id=f"s{i}", demo_id=f"d{demo}", source="synthetic", kind=kind, object_name=obj,
                   relation=rel, held_name=held, T_obj_eef=np.tile(np.eye(4), (3, 1, 1)), gripper=np.zeros(3),
                   T_eef_held=None, object_points=np.random.default_rng(i).normal(size=(20, 3)) * 0.05,
                   task_language="", meta={"extent": np.array([0.1, 0.1, 0.05]) * (1 + i % 3)})


segments_st = st.lists(
    st.tuples(st.sampled_from(KINDS), st.sampled_from(NAMES), st.sampled_from(RELS),
              st.one_of(st.none(), st.sampled_from(NAMES)), st.integers(0, 9)),
    min_size=1, max_size=40,
).map(lambda rows: [make_seg(i, *r) for i, r in enumerate(rows)])


@settings(max_examples=150, deadline=None)
@given(segments_st, st.sets(st.sampled_from(NAMES), max_size=3), st.sets(st.integers(0, 9).map(lambda d: f"d{d}"),
       max_size=5), st.sampled_from(KINDS), st.one_of(st.none(), st.sampled_from(NAMES + ["black bowl", "stove"])),
       st.sampled_from(RELS), st.one_of(st.none(), st.sampled_from(NAMES)))
def test_removed_never_returned(segs, rm_obj, rm_demo, kind, qname, rel, held):
    mem = MotorMemory(segs)
    view = mem.without(objects=rm_obj, demo_ids=rm_demo)
    res = view.query(kind, qname, relation=rel, held_name=held, k=100)
    for s in res:
        assert s.demo_id not in rm_demo
        assert s.object_name not in rm_obj
        assert s.held_name not in rm_obj
    for n in rm_obj:
        assert len(view.object_points(n)) == 0
    # the original memory is untouched and the view composes
    assert len(mem.query(kind, k=1000)) == sum(1 for s in segs if s.kind == kind)
    view2 = view.without(demo_ids=["d0"])
    assert all(s.demo_id != "d0" and s.demo_id not in rm_demo for s in view2.query(kind, qname, k=100))


def test_query_ranking_and_canonical(tmp_path):
    segs = [make_seg(0, SegmentKind.GRASP, "akita_black_bowl", None, None, 0),
            make_seg(1, SegmentKind.GRASP, "plate", None, None, 1),
            make_seg(2, SegmentKind.PLACE, "plate", "on", "akita_black_bowl", 2),
            make_seg(3, SegmentKind.PLACE, "plate", "in", "cream_cheese", 3),
            make_seg(4, SegmentKind.ACTUATE, "flat_stove", "turn_on", None, 4)]
    mem = MotorMemory(segs)
    assert mem.canonical("black bowl") == "akita_black_bowl"
    assert mem.canonical("the stove") == "flat_stove"
    assert mem.canonical("plates") == "plate"
    assert mem.canonical("giraffe") is None
    assert mem.query(SegmentKind.GRASP, "the black bowl")[0].object_name == "akita_black_bowl"
    r = mem.query(SegmentKind.PLACE, "plate", relation="on", held_name="bowl")
    assert r[0].segment_id == "s2"
    # unknown name falls back to same kind ranked by extent
    r = mem.query(SegmentKind.GRASP, "mystery mug", extent=np.array([0.2, 0.2, 0.1]))
    assert [s.segment_id for s in r] == ["s1", "s0"]
    assert mem.object_points("bowl").shape[1] == 3
    p = tmp_path / "m.pkl"
    mem.without(objects=["plate"]).save(p)
    m2 = MotorMemory.load(p)
    assert {s.segment_id for s in m2.segments} == {"s0", "s4"}


# ------------------------------------------------------------ editability: excluded objects stay gone
ED_NAMES = ["akita_black_bowl", "black_book", "yellow_book", "plate", "white_yellow_mug", "porcelain_mug",
            "red_coffee_mug", "wine_bottle", "wine_rack", "wooden_cabinet", "white_cabinet", "desk_caddy",
            "cream_cheese", "flat_stove", "ketchup", "wooden_tray"]
WORDS = sorted({t for n in ED_NAMES for t in n.split("_")} | {"the", "top", "bowl", "book", "mug", "stove",
                                                               "bottle", "drawer", "left", "cup", "can"})
ed_segments_st = st.lists(
    st.tuples(st.sampled_from(KINDS), st.sampled_from(ED_NAMES), st.sampled_from(RELS),
              st.one_of(st.none(), st.sampled_from(ED_NAMES)), st.integers(0, 9)),
    min_size=1, max_size=40,
).map(lambda rows: [make_seg(i, *r) for i, r in enumerate(rows)])


@settings(max_examples=200, deadline=None)
@given(ed_segments_st, st.sets(st.sampled_from(ED_NAMES), min_size=1, max_size=3),
       st.lists(st.sampled_from(WORDS), min_size=1, max_size=4), st.sampled_from(KINDS),
       st.one_of(st.none(), st.sampled_from(ED_NAMES)))
def test_excluded_never_canonical_and_strict_empty(segs, rm, words, kind, held):
    view = MotorMemory(segs).without(objects=rm)
    labels = [" ".join(words)] + [n for n in rm] + [n.replace("_", " ") for n in rm]
    for lab in labels:
        assert view.canonical(lab) not in rm
    for n in rm:
        for lab in (n, n.replace("_", " ")):
            assert view.query(kind, lab, strict=True, k=100) == []
            assert view.query(kind, lab, held_name=held, strict=True, k=100) == []
            assert all(s.object_name != n for s in view.query(kind, lab, k=100))
    # strict results are always exactly the requested (canonical) object
    name = view.canonical(" ".join(words))
    for s in view.query(kind, " ".join(words), held_name=held, strict=True, k=100):
        assert s.object_name == name and (held is None or s.held_name == held)


def test_canonical_distinctive_tokens():
    names = ["black_book", "yellow_book", "akita_black_bowl", "ketchup", "wine_bottle", "wine_rack",
             "wooden_cabinet", "white_yellow_mug"]
    mem = MotorMemory([make_seg(i, SegmentKind.GRASP, n, None, None, i) for i, n in enumerate(names)])
    assert mem.canonical("black book") == "black_book"
    assert mem.canonical("book") in ("black_book", "yellow_book")
    assert mem.canonical("ketchup bottle") == "ketchup"
    assert mem.canonical("wine bottle") == "wine_bottle"
    assert mem.canonical("top drawer of the cabinet") == "wooden_cabinet"
    assert mem.canonical("yellow book") == "yellow_book"
    no_black = mem.without(objects=["black book"])
    assert no_black.canonical("black book") is None          # never yellow_book or akita_black_bowl
    assert no_black.canonical("black_book") is None
    assert no_black.canonical("book") == "yellow_book"       # the category alone may be any book
    assert no_black.canonical("black bowl") == "akita_black_bowl"
    assert no_black.query(SegmentKind.GRASP, "black book", strict=True) == []
    assert no_black.query(SegmentKind.GRASP, "black book")  # non-strict still ranks other segments
    only_yellow = MotorMemory([make_seg(0, SegmentKind.GRASP, "yellow_book", None, None, 0),
                               make_seg(1, SegmentKind.GRASP, "akita_black_bowl", None, None, 1)])
    assert only_yellow.canonical("black book") is None
    assert mem.without(objects=["wine_bottle"]).canonical("wine bottle") is None
