"""MotorMemory: the searchable collection of object-frame demo segments."""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from motormem.types import Segment, SegmentKind

# free-text word -> canonical-name tokens it stands for
SYNONYMS: dict[str, list[str]] = {
    "stove": ["flat_stove"], "burner": ["flat_stove"], "cooktop": ["flat_stove"],
    "cabinet": ["wooden_cabinet", "white_cabinet"], "drawer": ["wooden_cabinet", "white_cabinet"],
    "rack": ["wine_rack"], "bottle": ["wine_bottle"], "wine": ["wine_bottle"],
    "bowl": ["akita_black_bowl"], "cheese": ["cream_cheese"], "book": ["black_book"],
    "caddy": ["desk_caddy"], "tray": ["wooden_tray"], "pan": ["chefmate_8_frypan"], "frypan": ["chefmate_8_frypan"],
    "cookie": ["cookies"], "cookies": ["cookies"], "ramekin": ["glazed_rim_porcelain_ramekin"],
    "mug": ["porcelain_mug", "red_coffee_mug", "white_yellow_mug"], "cup": ["porcelain_mug", "red_coffee_mug"],
    "soup": ["alphabet_soup"], "pudding": ["chocolate_pudding"], "dressing": ["salad_dressing"],
    "juice": ["orange_juice"], "moka": ["moka_pot"], "pot": ["moka_pot"], "microwave": ["microwave"],
    "basket": ["basket"], "can": ["can", "alphabet_soup", "tomato_sauce"], "milk": ["milk"],
    "bin": ["bin"], "cereal": ["cereal"], "bread": ["bread"],
}
STOP = {"the", "a", "an", "of", "to", "on", "in", "and", "with", "that", "this", "object", "thing", "it", "its"}
# attribute words: if a label says one, the name must contain it ("black book" is never yellow_book)
ATTRS = {"black", "white", "red", "yellow", "blue", "green", "orange", "purple", "pink", "brown", "gray",
         "silver", "gold", "wooden", "metal", "glass", "plastic", "porcelain", "ceramic"}
ALIAS = {"wood": "wooden", "grey": "gray", "metallic": "metal"}
# container words that do not identify an object when unmatched ("ketchup bottle" is ketchup)
GENERIC = {"bottle", "can", "box", "carton", "jar", "pack", "packet", "container", "bag", "tin"}


def _tok(s: str) -> list[str]:
    return [ALIAS.get(t, t) for t in re.split(r"[^a-z0-9]+", s.lower()) if t and t not in STOP]


def _plural(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t


class MotorMemory:
    def __init__(self, segments: Iterable[Segment] = (), _excluded_objects: frozenset = frozenset(),
                 _excluded_demos: frozenset = frozenset()):
        self._segments: list[Segment] = []
        self._ex_obj = frozenset(_excluded_objects)
        self._ex_demo = frozenset(_excluded_demos)
        self._points_cache: dict[str, np.ndarray] = {}
        self.add_segments(list(segments))

    # ------------------------------------------------------------ building
    def add_segments(self, segs: list[Segment]) -> None:
        self._segments.extend(segs)
        self._points_cache.clear()

    def _visible(self, s: Segment) -> bool:
        if s.demo_id in self._ex_demo:
            return False
        if s.object_name in self._ex_obj or (s.held_name is not None and s.held_name in self._ex_obj):
            return False
        return True

    @property
    def segments(self) -> list[Segment]:
        return [s for s in self._segments if self._visible(s)]

    def __len__(self) -> int:
        return len(self.segments)

    def names(self) -> set[str]:
        out = set()
        for s in self.segments:
            out.add(s.object_name)
            if s.held_name:
                out.add(s.held_name)
        return out

    def demo_ids(self) -> set[str]:
        return {s.demo_id for s in self.segments}

    # ------------------------------------------------------------ names
    def _all_names(self) -> set[str]:
        out = set()
        for s in self._segments:
            out.add(s.object_name)
            if s.held_name:
                out.add(s.held_name)
        return out

    def _best_name(self, label: str) -> str | None:
        """Best name for `label` among ALL recorded names, excluded ones included.

        A name qualifies only if every distinctive label word (attribute, category noun or synonym word)
        matches it, so "black book" can be black_book but never yellow_book or akita_black_bowl, while
        "book" may be any book. Unknown words ("top", "left") are ignored.
        """
        if not label:
            return None
        names = self._all_names()
        raw = label.strip().lower().replace(" ", "_")
        if raw in names:
            return raw
        toks = [_plural(t) for t in _tok(label)]
        if not toks:
            return None
        distinct = (ATTRS | {_plural(n.split("_")[-1]) for n in names} | set(SYNONYMS)) - GENERIC
        visible = self.names()
        counts: dict[str, int] = {}
        for s in self.segments:
            counts[s.object_name] = counts.get(s.object_name, 0) + 1
        best, best_key = None, None
        for n in names:
            ntok = {_plural(t) for t in n.split("_")}
            score = 0.0
            for t in toks:
                if t in ntok:
                    score += 1.0
                elif any(n == syn or set(syn.split("_")) <= ntok for syn in SYNONYMS.get(t, [])):
                    score += 0.9
                elif t in distinct:
                    score = -1.0
                    break
            if score <= 0:
                continue
            # fewest unexplained name tokens first; then visible over excluded, then frequent names
            key = (round(score - 0.05 * len(ntok - set(toks)), 3), n in visible, counts.get(n, 0), n)
            if best_key is None or key > best_key:
                best, best_key = n, key
        return best

    def canonical(self, label: str) -> str | None:
        """Map free text ("black bowl", "the stove") to a canonical object name known to this memory.

        Returns None when nothing matches or when the best match is an excluded object.
        """
        best = self._best_name(label)
        return best if best is not None and best in self.names() else None

    def _canon_or_self(self, name: str | None) -> str | None:
        if name is None:
            return None
        return self.canonical(name) or name

    # ------------------------------------------------------------ queries
    def query(self, kind: SegmentKind, object_name: str | None = None, relation: str | None = None,
              held_name: str | None = None, extent: np.ndarray | None = None, k: int = 8,
              strict: bool = False) -> list[Segment]:
        """Exact canonical name (+ relation, + held) first, then same kind ranked by extent similarity.

        strict=True: only segments of exactly the requested object (and held object, when given); an
        unknown or excluded name returns [].
        """
        kind = SegmentKind(kind)
        cands = [s for s in self.segments if s.kind == kind]
        name = self._canon_or_self(object_name)
        held = self._canon_or_self(held_name)
        if strict:
            if name is None:
                return []
            cands = [s for s in cands if s.object_name == name and (held is None or s.held_name == held)]
        if not cands:
            return []
        if extent is None and name is not None:
            exts = [s.meta.get("extent") for s in cands if s.object_name == name and s.meta.get("extent") is not None]
            if exts:
                extent = np.median(np.stack(exts), axis=0)
        ext_q = None if extent is None else np.log(np.maximum(np.asarray(extent, float), 0.005))

        def key(s: Segment):
            e = s.meta.get("extent")
            if ext_q is not None and e is not None:
                ed = float(np.linalg.norm(np.log(np.maximum(np.asarray(e, float), 0.005)) - ext_q))
            else:
                ed = 0.0
            return (0 if (name is not None and s.object_name == name) else 1,
                    0 if (relation is None or s.relation == relation) else 1,
                    0 if (held is None or s.held_name == held) else 1,
                    round(ed, 3), len(s.gripper), s.segment_id)

        return sorted(cands, key=key)[:k]

    def object_points(self, name: str, max_points: int = 1500) -> np.ndarray:
        """Merged surface points of an object (object frame) from all visible segments' demos."""
        canon = self._canon_or_self(name)
        if canon is None or canon in self._ex_obj:
            return np.zeros((0, 3))
        if canon in self._points_cache:
            return self._points_cache[canon]
        clouds, seen = [], set()
        for s in self.segments:
            if s.object_name == canon and s.demo_id not in seen and len(s.object_points):
                clouds.append(np.asarray(s.object_points, np.float64))
                seen.add(s.demo_id)
            if len(clouds) >= 10:
                break
        if not clouds:
            P = np.zeros((0, 3))
        else:
            from motormem.memory.replay import voxel_downsample
            P = voxel_downsample(np.concatenate(clouds), max_points)
        self._points_cache[canon] = P
        return P

    # ------------------------------------------------------------ editing
    def without(self, objects=(), demo_ids=()) -> "MotorMemory":
        """Filtered view (no copy of the segment arrays). Objects may be free text or canonical."""
        objs = set()
        known = self._all_names()
        for o in objects:
            objs.add(o)
            raw = o.strip().lower().replace(" ", "_")
            c = raw if raw in known else (None if "_" in o else self._best_name(o))
            if c:
                objs.add(c)
        m = MotorMemory.__new__(MotorMemory)
        m._segments = self._segments
        m._ex_obj = self._ex_obj | frozenset(objs)
        m._ex_demo = self._ex_demo | frozenset(demo_ids)
        m._points_cache = {}
        return m

    def summary(self) -> dict:
        out: dict[str, dict[str, int]] = {}
        for s in self.segments:
            out.setdefault(s.kind.value, {}).setdefault(s.object_name, 0)
            out[s.kind.value][s.object_name] += 1
        return out

    # ------------------------------------------------------------ persistence
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"version": 1, "segments": self.segments}, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    @classmethod
    def load(cls, path) -> "MotorMemory":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["segments"])
