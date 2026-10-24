"""Stream C: motor memory (demo replay, segmentation into skills, object-frame index)."""
from motormem.memory.index import MotorMemory  # noqa: F401


def build_index(sources, out, max_demos_per_task=None, **kw):
    from motormem.memory.build import build_index as _b
    return _b(sources, out, max_demos_per_task, **kw)
