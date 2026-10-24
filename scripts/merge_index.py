"""Merge motor-memory indexes (e.g. LIBERO suites + LIBERO-90 living/study scenes) into one file."""
import sys
from motormem.memory import MotorMemory
out, *srcs = sys.argv[1:]
m = MotorMemory.load(srcs[0])
for p in srcs[1:]:
    m.add_segments(list(MotorMemory.load(p).segments))
m.save(out)
print(out, len(m.segments), "segments")
