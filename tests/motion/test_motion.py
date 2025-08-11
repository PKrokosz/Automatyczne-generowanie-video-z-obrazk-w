import math
import numpy as np
from ken_burns_reel import motion


def test_arc_path_curvature_positive():
    start = (0.0, 0.0)
    end = (1.0, 0.0)
    pts = np.array([motion.arc_path(start, end, p) for p in np.linspace(0, 1, 5)])
    # approximate curvature via angle differences
    angles = []
    for i in range(len(pts) - 2):
        a, b, c = pts[i], pts[i + 1], pts[i + 2]
        v1 = b - a
        v2 = c - b
        ang = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
        angles.append(ang)
    avg_curv = sum(abs(a) for a in angles) / len(angles)
    assert avg_curv > 0


def test_subtle_drift_deterministic():
    p = 0.5
    z1 = motion.subtle_drift("bg", 123, p)
    z2 = motion.subtle_drift("bg", 123, p)
    assert z1 == z2
    z3 = motion.subtle_drift("bg", 124, p)
    assert z1 != z3
