# LOOK

Reference visualisations used in documentation are generated locally and saved under `artifacts/` (ignored by git).

## Arc vs Linear motion

Generate a comparison plot for camera paths:

```bash
python - <<'PY'
import numpy as np, matplotlib.pyplot as plt
from ken_burns_reel import motion

p = np.linspace(0, 1, 100)
arc = np.array([motion.arc_path((0, 0), (1, 0), t) for t in p])
plt.plot(p, np.zeros_like(p), label="linear")
plt.plot(p, arc[:, 1], label="arc")
plt.legend()
plt.savefig("artifacts/arc_vs_linear.png")
PY
```

## overlay_lift transition frames

Golden frames for `overlay_lift` can be generated via tests:

```bash
pytest tests/test_overlay_lift.py --look
```

All outputs will appear in the local `artifacts/` directory.

## Bubble overlay

The bubble layer ensures speech balloons render above panels without being clipped.
Use tests to generate reference frames:

```bash
pytest tests/test_bubbles.py --look
```
