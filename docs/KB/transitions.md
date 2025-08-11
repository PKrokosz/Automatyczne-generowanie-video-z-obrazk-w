# Transitions

The `ken_burns_reel/transitions.py` module implements several transitions. Notable is `fg_fade` which fades only the foreground mask while leaving the background static.

```python
from ken_burns_reel.transitions import fg_fade
```

The CLI enforces a minimum panel exposure of 1400 ms via the `--readability-ms` option; passing `--validate` or `--deterministic` does not reduce this guard.
