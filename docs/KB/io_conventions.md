# I/O Conventions

Output video files follow a human-readable naming scheme. The `--out-naming` flag controls behaviour:

| Mode | Description |
|------|-------------|
| `auto` (default) | `<out-prefix><input-folder>-<mode>_<timestamp>.mp4` in the target directory |
| `keep` | legacy behaviour using provided `--output` path or `final_video.mp4` |
| `custom` | `<out-prefix>_<timestamp>.mp4` |

Use `--out-prefix` to prepend a slug.
