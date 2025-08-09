# AGENTS.md — Sprint 0: Hardening & Hygiene

> **Repo:** `PKrokosz/Automatyczne-generowanie-video-z-obrazk-w`\
> **Branch:** `codex/hardening-hygiene` → PR to `main`

This file is a working playbook for autonomous agents and human contributors in this repository. It specifies scope/precedence, environment setup, external binary resolution, coding conventions, programmatic checks, and the exact Sprint 0 tasks to complete.

---

## 1. Scope & Precedence

- Applies to the **entire repository**, unless a deeper directory contains its own `AGENTS.md` that overrides these rules for that subtree.
- If a task prompt conflicts with this document, **the task prompt wins**.
- If two `AGENTS.md` files conflict, **the deeper path wins**.
- Always run the **programmatic checks** defined here (tests, smoke builds) before declaring a task done.

## 2. Environment & Tools

**Language:** Python 3.10–3.12

**Local setup:**

```bash
python -m venv .venv
. .venv/bin/activate     # PowerShell: .venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .[dev] || pip install -r requirements.txt -r requirements-dev.txt
```

**External binaries (not vendored):**

- **ImageMagick** (`magick` CLI) — used for caption rendering.
- **Tesseract OCR** (`tesseract` CLI) — used for OCR paths.

**Binary resolution policy (must be implemented and used in code):** Create `ken_burns_reel/bin_config.py` exposing:

- `resolve_imagemagick()` — precedence:
  1. CLI flag `--magick`
  2. Env `IMAGEMAGICK_BINARY`
  3. MoviePy config value (if present)
  4. `shutil.which("magick")`
- `resolve_tesseract()` — precedence:
  1. CLI flag `--tesseract`
  2. Env `TESSERACT_BINARY`
  3. `pytesseract.pytesseract.tesseract_cmd`
  4. `shutil.which("tesseract")`
- **Validation:** resolvers must verify the binary exists and is executable. If not found, raise a friendly `EnvironmentError` with actionable hints.
- **Runtime usage:** Entry points (e.g., `__main__.py`) must call these resolvers instead of hard-coding absolute paths. Keep support for the existing `--magick` and `--tesseract` flags (flags override discovery).

**Windows notes:**

- Prefer the unified `magick` wrapper over legacy `convert`.
- Ensure Tesseract’s install directory is on `PATH` (e.g., `C:\Program Files\Tesseract-OCR`).

## 3. Troubleshooting

- **Tesseract not found:** set `TESSERACT_BINARY` or add the install directory to `PATH`.
- **ImageMagick label errors:** confirm `magick` exists and security policy permits `label:`; use the `magick` wrapper on Windows.
- **MoviePy warnings about ImageMagick:** typically indicate missing/blocked IM — verify resolver output and permissions.

## 4. Required Commands

**Install:**

```bash
pip install -e .[dev] || pip install -r requirements.txt -r requirements-dev.txt
```

**Run tests:**

```bash
pytest -q
```

**Optional lints (don’t fail the sprint if tools are absent):**

```bash
ruff check . || true
mypy . || true
```

**Smoke build (example):**

```bash
python -m ken_burns_reel . --dry-run || true
```

## 5. Coding Conventions

- **No hard-coded absolute paths** to external binaries.
- **Context-manage images:**
  ```python
  from PIL import Image
  with Image.open(path) as im:
      img = im.convert("RGB")
  ```
- **Guard divisions:** in `detect_focus_point`, protect denominators with `eps = 1e-6` and `den = max(eps, den)`.
- **Graceful degradation:** If captions require ImageMagick and it’s unavailable, **do not crash**—log a warning and skip captions for that clip.
- Prefer PEP 8, type hints in new/modified code, and small, atomic commits.

## 6. Programmatic Checks (must pass before finishing)

1. **Unit tests pass** locally: `pytest -q`.
2. **Video smoke test** using synthetic frames produces a non-empty file.
3. **No **`` for unclosed files in test output.
4. Tests that require external binaries are **skipped** with `@pytest.mark.skipif` when binaries are unresolved.

## 7. Sprint 0 — Tasks (authoritative list)

### 7.1 — Binary config module

- Add `ken_burns_reel/bin_config.py` with resolvers above; wire into entry points; keep `--magick`/`--tesseract` overrides.
- **Acceptance:** `python -m ken_burns_reel .` works when binaries are discoverable via PATH/ENV; flags still override.

### 7.2 — Captions dedup

- Create `ken_burns_reel/captions.py` with:
  - `render_caption_clip(text, size, margin=50, method="label")`
  - `overlay_caption(clip, text, size, ...)`
- Migrate caption logic from `ken_burns_scroll_audio.py` and `utils.py`. If IM is missing, log a clear warning and **skip** captions; keep the pipeline running.
- **Acceptance:** One authoritative captions module; all entry points consume it.

### 7.3 — Focus guard

- Add epsilon guard for zero-brightness frames; if uniformly dark, **return frame center** and log a warning.
- **Acceptance:** no `ZeroDivisionError`; black frames still render.

### 7.4 — Context-manage images

- Replace raw `Image.open(path)` with context-managed form everywhere; avoid keeping global PIL objects.
- **Acceptance:** no `ResourceWarning: unclosed file` during tests.

### 7.5 — Tests

- Add/extend under `tests/`:
  - `test_builder_sequence_basic.py` — create synthetic frames, run builder, assert output exists and size > 0 (use `tmp_path`).
  - `test_transitions.py` — compose two clips + crossfade; assert duration and no exceptions.
  - `test_focus_guard.py` — pure black frame → returns center, no exception.
- Use `@pytest.mark.skipif` when resolvers cannot find IM/Tesseract.
- **Acceptance:** `pytest -q` passes; IM/TE-dependent tests skip cleanly if binaries are missing.

### 7.6 — README

- Add a **Tech/Setup** section explaining: binary resolution order, env vars, CLI flags, Windows notes, and quick-start/test commands.

## 8. Git & PR Conventions

- **Branch:** `codex/hardening-hygiene`.
- **Commits:** small and scoped (ideally one task per commit) with meaningful messages.
- **PR to **``**:** include:
  - a short per-task summary,
  - test run snippet (e.g., `pytest -q` with any skips),
  - notes on behavioral changes (e.g., captions skipped when IM missing).

## 9. Agent Operating Rules

- Make changes and **commit** your patch; keep the working tree clean (no stray files).
- **Do not** amend commits authored by others.
- Respect this `AGENTS.md` in any path you modify.
- Before finishing, run the **Programmatic Checks** and paste a brief verification log into the PR description.

---

**Definition of Done (Sprint 0)**

- `python -m ken_burns_reel .` works on systems with PATH/ENV-configured binaries (no hard-coded paths).
- Single captions module with graceful fallback.
- `detect_focus_point` epsilon guard in place; no division-by-zero.
- All images opened via context managers; no unclosed file warnings.
- New tests added and passing/skipping correctly.
- README updated with clear binary configuration and setup.

