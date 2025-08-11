# CLI Examples

## Quick start

**Bash**

```bash
python -m ken_burns_reel . --mode panels \
  --bg-mode blur --page-scale 0.94 --bg-parallax 0.85 \
  --profile social
```

**PowerShell**

```powershell
python -m ken_burns_reel . `
  --mode panels `
  --bg-mode blur `
  --page-scale 0.94 `
  --bg-parallax 0.85 `
  --profile social
```

**CMD**

```cmd
python -m ken_burns_reel . --mode panels ^
  --bg-mode blur ^
  --page-scale 0.94 ^
  --bg-parallax 0.85 ^
  --profile social
```

## One-click mode

**PowerShell**

```powershell
python -m ken_burns_reel . --oneclick --limit-items 10 --align-beat --profile preview --aspect 9:16 --height 1080
```

**Bash**

```bash
python -m ken_burns_reel . --oneclick --limit-items 10 --align-beat --profile preview --aspect 9:16 --height 1080
```

## Export panels

```bash
python -m ken_burns_reel input_pages --export-panels panels --export-mode rect
```

## Panel-first mode

**PowerShell**

```powershell
python -m ken_burns_reel .\panels `
  --mode panels-items `
  --trans smear --trans-dur 0.32 --smear-strength 1.1 `
  --bg-mode blur --page-scale 0.92 --bg-parallax 0.85 `
  --profile preview
```

**Bash**

```bash
python -m ken_burns_reel panels --mode panels-items \
  --size 1920x1080 --trans whip --trans-dur 0.28 \
  --profile social
```

## Overlay mode (page + masked panels)

```bash
python -m ken_burns_reel . --mode panels-overlay \
  --overlay-fit 0.75 --bg-source page \
  --parallax-bg 0.85 --parallax-fg 0.08 \
  --travel-ease inout
```

## Formaty i presety

**Bash**

```bash
python -m ken_burns_reel folder --bg-mode blur --profile social
```

**CMD**

```cmd
python -m ken_burns_reel folder --bg-mode blur --profile social
```

**PowerShell**

```powershell
python -m ken_burns_reel folder `
  --bg-mode blur `
  --page-scale 0.92
```

Zalecane `--page-scale` mieści się w zakresie `0.90–0.95`. Wideo można wymiarować przez `--size WxH` albo `--aspect 9:16 --height 1080`.

