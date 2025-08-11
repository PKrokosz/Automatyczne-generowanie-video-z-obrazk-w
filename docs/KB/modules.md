# Modules
### `color`
- `srgb_to_linear16`: Convert 16‑bit sRGB values to linear light floats.
- `linear16_to_srgb`: Convert linear light floats back to 16‑bit sRGB values.
### `transitions`
- `ease_in_out`: Cosine ease-in-out for ``t`` in [0,1].
- `ease_in`: Cosine ease-in for ``t`` in [0,1].
- `ease_out`: Cosine ease-out for ``t`` in [0,1].
- `slide_transition`: Simple horizontal slide transition between clips.
- `smear_transition`: Directional smear (pseudo motion blur) between clips.
- `whip_pan_transition`: Whip-pan style transition with easing and brightness dip.
- `fg_fade`: Fade only the foreground alpha of ``panel_clip``.
- `smear_bg_crossfade_fg`: Smear transition for backgrounds with foreground crossfade.
- `overlay_lift`: Lift-in effect for overlay panels.
### `utils`
- `smart_crop`: Crop image to target aspect ratio while keeping center.
- `gaussian_blur`: Apply Gaussian blur with validated kernel parameters.
### `audio`
- `extract_beats`: Return beat times for an audio file.
### `__init__`
- `make_filmstrip`: 
### `builder`
- `ease_in_out`: Cosine ease-in-out for t in [0,1].
- `ease_in`: Cosine ease-in for t in [0,1].
- `ease_out`: Cosine ease-out for t in [0,1].
- `apply_clahe_rgb`: 
- `enhance_panel`: Enhance RGBA panel; process RGB and keep alpha.
- `make_panels_cam_clip`: Animate camera between comic panels detected in the image.
- `make_panels_cam_sequence`: Buduje jeden film, sklejając panel-camera clippy dla wszystkich stron.
- `make_panels_items_sequence`: Build a sequence from pre-cropped panel images.
- `compute_segment_timing`: 
- `make_panels_overlay_sequence`: Render overlay sequence with static foreground panels.
- `ken_burns_scroll`: Create a single Ken Burns style clip.
- `make_filmstrip`: Build final video from assets in *input_folder*.
### `panels`
- `alpha_bbox`: Return bounding box of non-zero alpha in RGBA array.
- `fill_holes`: Fill holes inside a binary mask using flood fill from the border.
- `roughen_alpha`: Add small irregularities to the mask edge.
- `detect_panels`: 
- `order_panels_lr_tb`: 
- `export_panels`: Detect panels in *image_path* and export them to *out_dir*.
- `debug_detect_panels`: 
### `ocr`
- `extract_caption`: Extract text from an image using pytesseract.
- `verify_tesseract_available`: Ensure tesseract binary is available or raise an error.
- `page_ocr_data`: Return raw OCR data for an entire page.
- `text_boxes_stats`: Return basic stats about OCR-detected word boxes in *img*.
### `cli`
- `build_parser`: 
- `validate_args`: 
- `main`: 
### `motion`
- `arc_path`: Interpolate between ``start`` and ``end`` along an arc.
- `DriftParams`: 
- `subtle_drift`: Return deterministic subtle drift parameters.
- `apply_transform`: Apply zoom/translation/rotation drift to ``img``.
### `layers`
- `page_shadow`: Apply drop shadow to ``img`` and return RGBA with premultiplied alpha.
- `shadow_cache_stats`: Return ``(hits, misses)`` for page shadow cache.
### `__main__`
- `parse_args`: 
- `main`: 
### `captions`
- `sanitize_caption`: 
- `is_caption_meaningful`: 
- `render_caption_clip`: Create a TextClip for *text* or return ``None`` if rendering fails.
- `overlay_caption`: Overlay *text* caption onto *clip* if possible.
### `focus`
- `detect_focus_point`: Detect a point of interest in an image.
### `bin_config`
- `resolve_imagemagick`: Resolve path to ImageMagick ``magick`` executable.
- `resolve_tesseract`: Resolve path to the Tesseract binary.

## Cross-imports