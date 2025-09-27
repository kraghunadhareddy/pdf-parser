# Copilot Instructions for AI Coding Agents

## What this project does
Parse medical intake PDFs to detect checkboxes and map them to labeled sections. Pipeline: PDF -> images (pdf2image+Poppler) -> preprocess (PIL) -> template match ticked/empty (OpenCV) -> OCR labels (Tesseract) -> find section regions -> align labels to nearest checkbox -> annotate debug images -> write pruned JSON.

## Key files
- `extractor.py`: Main logic (`CheckboxExtractor`). CLI flags: `--pdf`, `--sections`, `--output`, `--ticked_template`, `--empty_template`, `--poppler`, `--threshold`.
- `extract_tick_coordinates.py`: Click to collect checkbox coords on rendered pages (template setup/debugging).
- `template_extractor.py`: Crop checkbox templates from PDFs given coords (produces `ticked.png`, `unticked.png`).
- `sections.json`: Array of `{ section_name, labels[] }` used as OCR targets and section anchors.

## Environment & dependencies
- Windows defaults: Tesseract path hardcoded at top of `extractor.py` and Poppler default provided by `--poppler` CLI. Override via CLI or edit the constant if paths differ.
- Requires: `pdf2image`, `pytesseract`, `opencv-python`, `Pillow`, `numpy`. Images rendered at 300 DPI; all coordinates are pixel-based at that DPI.

## Core conventions the code relies on
- Templates: two grayscale PNGs (same size) for ticked and empty; default match threshold `--threshold=0.6`. Template dedupe uses `max_dist=5` on top-left proximity.
- Checkbox statuses are one of: `ticked`, `empty`, `missing` (only non-`missing` are written to output).
- Label text normalization for matching: remove control chars, spaces, slashes, dashes; map I/l/L/i/1 -> L; uppercase. Multi-line label matching uses row lookahead with x-alignment tolerance ≈50 px.
- Section detection: normalize section header text, find its OCR line, set region from that y down to the last nearby checkbox (max gap ≈100 px) with a small buffer; region is roughly x:[0,2000].
- Row clustering groups checkboxes by y with gap ≈50 px; label-to-checkbox assignment prefers the nearest box in the same row if delta-y ≤ 60 px.
- Debug annotations: blue rectangles for section bounds; checkbox rectangles colored green=ticked, red=empty, yellow=missing; arrows from matched label positions to boxes. Files: `debug_page_<N>.png`.
- Output JSON (after pruning): `{"pages":[{"page_number", "sections":[{"section","checkboxes":[{"label","status","score","confidence","position":[x,y,w,h]}]}]}]}`. Pruning rules: drop `missing` checkboxes, drop sections with none left, de-duplicate sections per page, drop empty pages.

## Typical workflows
1) Template setup: run `extract_tick_coordinates.py` to capture coords -> run `template_extractor.py` to crop `ticked.png` and `unticked.png` that visually match the form style.
2) Configure sections: update `sections.json` with `section_name` headers as they appear on the form and the exact label phrases to target (normalization makes matching resilient to minor OCR noise and line breaks).
3) Extract: run `extractor.py --pdf <file.pdf> --sections sections.json --ticked_template ticked.png --empty_template unticked.png [--output output.json] [--poppler <path>] [--threshold 0.6]`. Inspect `debug_page_*.png` and console logs (`[CHECKBOX POSITIONS]`, `[LABEL POSITIONS]`, `[ANCHOR TEXT]`, `[MATCH]`).

## When extending or debugging
- If labels fail to match, inspect logged OCR tokens and normalization; consider adjusting the label text in `sections.json` to include more of the visible phrase (multi-line is handled).
- If section bounds are wrong, check the anchor header text exactly as OCR sees it and verify checkboxes exist below it; tune gaps in code only if necessary.
- If checkboxes mis-assign across rows, review row gap (≈50) and allowed delta-y (≤60) in `assign_checkboxes_sectionwise`.

Questions or gaps? Tell us which step (templates, OCR labels, section anchors, or assignment) is unclear and we’ll refine these rules.