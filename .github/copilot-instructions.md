# Copilot Instructions for AI Coding Agents

## What this project does
Extract checkbox selections from medical intake PDFs and map them to labeled sections. Flow: PDF → images (pdf2image + Poppler) → preprocess (PIL) → template match ticked/empty (OpenCV) → OCR labels (Tesseract) → detect section regions → align labels to nearest checkbox → annotate debug images → emit pruned JSON.

## Key files and entry points
- `extractor.py`: Main pipeline (`CheckboxExtractor`). Flags: `--pdf`, `--sections`, `--output`, `--ticked_template`, `--empty_template`, `--poppler`, `--threshold`.
- `extract_tick_coordinates.py`: Click on rendered pages to record checkbox coords (for template setup/debugging).
- `template_extractor.py`: Crop checkbox templates from a PDF given coords; outputs `ticked.png`/`unticked.png`.
- `sections.json`: Array of `{ section_name, labels[] }` used for section anchors and expected label texts.

## Environment and assumptions (Windows-first)
- Tesseract path is hardcoded in `extractor.py` (pytesseract.tesseract_cmd). Override by editing that constant or letting system PATH resolve it.
- Poppler path defaults via `--poppler` in all scripts. Pass the Poppler bin directory path when needed.
- Dependencies: `pdf2image`, `pytesseract`, `opencv-python`, `Pillow`, `numpy`. Images render at 300 DPI; all coords assume 300 DPI.
- A venv exists at `pdf-env/`; prefer `pdf-env\Scripts\python.exe` when running scripts locally.

## Core conventions this code relies on
- Templates: two same-sized grayscale PNGs for ticked and empty; default match threshold `--threshold=0.6`. Dedupe by top-left proximity with `max_dist=5`.
- Status values: `ticked`, `empty`, `missing` (only non-`missing` survive pruning/output).
- Label matching normalization: remove control chars, spaces, slashes, dashes; preserve case. Flexible I/L/l/1 matching is enabled only where the expected label contains uppercase `I`. Multi-line matching uses lookahead across up to 5 lines with x-alignment tolerance ≈80 px.
- Section detection: normalize header text similarly; anchor at first matching OCR line; extend region down while checkboxes appear with ≤100 px vertical gaps; x-range is [0, 2000].
- Row clustering: group by y with gap ≈50 px; label→checkbox assignment keeps same-row preference and requires Δy ≤ 60 px.
- Debug artifacts: `debug_page_<N>.png` with blue section boxes; checkbox boxes colored green=ticked, red=empty, yellow=missing; arrows from label origins to assigned boxes.

## Output shape and pruning
`{"pages":[{"page_number":N,"sections":[{"section":"Name","checkboxes":[{"label","status","score","confidence","position":[x,y,w,h]}]}]}]}`.
- Pruning: drop `missing` checkboxes; drop sections with none left; skip empty pages. Sections are emitted once per page as processed.

## Typical workflows
1) Template setup: run `extract_tick_coordinates.py` to capture coords → run `template_extractor.py` to crop `ticked.png`/`unticked.png` that visually match the form.
2) Configure sections: update `sections.json` with visible section headers and exact label phrases (normalization and multi-line lookahead are already handled).
3) Extract: run `extractor.py --pdf <file.pdf> --sections sections.json --ticked_template ticked.png --empty_template unticked.png [--output output.json] [--poppler <path>] [--threshold 0.6]`. Inspect `debug_page_*.png` and logs.

## Logs to watch when debugging
`[CHECKBOX POSITIONS]`, `[LABEL POSITIONS]`, `[MULTILINE LOOKAHEAD]`, `[ANCHOR TEXT]`, `[MATCH]`, `[LABEL -> CHECKBOX]`, `[MISS]`, `[SKIP]`, `[WARN]`.

## Repo-specific gotchas for agents
- `get_label_positions` contains a duplicate, unreachable block after a `return`; rely on the first implementation (with flexible I/L/1 and multi-line lookahead).
- All geometric thresholds (template threshold, row gap 50, Δy 60, section gap 100, x_tolerance 80, lookahead 5) are centralized in method defaults—tune there when adapting to new forms.