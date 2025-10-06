"""
Project-wide constants used across extractor and response logic.

All geometric thresholds are defined here to avoid magic numbers scattered
throughout the codebase.
"""

# Minimum vertical offset (in pixels at 300 DPI) to start searching for
# labels/questions after a section anchor. Prevents consuming the header line.
ANCHOR_OFFSET_PX: int = 40

# Rendering / DPI
RENDER_DPI: int = 300  # pdf2image render DPI and pypdfium2 fallback scale baseline

# Template matching / detection
TEMPLATE_THRESHOLD_DEFAULT: float = 0.6  # default; can be overridden by CLI/config
DEDUPE_MAX_DIST: int = 5  # pixels; to dedupe overlapping template hits by top-left proximity

# Label/question multiline matcher tolerances
LABEL_MULTILINE_BASE_X_TOLERANCE: int = 160  # px tolerance for x alignment across wrapped lines
LABEL_MULTILINE_MAX_LOOKAHEAD: int = 5       # number of subsequent lines to search for continuation

# Checkbox row clustering and alignment
LABEL_ROW_GAP_PX: int = 50                  # group checkboxes into rows by Y gap
LABEL_CB_ASSIGN_MAX_DELTA_Y: int = 60       # max allowed delta Y between label and checkbox row center

# Section detection and geometry
SECTION_CB_MAX_GAP_PX: int = 100            # grow section down while checkbox gap <= this value
SECTION_X1: int = 0                         # section region left bound for debug/annotation
SECTION_X2: int = 2000                      # section region right bound for debug/annotation
SECTION_BOTTOM_BUFFER_PX: int = 50          # small buffer added below last checkbox within section

# Answer extraction parameters
ANSWER_COL_GAP_PX: int = 5                  # horizontal gap (px) before next question's start x used as right bound
ANSWER_PAGE_RIGHT_MARGIN_PX: int = 20       # right margin (px) when no next question on the same row
ANSWER_ROW_Y_TOLERANCE_PX: int = 12         # y tolerance (px) to treat questions as belonging to the same row
ANSWER_MIN_LINE_HEIGHT_PX: int = 6          # ignore OCR noise lines below this height
ANSWER_MAX_VERTICAL_GAP_PX: int = 1_000     # hard stop safety to avoid runaway scans (very large pages)
ANSWER_STOP_ON_BLANK: bool = True           # stop collecting when a blank line encountered after starting
ANSWER_LEFT_MARGIN_PX: int = 30             # allow capturing words slightly left of question x_start in answers
ANSWER_BLANK_LINE_GAP_PX: int = 45          # if vertical gap between consecutive captured answer lines exceeds this, treat as blank separator

# Answer continuation (multi-line) tuning
# After capturing the first answer line inside the fixed band, additional wrapped lines are
# accepted if their baseline y satisfies: y > (first_line_y + ANSWER_CONTINUATION_MIN_DELTA_Y)
# and y <= (first_line_y + ANSWER_CONTINUATION_MAX_DELTA_Y). Raising MAX helps narrative
# answers (e.g., family history medical conditions) that wrap slightly further down while
# still remaining well above the next row's question band.
ANSWER_CONTINUATION_MIN_DELTA_Y: int = 10   # minimum delta (px) above first answer baseline before accepting continuation
ANSWER_CONTINUATION_MAX_DELTA_Y: int = 50   # maximum delta (px) for continuation capture per spec (restore strict 10..50 window)

# Yes/No highlighted answer recovery tuning
YESNO_SLIDE_OFFSETS: list[int] = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]
YESNO_PROBE_MAX_BANDS: int = 2              # number of downward same-height bands to probe after base window
YESNO_HIGHLIGHT_CONFIDENCE: float = 0.95    # confidence assigned to direct highlight recovery
YESNO_INFERRED_CONFIDENCE: float = 0.80     # confidence when inferred from follow-up logic

# Global OCR tuning (generic, not field-specific)
OCR_PSM: int = 6                            # Tesseract page segmentation mode for block of text
OCR_LANG: str | None = None                 # Set to language code string if needed (e.g., 'eng')
OCR_ENABLE_PREPROCESS: bool = True          # Enable pre-OCR enhancement pass
PREPROC_UNSHARP_RADIUS: float = 1.2         # Unsharp mask radius (Pillow / manual kernel equivalent)
PREPROC_UNSHARP_AMOUNT: float = 1.3         # Unsharp mask amount multiplier
PREPROC_CLAHE: bool = True                  # Apply CLAHE (adaptive contrast) if OpenCV available
PREPROC_CLAHE_CLIP: float = 2.0             # CLAHE clip limit
PREPROC_CLAHE_TILE: int = 8                 # CLAHE tile grid size

# Debug toggles
DEBUG_ANSWER_GEOMETRY: bool = True          # TEMP enable for diagnostics; set back to False after analysis
