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
