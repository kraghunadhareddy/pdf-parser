# PDF Medical Intake Form Parser

This project extracts checkbox, medication, supplement, and lifestyle information from medical intake PDF forms using image processing and OCR.

## Prerequisites
- Python 3.8+
- Poppler for Windows (for PDF conversion)
- Tesseract OCR
- Required Python packages: `pdf2image`, `pytesseract`, `opencv-python`, `Pillow`, `numpy`
- Checkbox template images: `ticked.png`, `unticked.png`

## Scripts Overview

### 1. `extractor.py`
**Purpose:** Main extraction logic for checkboxes and labels from PDF forms.

**Usage:**
```sh
python extractor.py --pdf <PDF_FILE> --sections <SECTIONS_JSON> --ticked_template <TICKED_IMG> --empty_template <EMPTY_IMG> [--output <OUTPUT_JSON>] [--poppler <POPPLER_PATH>] [--threshold <MATCH_THRESHOLD>]
```
- Extracts structured data to `output.json`.
- Debug images are saved as `debug_page_<N>.png`.
- Logs are printed to console and can be redirected to `log.txt`.

### 2. `extract_tick_coordinates.py`
**Purpose:** Interactive tool to click on PDF images and get coordinates for checkbox regions.

**Usage:**
```sh
python extract_tick_coordinates.py --pdf <PDF_FILE> [--poppler <POPPLER_PATH>]
```
- Click on checkboxes in the displayed image to record coordinates.
- Useful for template setup and debugging.

### 3. `template_extractor.py`
**Purpose:** Extracts image templates for checkboxes from PDFs based on coordinates.

**Usage:**
```sh
python template_extractor.py --pdf <PDF_FILE> --coords <COORDS_FILE> --output <OUTPUT_IMG>
```
- Crops and saves checkbox template images (`ticked.png`, `unticked.png`).
- Coordinates file is generated from `extract_tick_coordinates.py`.

### 4. `sections.json`
**Purpose:** Defines form sections and expected labels for extraction.

- Update this file to support new form types or label sets.
- Each section must have a `section_name` and a list of `labels`.

## Workflow Example
1. Use `extract_tick_coordinates.py` to interactively find checkbox coordinates on a sample PDF.
2. Use `template_extractor.py` to crop and save checkbox templates.
3. Update `sections.json` with section names and labels matching the form.
4. Run `extractor.py` to process the PDF and output results to `output.json`.

## Output
- Results are written to `output.json` in structured format.
- Debug images: `debug_page_<N>.png`
- Logs: `log.txt`, `page <N> log.txt`

## Troubleshooting
- Ensure Poppler and Tesseract paths are set correctly for your OS.
- Use debug images and logs to troubleshoot extraction issues.
- Update `sections.json` and template images if form style changes.

## License
MIT
