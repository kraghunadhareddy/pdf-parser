# PDF Medical Intake Form Parser

This project extracts checkbox, medication, supplement, and lifestyle information from medical intake PDF forms using image processing and OCR.

## Prerequisites
- Python 3.8+
- Poppler for Windows (for PDF conversion)
- Tesseract OCR
- Python packages (see `requirements.txt`): `pdf2image`, `pytesseract`, `opencv-python`, `Pillow`, `numpy`
- Checkbox template images: `ticked.png`, `unticked.png`

## Scripts Overview

### 1. `extractor.py`
**Purpose:** Main extraction logic for checkboxes and labels from PDF forms.

**Usage:**
```sh
python extractor.py --pdf <PDF_FILE> [--output <OUTPUT_JSON>] [--sections <SECTIONS_JSON>] [--ticked_template <TICKED_IMG>] [--empty_template <EMPTY_IMG>] [--poppler <POPPLER_PATH>] [--threshold <MATCH_THRESHOLD>] [--config <CONFIG_JSON>]
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
4. Configure defaults in `config.json`.
5. Run `extractor.py` to process the PDF and output results to `output.json`.

### Install dependencies
```powershell
.\u005cpdf-envScripts\Activate.ps1
pip install -r requirements.txt
```

### Check your environment
```powershell
python extractor.py --pdf test-intake-form.pdf --check
```
This validates Python packages and prints the resolved Tesseract/Poppler paths.

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

## Config
Define project defaults in `config.json` at the repo root:
```json
{
	"sections": "sections.json",
	"ticked_template": "ticked.png",
	"empty_template": "unticked.png",
	"poppler": "C:\\Path\\To\\poppler\\Library\\bin",
	"threshold": 0.6,
	"tesseract": "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
}
```
CLI flags override config values. Use `--config` to point at an alternate file.

## Programmatic API
You can call the extractor directly from Python using the built-in helper:
```python
from extractor import run_extractor_from_config

data = run_extractor_from_config(
		pdf_path=r"C:\\Path\\To\\test-intake-form.pdf",
		output_path=r"C:\\Path\\To\\output.json",          # optional
		config_path=None,                                        # optional, defaults to repo config.json
		threshold=0.6                                            # optional override
)
```
This loads defaults from `config.json` and lets you override any value via keyword args.
