# Copilot Instructions for AI Coding Agents

## Project Overview
This project parses medical intake PDF forms to extract checkbox, medication, supplement, and lifestyle information. It uses image processing and OCR to identify checked/unchecked boxes and extract structured data.

### Key Components
- `extractor.py`: Main logic for extracting checkbox states from PDF forms using template matching (OpenCV) and OCR (pytesseract). Defines `CheckboxExtractor` class.
- `extract_tick_coordinates.py`: Interactive tool to click on PDF images and get coordinates for checkbox regions. Useful for template setup and debugging.
- `template_extractor.py`: Extracts image templates for checkboxes from PDFs based on coordinates. Used to generate reference images for matching.
- `sections.json`: Defines form sections and expected labels for extraction. Update this file to support new form types or label sets.

## Developer Workflows
- **Template Setup**: Use `extract_tick_coordinates.py` to find checkbox coordinates, then `template_extractor.py` to save template images (`ticked.png`, `unticked.png`).
- **Extraction**: Run `extractor.py` with correct template paths and poppler/tesseract configuration to process PDFs.
- **Configuration**: Update `sections.json` for new forms or label changes. Ensure template images match the form style.
- **Debugging**: Use debug images (`debug_page_*.png`) and logs (`log.txt`, `page 3 log.txt`) to troubleshoot extraction issues.

## Conventions & Patterns
- All image processing uses PIL and OpenCV; templates are grayscale PNGs.
- Poppler and Tesseract paths are hardcoded for Windows; update as needed for other OSes.
- Extraction is page-based; coordinates are 0-indexed and must match PDF DPI (default 300).
- Output is written to `output.json` in structured format.
- Use argparse for CLI tools; see each script for required arguments.

## Integration Points
- External dependencies: `pdf2image`, `pytesseract`, `opencv-python`, `Pillow`, `numpy`.
- Poppler binaries required for PDF conversion (Windows path set by default).
- Tesseract OCR must be installed and path set in `extractor.py`.

## Example Workflow
1. Use `extract_tick_coordinates.py` to interactively find checkbox coordinates on a sample PDF.
2. Use `template_extractor.py` to crop and save checkbox templates.
3. Update `sections.json` with section names and labels matching the form.
4. Run `extractor.py` to process the PDF and output results to `output.json`.

## Key Files
- `extractor.py`, `extract_tick_coordinates.py`, `template_extractor.py`, `sections.json`, `ticked.png`, `unticked.png`

---
If any section is unclear or missing, please provide feedback or specify which workflow or convention needs more detail.