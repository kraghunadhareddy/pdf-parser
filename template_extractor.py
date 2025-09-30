from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import argparse
import os
import json

def load_config(config_path=None):
    cfg = {}
    path = config_path or os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    return cfg

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    return image

def extract_template(pdf_path, page_number, x, y, w, h, output_path, poppler_path=None):
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    if page_number < 1 or page_number > len(pages):
        raise ValueError(f"Page {page_number} is out of range. PDF has {len(pages)} pages.")

    page = preprocess_image(pages[page_number - 1])
    cropped = page.crop((x, y, x + w, y + h))
    cropped.save(output_path)
    print(f"✅ Saved template to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract checkbox template from PDF")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--page", type=int, required=True, help="Page number (1-based)")
    parser.add_argument("--x", type=int, required=True, help="X coordinate of top-left corner")
    parser.add_argument("--y", type=int, required=True, help="Y coordinate of top-left corner")
    parser.add_argument("--w", type=int, required=True, help="Width of region")
    parser.add_argument("--h", type=int, required=True, help="Height of region")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--poppler", help="Path to poppler bin directory (default from config.json)")
    parser.add_argument("--config", help="Path to config.json (optional)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    poppler = args.poppler or cfg.get("poppler")

    extract_template(args.pdf, args.page, args.x, args.y, args.w, args.h, args.output, poppler)

if __name__ == "__main__":
    main()