import cv2
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import argparse
import os, json

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

def click_to_get_coordinates(pdf_path, page_number=1, poppler_path=None, scale=0.5, box_size=25):
    # Convert PDF to image
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    if page_number < 1 or page_number > len(pages):
        raise ValueError(f"Page {page_number} is out of range. PDF has {len(pages)} pages.")

    # Preprocess and convert to NumPy
    page = preprocess_image(pages[page_number - 1])
    img_np = np.array(page)

    # Resize for display
    display_img = cv2.resize(img_np, None, fx=scale, fy=scale)
    clone = display_img.copy()

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert scaled coordinates back to original
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            print(f"🧭 Clicked at: x={orig_x}, y={orig_y}, w={box_size}, h={box_size}")

            # Draw rectangle for visual feedback
            cv2.rectangle(clone, (x - int(box_size * scale / 2), y - int(box_size * scale / 2)),
                          (x + int(box_size * scale / 2), y + int(box_size * scale / 2)), (0, 255, 0), 2)
            cv2.imshow("PDF Page", clone)

    cv2.namedWindow("PDF Page", cv2.WINDOW_NORMAL)
    cv2.imshow("PDF Page", display_img)
    cv2.setMouseCallback("PDF Page", click_event)
    print("🖱️ Click on a checkbox to get coordinates. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Click to extract checkbox coordinates from PDF")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--page", type=int, default=1, help="Page number (1-based)")
    parser.add_argument("--scale", type=float, default=0.5, help="Display scale factor (0.1–1.0)")
    parser.add_argument("--box_size", type=int, default=25, help="Suggested box size for template")
    parser.add_argument("--poppler", help="Path to poppler bin directory (default from config.json)")
    parser.add_argument("--config", help="Path to config.json (optional)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    poppler = args.poppler or cfg.get("poppler")

    click_to_get_coordinates(args.pdf, page_number=args.page, poppler_path=poppler,
                             scale=args.scale, box_size=args.box_size)

if __name__ == "__main__":
    main()