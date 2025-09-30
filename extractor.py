import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import json
import os
import argparse
from difflib import SequenceMatcher
from collections import defaultdict
import pprint
import re

def load_config(config_path=None):
    cfg = {}
    base_dir = os.path.dirname(__file__)
    path = config_path or os.path.join(base_dir, "config.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            base_dir = os.path.dirname(os.path.abspath(path))
        except Exception:
            cfg = {}
    return cfg, base_dir

def resolve_path(p, base_dir):
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(base_dir, p))

CONFIG, CONFIG_BASEDIR = load_config()

# Configure Tesseract if provided in config; otherwise rely on system PATH or prior configuration
if CONFIG.get("tesseract"):
    pytesseract.pytesseract.tesseract_cmd = resolve_path(CONFIG["tesseract"], CONFIG_BASEDIR)

class CheckboxExtractor:
    def __init__(self, poppler_path=None, ticked_template_path=None, empty_template_path=None, match_threshold=0.6):
        self.poppler_path = poppler_path
        self.match_threshold = match_threshold
        self.ticked_template = cv2.imread(ticked_template_path, cv2.IMREAD_GRAYSCALE)
        self.empty_template = cv2.imread(empty_template_path, cv2.IMREAD_GRAYSCALE)
        if self.ticked_template is None or self.empty_template is None:
            raise ValueError("❌ Failed to load one or both template images.")
        self.template_size = self.ticked_template.shape[::-1]

    def preprocess_image(self, image):
        image = image.convert("RGB")
        image = image.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        return image

    def match_template(self, image_gray, template, label):
        result = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= self.match_threshold)
        matches = []
        for pt in zip(*locations[::-1]):
            x, y = pt
            score = float(result[y, x])
            matches.append({
                "position": [int(x), int(y), self.template_size[0], self.template_size[1]],
                "score": round(score, 2),
                "label": label,
                "status": label,
                "confidence": round(score, 2)
            })
        return matches

    def deduplicate_matches(self, ticked_matches, empty_matches, max_dist=5):
        all_boxes = []

        # Tag status
        for box in ticked_matches:
            box["status"] = "ticked"
            all_boxes.append(box)
        for box in empty_matches:
            box["status"] = "empty"
            all_boxes.append(box)

        # Sort by confidence descending
        all_boxes.sort(key=lambda b: b["confidence"], reverse=True)

        deduped = []
        for box in all_boxes:
            x, y, w, h = box["position"]
            is_duplicate = False
            for existing in deduped:
                ex, ey, ew, eh = existing["position"]
                if abs(x - ex) < max_dist and abs(y - ey) < max_dist:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduped.append(box)

        return sorted(deduped, key=lambda b: (b["position"][1], b["position"][0]))

    def detect_checkboxes(self, pil_image):
        img = np.array(pil_image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ticked_matches = self.match_template(gray, self.ticked_template, "ticked")
        empty_matches = self.match_template(gray, self.empty_template, "empty")
        boxes = self.deduplicate_matches(ticked_matches, empty_matches)
        
        print("[CHECKBOX POSITIONS]")
        for i, box in enumerate(boxes):
            print(f"  Checkbox {i}: {box}, type={type(box)}")

        print(f"Template matches: {len(boxes)} checkboxes detected")
        return boxes, img

    def get_label_positions(self, pil_image, expected_labels, match_threshold=0.8):
        import unicodedata
        import re
        ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        label_positions = defaultdict(list)

        def normalize_text(text):
            text = unicodedata.normalize('NFKD', text)
            text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')
            text = text.replace('/', '').replace(' ', '').replace('-', '')
            text = re.sub(r'^[^a-zA-Z]+', '', text)
            text = re.sub(r'[^a-zA-Z]+$', '', text)
            return text

        def clean_label_sequence(seq):
            normed = [normalize_text(s) for s in seq]
            joined = ''.join(normed)
            joined = unicodedata.normalize('NFKD', joined)
            joined = ''.join(c for c in joined if unicodedata.category(c)[0] != 'C')
            joined = joined.replace(' ', '')
            return joined

        # Helpers for conditional I/L/l/1 matching based on expected having uppercase 'I'
        def build_expected_masked_upper(text):
            t = unicodedata.normalize('NFKD', text)
            t = ''.join(c for c in t if unicodedata.category(c)[0] != 'C')
            t = t.replace('/', '').replace(' ', '').replace('-', '')
            expected_s = []
            i_mask = set()
            for idx, ch in enumerate(t):
                if ch == 'I':  # only uppercase I in expected triggers flexibility
                    i_mask.add(idx)
                expected_s.append(ch)
            return ''.join(expected_s), i_mask

        def flex_equal(expected_s, i_mask, candidate_s):
            if len(expected_s) != len(candidate_s):
                return False
            for i, (e, c) in enumerate(zip(expected_s, candidate_s)):
                if i in i_mask and e == 'I':
                    if c not in ('I', 'L', 'l', '1'):
                        return False
                else:
                    if e != c:
                        return False
            return True

        def flex_contains(expected_s, i_mask, haystack_s):
            m, n = len(expected_s), len(haystack_s)
            if m == 0:
                return True
            for i in range(0, n - m + 1):
                if flex_equal(expected_s, i_mask, haystack_s[i:i+m]):
                    return True
            return False

        # Build tokens for single-line matching
        tokens = []
        for i in range(len(ocr_data["text"])):
            word = ocr_data["text"][i].strip()
            if not word:
                continue
            norm = normalize_text(word)
            tokens.append({
                "text": norm,
                "orig": word,
                "x": ocr_data["left"][i],
                "y": ocr_data["top"][i]
            })

        # Build lines for multi-line matching
        lines = []
        for i in range(len(ocr_data["text"])):
            word = ocr_data["text"][i].strip()
            if not word:
                continue
            x = ocr_data["left"][i]
            y = ocr_data["top"][i]
            block = ocr_data["block_num"][i]
            par = ocr_data["par_num"][i]
            line_num = ocr_data["line_num"][i]
            found = False
            for l in lines:
                if l["block"] == block and l["par"] == par and l["line_num"] == line_num:
                    l["words"].append({"text": word, "x": x, "y": y})
                    found = True
                    break
            if not found:
                lines.append({"block": block, "par": par, "line_num": line_num, "words": [{"text": word, "x": x, "y": y}], "y": y})
        lines = sorted(lines, key=lambda l: l["y"])

        for lbl in expected_labels:
            lbl_words = lbl.split()
            first_word = normalize_text(lbl_words[0])
            last_word = normalize_text(lbl_words[-1])
            lbl_clean = clean_label_sequence(lbl_words)
            exp_s, i_mask = build_expected_masked_upper(lbl)
            n = len(tokens)
            max_len = len(lbl_words)
            found = False
            # Try single-line (token sequence) match first
            for i in range(n):
                for j in range(i, min(i + max_len, n)):
                    seq = tokens[i:j+1]
                    if not seq:
                        continue
                    if first_word in normalize_text(seq[0]["orig"]) and last_word in normalize_text(seq[-1]["orig"]):
                        seq_clean = clean_label_sequence([t["orig"] for t in seq])
                        if flex_contains(exp_s, i_mask, seq_clean):
                            label_positions[lbl].append((seq[0]["x"], seq[0]["y"]))
                            found = True
            # Multi-line lookahead (contiguous tokens, x tolerance on next lines)
            if not found:
                x_tolerance = 80
                max_lookahead = 5
                for i, line in enumerate(lines):
                    words = line["words"]
                    if not words:
                        continue
                    # Try all possible starting tokens on this line; pick the one that matches the most leading words contiguously
                    best_start = None
                    best_matched_here = 0
                    for start_idx in range(len(words)):
                        # Match first expected word at this token
                        matched_here = 0
                        k = start_idx
                        for lbl_idx in range(0, len(lbl_words)):
                            if k >= len(words):
                                break
                            wu, w_mask = build_expected_masked_upper(lbl_words[lbl_idx])
                            tok_clean = normalize_text(words[k]["text"])
                            if flex_contains(wu, w_mask, tok_clean):
                                matched_here += 1
                                k += 1
                            else:
                                break
                        if matched_here > best_matched_here:
                            best_matched_here = matched_here
                            best_start = start_idx
                        # Early exit if whole label fits here
                        if matched_here == len(lbl_words):
                            break
                    if best_matched_here == 0:
                        continue
                    # Set starting position and x_ref from the chosen start
                    start_x = words[best_start]["x"]
                    start_y = words[best_start]["y"]
                    x_ref = start_x
                    curr_lbl_idx = best_matched_here
                    curr_idx = i
                    matched_all = (curr_lbl_idx == len(lbl_words))
                    lookahead_used = 0
                    # Continue to next lines within constraints
                    while not matched_all and lookahead_used < max_lookahead:
                        if curr_idx + 1 >= len(lines):
                            break
                        next_line = lines[curr_idx + 1]
                        next_tokens = next_line["words"]
                        if not next_tokens:
                            break
                        # Candidates within x_tolerance for the next expected word
                        expected_word = lbl_words[curr_lbl_idx]
                        wuN, w_maskN = build_expected_masked_upper(expected_word)
                        candidate_indices = [idx for idx, tok in enumerate(next_tokens)
                                             if abs(tok["x"] - x_ref) <= x_tolerance and flex_contains(wuN, w_maskN, normalize_text(tok["text"]))]
                        if not candidate_indices:
                            break
                        # For each candidate, compute how many contiguous words match to the right
                        best_line_match = 0
                        best_line_start = None
                        for ci in candidate_indices:
                            matched_in_line = 0
                            k = ci
                            for lbl_idx in range(curr_lbl_idx, len(lbl_words)):
                                if k >= len(next_tokens):
                                    break
                                wu2, w2_mask = build_expected_masked_upper(lbl_words[lbl_idx])
                                if flex_contains(wu2, w2_mask, normalize_text(next_tokens[k]["text"])):
                                    matched_in_line += 1
                                    k += 1
                                else:
                                    break
                            if matched_in_line > best_line_match:
                                best_line_match = matched_in_line
                                best_line_start = ci
                        if best_line_match == 0:
                            break
                        # Advance
                        curr_lbl_idx += best_line_match
                        curr_idx += 1
                        lookahead_used += 1
                        x_ref = next_tokens[best_line_start]["x"]
                        matched_all = (curr_lbl_idx == len(lbl_words))
                    if matched_all:
                        label_positions[lbl].append((start_x, start_y))
                        print(f"[MULTILINE LOOKAHEAD] '{lbl}' matched starting at y={start_y}")

        print("[LABEL POSITIONS]")
        for lbl, positions in label_positions.items():
            for pos in positions:
                print(f"  Label '{lbl}' matched at (x={pos[0]}, y={pos[1]})")

        return label_positions
        ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        import unicodedata
        import re
        def normalize_text(text):
            text = unicodedata.normalize('NFKD', text)
            text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')
            text = text.replace('/', '').replace(' ', '').replace('-', '')
            text = re.sub(r'^[^a-zA-Z]+', '', text)
            text = re.sub(r'[^a-zA-Z]+$', '', text)
            return text

        def clean_label_sequence(seq):
            normed = [normalize_text(s) for s in seq]
            joined = ''.join(normed)
            joined = unicodedata.normalize('NFKD', joined)
            joined = ''.join(c for c in joined if unicodedata.category(c)[0] != 'C')
            joined = joined.replace(' ', '')
            return joined

        tokens = []
        for i in range(len(ocr_data["text"])):
            word = ocr_data["text"][i].strip()
            if not word:
                continue
            norm = normalize_text(word)
            tokens.append({
                "text": norm,
                "orig": word,
                "x": ocr_data["left"][i],
                "y": ocr_data["top"][i]
            })
        ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        label_positions = defaultdict(list)

        import unicodedata
        import re
        import re
        def normalize_text(text):
            # Remove special/unprintable characters, spaces, slashes, dashes, lowercase
            import unicodedata
            text = unicodedata.normalize('NFKD', text)
            text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')
            text = text.replace('/', '').replace(' ', '').replace('-', '').lower()
            # Strip leading/trailing non-alpha chars
            text = re.sub(r'^[^a-zA-Z]+', '', text)
            text = re.sub(r'[^a-zA-Z]+$', '', text)
            return text

        tokens = []
        print("[OCR TOKENS]")
        for i in range(len(ocr_data["text"])):
            word = ocr_data["text"][i].strip()
            if not word:
                continue
            norm = normalize_text(word)
            print(f"  OCR token: '{word}' normalized: '{norm}' at (x={ocr_data['left'][i]}, y={ocr_data['top'][i]})")
            tokens.append({
                "text": norm,
                "orig": word,
                "x": ocr_data["left"][i],
                "y": ocr_data["top"][i]
            })


        def clean_label_sequence(seq):
            import unicodedata
            # Normalize each word/token, join, remove spaces and non-printable chars, lowercase
            normed = [normalize_text(s) for s in seq]
            joined = ''.join(normed)
            joined = unicodedata.normalize('NFKD', joined)
            joined = ''.join(c for c in joined if unicodedata.category(c)[0] != 'C')
            joined = joined.replace(' ', '').lower()
            return joined

        for lbl in expected_labels:
            lbl_words = lbl.split()
            first_word = normalize_text(lbl_words[0])
            last_word = normalize_text(lbl_words[-1])
            lbl_clean = clean_label_sequence(lbl_words)
            n = len(tokens)
            max_len = len(lbl_words)
            for i in range(n):
                # Only consider sequences up to max_len tokens
                for j in range(i, min(i + max_len, n)):
                    seq = tokens[i:j+1]
                    if not seq:
                        continue
                    # Non-strict match: first/last word is substring of normalized token
                    if first_word in normalize_text(seq[0]["orig"]) and last_word in normalize_text(seq[-1]["orig"]):
                        seq_clean = clean_label_sequence([t["orig"] for t in seq])
                        # Substring match: label exists in cleaned OCR token sequence
                        if lbl_clean in seq_clean:
                            label_positions[lbl].append((seq[0]["x"], seq[0]["y"]))

        print("[LABEL POSITIONS]")
        for lbl, positions in label_positions.items():
            for pos in positions:
                print(f"  Label '{lbl}' matched at (x={pos[0]}, y={pos[1]})")

        return label_positions

    def detect_section_regions(self, pil_image, sections, label_positions, checkbox_positions, max_gap=100):
        ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        lines = {}
        for i in range(len(ocr_data["text"])):
            word = ocr_data["text"][i].strip()
            if not word:
                continue
            line_id = (ocr_data["block_num"][i], ocr_data["par_num"][i], ocr_data["line_num"][i])
            if line_id not in lines:
                lines[line_id] = {
                    "text": word,
                    "x": ocr_data["left"][i],
                    "y": ocr_data["top"][i],
                    "h": ocr_data["height"][i]
                }
            else:
                lines[line_id]["text"] += " " + word

        sorted_lines = sorted(lines.values(), key=lambda l: l["y"])
        checkbox_y_positions = sorted([cb["position"][1] for cb in checkbox_positions])
        section_regions = {}

        # Helper functions for conditional matching on section names
        import unicodedata
        def build_expected_masked_upper(text):
            t = unicodedata.normalize('NFKD', text)
            t = ''.join(c for c in t if unicodedata.category(c)[0] != 'C')
            t = t.replace('/', '').replace(' ', '').replace('-', '')
            expected_s = []
            i_mask = set()
            for idx, ch in enumerate(t):
                if ch == 'I':
                    i_mask.add(idx)
                expected_s.append(ch)
            return ''.join(expected_s), i_mask

        def clean_line_preserve_case(text):
            t = unicodedata.normalize('NFKD', text)
            t = ''.join(c for c in t if unicodedata.category(c)[0] != 'C')
            t = t.replace('/', '').replace(' ', '').replace('-', '')
            return t

        def flex_equal(expected_s, i_mask, candidate_s):
            if len(expected_s) != len(candidate_s):
                return False
            for i, (e, c) in enumerate(zip(expected_s, candidate_s)):
                if i in i_mask and e == 'I':
                    if c not in ('I', 'L', 'l', '1'):
                        return False
                else:
                    if e != c:
                        return False
            return True

        def flex_contains(expected_s, i_mask, haystack_s):
            m, n = len(expected_s), len(haystack_s)
            if m == 0:
                return True
            for i in range(0, n - m + 1):
                if flex_equal(expected_s, i_mask, haystack_s[i:i+m]):
                    return True
            return False

        for section in sections:
            section_name = section["section_name"]
            anchor_y = None
            exp_s, i_mask = build_expected_masked_upper(section_name)
            for line in sorted_lines:
                line_s = clean_line_preserve_case(line["text"])
                if flex_contains(exp_s, i_mask, line_s):
                    anchor_y = line["y"]
                    print(f"[ANCHOR TEXT] '{line['text']}' matched section '{section_name}' at y={anchor_y}")
                    break
            if anchor_y is None:
                print(f"[WARN] No anchor found for section '{section_name}'")
                continue

            # Extend downward until checkbox silence
            y2 = anchor_y
            last_cb_y = None
            for cb_y in checkbox_y_positions:
                if cb_y < anchor_y:
                    continue
                if last_cb_y is None or cb_y - last_cb_y <= max_gap:
                    y2 = cb_y
                    last_cb_y = cb_y
                else:
                    break

            section_regions[section_name] = {
                "x1": 0,
                "y1": anchor_y,
                "x2": 2000,
                "y2": y2 + 50  # small buffer
            }
            print(f"[MATCH] Section '{section_name}' anchored at y={anchor_y}, extended to y={y2 + 50}")

        return section_regions

    def filter_checkboxes_in_region(self, checkboxes, region):
        return [box for box in checkboxes if region["y1"] <= box["position"][1] <= region["y2"]]

    def cluster_checkboxes_by_rows(self, checkboxes, gap_threshold=50):
        sorted_boxes = sorted(checkboxes, key=lambda b: b["position"][1])
        rows = []
        current_row = []

        for i, box in enumerate(sorted_boxes):
            y = box["position"][1]
            if not current_row:
                current_row.append(box)
                continue

            prev_y = current_row[-1]["position"][1]
            if abs(y - prev_y) > gap_threshold:
                rows.append({
                    "y": sum(b["position"][1] for b in current_row) / len(current_row),
                    "boxes": current_row
                })
                current_row = [box]
            else:
                current_row.append(box)

        if current_row:
            rows.append({
                "y": sum(b["position"][1] for b in current_row) / len(current_row),
                "boxes": current_row
            })

        return rows

    def assign_checkboxes_sectionwise(self, checkboxes, sections, label_positions, section_regions,
                                    column_tolerance=100, min_row_gap=60):
        output_sections = []

        output_sections = []
        used_boxes = set()
        assigned_labels = set()

        for sec in sections:
            sec_name = sec["section_name"]
            sec_checkboxes = []
            if sec_name not in section_regions:
                print(f"[WARN] Skipping section '{sec_name}' — no region found")
                continue
            region = section_regions[sec_name]
            section_boxes = self.filter_checkboxes_in_region(checkboxes, region)
            rows = self.cluster_checkboxes_by_rows(section_boxes)
            for lbl in sec["labels"]:
                assigned = False
                if lbl not in label_positions:
                    print(f"[MISS] Label '{lbl}' not found by OCR")
                    sec_checkboxes.append({
                        "label": lbl,
                        "status": "missing",
                        "score": 0.0,
                        "confidence": 0.0,
                        "position": [0, 0, 0, 0]
                    })
                    continue
                for pos in label_positions[lbl]:
                    lx, ly = pos
                    # Only consider labels within section bounds
                    if not (region["y1"] <= ly <= region["y2"]):
                        continue
                    # Find closest checkbox in section
                    best_distance = None
                    best_assignment = None
                    for row in rows:
                        for cb in row["boxes"]:
                            cb_x, cb_y, _, _ = cb["position"]
                            dist = abs(cb_y - ly) + abs(cb_x - lx)
                            if best_distance is None or dist < best_distance:
                                best_distance = dist
                                best_assignment = (lx, ly, cb)
                    if best_assignment:
                        lx, ly, closest_box = best_assignment
                        cb_x, cb_y, _, _ = closest_box["position"]
                        best_row = next((row for row in rows if closest_box in row["boxes"]), None)
                        delta_y = abs(best_row["y"] - ly) if best_row else None
                        box_id = id(closest_box)
                        if best_row is None or delta_y > 60:
                            continue
                        elif box_id in used_boxes:
                            continue
                        else:
                            used_boxes.add(box_id)
                            assigned_labels.add(lbl)
                            sec_checkboxes.append({
                                "label": lbl,
                                "status": closest_box["status"],
                                "score": closest_box["score"],
                                "confidence": closest_box["confidence"],
                                "position": closest_box["position"]
                            })
                            assigned = True
                            break
                if not assigned:
                    sec_checkboxes.append({
                        "label": lbl,
                        "status": "missing",
                        "score": 0.0,
                        "confidence": 0.0,
                        "position": [0, 0, 0, 0]
                    })
            # Only add section if at least one checkbox is not 'missing'
            if any(cb.get("status") != "missing" for cb in sec_checkboxes):
                output_sections.append({
                    "section": sec_name,
                    "checkboxes": sec_checkboxes
                })
        return output_sections
   
    def extract_pdf_with_sections(self, pdf_path, sections_file):
        if not os.path.exists(sections_file):
            raise FileNotFoundError(f"Sections JSON file not found: {sections_file}")
        with open(sections_file, "r", encoding="utf-8") as f:
            sections = json.load(f)

        structured_data = {"pages": []}
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=self.poppler_path)
        print(f"Loaded {len(pages)} pages from PDF: {pdf_path}")

        for page_number, page_image in enumerate(pages, start=1):
            print(f"Processing page {page_number}")
            processed_img = self.preprocess_image(page_image)
            checkboxes, raw_img = self.detect_checkboxes(processed_img)

            all_labels = [lbl for sec in sections for lbl in sec["labels"]]
            label_positions = self.get_label_positions(processed_img, all_labels)
            section_regions = self.detect_section_regions(processed_img, sections, label_positions, checkboxes)

            #print("[LABEL -> CHECKBOX DISTANCES]")
            #for lbl, (lx, ly) in label_positions.items():
                #if not checkboxes:
                    #print(f"[WARN] No checkboxes detected on page {page_number}, skipping distance calc for label '{lbl}'")
                    #continue
                #closest = min(checkboxes, key=lambda cb: abs(cb['position'][1] - ly) + abs(cb['position'][0] - lx))
                #cb_x, cb_y = closest['position'][0], closest['position'][1]
                #dy = abs(cb_y - ly)
                #dx = abs(cb_x - lx)
                #print(f"Label '{lbl}' -> Closest checkbox at (x={cb_x}, y={cb_y}) | Delta-x={dx}, Delta-y={dy}")

            for section in sections:
                sec_name = section["section_name"]
                if sec_name not in section_regions:
                    continue
                region = section_regions[sec_name]
                for lbl in section["labels"]:
                    if lbl not in label_positions:
                        continue
                    for pos in label_positions[lbl]:
                        lx, ly = pos
                    if not (region["y1"] <= ly <= region["y2"]):
                        print(f"[SKIP] Label '{lbl}' at y={ly} is outside section '{sec_name}' bounds ({region['y1']}–{region['y2']})")
                        continue
                    section_boxes = self.filter_checkboxes_in_region(checkboxes, region)
                    if not section_boxes:
                        print(f"[SKIP] No checkboxes in section '{sec_name}' for label '{lbl}'")
                        continue
                    closest = min(section_boxes, key=lambda cb: abs(cb['position'][1] - ly) + abs(cb['position'][0] - lx))
                    cb_x, cb_y = closest['position'][0], closest['position'][1]
                    dy = abs(cb_y - ly)
                    dx = abs(cb_x - lx)
                    print(f"[LABEL -> CHECKBOX] Section '{sec_name}' | Label '{lbl}' -> Closest checkbox at (x={cb_x}, y={cb_y}) | Delta-x={dx}, Delta-y={dy}")

            sections_data = self.assign_checkboxes_sectionwise(
                checkboxes, sections, label_positions, section_regions
            )

            self.annotate_debug_image(raw_img, sections_data, label_positions, page_number, section_regions)

            structured_data["pages"].append({
                "page_number": page_number,
                "sections": sections_data
            })

        print("[FINAL STRUCTURED DATA]")
        pprint.pprint(structured_data)

        return structured_data

    def annotate_debug_image(self, img, sections_data, label_positions, page_number, section_regions=None):
        if section_regions:
            for name, region in section_regions.items():
                cv2.rectangle(img, (region["x1"], region["y1"]), (region["x2"], region["y2"]), (255, 0, 0), 2)
                cv2.putText(img, name, (region["x1"], region["y1"] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        for section in sections_data:
            for box in section["checkboxes"]:
                x, y, w, h = box["position"]
                label = box.get("label", "?")
                score = box.get("score", 0.0)
                status = box.get("status", "unknown")

                color = {
                    "ticked": (0, 255, 0),
                    "empty": (0, 0, 255),
                    "missing": (0, 255, 255)
                }.get(status, (128, 128, 128))

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f"{label} ({status} S:{score:.2f})", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                if label in label_positions:
                    for pos in label_positions[label]:
                        if len(pos) >= 2:
                            lx, ly = pos
                            cv2.arrowedLine(img, (lx, ly), (x, y), color, 1, tipLength=0.2)
                        else:
                            print(f"[WARN] Malformed label position for '{label}': {pos}")

        cv2.imwrite(f"debug_page_{page_number}.png", img)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def run_extractor_from_config(pdf_path,
                              output_path=None,
                              config_path=None,
                              sections=None,
                              ticked_template=None,
                              empty_template=None,
                              poppler=None,
                              threshold=None,
                              tesseract=None):
    """
    Convenience API for calling the extractor from another Python program.

    - Loads defaults from config.json (or a provided config_path)
    - Optional keyword args override config values
    - If output_path is provided, writes JSON there; always returns the dict
    """
    if config_path:
        cfg, base_dir = load_config(config_path)
    else:
        cfg, base_dir = CONFIG, CONFIG_BASEDIR

    # Allow tesseract override
    tess_path = tesseract or cfg.get("tesseract")
    if tess_path:
        pytesseract.pytesseract.tesseract_cmd = resolve_path(tess_path, base_dir)

    sections_path = sections or resolve_path(cfg.get("sections"), base_dir)
    ticked_path = ticked_template or resolve_path(cfg.get("ticked_template"), base_dir)
    empty_path = empty_template or resolve_path(cfg.get("empty_template"), base_dir)
    poppler_path = poppler or resolve_path(cfg.get("poppler"), base_dir)
    th = threshold if threshold is not None else cfg.get("threshold", 0.6)

    missing = []
    if not sections_path: missing.append("sections")
    if not ticked_path: missing.append("ticked_template")
    if not empty_path: missing.append("empty_template")
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(missing)}")

    extractor = CheckboxExtractor(
        poppler_path=poppler_path,
        ticked_template_path=ticked_path,
        empty_template_path=empty_path,
        match_threshold=th,
    )
    data = extractor.extract_pdf_with_sections(pdf_path, sections_path)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False, cls=NpEncoder)
    return data

def main():
    parser = argparse.ArgumentParser(description="PDF Checkbox Extractor with Section-Aware Label Alignment")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--sections", help="Path to sections.json (default from config.json)")
    parser.add_argument("--output", default="output.json", help="Output JSON file")
    parser.add_argument("--ticked_template", help="Path to ticked checkbox image (default from config.json)")
    parser.add_argument("--empty_template", help="Path to empty checkbox image (default from config.json)")
    parser.add_argument("--poppler", help="Path to poppler bin directory (default from config.json)")
    parser.add_argument("--threshold", type=float, help="Template match threshold (0–1, default from config.json)")
    parser.add_argument("--config", help="Path to config.json (optional)")
    args = parser.parse_args()

    # Reload config if a custom path is provided
    if args.config:
        cfg, base_dir = load_config(args.config)
    else:
        cfg, base_dir = CONFIG, CONFIG_BASEDIR

    # Allow overriding Tesseract path via config flag as well
    if cfg.get("tesseract"):
        pytesseract.pytesseract.tesseract_cmd = resolve_path(cfg["tesseract"], base_dir)

    sections_path = args.sections or resolve_path(cfg.get("sections"), base_dir)
    ticked_path = args.ticked_template or resolve_path(cfg.get("ticked_template"), base_dir)
    empty_path = args.empty_template or resolve_path(cfg.get("empty_template"), base_dir)
    poppler_path = args.poppler or resolve_path(cfg.get("poppler"), base_dir)
    threshold = args.threshold if args.threshold is not None else cfg.get("threshold", 0.6)

    # Validate required inputs now that config has been considered
    missing = []
    if not sections_path:
        missing.append("--sections (or config.sections)")
    if not ticked_path:
        missing.append("--ticked_template (or config.ticked_template)")
    if not empty_path:
        missing.append("--empty_template (or config.empty_template)")
    if missing:
        raise SystemExit(f"Missing required inputs: {', '.join(missing)}")

    extractor = CheckboxExtractor(
        poppler_path=poppler_path,
        ticked_template_path=ticked_path,
        empty_template_path=empty_path,
        match_threshold=threshold
    )

    print(f"Starting extraction for: {args.pdf}")
    data = extractor.extract_pdf_with_sections(args.pdf, sections_path)

    print(f"Writing structured data to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, cls=NpEncoder)

    print(f"Extraction complete! Data saved to {args.output}")

if __name__ == "__main__":
        main()