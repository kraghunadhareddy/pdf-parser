import pytesseract
from pdf2image import convert_from_path
try:
    from pdf2image.exceptions import PDFInfoNotInstalledError
except Exception:
    class PDFInfoNotInstalledError(Exception):
        pass
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import json
import os
import argparse
from collections import defaultdict, Counter
import pprint
import re
from response_extractor import extract_responses_from_page
from constants import (
    ANCHOR_OFFSET_PX,
    RENDER_DPI,
    TEMPLATE_THRESHOLD_DEFAULT,
    DEDUPE_MAX_DIST,
    LABEL_MULTILINE_BASE_X_TOLERANCE,
    LABEL_MULTILINE_MAX_LOOKAHEAD,
    LABEL_ROW_GAP_PX,
    LABEL_CB_ASSIGN_MAX_DELTA_Y,
    SECTION_CB_MAX_GAP_PX,
    SECTION_X1,
    SECTION_X2,
    SECTION_BOTTOM_BUFFER_PX,
)
try:
    import pypdfium2 as pdfium  # optional fallback renderer
except Exception:
    pdfium = None

# Simple tee to duplicate stdout/stderr to a file while preserving console output
class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

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
    def __init__(self, poppler_path=None, ticked_template_path=None, empty_template_path=None, match_threshold=0.6, artifacts_dir: str | None = None):
        self.poppler_path = poppler_path
        self.match_threshold = match_threshold
        self.ticked_template = cv2.imread(ticked_template_path, cv2.IMREAD_GRAYSCALE)
        self.empty_template = cv2.imread(empty_template_path, cv2.IMREAD_GRAYSCALE)
        if self.ticked_template is None or self.empty_template is None:
            raise ValueError(
                "❌ Failed to load one or both template images. "
                f"Resolved ticked='{ticked_template_path}', empty='{empty_template_path}'. "
                "Ensure paths exist and are accessible (config paths are resolved relative to config.json)."
            )
        self.template_size = self.ticked_template.shape[::-1]
        # intermediate artifacts (debug images, crops, etc.)
        self.artifacts_dir = artifacts_dir or os.path.join(os.getcwd(), "artifacts")
        try:
            os.makedirs(self.artifacts_dir, exist_ok=True)
        except Exception as _e:
            # Fall back to CWD if cannot create
            self.artifacts_dir = os.getcwd()

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

    def deduplicate_matches(self, ticked_matches, empty_matches, max_dist=DEDUPE_MAX_DIST):
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

        # Summary only; detailed checkbox dumps removed
        print(f"Template matches: {len(boxes)} checkboxes detected")
        return boxes, img

    def get_label_positions(self, pil_image, expected_labels, match_threshold=0.8, next_page_image=None, next_page_head_lines: int = 5):
        import unicodedata
        import re
        ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        label_positions = defaultdict(list)

        def normalize_text(text):
            # Remove control chars and common separators; compare case-insensitively
            text = unicodedata.normalize('NFKD', text)
            text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')
            text = text.replace('/', '').replace(' ', '').replace('-', '')
            text = re.sub(r'^[^a-zA-Z]+', '', text)
            text = re.sub(r'[^a-zA-Z]+$', '', text)
            return text.upper()

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
            t = t.upper()
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
                    # Candidate is already normalized to upper except digits
                    if c not in ('I', 'L', '1'):
                        return False
                else:
                    if e != c.upper():
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

        def clean_token_preserve_case(text: str):
            t = unicodedata.normalize('NFKD', text)
            t = ''.join(c for c in t if unicodedata.category(c)[0] != 'C')
            t = t.replace('/', '').replace(' ', '').replace('-', '')
            return t

        def clean_token_preserve_case(text):
            t = unicodedata.normalize('NFKD', text)
            t = ''.join(c for c in t if unicodedata.category(c)[0] != 'C')
            t = t.replace('/', '').replace(' ', '').replace('-', '')
            return t

        def clean_token_preserve_case(text):
            t = unicodedata.normalize('NFKD', text)
            t = ''.join(c for c in t if unicodedata.category(c)[0] != 'C')
            t = t.replace('/', '').replace('-', '')
            return t

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

        # We'll keep current-page lines separate; cross-page continuation will be attempted later only for unfound labels
        try:
            _, img_h = pil_image.size
        except Exception:
            img_h = 10000

        # Helper to build a version of lines with the head of the next page appended (y-offset by current page height)
        def build_lines_with_next_head(lines_input):
            if next_page_image is None:
                return lines_input
            try:
                np_data = pytesseract.image_to_data(next_page_image, output_type=pytesseract.Output.DICT)
                next_lines = []
                for i in range(len(np_data["text"])):
                    word = np_data["text"][i].strip()
                    if not word:
                        continue
                    x = np_data["left"][i]
                    y = np_data["top"][i] + img_h
                    block = np_data["block_num"][i]
                    par = np_data["par_num"][i]
                    line_num = np_data["line_num"][i]
                    found_local = False
                    for l in next_lines:
                        if l["block"] == block and l["par"] == par and l["line_num"] == line_num:
                            l["words"].append({"text": word, "x": x, "y": y})
                            found_local = True
                            break
                    if not found_local:
                        next_lines.append({
                            "block": block,
                            "par": par,
                            "line_num": line_num,
                            "words": [{"text": word, "x": x, "y": y}],
                            "y": y
                        })
                next_lines = sorted(next_lines, key=lambda l: l["y"])            
                kept = []
                seen_groups = 0
                i2 = 0
                while i2 < len(next_lines) and seen_groups < max(0, int(next_page_head_lines)):
                    group_y = next_lines[i2]["y"]
                    group = []
                    while i2 < len(next_lines) and next_lines[i2]["y"] == group_y:
                        group.append(next_lines[i2])
                        i2 += 1
                    kept.extend(group)
                    seen_groups += 1
                return lines_input + kept
            except Exception:
                return lines_input

        # Multiline helper parameterized by the set of lines to search across
        def try_multiline_on_lines(lines_input, lbl_words_seq):
            # Slightly wider tolerance helps wrapped paragraphs with indent/outdent
            base_x_tolerance = LABEL_MULTILINE_BASE_X_TOLERANCE
            max_lookahead = LABEL_MULTILINE_MAX_LOOKAHEAD

            def _inner(lbl_words_seq_local):
                for i, line in enumerate(lines_input):
                    words = line["words"]
                    if not words:
                        continue
                    # Try all possible starting tokens on this line; pick the one that matches the most leading words contiguously
                    best_start = None
                    best_matched_here = 0
                    for start_idx in range(len(words)):
                        matched_here = 0
                        k = start_idx
                        for lbl_idx in range(0, len(lbl_words_seq_local)):
                            if k >= len(words):
                                break
                            wu, w_mask = build_expected_masked_upper(lbl_words_seq_local[lbl_idx])
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
                        if matched_here == len(lbl_words_seq_local):
                            break
                    if best_matched_here == 0:
                        continue
                    # Set starting position and x_ref from the chosen start
                    start_x = words[best_start]["x"]
                    start_y = words[best_start]["y"]
                    x_ref = start_x
                    curr_lbl_idx = best_matched_here
                    curr_idx = i
                    matched_all = (curr_lbl_idx == len(lbl_words_seq_local))
                    lookahead_used = 0
                    x_tolerance = base_x_tolerance
                    # Continue to next lines within constraints
                    while not matched_all and lookahead_used < max_lookahead:
                        if curr_idx + 1 >= len(lines_input):
                            break
                        next_line = lines_input[curr_idx + 1]
                        next_tokens = next_line["words"]
                        if not next_tokens:
                            break
                        # Candidates within x_tolerance for the next expected word
                        expected_word = lbl_words_seq_local[curr_lbl_idx]
                        wuN, w_maskN = build_expected_masked_upper(expected_word)
                        candidate_indices = [idx for idx, tok in enumerate(next_tokens)
                                             if abs(tok["x"] - x_ref) <= x_tolerance and flex_contains(wuN, w_maskN, normalize_text(tok["text"]))]
                        # If none within tolerance, relax by scanning the whole line (fallback)
                        if not candidate_indices:
                            candidate_indices = [idx for idx, tok in enumerate(next_tokens)
                                                 if flex_contains(wuN, w_maskN, normalize_text(tok["text"]))]
                        if not candidate_indices:
                            break
                        # For each candidate, compute how many contiguous words match to the right
                        best_line_match = 0
                        best_line_start = None
                        for ci in candidate_indices:
                            matched_in_line = 0
                            k = ci
                            for lbl_idx in range(curr_lbl_idx, len(lbl_words_seq_local)):
                                if k >= len(next_tokens):
                                    break
                                wu2, w2_mask = build_expected_masked_upper(lbl_words_seq_local[lbl_idx])
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
                        matched_all = (curr_lbl_idx == len(lbl_words_seq_local))
                    if matched_all:
                        return (start_x, start_y)
                return None

            return _inner(lbl_words_seq)

        # Pass 1: in-page only search
        found_labels = set()
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
                # First, try with full label words on in-page lines only
                pos = try_multiline_on_lines(lines, lbl_words)
                if pos is not None:
                    label_positions[lbl].append(pos)
                    found_labels.add(lbl)
                else:
                    # Fallback: allow starting match from later words if early words were mis-OCR'd (in-page only)
                    # For long labels, skipping up to 4 words reduces sensitivity to noisy line starts.
                    for skip in range(1, min(5, len(lbl_words))):
                        suffix = lbl_words[skip:]
                        pos2 = try_multiline_on_lines(lines, suffix)
                        if pos2 is not None:
                            label_positions[lbl].append(pos2)
                            found_labels.add(lbl)
                            break

        # Pass 2: only for labels not found in-page, allow cross-page continuation by appending head of next page
        if next_page_image is not None:
            lines_with_next = build_lines_with_next_head(lines)
            for lbl in expected_labels:
                if lbl in label_positions and label_positions[lbl]:
                    continue
                lbl_words = lbl.split()
                # Attempt multiline with next-page head lines available
                pos = try_multiline_on_lines(lines_with_next, lbl_words)
                if pos is not None:
                    label_positions[lbl].append(pos)
                    continue
                # Fallback with skipped leading words
                for skip in range(1, min(5, len(lbl_words))):
                    suffix = lbl_words[skip:]
                    pos2 = try_multiline_on_lines(lines_with_next, suffix)
                    if pos2 is not None:
                        label_positions[lbl].append(pos2)
                        break

        # Detailed label position dumps removed

        return label_positions

    def detect_section_regions(self, pil_image, sections, label_positions, checkbox_positions, max_gap=SECTION_CB_MAX_GAP_PX):
        ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        try:
            img_w, img_h = pil_image.size
        except Exception:
            img_w, img_h = (2000, 10000)
        anchor_x_threshold = int(0.10 * img_w)

        # Build lines with token geometry, sorted by y, tokens sorted by x
        lines_map = {}
        for i in range(len(ocr_data["text"])):
            word = (ocr_data["text"][i] or '').strip()
            if not word:
                continue
            key = (ocr_data["block_num"][i], ocr_data["par_num"][i], ocr_data["line_num"][i])
            tok = {
                "text": word,
                "x": int(ocr_data["left"][i]),
                "y": int(ocr_data["top"][i]),
                "w": int(ocr_data["width"][i]),
                "h": int(ocr_data["height"][i]),
            }
            if key not in lines_map:
                lines_map[key] = {"words": [tok], "y": tok["y"], "text": word}
            else:
                lines_map[key]["words"].append(tok)
                lines_map[key]["text"] += " " + word
        for ln in lines_map.values():
            ln["words"].sort(key=lambda t: t["x"])
        sorted_lines = sorted(lines_map.values(), key=lambda l: l["y"]) 
        checkbox_y_positions = sorted([cb["position"][1] for cb in checkbox_positions])
        section_regions = {}

        # Helper functions for conditional matching on section names
        import unicodedata
        def build_expected_masked_upper(text):
            t = unicodedata.normalize('NFKD', text)
            t = ''.join(c for c in t if unicodedata.category(c)[0] != 'C')
            t = t.replace('/', '').replace(' ', '').replace('-', '')
            t = t.upper()
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

        # Strict token cleaner for short headers: keep letters only (drop punctuation like ':' or '.')
        def letters_only_token(text: str) -> str:
            t = unicodedata.normalize('NFKD', text)
            t = ''.join(c for c in t if c.isalpha())
            return t

        # OCR-side normalization for anchor matching: strip common separators to mirror expected-side cleaning
        # This ensures tokens like 'Surgeries/Major' can match 'Surgeries Major' or 'Surgeries-Major'.
        def _ocr_norm_preserve_punct_upper(text: str) -> str:
            t = unicodedata.normalize('NFKD', text)
            # remove control chars
            t = ''.join(c for c in t if unicodedata.category(c)[0] != 'C')
            # strip common separators used in headers
            t = t.replace('/', '').replace(' ', '').replace('-', '')
            return t.upper()

        def _flex_startswith_upper(expected_s: str, i_mask: set[int], haystack_s: str) -> bool:
            m = len(expected_s)
            if m == 0:
                return True
            if len(haystack_s) < m:
                return False
            return flex_equal(expected_s, i_mask, haystack_s[:m])

        base_x_tolerance = LABEL_MULTILINE_BASE_X_TOLERANCE
        max_lookahead = LABEL_MULTILINE_MAX_LOOKAHEAD

        def best_span_in_line(words, name_words):
            # Try to match as many consecutive expected words as possible starting at each token
            best_start = None
            best_matched = 0
            # Precompute concatenated expected string for merged-token OCR cases
            exp_concat_s, exp_concat_mask = build_expected_masked_upper(' '.join(name_words))
            for start_idx in range(len(words)):
                tok0_clean = _ocr_norm_preserve_punct_upper(words[start_idx]["text"])
                # Fast path: entire phrase merged into one token at the start
                if _flex_startswith_upper(exp_concat_s, exp_concat_mask, tok0_clean):
                    return start_idx, len(name_words)
                matched_here = 0
                k = start_idx
                for lbl_idx in range(len(name_words)):
                    if k >= len(words):
                        break
                    wu, w_mask = build_expected_masked_upper(name_words[lbl_idx])
                    tok_clean = _ocr_norm_preserve_punct_upper(words[k]["text"])
                    ok = _flex_startswith_upper(wu, w_mask, tok_clean) if lbl_idx == 0 else flex_contains(wu, w_mask, tok_clean)
                    if ok:
                        matched_here += 1
                        k += 1
                    else:
                        break
                if matched_here > best_matched:
                    best_matched = matched_here
                    best_start = start_idx
            return best_start, best_matched

        def continue_multiline_from(lines_local, i_start, start_idx, name_words, matched_here):
            # Continue matching expected words onto subsequent line groups with x-alignment tolerance
            curr_idx = i_start
            curr_lbl_idx = matched_here
            x_ref = lines_local[i_start]["words"][start_idx]["x"]
            segments = [{
                "line_y": int(lines_local[i_start]["words"][start_idx]["y"]),
                "start_x": x_ref,
                "end_x": lines_local[i_start]["words"][max(start_idx, start_idx + matched_here - 1)]["x"] + lines_local[i_start]["words"][max(start_idx, start_idx + matched_here - 1)]["w"],
                "count": matched_here,
                "tokens": [t["text"] for t in lines_local[i_start]["words"][start_idx: start_idx + matched_here]]
            }]
            lookahead_used = 0
            while curr_lbl_idx < len(name_words) and lookahead_used < max_lookahead:
                # find next y strictly greater than current
                j = curr_idx + 1
                curr_y = lines_local[curr_idx]["y"]
                while j < len(lines_local) and lines_local[j]["y"] <= curr_y:
                    j += 1
                if j >= len(lines_local):
                    break
                next_y = lines_local[j]["y"]
                # all sibling lines at this y
                sibling_indices = []
                kidx = j
                while kidx < len(lines_local) and lines_local[kidx]["y"] == next_y:
                    sibling_indices.append(kidx)
                    kidx += 1
                expected_word = name_words[curr_lbl_idx]
                wuN, w_maskN = build_expected_masked_upper(expected_word)
                best_line_overall_match = 0
                best_line_overall_start = None
                best_line_overall_idx = None
                best_line_overall_segtoks = None
                for li in sibling_indices:
                    toks = lines_local[li]["words"]
                    if not toks:
                        continue
                    # Prefer startswith within x tolerance
                    candidates = [idx for idx, tok in enumerate(toks)
                                  if abs(tok["x"] - x_ref) <= base_x_tolerance and _flex_startswith_upper(wuN, w_maskN, _ocr_norm_preserve_punct_upper(tok["text"]))]
                    if not candidates:
                        candidates = [idx for idx, tok in enumerate(toks)
                                      if _flex_startswith_upper(wuN, w_maskN, _ocr_norm_preserve_punct_upper(tok["text"]))]
                    if not candidates:
                        candidates = [idx for idx, tok in enumerate(toks)
                                      if abs(tok["x"] - x_ref) <= base_x_tolerance and flex_contains(wuN, w_maskN, _ocr_norm_preserve_punct_upper(tok["text"]))]
                    if not candidates:
                        continue
                    best_line_match = 0
                    best_line_start = None
                    best_line_segtoks = None
                    for ci in candidates:
                        matched_in_line = 0
                        k = ci
                        while k < len(toks) and (curr_lbl_idx + matched_in_line) < len(name_words):
                            wu2, w2_mask = build_expected_masked_upper(name_words[curr_lbl_idx + matched_in_line])
                            ok2 = _flex_startswith_upper(wu2, w2_mask, _ocr_norm_preserve_punct_upper(toks[k]["text"])) if matched_in_line == 0 else flex_contains(wu2, w2_mask, _ocr_norm_preserve_punct_upper(toks[k]["text"]))
                            if ok2:
                                matched_in_line += 1
                                k += 1
                            else:
                                break
                        if matched_in_line > best_line_match:
                            best_line_match = matched_in_line
                            best_line_start = ci
                            best_line_segtoks = toks[best_line_start: best_line_start + best_line_match]
                    if best_line_match > best_line_overall_match:
                        best_line_overall_match = best_line_match
                        best_line_overall_start = best_line_start
                        best_line_overall_idx = li
                        best_line_overall_segtoks = best_line_segtoks
                if not best_line_overall_match or best_line_overall_segtoks is None or best_line_overall_idx is None:
                    break
                seg_tokens = best_line_overall_segtoks
                segments.append({
                    "line_y": int(min(t["y"] for t in seg_tokens)),
                    "start_x": seg_tokens[0]["x"],
                    "end_x": seg_tokens[-1]["x"] + seg_tokens[-1]["w"],
                    "count": best_line_overall_match,
                    "tokens": [t["text"] for t in seg_tokens]
                })
                curr_lbl_idx += best_line_overall_match
                curr_idx = best_line_overall_idx
                x_ref = seg_tokens[0]["x"]
                lookahead_used += 1
            return curr_lbl_idx, segments

        def flex_equal(expected_s, i_mask, candidate_s):
            """
            Case-insensitive token equality with IL1-flexibility (when expected has uppercase 'I'),
            mirroring response_extractor’s behavior. Candidate characters are uppercased for
            comparison except when matching the flexible 'I' class.
            """
            if len(expected_s) != len(candidate_s):
                return False
            for i, (e, c) in enumerate(zip(expected_s, candidate_s)):
                if i in i_mask and e == 'I':
                    # Accept I/L/l/1 when expected has uppercase 'I'
                    if c not in ('I', 'L', 'l', '1'):
                        return False
                else:
                    if e != c.upper():
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

        claimed_anchor_ys = set()
        for section in sections:
            section_name = section["section_name"]
            name_words = [w for w in section_name.split() if w]
            if not name_words:
                print(f"[WARN] No anchor found for section '{section_name}'")
                continue
            best_candidate = None  # (matched_count, start_y, start_x)
            best_segments = None
            # Evaluate each line as a potential start; prefer longest match beginning within the first 10%
            for i, line in enumerate(sorted_lines):
                words = line.get("words", [])
                if not words:
                    continue
                start_idx, matched_here = best_span_in_line(words, name_words)
                if matched_here == 0 or start_idx is None:
                    continue
                start_tok = words[start_idx]
                start_x = int(start_tok.get("x", 0))
                start_y = int(start_tok.get("y", line.get("y", 0)))
                # Enforce first-10% on the start token
                if start_x > anchor_x_threshold:
                    continue
                # Avoid reusing an already-claimed y (header-line collisions)
                if start_y in claimed_anchor_ys:
                    continue
                # Continue across following lines to match as many words as possible
                total_matched, segments = continue_multiline_from(sorted_lines, i, start_idx, name_words, matched_here)
                cand = (int(total_matched), start_y, start_x)
                if best_candidate is None or cand > best_candidate:
                    best_candidate = cand
                    best_segments = segments
                # Fast path: full-phrase match achieved
                if total_matched >= len(name_words):
                    break
            if best_candidate is None:
                print(f"[WARN] No anchor found for section '{section_name}'")
                continue
            # Enforce minimum words matched: 1 for single-word headers, 2+ for multi-word headers
            min_required = 1 if len(name_words) == 1 else 2
            if int(best_candidate[0]) < min_required:
                print(f"[WARN] No anchor found for section '{section_name}'")
                continue
            _, anchor_y, _ = best_candidate
            claimed_anchor_ys.add(anchor_y)

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
                "x1": SECTION_X1,
                "y1": anchor_y,
                "x2": SECTION_X2,
                "y2": y2 + SECTION_BOTTOM_BUFFER_PX  # small buffer
            }
            # Section region computed; verbose log removed

        return section_regions

    def filter_checkboxes_in_region(self, checkboxes, region):
        return [box for box in checkboxes if region["y1"] <= box["position"][1] <= region["y2"]]

    def cluster_checkboxes_by_rows(self, checkboxes, gap_threshold=LABEL_ROW_GAP_PX):
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

    def assign_checkboxes_sectionwise(self, checkboxes, sections, label_positions, section_regions):
        output_sections = []

        output_sections = []
        used_boxes = set()
        assigned_labels = set()

        for sec in sections:
            sec_name = sec["section_name"]
            sec_checkboxes = []
            if sec_name not in section_regions:
                # Use ASCII hyphen to avoid mojibake in Windows PowerShell logs
                print(f"[WARN] Skipping section '{sec_name}' - no region found")
                continue
            region = section_regions[sec_name]
            # Apply post-anchor offset for labels to avoid consuming header line
            y1_effective = region["y1"] + ANCHOR_OFFSET_PX
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
                    # Only consider labels within section bounds, enforcing post-anchor offset
                    if not (y1_effective <= ly <= region["y2"]):
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
                        if best_row is None or delta_y > LABEL_CB_ASSIGN_MAX_DELTA_Y:
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
        pages = None
        try:
            pages = convert_from_path(pdf_path, dpi=RENDER_DPI, poppler_path=self.poppler_path)
            print(f"Loaded {len(pages)} pages from PDF: {pdf_path}")
        except (PDFInfoNotInstalledError, FileNotFoundError) as e:
            print(f"[WARN] Poppler/pdf2image path issue: {e}")
            if pdfium is None:
                raise
            # Fallback: render with pypdfium2 at 300 DPI
            print(f"[INFO] Falling back to pypdfium2 renderer at {RENDER_DPI} DPI")
            scale = float(RENDER_DPI) / 72.0
            try:
                doc = pdfium.PdfDocument(pdf_path)
                pages = []
                for i in range(len(doc)):
                    page = doc[i]
                    bitmap = page.render(scale=scale)
                    img = bitmap.to_pil()
                    pages.append(img)
                print(f"Loaded {len(pages)} pages via pypdfium2: {pdf_path}")
            except Exception as e2:
                print(f"[ERROR] pypdfium2 fallback failed: {e2}")
                raise

        # Track remaining/complete state across pages
        # For labels: map section -> set of remaining label strings
        label_sections = None
        remaining_labels_by_section = {}
        completed_label_sections = set()
        # For questions: map section -> list of remaining question strings (preserve duplicates and order)
        remaining_questions_by_section = {}
        completed_question_sections = set()

        for page_number, page_image in enumerate(pages, start=1):
            print(f"Processing page {page_number}")
            processed_img = self.preprocess_image(page_image)
            checkboxes, raw_img = self.detect_checkboxes(processed_img)

            # Only collect labels from sections that actually define them
            if label_sections is None:
                label_sections = [sec for sec in sections if isinstance(sec.get("labels"), list) and sec.get("labels")]
                for sec in label_sections:
                    remaining_labels_by_section[sec["section_name"]] = set(sec["labels"])
            # Build the list of labels to search on this page, skipping sections already complete
            labels_to_search = []
            active_label_sections = []
            for sec in label_sections:
                name = sec["section_name"]
                if name in completed_label_sections:
                    continue
                rem = remaining_labels_by_section.get(name, set())
                if not rem:
                    completed_label_sections.add(name)
                    continue
                active_label_sections.append(sec)
                labels_to_search.extend(sorted(rem))
            # Build next-page preprocessed image up-front for cross-page continuation
            next_page_image_for_labels = None
            if page_number < len(pages):
                try:
                    next_page_image_for_labels = self.preprocess_image(pages[page_number])
                except Exception:
                    next_page_image_for_labels = None
            label_positions = self.get_label_positions(processed_img, labels_to_search, next_page_image=next_page_image_for_labels)
            section_regions = self.detect_section_regions(processed_img, sections, label_positions, checkboxes)

            # Per-label diagnostics removed to reduce verbosity

            # Update remaining/complete sets for labels based purely on OCR detections in section regions
            for sec in active_label_sections:
                sname = sec["section_name"]
                region = section_regions.get(sname)
                if not region:
                    continue
                y1_effective = region["y1"] + ANCHOR_OFFSET_PX
                y2 = region["y2"]
                still_needed = set()
                for lbl in remaining_labels_by_section.get(sname, set()):
                    found_in_region = False
                    for pos in label_positions.get(lbl, []):
                        lx, ly = pos
                        if y1_effective <= ly <= y2:
                            found_in_region = True
                            break
                    if not found_in_region:
                        still_needed.add(lbl)
                remaining_labels_by_section[sname] = still_needed
                if not still_needed:
                    completed_label_sections.add(sname)

            sections_data = self.assign_checkboxes_sectionwise(
                checkboxes, active_label_sections, label_positions, section_regions
            )

            # No further update needed here; we treat labels as 'found' based on OCR presence in region above

            # Extract free-text responses for sections that define "questions"
            next_page_image = None
            if page_number < len(pages):
                try:
                    next_page_image = self.preprocess_image(pages[page_number])  # pages is 0-indexed; current is start=1
                except Exception:
                    next_page_image = None
            # Build a filtered sections list for questions: only include sections with questions still remaining
            if page_number == 1:
                # Initialize remaining questions per section on first pass (keep duplicates)
                for sec in sections:
                    qs = sec.get("questions") or []
                    if qs:
                        remaining_questions_by_section[sec["section_name"]] = list(qs)
            question_sections_active = []
            for sec in sections:
                sname = sec["section_name"]
                qs = sec.get("questions") or []
                if not qs:
                    continue
                if sname in completed_question_sections:
                    continue
                remaining_qs = remaining_questions_by_section.get(sname, [])
                if not remaining_qs:
                    completed_question_sections.add(sname)
                    continue
                # Shallow copy with reduced questions set for this page
                sec_copy = dict(sec)
                # Pass a copy to avoid in-loop mutation; keep original order and duplicates
                sec_copy["questions"] = list(remaining_qs)
                question_sections_active.append(sec_copy)

            responses_data = extract_responses_from_page(
                processed_img,
                question_sections_active,
                section_regions,
                artifacts_dir=self.artifacts_dir,
                next_page_image=next_page_image,
            )

            # Update remaining/complete sets for questions based on matches we just made
            for sec in responses_data or []:
                sname = sec.get("section")
                for q in sec.get("questions", []):
                    qt = q.get("question")
                    rem_list = remaining_questions_by_section.get(sname, [])
                    # Remove only a single occurrence to respect duplicates in the section definition
                    if qt in rem_list:
                        try:
                            rem_list.remove(qt)
                        except ValueError:
                            pass
                        remaining_questions_by_section[sname] = rem_list
                if not remaining_questions_by_section.get(sname):
                    completed_question_sections.add(sname)

            self.annotate_debug_image(raw_img, sections_data, label_positions, page_number, section_regions)

            structured_data["pages"].append({
                "page_number": page_number,
                "sections": sections_data,
                "responses": responses_data
            })

        # Final structured data dump removed

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

        out_path = os.path.join(self.artifacts_dir, f"debug_page_{page_number}.png")
        try:
            cv2.imwrite(out_path, img)
        except Exception as _e:
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
                              tesseract=None,
                              artifacts=None):
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
    artifacts_dir = artifacts or resolve_path(cfg.get("artifacts"), base_dir) or os.path.join(os.getcwd(), "artifacts")

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
        artifacts_dir=artifacts_dir,
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
    parser.add_argument("--artifacts", help="Directory for intermediate artifacts (debug images, crops). Defaults to 'artifacts' in CWD or config.artifacts")
    parser.add_argument("--check", action="store_true", help="Only check environment/dependencies and exit")
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
    artifacts_dir = args.artifacts or resolve_path(cfg.get("artifacts"), base_dir) or os.path.join(os.getcwd(), "artifacts")

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

    # Optional environment check mode
    if args.check:
        print("[CHECK] Python packages present: pdf2image, pytesseract, opencv-python, Pillow, numpy")
        try:
            from pdf2image import pdfinfo_from_path  # noqa
            import pytesseract as _pt  # noqa
            import cv2 as _cv  # noqa
            from PIL import Image as _im  # noqa
            import numpy as _np  # noqa
            print("[CHECK] Python dependencies: OK")
        except Exception as e:
            print(f"[CHECK] Python dependency error: {e}")
        # External tools
        print(f"[CHECK] Tesseract path: {pytesseract.pytesseract.tesseract_cmd}")
        if not (pytesseract.pytesseract.tesseract_cmd and os.path.exists(str(pytesseract.pytesseract.tesseract_cmd))):
            print("[CHECK] WARN: Tesseract not found at configured path. Ensure it's installed or on PATH.")
        print(f"[CHECK] Poppler path: {poppler_path or '(using PATH)'}")
        if poppler_path and not os.path.exists(poppler_path):
            print("[CHECK] WARN: Poppler path does not exist. Use config.json or --poppler to set correctly.")
        print("[CHECK] Done.")
        return

    # Print logs to stdout/stderr only; let the caller redirect to a file if desired
    extractor = CheckboxExtractor(
        poppler_path=poppler_path,
        ticked_template_path=ticked_path,
        empty_template_path=empty_path,
        match_threshold=threshold,
        artifacts_dir=artifacts_dir
    )

    print(f"Starting extraction for: {args.pdf}")
    data = extractor.extract_pdf_with_sections(args.pdf, sections_path)

    print(f"Writing structured data to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, cls=NpEncoder)

    print(f"Extraction complete! Data saved to {args.output}")

if __name__ == "__main__":
        main()