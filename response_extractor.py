import pytesseract
import numpy as np  # highlight fallback
import pytesseract
import numpy as np  # highlight fallback
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency already in main pipeline
    cv2 = None
import unicodedata
import re
from collections import defaultdict
from collections import defaultdict
from PIL import Image

# Runtime debug toggle (set to True only when deep troubleshooting)
DEBUG_VERBOSE = False
from constants import (
    ANCHOR_OFFSET_PX,
    LABEL_MULTILINE_BASE_X_TOLERANCE,
    LABEL_MULTILINE_MAX_LOOKAHEAD,
    ANSWER_COL_GAP_PX,
    ANSWER_PAGE_RIGHT_MARGIN_PX,
    ANSWER_ROW_Y_TOLERANCE_PX,
    ANSWER_MIN_LINE_HEIGHT_PX,
    ANSWER_MAX_VERTICAL_GAP_PX,
    ANSWER_STOP_ON_BLANK,
    ANSWER_LEFT_MARGIN_PX,
    ANSWER_BLANK_LINE_GAP_PX,
    YESNO_SLIDE_OFFSETS,
    YESNO_PROBE_MAX_BANDS,
    YESNO_HIGHLIGHT_CONFIDENCE,
    YESNO_INFERRED_CONFIDENCE,
    ANSWER_CONTINUATION_MIN_DELTA_Y,
    ANSWER_CONTINUATION_MAX_DELTA_Y,
    OCR_PSM,
    OCR_LANG,
    DEBUG_ANSWER_GEOMETRY,
)


# -----------------------------
# Helpers cloned from extractor.get_label_positions (uppercased with IL1-flex)
# -----------------------------
# Note: helper functions mirror label-matching behavior; keep minimal surface


def _build_expected_masked_upper(text: str):
    """
    Expected-side normalization cloned from label matcher: remove '/', ' ', '-' and uppercase.
    Do NOT drop other non-alphanumerics or trim edges; preserve characters to mirror labels logic.
    """
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


def _flex_equal_upper(expected_s: str, i_mask: set[int], candidate_s: str) -> bool:
    if len(expected_s) != len(candidate_s):
        return False
    for i, (e, c) in enumerate(zip(expected_s, candidate_s)):
        if i in i_mask and e == 'I':
            # Parity with label logic: accept I/L/l/1 when expected has uppercase 'I'
            if c not in ('I', 'L', 'l', '1'):
                return False
        else:
            if e != c.upper():
                return False
    return True


def _flex_contains_upper(expected_s: str, i_mask: set[int], haystack_s: str) -> bool:
    m, n = len(expected_s), len(haystack_s)
    if m == 0:
        return True
    for i in range(0, n - m + 1):
        if _flex_equal_upper(expected_s, i_mask, haystack_s[i:i + m]):
            return True
    return False


def _flex_startswith_upper(expected_s: str, i_mask: set[int], haystack_s: str) -> bool:
    """
    Returns True if haystack_s begins with expected_s under IL1-flex rules.
    Prevents substring matches mid-token (e.g., 'ARE' in 'CARE'),
    while allowing joined tokens like 'AREYOU' to match 'ARE'.
    """
    m = len(expected_s)
    if m == 0:
        return True
    if len(haystack_s) < m:
        return False
    return _flex_equal_upper(expected_s, i_mask, haystack_s[:m])


def _build_lines_with_geometry(pil_image, ocr_data=None):
    """Return list of line dicts with token geometry.

    If precomputed Tesseract OCR dict (image_to_data Output.DICT) is supplied,
    reuse it to avoid re-running OCR (performance path aligned with extractor's
    single-pass OCR optimization). Falls back to invoking pytesseract when not
    provided for backward compatibility.
    """
    ocr_cfg = f"--psm {OCR_PSM}" + (f" -l {OCR_LANG}" if OCR_LANG else "")
    data = ocr_data or pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config=ocr_cfg)
    lines = {}
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        tok = {
            "text": word,
            "x": data["left"][i],
            "y": data["top"][i],
            "w": data["width"][i],
            "h": data["height"][i],
        }
        if key not in lines:
            lines[key] = {"words": [tok], "y": data["top"][i]}
        else:
            lines[key]["words"].append(tok)
    for ln in lines.values():
        ln["words"].sort(key=lambda t: t["x"])
    sorted_lines = sorted(lines.values(), key=lambda ln: ln["y"])
    return sorted_lines


def _letters_only_upper(text: str) -> str:
    t = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in t if c.isalpha()).upper()


# -----------------------------
# Section header matching (clone of extractor.detect_section_regions' anchor logic)
# -----------------------------
def _match_section_anchors(pil_image, sections: list[dict], ocr_data=None):
    """Find approximate y anchor for each section header.

    Accepts optional precomputed OCR data for performance. If not provided,
    executes a fresh OCR call.
    """
    ocr_cfg = f"--psm {OCR_PSM}" + (f" -l {OCR_LANG}" if OCR_LANG else "")
    data = ocr_data or pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config=ocr_cfg)
    lines = {}
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in lines:
            lines[key] = {
                "text": word,
                "x": data["left"][i],
                "y": data["top"][i],
                "h": data["height"][i],
                "tokens": [
                    {"text": word, "x": data["left"][i]}
                ]
            }
        else:
            lines[key]["text"] += " " + word
            lines[key].setdefault("tokens", []).append({"text": word, "x": data["left"][i]})
    sorted_lines = sorted(lines.values(), key=lambda l: l["y"])
    # Determine 10%-of-page-x threshold
    try:
        img_w, img_h = pil_image.size
    except Exception:
        img_w, img_h = (2000, 10000)
    anchor_x_threshold = int(0.10 * img_w)

    def clean_line_preserve_case(text: str) -> str:
        t = unicodedata.normalize('NFKD', text)
        t = ''.join(c for c in t if unicodedata.category(c)[0] != 'C')
        t = t.replace('/', '').replace(' ', '').replace('-', '')
        return t

    def letters_only_token(text: str) -> str:
        t = unicodedata.normalize('NFKD', text)
        return ''.join(c for c in t if c.isalpha())

    anchors = {}
    claimed_anchor_ys = set()
    for section in sections:
        name = section["section_name"]
        exp_s, i_mask = _build_expected_masked_upper(name)
        anchor_y = None
        for line in sorted_lines:
            raw = line["text"]
            cleaned_expected = clean_line_preserve_case(name)
            name_words = name.split()
            # Use letters-only token for first word, enforce token-level equality
            exp_first_token_letters = letters_only_token(name_words[0]) if name_words else cleaned_expected
            exp_first_token_masked, exp_first_token_mask = _build_expected_masked_upper(exp_first_token_letters)

            line_tokens = line.get("tokens", [])
            require_token_equality = (len(cleaned_expected) <= 3) or (len(name_words) == 1)
            matched_here = False
            for tok in line_tokens:
                tok_letters = letters_only_token(tok.get("text", ""))
                if not tok_letters:
                    continue
                if require_token_equality:
                    ok = _flex_equal_upper(exp_first_token_masked, exp_first_token_mask, tok_letters)
                else:
                    ok = _flex_equal_upper(exp_first_token_masked, exp_first_token_mask, tok_letters)
                if not ok:
                    continue
                if line["y"] in claimed_anchor_ys:
                    continue
                # Enforce first 10% based on matched token's x
                if int(tok.get("x", 0)) > anchor_x_threshold:
                    continue
                anchor_y = line["y"]
                matched_here = True
                break
            if matched_here:
                break
        if anchor_y is None:
            print(f"[WARN] No anchor found for section '{name}'")
        else:
            anchors[name] = anchor_y
            # Claim anchor y so subsequent sections cannot bind at the same y
            claimed_anchor_ys.add(anchor_y)
    return anchors


# -----------------------------
# Question matching (clone of label matching with multiline lookahead + detailed logs)
# -----------------------------
def _match_questions_like_labels(pil_image, questions: list[str], *, ocr_data=None,
                                 next_page_image=None, next_page_ocr_data=None,
                                 next_page_head_lines: int = 5):
    """Label-like multiline matching for question prompts.

    Parameters:
        pil_image: PIL image for the current page (already preprocessed upstream)
        questions: list of question strings
        ocr_data: optional OCR dict for current page
        next_page_image: optional PIL image for next page (used only if next_page_ocr_data absent)
        next_page_ocr_data: optional OCR dict for next page (head lines appended)
        next_page_head_lines: number of distinct y-line groups from next page to append
    """
    ocr_cfg = f"--psm {OCR_PSM}" + (f" -l {OCR_LANG}" if OCR_LANG else "")
    data = ocr_data or pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config=ocr_cfg)

    def clean_label_sequence(seq):
        # Preserve punctuation as-is; only remove control chars and spaces when concatenating
        joined = ' '.join(seq)
        joined = unicodedata.normalize('NFKD', joined)
        joined = ''.join(c for c in joined if unicodedata.category(c)[0] != 'C')
        joined = joined.replace(' ', '')
        return joined

    # Build lines for multiline processing
    lines = []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word:
            continue
        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]
        block = data["block_num"][i]
        par = data["par_num"][i]
        line_num = data["line_num"][i]
        found = False
        for l in lines:
            if l["block"] == block and l["par"] == par and l["line_num"] == line_num:
                l["words"].append({"text": word, "x": x, "y": y, "w": w, "h": h})
                found = True
                break
        if not found:
            lines.append({
                "block": block,
                "par": par,
                "line_num": line_num,
                "words": [{"text": word, "x": x, "y": y, "w": w, "h": h}],
                "y": y
            })
    for ln in lines:
        ln["words"].sort(key=lambda t: t["x"])
    lines = sorted(lines, key=lambda l: l["y"]) 

    # Optionally append the first few line-groups from the next page, offset by this page's height,
    # so multiline matching can continue across a page break while preserving the start position on this page.
    try:
        _, img_h = pil_image.size
    except Exception:
        img_h = 10000
    if next_page_ocr_data is not None or next_page_image is not None:
        try:
            data2 = next_page_ocr_data or pytesseract.image_to_data(next_page_image, output_type=pytesseract.Output.DICT)
            next_lines = []
            for i in range(len(data2["text"])):
                word = data2["text"][i].strip()
                if not word:
                    continue
                x = data2["left"][i]
                y = data2["top"][i] + img_h  # offset into virtual space below current page
                w = data2["width"][i]
                h = data2["height"][i]
                block = data2["block_num"][i]
                par = data2["par_num"][i]
                line_num = data2["line_num"][i]
                found = False
                for l in next_lines:
                    if l["block"] == block and l["par"] == par and l["line_num"] == line_num:
                        l["words"].append({"text": word, "x": x, "y": y, "w": w, "h": h})
                        found = True
                        break
                if not found:
                    next_lines.append({
                        "block": block,
                        "par": par,
                        "line_num": line_num,
                        "words": [{"text": word, "x": x, "y": y, "w": w, "h": h}],
                        "y": y
                    })
            for ln2 in next_lines:
                ln2["words"].sort(key=lambda t: t["x"])
            next_lines = sorted(next_lines, key=lambda l: l["y"]) 
            # Keep only the first K unique-y groups
            kept = []
            seen_groups = 0
            i = 0
            while i < len(next_lines) and seen_groups < max(0, int(next_page_head_lines)):
                group_y = next_lines[i]["y"]
                group = []
                while i < len(next_lines) and next_lines[i]["y"] == group_y:
                    group.append(next_lines[i])
                    i += 1
                kept.extend(group)
                seen_groups += 1
            lines.extend(kept)
        except Exception:
            pass

    # OCR-side normalization that preserves visible punctuation; only strips control chars and uppercases
    def _ocr_norm_preserve_punct_upper(text: str) -> str:
        """OCR-side normalization.

        Original version preserved most punctuation, but expected-side normalization
        (in _build_expected_masked_upper) strips '/', spaces, and '-'. This caused
        mismatches for single-token questions containing slashes (e.g., 'Packs/Day')
        because expected becomes 'PACKSDAY' while OCR token remained 'PACKS/DAY'.

        To make matching symmetric, we now remove the same trio ('/', ' ', '-') on
        the OCR side before uppercasing, preventing false negatives for slash-joined
        tokens without widening matching rules elsewhere.
        """
        t = unicodedata.normalize('NFKD', text)
        t = ''.join(c for c in t if unicodedata.category(c)[0] != 'C')
        t = t.replace('/', '').replace(' ', '').replace('-', '')
        return t.upper()

    def best_span_in_line(words, lbl_words):
        # Precompute concatenated expected string (normalized) for merged-token OCR cases
        exp_concat_s, exp_concat_mask = _build_expected_masked_upper(' '.join(lbl_words))
        best_start = None
        best_matched_here = 0
        for start_idx in range(len(words)):
            matched_here = 0
            k = start_idx
            # Fast path: if the entire expected phrase appears as a prefix of this single token, accept full-line match
            tok0_clean = _ocr_norm_preserve_punct_upper(words[start_idx]["text"])
            if _flex_startswith_upper(exp_concat_s, exp_concat_mask, tok0_clean):
                return start_idx, len(lbl_words)
            for lbl_idx in range(0, len(lbl_words)):
                if k >= len(words):
                    break
                wu, w_mask = _build_expected_masked_upper(lbl_words[lbl_idx])
                tok_clean = _ocr_norm_preserve_punct_upper(words[k]["text"])
                # For the first expected word, require startswith to avoid substring starts (e.g., 'ARE' in 'CARE') but allow 'AREYOU'
                if lbl_idx == 0:
                    ok = _flex_startswith_upper(wu, w_mask, tok_clean)
                else:
                    ok = _flex_contains_upper(wu, w_mask, tok_clean)
                if ok:
                    matched_here += 1
                    k += 1
                else:
                    break
            if matched_here > best_matched_here:
                best_matched_here = matched_here
                best_start = start_idx
            if matched_here == len(lbl_words):
                break
        return best_start, best_matched_here

    def find_all_full_in_line(words, lbl_words):
        """Return a list of segment dicts for all same-line full matches on this line."""
        segments = []
        if not words:
            return segments
        exp_concat_s, exp_concat_mask = _build_expected_masked_upper(' '.join(lbl_words))
        for start_idx in range(len(words)):
            # Fast path: entire phrase merged into one token
            tok0_clean = _ocr_norm_preserve_punct_upper(words[start_idx]["text"])
            if _flex_startswith_upper(exp_concat_s, exp_concat_mask, tok0_clean):
                seg_tokens = [words[start_idx]]
                segments.append({
                    "line_y": int(min(t["y"] for t in seg_tokens)),
                    "start_x": seg_tokens[0]["x"],
                    "end_x": seg_tokens[-1]["x"] + seg_tokens[-1]["w"],
                    "count": len(lbl_words),
                    "tokens": [t["text"] for t in seg_tokens],
                })
                continue
            matched_here = 0
            k = start_idx
            for lbl_idx in range(0, len(lbl_words)):
                if k >= len(words):
                    matched_here = 0
                    break
                wu, w_mask = _build_expected_masked_upper(lbl_words[lbl_idx])
                tok_clean = _ocr_norm_preserve_punct_upper(words[k]["text"])
                if lbl_idx == 0:
                    ok = _flex_startswith_upper(wu, w_mask, tok_clean)
                else:
                    ok = _flex_contains_upper(wu, w_mask, tok_clean)
                if ok:
                    matched_here += 1
                    k += 1
                else:
                    matched_here = 0
                    break
            if matched_here == len(lbl_words):
                seg_tokens = words[start_idx: start_idx + matched_here]
                segments.append({
                    "line_y": int(min(t["y"] for t in seg_tokens)),
                    "start_x": seg_tokens[0]["x"],
                    "end_x": seg_tokens[-1]["x"] + seg_tokens[-1]["w"],
                    "count": matched_here,
                    "tokens": [t["text"] for t in seg_tokens],
                })
        return segments

    base_x_tolerance = LABEL_MULTILINE_BASE_X_TOLERANCE
    max_lookahead = LABEL_MULTILINE_MAX_LOOKAHEAD

    def try_multiline(lbl_words_seq):
        for i, line in enumerate(lines):
            words = line["words"]
            if not words:
                continue
            best_start, matched_here = best_span_in_line(words, lbl_words_seq)
            if matched_here == 0:
                continue
            start_x = words[best_start]["x"]
            start_y = words[best_start]["y"]
            x_ref = start_x
            curr_lbl_idx = matched_here
            curr_idx = i
            matched_all = (curr_lbl_idx == len(lbl_words_seq))
            lookahead_used = 0
            # Debug start logs removed for cleanliness
            segments = [{
                "line_y": int(min(t["y"] for t in words[best_start: best_start + matched_here]) if matched_here > 0 else line.get("y", 0)),
                "start_x": start_x,
                "end_x": words[min(best_start + max(0, matched_here - 1), len(words)-1)]["x"] + words[min(best_start + max(0, matched_here - 1), len(words)-1)]["w"],
                "count": matched_here,
                "tokens": [t["text"] for t in words[best_start: best_start + matched_here]]
            }]
            while not matched_all and lookahead_used < max_lookahead:
                # Find the first y that is strictly greater than current, then consider ALL lines with that same y
                j = curr_idx + 1
                curr_y = lines[curr_idx]["y"] if curr_idx < len(lines) else None
                while j < len(lines) and curr_y is not None and lines[j]["y"] <= curr_y:
                    j += 1
                if j >= len(lines):
                    break
                next_y = lines[j]["y"]
                # Collect sibling lines sharing the same y value
                group_indices = []
                kidx = j
                while kidx < len(lines) and lines[kidx]["y"] == next_y:
                    group_indices.append(kidx)
                    kidx += 1

                expected_word = lbl_words_seq[curr_lbl_idx]
                wuN, w_maskN = _build_expected_masked_upper(expected_word)

                best_line_overall_match = 0
                best_line_overall_start = None
                best_line_overall_idx = None
                best_line_overall_segtoks = None

                # Evaluate each sibling line at this y and pick the best continuation
                for li in group_indices:
                    line_tokens = lines[li]["words"]
                    if not line_tokens:
                        continue
                    # 1) Prefer startswith matches within x tolerance
                    candidate_indices = [idx for idx, tok in enumerate(line_tokens)
                                         if abs(tok["x"] - x_ref) <= base_x_tolerance and _flex_startswith_upper(wuN, w_maskN, _ocr_norm_preserve_punct_upper(tok["text"]))]
                    # 2) Relax to startswith anywhere on the line
                    if not candidate_indices:
                        candidate_indices = [idx for idx, tok in enumerate(line_tokens)
                                             if _flex_startswith_upper(wuN, w_maskN, _ocr_norm_preserve_punct_upper(tok["text"]))]
                    # 3) Last resort: contains within x tolerance
                    if not candidate_indices:
                        candidate_indices = [idx for idx, tok in enumerate(line_tokens)
                                             if abs(tok["x"] - x_ref) <= base_x_tolerance and _flex_contains_upper(wuN, w_maskN, _ocr_norm_preserve_punct_upper(tok["text"]))]
                    if not candidate_indices:
                        continue

                    # For each candidate start on this line, compute how many expected words we can match contiguously
                    best_line_match = 0
                    best_line_start = None
                    best_line_segtoks = None
                    for ci in candidate_indices:
                        matched_in_line = 0
                        k = ci
                        while k < len(line_tokens) and (curr_lbl_idx + matched_in_line) < len(lbl_words_seq):
                            wu2, w2_mask = _build_expected_masked_upper(lbl_words_seq[curr_lbl_idx + matched_in_line])
                            if matched_in_line == 0:
                                ok2 = _flex_startswith_upper(wu2, w2_mask, _ocr_norm_preserve_punct_upper(line_tokens[k]["text"]))
                            else:
                                ok2 = _flex_contains_upper(wu2, w2_mask, _ocr_norm_preserve_punct_upper(line_tokens[k]["text"]))
                            if ok2:
                                matched_in_line += 1
                                k += 1
                            else:
                                break
                        if matched_in_line > best_line_match:
                            best_line_match = matched_in_line
                            best_line_start = ci
                            best_line_segtoks = line_tokens[best_line_start: best_line_start + best_line_match]

                    if best_line_match > best_line_overall_match:
                        best_line_overall_match = best_line_match
                        best_line_overall_start = best_line_start
                        best_line_overall_idx = li
                        best_line_overall_segtoks = best_line_segtoks

                # If none of the sibling lines at this y produced a continuation, stop
                if best_line_overall_match == 0 or best_line_overall_idx is None or best_line_overall_segtoks is None:
                    break

                # Advance using the best sibling line for this y
                seg_tokens = best_line_overall_segtoks
                next_line = lines[best_line_overall_idx]
                segments.append({
                    "line_y": int(min(t["y"] for t in seg_tokens) if seg_tokens else next_line.get("y", 0)),
                    "start_x": seg_tokens[0]["x"],
                    "end_x": seg_tokens[-1]["x"] + seg_tokens[-1]["w"],
                    "count": best_line_overall_match,
                    "tokens": [t["text"] for t in seg_tokens]
                })

                curr_lbl_idx += best_line_overall_match
                curr_idx = best_line_overall_idx
                lookahead_used += 1
                x_ref = seg_tokens[0]["x"]
                matched_all = (curr_lbl_idx == len(lbl_words_seq))
            if matched_all:
                return {
                    "start": (start_x, start_y),
                    "segments": segments
                }
        return None

    results = defaultdict(list)
    # De-duplicate by question text to avoid redundant identical scans for repeated entries
    unique_questions = []
    seen_qtexts = set()
    for q in questions:
        if q not in seen_qtexts:
            unique_questions.append(q)
            seen_qtexts.add(q)
    for q in unique_questions:
        raw_words = q.split()
        q_words = [w for w in raw_words if w]
        if not q_words:
            continue
        # Pass 1: collect all same-line full matches across the page
        seen_starts = set()
        for line in lines:
            segs = find_all_full_in_line(line["words"], q_words)
            for seg in segs:
                sx = int(seg["start_x"])
                sy = int(seg["line_y"])
                key = (sx, sy)
                if key in seen_starts:
                    continue
                seen_starts.add(key)
                results[q].append({
                    "x": sx,
                    "y": sy,
                    "segments": [seg]
                })
        # Pass 2: multiline fallback – find additional wrapped matches not caught in pass 1
        hit = try_multiline(q_words)
        if hit is not None:
            sx, sy = int(hit["start"][0]), int(hit["start"][1])
            key = (sx, sy)
            if key not in seen_starts:
                results[q].append({
                    "x": sx,
                    "y": sy,
                    "segments": hit["segments"],
                })
    return results


# -----------------------------
# Public API: match sections and their questions with positions
# -----------------------------
def match_sections_and_questions(pil_image, sections: list[dict], section_regions: dict | None = None,
                                 *, ocr_data=None, next_page_image=None, next_page_ocr_data=None):
    # 1) Section anchors: prefer provided section_regions from extractor; fallback to internal anchor detection
    anchors = {}
    bands = {}
    if section_regions:
        # Capture anchors from provided regions first
        for sec in sections:
            name = sec["section_name"]
            reg = section_regions.get(name)
            if not reg:
                continue
            anchors[name] = reg.get("y1")
        # Build vertical bands using region.y2 as primary bottom, but never past next anchor
        try:
            img_w, img_h = pil_image.size
        except Exception:
            img_h = 10_000
        ordered = sorted([(n, y) for n, y in anchors.items() if y is not None], key=lambda t: t[1])
        name_to_next_anchor = {}
        for idx, (name, y1) in enumerate(ordered):
            next_y = ordered[idx + 1][1] - 1 if (idx + 1) < len(ordered) else img_h
            name_to_next_anchor[name] = next_y
        for name, anchor_y in ordered:
            reg = section_regions.get(name) or {}
            region_y2 = int(reg.get("y2")) if isinstance(reg.get("y2"), (int, float)) else None
            next_anchor_bottom = name_to_next_anchor.get(name, img_h)
            if region_y2 is not None:
                bands[name] = (anchor_y, min(region_y2, next_anchor_bottom))
            else:
                bands[name] = (anchor_y, next_anchor_bottom)
    else:
        anchors = _match_section_anchors(pil_image, sections, ocr_data=ocr_data)
        # Build vertical bands per section using next anchor as bottom; fallback to image height
        try:
            img_w, img_h = pil_image.size
        except Exception:
            img_h = 10_000
        ordered = sorted([(name, y) for name, y in anchors.items()], key=lambda t: t[1])
        for idx, (name, y) in enumerate(ordered):
            y1 = y
            y2 = ordered[idx + 1][1] - 1 if (idx + 1) < len(ordered) else img_h
            bands[name] = (y1, y2)

    # 2) For each section, match its questions using label-like matcher
    out = []
    # Enforce questions to start below anchor by the shared offset
    anchor_offset_px = ANCHOR_OFFSET_PX
    for sec in sections:
        sec_name = sec["section_name"]
        qs = sec.get("questions") or []
        if not qs:
            continue
        # First pass: match only within this page; we'll only use cross-page continuation if needed per-question
        qhits = _match_questions_like_labels(
            pil_image,
            qs,
            ocr_data=ocr_data,
            next_page_image=None,
            next_page_ocr_data=None
        )
        sec_hits = []
        # Lock exact start positions per question so repeated entries pick subsequent occurrences
        claimed_start_positions_by_q = defaultdict(set)
        yband = bands.get(sec_name)
        # Determine the minimum y allowed for question starts (below anchor)
        sec_anchor_y = anchors.get(sec_name)
        min_start_y = None
        try:
            if isinstance(sec_anchor_y, (int, float)):
                min_start_y = int(sec_anchor_y) + anchor_offset_px
        except Exception:
            min_start_y = None
        for q in qs:
            hits = qhits.get(q, [])
            # Discard any hits that indicate skipped words (full-match requirement)
            hits = [h for h in hits if not h.get("skipped")]
            # Attribute to this section by y-band if available
            if yband:
                y1, y2 = yband
                # Enforce both band and post-anchor minimum start y (if anchor known)
                hits = [h for h in hits if y1 <= h.get("y", 0) <= y2 and (min_start_y is None or h.get("y", 0) >= min_start_y)]
            # If no in-band hits found, retry by restricting OCR to the band's crop (within-section scan)
            if not hits and yband:
                try:
                    img_w, img_h = pil_image.size
                except Exception:
                    img_w, img_h = (2000, 10000)
                y1, y2 = yband
                # Guard bounds
                # Start crop below the anchor if known
                y1_effective = int(y1)
                if isinstance(sec_anchor_y, (int, float)):
                    y1_effective = max(y1_effective, int(sec_anchor_y) + anchor_offset_px)
                y1c = max(0, y1_effective)
                # First try with the band bottom (may be region.y2 clamped by next anchor)
                y2c_primary = max(y1c + 1, int(min(img_h, y2)))
                def try_crop(y2c_local):
                    local_hits = []
                    try:
                        region_img = pil_image.crop((0, y1c, img_w, y2c_local))
                        bhits = _match_questions_like_labels(region_img, [q]).get(q, [])
                        for bh in bhits:
                            bh["y"] = int(bh.get("y", 0)) + y1c
                            for seg in bh.get("segments", []):
                                if isinstance(seg, dict) and "line_y" in seg:
                                    try:
                                        seg["line_y"] = int(seg["line_y"]) + y1c
                                    except Exception:
                                        pass
                            if (min_start_y is None) or (bh.get("y", 0) >= min_start_y):
                                local_hits.append(bh)
                    except Exception:
                        return []
                    return local_hits
                band_hits = try_crop(y2c_primary)
                # If still empty and we have anchors, widen to next-anchor bottom ignoring region.y2
                if not band_hits and section_regions and sec_name in anchors:
                    try:
                        # compute next anchor bottom
                        ordered = sorted([(n, y) for n, y in anchors.items() if y is not None], key=lambda t: t[1])
                        idx = next((i for i, (n, _) in enumerate(ordered) if n == sec_name), None)
                        if idx is not None:
                            next_anchor_bottom = ordered[idx + 1][1] - 1 if (idx + 1) < len(ordered) else img_h
                            y2c_wide = max(y1c + 1, int(min(img_h, next_anchor_bottom)))
                            if y2c_wide > y2c_primary:
                                band_hits = try_crop(y2c_wide)
                    except Exception:
                        pass
                for bh in band_hits:
                    hits.append(bh)
                # Suppress targeted retry logs
            # Cross-page fallback: if still no hits for this question starting on this page, allow continuation into next page
            if not hits and (next_page_image is not None or next_page_ocr_data is not None):
                try:
                    xhits = _match_questions_like_labels(
                        pil_image,
                        [q],
                        ocr_data=ocr_data,
                        next_page_image=next_page_image,
                        next_page_ocr_data=next_page_ocr_data
                    ).get(q, [])
                    if yband:
                        y1, y2 = yband
                        xhits = [h for h in xhits if y1 <= h.get("y", 0) <= y2 and (min_start_y is None or h.get("y", 0) >= min_start_y)]
                    for h in xhits:
                        hits.append(h)
                except Exception:
                    pass
            if not hits:
                continue
            # Sort stable and pick the first hit whose exact start (x,y) isn't already claimed for this question
            for chosen in sorted(hits, key=lambda h: (h.get("y", 0), h.get("x", 0))):
                cx, cy = int(chosen.get("x", 0)), int(chosen.get("y", 0))
                key = (cx, cy)
                if key in claimed_start_positions_by_q[q]:
                    continue
                claimed_start_positions_by_q[q].add(key)
                sec_hits.append({
                    "question": q,
                    "position": [cx, cy],
                    "segments": chosen.get("segments", []),
                    "skipped": chosen.get("skipped") if "skipped" in chosen else None
                })
                break
        # Only include sections that have at least one matched question
        if sec_hits:
            out.append({
                "section": sec_name,
                "anchor_y": anchors.get(sec_name),
                "questions": sec_hits
            })
    return out


# -----------------------------
# Compatibility wrapper for extractor.py
# -----------------------------
def extract_responses_from_page(pil_image, sections: list, section_regions: dict | None = None,
                                artifacts_dir: str | None = None, next_page_image=None,
                                *, ocr_data=None, next_page_ocr_data=None, checkboxes=None):
    """Wrapper used by extractor to pull question anchor positions.

    Added performance parameters:
        ocr_data: precomputed OCR dict for current page
        next_page_ocr_data: precomputed OCR dict for next page (for cross-page continuation)
    """
    matches = match_sections_and_questions(
        pil_image,
        sections,
        section_regions=section_regions,
        ocr_data=ocr_data,
        next_page_image=next_page_image,
        next_page_ocr_data=next_page_ocr_data,
    )
    # --- Fallback utilities (defined inside to avoid polluting module namespace) ---
    def _looks_yes_no_question(q_text: str) -> bool:
        if not q_text:
            return False
        lower = q_text.lower()
        # heuristic triggers: direct yes/no phrasing or opt-in verbs
        trig = ("would you" in lower or "do you" in lower or "are you" in lower or "have you" in lower or "yes" in lower or "no" in lower)
        return trig and lower.strip().endswith('?')

    def _yellow_highlight_ocr(pil_img, box: dict):
        """Attempt to recover short printed YES/NO under highlighter.
        box: {x_start,y_start,x_end,y_end} from answer_window.
        Returns cleaned_text or ''.
        """
        if cv2 is None:
            return ""
        try:
            xs, ys, xe, ye = box["x_start"], box["y_start"], box["x_end"], box["y_end"]
            # Expand a little vertically & horizontally
            pad_x = 15
            pad_y = 12
            xs2 = max(0, xs - pad_x)
            ys2 = max(0, ys - pad_y)
            xe2 = xe + pad_x
            ye2 = ye + pad_y
            crop = pil_img.crop((xs2, ys2, xe2, ye2))
            cv = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(cv, cv2.COLOR_BGR2HSV)
            # broad yellow mask
            yellow_mask = cv2.inRange(hsv, (18, 60, 140), (40, 255, 255))
            # neutralize highlight: set masked regions to white
            cv[yellow_mask > 0] = (255, 255, 255)
            gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
            # adaptive threshold to bring out dark glyphs
            thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 21, 9)
            # slight dilation to connect thin strokes
            kernel = np.ones((2, 2), np.uint8)
            thr = cv2.dilate(thr, kernel, iterations=1)
            # restrict character set to YESNO for precision
            config = "--psm 7 -c tessedit_char_whitelist=YESNOyesno"
            txt = pytesseract.image_to_string(thr, config=config).strip()
            txt = txt.replace('\n', ' ').strip()
            # normalize common OCR noise (e.g., 'YEs', 'NOO')
            if txt.lower().startswith('yes'):
                return 'Yes'
            if txt.lower().startswith('no'):
                return 'No'
            # Extremely short outputs; sometimes returns 'Y' or 'N'
            if txt in {'Y', 'y'}:
                return 'Yes'
            if txt in {'N', 'n'}:
                return 'No'
            return ''
        except Exception as e:  # pragma: no cover
            print(f"[YESNO-FALLBACK-ERROR] {e}")
            return ""
    # Augment with answer extraction
    try:
        # Build line index from OCR once here (ensure we use global OCR config for consistency)
        ocr_cfg = f"--psm {OCR_PSM}" + (f" -l {OCR_LANG}" if OCR_LANG else "")
        data = ocr_data or pytesseract.image_to_data(
            pil_image,
            output_type=pytesseract.Output.DICT,
            config=ocr_cfg,
        )
        lines = {}
        for i in range(len(data["text"])):
            word = data["text"][i].strip()
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            if not word:
                # still record blank structure for gap detection by line grouping
                pass
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            if key not in lines:
                lines[key] = {"words": [], "y": y, "h": h}
            if word:
                lines[key]["words"].append({"text": word, "x": x, "y": y, "w": w, "h": h})
                lines[key]["h"] = max(lines[key]["h"], h)
        line_list = sorted(lines.values(), key=lambda ln: ln["y"])
        # Precompute plain text per line
        for ln in line_list:
            ln["text"] = " ".join(w["text"] for w in ln["words"]) if ln["words"] else ""

        # Helper: find next question start x on approximately same y-band within same section
        def compute_right_bound(section_questions, current_q):
            cx = current_q["position"][0]
            cy = current_q["position"][1]
            segments = current_q.get("segments") or []
            # Detect narrative (Medical Conditions) and allow full width for those only
            is_narrative = False
            try:
                seg_tokens_lower = []
                for seg in segments:
                    if isinstance(seg, dict):
                        seg_tokens_lower.extend([t.lower() for t in seg.get("tokens", [])])
                if "medical" in seg_tokens_lower and "conditions" in seg_tokens_lower:
                    is_narrative = True
            except Exception:
                pass
            candidates = []
            for q in section_questions:
                if q is current_q:
                    continue
                qx, qy = q.get("position", [None, None])
                if qx is None or qy is None:
                    continue
                if abs(qy - cy) <= ANSWER_ROW_Y_TOLERANCE_PX and qx > cx:
                    candidates.append(qx)
            if candidates and not is_narrative:
                return min(candidates) - ANSWER_COL_GAP_PX
            # Fallback: extend to page width minus margin
            try:
                img_w, _ = pil_image.size
            except Exception:
                img_w = 2000
            return img_w - ANSWER_PAGE_RIGHT_MARGIN_PX

        # Helper: gather lines of answer window
        def collect_answer_lines(q_start_x, q_start_y, q_segments, right_x, section_bottom_y=None, question_line_token_set=None):
            # Determine baseline start y from last segment line if multiline
            last_seg_y = q_start_y
            if q_segments:
                try:
                    last_seg_y = max(int(seg.get("line_y", last_seg_y)) for seg in q_segments if isinstance(seg, dict))
                except Exception:
                    pass
            # FIXED OFFSET LOGIC (per explicit requirement):
            # Start Y is always the last question segment line plus ANCHOR_OFFSET_PX.
            # The answer window vertical extent is exactly one offset tall: [start_collect_y, start_collect_y + ANCHOR_OFFSET_PX].
            # We deliberately ignore any adaptive lowering and do NOT capture inline same-line tokens.
            start_collect_y = last_seg_y + ANCHOR_OFFSET_PX
            band_bottom_y = start_collect_y + ANCHOR_OFFSET_PX
            answers = []
            # Measure vertical gap from actual last question line so immediate answers aren't skipped
            prev_y = last_seg_y
            last_line_y2 = None
            captured_any = False
            # Specialized debug for 'Medical Conditions' narrative answers (late page family history rows)
            question_tokens_flat = []
            for seg in (q_segments or []):
                if isinstance(seg, dict):
                    question_tokens_flat.extend([t.lower() for t in seg.get("tokens", [])])
            is_medical_conditions_q = ("medical" in question_tokens_flat and "conditions" in question_tokens_flat)
            # Detect the long reminder opt-in question (appointment reminders) for targeted debug
            is_reminder_optin_q = False
            if question_tokens_flat:
                # Look for distinctive tokens from the long question
                key_hits = 0
                for kw in ("reminders?", "automatic", "appointment", "messages."):
                    if kw.rstrip('?').lower().strip('.') in question_tokens_flat:
                        key_hits += 1
                if key_hits >= 2 and ("reminders?" in [t.rstrip('?') for t in question_tokens_flat] or "reminders" in question_tokens_flat):
                    is_reminder_optin_q = True
            if DEBUG_VERBOSE and is_medical_conditions_q and start_collect_y > 2500:
                print(f"[MC] q_y={q_start_y} band={start_collect_y}-{band_bottom_y}")
            # Precompute a simple function to guess if a line looks like a new section header (many capitalized words, long span)
            def looks_like_header(text: str) -> bool:
                if not text:
                    return False
                tokens = text.strip().split()
                if len(tokens) <= 1:
                    return False
                cap_tokens = sum(1 for t in tokens if t.isupper() and len(t) >= 3)
                # heuristics: at least half tokens uppercase and total length > 12
                if cap_tokens >= max(2, len(tokens)//2) and sum(len(t) for t in tokens) > 12:
                    return True
                return False

            # --- DEBUG BLOCK (TEMPORARY) ---
            # If this looks like the pregnancy question (multi-line, specific y deltas) or any question whose offset is 1373,
            # dump nearby OCR lines to understand why answer not captured.
            # Removed ad-hoc pregnancy debug; rely on DEBUG_VERBOSE + targeted logging when needed.
            # Additional window debug for reminder opt-in question
            if DEBUG_VERBOSE and is_reminder_optin_q:
                print(f"[REMINDER] q_y={q_start_y} band={start_collect_y}-{band_bottom_y}")
                band_low2 = start_collect_y - 20
                band_high2 = band_bottom_y + 20
                for ln_dbg2 in line_list:
                    ly2 = ln_dbg2.get("y")
                    if ly2 is None or ly2 < band_low2 or ly2 > band_high2:
                        continue
                    wds2 = ln_dbg2.get("words", [])
                    all_tokens2 = [w.get("text","") for w in wds2]
                    in_tokens2 = [w.get("text","") for w in wds2 if q_start_x <= w.get("x",0) <= right_x]
                    print(f"[REMINDER-INSPECT] line_y={ly2} in_tokens={in_tokens2} all={all_tokens2}")
                # Extended sweep (broader vertical) to locate where 'Yes' might actually be
                sweep_low = last_seg_y + 1
                sweep_high = last_seg_y + 240  # wider than answer band
                yesno_hits = []
                for ln_dbg3 in line_list:
                    ly3 = ln_dbg3.get("y")
                    if ly3 is None or ly3 < sweep_low or ly3 > sweep_high:
                        continue
                    wds3 = ln_dbg3.get("words", [])
                    if not wds3:
                        continue
                    all_tokens3 = [w.get("text","") for w in wds3]
                    in_tokens3 = [w.get("text","") for w in wds3 if q_start_x <= w.get("x",0) <= right_x]
                    if any(tok.lower() in ("yes","no","y","n") for tok in all_tokens3):
                        yesno_hits.append((ly3, all_tokens3, in_tokens3))
                    # Log every 60px stratum or lines that have any tokens in window
                    if in_tokens3 or (ly3 - sweep_low) % 60 < 5:
                        print(f"[REMINDER-SWEEP] line_y={ly3} in_tokens={in_tokens3} all={all_tokens3}")
                if yesno_hits:
                    for hit_y, hit_all, hit_in in yesno_hits:
                        print(f"[REMINDER-HIT] y={hit_y} tokens={hit_all} in_window={hit_in}")
                else:
                    print(f"[REMINDER-HIT] none_found within_y={sweep_low}-{sweep_high}")
            # --- END DEBUG BLOCK ---
            # Inline same-line capture intentionally disabled under fixed-offset rule.
            # Phase 1: Band capture (strictly within initial fixed window)
            left_bound = max(0, q_start_x - ANSWER_LEFT_MARGIN_PX)
            captured_line_ys: list[int] = []
            for ln in line_list:
                ly = ln.get("y")
                if ly is None or ly <= last_seg_y:
                    continue
                if ly < start_collect_y or ly > band_bottom_y:
                    continue
                words = ln.get("words", [])
                window_tokens = [w.get("text", "") for w in words if left_bound <= w.get("x", 0) <= right_x]
                if window_tokens:
                    answers.append(" ".join(t for t in window_tokens if t))
                    captured_any = True
                    captured_line_ys.append(ly)
                    try:
                        h = ln.get("h") or (max((wd.get("h", 0) for wd in words), default=0)) or 0
                    except Exception:
                        h = 0
                    last_line_y2 = ly + h
                # (debug capture removed)
                if is_medical_conditions_q and window_tokens:
                    print(f"[MC-CAPTURE] band_line_y={ly} tokens={window_tokens}")
                if is_reminder_optin_q and window_tokens:
                    print(f"[REMINDER-CAPTURE] y={ly} tokens={window_tokens}")
            # New continuation rule (replaces discrete +40px step probing):
            # If an answer line was found, define first answer baseline y_answer as the
            # minimum captured line y. Then look for additional wrapped lines whose
            # baseline y satisfies: y > y_answer + 10 and y <= y_answer + 50.
            # Capture all qualifying lines in that range (still confined to the
            # same horizontal window) in ascending order. This targets natural
            # immediate wraps that occur just below the initial band without
            # waiting a full ANCHOR_OFFSET_PX jump.
            if captured_any and captured_line_ys:
                y_answer = min(captured_line_ys)
                cont_start = y_answer + ANSWER_CONTINUATION_MIN_DELTA_Y
                cont_end = y_answer + ANSWER_CONTINUATION_MAX_DELTA_Y
                for ln2 in line_list:
                    ly2 = ln2.get("y")
                    if ly2 is None:
                        continue
                    if ly2 <= y_answer:  # never go back upwards or reuse same baseline
                        continue
                    if ly2 <= cont_start:
                        continue
                    if ly2 > cont_end:
                        break
                    if ly2 in captured_line_ys:
                        continue  # already captured inside initial band
                    words2 = ln2.get("words", [])
                    c_tokens = [w.get("text", "") for w in words2 if left_bound <= w.get("x", 0) <= right_x]
                    if not c_tokens:
                        continue
                    # Skip if this line's token sequence exactly matches any question line tokens (prevents grabbing next question row)
                    if question_line_token_set:
                        joined_lower = " ".join(c_tokens).strip().lower()
                        if joined_lower in question_line_token_set:
                            if DEBUG_VERBOSE:
                                print(f"[ANS-CONT-SKIP-Q] y={ly2} tokens={c_tokens}")
                            continue
                    answers.append(" ".join(t for t in c_tokens if t))
                    captured_line_ys.append(ly2)
                    if DEBUG_VERBOSE:
                        print(f"[ANS-CONT] y={ly2} tokens={c_tokens} window=({left_bound},{right_x}) base={y_answer} range=({cont_start},{cont_end})")
            # Final answer assembly: previously lines were joined with '\n'. Per new requirement,
            # join multi-line captures into a single line separated by spaces only.
            if answers:
                cleaned_lines = [" ".join(a.split()) for a in answers if a]
                # Deduplicate identical consecutive lines (e.g., Likert option captured twice: 'Not at all')
                deduped_lines: list[str] = []
                for cl in cleaned_lines:
                    if not deduped_lines or deduped_lines[-1] != cl:
                        deduped_lines.append(cl)
                # Additionally, if the entire concatenated text is an exact double repeat of the first half, collapse it.
                # This guards against two identical lines separated only by the join operation producing 'X X'.
                if len(deduped_lines) == 2 and deduped_lines[0] == deduped_lines[1]:
                    deduped_lines = [deduped_lines[0]]
                ans_text = " ".join(deduped_lines).strip()
            else:
                ans_text = ""
            # Dynamically extend the answer window's vertical end if we captured continuation
            # lines beyond the initial fixed band. This prevents the later strict horizontal
            # pruning (which re-OCRs only inside y_start..y_end) from discarding trailing
            # continuation tokens that we logically accepted above.
            dynamic_y_end = band_bottom_y
            if captured_line_ys:
                # Estimate per-line height: use last_line_y2 - last captured baseline if available
                try:
                    last_captured_y = max(captured_line_ys)
                    # Find the line object to get its height
                    last_ln = next((ln for ln in line_list if ln.get("y") == last_captured_y), None)
                    last_h = 0
                    if last_ln:
                        try:
                            last_h = int(last_ln.get("h") or 0)
                        except Exception:
                            last_h = 0
                    # Only extend if the last captured line baseline sits below band_bottom_y
                    if last_captured_y > band_bottom_y:
                        dynamic_y_end = last_captured_y + max(last_h, 0)
                except Exception:
                    pass
            # Word-level scan of the exact answer window for the reminder opt-in question
            if is_reminder_optin_q:
                try:
                    if DEBUG_VERBOSE and is_reminder_optin_q:
                        print(f"[REMINDER] window=({q_start_x},{right_x},{start_collect_y},{dynamic_y_end})")
                    window_word_count = 0
                    for ln_scan in line_list:
                        words_scan = ln_scan.get("words", [])
                        if not words_scan:
                            continue
                        for w in words_scan:
                            wx = w.get("x", -1)
                            wy = w.get("y", -1)
                            wh = w.get("h", 0) or 0
                            if wx is None or wy is None:
                                continue
                            if q_start_x <= wx <= right_x and start_collect_y <= wy <= dynamic_y_end:
                                window_word_count += 1
                                print(f"[REMINDER-WINDOW-WORD] text='{w.get('text','')}' x={wx} y={wy} h={wh}")
                    if window_word_count == 0:
                        print("[REMINDER-WINDOW-EMPTY] no_words_in_window")
                except Exception as e:
                    if DEBUG_VERBOSE:
                        print(f"[REMINDER-WINDOW-ERROR] {e}")
            # Under fixed band rule, y_end (window) remains band_bottom_y regardless of text extension.
            # Final answer debug removed (answers are in JSON). Enable here if needed.
            return ans_text, start_collect_y, dynamic_y_end

        # Iterate sections and attach answers
        # Precompute a set of lowercased token sequences for each question's own line(s) per section
        for sec in matches:
            qlist = sec.get("questions", [])
            # Determine section vertical bottom limit if region info was passed in
            sec_region_bottom = None
            if section_regions and isinstance(section_regions, dict):
                reg = section_regions.get(sec.get("section")) if sec.get("section") else None
                if reg and isinstance(reg, dict):
                    y2 = reg.get("y2")
                    if isinstance(y2, (int, float)):
                        sec_region_bottom = int(y2)
            # Build token sequence set for this section's question lines
            question_line_token_set = set()
            for q_line in qlist:
                for seg in (q_line.get("segments") or []):
                    if isinstance(seg, dict):
                        toks = [t.lower() for t in seg.get("tokens", []) if t]
                        if toks:
                            question_line_token_set.add(" ".join(toks))
            for q in qlist:
                q_start_x, q_start_y = q.get("position", [None, None])
                if q_start_x is None:
                    continue
                right_bound = compute_right_bound(qlist, q)
                ans_text, win_y1, win_y2 = collect_answer_lines(
                    q_start_x,
                    q_start_y,
                    q.get("segments"),
                    right_bound,
                    section_bottom_y=sec_region_bottom,
                    question_line_token_set=question_line_token_set,
                )
                # Store the answer window with the left margin applied (mirrors capture logic using left_bound)
                # This avoids clipping the first glyph during later strict horizontal pruning re-OCR (e.g., 'j' -> 'i').
                expanded_x_start = max(0, int(q_start_x) - ANSWER_LEFT_MARGIN_PX)
                q["answer_window"] = {
                    "x_start": expanded_x_start,
                    "y_start": int(win_y1),
                    "x_end": int(right_bound),
                    "y_end": int(win_y2),
                }
                if DEBUG_ANSWER_GEOMETRY:
                    print(f"[ANS-GEOM] q='{(q.get('question') or '')[:40]}' window=({expanded_x_start},{win_y1},{right_bound},{win_y2}) ans='{(ans_text or '')[:60]}'")
                if ans_text:
                    q["answer"] = ans_text
            # Pass 1.5: highlight fallback OCR for empty yes/no windows
            for q in qlist:
                try:
                    if q.get("answer"):
                        continue
                    aw = q.get("answer_window")
                    if not aw:
                        continue
                    q_text = q.get("question", "")
                    if not _looks_yes_no_question(q_text):
                        continue
                    # Attempt highlight fallback extraction
                    recovered = _yellow_highlight_ocr(pil_image, aw)
                    if recovered in ("Yes", "No"):
                        q["answer"] = recovered
                        q["answer_inferred"] = False
                        q["answer_method"] = "highlight_fallback"
                        print(f"[YESNO-FALLBACK] recovered='{recovered}' q='{q_text[:40]}...' window={aw}")
                        print(f"[YESNO] recovered='{recovered}' method=base")
                    else:
                        print(f"[YESNO-FALLBACK] no_recovery q='{q_text[:40]}...' window={aw}")
                        if DEBUG_VERBOSE:
                            print(f"[YESNO] base_no_recovery q='{q_text[:40]}...'")
                        # Sliding offset probes (upwards/nearby) in case the short 'Yes'/'No' sits closer than fixed offset
                        try:
                            segs = q.get("segments") or []
                            last_seg_y = None
                            for s in segs:
                                ly = s.get("line_y")
                                if isinstance(ly, (int, float)):
                                    last_seg_y = ly if last_seg_y is None else max(last_seg_y, ly)
                            band_h = aw["y_end"] - aw["y_start"]
                            if last_seg_y is not None and band_h > 0 and not q.get("answer"):
                                # Use configured slide offsets; include band_h if not already present
                                slide_offsets = list(YESNO_SLIDE_OFFSETS)
                                if band_h not in slide_offsets:
                                    slide_offsets.append(band_h)
                                slide_offsets = sorted(slide_offsets)
                                img_h = pil_image.height
                                for idx, off in enumerate(slide_offsets):
                                    y1 = int(last_seg_y + off)
                                    y2 = y1 + band_h
                                    if y2 > img_h:
                                        break
                                    slide_box = {
                                        "x_start": aw["x_start"],
                                        "y_start": y1,
                                        "x_end": aw["x_end"],
                                        "y_end": y2,
                                    }
                                    rec_slide = _yellow_highlight_ocr(pil_image, slide_box)
                                    if rec_slide in ("Yes", "No"):
                                        q["answer"] = rec_slide
                                        q["answer_inferred"] = False
                                        q["answer_method"] = f"highlight_slide_{off}"
                                        q["answer_confidence"] = YESNO_HIGHLIGHT_CONFIDENCE
                                        print(f"[YESNO-SLIDE] recovered='{rec_slide}' off={off} band={y1}-{y2}")
                                        print(f"[YESNO] recovered='{rec_slide}' method=slide off={off}")
                                        break
                                    else:
                                        print(f"[YESNO-SLIDE] miss off={off} band={y1}-{y2} rec='{rec_slide}'")
                                        if DEBUG_VERBOSE:
                                            print(f"[YESNO] slide_miss off={off}")
                        except Exception as e_slide:  # pragma: no cover
                            print(f"[YESNO-SLIDE-ERROR] {e_slide}")
                            if DEBUG_VERBOSE:
                                print(f"[YESNO-SLIDE-ERROR] {e_slide}")
                        # Vertical expansion probes (downward) only if still empty
                        if not q.get("answer"):
                            band_h = aw["y_end"] - aw["y_start"]
                            max_probes = YESNO_PROBE_MAX_BANDS
                            for probe_idx in range(1, max_probes + 1):
                                if q.get("answer"):
                                    break
                                probe_y1 = aw["y_start"] + probe_idx * band_h
                                probe_y2 = probe_y1 + band_h
                                probe_box = {
                                    "x_start": aw["x_start"],
                                    "y_start": probe_y1,
                                    "x_end": aw["x_end"],
                                    "y_end": probe_y2,
                                }
                                rec2 = _yellow_highlight_ocr(pil_image, probe_box)
                                if rec2 in ("Yes", "No"):
                                    q["answer"] = rec2
                                    q["answer_inferred"] = False
                                    q["answer_method"] = f"highlight_probe_{probe_idx}"
                                    q["answer_confidence"] = YESNO_HIGHLIGHT_CONFIDENCE
                                    print(f"[YESNO-PROBE] idx={probe_idx} recovered='{rec2}' band={probe_y1}-{probe_y2}")
                                    print(f"[YESNO] recovered='{rec2}' method=probe idx={probe_idx}")
                                    break
                                else:
                                    print(f"[YESNO-PROBE] idx={probe_idx} no_hit band={probe_y1}-{probe_y2} rec='{rec2}'")
                                    if DEBUG_VERBOSE:
                                        print(f"[YESNO] probe_miss idx={probe_idx}")
                        # Optional: save diagnostic crops if still empty
                        if not q.get("answer") and artifacts_dir:
                            try:
                                import os
                                os.makedirs(artifacts_dir, exist_ok=True)
                                diag_base = os.path.join(artifacts_dir, "yesno_diag")
                                for probe_idx in range(0, (YESNO_PROBE_MAX_BANDS if not q.get("answer") else 0) + 1):
                                    if probe_idx == 0:
                                        y1, y2 = aw["y_start"], aw["y_end"]
                                    else:
                                        y1 = aw["y_start"] + probe_idx * band_h
                                        y2 = y1 + band_h
                                    crop = pil_image.crop((aw["x_start"], y1, aw["x_end"], y2))
                                    crop.save(f"{diag_base}_probe{probe_idx}.png")
                                print(f"[YESNO-PROBE] saved_crops prefix={diag_base}_probe*.png")
                                if DEBUG_VERBOSE:
                                    print(f"[YESNO] saved_crops prefix={diag_base}_probe*.png")
                            except Exception as e2:  # pragma: no cover
                                print(f"[YESNO-PROBE-ERROR] save_crops {e2}")
                                if DEBUG_VERBOSE:
                                    print(f"[YESNO-PROBE-ERROR] save_crops {e2}")
                    if DEBUG_VERBOSE:
                        print(f"[YESNO-FALLBACK-ERROR] {e}")
                except Exception as e:
                    print(f"[YESNO-FALLBACK-ERROR] {e}")
            # Second pass: adjust x_end to ensure no overlap into a subsequent question column
            # If a window's x_end would intrude into another question's x_start (same horizontal row band), trim it.
            for q in qlist:
                try:
                    aw = q.get("answer_window")
                    if not aw:
                        continue
                    qx = aw.get("x_start")
                    q_end = aw.get("x_end")
                    qy = q.get("position", [0, 0])[1]
                    if qx is None or q_end is None:
                        continue
                    min_end = q_end
                    for other in qlist:
                        if other is q:
                            continue
                        ox, oy = other.get("position", [None, None])
                        if ox is None:
                            continue
                        if ox <= qx:
                            continue
                        # Same row (allow a slightly wider tolerance) so that columns on the same horizontal band constrain width
                        if abs(oy - qy) <= ANSWER_ROW_Y_TOLERANCE_PX * 2:
                            candidate = ox - ANSWER_COL_GAP_PX
                            if candidate < min_end and candidate >= qx:
                                min_end = candidate
                    if min_end < q_end:
                        aw["x_end"] = max(qx, min_end)
                except Exception:
                    pass
    except Exception:
        # Silently continue; answer field will just be absent
        pass
    # Global pass: ensure no answer_window overlaps any other answer_window on the same horizontal row (across sections)
    try:
        # Collect all questions with windows
        all_entries = []  # list of (question_dict, y, x_start)
        for sec in matches:
            for q in sec.get("questions", []):
                aw = q.get("answer_window")
                pos = q.get("position", [None, None])
                if aw and pos and pos[0] is not None and pos[1] is not None:
                    all_entries.append((q, int(pos[1]), int(aw.get("x_start", pos[0]))))
        # Group by approximate row using expanded tolerance
        all_entries.sort(key=lambda t: t[1])
        groups = []
        for entry in all_entries:
            placed = False
            for g in groups:
                # compare with first element's y in group
                if abs(g[0][1] - entry[1]) <= ANSWER_ROW_Y_TOLERANCE_PX * 2:
                    g.append(entry)
                    placed = True
                    break
            if not placed:
                groups.append([entry])
        # For each group, sort by x_start and trim overlaps
        for g in groups:
            g.sort(key=lambda t: t[2])
            for i in range(len(g) - 1):
                q_curr, y_curr, x_curr = g[i]
                q_next, y_next, x_next = g[i + 1]
                aw_curr = q_curr.get("answer_window")
                aw_next = q_next.get("answer_window")
                if not aw_curr or not aw_next:
                    continue
                x_end_curr = aw_curr.get("x_end")
                if x_end_curr is None:
                    continue
                # If current window intrudes into next window's start, trim it to (next_start - ANSWER_COL_GAP_PX)
                if x_end_curr >= x_next:
                    new_end = x_next - ANSWER_COL_GAP_PX
                    if new_end < aw_curr.get("x_start", x_curr):
                        new_end = aw_curr.get("x_start", x_curr)
                    aw_curr["x_end"] = new_end

        # Row-group cleanup (tolerant):
        # Some multi-column rows have slightly different y_start values (few px) due to OCR variance.
        # We cluster answer windows whose y_start are within tolerance, then collapse unintended multi-line
        # answers for non-narrative questions so they don't swallow the first line of the next column.
        try:
            # Gather (q, y_start)
            entries = []
            for sec in matches:
                for q in sec.get("questions", []):
                    aw = q.get("answer_window")
                    if aw and isinstance(aw.get("y_start"), int):
                        entries.append((q, aw.get("y_start")))
            entries.sort(key=lambda t: t[1])
            grouped: list[list[tuple]] = []
            for q, y in entries:
                placed = False
                for g in grouped:
                    if abs(g[0][1] - y) <= ANSWER_ROW_Y_TOLERANCE_PX * 2:  # broaden tolerance
                        g.append((q, y))
                        placed = True
                        break
                if not placed:
                    grouped.append([(q, y)])
            for g in grouped:
                if len(g) <= 1:
                    continue  # single column row
                # multi-column row -> enforce single line unless narrative
                for q, y in g:
                    ans = q.get("answer")
                    if not ans or "\n" not in ans:
                        continue
                    # Narrative detection (Medical Conditions)
                    try:
                        seg_tokens_lower = []
                        for seg in (q.get("segments") or []):
                            if isinstance(seg, dict):
                                seg_tokens_lower.extend([t.lower() for t in seg.get("tokens", [])])
                        is_narrative = ("medical" in seg_tokens_lower and "conditions" in seg_tokens_lower)
                    except Exception:
                        is_narrative = False
                    if is_narrative:
                        continue  # preserve
                    first_line = ans.split("\n", 1)[0].strip()
                    q["answer"] = first_line
                    aw = q.get("answer_window")
                    if aw:
                        aw["y_end"] = aw.get("y_start", y)  # collapse vertically
        except Exception:
            pass
    except Exception:
        pass
    # STRICT HORIZONTAL PRUNING: remove any tokens accidentally concatenated that lie outside stored x window
    try:
        for sec in matches:
            for q in sec.get("questions", []):
                aw = q.get("answer_window")
                ans = q.get("answer")
                if not aw or not ans:
                    continue
                x0 = aw.get("x_start"); x1 = aw.get("x_end")
                y0 = aw.get("y_start"); y1 = aw.get("y_end")
                if None in (x0, x1, y0, y1):
                    continue
                try:
                    from PIL import Image as _Image  # noqa: F401
                    import pytesseract as _pt
                    ocr_cfg = f"--psm {OCR_PSM}" + (f" -l {OCR_LANG}" if OCR_LANG else "")
                    crop = pil_image.crop((int(x0), int(y0), int(x1), int(y1)))
                    data = _pt.image_to_data(crop, output_type=_pt.Output.DICT, config=ocr_cfg)
                    kept_tokens = [data["text"][i].strip() for i in range(len(data["text"])) if data["text"][i].strip()]
                    if not kept_tokens:
                        continue
                    orig_tokens = ans.split()
                    removed = [tok for tok in orig_tokens if tok not in kept_tokens]
                    # Heuristic: don't replace if new tokens introduce evident OCR degradations (qmail->gmail, Davs->Days variants)
                    degraded = False
                    joined_new = " ".join(kept_tokens)
                    joined_orig = ans
                    # If original contains 'gmail' and new contains 'qmail', treat as degradation
                    if 'gmail' in joined_orig.lower() and 'qmail' in joined_new.lower():
                        degraded = True
                    # If original contains 'Days' and new contains 'Davs' treat as degradation
                    if 'days' in joined_orig.lower() and 'davs' in joined_new.lower():
                        degraded = True
                    # Only replace if we actually pruned something AND quality not degraded
                    if removed and not degraded:
                        q["answer"] = " ".join(kept_tokens)
                except Exception:
                    continue
    except Exception:
        pass
    # Post-processing: attempt checkbox-based inference for specific questions (e.g., reminder opt-in)
    try:
        if checkboxes:
            # Build a light index of checkboxes by y proximity for quick scan
            cb_list = [cb for cb in checkboxes if isinstance(cb, dict) and cb.get("position")]
            # Heuristic mapping: for a YES/NO pair, assume left -> Yes, right -> No (document if reversed later)
            for sec in matches:
                for q in sec.get("questions", []):
                    if q.get("answer"):
                        continue  # already has textual answer
                    segments = q.get("segments") or []
                    # Derive a flattened token list (lowercased) from segments to identify the reminder question
                    tokens_flat = []
                    for seg in segments:
                        if isinstance(seg, dict):
                            tokens_flat.extend([t.lower() for t in seg.get("tokens", [])])
                    if not tokens_flat:
                        continue
                    # Reuse earlier detection: look for at least two distinctive keywords
                    key_hits = 0
                    for kw in ("reminders?", "automatic", "appointment", "messages."):
                        base_kw = kw.rstrip('?').rstrip('.').lower()
                        if base_kw in tokens_flat or kw.rstrip('?').rstrip('.').lower() in tokens_flat:
                            key_hits += 1
                    is_reminder = key_hits >= 2 and ("reminders" in tokens_flat or any(t.startswith("reminder") for t in tokens_flat))
                    if not is_reminder:
                        continue
                    # Debug: enumerate nearby checkboxes for analysis
                    try:
                        dbg_band_low = (q.get("position", [0,0])[1] or 0) - 200
                        dbg_band_high = (q.get("position", [0,0])[1] or 0) + 400
                        dbg_cbs = [cb for cb in cb_list if dbg_band_low <= cb.get("position", [0,0,0,0])[1] <= dbg_band_high]
                        print(f"[REMINDER-INFER-DEBUG] q_y={q.get('position',[0,0])[1]} last_seg_y_candidate scan_n={len(dbg_cbs)}")
                        for dcb in dbg_cbs:
                            print(f"[REMINDER-INFER-CB] pos={dcb.get('position')} status={dcb.get('status')} conf={dcb.get('confidence')}")
                    except Exception:
                        pass
                    # Compute anchor y using last segment line_y if present
                    last_seg_y = None
                    for seg in segments:
                        if isinstance(seg, dict) and seg.get("line_y") is not None:
                            ly = seg.get("line_y")
                            if last_seg_y is None or ly > last_seg_y:
                                last_seg_y = ly
                    if last_seg_y is None:
                        # fallback to question position y
                        last_seg_y = q.get("position", [0, 0])[1]
                    # Collect candidate checkboxes within vertical band around last segment
                    V_TOL_TOP = 20
                    V_TOL_BOTTOM = 70  # allow some space below if boxes sit slightly lower
                    y_low = last_seg_y - V_TOL_TOP
                    y_high = last_seg_y + V_TOL_BOTTOM
                    # Horizontal window: start near question x_start - small margin to page right
                    qx = q.get("position", [0, 0])[0] or 0
                    # We don't know exact page width here; rely on max of checkbox x positions
                    max_cb_x = max((cb["position"][0] for cb in cb_list), default=qx + 200)
                    # Filter checkboxes in band
                    nearby = []
                    for cb in cb_list:
                        cx, cy, cw, ch = cb.get("position")
                        if y_low <= cy <= y_high:
                            nearby.append(cb)
                    # Sort left -> right
                    nearby.sort(key=lambda c: c["position"][0])
                    # Heuristic: choose first two distinct-x boxes as the YES/NO pair
                    pair = []
                    seen_x = set()
                    for cb in nearby:
                        x = cb["position"][0]
                        if all(abs(x - ex) >= 5 for ex in seen_x):
                            pair.append(cb)
                            seen_x.add(x)
                        if len(pair) == 2:
                            break
                    if len(pair) != 2:
                        # Not enough boxes to infer
                        print(f"[REMINDER-INFER] insufficient_pair candidates={len(pair)} nearby={len(nearby)} y_band=({y_low},{y_high})")
                        continue
                    # Determine ticked states
                    statuses = [cb.get("status") for cb in pair]
                    ticked_indices = [i for i,s in enumerate(statuses) if s == "ticked"]
                    inferred_answer = None
                    reason = None
                    if len(ticked_indices) == 1:
                        # Mapping assumption: left (index 0) => Yes, right (index 1) => No
                        inferred_answer = "Yes" if ticked_indices[0] == 0 else "No"
                        reason = f"single_ticked_index={ticked_indices[0]}"
                    elif len(ticked_indices) == 2:
                        # Both ticked - ambiguous; represent as 'Yes' (conservative) but flag ambiguity
                        inferred_answer = "Yes"
                        reason = "both_ticked"
                    else:  # none ticked
                        inferred_answer = None
                        reason = "none_ticked"
                    if inferred_answer:
                        q["answer"] = inferred_answer
                        q["answer_inferred"] = True
                        q["answer_source"] = "checkbox"
                        q["answer_confidence"] = 0.6 if reason == "single_ticked_index=0" else 0.5
                        print(f"[REMINDER-INFER] y_anchor={last_seg_y} boxes={[cb['position'] for cb in pair]} statuses={statuses} -> answer={inferred_answer} reason={reason}")
                    else:
                        print(f"[REMINDER-INFER] y_anchor={last_seg_y} boxes={[cb['position'] for cb in pair]} statuses={statuses} -> no_inference ({reason})")
    except Exception as e:
        print(f"[REMINDER-INFER-ERROR] {e}")
    # Secondary inference: if provider question is answered, infer 'Yes' for prior opt-in question
    try:
        for sec in matches:
            qs = sec.get("questions", [])
            for i, q in enumerate(qs):
                text = q.get("question", "").lower()
                if "appointment" in text and "reminder" in text and not q.get("answer"):
                    # Look ahead to next question for provider
                    if i + 1 < len(qs):
                        next_q = qs[i+1]
                        next_text = (next_q.get("question") or "").lower()
                        if next_q.get("answer") and next_text.startswith("if yes"):
                            q["answer"] = "Yes"
                            q["answer_inferred"] = True
                            q["answer_source"] = "followup_inference"
                            q["answer_confidence"] = 0.8
                            print(f"[REMINDER-INFER-FOLLOWUP] Inferred 'Yes' based on answered follow-up provider question at index {i+1}")
    except Exception as e:
        print(f"[REMINDER-INFER-FOLLOWUP-ERROR] {e}")
    # Field-specific sanitization (lightweight, post-pruning)
    try:
        zip_re = re.compile(r"^\d{5}(-\d{4})?$")
        for sec in matches:
            for q in sec.get("questions", []):
                qtext = (q.get("question") or "").lower()
                ans = q.get("answer")
                if not ans:
                    continue
                # Zip Code: keep only first ZIP-looking token if extra tokens (e.g., leaked email) present
                if "zip" in qtext and "code" in qtext:
                    tokens = ans.split()
                    if tokens:
                        # find first token that matches zip pattern
                        for tok in tokens:
                            if zip_re.match(tok):
                                if tok != ans:
                                    q["answer"] = tok
                                break
    except Exception:
        pass
    return matches
