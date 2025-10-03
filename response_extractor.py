import pytesseract
import unicodedata
from collections import defaultdict
from constants import ANCHOR_OFFSET_PX, LABEL_MULTILINE_BASE_X_TOLERANCE, LABEL_MULTILINE_MAX_LOOKAHEAD


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


def _build_lines_with_geometry(pil_image):
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
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
def _match_section_anchors(pil_image, sections: list[dict]):
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
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
            }
        else:
            lines[key]["text"] += " " + word
    sorted_lines = sorted(lines.values(), key=lambda l: l["y"])

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
            line_s = clean_line_preserve_case(raw)
            cleaned_expected = clean_line_preserve_case(name)
            if len(cleaned_expected) <= 3:
                tokens = [letters_only_token(tok) for tok in raw.split()]
                exp_short, i_mask_short = _build_expected_masked_upper(letters_only_token(name))
                for tok in tokens:
                    if _flex_equal_upper(exp_short, i_mask_short, tok):
                        # Do not reuse an anchor y already claimed
                        if line["y"] in claimed_anchor_ys:
                            continue
                        anchor_y = line["y"]
                        # anchor matched (token flex) – verbose log removed
                        break
                if anchor_y is not None:
                    break
            else:
                if _flex_contains_upper(exp_s, i_mask, line_s):
                    # Do not reuse an anchor y already claimed
                    if line["y"] in claimed_anchor_ys:
                        continue
                    anchor_y = line["y"]
                    # anchor matched – verbose log removed
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
def _match_questions_like_labels(pil_image, questions: list[str]):
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)

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

    # OCR-side normalization that preserves visible punctuation; only strips control chars and uppercases
    def _ocr_norm_preserve_punct_upper(text: str) -> str:
        t = unicodedata.normalize('NFKD', text)
        t = ''.join(c for c in t if unicodedata.category(c)[0] != 'C')
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
def match_sections_and_questions(pil_image, sections: list[dict], section_regions: dict | None = None):
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
        anchors = _match_section_anchors(pil_image, sections)
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
        qhits = _match_questions_like_labels(pil_image, qs)
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
def extract_responses_from_page(pil_image, sections: list, section_regions: dict | None = None, artifacts_dir: str | None = None):
    """
    For now, return just section/question matches with positions (no answers),
    using label-like matching and detailed multiline logs.
    """
    matches = match_sections_and_questions(pil_image, sections, section_regions=section_regions)
    # Shape similar enough for extractor to include under "responses"
    return matches
