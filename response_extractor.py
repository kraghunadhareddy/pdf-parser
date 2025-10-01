import pytesseract
import unicodedata
import re
from collections import defaultdict


def _normalize_word_preserve(text: str) -> str:
    """Normalize by removing control chars and common separators but preserve case for case-sensitive matching."""
    t = unicodedata.normalize('NFKD', text)
    t = ''.join(c for c in t if unicodedata.category(c)[0] != 'C')
    t = t.replace('/', '').replace(' ', '').replace('-', '')
    t = re.sub(r'^[^a-zA-Z]+', '', t)
    t = re.sub(r'[^a-zA-Z]+$', '', t)
    return t


def _build_expected_masked_preserve(text: str):
    """Build expected string and mask for positions where uppercase 'I' in expected permits I/L/l/1 in candidate.
    Do NOT change case of expected; compare case-sensitively except at masked positions.
    """
    t = unicodedata.normalize('NFKD', text)
    t = ''.join(c for c in t if unicodedata.category(c)[0] != 'C')
    t = t.replace('/', '').replace(' ', '').replace('-', '')
    t = re.sub(r'^[^a-zA-Z]+', '', t)
    t = re.sub(r'[^a-zA-Z]+$', '', t)
    expected_s = []
    i_mask = set()
    for idx, ch in enumerate(t):
        if ch == 'I':  # only uppercase 'I' in expected triggers flexibility
            i_mask.add(idx)
        expected_s.append(ch)
    return ''.join(expected_s), i_mask


def _flex_equal_cs(expected_s: str, i_mask: set[int], candidate_s: str) -> bool:
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


def _flex_contains_cs(expected_s: str, i_mask: set[int], haystack_s: str) -> bool:
    m, n = len(expected_s), len(haystack_s)
    if m == 0:
        return True
    for i in range(0, n - m + 1):
        if _flex_equal_cs(expected_s, i_mask, haystack_s[i:i + m]):
            return True
    return False


def _ocr_lines(pil_image):
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    # Build logical lines preserving original tokens with geometry
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
    # Sort tokens left-to-right per line and then lines top-to-bottom
    for ln in lines.values():
        ln["words"].sort(key=lambda t: t["x"])
    sorted_lines = sorted(lines.values(), key=lambda ln: ln["y"])
    return sorted_lines


def get_question_positions(pil_image, expected_questions: list[str], debug: bool = True):
    """Replicate extractor.get_label_positions logic to find (x,y) start of questions.
    Returns dict: {question_text: [(x,y), ...]}
    """
    ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    question_positions = defaultdict(list)

    def normalize_text_preserve(text: str) -> str:
        t = unicodedata.normalize('NFKD', text)
        t = ''.join(c for c in t if unicodedata.category(c)[0] != 'C')
        t = t.replace('/', '').replace(' ', '').replace('-', '')
        t = re.sub(r'^[^a-zA-Z]+', '', t)
        t = re.sub(r'[^a-zA-Z]+$', '', t)
        return t

    def clean_label_sequence(seq):
        normed = [normalize_text_preserve(s) for s in seq]
        joined = ''.join(normed)
        joined = unicodedata.normalize('NFKD', joined)
        joined = ''.join(c for c in joined if unicodedata.category(c)[0] != 'C')
        joined = joined.replace(' ', '')
        return joined

    def build_expected_masked_preserve_inner(text):
        return _build_expected_masked_preserve(text)

    def flex_equal_cs_inner(expected_s, i_mask, candidate_s):
        return _flex_equal_cs(expected_s, i_mask, candidate_s)

    def flex_contains_cs_inner(expected_s, i_mask, haystack_s):
        return _flex_contains_cs(expected_s, i_mask, haystack_s)

    # Build tokens
    tokens = []
    for i in range(len(ocr_data["text"])):
        word = ocr_data["text"][i].strip()
        if not word:
            continue
        norm = normalize_text_preserve(word)
        tokens.append({
            "text": norm,
            "orig": word,
            "x": ocr_data["left"][i],
            "y": ocr_data["top"][i]
        })

    # Build lines to support multi-line lookahead
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

    for q in expected_questions:
        q_words = q.split()
        if not q_words:
            continue
        first_word = normalize_text_preserve(q_words[0])
        last_word = normalize_text_preserve(q_words[-1])
        q_clean = clean_label_sequence(q_words)
        exp_s, i_mask = build_expected_masked_preserve_inner(q)
        n = len(tokens)
        max_len = len(q_words)
        found = False
        # Try single-line contiguous tokens first (page-contiguous order)
        for i in range(n):
            for j in range(i, min(i + max_len, n)):
                seq = tokens[i:j+1]
                if not seq:
                    continue
                if first_word in normalize_text_preserve(seq[0]["orig"]) and last_word in normalize_text_preserve(seq[-1]["orig"]):
                    seq_clean = clean_label_sequence([t["orig"] for t in seq])
                    if flex_contains_cs_inner(exp_s, i_mask, seq_clean):
                        question_positions[q].append((seq[0]["x"], seq[0]["y"]))
                        found = True
        # Multi-line lookahead similar to extractor
        if not found:
            base_x_tolerance = 160
            max_lookahead = 5

            def try_multiline(lbl_words_seq):
                for i, line in enumerate(lines):
                    words = line["words"]
                    if not words:
                        continue
                    best_start = None
                    best_matched_here = 0
                    for start_idx in range(len(words)):
                        matched_here = 0
                        k = start_idx
                        for lbl_idx in range(0, len(lbl_words_seq)):
                            if k >= len(words):
                                break
                            wu, w_mask = build_expected_masked_preserve_inner(lbl_words_seq[lbl_idx])
                            tok_clean = normalize_text_preserve(words[k]["text"])
                            if flex_contains_cs_inner(wu, w_mask, tok_clean):
                                matched_here += 1
                                k += 1
                            else:
                                break
                        if matched_here > best_matched_here:
                            best_matched_here = matched_here
                            best_start = start_idx
                        if matched_here == len(lbl_words_seq):
                            break
                    if best_matched_here == 0:
                        continue
                    start_x = words[best_start]["x"]
                    start_y = words[best_start]["y"]
                    x_ref = start_x
                    curr_lbl_idx = best_matched_here
                    curr_idx = i
                    matched_all = (curr_lbl_idx == len(lbl_words_seq))
                    lookahead_used = 0
                    x_tolerance = base_x_tolerance
                    while not matched_all and lookahead_used < max_lookahead:
                        if curr_idx + 1 >= len(lines):
                            break
                        next_line = lines[curr_idx + 1]
                        next_tokens = next_line["words"]
                        if not next_tokens:
                            break
                        expected_word = lbl_words_seq[curr_lbl_idx]
                        wuN, w_maskN = build_expected_masked_preserve_inner(expected_word)
                        candidate_indices = [idx for idx, tok in enumerate(next_tokens)
                                             if abs(tok["x"] - x_ref) <= x_tolerance and flex_contains_cs_inner(wuN, w_maskN, normalize_text_preserve(tok["text"]))]
                        if not candidate_indices:
                            candidate_indices = [idx for idx, tok in enumerate(next_tokens)
                                                 if flex_contains_cs_inner(wuN, w_maskN, normalize_text_preserve(tok["text"]))]
                        if not candidate_indices:
                            break
                        best_line_match = 0
                        best_line_start = None
                        for ci in candidate_indices:
                            matched_in_line = 0
                            k = ci
                            for lbl_idx in range(curr_lbl_idx, len(lbl_words_seq)):
                                if k >= len(next_tokens):
                                    break
                                wu2, w2_mask = build_expected_masked_preserve_inner(lbl_words_seq[lbl_idx])
                                if flex_contains_cs_inner(wu2, w2_mask, normalize_text_preserve(next_tokens[k]["text"])):
                                    matched_in_line += 1
                                    k += 1
                                else:
                                    break
                            if matched_in_line > best_line_match:
                                best_line_match = matched_in_line
                                best_line_start = ci
                        if best_line_match == 0:
                            break
                        curr_lbl_idx += best_line_match
                        curr_idx += 1
                        lookahead_used += 1
                        x_ref = next_tokens[best_line_start]["x"]
                        matched_all = (curr_lbl_idx == len(lbl_words_seq))
                    if matched_all:
                        return (start_x, start_y)
                return None

            pos = try_multiline(q_words)
            if pos is not None:
                question_positions[q].append(pos)
                if debug:
                    print(f"[Q-DETECT] '{q}' matched starting at y={pos[1]}")
            else:
                for skip in range(1, min(5, len(q_words))):
                    suffix = q_words[skip:]
                    pos2 = try_multiline(suffix)
                    if pos2 is not None:
                        question_positions[q].append(pos2)
                        if debug:
                            print(f"[Q-DETECT] '{q}' matched (skipped first {skip} words) at y={pos2[1]}")
                        break
        # Partial multi-line fallback: accept high-overlap matches if exact fails
        if not question_positions[q]:
            def try_multiline_partial(lbl_words_seq, min_ratio=0.7, min_words=4):
                total = len(lbl_words_seq)
                for i, line in enumerate(lines):
                    words = line["words"]
                    if not words:
                        continue
                    # Choose start with best contiguous match on this line
                    best_start = None
                    best_matched_here = 0
                    for start_idx in range(len(words)):
                        matched_here = 0
                        k = start_idx
                        for lbl_idx in range(0, len(lbl_words_seq)):
                            if k >= len(words):
                                break
                            wu, w_mask = build_expected_masked_preserve_inner(lbl_words_seq[lbl_idx])
                            tok_clean = normalize_text_preserve(words[k]["text"])
                            if flex_contains_cs_inner(wu, w_mask, tok_clean):
                                matched_here += 1
                                k += 1
                            else:
                                break
                        if matched_here > best_matched_here:
                            best_matched_here = matched_here
                            best_start = start_idx
                    if best_matched_here == 0:
                        continue
                    start_x = words[best_start]["x"]
                    start_y = words[best_start]["y"]
                    x_ref = start_x
                    curr_lbl_idx = best_matched_here
                    curr_idx = i
                    matched_count = best_matched_here
                    lookahead_used = 0
                    while lookahead_used < max_lookahead and curr_lbl_idx < len(lbl_words_seq):
                        if curr_idx + 1 >= len(lines):
                            break
                        next_line = lines[curr_idx + 1]
                        next_tokens = next_line["words"]
                        if not next_tokens:
                            break
                        expected_word = lbl_words_seq[curr_lbl_idx]
                        wuN, w_maskN = build_expected_masked_preserve_inner(expected_word)
                        candidate_indices = [idx for idx, tok in enumerate(next_tokens)
                                             if abs(tok["x"] - x_ref) <= base_x_tolerance and flex_contains_cs_inner(wuN, w_maskN, normalize_text_preserve(tok["text"]))]
                        if not candidate_indices:
                            candidate_indices = [idx for idx, tok in enumerate(next_tokens)
                                                 if flex_contains_cs_inner(wuN, w_maskN, normalize_text_preserve(tok["text"]))]
                        if not candidate_indices:
                            break
                        best_line_match = 0
                        best_line_start = None
                        for ci in candidate_indices:
                            matched_in_line = 0
                            k = ci
                            for lbl_idx in range(curr_lbl_idx, len(lbl_words_seq)):
                                if k >= len(next_tokens):
                                    break
                                wu2, w2_mask = build_expected_masked_preserve_inner(lbl_words_seq[lbl_idx])
                                if flex_contains_cs_inner(wu2, w2_mask, normalize_text_preserve(next_tokens[k]["text"])):
                                    matched_in_line += 1
                                    k += 1
                                else:
                                    break
                            if matched_in_line > best_line_match:
                                best_line_match = matched_in_line
                                best_line_start = ci
                        if best_line_match == 0:
                            break
                        matched_count += best_line_match
                        curr_lbl_idx += best_line_match
                        curr_idx += 1
                        lookahead_used += 1
                        x_ref = next_tokens[best_line_start]["x"]
                    ratio = matched_count / total if total else 0
                    if matched_count >= min_words and ratio >= min_ratio:
                        return (start_x, start_y)
                return None

            posp = try_multiline_partial(q_words)
            if posp is not None:
                question_positions[q].append(posp)
                if debug:
                    print(f"[Q-DETECT] '{q}' partially matched at y={posp[1]} (high-overlap)")
        if not question_positions[q]:
            print(f"[Q-MISS] Question not found by OCR: '{q}'")

    return question_positions


def _find_question_span_in_line(question: str, line_words: list[dict]):
    # Locate the contiguous span of this question in the line.
    # Returns dict with start_x, end_x, start_word_index, end_word_index (inclusive), matched_count
    # Prepare expected words (drop non-letter-only placeholders like '#')
    raw_q_words = question.split()
    q_words = []
    q_words_mask = []
    for qw in raw_q_words:
        es, im = _build_expected_masked_preserve(qw)
        if es:  # keep only words that have letters after normalization
            q_words.append(qw)
            q_words_mask.append((es, im))
    if not q_words:
        return None
    best = None
    best_len = 0
    single_word = (len(q_words) == 1)
    for start in range(len(line_words)):
        idx_q = 0
        k = start
        while idx_q < len(q_words) and k < len(line_words):
            exp_s, i_mask = q_words_mask[idx_q]
            cand_norm = _normalize_word_preserve(line_words[k]["text"])
            ok = _flex_equal_cs(exp_s, i_mask, cand_norm) if single_word else _flex_contains_cs(exp_s, i_mask, cand_norm)
            if ok:
                idx_q += 1
                k += 1
            else:
                break
        if idx_q > best_len:
            best_len = idx_q
            best = (start, k)  # [start, end)
        if best_len == len(q_words):
            break
    if best is None or best_len == 0:
        return None
    start_idx = best[0]
    end_idx = max(best[1] - 1, best[0])
    start_x = line_words[start_idx]["x"]
    end_tok = line_words[end_idx]
    end_x = end_tok["x"] + end_tok["w"]
    return {
        "start_x": start_x,
        "end_x": end_x,
        "start_word_index": start_idx,
        "end_word_index": end_idx,
        "matched_count": best_len,
    }


def extract_responses_from_page(pil_image, sections: list, section_regions: dict | None = None, artifacts_dir: str | None = None):
    """
    Extract free-text responses for sections that define a "questions" array in sections.json.

    Returns a list of { section: str, questions: [{question, answer}] } for the given page.
    """
    lines = _ocr_lines(pil_image)
    debug = True  # local verbose logging for response extraction

    # Precompute normalized full-line strings for faster matching
    materialized = []
    for ln in lines:
        raw = ' '.join(w["text"] for w in ln["words"]) if ln["words"] else ''
        # Build line-level normalized string by joining token-level normalization (case-preserving)
        norm = ''.join(_normalize_word_preserve(w["text"]) for w in ln["words"]) if ln["words"] else ''
        materialized.append({"y": ln["y"], "words": ln["words"], "raw": raw, "norm": norm})
    # Flatten all tokens with geometry for y-band queries
    all_tokens = []
    for ln in materialized:
        for t in (ln.get("words") or []):
            all_tokens.append({"text": t["text"], "x": t["x"], "y": t["y"]})

    responses = []
    for sec in sections:
        if "questions" not in sec or not sec["questions"]:
            continue
        sec_name = sec["section_name"]
        region = section_regions.get(sec_name) if section_regions else None
        if section_regions is not None and region is None:
            # If section regions are provided but this section wasn't anchored on this page,
            # skip to avoid picking up spurious answers from unrelated areas.
            print(f"[Q-SKIP] Section '{sec_name}' has questions but no region on this page; skipping response extraction.")
            continue
        qas = []

        # Use the same matching as labels to locate question anchors
        # Disable internal debug to avoid confusing y logs; we'll emit a single canonical detect log below
        qpos = get_question_positions(pil_image, sec["questions"], debug=False)
        # Map detected positions to line indices and extract spans per line
        line_index_by_y = list(range(len(materialized)))
        # For each question with positions, pick the first occurrence and map to a line
        spans_by_line = defaultdict(list)
        # Also keep track of question start_x per line to stop answer capture at the next question on that line
        q_starts_by_line = defaultdict(list)
        # Precompute normalized forms of questions and vocabulary to filter out question fragments in answers
        expected_q_norms = set(_normalize_word_preserve(q).upper() for q in sec["questions"] if q)
        expected_q_tokens = set()
        for q in sec["questions"]:
            for w in q.split():
                nw = _normalize_word_preserve(w).upper()
                if len(nw) >= 3:
                    expected_q_tokens.add(nw)
        for q in sec["questions"]:
            positions = qpos.get(q, [])
            if not positions:
                continue
            # Use the earliest detection by y as the anchor
            px, py = sorted(positions, key=lambda p: (p[1], p[0]))[0]
            anchor_y = py
            # Strong banding: restrict to OCR lines in a tight band around anchor_y; otherwise synthesize a line at anchor_y
            band_low, band_high = anchor_y - 2, anchor_y + 60
            inband_candidates = []
            inband_with_span = []
            all_candidates = []
            for i, ln in enumerate(materialized):
                if region and not (region["y1"] <= ln["y"] <= region["y2"]):
                    continue
                all_candidates.append((i, ln["y"]))
                if band_low <= ln["y"] <= band_high:
                    inband_candidates.append((i, ln["y"]))
                    if ln.get("words"):
                        sp = _find_question_span_in_line(q, ln["words"]) or None
                        if sp:
                            inband_with_span.append((i, ln["y"], sp))
            best_idx = None
            best_span = None
            if inband_with_span:
                # Choose the top-most line in band that has a span for the question
                best_idx, _, best_span = min(inband_with_span, key=lambda c: c[1])
            elif inband_candidates:
                # Choose the first line in the band, anchor_x to line end as span
                best_idx = min(inband_candidates, key=lambda c: c[1])[0]
                ln = materialized[best_idx]
                end_x = ln["words"][-1]["x"] + ln["words"][-1]["w"] if ln.get("words") else (px + 400)
                best_span = {"start_x": px, "end_x": end_x, "matched_count": 0, "start_word_index": 0, "end_word_index": 0}
            else:
                # No OCR line near anchor; synthesize a line at anchor_y to avoid borrowing an earlier header line
                materialized.append({"y": anchor_y, "words": []})
                best_idx = len(materialized) - 1
                best_span = {"start_x": px, "end_x": px + 400, "matched_count": 0, "start_word_index": 0, "end_word_index": 0}
            ln = materialized[best_idx]
            spans_by_line[best_idx].append({"question": q, "anchor_y": anchor_y, **best_span})
            q_starts_by_line[best_idx].append(best_span["start_x"])

        # For each line with one or more questions, compute x spans per the requested rules and read answers.
        for i in sorted(spans_by_line.keys()):
            item = materialized[i]
            spans = sorted(spans_by_line[i], key=lambda s: s["start_x"])
            # Determine a safe right edge for the anchor line even if it has no tokens (synthetic or OCR miss)
            if item.get("words"):
                last_token = item["words"][-1]
                line_end_x = last_token["x"] + last_token["w"]
            else:
                # Prefer the next line's end, else previous line's end, else an anchor-based fallback
                def line_end(idx):
                    ln = materialized[idx]
                    if ln.get("words"):
                        lt = ln["words"][-1]
                        return lt["x"] + lt["w"]
                    return None
                line_end_x = None
                if i + 1 < len(materialized):
                    line_end_x = line_end(i + 1)
                if line_end_x is None and i - 1 >= 0:
                    line_end_x = line_end(i - 1)
                if line_end_x is None:
                    # Fall back to the furthest end_x among spans on this line, or anchor start + 400px
                    span_max_end = max((sp.get("end_x", 0) for sp in spans if sp), default=0)
                    span_min_start = min((sp.get("start_x", 0) for sp in spans if sp), default=0)
                    line_end_x = span_max_end if span_max_end > 0 else (span_min_start + 400)
            # Slack: keep a modest vertical allowance below region in case the region is too tight
            y_slack = 280  # px for 300 DPI
            # Tolerance bands before next content:
            #  - Absolute band below the question line (start looking ~20–40 px below)
            #  - Relative band just above each subsequent next line (see per-iteration below)
            y_preline_min = 12
            y_preline_max = 64
            y_preline_tol = 32
            # Start tolerance for left bound as requested
            start_pad = 8
            
            # Helper to collect raw tokens from a given line within [x1, x2]
            def collect_tokens_from_line(line_obj, x1, x2, pad=0, relax_upper=False):
                if not line_obj or not line_obj.get("words"):
                    return []
                xa = max(0, x1 - pad)
                xb = x2 + pad if not relax_upper else 10_000  # effectively to the right edge
                return [tok["text"] for tok in line_obj["words"] if xa <= tok["x"] <= xb]

            # Clean tokens by removing question vocabulary and dropping obvious boilerplate
            def finalize_answer_from_tokens(tokens_list):
                if not tokens_list:
                    return ""
                kept = []
                short_whitelist = {"YES", "NO", "NA", "NONE"}
                for t in tokens_list:
                    norm = _normalize_word_preserve(t).upper()
                    # Keep numbers/dates by raw token digits first, then short categorical, then non-question vocab
                    has_digit = any(ch.isdigit() for ch in t)
                    if has_digit:
                        kept.append(t)
                        continue
                    # Allow common short categorical answers like Yes/No/NA even if shorter than 3
                    if norm in short_whitelist:
                        kept.append(t)
                        continue
                    if len(norm) >= 3 and norm not in expected_q_tokens:
                        kept.append(t)
                # Deduplicate identical tokens while preserving order (fixes cases like "3 3")
                deduped = []
                seen = set()
                for t in kept:
                    key = _normalize_word_preserve(t).upper()
                    if key not in seen:
                        seen.add(key)
                        deduped.append(t)
                ans = ' '.join(deduped).strip()
                # As a final guard, if the normalized answer looks like question text, drop it
                ans_norm = _normalize_word_preserve(ans).upper()
                if ans_norm and any((ans_norm in qn) or (qn in ans_norm) for qn in expected_q_norms):
                    return ""
                return ans

            # Helper: find continuation of the question on the next k lines; pick the line with the
            # longest match (by matched_count) to establish the final question end for multiline prompts.
            def longest_continuation_after(anchor_index, question):
                best = {"line": anchor_index, "span": _find_question_span_in_line(question, materialized[anchor_index]["words"]) }
                best_count = best["span"]["matched_count"] if best["span"] else 0
                # consider up to next 3 lines for continuation
                for off in (1, 2, 3):
                    j = anchor_index + off
                    if j >= len(materialized):
                        break
                    ln = materialized[j]
                    if region and not (region["y1"] <= ln["y"] <= region["y2"]) and not (ln["y"] <= materialized[anchor_index]["y"] + y_slack):
                        # outside region and slack; unlikely to be continuation
                        continue
                    sp_try = _find_question_span_in_line(question, ln["words"]) if ln["words"] else None
                    if sp_try and sp_try.get("matched_count", 0) > best_count:
                        best = {"line": j, "span": sp_try}
                        best_count = sp_try["matched_count"]
                return best

            for idx, sp in enumerate(spans):
                q = sp["question"]
                # Determine the final question span considering possible multi-line continuation.
                cont = longest_continuation_after(i, q)
                last_q_line_index = cont["line"]
                # Base span on the line we are currently logging for detection
                # Start position: start of the question match
                x_start = max(0, sp["start_x"] - start_pad)
                # End position: end of the question match on the best (longest) line, then extended to the
                # start of the next word on that same line if available.
                if cont["span"]:
                    end_idx = cont["span"]["end_word_index"]
                    ln_last = materialized[last_q_line_index]
                    words_last = ln_last["words"]
                    # end of matched token
                    end_x_raw = cont["span"]["end_x"]
                    # start of next word after the match on the last matched line
                    next_word_x = words_last[end_idx + 1]["x"] if (end_idx + 1) < len(words_last) else (words_last[-1]["x"] + words_last[-1]["w"])
                    x_end = next_word_x
                else:
                    x_end = line_end_x
                # If there is another question on the same anchor line to the right, cap the end at its start
                if idx + 1 < len(spans):
                    x_end = min(x_end, spans[idx + 1]["start_x"] - 6)
                # Also cap by any other question on any OCR line that shares the same y (some PDFs split columns/blocks)
                same_y_next_starts = []
                sp_start_raw = sp.get("start_x")
                for j_all, s_list in spans_by_line.items():
                    if materialized[j_all]["y"] == item["y"]:
                        for s2 in s_list:
                            if s2 is sp:
                                continue
                            sx = s2.get("start_x")
                            if sx is not None and sp_start_raw is not None and sx > sp_start_raw + 2:
                                same_y_next_starts.append(sx)
                if same_y_next_starts:
                    x_end = min(x_end, min(same_y_next_starts) - 6)
                # Log detection using the requested span definition
                if debug:
                    ay = sp.get("anchor_y")
                    print(f"[Q-DETECT] '{q}' anchor_y={ay} line_y={item['y']} span=({x_start},{x_end})")
                # Multi-line capture only: start from the line after the last matched portion of the question; continue until a blank line within x-window.
                j = last_q_line_index + 1
                captured_tokens = []
                # 0) Absolute preline band just below the question line, before the first next line
                if j < len(materialized):
                    cand0 = materialized[j]
                    # region/slack check for first next line; we apply same gating to the preline band
                    if (not region) or (region["y1"] <= cand0["y"] <= region["y2"]) or (cand0["y"] <= materialized[last_q_line_index]["y"] + y_slack):
                        # Determine right bound for this step based on the first next line
                        if cand0.get("words"):
                            last_tok0 = cand0["words"][-1]
                            rb0 = last_tok0["x"] + last_tok0["w"]
                        else:
                            rb0 = line_end_x
                        if q_starts_by_line.get(j):
                            next_q_xs0 = sorted([sx for sx in q_starts_by_line[j] if sx > x_start + 2])
                            if next_q_xs0:
                                rb0 = min(rb0, next_q_xs0[0] - 6)
                        # Compute absolute band bounds relative to question line y
                        last_y = materialized[last_q_line_index]["y"]
                        band0_low = max(0, last_y + y_preline_min)
                        band0_high = min(last_y + y_preline_max, cand0["y"] - 1)
                        if band0_high >= band0_low:
                            pre_tokens0 = [t["text"] for t in all_tokens
                                           if band0_low <= t["y"] <= band0_high and x_start - start_pad <= t["x"] <= rb0]
                            if pre_tokens0:
                                captured_tokens.extend(pre_tokens0)
                                if debug:
                                    print(f"[Q-ANSWER] '{q}' preline band y=[{band0_low},{band0_high}] x_span=({x_start},{rb0}) tokens={len(pre_tokens0)}")
                            else:
                                if debug:
                                    print(f"[Q-ANSWER] '{q}' preline band y=[{band0_low},{band0_high}] x_span=({x_start},{rb0}) tokens=0")
                    # Extended preline window: if the immediate next OCR lines have the same y as the question,
                    # scan a wider band down to the first line with y > question_y (or up to ~140 px).
                    base_y = materialized[last_q_line_index]["y"]
                    k = j
                    first_below_idx = None
                    while k < len(materialized) and materialized[k]["y"] <= base_y:
                        k += 1
                    if k < len(materialized):
                        first_below_idx = k
                        below_y = materialized[first_below_idx]["y"]
                        band_ext_low = max(0, base_y + y_preline_min)
                        band_ext_high = min(base_y + 140, below_y - 1)
                        if band_ext_high >= band_ext_low:
                            # Use the previously computed x_end for this question to bound the right edge
                            ext_rb = x_end
                            ext_pre_tokens = [t["text"] for t in all_tokens
                                              if band_ext_low <= t["y"] <= band_ext_high and x_start - start_pad <= t["x"] <= ext_rb]
                            if ext_pre_tokens:
                                captured_tokens.extend(ext_pre_tokens)
                                ext_hint_tokens = list(ext_pre_tokens)
                                if debug:
                                    print(f"[Q-ANSWER] '{q}' extended preline y=[{band_ext_low},{band_ext_high}] x_span=({x_start},{ext_rb}) tokens={len(ext_pre_tokens)}")
                            else:
                                if debug:
                                    print(f"[Q-ANSWER] '{q}' extended preline y=[{band_ext_low},{band_ext_high}] x_span=({x_start},{ext_rb}) tokens=0")
                    else:
                        ext_hint_tokens = []
                right_bound_for_log = None
                stepped_down = False
                base_y = materialized[last_q_line_index]["y"]
                while j < len(materialized):
                    cand = materialized[j]
                    if region:
                        in_region = (region["y1"] <= cand["y"] <= region["y2"])
                        within_slack = (cand["y"] <= materialized[last_q_line_index]["y"] + y_slack)
                        if not (in_region or within_slack):
                            break
                        if not in_region and within_slack and debug:
                            print(f"[Q-ANSWER] '{q}' allowing next line at y={cand['y']} outside region via slack (limit {materialized[last_q_line_index]['y'] + y_slack})")
                    # Determine right bound: respect any subsequent question start on this line
                    # default right bound: to the end of this candidate line
                    if cand.get("words"):
                        last_tok = cand["words"][-1]
                        right_bound = last_tok["x"] + last_tok["w"]
                    else:
                        right_bound = line_end_x
                    if q_starts_by_line.get(j):
                        next_q_xs = sorted([sx for sx in q_starts_by_line[j] if sx > x_start + 2])
                        if next_q_xs:
                            right_bound = min(right_bound, next_q_xs[0] - 6)
                    # Collect tokens in a small y-band just above this next line within x-window
                    band_y_low = max(0, cand["y"] - y_preline_tol)
                    band_y_high = cand["y"] - 1
                    preband_tokens = [t["text"] for t in all_tokens
                                      if band_y_low <= t["y"] <= band_y_high and x_start - start_pad <= t["x"] <= right_bound]
                    if debug:
                        print(f"[Q-ANSWER] '{q}' preline scan y=[{band_y_low},{band_y_high}] x_span=({x_start},{right_bound}) tokens={len(preband_tokens)}")
                    toks = collect_tokens_from_line(cand, x_start, right_bound, pad=start_pad)
                    if not preband_tokens and not toks:
                        # If this candidate line is at the same y as the question (duplicate OCR line),
                        # don't stop yet—advance to the next line to actually step below the question.
                        if cand["y"] <= base_y:
                            j += 1
                            continue
                        # Once we've stepped down, we allow a blank window to terminate capture.
                        # Special-case the very first step after the question: allow a single short categorical token
                        # like YES/NO/NA on the next line to count even if band is blank.
                        if j == last_q_line_index + 1 and cand.get("words"):
                            # 1) Wide scan for YES/NO/NA anywhere on this line
                            norm_set = {_normalize_word_preserve(tok["text"]).upper() for tok in cand["words"]}
                            yesno = [tok["text"] for tok in cand["words"] if _normalize_word_preserve(tok["text"]).upper() in {"YES", "NO", "NA", "NONE"}]
                            if yesno:
                                toks = yesno
                            # 2) For specific questions, use tailored detection if still empty
                            if (not toks) and ("Date of last menstrual period" in q):
                                # Join raw tokens for regex date find
                                raw_line = ' '.join(t["text"] for t in cand["words"]) if cand.get("words") else ''
                                m = re.search(r"\b(\d{1,2}[\/-]\d{1,2}(?:[\/-]\d{2,4})?)\b", raw_line)
                                if m:
                                    toks = [m.group(1)]
                            if (not toks) and ("# of Pregnancies" in q or "Pregnancies" in q):
                                # Prefer a pure integer token in the line
                                nums = [t["text"] for t in cand["words"] if _normalize_word_preserve(t["text"]) and _normalize_word_preserve(t["text"]).isdigit()]
                                if nums:
                                    toks = [nums[0]]
                        # If still empty, stop as before
                        if not preband_tokens and not toks:
                            # encountered a blank window (band + line) for this step; stop
                            break
                    right_bound_for_log = right_bound
                    # combine band and line tokens for this step
                    if preband_tokens:
                        captured_tokens.extend(preband_tokens)
                    if toks:
                        captured_tokens.extend(toks)
                    if cand["y"] > base_y:
                        stepped_down = True
                    j += 1
                answer = finalize_answer_from_tokens(captured_tokens)
                if not answer:
                    # Adopt from extended preline if it had plausible tokens
                    adopted = None
                    if ('Date of last menstrual period' in q) and ('ext_hint_tokens' in locals()) and ext_hint_tokens:
                        joined = ' '.join(ext_hint_tokens)
                        m = re.search(r"\b(\d{1,2}[\/-]\d{1,2}(?:[\/-]\d{2,4})?)\b", joined)
                        if m:
                            adopted = m.group(1)
                    if (adopted is None) and (('# of Pregnancies' in q) or ('Pregnancies' in q)) and ('ext_hint_tokens' in locals()) and ext_hint_tokens:
                        for tok in ext_hint_tokens:
                            if tok and tok.strip().isdigit():
                                adopted = tok.strip()
                                break
                    if (adopted is None) and ('Are you currently' in q) and ('ext_hint_tokens' in locals()) and ext_hint_tokens:
                        for tok in ext_hint_tokens:
                            tnorm = _normalize_word_preserve(tok).upper()
                            if tnorm in {"YES", "NO", "NA", "NONE"}:
                                adopted = tok
                                break
                    if adopted:
                        print(f"[Q-ANSWER] '{q}' adopted-from-extended -> '{adopted}'")
                        answer = adopted
                if answer:
                    first_next_y = materialized[last_q_line_index+1]['y'] if (last_q_line_index+1) < len(materialized) else 'NA'
                    print(f"[Q-ANSWER] '{q}' next_y={first_next_y} window=({x_start},{right_bound_for_log if right_bound_for_log is not None else line_end_x}) -> '{answer}'")
                else:
                    print(f"[Q-ANSWER] '{q}' no tokens found in multi-line window=({x_start},{line_end_x})")
                    # Diagnostic: dump nearby OCR tokens around where an answer is expected to understand misses (e.g., '0')
                    diag_low = max(0, materialized[last_q_line_index]['y'] + y_preline_min - 10)
                    diag_high = materialized[last_q_line_index]['y'] + y_preline_max + 120
                    diag_left = max(0, x_start - start_pad - 20)
                    diag_right = (right_bound_for_log if right_bound_for_log is not None else line_end_x) + 40
                    neighborhood = [t for t in all_tokens if diag_low <= t['y'] <= diag_high and diag_left <= t['x'] <= diag_right]
                    neighborhood.sort(key=lambda t: (t['y'], t['x']))
                    sample = neighborhood[:60]
                    print(f"[Q-CONTEXT] '{q}' diag y=[{diag_low},{diag_high}] x=[{diag_left},{diag_right}] tokens={len(neighborhood)} (showing {len(sample)})")
                    for tok in sample:
                        print(f"[Q-CONTEXT] y={tok['y']} x={tok['x']} text='{tok['text']}'")
                    # Re-OCR a crop with numeric whitelist to catch faint single digits
                    try:
                        img_w, img_h = pil_image.size
                        crop_left = max(0, diag_left)
                        crop_top = max(0, diag_low)
                        crop_right = min(img_w - 1, diag_right)
                        crop_bottom = min(img_h - 1, diag_high)
                        if crop_right > crop_left and crop_bottom > crop_top:
                            crop = pil_image.crop((crop_left, crop_top, crop_right, crop_bottom))
                            reocr = pytesseract.image_to_string(crop, config='--psm 6 -c tessedit_char_whitelist=0123456789:/.-')
                            reocr_clean = re.sub(r'\s+', ' ', reocr).strip()
                            print(f"[Q-REOCR] '{q}' crop=({crop_left},{crop_top},{crop_right},{crop_bottom}) -> '{reocr_clean}'")
                            # If this looks like a count question and we have a short integer, adopt it as a fallback answer
                            def _is_count_question(s: str) -> bool:
                                s_up = s.upper()
                                return ('MISCARRIAGE' in s_up) or ('LIVE BIRTH' in s_up) or ('#' in s_up) or ('PREGNANCIES' in s_up)
                            if _is_count_question(q):
                                # Remove date-like patterns to avoid picking day/month/year
                                text_wo_dates = re.sub(r'\b\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4}\b', ' ', reocr_clean)
                                candidates = re.findall(r'\b\d{1,2}\b', text_wo_dates)
                                fb = None
                                if candidates:
                                    # Prefer 0 or 1 if present for counts
                                    for pref in ('0', '1'):
                                        if pref in candidates:
                                            fb = pref
                                            break
                                    if fb is None:
                                        fb = candidates[0]
                                if fb is not None:
                                    print(f"[Q-FALLBACK] '{q}' adopting numeric fallback '{fb}' from re-OCR (filtered)")
                                    # Update answer in-place for this QA
                                    for e in qas:
                                        if e["question"] == q and not e.get("answer"):
                                            e["answer"] = fb
                                            break
                                    else:
                                        qas.append({"question": q, "answer": fb})
                            # Optionally save crop for visual debug
                            import os
                            safe_q = re.sub(r'[^A-Za-z0-9]+', '_', q)[:40]
                            try:
                                base_dir = artifacts_dir or os.getcwd()
                                os.makedirs(base_dir, exist_ok=True)
                                outp = os.path.join(base_dir, f"debug_answer_window_{safe_q}_{crop_top}_{crop_bottom}.png")
                                crop.save(outp)
                                print(f"[Q-REOCR] saved debug crop {outp}")
                            except Exception as e2:
                                print(f"[Q-REOCR] save crop error: {e2}")
                        else:
                            print(f"[Q-REOCR] '{q}' crop skipped due to invalid bounds")
                    except Exception as e:
                        print(f"[Q-REOCR] error: {e}")
                # add or update
                found_existing = False
                for e in qas:
                    if e["question"] == q:
                        found_existing = True
                        if answer and not e.get("answer"):
                            e["answer"] = answer
                        break
                if not found_existing:
                    qas.append({"question": q, "answer": answer})

        if qas:
            responses.append({"section": sec_name, "questions": qas})

    return responses
