import json, os, sys
from typing import Dict, Any, Tuple, List, Set

def load(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def page_index(d: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {p.get('page_number', i+1): p for i,p in enumerate(d.get('pages', []))}

def section_label_status_map(p: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for s in (p.get('sections') or []):
        name = s.get('section')
        if not name: continue
        cbs = s.get('checkboxes')
        if cbs is None: continue
        out[name] = {cb.get('label'): cb.get('status') for cb in cbs if cb.get('label') is not None}
    return out

def responses_questions_map(p: Dict[str, Any]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for s in (p.get('responses') or []):
        name = s.get('section')
        if not name: continue
        qs = set()
        for q in s.get('questions', []):
            qt = q.get('question')
            if qt:
                qs.add(qt)
        out[name] = qs
    return out

def responses_question_values_map(p: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return: section -> question -> value (if present), else None.
    This focuses on explicit 'value' fields; it does not attempt to derive from segments.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for s in (p.get('responses') or []):
        name = s.get('section')
        if not name: continue
        m: Dict[str, Any] = {}
        for q in s.get('questions', []):
            qt = q.get('question')
            if not qt:
                continue
            m[qt] = q.get('value') if 'value' in q else None
        if m:
            out[name] = m
    return out

def diff_json(out_path: str, golden_path: str) -> str:
    a = load(out_path)
    b = load(golden_path)
    pa, pb = page_index(a), page_index(b)
    pages = sorted(set(pa) | set(pb))
    lines: List[str] = []
    lines.append(f"Pages: output={len(pa)} golden={len(pb)}")
    for pn in pages:
        A, B = pa.get(pn), pb.get(pn)
        if A is None or B is None:
            lines.append(f"Page {pn}: present? output={A is not None} golden={B is not None}")
            continue
        # Sections presence
        sa, sb = section_label_status_map(A), section_label_status_map(B)
        names_a, names_b = set(sa), set(sb)
        only_a = sorted(names_a - names_b)
        only_b = sorted(names_b - names_a)
        if only_a or only_b:
            lines.append(f"Page {pn} sections: only-in-output={only_a} only-in-golden={only_b}")
        # Section label sets/statuses
        for name in sorted(names_a & names_b):
            la, lb = sa[name], sb[name]
            la_keys, lb_keys = set(la), set(lb)
            miss = sorted(lb_keys - la_keys)
            extra = sorted(la_keys - lb_keys)
            if miss or extra:
                lines.append(f"Page {pn} section '{name}': labels only-in-output={extra} only-in-golden={miss}")
            # Status diffs for overlapping labels
            both_labels = la_keys & lb_keys
            status_diffs = [(lab, la[lab], lb[lab]) for lab in sorted(both_labels) if la[lab] != lb[lab]]
            if status_diffs:
                pretty = ", ".join([f"{lab}: out={oa} gold={ob}" for lab, oa, ob in status_diffs])
                lines.append(f"Page {pn} section '{name}': status diffs -> {pretty}")
        # Responses sections and questions
        ra, rb = responses_questions_map(A), responses_questions_map(B)
        rn_a, rn_b = set(ra), set(rb)
        r_only_a = sorted(rn_a - rn_b)
        r_only_b = sorted(rn_b - rn_a)
        if r_only_a or r_only_b:
            lines.append(f"Page {pn} responses: only-in-output={r_only_a} only-in-golden={r_only_b}")
        for name in sorted(rn_a & rn_b):
            qa, qb = ra[name], rb[name]
            qa_only = sorted(qa - qb)
            qb_only = sorted(qb - qa)
            if qa_only or qb_only:
                lines.append(f"Page {pn} responses '{name}': questions only-in-output={qa_only} only-in-golden={qb_only}")
        # Field value compare for responses ('value' fields if present)
        va, vb = responses_question_values_map(A), responses_question_values_map(B)
        common_resp_secs = sorted(set(va) & set(vb))
        for name in common_resp_secs:
            qva, qvb = va[name], vb[name]
            common_qs = sorted(set(qva) & set(qvb))
            diffs = []
            for qn in common_qs:
                if qva[qn] != qvb[qn]:
                    diffs.append((qn, qva[qn], qvb[qn]))
            if diffs:
                pretty = ", ".join([f"{qn}: out={oa!r} gold={ob!r}" for qn, oa, ob in diffs])
                lines.append(f"Page {pn} responses '{name}': value diffs -> {pretty}")
        # Note about missing values presence
        # If values exist on only one side, mention which sections have presence asymmetry
        only_val_a = sorted([s for s in va if s not in vb])
        only_val_b = sorted([s for s in vb if s not in va])
        if only_val_a or only_val_b:
            lines.append(f"Page {pn} responses: sections with 'value' only-in-output={only_val_a} only-in-golden={only_val_b}")
    if len(lines) == 1:
        lines.append("No per-page differences found (sections/labels/statuses/responses)")
    return "\n".join(lines)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: compare_json.py <output.json> <golden.json>")
        sys.exit(2)
    print(diff_json(sys.argv[1], sys.argv[2]))
