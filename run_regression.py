import os
import json
import shutil
import sys
import argparse
from typing import List, Tuple

from extractor import run_extractor_from_config


def ensure_dirs(base: str):
    inputs = os.path.join(base, "inputs")
    goldens = os.path.join(base, "goldens")
    os.makedirs(inputs, exist_ok=True)
    os.makedirs(goldens, exist_ok=True)
    return inputs, goldens


def discover_pdfs(src_dir: str) -> List[str]:
    return [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.lower().endswith(".pdf")]


def copy_if_missing(src: str, dst_dir: str):
    dst = os.path.join(dst_dir, os.path.basename(src))
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
    return dst


def run_one(pdf_path: str, out_json_path: str):
    data = run_extractor_from_config(
        pdf_path=pdf_path,
        output_path=out_json_path,
        # Respect config.json for sections/templates/poppler/tesseract; we only override output
    )
    return data


def _normalize_for_compare(obj: dict, ignore_responses: bool) -> dict:
    if not isinstance(obj, dict):
        return obj
    if ignore_responses and "pages" in obj and isinstance(obj["pages"], list):
        pages = []
        for p in obj["pages"]:
            if isinstance(p, dict):
                q = dict(p)
                # Drop responses when comparing if requested
                q.pop("responses", None)
                pages.append(q)
            else:
                pages.append(p)
        obj = dict(obj)
        obj["pages"] = pages
    return obj


def compare_json(golden_path: str, new_path: str, ignore_responses: bool = False) -> Tuple[bool, str]:
    try:
        with open(golden_path, "r", encoding="utf-8") as f:
            g = json.load(f)
        with open(new_path, "r", encoding="utf-8") as f:
            n = json.load(f)
    except Exception as e:
        return False, f"Failed to load JSON: {e}"

    if ignore_responses:
        g = _normalize_for_compare(g, ignore_responses=True)
        n = _normalize_for_compare(n, ignore_responses=True)

    if g == n:
        return True, "OK"
    return False, "Mismatch"


def main():
    parser = argparse.ArgumentParser(description="Run PDF extraction regression tests")
    parser.add_argument("--no-seed", action="store_true", help="Do not seed missing goldens; treat missing goldens as warnings and skip compare")
    parser.add_argument("--update", action="store_true", help="Overwrite goldens with current outputs (use with care)")
    parser.add_argument("--prune", action="store_true", help="Delete golden files that have no corresponding input PDF")
    parser.add_argument("--ignore-responses", action="store_true", help="Ignore 'responses' when comparing JSON (useful when responses format changes)")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    artifacts = os.path.join(repo_root, "artifacts")
    tests_dir = os.path.join(artifacts, "test-cases")
    inputs_dir, goldens_dir = ensure_dirs(tests_dir)

    # Seed: copy test PDFs from test-inputs into test-cases/inputs
    seed_dir = os.path.join(repo_root, "test-inputs")
    if os.path.exists(seed_dir):
        for pdf in discover_pdfs(seed_dir):
            copy_if_missing(pdf, inputs_dir)

    # Optional pruning: remove goldens with no matching input
    if args.prune:
        inputs_basenames = {os.path.splitext(f)[0] for f in os.listdir(inputs_dir) if f.lower().endswith('.pdf')}
        removed = []
        for g in os.listdir(goldens_dir):
            if not g.lower().endswith('.json'):
                continue
            base = os.path.splitext(g)[0]
            # goldens are named output_<basename>
            if base.startswith("output_"):
                name = base[len("output_"):]
            else:
                name = base
            if name not in inputs_basenames:
                os.remove(os.path.join(goldens_dir, g))
                removed.append(g)
        if removed:
            print(f"[REG][PRUNE] Removed {len(removed)} golden(s): {', '.join(removed)}")

    # Run regression for each input PDF: write fresh outputs to artifacts/test-cases/outputs
    outputs_dir = os.path.join(tests_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    pdfs = discover_pdfs(inputs_dir)
    if not pdfs:
        print("[REG] No PDFs found under artifacts/test-cases/inputs. Nothing to run.")
        sys.exit(0)

    failures = []
    for pdf in pdfs:
        name = os.path.splitext(os.path.basename(pdf))[0]
        out_json = os.path.join(outputs_dir, f"output_{name}.json")
        print(f"[REG] Running: {os.path.basename(pdf)} -> {os.path.relpath(out_json, repo_root)}")
        run_one(pdf, out_json)
        # Prefer deterministic golden name: output_{input_basename}.json
        golden_expected = os.path.join(goldens_dir, f"output_{name}.json")
        golden_path = golden_expected if os.path.exists(golden_expected) else None
        # Fallback: legacy golden filenames that partially match the name
        if golden_path is None:
            golden_candidates = [g for g in os.listdir(goldens_dir) if g.lower().endswith('.json') and name in g]
            golden_path = os.path.join(goldens_dir, golden_candidates[0]) if golden_candidates else None

        # Seed: if still missing and allowed, seed golden with current output
        if golden_path is None and not args.no_seed:
            os.makedirs(goldens_dir, exist_ok=True)
            shutil.copy2(out_json, golden_expected)
            print(f"[REG][SEED] Created golden for {name}: {os.path.relpath(golden_expected, repo_root)}")
            golden_path = golden_expected
        elif golden_path is None and args.no_seed:
            print(f"[REG][WARN] No golden found for {name} and seeding is disabled; skipping compare")
            continue

        # Optional update: overwrite golden with current output
        if args.update:
            shutil.copy2(out_json, golden_expected)
            golden_path = golden_expected
            print(f"[REG][UPDATE] Overwrote golden for {name}: {os.path.relpath(golden_expected, repo_root)}")
        ok, msg = compare_json(golden_path, out_json, ignore_responses=args.ignore_responses)
        if ok:
            note = " (ignoring responses)" if args.ignore_responses else ""
            print(f"[REG][PASS] {name}{note}")
        else:
            failures.append((name, msg, golden_path, out_json))
            print(f"[REG][FAIL] {name}: {msg}")

    if failures:
        print("\n[REG] Failures:")
        for name, msg, g, n in failures:
            print(f"  - {name}: {msg}\n    golden: {g}\n    output: {n}")
        sys.exit(1)
    else:
        print("\n[REG] All tests passed.")


if __name__ == "__main__":
    main()
