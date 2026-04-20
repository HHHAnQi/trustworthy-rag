"""
scripts/prepare_triviaqa.py
----------------------------
Convert your existing ConU cleaned_generations.pkl into the two files
this project needs:

  data/docs.txt           — one Wikipedia passage per line (RAG corpus)
  data/calibration.json   — [{"question":..., "answer":...}, ...]

It also optionally downloads the TriviaQA 'rc' (with-context) split from
HuggingFace to extract Wikipedia passages.  If you already have the passages
locally, point --passages_pkl at them instead.

Usage (recommended — downloads passages automatically)
------------------------------------------------------
    python scripts/prepare_triviaqa.py \
        --generations_pkl  path/to/cleaned_generations.pkl \
        --use_hf_passages \
        --max_docs         50000 \
        --max_cal          500

Usage (no internet — manual passages)
--------------------------------------
    python scripts/prepare_triviaqa.py \
        --generations_pkl  path/to/cleaned_generations.pkl \
        --passages_txt     path/to/existing_passages.txt \
        --max_cal          500
"""

from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_generations(pkl_path: str) -> list:
    with open(pkl_path, "rb") as f:
        gens = pickle.load(f)
    print(f"Loaded {len(gens)} generations from {pkl_path}")
    return gens


def build_calibration_json(generations: list, max_cal: int, out_path: str):
    """Extract question/answer pairs into calibration.json."""
    data = []
    for g in generations[:max_cal]:
        data.append({"question": g["question"], "answer": g["answer"]})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} calibration samples → {out_path}")


def build_docs_from_hf(max_docs: int, out_path: str):
    """Download TriviaQA rc passages from HuggingFace and write docs.txt."""
    try:
        import datasets
    except ImportError:
        print("ERROR: `pip install datasets` first.")
        sys.exit(1)

    print("Downloading TriviaQA 'rc' validation split from HuggingFace …")
    ds = datasets.load_dataset("trivia_qa", "rc", split="validation")

    passages = []
    for item in ds:
        for ctx in item.get("search_results", {}).get("search_context", []):
            chunk = ctx.replace("\n", " ").strip()
            if len(chunk) > 50:            # skip very short fragments
                passages.append(chunk[:500])  # cap length per passage
            if len(passages) >= max_docs:
                break
        if len(passages) >= max_docs:
            break

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(passages))
    print(f"Saved {len(passages)} passages → {out_path}")


def build_docs_from_file(passages_txt: str, out_path: str):
    """Copy an existing passages file to docs.txt."""
    import shutil
    shutil.copy(passages_txt, out_path)
    n = sum(1 for ln in open(out_path, encoding="utf-8") if ln.strip())
    print(f"Copied {n} passages → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generations_pkl", required=True,
                   help="Path to ConU cleaned_generations.pkl")
    p.add_argument("--use_hf_passages", action="store_true",
                   help="Download TriviaQA rc passages from HuggingFace")
    p.add_argument("--passages_txt", default=None,
                   help="Use an existing passages file instead of HF")
    p.add_argument("--max_docs", type=int, default=50000,
                   help="Max passages to put in docs.txt")
    p.add_argument("--max_cal",  type=int, default=500,
                   help="Max questions to put in calibration.json")
    p.add_argument("--docs_out", default="data/docs.txt")
    p.add_argument("--cal_out",  default="data/calibration.json")
    args = p.parse_args()

    Path("data").mkdir(exist_ok=True)

    # 1. calibration.json
    generations = load_generations(args.generations_pkl)
    build_calibration_json(generations, args.max_cal, args.cal_out)

    # 2. docs.txt
    if args.use_hf_passages:
        build_docs_from_hf(args.max_docs, args.docs_out)
    elif args.passages_txt:
        build_docs_from_file(args.passages_txt, args.docs_out)
    else:
        print(
            "WARNING: No passages source specified.\n"
            "Supply --use_hf_passages or --passages_txt.\n"
            "docs.txt not created — build_index.py will fail."
        )


if __name__ == "__main__":
    main()
