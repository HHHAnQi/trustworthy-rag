"""
scripts/run_all_and_split.py
-----------------------------
对全部 700 条问题跑完整 RAG pipeline，收集 nonconformity score，
然后随机 7:5/2 划分，同时生成 qhat.json 和评估结果。

这样保证 calibration 和 test 的 score 来自同一分布，
conformal 的 exchangeability 条件成立。
"""

from __future__ import annotations
import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conformal import compute_qhat
from src.pipeline import TrustworthyRAGPipeline
from src.utils import load_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    "Answer the question in ONE sentence using only the context below. "
    "Do not explain your reasoning. Do not mention the context. "
    "If the answer is not in the context, say: I don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer (one sentence):"
)


def is_correct(pred: str, gold: str, threshold: float = 0.5) -> bool:
    if not pred or not pred.strip():
        return False
    import re
    def tokenize(s):
        return set(re.findall(r'\b\w+\b', s.lower()))
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return False
    if gold_tokens.issubset(pred_tokens):
        return True
    common = pred_tokens & gold_tokens
    if not common:
        return False
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gold_tokens)
    return (2 * p * r / (p + r)) >= threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--data_path", required=True,
                        help="全部 700 条问题的 JSON")
    parser.add_argument("--cal_size",  type=int, default=500)
    parser.add_argument("--alphas",    nargs="+", type=float,
                        default=[0.05, 0.1, 0.2])
    parser.add_argument("--qhat_out",  default="data/qhat.json")
    parser.add_argument("--eval_out",  default="data/eval_results.json")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    config   = load_config(args.config)
    pipeline = TrustworthyRAGPipeline(config)

    with open(args.data_path, encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info("总数据量: %d 条", len(dataset))

    # ── Step 1: 对全部数据跑 pipeline，收集 score ──────────────────────
    records = []
    logger.info("Step 1/3: 对全部 %d 条问题收集 score …", len(dataset))

    for item in tqdm(dataset, desc="Collecting scores"):
        question = item["question"]
        gold     = item["answer"]
        try:
            # 用最宽松的 alpha，只是为了拿到 score 和 answer
            result = pipeline.run(question, alpha=args.alphas[-1],
                                  return_details=False)
            records.append({
                "question":   question,
                "gold":       gold,
                "answer":     result["answer"],
                "score":      result["score"],
                "confidence": result["confidence"],
                "accepted":   result["accepted"],
            })
        except Exception as e:
            logger.warning("跳过 %r: %s", question[:50], e)

    logger.info("成功收集 %d 条", len(records))

    # ── Step 2: 随机划分 cal / test ────────────────────────────────────
    logger.info("Step 2/3: 随机划分 %d cal / %d test …",
                args.cal_size, len(records) - args.cal_size)

    indices = list(range(len(records)))
    random.shuffle(indices)
    cal_indices  = indices[:args.cal_size]
    test_indices = indices[args.cal_size:]

    cal_records  = [records[i] for i in cal_indices]
    test_records = [records[i] for i in test_indices]

    cal_scores = np.array([r["score"] for r in cal_records])
    logger.info(
        "Calibration score 统计: min=%.4f mean=%.4f max=%.4f",
        cal_scores.min(), cal_scores.mean(), cal_scores.max()
    )

    test_scores = np.array([r["score"] for r in test_records])
    logger.info(
        "Test score 统计:        min=%.4f mean=%.4f max=%.4f",
        test_scores.min(), test_scores.mean(), test_scores.max()
    )

    # ── Step 3: 计算 qhat 并评估 ───────────────────────────────────────
    logger.info("Step 3/3: 计算 q̂ 并评估 …")

    qhat_table = {}
    results_by_alpha = {}

    # ── Baseline 统计（不带 CP，直接回答所有问题）──────────────────────────
    baseline_correct = sum(
        1 for r in test_records
        if r["answer"] and is_correct(r["answer"], r["gold"])
    )
    baseline_total   = len(test_records)
    baseline_accuracy = baseline_correct / baseline_total if baseline_total else 0
    logger.info(
        "Baseline（无CP，直接回答所有问题）: accuracy=%.1f%% (%d/%d)",
        baseline_accuracy * 100, baseline_correct, baseline_total
    )

    for alpha in args.alphas:
        qhat = compute_qhat(cal_scores.tolist(), alpha)
        qhat_table[str(alpha)] = round(qhat, 6)

        # 用新 qhat 对 test 重新做 accept/abstain 决策
        n_total    = len(test_records)
        answered   = []
        n_abstain  = 0

        for r in test_records:
            # 模型说不知道（空答案）强制 abstain
            if not r["answer"] or not r["answer"].strip():
                n_abstain += 1
                continue
            if r["score"] <= qhat:
                answered.append(r)
            else:
                n_abstain += 1

        n_answered = len(answered)
        n_correct  = sum(1 for r in answered
                         if is_correct(r["answer"], r["gold"]))
        n_wrong    = n_answered - n_correct

        miscoverage = n_wrong / n_answered if n_answered > 0 else 0.0
        abstention  = (n_total - n_answered) / n_total
        accuracy    = n_correct / n_answered if n_answered > 0 else 0.0
        holds       = miscoverage <= alpha

        results_by_alpha[str(alpha)] = {
            "alpha":                alpha,
            "threshold_qhat":       round(qhat, 4),
            "n_total":              n_total,
            "n_answered":           n_answered,
            "n_abstained":          n_total - n_answered,
            "n_correct":            n_correct,
            "n_wrong":              n_wrong,
            "miscoverage_rate":     round(miscoverage, 4),
            "cp_guarantee_holds":   holds,
            "abstention_rate":      round(abstention, 4),
            "accuracy_on_answered": round(accuracy, 4),
        }

        logger.info(
            "α=%.2f | q̂=%.4f | answered=%d/%d | correct=%d | "
            "miscov=%.4f (≤%.2f: %s) | abstain=%.1f%%",
            alpha, qhat, n_answered, n_total, n_correct,
            miscoverage, alpha,
            "✅" if holds else "❌",
            abstention * 100,
        )

    # ── 保存 qhat.json ─────────────────────────────────────────────────
    Path(args.qhat_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.qhat_out, "w") as f:
        json.dump(qhat_table, f, indent=2)
    logger.info("q̂ 表已保存 → %s", args.qhat_out)

    # ── 保存 eval_results.json ─────────────────────────────────────────
    output = {
        "n_cal":  len(cal_records),
        "n_test": len(test_records),
        "cal_score_stats": {
            "min":  round(float(cal_scores.min()),  4),
            "mean": round(float(cal_scores.mean()), 4),
            "max":  round(float(cal_scores.max()),  4),
        },
        "test_score_stats": {
            "min":  round(float(test_scores.min()),  4),
            "mean": round(float(test_scores.mean()), 4),
            "max":  round(float(test_scores.max()),  4),
        },
        "results_by_alpha": results_by_alpha,
        "baseline": {
            "total":    baseline_total,
            "correct":  baseline_correct,
            "accuracy": round(baseline_accuracy, 4),
        },
    }
    with open(args.eval_out, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("评估结果已保存 → %s", args.eval_out)

    # ── 打印汇总表 ─────────────────────────────────────────────────────
    print("\n" + "="*72)
    print(f"{'α':>6} {'q̂':>8} {'answered':>10} {'abstain%':>10} "
          f"{'accuracy':>10} {'miscov':>10} {'CP holds':>10}")
    print("-"*72)
    for alpha in args.alphas:
        r = results_by_alpha[str(alpha)]
        print(
            f"{r['alpha']:>6.2f} "
            f"{r['threshold_qhat']:>8.4f} "
            f"{r['n_answered']:>4}/{r['n_total']:<4} "
            f"{r['abstention_rate']*100:>9.1f}% "
            f"{r['accuracy_on_answered']*100:>9.1f}% "
            f"{r['miscoverage_rate']:>10.4f} "
            f"{'✅' if r['cp_guarantee_holds'] else '❌':>10}"
        )
    print("="*72)


if __name__ == "__main__":
    main()
