"""
scripts/evaluate.py  —  验证 Conformal Prediction 保证是否成立

在独立测试集上运行完整 pipeline，统计：
  1. 实际 miscoverage rate（应 ≤ α）
  2. Abstention rate（不同 α 下拒答比例）
  3. Answered queries 的准确率

用法：
    python scripts/evaluate.py \
        --config    configs/config.yaml \
        --data_path data/test.json \
        --alphas    0.05 0.1 0.2 \
        --output    data/eval_results.json

测试集格式（和 calibration.json 相同）：
    [{"question": "...", "answer": "..."}, ...]
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm import LLM
from src.rag import RAG
from src.uncertainty import Uncertainty
from src.conformal import ConformalPredictor
from src.pipeline import TrustworthyRAGPipeline
from src.utils import load_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ── 答案正确性判断 ──────────────────────────────────────────────────────────
def is_correct(pred: str, gold: str, threshold: float = 0.6) -> bool:
    """
    用 token overlap (F1) 判断答案是否正确。
    不依赖额外模型，工程友好。
    threshold=0.6 和 TriviaQA 论文一致。
    """
    if pred is None:
        return False

    def tokenize(s: str) -> set:
        import re
        return set(re.findall(r'\b\w+\b', s.lower()))

    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)

    if not pred_tokens or not gold_tokens:
        return False

    common = pred_tokens & gold_tokens
    if not common:
        return False

    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1 >= threshold


# ── 主函数 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate conformal RAG system")
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--data_path", required=True, help="测试集 JSON 路径")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--output",    default="data/eval_results.json")
    args = parser.parse_args()

    config   = load_config(args.config)
    pipeline = TrustworthyRAGPipeline(config)

    with open(args.data_path, encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info("测试集: %d 条", len(dataset))

    # ── 收集每条样本的结果 ──────────────────────────────────────────────────
    records = []
    for item in tqdm(dataset, desc="Evaluating"):
        question = item["question"]
        gold     = item["answer"]
        try:
            result = workflow_obj.run(
                query=question,
                alpha=args.alphas[-1],
                session_id="evaluate",
            )
            answer = result.get("answer")
            if not answer or not answer.strip():
                answer = None
                result["score"] = 1.0
            records.append({
                "question":    question,
                "gold":        gold,
                "answer":      answer,
                "score":       result["score"],
                "confidence":  result.get("confidence", 0.0),
                "faithfulness": result.get("faithfulness"),
            })
        except Exception as e:
            logger.warning("跳过 %r: %s", question[:50], e)

    if not records:
        raise RuntimeError("没有收集到任何结果")

    scores = np.array([r["score"] for r in records])
    logger.info("Score 统计: min=%.4f mean=%.4f max=%.4f",
                scores.min(), scores.mean(), scores.max())

    # ── 对每个 alpha 计算指标 ───────────────────────────────────────────────
    # 加载校准阈值
    conformal = pipeline.conformal
    results_by_alpha = {}

    for alpha in args.alphas:
        threshold = conformal._get_threshold(alpha)

        answered    = []   # 被接受的样本
        abstained_n = 0

        for r in records:
            accepted = r["score"] <= threshold
            if accepted:
                answered.append(r)
            else:
                abstained_n += 1

        n_total    = len(records)
        n_answered = len(answered)
        n_abstain  = abstained_n

        # 在被回答的里面判断正确率
        n_correct = sum(1 for r in answered
                        if is_correct(r["answer"], r["gold"]))

        # miscoverage rate = 在被回答的里面答错的比例（CP 标准定义）
        # 分母是 n_answered（不含拒答），这才是 CP 保证的对象
        n_wrong   = n_answered - n_correct
        miscoverage_rate = n_wrong / n_answered if n_answered > 0 else 0.0

        accuracy_on_answered = n_correct / n_answered if n_answered > 0 else 0.0
        abstention_rate      = n_abstain / n_total

        results_by_alpha[str(alpha)] = {
            "alpha":                alpha,
            "threshold_qhat":       round(threshold, 4),
            "n_total":              n_total,
            "n_answered":           n_answered,
            "n_abstained":          n_abstain,
            "n_correct":            n_correct,
            "n_wrong":              n_wrong,
            "miscoverage_rate":     round(miscoverage_rate, 4),
            "cp_guarantee_holds":   miscoverage_rate <= alpha,   # 核心验证
            "abstention_rate":      round(abstention_rate, 4),
            "accuracy_on_answered": round(accuracy_on_answered, 4),
        }

        logger.info(
            "α=%.2f | threshold=%.4f | answered=%d/%d | "
            "correct=%d | miscoverage=%.4f (≤%.2f: %s) | abstain=%.1f%%",
            alpha, threshold, n_answered, n_total,
            n_correct, miscoverage_rate, alpha,
            "✅" if miscoverage_rate <= alpha else "❌",
            abstention_rate * 100,
        )

    # ── Baseline 统计（不带 CP，直接回答所有问题）────────────────────────
    baseline_correct = 0
    baseline_total   = len(records)
    for r in records:
        if r["answer"] and is_correct(r["answer"], r["gold"]):
            baseline_correct += 1
    baseline_accuracy = baseline_correct / baseline_total if baseline_total else 0
    logger.info(
        "Baseline（无CP，直接回答所有问题）: accuracy=%.1f%% (%d/%d)",
        baseline_accuracy * 100, baseline_correct, baseline_total
    )

    # ── Faithfulness 统计（全部样本，不分 alpha）────────────────────────────
    faith_records = [r for r in records if r.get("faithfulness") and r["answer"]]
    if faith_records:
        labels = [r["faithfulness"]["label"] for r in faith_records]
        n_faith  = len(faith_records)
        n_faithful     = labels.count("faithful")
        n_uncertain    = labels.count("uncertain")
        n_hallucinated = labels.count("hallucinated")
        avg_faith_score = sum(
            r["faithfulness"]["faithfulness_score"] for r in faith_records
        ) / n_faith

        logger.info(
            "Faithfulness 统计 (n=%d answered): faithful=%d(%.1f%%) "
            "uncertain=%d(%.1f%%) hallucinated=%d(%.1f%%) avg_score=%.4f",
            n_faith,
            n_faithful,     n_faithful/n_faith*100,
            n_uncertain,    n_uncertain/n_faith*100,
            n_hallucinated, n_hallucinated/n_faith*100,
            avg_faith_score,
        )
        faithfulness_stats = {
            "n_answered":       n_faith,
            "n_faithful":       n_faithful,
            "n_uncertain":      n_uncertain,
            "n_hallucinated":   n_hallucinated,
            "faithful_rate":    round(n_faithful/n_faith, 4),
            "hallucinated_rate":round(n_hallucinated/n_faith, 4),
            "avg_score":        round(avg_faith_score, 4),
        }
    else:
        faithfulness_stats = {}
        logger.warning("没有 faithfulness 数据，跳过统计")

    # ── 保存结果 ────────────────────────────────────────────────────────────
    output = {
        "n_samples": len(records),
        "score_stats": {
            "min":  round(float(scores.min()),  4),
            "mean": round(float(scores.mean()), 4),
            "max":  round(float(scores.max()),  4),
        },
        "results_by_alpha":  results_by_alpha,
        "faithfulness_stats": faithfulness_stats,
        "baseline": {
            "total":    baseline_total,
            "correct":  baseline_correct,
            "accuracy": round(baseline_accuracy, 4),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("结果已保存 → %s", out_path)

    # ── 打印汇总表格 ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'α':>6} {'q̂':>8} {'answered':>10} {'abstain%':>10} "
          f"{'accuracy':>10} {'miscov':>10} {'CP holds':>10}")
    print("-"*70)
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
    print("="*70)


if __name__ == "__main__":
    main()
