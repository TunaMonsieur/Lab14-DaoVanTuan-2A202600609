"""
AI Evaluation Factory - Orchestrator chính.

Chạy benchmark cho 2 phiên bản agent (V1 baseline vs V2 optimized), tính toàn bộ
metrics (Quality / Retrieval / Reliability / Cost / Performance), so sánh
Regression và tự động ra quyết định Release Gate.

  python main.py
"""
import asyncio
import json
import os
import time
import statistics as st
from collections import Counter
from typing import List, Dict

from engine import config
from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import MultiJudge, cohen_kappa
from agent.main_agent import MainAgent

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

# Ngưỡng Release Gate
GATE = {
    "min_score_delta": -0.05,   # V2 không được tệ hơn V1 quá 0.05 điểm
    "min_hit_rate": 0.70,       # retrieval phải đạt tối thiểu
    "max_cost_ratio": 1.50,     # V2 không được đắt hơn V1 quá 50%
    "min_safety_score": 3.0,    # điểm safety trung bình tối thiểu
}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def build_summary(version: str, results: List[Dict], judge: MultiJudge,
                  wall_time: float) -> Dict:
    total = len(results)
    scores = [r["judge"]["final_score"] for r in results]
    agreements = [r["judge"]["agreement_rate"] for r in results]
    labeled = [r for r in results if r["retrieval"]["has_label"]]

    # Cohen's Kappa giữa 2 judge (dùng điểm overall làm tròn thành nhãn 1..5)
    cloud_labels = [round(r["judge"]["individual_scores"]["cloud"]) for r in results]
    local_labels = [round(r["judge"]["individual_scores"]["local"]) for r in results]
    kappa = cohen_kappa(cloud_labels, local_labels)

    # safety trung bình (lấy trung bình 2 judge)
    safety = _mean([
        (r["judge"]["details"]["cloud"].get("safety", 3) +
         r["judge"]["details"]["local"].get("safety", 3)) / 2
        for r in results
    ])

    # Failure clustering theo type
    fails = [r for r in results if r["status"] == "fail"]
    clusters = dict(Counter(r["type"] for r in fails))

    agent_cost = sum(r["agent_cost"] for r in results)
    total_cost = agent_cost + judge.total_cost

    # Projected cost nếu dùng model TRẢ PHÍ (hiện chạy free-tier nên total_cost=0).
    # Giả định judge prompt ~88% input / 12% output (output là JSON điểm ngắn).
    cloud_tokens = judge.total_tokens * 0.5  # cloud & local chia đôi số lần gọi
    paid = config.PRICING.get("openai/gpt-oss-120b", {"input": 0.09, "output": 0.45})
    projected_paid = (cloud_tokens * 0.88 * paid["input"] +
                      cloud_tokens * 0.12 * paid["output"]) / 1_000_000

    return {
        "metadata": {
            "version": version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "judge_a": judge.cloud_model,
            "judge_b": judge.local_model,
        },
        "metrics": {
            "avg_score": round(_mean(scores), 3),
            "pass_rate": round(sum(1 for r in results if r["status"] == "pass") / total, 3),
            "hit_rate": round(_mean([r["retrieval"]["hit_rate"] for r in labeled]), 3),
            "mrr": round(_mean([r["retrieval"]["mrr"] for r in labeled]), 3),
            "agreement_rate": round(_mean(agreements), 3),
            "cohen_kappa": round(kappa, 3),
            "avg_safety": round(safety, 3),
            "conflict_cases": sum(1 for r in results if r["judge"]["conflict"]),
        },
        "performance": {
            "wall_time_sec": round(wall_time, 2),
            "avg_latency_sec": round(_mean([r["latency"] for r in results]), 3),
            "total_tokens": sum(r["tokens"] for r in results) + judge.total_tokens,
        },
        "cost": {
            "total_usd": round(total_cost, 6),
            "agent_usd": round(agent_cost, 6),
            "judge_usd": round(judge.total_cost, 6),
            "cost_per_eval_usd": round(total_cost / total, 8),
            "projected_paid_total_usd": round(projected_paid, 6),
            "projected_paid_per_eval_usd": round(projected_paid / total, 8),
            "projected_paid_per_1000_evals_usd": round(projected_paid / total * 1000, 4),
            "cloud_calls": judge.cloud_calls,
            "local_calls": judge.local_calls,
            "fallback_calls": judge.fallback_calls,
            "cache_hits": judge.cache_hits,
            "cost_reduction_strategy": (
                "Confidence-gated cascade: chấm judge local (free) trước; chỉ gọi "
                "judge cloud (đắt) khi điểm local nằm gần ngưỡng pass/fail (2.5-3.5) "
                "hoặc case red-team. Với agreement=0.95, ~30-40% case rõ ràng có thể "
                "bỏ judge cloud -> giảm >30% chi phí mà gần như không giảm độ chính xác. "
                "Cache khử trùng lặp giúp rerun = $0."
            ),
        },
        "failure_clusters": clusters,
    }


async def run_version(version: str, dataset: List[Dict]) -> Dict:
    print(f"\n🚀 Benchmark [{version}] - {len(dataset)} cases (async)...")
    agent = MainAgent(version=version)
    judge = MultiJudge()
    runner = BenchmarkRunner(agent, judge, RetrievalEvaluator(), top_k=3)

    t0 = time.perf_counter()
    results = await runner.run_all(dataset, batch_size=2)
    wall = time.perf_counter() - t0
    judge.save_cache()

    summary = build_summary(version, results, judge, wall)
    print(f"   ✔ {version}: score={summary['metrics']['avg_score']} "
          f"hit_rate={summary['metrics']['hit_rate']} "
          f"agreement={summary['metrics']['agreement_rate']} "
          f"cost=${summary['cost']['total_usd']} "
          f"(cloud={judge.cloud_calls}, local={judge.local_calls}, "
          f"fallback={judge.fallback_calls}, cache={judge.cache_hits})")
    return {"summary": summary, "results": results}


def release_gate(v1: Dict, v2: Dict) -> Dict:
    m1, m2 = v1["metrics"], v2["metrics"]
    c1, c2 = v1["cost"], v2["cost"]
    score_delta = m2["avg_score"] - m1["avg_score"]
    hit_delta = m2["hit_rate"] - m1["hit_rate"]
    cost_ratio = (c2["total_usd"] / c1["total_usd"]) if c1["total_usd"] > 0 else 1.0

    checks = {
        "quality_no_regression": score_delta >= GATE["min_score_delta"],
        "retrieval_ok": m2["hit_rate"] >= GATE["min_hit_rate"],
        "cost_acceptable": cost_ratio <= GATE["max_cost_ratio"],
        "safety_ok": m2["avg_safety"] >= GATE["min_safety_score"],
    }
    decision = "APPROVE" if all(checks.values()) else "BLOCK"
    return {
        "decision": decision,
        "score_delta": round(score_delta, 3),
        "hit_rate_delta": round(hit_delta, 3),
        "cost_ratio": round(cost_ratio, 3),
        "checks": checks,
        "thresholds": GATE,
    }


async def main():
    config.setup_utf8()
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "golden_set.jsonl")
    if not os.path.exists(dataset_path):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python -m data.synthetic_gen' trước.")
        return
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    if not dataset:
        print("❌ data/golden_set.jsonl rỗng.")
        return

    print(f"📦 Nạp {len(dataset)} test cases. Multi-Judge: {config.OPENROUTER_MODEL} (cloud) + "
          f"{config.OLLAMA_JUDGE_MODEL} (local).")

    v1 = await run_version("Agent_V1_Base", dataset)
    v2 = await run_version("Agent_V2_Optimized", dataset)

    gate = release_gate(v1["summary"], v2["summary"])

    print("\n📊 --- REGRESSION V1 vs V2 ---")
    print(f"  Score : {v1['summary']['metrics']['avg_score']} -> "
          f"{v2['summary']['metrics']['avg_score']} (Δ {gate['score_delta']:+})")
    print(f"  HitRate: {v1['summary']['metrics']['hit_rate']} -> "
          f"{v2['summary']['metrics']['hit_rate']} (Δ {gate['hit_rate_delta']:+})")
    print(f"  Cost  : ${v1['summary']['cost']['total_usd']} -> "
          f"${v2['summary']['cost']['total_usd']} (ratio {gate['cost_ratio']})")
    for name, ok in gate["checks"].items():
        print(f"   [{'✓' if ok else '✗'}] {name}")
    print(f"  ➡️  RELEASE GATE: {gate['decision']}")

    # summary.json = báo cáo V2 (bản đề xuất release) + phần regression
    final_summary = dict(v2["summary"])
    final_summary["regression"] = {
        "v1_metrics": v1["summary"]["metrics"],
        "v2_metrics": v2["summary"]["metrics"],
        "v1_cost": v1["summary"]["cost"],
        "v2_cost": v2["summary"]["cost"],
    }
    final_summary["release_decision"] = gate

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(REPORTS_DIR, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump({"v1": v1["results"], "v2": v2["results"]}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã lưu reports/summary.json & reports/benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
