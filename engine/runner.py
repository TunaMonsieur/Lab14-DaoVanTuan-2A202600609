"""
Async Benchmark Runner.

Mỗi test case chạy qua pipeline:
  1. Agent (RAG) trả lời  -> đo latency, token, cost.
  2. Retrieval Eval        -> Hit Rate@k, MRR (nếu case có expected_retrieval_ids).
  3. Multi-Judge           -> điểm consensus, agreement, conflict.

Chạy song song theo batch (asyncio.gather) để nhanh nhưng vẫn giới hạn để
không vượt rate limit của Gemini.
"""
import asyncio
import time
from typing import List, Dict

from engine.retrieval_eval import RetrievalEvaluator


class BenchmarkRunner:
    def __init__(self, agent, judge, retrieval_eval: RetrievalEvaluator = None,
                 top_k: int = 3):
        self.agent = agent
        self.judge = judge
        self.retrieval = retrieval_eval or RetrievalEvaluator()
        self.top_k = top_k

    async def run_single_test(self, test_case: Dict) -> Dict:
        start = time.perf_counter()
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start

        # --- Retrieval metrics ---
        expected_ids = test_case.get("expected_retrieval_ids") or []
        retrieved_ids = response.get("retrieved_ids", [])
        if expected_ids:
            hit = self.retrieval.calculate_hit_rate(expected_ids, retrieved_ids, self.top_k)
            mrr = self.retrieval.calculate_mrr(expected_ids, retrieved_ids)
            has_label = True
        else:
            hit, mrr, has_label = None, None, False

        # --- Multi-Judge ---
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"], response["answer"], test_case["expected_answer"]
        )

        meta = response.get("metadata", {})
        return {
            "question": test_case["question"],
            "category": test_case.get("metadata", {}).get("category"),
            "type": test_case.get("metadata", {}).get("type"),
            "agent_response": response["answer"],
            "expected_answer": test_case["expected_answer"],
            "expected_retrieval_ids": expected_ids,
            "retrieved_ids": retrieved_ids,
            "retrieval": {"hit_rate": hit, "mrr": mrr, "has_label": has_label},
            "judge": judge_result,
            "latency": round(latency, 3),
            "tokens": meta.get("prompt_tokens", 0) + meta.get("completion_tokens", 0),
            "agent_cost": meta.get("cost", 0.0),
            "status": "pass" if judge_result["final_score"] >= 3 else "fail",
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 4) -> List[Dict]:
        results: List[Dict] = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self.run_single_test(c) for c in batch]
            )
            results.extend(batch_results)
            print(f"  ...đã chạy {len(results)}/{len(dataset)} cases")
            # lưu cache judge sau mỗi batch (an toàn nếu hết quota giữa chừng)
            if hasattr(self.judge, "save_cache"):
                self.judge.save_cache()
        return results
