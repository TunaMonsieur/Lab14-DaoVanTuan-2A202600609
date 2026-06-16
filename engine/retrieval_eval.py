"""
Đánh giá tầng Retrieval (độc lập với Generation).

Phải chứng minh retriever lấy đúng tài liệu TRƯỚC khi đánh giá câu trả lời,
nếu không sẽ không biết lỗi đến từ Retrieval hay từ Prompting/LLM.

Chỉ số:
  - Hit Rate@k: tỉ lệ case có ít nhất 1 expected_id nằm trong top-k.
  - MRR: trung bình 1/(vị trí expected_id đầu tiên xuất hiện).
"""
from typing import List, Dict, Optional

from engine.knowledge_base import TfidfRetriever, get_retriever


class RetrievalEvaluator:
    def __init__(self, retriever: Optional[TfidfRetriever] = None):
        self.retriever = retriever or get_retriever()

    @staticmethod
    def calculate_hit_rate(expected_ids: List[str], retrieved_ids: List[str],
                           top_k: int = 3) -> float:
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    @staticmethod
    def calculate_mrr(expected_ids: List[str], retrieved_ids: List[str]) -> float:
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict], top_k: int = 3) -> Dict:
        """
        Chạy retrieval cho mọi case trong dataset và tính Hit Rate / MRR.
        Mỗi case cần có 'expected_retrieval_ids'. Bỏ qua case không có nhãn này.
        """
        per_case = []
        for case in dataset:
            expected = case.get("expected_retrieval_ids") or []
            if not expected:
                continue
            hits = self.retriever.retrieve(case["question"], top_k=max(top_k, 5))
            retrieved_ids = [p.id for p, _ in hits]
            per_case.append({
                "question": case["question"],
                "expected_retrieval_ids": expected,
                "retrieved_ids": retrieved_ids[:top_k],
                "hit_rate": self.calculate_hit_rate(expected, retrieved_ids, top_k),
                "mrr": self.calculate_mrr(expected, retrieved_ids),
            })

        n = len(per_case) or 1
        return {
            "avg_hit_rate": sum(c["hit_rate"] for c in per_case) / n,
            "avg_mrr": sum(c["mrr"] for c in per_case) / n,
            "evaluated_cases": len(per_case),
            "top_k": top_k,
            "per_case": per_case,
        }
