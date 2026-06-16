"""
NovaCloud Support Agent - kiến trúc RAG THẬT.

Hỗ trợ 2 phiên bản để chạy Regression Testing:
  - "v1": baseline yếu (ít context, prompt sơ sài, dễ hallucination).
  - "v2": tối ưu (nhiều context hơn, prompt grounded + chỉ dẫn từ chối khi
    không có thông tin trong tài liệu).

Agent dùng:
  - Retrieval: TfidfRetriever trên corpus NovaCloud (engine/knowledge_base.py).
  - Generation: model local qua Ollama (engine/llm_client.py).
"""
import asyncio
from typing import Dict

from engine import config
from engine.knowledge_base import get_retriever
from engine.llm_client import ollama_chat

_V1_SYSTEM = (
    "Bạn là trợ lý hỗ trợ khách hàng của NovaCloud. Trả lời câu hỏi của người dùng."
)

_V2_SYSTEM = (
    "Bạn là trợ lý hỗ trợ khách hàng của NovaCloud. "
    "CHỈ trả lời dựa trên CONTEXT được cung cấp bên dưới. "
    "Nếu CONTEXT không chứa thông tin để trả lời, hãy nói rõ: "
    "'Tôi không tìm thấy thông tin này trong tài liệu.' Tuyệt đối không bịa đặt. "
    "Trả lời ngắn gọn, chuyên nghiệp, đúng trọng tâm bằng tiếng Việt."
)


class MainAgent:
    def __init__(self, version: str = "v1"):
        self.version = version
        self.name = f"NovaSupportAgent-{version}"
        self.retriever = get_retriever()
        if version == "v2":
            self.top_k = 4
            self.system = _V2_SYSTEM
            self.temperature = 0.1
        else:
            self.top_k = 2
            self.system = _V1_SYSTEM
            self.temperature = 0.5

    async def query(self, question: str) -> Dict:
        # 1. Retrieval
        hits = self.retriever.retrieve(question, top_k=self.top_k)
        contexts = [p.text for p, _ in hits]
        retrieved_ids = [p.id for p, _ in hits]

        # 2. Generation
        context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        user_prompt = f"CONTEXT:\n{context_block}\n\nCÂU HỎI: {question}\n\nTRẢ LỜI:"
        resp = await ollama_chat(
            config.AGENT_MODEL, self.system, user_prompt, temperature=self.temperature
        )

        return {
            "answer": resp.text.strip(),
            "contexts": contexts,
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "version": self.version,
                "model": resp.model,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "cost": resp.cost,
            },
        }


if __name__ == "__main__":
    config.setup_utf8()

    async def test():
        for v in ("v1", "v2"):
            agent = MainAgent(version=v)
            resp = await agent.query("Làm thế nào để đổi mật khẩu?")
            print(f"\n=== {agent.name} ===")
            print("Retrieved:", resp["retrieved_ids"])
            print("Answer:", resp["answer"][:300])
            print("Tokens:", resp["metadata"]["prompt_tokens"], "+",
                  resp["metadata"]["completion_tokens"])

    asyncio.run(test())
