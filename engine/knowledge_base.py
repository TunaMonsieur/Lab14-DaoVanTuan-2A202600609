"""
Knowledge Base + Retriever cho NovaCloud Support Agent.

- Đọc corpus markdown trong data/corpus/, chunk theo heading cấp 2 (##).
- Mỗi passage có ID ổn định dạng "<file_stem>#<index>" để dùng làm
  Ground Truth ID khi tính Hit Rate / MRR.
- Retriever dùng TF-IDF + cosine similarity (thuần Python, không cần model ngoài).
  Đây là retrieval THẬT (không mock): điểm số được tính từ nội dung corpus.
"""
import os
import re
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")

# Stopwords tiếng Việt + tiếng Anh phổ biến (giảm nhiễu cho TF-IDF)
_STOPWORDS = {
    "và", "là", "của", "có", "được", "cho", "các", "một", "với", "khi", "trong",
    "để", "không", "này", "đó", "những", "theo", "đã", "sẽ", "bị", "hay", "hoặc",
    "thì", "mà", "nếu", "tôi", "bạn", "người", "dùng", "về", "trên", "vào", "ra",
    "the", "a", "an", "of", "to", "in", "is", "are", "for", "and", "or", "on",
}

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Tách token: hạ thường, bỏ stopwords và token quá ngắn."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


@dataclass
class Passage:
    id: str
    doc: str          # tên file nguồn
    title: str        # heading của section
    text: str         # nội dung section (kèm title)
    tokens: List[str] = field(default_factory=list)


def load_corpus(corpus_dir: str = CORPUS_DIR) -> List[Passage]:
    """Đọc toàn bộ file .md, chunk theo heading '## '."""
    passages: List[Passage] = []
    corpus_dir = os.path.abspath(corpus_dir)
    if not os.path.isdir(corpus_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục corpus: {corpus_dir}")

    for fname in sorted(os.listdir(corpus_dir)):
        if not fname.endswith(".md"):
            continue
        stem = os.path.splitext(fname)[0]
        with open(os.path.join(corpus_dir, fname), "r", encoding="utf-8") as f:
            content = f.read()

        # Chunk theo heading cấp 2
        sections = re.split(r"(?m)^##\s+", content)
        idx = 0
        for sec in sections:
            sec = sec.strip()
            if not sec or sec.startswith("# "):  # bỏ tiêu đề cấp 1 đứng đầu file
                continue
            lines = sec.split("\n", 1)
            title = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            full = f"{title}. {body}"
            pid = f"{stem}#{idx}"
            passages.append(Passage(
                id=pid, doc=fname, title=title, text=full, tokens=tokenize(full)
            ))
            idx += 1
    return passages


class TfidfRetriever:
    """Vector retriever đơn giản dựa trên TF-IDF + cosine similarity."""

    def __init__(self, passages: List[Passage]):
        self.passages = passages
        self.N = len(passages)
        self.idf: Dict[str, float] = self._compute_idf()
        self.vectors: List[Dict[str, float]] = [
            self._tfidf_vector(p.tokens) for p in passages
        ]
        self.norms: List[float] = [self._norm(v) for v in self.vectors]

    def _compute_idf(self) -> Dict[str, float]:
        df = Counter()
        for p in self.passages:
            for term in set(p.tokens):
                df[term] += 1
        # smoothed idf
        return {t: math.log((self.N + 1) / (c + 1)) + 1.0 for t, c in df.items()}

    def _tfidf_vector(self, tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        total = sum(tf.values()) or 1
        return {
            t: (cnt / total) * self.idf.get(t, math.log(self.N + 1) + 1.0)
            for t, cnt in tf.items()
        }

    @staticmethod
    def _norm(vec: Dict[str, float]) -> float:
        return math.sqrt(sum(v * v for v in vec.values())) or 1e-9

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Passage, float]]:
        q_vec = self._tfidf_vector(tokenize(query))
        q_norm = self._norm(q_vec)
        scored = []
        for i, p in enumerate(self.passages):
            vec, norm = self.vectors[i], self.norms[i]
            # cosine = dot / (|q| * |d|); duyệt theo vector ngắn hơn
            small, big = (q_vec, vec) if len(q_vec) < len(vec) else (vec, q_vec)
            dot = sum(val * big.get(term, 0.0) for term, val in small.items())
            scored.append((p, dot / (q_norm * norm)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# Singleton tiện dùng lại
_RETRIEVER: TfidfRetriever | None = None


def get_retriever() -> TfidfRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = TfidfRetriever(load_corpus())
    return _RETRIEVER


if __name__ == "__main__":
    r = get_retriever()
    print(f"Đã nạp {r.N} passages từ corpus.")
    for p in r.passages:
        print(f"  {p.id:20s} | {p.title}")
    print("\nTest truy vấn: 'làm sao đổi mật khẩu?'")
    for p, score in r.retrieve("làm sao đổi mật khẩu?", top_k=3):
        print(f"  [{score:.3f}] {p.id} - {p.title}")
