"""
检索器抽象层。

支持的 retriever kind（通过 RAGConfig.embedding_model 字段切换）：
  minilm   - FAISS IndexFlatL2 + all-MiniLM-L6-v2（论文 §5.2 默认）
  bge      - FAISS IndexFlatL2 + BAAI/bge-small-en-v1.5（论文 §6 消融）
  bm25     - rank_bm25.BM25Okapi（论文 §6 消融，稀疏检索）
  mock     - 随机向量（仅 dev/CI 用，无 GPU 时跑流程）

所有实现统一暴露：
    build_index(docs: List[str])              # 文本列表
    retrieve(query: str, top_k: int) -> List[str]
    encode(text: str) -> np.ndarray           # 仅 dense 类有意义；BM25/mock raise

设计原则：retriever 在 main.py 构造一次，跨 SimpleRAG / DCMIA 共享。
SimpleRAG 不做 ABC（单实现胶水），DCMIA 也不做 ABC（单实现胶水）。
"""
import abc
import hashlib
import numpy as np
from typing import List, Dict, Optional


class BaseRetriever(abc.ABC):
    def __init__(self, kind: str, name: str):
        self.kind = kind
        self.name = name

    @abc.abstractmethod
    def build_index(self, docs: List[str]):
        """docs 是文本列表（不是 dict 列表）。"""

    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[str]:
        """返回 top-k 相关文档文本。"""

    def encode(self, text: str) -> np.ndarray:
        """仅 dense retriever 适用；BM25 / mock raise NotImplementedError。"""
        raise NotImplementedError(f"{self.kind} retriever 不支持 encode()")

    def warmup(self, texts: List[str]):
        """可选：批量预热缓存（仅 DenseRetriever 实现）。"""
        pass


# ============================================================================
# Dense Retriever: FAISS + SentenceTransformer
# ============================================================================

class DenseRetriever(BaseRetriever):
    """FAISS IndexFlatL2 + SentenceTransformer。共享文本级 cache。"""

    def __init__(self, model_name: str, kind: str):
        super().__init__(kind=kind, name=model_name)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._cache: Dict[str, np.ndarray] = {}
        self.index = None
        self.docs: List[str] = []
        print(f"[DenseRetriever/{kind}] loaded {model_name} (dim={self.model.get_sentence_embedding_dimension()})")

    def encode(self, text: str) -> np.ndarray:
        if text not in self._cache:
            emb = self.model.encode([text], normalize_embeddings=True)[0]
            self._cache[text] = emb.astype("float32")
        return self._cache[text]

    def warmup(self, texts: List[str]):
        """批量预热：避免影子 RAG × 8000 docs 重复编码。"""
        uncached = [t for t in texts if t not in self._cache]
        if not uncached:
            return
        embs = self.model.encode(uncached, normalize_embeddings=True, batch_size=64)
        for t, e in zip(uncached, embs):
            self._cache[t] = e.astype("float32")
        print(f"[DenseRetriever/{self.kind}] warmup cached {len(uncached)} texts (total: {len(self._cache)})")

    def build_index(self, docs: List[str]):
        self.docs = docs
        embeddings = np.stack([self.encode(t) for t in docs])
        import faiss
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        print(f"[DenseRetriever/{self.kind}] FAISS IndexFlatL2 built: {len(docs)} docs, dim={dim}")

    def retrieve(self, query: str, top_k: int) -> List[str]:
        if self.index is None or not self.docs:
            return []
        q_emb = self.encode(query).reshape(1, -1)
        distances, indices = self.index.search(q_emb, top_k)
        return [self.docs[i] for i in indices[0] if 0 <= i < len(self.docs)]


# ============================================================================
# BM25 Retriever: 稀疏检索（论文 §6 消融）
# ============================================================================

class BM25Retriever(BaseRetriever):
    """rank_bm25.BM25Okapi 稀疏检索。"""

    def __init__(self):
        super().__init__(kind="bm25", name="bm25-okapi")
        self.bm25 = None
        self.docs: List[str] = []
        self.tokenized_docs: List[List[str]] = []
        print("[BM25Retriever] initialized (sparse retrieval)")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    def build_index(self, docs: List[str]):
        from rank_bm25 import BM25Okapi
        self.docs = docs
        self.tokenized_docs = [self._tokenize(d) for d in docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"[BM25Retriever] BM25Okapi built: {len(docs)} docs")

    def retrieve(self, query: str, top_k: int) -> List[str]:
        if self.bm25 is None or not self.docs:
            return []
        scores = self.bm25.get_scores(self._tokenize(query))
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.docs[i] for i in top_indices if 0 <= i < len(self.docs)]


# ============================================================================
# Mock Retriever: 随机向量（dev/CI 用）
# ============================================================================

class MockRetriever(BaseRetriever):
    """纯随机——只在没 GPU / 不下载模型时跑流程。"""

    def __init__(self):
        super().__init__(kind="mock", name="mock-random")
        self.docs: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        print("[MockRetriever] initialized (random retrieval, no model)")

    def encode(self, text: str) -> np.ndarray:
        # 用 text 的 hash 做 seed 让 encode 行为稳定
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return rng.random(384).astype("float32")

    def build_index(self, docs: List[str]):
        self.docs = docs
        self.embeddings = np.stack([self.encode(d) for d in docs])

    def retrieve(self, query: str, top_k: int) -> List[str]:
        if not self.docs:
            return []
        rng = np.random.default_rng(int(hashlib.md5(query.encode("utf-8")).hexdigest()[:8], 16))
        idx = rng.choice(len(self.docs), size=min(top_k, len(self.docs)), replace=False)
        return [self.docs[i] for i in idx]


# ============================================================================
# 工厂：从 RAGConfig 构造
# ============================================================================

def build_retriever_from_config(embedding_model: str):
    """
    embedding_model 取值:
      minilm / MiniLM / all-MiniLM-L6-v2 → FAISS + MiniLM
      bge    / BGE / BAAI/bge-*           → FAISS + BGE-en
      bm25   / bm25-okapi                 → rank_bm25 稀疏
      mock   / random                     → 随机向量
    """
    if embedding_model in ("mock", "random"):
        return MockRetriever()
    if embedding_model in ("bm25", "bm25-okapi"):
        return BM25Retriever()
    if embedding_model.startswith("BAAI/") or embedding_model.lower() == "bge":
        return DenseRetriever(model_name=embedding_model, kind="bge")
    if "MiniLM" in embedding_model or embedding_model == "minilm":
        return DenseRetriever(
            model_name="all-MiniLM-L6-v2" if embedding_model == "minilm" else embedding_model,
            kind="minilm",
        )
    print(f"[RetrieverFactory] 未知 embedding_model='{embedding_model}', fallback 到 minilm")
    return DenseRetriever(model_name="all-MiniLM-L6-v2", kind="minilm")
