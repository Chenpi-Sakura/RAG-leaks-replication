"""
基于 FAISS（内存版）的简化 RAG 系统。

当前实现是 mock（随机 embedding + print 占位）；
真实 all-MiniLM-L6-v2 + faiss.IndexFlatL2 接入留给后续 PR。
"""
import numpy as np
from typing import List, Dict


class SimpleRAG:
    def __init__(self, llm_service, top_k: int = 4, prompt_template: str = None):
        self.llm = llm_service
        self.top_k = top_k
        self.prompt_template = prompt_template or (
            "Please answer the question based on the provided context.\n"
            "Context: {context}\nQuestion: {question}"
        )
        self.index = None
        self.texts = []
        self.embedder = None          # 真实 SentenceTransformer 由后续 PR 注入
        self.is_mock_embed = True
        if self.is_mock_embed:
            print("[SimpleRAG] 使用 Mock Embedder（随机向量）...")

    def _get_embedding(self, text: str) -> np.ndarray:
        if self.is_mock_embed:
            return np.random.rand(384).astype("float32")
        return None

    def build_index(self, data: List[Dict]):
        self.texts = [d.get("answer", "") for d in data]
        # 真实环境会调 faiss.IndexFlatL2 建索引
        print(f"[SimpleRAG] 构建包含 {len(data)} 条数据的索引 (Mock).")

    def retrieve(self, query: str, top_k: int = None) -> List[str]:
        k = top_k or self.top_k
        # 真实环境会跑 faiss 检索
        return [f"Mocked context {i} for {query}" for i in range(k)]

    def generate_answer(self, query: str) -> str:
        contexts = self.retrieve(query)
        context_str = "\n".join(contexts)
        prompt = self.prompt_template.format(context=context_str, question=query)
        return self.llm.generate(prompt)
