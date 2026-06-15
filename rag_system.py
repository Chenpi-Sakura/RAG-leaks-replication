import numpy as np
from typing import List, Dict
import os

class SimpleRAG:
    """
    一个基于 FAISS (内存版) 的简化 RAG 系统。
    提供快速的 Index 构建、相似度检索和 LLM 生成。
    """
    def __init__(self, llm_service, embedding_model=None):
        self.llm = llm_service
        self.index = None
        self.texts = []
        
        # 为了避免重依赖，此处 Mock Embedding 模型。
        # 真实环境中应该使用: 
        # from sentence_transformers import SentenceTransformer
        # self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedder = embedding_model
        
        self.is_mock_embed = (self.embedder is None)
        if self.is_mock_embed:
            print("[SimpleRAG] 正在使用 Mock Embedder (随机向量) 加速测试...")

    def _get_embedding(self, text: str) -> np.ndarray:
        if self.is_mock_embed:
            # 返回随机 384 维向量 (MiniLM 的维度)
            return np.random.rand(384).astype('float32')
        # return self.embedder.encode(text).astype('float32')
        return None

    def build_index(self, data: List[Dict]):
        """
        根据输入数据列表构建 FAISS 索引。
        """
        # import faiss
        # self.texts = [d['answer'] for d in data] # 真实 RAG 这里存的应该是上下文段落
        # embeddings = np.array([self._get_embedding(t) for t in self.texts])
        # dim = embeddings.shape[1]
        # self.index = faiss.IndexFlatL2(dim)
        # self.index.add(embeddings)
        print(f"[SimpleRAG] 成功构建包含 {len(data)} 条数据的 FAISS 索引 (Mock).")
        
    def retrieve(self, query: str, top_k: int = 4) -> List[str]:
        """
        返回检索到的 Top-K 上下文文本。
        """
        # if self.index is None: return []
        # q_emb = self._get_embedding(query).reshape(1, -1)
        # distances, indices = self.index.search(q_emb, top_k)
        # return [self.texts[i] for i in indices[0] if i < len(self.texts)]
        
        return [f"Mocked context {i} for {query}" for i in range(top_k)]

    def generate_answer(self, query: str) -> str:
        """
        端到端：检索 + 生成
        """
        contexts = self.retrieve(query)
        context_str = "\n".join(contexts)
        
        prompt = f"Please answer the question based on the provided context.\nContext: {context_str}\nQuestion: {query}"
        
        answer = self.llm.generate(prompt)
        return answer
