"""
SimpleRAG: 胶水层，组合 retriever + LLM + prompt。
单一实现，不做 ABC（设计原则）。

默认 prompt（论文 §5.2 英文版）：
    "Please answer the question based on the context.\n"
    "Context: {context}\nQuestion: {query}"

AgNews 专用 prompt（在 yaml 里覆盖）：
    "Complete the sentence based on the context: {context}"
"""
from typing import List, Dict


class SimpleRAG:
    def __init__(self, llm_service, retriever, top_k: int = 4,
                 prompt_template: str = None, no_rag: bool = False):
        self.llm = llm_service
        self.retriever = retriever
        self.top_k = top_k
        # 论文 §5.2 原文："Please answer the question based on the context."
        self.prompt_template = prompt_template or (
            "Please answer the question based on the context.\n"
            "Context: {context}\nQuestion: {query}"
        )
        # §7.1：无 RAG 对照实验，跳过 retrieval
        self.no_rag = no_rag

    def build_index(self, data: List[Dict]):
        docs = [d.get("answer", "") for d in data]
        self.retriever.build_index(docs)
        print(f"[SimpleRAG] built on retriever={self.retriever.kind}, {len(docs)} docs, no_rag={self.no_rag}")

    def retrieve(self, query: str, top_k: int = None) -> List[str]:
        if self.no_rag:
            return []
        return self.retriever.retrieve(query, top_k or self.top_k)

    def generate_answer(self, query: str) -> str:
        contexts = self.retrieve(query)
        context_str = "\n".join(contexts) if contexts else "(no context)"
        # 同时支持 {query} 和 {text} 两种占位符（AgNews 用 {text}，默认用 {query}）
        prompt = self.prompt_template.format(context=context_str, query=query, text=query)
        return self.llm.generate(prompt)
