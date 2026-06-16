"""
SimpleRAG: 胶水层，组合 retriever + LLM + prompt。
单一实现，不做 ABC（设计原则）。
"""
from typing import List, Dict


class SimpleRAG:
    def __init__(self, llm_service, retriever, top_k: int = 4, prompt_template: str = None):
        self.llm = llm_service
        self.retriever = retriever
        self.top_k = top_k
        self.prompt_template = prompt_template or (
            "Please answer the question based on the provided context.\n"
            "Context: {context}\nQuestion: {question}"
        )

    def build_index(self, data: List[Dict]):
        docs = [d.get("answer", "") for d in data]
        self.retriever.build_index(docs)
        print(f"[SimpleRAG] built on retriever={self.retriever.kind}, {len(docs)} docs")

    def retrieve(self, query: str, top_k: int = None) -> List[str]:
        return self.retriever.retrieve(query, top_k or self.top_k)

    def generate_answer(self, query: str) -> str:
        contexts = self.retrieve(query)
        context_str = "\n".join(contexts)
        prompt = self.prompt_template.format(context=context_str, question=query)
        return self.llm.generate(prompt)
