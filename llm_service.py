import abc
import os
import time

class LLMService(abc.ABC):
    """
    LLM 服务抽象基类。
    负责向 RAG 系统提供统一的文本生成接口。
    """
    @abc.abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        pass


class LocalTransformersLLM(LLMService):
    """
    基于 HuggingFace Transformers 的本地推理实现。
    如果你有本地 GPU 并且想用 Unsloth 优化，可以在这里引入 Unsloth 的 FastLanguageModel。
    """
    def __init__(self, model_name_or_path: str, use_unsloth: bool = False):
        self.model_name_or_path = model_name_or_path
        self.use_unsloth = use_unsloth
        print(f"[LLMService] 正在加载本地模型: {model_name_or_path} (Unsloth: {use_unsloth})")
        
        # 为了演示，此处不实际加载模型（会阻塞太久），预留加载代码位置：
        # if self.use_unsloth:
        #     from unsloth import FastLanguageModel
        #     self.model, self.tokenizer = FastLanguageModel.from_pretrained(model_name_or_path, load_in_4bit=True)
        # else:
        #     from transformers import AutoModelForCausalLM, AutoTokenizer
        #     self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        #     self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
        
        # 标记为 Mock 状态
        self.is_mock = True 

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        if self.is_mock:
            # 假装在推理
            return "This is a mocked response based on the context."
        
        # 实际推理代码预留：
        # inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        # outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class APIBasedLLM(LLMService):
    """
    基于外部 API（如 OpenAI, vLLM Server 等）的推理实现。
    """
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "mock-key")
        self.base_url = base_url
        self.model_name = model_name
        print(f"[LLMService] 已初始化 API LLM 客户端: {self.model_name}")
        # import openai
        # self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        # 实际 API 调用预留：
        # response = self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=max_new_tokens
        # )
        # return response.choices[0].message.content
        return "This is a mocked API response based on the context."
