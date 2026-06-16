"""
LLM 服务抽象层。

支持的 LLM_KIND：
    echo         — 无网络测试用，文本回显
    openai       — OpenAI 官方及 OpenAI 兼容 API（智谱、DeepSeek、Kimi 等）
    vllm-server  — 通过 HTTP 调用本地起的 vLLM OpenAI 兼容 server
    vllm-local   — 直接 import vllm 库调用，省 HTTP 开销，适合大批量推理

所有实现统一暴露：
    generate(prompt: str, config=None) -> str
    chat(messages: List[Message], config=None) -> str
其中 messages 的 content 可以是 str，也可以是 List[ContentPart]（图文混排）。

复现实验建议用 .env 文件管理配置，参考 .env.example。
"""
import abc
import os
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Literal

# 模块级加载 .env，确保单独 import 也能拿到 key
from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# 类型 / 数据结构
# ============================================================================

ChatTemplate = Literal["raw", "llama3", "mistral", "chatml", "auto"]

# Message: {"role": "user"|"assistant"|"system", "content": str | List[ContentPart]}
# ContentPart: {"type": "text", "text": str} | {"type": "image_url", "image_url": {"url": str}} | {"type": "image_pil", "image_pil": PIL.Image}
Message = Dict[str, Any]


@dataclass
class GenerationConfig:
    """所有 LLM 实现必须遵守这套生成参数以保证可复现性。"""
    max_new_tokens: int = 128
    temperature: float = 0.0   # DC-MIA 复现建议 0，否则 AUC 波动
    top_p: float = 1.0
    stop: Optional[List[str]] = None


@dataclass
class ModelInfo:
    """模型元信息。"""
    name: str
    provider: str
    chat_template: ChatTemplate = "raw"
    supports_multimodal: bool = False


# ============================================================================
# 抽象基类
# ============================================================================

class BaseLLM(abc.ABC):
    """所有 LLM 客户端的基类。"""

    def __init__(self, info: ModelInfo):
        self.info = info
        # 多模态预声明：vLLM library 模式在 __init__ 时需要这个
        self._mm_limits = {"image": 1} if info.supports_multimodal else {}

    @abc.abstractmethod
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """纯文本便利接口。内部应包成单条 user message 调 chat()。"""

    @abc.abstractmethod
    def chat(self, messages: List[Message], config: Optional[GenerationConfig] = None) -> str:
        """通用接口：messages 支持 str content 和 List[ContentPart]（图文混排）。"""


# ============================================================================
# 实现 0: EchoLLM —— 无网络测试用
# ============================================================================

class EchoLLM(BaseLLM):
    """把最后一行文本回显作为回答。取代旧的 is_mock 静默兜底。"""

    def __init__(self):
        super().__init__(ModelInfo(name="echo", provider="echo"))

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        return self.chat([{"role": "user", "content": prompt}], config)

    def chat(self, messages: List[Message], config: Optional[GenerationConfig] = None) -> str:
        parts = []
        for m in messages:
            c = m["content"]
            if isinstance(c, str):
                parts.append(c)
            else:
                for p in c:
                    if p.get("type") == "text":
                        parts.append(p["text"])
        text = "\n".join(parts)
        return (text.split("\n")[-1] or text)[:200]


# ============================================================================
# 实现 1: OpenAI 兼容客户端（覆盖 OpenAI / vllm-server / Ollama / TGI / 国产 API）
# ============================================================================

def _retry_with_tenacity(callable_, max_attempts: int = 5):
    """用 tenacity 指数退避重试；装不上时降级为手写循环 3 次。"""
    try:
        from tenacity import (
            retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
        )
        from openai import RateLimitError, APITimeoutError, APIConnectionError

        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
            reraise=True,
        )
        def _do():
            return callable_()

        return _do()
    except ImportError:
        # tenacity 没装，降级为手写重试 3 次
        last_err = None
        for attempt in range(3):
            try:
                return callable_()
            except Exception as e:
                last_err = e
                print(f"[OpenAICompatLLM] 重试 {attempt + 1}/3 失败: {e}")
                time.sleep(2 ** attempt)
        raise last_err


class OpenAICompatLLM(BaseLLM):
    """
    基于 OpenAI 协议的客户端。
    同一份代码覆盖：
      - OpenAI 官方（base_url=None）
      - vLLM 自托管 OpenAI server（http://localhost:8000/v1, key=EMPTY）
      - Ollama（http://localhost:11434/v1）
      - TGI / LM Studio
      - 国产 API（智谱 GLM / DeepSeek / Kimi 等）
    """

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        chat_template: ChatTemplate = "raw",
        provider: str = "openai",
        supports_multimodal: bool = False,
    ):
        api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        if not api_key:
            raise ValueError(
                "[OpenAICompatLLM] 未找到 API key（请通过参数传入，或设置环境变量 OPENAI_API_KEY / LLM_API_KEY）"
            )

        super().__init__(ModelInfo(
            name=model_name,
            provider=provider,
            chat_template=chat_template,
            supports_multimodal=supports_multimodal,
        ))

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        print(f"[OpenAICompatLLM] {provider} | model={model_name} | base={base_url or 'api.openai.com'}")

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        return self.chat([{"role": "user", "content": prompt}], config)

    def chat(self, messages: List[Message], config: Optional[GenerationConfig] = None) -> str:
        cfg = config or GenerationConfig()
        # 调前把温度等参数冻结到 messages 元数据（方便日志/调试时看）
        def _do_call():
            return self.client.chat.completions.create(
                model=self.info.name,
                messages=messages,
                max_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                stop=cfg.stop,
            )

        try:
            resp = _retry_with_tenacity(_do_call, max_attempts=5)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[OpenAICompatLLM] 调用失败（已重试完）: {e}")
            return ""


# ============================================================================
# 实现 2: vLLM 当 Python 库直接调（省 HTTP 开销）
# ============================================================================

class VLLMLibraryLLM(BaseLLM):
    """把 vLLM 当库直接 import 使用，适合大批量推理。"""

    def __init__(
        self,
        model_path: str,
        chat_template: ChatTemplate = "auto",
        max_model_len: int = 4096,
        supports_multimodal: Optional[bool] = None,
        gpu_memory_utilization: Optional[float] = None,
    ):
        # gpu_memory_utilization 优先从 env 读 (LLM_GPU_MEM_UTIL)，默认 0.85
        # 同卡共享 embedder 时建议 0.7-0.85；单独 vLLM 时可 0.9
        if gpu_memory_utilization is None:
            gpu_memory_utilization = float(os.environ.get("LLM_GPU_MEM_UTIL", "0.85"))

        # 自动检测多模态
        if supports_multimodal is None:
            supports_multimodal = self._detect_multimodal(model_path)

        super().__init__(ModelInfo(
            name=model_path,
            provider="vllm-local",
            chat_template=chat_template,
            supports_multimodal=supports_multimodal,
        ))

        from vllm import LLM
        kwargs = dict(
            model=model_path,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if supports_multimodal:
            kwargs["limit_mm_per_prompt"] = {"image": 1}

        self._llm = LLM(**kwargs)
        try:
            import torch
            n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except ImportError:
            n_gpu = 0
        print(f"[VLLMLibraryLLM] loaded {model_path} (multimodal={supports_multimodal}, "
              f"gpu={n_gpu}, mem_util={gpu_memory_utilization})")

    @staticmethod
    def _detect_multimodal(model_path: str) -> bool:
        """读 config.json 的 architectures 字段启发式判断。"""
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            arch = getattr(cfg, "architectures", []) or []
            return any(("Vision" in a) or ("VLM" in a) or ("VL" in a) for a in arch)
        except Exception:
            return False

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        return self.chat([{"role": "user", "content": prompt}], config)

    def chat(self, messages: List[Message], config: Optional[GenerationConfig] = None) -> str:
        cfg = config or GenerationConfig()

        from vllm import SamplingParams
        sp = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            stop=cfg.stop,
        )

        # 把 messages 转换成 vLLM 协议；image_url 远程的留 URL，本地路径转 image_pil
        from PIL import Image
        vllm_msgs = []
        for m in messages:
            c = m["content"]
            if isinstance(c, str):
                vllm_msgs.append({"role": m["role"], "content": c})
                continue

            parts = []
            for p in c:
                ptype = p.get("type")
                if ptype == "text":
                    parts.append({"type": "text", "text": p["text"]})
                elif ptype == "image_url":
                    url = p["image_url"]["url"]
                    if url.startswith(("data:image", "http://", "https://")):
                        parts.append({"type": "image_url", "image_url": {"url": url}})
                    else:
                        # 本地路径
                        parts.append({"type": "image_pil", "image_pil": Image.open(url)})
                elif ptype == "image_pil":
                    parts.append({"type": "image_pil", "image_pil": p["image_pil"]})
                else:
                    print(f"[VLLMLibraryLLM] 未知 content part type: {ptype}，已跳过")
            vllm_msgs.append({"role": m["role"], "content": parts})

        chat_template_arg = None if self.info.chat_template == "auto" else self.info.chat_template
        out = self._llm.chat(vllm_msgs, sampling_params=sp, chat_template=chat_template_arg)
        return out[0].outputs[0].text.strip()


# ============================================================================
# 向后兼容：旧类名 alias
# ============================================================================

LLMService = BaseLLM         # 旧名
APIBasedLLM = OpenAICompatLLM  # 旧名


# ============================================================================
# 工厂：从 .env 自动构造
# ============================================================================

def build_llm_from_env() -> BaseLLM:
    """
    从环境变量（.env）自动构造 LLM。

    必选：
      LLM_KIND = echo | openai | vllm-server | vllm-local

    openai / vllm-server / ollama / tgi 模式额外需要：
      LLM_MODEL, LLM_API_KEY (或 OPENAI_API_KEY)
      可选: LLM_BASE_URL, LLM_CHAT, LLM_MULTIMODAL

    vllm-local 模式额外需要：
      LLM_PATH
      可选: LLM_CHAT, LLM_MAX_LEN, LLM_MULTIMODAL
    """
    kind = os.environ.get("LLM_KIND", "").lower()
    if not kind:
        raise ValueError(
            "LLM_KIND 未设置。可选: echo | openai | vllm-server | vllm-local"
        )

    if kind == "echo":
        return EchoLLM()

    if kind == "vllm-local":
        path = os.environ.get("LLM_PATH")
        if not path:
            raise ValueError("LLM_KIND=vllm-local 时必须设置 LLM_PATH")
        mm = os.environ.get("LLM_MULTIMODAL", "auto").lower()
        return VLLMLibraryLLM(
            model_path=path,
            chat_template=os.environ.get("LLM_CHAT", "auto"),  # type: ignore[arg-type]
            max_model_len=int(os.environ.get("LLM_MAX_LEN", "4096")),
            supports_multimodal=None if mm == "auto" else (mm == "true"),
        )

    # openai / vllm-server / ollama / tgi 全部走 OpenAICompatLLM
    model = os.environ.get("LLM_MODEL")
    if not model:
        raise ValueError(f"LLM_KIND={kind} 时必须设置 LLM_MODEL")
    return OpenAICompatLLM(
        model_name=model,
        base_url=os.environ.get("LLM_BASE_URL"),
        api_key=os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        chat_template=os.environ.get("LLM_CHAT", "raw"),  # type: ignore[arg-type]
        provider=kind,
        supports_multimodal=os.environ.get("LLM_MULTIMODAL", "false").lower() == "true",
    )
