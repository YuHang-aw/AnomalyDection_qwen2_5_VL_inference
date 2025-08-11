# inference_core.py
# -*- coding: utf-8 -*-
import abc
import base64
import time
from io import BytesIO
from typing import List, Dict, Optional, Sequence

# ✅ 正确的导入路径（注意：是 llamafactory，没有下划线）
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc

# OpenAI v1 客户端
try:
    from openai import OpenAI
except Exception:
    # 兼容旧写法：import openai; openai.OpenAI
    import openai as _openai
    OpenAI = getattr(_openai, "OpenAI")

from PIL import Image


# ----------------------------
# 抽象基类
# ----------------------------
class BaseInferenceCore(abc.ABC):
    @abc.abstractmethod
    def predict(self, text_prompt: str, image_paths: Sequence[str]) -> str:
        """执行一次预测。"""
        raise NotImplementedError

    def cleanup(self):
        """清理资源（默认空实现）。"""
        pass


# ----------------------------
# 工具：将本地文件转为 data URI（API 模式使用）
# ----------------------------
def _path_to_data_uri(image_path: str) -> str:
    with Image.open(image_path) as img:
        fmt = (img.format or "PNG").upper()
        if fmt not in {"PNG", "JPEG", "JPG", "WEBP"}:
            fmt = "PNG"
        buf = BytesIO()
        img.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = "jpeg" if fmt in {"JPG", "JPEG"} else fmt.lower()
        return f"data:image/{mime};base64,{b64}"


def _as_image_url_block(path_or_url: str) -> dict:
    """支持 http(s)/data URI 或本地路径（会自动转 data URI）"""
    if path_or_url.startswith(("http://", "https://", "data:")):
        url = path_or_url
    else:
        url = _path_to_data_uri(path_or_url)
    return {"type": "image_url", "image_url": {"url": url}}


# ----------------------------
# 模式一：集成（本地）推理
# ----------------------------
import re

class IntegratedInferenceCore(BaseInferenceCore):
    def __init__(self, model_config: Dict):
        print("正在初始化 [集成模式] 推理核心……")
        try:
            self.model = ChatModel(model_config)
        except Exception as e:
            msg = str(e)
            # 捕获 HfArgumentParser 的未使用键提示
            m = re.search(r"Some keys are not used by the HfArgumentParser:\s*\[(.*?)\]", msg)
            if m:
                bad = [k.strip(" '\"") for k in m.group(1).split(",")]
                for k in bad:
                    model_config.pop(k, None)
                print(f"[WARN] 自动移除未识别参数并重试：{bad}")
                self.model = ChatModel(model_config)  # 重试一次
            else:
                raise
        print("模型加载成功。")

    def predict(self, text_prompt: str, image_paths: Sequence[str]) -> str:
        # ChatModel 要求：文字放 messages，图片单独放 images 参数
        messages = [{"role": "user", "content": text_prompt}]
        try:
            response = []
            # stream_chat 返回生成器，逐块累加
            for new_text in self.model.stream_chat(messages, images=list(image_paths or [])):
                if new_text:
                    response.append(new_text)
            return "".join(response) if response else ""
        except RuntimeError as e:
            # 常见：CUDA OOM 等
            return f"[集成] 推理失败（RuntimeError）: {e}"
        except Exception as e:
            return f"[集成] 推理失败: {e}"

    def cleanup(self):
        try:
            if hasattr(self, "model"):
                del self.model
            torch_gc()
        except Exception:
            pass
        print("已清理本地模型与显存。")


# ----------------------------
# 模式二：API 客户端（OpenAI 兼容）
# ----------------------------
class ApiClientInferenceCore(BaseInferenceCore):
    def __init__(self, api_config: Dict):
        print("正在初始化 [API 客户端模式] 推理核心……")
        try:
            base_url = api_config.get("base_url")
            api_key = api_config.get("api_key")
            self.model_name = api_config.get("model_name")
            self.timeout_s = int(api_config.get("request_timeout_s", 120))
            self.max_retries = int(api_config.get("max_retries", 3))
            self.temperature = float(api_config.get("temperature", 0.1))
            self.max_tokens = int(api_config.get("max_tokens", 2048))

            if not base_url or not self.model_name:
                raise ValueError("api_client_settings.base_url 或 model_name 未配置")

            self.client = OpenAI(api_key=api_key or "sk-local", base_url=base_url)
            print(f"API 客户端就绪: {base_url}，模型={self.model_name}")
        except Exception as e:
            print(f"[API] 客户端初始化失败: {e}")
            raise

    def _once(self, text_prompt: str, image_paths: Sequence[str]) -> str:
        # 构造 OpenAI 风格消息：文本块 + 图片块
        content = [{"type": "text", "text": text_prompt}]
        for p in image_paths or []:
            content.append(_as_image_url_block(p))

        messages = [{"role": "user", "content": content}]

        # 发起请求
        r = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout_s,   # openai>=1.40 支持；低版本可改用 client.with_options(timeout=...)
        )
        return r.choices[0].message.content or ""

    def predict(self, text_prompt: str, image_paths: Sequence[str]) -> str:
        # 带重试的推理
        delay = 1.5
        last_err: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._once(text_prompt, image_paths)
            except Exception as e:
                last_err = f"[API] 第{attempt}次请求失败: {e}"
                print(last_err)
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
        return last_err or "[API] 未知错误"

    # API 客户端无需清理
    def cleanup(self):
        print("API 客户端模式无需清理。")
