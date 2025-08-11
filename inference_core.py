import abc
import base64
from io import BytesIO
from typing import List, Dict

from llama_factory.model import ChatModel
from llama_factory.extras.misc import torch_gc
import openai
from PIL import Image

# --- 抽象基类，定义统一接口 ---
class BaseInferenceCore(abc.ABC):
    @abc.abstractmethod
    def predict(self, prompt: str, image_paths: List[str]) -> str:
        """
        执行一次预测。
        子类必须实现此方法。
        """
        raise NotImplementedError

    def cleanup(self):
        """
        清理资源。
        默认什么都不做，子类可以重写。
        """
        pass

# --- 模式一：集成模式，直接加载模型 ---
class IntegratedInferenceCore(BaseInferenceCore):
    def __init__(self, model_config: Dict):
        print("正在初始化 [集成模式] 推理核心...")
        try:
            self.model = ChatModel(model_config)
            print("模型加载成功。")
        except Exception as e:
            print(f"集成模式下模型加载失败: {e}")
            raise

    def predict(self, prompt: str, image_paths: List[str]) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            response = ""
            for new_text in self.model.stream_chat(messages, images=image_paths):
                response += new_text
            return response
        except Exception as e:
            print(f"集成模式推理出错: {e}")
            return "推理失败"

    def cleanup(self):
        print("正在卸载模型并清理GPU缓存...")
        del self.model
        torch_gc()
        print("清理完成。")

# --- 模式二：API客户端模式 ---
class ApiClientInferenceCore(BaseInferenceCore):
    def __init__(self, api_config: Dict):
        print("正在初始化 [API客户端模式] 推理核心...")
        try:
            self.client = openai.OpenAI(
                api_key=api_config['api_key'],
                base_url=api_config['base_url'],
            )
            self.model_name = api_config['model_name']
            print(f"API客户端已配置，目标服务: {api_config['base_url']}")
        except Exception as e:
            print(f"API客户端初始化失败: {e}")
            raise

    def _image_to_base64_url(self, image_path: str) -> str:
        try:
            with Image.open(image_path) as img:
                buffered = BytesIO()
                # 确定格式，优先使用源格式
                format = img.format if img.format in ['JPEG', 'PNG'] else 'PNG'
                img.save(buffered, format=format)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/{format.lower()};base64,{img_str}"
        except Exception as e:
            print(f"图片转Base64失败: {image_path}, 错误: {e}")
            return ""

    def predict(self, prompt: str, image_paths: List[str]) -> str:
        content = [{"type": "text", "text": prompt}]
        for path in image_paths:
            base64_url = self._image_to_base64_url(path)
            if base64_url:
                content.append({"type": "image_url", "image_url": {"url": base64_url}})
        
        messages = [{"role": "user", "content": content}]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=2048, # 可以根据需要调整
                temperature=0.1, # 可以根据需要调整
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用出错: {e}")
            return "API调用失败"

    # cleanup 在客户端模式下不需要做任何事
