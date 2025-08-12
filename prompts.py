# prompts.py
# -*- coding: utf-8 -*-
"""
Prompt builders for Qwen-VL anomaly detection.

要点：
- 文本只负责“话术”，不放图片占位符。
- 图片（few-shot + 待测）由上层以列表形式传入推理引擎。
- 如走 OpenAI 兼容 API，可用 build_api_messages 构造 blocks。
"""
from __future__ import annotations
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Sequence, Optional, Union, Dict
from PIL import Image

# ----------------------------
# Few-shot 文本块（仅文字说明）
# ----------------------------
def build_few_shot_examples_text(example_images: List[Tuple[str, str]],
                                 show_label: bool = True) -> str:
    """
    :param example_images: [(img_path, label)], label: 'p'(异常) / 'n'(正常) / 其它
    :return: 多个示例的文本描述（不包含图片占位符）
    """
    if not example_images:
        return "（示例图像见上）"
    parts: List[str] = []
    for i, (_img, label) in enumerate(example_images, start=1):
        if show_label and label in ("p", "n"):
            status = "异常" if label == "p" else "正常"
            parts.append(
                f"示例 {i}（{status}）\n"
                f"【判断】: {status}\n"
                f"【分析】: 请参考上方示例图像中的关键区域与纹理特征。\n"
            )
        else:
            parts.append(f"示例 {i}\n【提示】: 参考上方示例图像。")
    return "\n".join(parts).strip()

# ----------------------------
# 模板渲染（只负责文本）
# ----------------------------
# prompts.py（只贴需要替换的函数）
from typing import List, Tuple, Optional, Dict

def build_few_shot_examples_text(example_images: List[Tuple[str, str]],
                                 show_label: bool = True) -> str:
    if not example_images:
        return "（示例图像见上）"
    parts = []
    for i, (_img, label) in enumerate(example_images, start=1):
        tag = "异常" if (show_label and label == "p") else ("正常" if (show_label and label == "n") else "")
        head = f"示例#{i}" + (f"（{tag}）" if tag else "")
        parts.append(f"{head}\n【提示】: 参考上方示例图像。")
    return "\n".join(parts).strip()

def build_text_prompt(
    prompt_template: str,
    few_shot_examples: Optional[List[Tuple[str, str]]] = None,
    show_fewshot_label: bool = True,
    placeholders: Optional[Dict[str, str]] = None
) -> str:
    text = (prompt_template or "").strip()
    if not text:
        return ""

    # few-shot 占位
    if "{few_shot_examples}" in text:
        ex = build_few_shot_examples_text(few_shot_examples or [], show_fewshot_label)
        text = text.replace("{few_shot_examples}", ex)

    # 其它占位（例如 {image_order_hint}）
    for k, v in (placeholders or {}).items():
        token = "{" + k + "}"
        if token in text:
            text = text.replace(token, v or "")

    # 清除遗留占位
    if "<image>" in text or "{image_token}" in text:
        text = text.replace("<image>", "").replace("{image_token}", "")
    return text


# ----------------------------
# API 模式：图片转 data URI（如传本地路径）
# ----------------------------
def _path_to_data_uri(path: Union[str, Path]) -> str:
    p = Path(path)
    with Image.open(p) as img:
        fmt = (img.format or "PNG").upper()
        if fmt not in {"PNG", "JPEG", "JPG", "WEBP"}:
            fmt = "PNG"
        buf = BytesIO()
        img.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = "jpeg" if fmt in {"JPG", "JPEG"} else fmt.lower()
        return f"data:image/{mime};base64,{b64}"

def _to_image_url_block(url_or_path: str) -> dict:
    """
    输入可以是 http(s) 链接，也可以是本地路径（将被转为 data URI）。
    """
    if url_or_path.startswith(("http://", "https://", "data:")):
        url = url_or_path
    else:
        url = _path_to_data_uri(url_or_path)
    return {"type": "image_url", "image_url": {"url": url}}

# ----------------------------
# API 模式：构造 OpenAI Chat Completions 的 messages
# ----------------------------
def build_api_messages(
    prompt_text: str,
    image_list: Sequence[str],
    system_text: Optional[str] = None,
) -> List[dict]:
    """
    生成 OpenAI 风格的 messages：
      [{"role":"system","content":"..."}?, {"role":"user","content":[text_block, img_block, ...]}]
    """
    content_blocks: List[dict] = [{"type": "text", "text": prompt_text}]
    for p in image_list:
        content_blocks.append(_to_image_url_block(p))

    messages: List[dict] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": content_blocks})
    return messages

# ----------------------------
# 集成模式（ChatModel）：只返回文本 messages；图片另行传参
# ----------------------------
def build_integrated_messages(prompt_text: str, system_text: Optional[str] = None) -> List[dict]:
    messages: List[dict] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": prompt_text})
    return messages
