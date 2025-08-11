# prompts.py
# -*- coding: utf-8 -*-
"""
Prompt builders for Qwen-VL anomaly detection with LLaMA-Factory.

- Integrated (ChatModel) mode:
    * Use build_text_prompt(...) to get the text prompt (NO <image> token).
    * Pass images as a separate list to ChatModel.stream_chat(..., images=images).

- API (OpenAI-style) mode:
    * Use build_api_messages(...) to get messages, where content is a list of
      {"type":"text"} / {"type":"image_url"} blocks.

Refs:
- Official API example uses image_url blocks (scripts/api_example/test_image.py).
- Chat engine signature shows images passed separately to stream_chat(..., images,...).
"""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Sequence, Optional, Union

from PIL import Image

# ----------------------------
# Few-shot 文本块（不插入图片占位符）
# ----------------------------
def build_few_shot_examples_text(example_images: List[Tuple[str, str]]) -> str:
    """
    :param example_images: [(img_path, label)], label: 'p'异常 / 'n'正常
    :return: 多个示例的文本描述（不包含图片占位符）
    """
    parts: List[str] = []
    for i, (_img, label) in enumerate(example_images, start=1):
        status = "异常" if label == "p" else "正常"
        if status == "异常":
            parts.append(
                f"示例 {i}（异常）\n"
                f"【判断】: 异常\n"
                f"【分析】: 图片中存在结构性缺陷（裂缝/渗水/破损等）。\n"
            )
        else:
            parts.append(
                f"示例 {i}（正常）\n"
                f"【判断】: 正常\n"
                f"【分析】: 衬砌板表面完整，无明显缺陷。\n"
            )
    return "\n".join(parts).strip()


# ----------------------------
# 模板渲染（只负责文本）
# ----------------------------
def build_text_prompt(
    prompt_template: str,
    few_shot_examples: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    渲染配置里的 prompt 模板，只产出【文本】。
    注意：不再使用 <image> 占位符，图片由上层分别传入（集成模式：images参数；API模式：image_url分块）。

    可用占位符：
      - {few_shot_examples}（可选）

    """
    text = prompt_template
    if "{few_shot_examples}" in text:
        if not few_shot_examples:
            raise ValueError("该模板需要 few_shot_examples，但未提供。")
        ex = build_few_shot_examples_text(few_shot_examples)
        text = text.replace("{few_shot_examples}", ex)
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
    输入可以是 http(s) 链接，也可以是本地路径。
    - http(s) 直接作为 url 传递；
    - 本地路径会被转为 data URI，兼容官方 API 结构。
    """
    if url_or_path.startswith("http://") or url_or_path.startswith("https://") or url_or_path.startswith("data:"):
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
      [
        {"role": "system", "content": "..."}?,            # 可选
        {"role": "user",   "content": [text_block, img_block, ...]}
      ]

    :param prompt_text: 渲染完成的文本提示词（build_text_prompt 的结果）
    :param image_list:  图片路径或 URL（按你希望模型看到的顺序）
    :param system_text: 可选，system 指令
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
# 集成模式（ChatModel）用：只返回 messages 文本，图片另行传参
# ----------------------------
def build_integrated_messages(prompt_text: str, system_text: Optional[str] = None) -> List[dict]:
    """
    ChatModel.stream_chat(...) 期望：
      - messages: 纯文本对话列表
      - images:   单独的图片列表参数（由调用方传入）
    """
    messages: List[dict] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": prompt_text})
    return messages
