#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

# LoRA / QLoRA
from peft import PeftModel

# 可选：官方工具（若缺失则用本地兜底）
try:
    from qwen_vl_utils import process_vision_info as _official_process_vision_info
except Exception:
    _official_process_vision_info = None


# ============== 数据读取 ==============
def load_json_or_jsonl(path: Path):
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, list) else [obj]
    else:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    rows.append(json.loads(ln))
        return rows


# ============== 消息构造 ==============
def to_qwen_messages_from_messages(sample: dict):
    """兼容你最初的数据格式：{'messages': [...]} —— 推理时去掉 assistant 段"""
    msgs = sample.get("messages", [])
    out = []
    for m in msgs:
        if m.get("role") == "assistant":
            break
        out.append(m)
    if not out:
        out = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
    return out


def to_qwen_messages_from_sharegpt(sample: dict):
    """
    兼容 LLaMA-Factory sharegpt+images：
    {
      "conversations":[{"from":"system","value":...},{"from":"human","value":"...<image>..."},{"from":"gpt","value": "..."}],
      "images": ["/path/img1.png", "/path/img2.png", ...]
    }
    把 human 文本中的 <image> 占位逐一替换成 Qwen 图像块，并与 images 一一匹配。
    """
    conv = sample.get("conversations", [])
    imgs: List[str] = list(sample.get("images", []))
    messages = []

    def add_user_turn(text: str, imgs_pool: List[str]):
        parts = text.split("<image>")
        content = []
        for i, seg in enumerate(parts):
            seg = seg.strip()
            if seg:
                content.append({"type": "text", "text": seg})
            if i < len(parts) - 1:
                if imgs_pool:
                    content.append({"type": "image", "image": imgs_pool.pop(0)})
                else:
                    # 若出现占位大于图片数，放一条提示文本避免空值
                    content.append({"type": "text", "text": "[[缺少对应图片]]"})
        if not content:
            content = [{"type": "text", "text": ""}]
        messages.append({"role": "user", "content": content})

    for turn in conv:
        role = turn.get("from")
        val = turn.get("value", "")
        if role == "system":
            messages.append({"role": "system", "content": [{"type": "text", "text": val}]})
        elif role in ("human", "user"):
            add_user_turn(val, imgs)
        elif role in ("gpt", "assistant"):
            break

    # 若还有多余图片（human 文本没有 <image>），附在最后一个 user
    if imgs:
        for m in reversed(messages):
            if m["role"] == "user":
                for p in imgs:
                    m["content"].append({"type": "image", "image": p})
                imgs = []
                break

    if not messages:
        messages = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
    return messages


def build_messages(sample: dict):
    if "messages" in sample:
        return to_qwen_messages_from_messages(sample)
    if "conversations" in sample:
        return to_qwen_messages_from_sharegpt(sample)
    raise ValueError("无法识别的数据格式：既无 'messages' 也无 'conversations'。")


# ============== 视觉输入准备（官方或兜底） ==============
def process_vision_info_local(messages) -> Tuple[list, list]:
    """
    兜底版：从 messages 中收集所有 image/video 字段。
    结构与官方 qwen_vl_utils.process_vision_info 返回一致：(image_inputs, video_inputs)
    """
    image_inputs, video_inputs = [], []
    for m in messages:
        for c in m.get("content", []):
            if c.get("type") == "image" and c.get("image"):
                image_inputs.append(c["image"])
            if c.get("type") == "video" and c.get("video"):
                video_inputs.append(c["video"])
    return image_inputs, video_inputs


def process_vision_info(messages):
    if _official_process_vision_info is not None:
        return _official_process_vision_info(messages)
    return process_vision_info_local(messages)


# ============== 模型加载 ==============
def build_quant_config(load_in_4bit=False, load_in_8bit=False, compute_dtype="bfloat16"):
    if not (load_in_4bit or load_in_8bit):
        return None
    dtype = torch.bfloat16 if str(compute_dtype).lower().startswith("bf16") else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_model_and_processor(
    base: str,
    adapter: str,
    device: str = "auto",         # "auto" / "cuda" / "cpu"
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    min_pixels: int = None,
    max_pixels: int = None,
    merge_lora: bool = False,     # QLoRA 不可 merge，普通 LoRA 可尝试
):
    qconf = build_quant_config(load_in_4bit, load_in_8bit)

    # 重要：不启用 flash-attn，这里不传 attn_implementation
    base_kwargs = dict(
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    if qconf is not None:
        base_kwargs["quantization_config"] = qconf

    if device == "auto":
        base_kwargs["device_map"] = "auto"
    elif device == "cpu":
        base_kwargs["device_map"] = None
    else:  # "cuda" 单卡
        base_kwargs["device_map"] = {"": 0}

    print(f"[INFO] loading base: {base} | device={device} | 4bit={load_in_4bit} 8bit={load_in_8bit}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base, **base_kwargs
    )

    # 单设备手动放到 cuda/cpu（注意：device_map=auto 时不能再手动 .to）
    if device in ("cuda", "cpu") and base_kwargs.get("device_map") is None:
        model = model.to(device)

    print(f"[INFO] attaching adapter: {adapter}")
    model = PeftModel.from_pretrained(model, adapter)

    if merge_lora:
        try:
            print("[INFO] trying merge_and_unload (skip if QLoRA 4bit)")
            model = model.merge_and_unload()
        except Exception as e:
            print(f"[WARN] merge_and_unload failed (this is expected for QLoRA): {e}")

    model.eval()

    # Processor（同 base）
    proc_kwargs = {}
    if min_pixels is not None: proc_kwargs["min_pixels"] = int(min_pixels)
    if max_pixels is not None: proc_kwargs["max_pixels"] = int(max_pixels)
    processor = AutoProcessor.from_pretrained(base, **proc_kwargs)
    return model, processor, (device == "auto")


# ============== 推理 ==============
@torch.inference_mode()
def infer_one(messages, model, processor, device_auto: bool, max_new_tokens=128, temperature=0.0):
    # 文本模板
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 视觉输入
    image_inputs, video_inputs = process_vision_info(messages)

    # 编码
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # device_map=auto 时，不要手动 .to("cuda")
    if not device_auto:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 生成
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    # 裁掉 prompt
    trimmed = gen[0, inputs["input_ids"].shape[1]:]
    out_text = processor.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return out_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="基础模型，如 Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", required=True, help="LLaMA-Factory 输出的 LoRA/QLoRA 目录（checkpoint-XXXX）")
    ap.add_argument("--test", required=True, help="测试集 .json / .jsonl")
    ap.add_argument("--out", required=True, help="输出预测 .jsonl")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--min_pixels", type=int, default=None)
    ap.add_argument("--max_pixels", type=int, default=None)
    ap.add_argument("--merge_lora", action="store_true", help="非 QLoRA 可尝试合并 LoRA")
    args = ap.parse_args()

    # 加载模型与处理器
    model, processor, device_auto = load_model_and_processor(
        base=args.base,
        adapter=args.adapter,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        merge_lora=args.merge_lora,
    )

    # 读数据
    data = load_json_or_jsonl(Path(args.test))
    print(f"[INFO] loaded {len(data)} samples")

    # 推理
    preds = []
    for i, sample in enumerate(data, start=1):
        try:
            messages = build_messages(sample)

            # 关键健壮性：至少要有一个 user 或 system
            if not messages or all(m.get("role") != "user" for m in messages):
                messages = [{"role": "user", "content": [{"type": "text", "text": ""}]}]

            # 另一处健壮性：若 user 文本出现多个 <image> 且 images 不够，已在构造函数里兜底
            out_text = infer_one(
                messages=messages,
                model=model,
                processor=processor,
                device_auto=device_auto,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            preds.append({"index": i - 1, "prediction": out_text})

        except Exception as e:
            print(f"[WARN] sample#{i} failed: {e}")
            preds.append({"index": i - 1, "error": str(e)})

        if i % 50 == 0:
            print(f"[INFO] processed {i}/{len(data)}")

    # 写结果
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] saved -> {outp} (total={len(preds)})")


if __name__ == "__main__":
    main()

# 例：单卡、4bit 量化基座 + LoRA 适配器推理
# python infer_qwen25vl_qlora.py \
#   --base Qwen/Qwen2.5-VL-7B-Instruct \
#   --adapter ./runs/roi_phase1/checkpoint-200 \
#   --test ./data/prepared/test.jsonl \
#   --out ./preds/test_preds.jsonl \
#   --device auto \
#   --load_in_4bit \
#   --max_new_tokens 64
