#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# 固定使用官方模型类（多模态）
from transformers import Qwen2_5_VLForConditionalGeneration

# PEFT / LoRA
from peft import PeftModel


def load_json_or_jsonl(path: Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        elif isinstance(obj, dict):
            return [obj]
        else:
            raise ValueError("JSON 格式需为数组或对象。")
    else:
        # jsonl
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    data.append(json.loads(ln))
        return data


def save_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def to_qwen_messages_from_messages(sample: dict):
    """
    兼容你原始的数据：{"messages":[ ... ]}
    仅用于推理：去掉 assistant，只保留 system + user（含 text / image）
    """
    msgs = sample.get("messages", [])
    # 过滤掉 assistant，避免把答案喂给模型
    filtered = []
    for m in msgs:
        if m.get("role") == "assistant":
            break
        filtered.append(m)
    return filtered


def to_qwen_messages_from_sharegpt(sample: dict):
    """
    兼容 LLaMA-Factory sharegpt+images:
    {
      "conversations":[{"from":"system","value":...},{"from":"human","value":"...<image>..."},{"from":"gpt","value": "..."}],
      "images": ["/path/img1.png", "/path/img2.png", ...]
    }
    转为 Qwen 的 messages（user 内部把 <image> 替换为 {"type":"image",...}，其余是 text）
    """
    conv = sample.get("conversations", [])
    imgs = list(sample.get("images", []))
    messages = []

    def add_user_with_images(text: str, images: list):
        # 把 text 按 <image> 切开，交替插入图片
        parts = text.split("<image>")
        content = []
        for i, seg in enumerate(parts):
            seg = seg.strip()
            if seg:
                content.append({"type": "text", "text": seg})
            if i < len(parts) - 1:
                if images:
                    imgp = images.pop(0)
                    content.append({"type": "image", "image": imgp})
                else:
                    # 若占位多于图片，留一个“占位缺图”的提示文本，避免空值
                    content.append({"type": "text", "text": "[[缺少对应图片]]"})
        if not content:
            # 没有文字也没有 <image>，至少放一个空文本，避免 processor 报错
            content = [{"type": "text", "text": ""}]
        messages.append({"role": "user", "content": content})

    for turn in conv:
        role = turn.get("from")
        val  = turn.get("value", "")
        if role == "system":
            messages.append({"role": "system", "content": [{"type":"text","text":val}]})
        elif role in ("human", "user"):
            add_user_with_images(val, imgs)
        elif role in ("gpt", "assistant"):
            # 推理不带答案
            break
        else:
            # 其他角色忽略
            pass

    # 若 sharegpt 中 images 仍有剩余（文本未显式 <image>），把剩余图片追加到最后一个 user
    if imgs:
        # 找最后一个 user
        for m in reversed(messages):
            if m["role"] == "user":
                for imgp in imgs:
                    m["content"].append({"type":"image", "image": imgp})
                imgs = []
                break
    return messages


def build_messages(sample: dict):
    if "messages" in sample:
        return to_qwen_messages_from_messages(sample)
    if "conversations" in sample:
        return to_qwen_messages_from_sharegpt(sample)
    raise ValueError("无法识别的数据格式：既无 'messages' 也无 'conversations'。")


def load_model_and_processor(
    base:str,
    adapter:str,
    device:str="cuda",
    torch_dtype:str="auto",
    flash_attn2:bool=False,
    min_pixels:int=None,
    max_pixels:int=None,
    merge_lora:bool=False,
):
    """
    两阶段加载，更稳妥：
    1) 先加载 BASE（Qwen2.5-VL）——可选 attn=fa2 / 设备映射
    2) 再用 PEFT 把 LoRA 适配器挂上；若需要可 merge_and_unload()
    """
    base_kwargs = dict(
        torch_dtype=torch_dtype if torch_dtype != "auto" else "auto",
    )
    if flash_attn2:
        base_kwargs.update(dict(attn_implementation="flash_attention_2"))

    # 设备策略
    if device == "auto":
        base_kwargs.update(dict(device_map="auto"))
    else:
        # 单设备（cuda/cpu），不使用 device_map
        base_kwargs.update(dict(device_map=None))

    print(f"[INFO] Loading BASE model: {base}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base, **base_kwargs
    )

    # 单设备时手动 .to(device)
    if device != "auto":
        model = model.to(device)

    print(f"[INFO] Attaching LoRA adapter from: {adapter}")
    model = PeftModel.from_pretrained(model, adapter)

    if merge_lora:
        # 注意：QLoRA（4bit）不可 merge；普通 LoRA 可 merge
        try:
            print("[INFO] Merging LoRA weights into base (merge_and_unload)...")
            model = model.merge_and_unload()
        except Exception as e:
            print(f"[WARN] merge_and_unload 失败，保留为 PEFT 形式继续推理。原因：{e}")

    model.eval()

    # Processor
    proc_kwargs = {}
    if min_pixels is not None: proc_kwargs["min_pixels"] = int(min_pixels)
    if max_pixels is not None: proc_kwargs["max_pixels"] = int(max_pixels)

    print(f"[INFO] Loading processor from BASE: {base} with params: {proc_kwargs or 'default'}")
    processor = AutoProcessor.from_pretrained(base, **proc_kwargs)
    return model, processor


@torch.inference_mode()
def infer_one(messages, model, processor, device="cuda", device_auto=False, max_new_tokens=128, temperature=0.0):
    """
    messages: Qwen 官方消息格式（包含 content: list[{"type":"text"/"image"}]）
    device_auto=True 表示 model 是 device_map="auto" 加载的，这时**不要**强制把 inputs .to(cuda)
    """
    # 准备输入（官方推荐流程）
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    if not device_auto:
        # 单设备：把输入放到 model.device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 生成
    gen_out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    # 裁掉 prompt 部分
    trimmed = gen_out[0, inputs["input_ids"].shape[1]:]
    out_text = processor.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return out_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="基础模型名或路径，如 Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", required=True, help="LLaMA-Factory 训练产出的 LoRA 目录（checkpoint-xxx）")
    ap.add_argument("--test", required=True, help="测试集 .jsonl（每行一个样本）或 .json（数组）")
    ap.add_argument("--out", required=True, help="输出预测 .jsonl")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu","auto"], help="推理设备策略")
    ap.add_argument("--flash_attn2", action="store_true", help="启用 FlashAttention2")
    ap.add_argument("--merge_lora", action="store_true", help="尝试将 LoRA 权重合并进基座（非 QLoRA）")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--min_pixels", type=int, default=None, help="可选：视觉 token 下界（像素）")
    ap.add_argument("--max_pixels", type=int, default=None, help="可选：视觉 token 上界（像素）")
    args = ap.parse_args()

    # 加载模型与处理器
    model, processor = load_model_and_processor(
        base=args.base,
        adapter=args.adapter,
        device=args.device,
        torch_dtype="auto",
        flash_attn2=args.flash_attn2,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        merge_lora=args.merge_lora,
    )
    device_auto = (args.device == "auto")

    # 读取测试集
    test_path = Path(args.test)
    data = load_json_or_jsonl(test_path)
    print(f"[INFO] Loaded {len(data)} test samples from {test_path}")

    preds = []
    for i, sample in enumerate(data, start=1):
        try:
            messages = build_messages(sample)
            # 基础健壮性：至少需要一个 user 或 system，否则补一个空 user
            if not messages or all(m.get("role") != "user" for m in messages):
                messages = [{"role":"user","content":[{"type":"text","text":""}]}]

            out_text = infer_one(
                messages=messages,
                model=model,
                processor=processor,
                device="cuda" if not device_auto else "cpu",   # 占位，不会在 device_auto 用到
                device_auto=device_auto,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            preds.append({
                "index": i-1,
                "prediction": out_text
            })
        except Exception as e:
            preds.append({
                "index": i-1,
                "error": str(e)
            })
            print(f"[WARN] sample#{i} failed: {e}")

        if i % 50 == 0:
            print(f"[INFO] processed {i}/{len(data)}")

    out_path = Path(args.out)
    save_jsonl(out_path, preds)
    print(f"[OK] Saved predictions -> {out_path} (total={len(preds)})")


if __name__ == "__main__":
    main()


# # 纯推理（单卡）
# python infer_qwen25vl_lora.py \
#   --base Qwen/Qwen2.5-VL-7B-Instruct \
#   --adapter ./runs/roi_phase1/checkpoint-200  \
#   --test ./data/prepared/test.jsonl \
#   --out ./preds/test_preds.jsonl \
#   --device cuda \
#   --flash_attn2 \
#   --max_new_tokens 64

# # 如果想自动分配多设备（不手动 .to(cuda)）
# python infer_qwen25vl_lora.py \
#   --base Qwen/Qwen2.5-VL-7B-Instruct \
#   --adapter ./runs/roi_phase1/checkpoint-200  \
#   --test ./data/prepared/test.jsonl \
#   --out ./preds/test_preds.jsonl \
#   --device auto \
#   --flash_attn2
