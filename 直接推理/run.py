#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, argparse, json
from pathlib import Path
from typing import List
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForCausalLM


def list_images(img_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
    files = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No images found in: {img_dir}")
    return files


def pick_dtype(device: str):
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_model_and_processor(model_path: str, device: str):
    # 处理器
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    # 模型（单卡/CPU 简单起见不做分片）
    dtype = pick_dtype(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        device_map=None,     # 不自动切分
    ).to(device)
    model.eval()
    # 生成时禁用缓存，避免梯度检查点等干扰（推理安全）
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return model, processor


def build_prompt(processor, image: Image.Image, question: str) -> (str, list):
    """
    用 Qwen2.5-VL 的聊天模板构造输入：
    - 模板里只放占位 {"type":"image"} + 问题文本
    - 真实图像通过 processor(images=[...]) 传入
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt, [image]


@torch.inference_mode()
def infer_one(model, processor, image_path: Path, question: str, device: str,
              max_new_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0):
    image = Image.open(image_path).convert("RGB")
    prompt, images = build_prompt(processor, image, question)

    # 注意：不要在编码阶段截断，避免 <image> 标记被截断导致不一致
    inputs = processor(
        text=[prompt],
        images=[images],              # 形状为 [batch, list_of_images]
        return_tensors="pt",
        padding=False,
        truncation=False,
    )

    # 将张量移到同一设备
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    output_ids = model.generate(
        **inputs,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    # 只取生成的新 tokens（去掉提示部分）
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


def main():
    ap = argparse.ArgumentParser(description="Offline inference for Qwen2.5-VL (3B).")
    ap.add_argument("--model", required=True, help="本地模型路径，例如 /models/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--image_dir", required=True, help="包含图片的目录")
    ap.add_argument("--question", required=True, help="要对每张图片提问的问题")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    choices=["cuda", "cpu"], help="推理设备（默认自动选择）")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--out_jsonl", type=str, default=None, help="保存结果到 JSONL")
    ap.add_argument("--out_csv", type=str, default=None, help="保存结果到 CSV")
    args = ap.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"  # 强制离线

    model, processor = load_model_and_processor(args.model, args.device)

    img_dir = Path(args.image_dir)
    images = list_images(img_dir)

    results = []
    for i, img_path in enumerate(images, 1):
        try:
            answer = infer_one(
                model, processor, img_path, args.question, args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p
            )
            print(f"[{i:03d}/{len(images)}] {img_path.name} → {answer}")
            results.append({"image": str(img_path), "question": args.question, "answer": answer})
        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}", file=sys.stderr)
            results.append({"image": str(img_path), "question": args.question, "answer": None, "error": str(e)})

    # --- 保存 ---
    if args.out_jsonl:
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved JSONL → {args.out_jsonl}")

    if args.out_csv:
        try:
            import csv
            with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["image", "question", "answer", "error"])
                for r in results:
                    w.writerow([r.get("image",""), r.get("question",""), r.get("answer",""), r.get("error","")])
            print(f"Saved CSV   → {args.out_csv}")
        except Exception as e:
            print(f"[WARN] CSV save failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

# # 单卡 GPU（自动 bf16/fp16）
# python infer_qwen2_5_vl.py \
#   --model /mnt/models/Qwen/Qwen2.5-VL-3B-Instruct \
#   --image_dir ./demo_images \
#   --question "请用一句话描述这张图的主要内容。" \
#   --out_jsonl ./preds.jsonl \
#   --out_csv   ./preds.csv

# # CPU 推理（慢一些）
# python infer_qwen2_5_vl.py \
#   --model /mnt/models/Qwen/Qwen2.5-VL-3B-Instruct \
#   --image_dir ./demo_images \
#   --question "图片里有哪些物体？" \
#   --device cpu
