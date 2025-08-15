#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_qwenvl_sft.py
- 适配你给出的 SFT jsonl 数据格式（messages 里 user 同时含 text+image）
- 统一处理 pixel_values 为 5D: [B, N, C, H, W]
- 自动生成/对齐 image_grid_thw / image_grid_idx（唯一可信来源）
- 支持根据 max_length 自适应缩放图片，避免超长
- HuggingFace Trainer 可直接跑

依赖：
- transformers >= 4.40（或兼容 Qwen-VL Processor 的版本）
- datasets
- pillow
- torch
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, set_seed

# 你项目里的加载函数（保持不变）
from src.modeling.load_qwen_vl import (
    load_model_and_processor,
    tune_image_processor_from_cfg,
    apply_freeze_and_lora,
)


# ============== 工具 & Collator ==============

class VLDataCollator:
    """
    针对“单图/少量 tiles”的通用版 Collator（兼容多图）：
    - 仅在一个地方（_normalize_enc）规范化 4D→5D，并创建匹配的 image_grid_* 元数据；
    - 不在 __call__ 再二次改写元数据，避免不一致；
    - 对 batch 内不同大小的 tile 做零填充（右下角补零），并生成批级 image_grid_*。
    """

    def __init__(
        self,
        processor,
        model_config,
        max_length: int = 4096,
        add_generation_prompt: bool = False,
        label_pad_token_id: int = -100,
        sanitize_user_image_token: bool = True,
        auto_downscale_if_needed: bool = True,
        prefer_short_side: int = 896,
        downscale_floor: int = 448,
        downscale_step: int = 64,
    ):
        self.processor = processor
        self.model_config = model_config
        self.max_length = int(max_length)
        self.add_generation_prompt = bool(add_generation_prompt)
        self.label_pad_token_id = int(label_pad_token_id)
        self.sanitize_user_image_token = bool(sanitize_user_image_token)
        self.auto_downscale_if_needed = bool(auto_downscale_if_needed)
        self.prefer_short_side = int(prefer_short_side) if prefer_short_side else None
        self.downscale_floor = int(downscale_floor)
        self.downscale_step = int(downscale_step)

        tok = processor.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.pad_token_id = tok.pad_token_id

        # 从 image_processor 读取 patch_size（用于网格计算）
        self.patch_size = getattr(self.processor.image_processor, "patch_size", 14)

        self._user_image_literal = "<image>"
        self._user_image_safe = "〈image〉"

    @staticmethod
    def _resize_keep_short(img, new_short):
        from PIL import Image
        w, h = img.size
        if min(w, h) <= new_short:
            return img
        if w < h:
            return img.resize((new_short, int(h * (new_short / w))), Image.BICUBIC)
        else:
            return img.resize((int(w * (new_short / h)), new_short), Image.BICUBIC)

    def _normalize_enc(self, enc: Dict[str, Any]) -> Dict[str, Any]:
        """
        统一把 pixel_values 处理为 5D，并保证 image_grid_thw / image_grid_idx 存在且匹配。
        仅此一处负责这些元数据的最终形态，避免重复构造造成不一致。
        """
        import math
        pv = enc.get("pixel_values", None)
        if pv is None:
            raise ValueError("processor 未返回 pixel_values。")

        # 5D: [B, N, C, H, W] —— 直接信任
        if pv.dim() == 5:
            pass

        # 4D: 可能是 [N, C, H, W]（无 batch）或 [B, C, H, W]（无 tiles）
        elif pv.dim() == 4:
            # 优先判断是否无 batch（存在 image_grid_thw 且首维匹配）
            if "image_grid_thw" in enc and enc["image_grid_thw"].shape[0] == pv.shape[0]:
                pv = pv.unsqueeze(0)  # [1, N, C, H, W]
                enc["pixel_values"] = pv
                if "image_grid_idx" not in enc:
                    N = pv.shape[1]
                    enc["image_grid_idx"] = torch.arange(N, dtype=torch.long).unsqueeze(0)
            else:
                # 视作 [B, C, H, W]，常见 B=1
                B, C, H, W = pv.shape
                pv = pv.unsqueeze(1)  # [B, 1, C, H, W]
                enc["pixel_values"] = pv

                # 创建网格（使用 ceil + 偶数对齐，兼容 2x2 merge）
                hp = math.ceil(H / self.patch_size)
                wp = math.ceil(W / self.patch_size)
                if hp % 2:
                    hp += 1
                if wp % 2:
                    wp += 1
                enc["image_grid_thw"] = torch.tensor([[1, hp, wp]], dtype=torch.long)
                enc["image_grid_idx"] = torch.tensor([[0]], dtype=torch.long)

        # 3D: [C, H, W] —— 极少见，补到 [1,1,C,H,W]
        elif pv.dim() == 3:
            import math
            C, H, W = pv.shape
            pv = pv.unsqueeze(0).unsqueeze(0)  # [1,1,C,H,W]
            enc["pixel_values"] = pv
            hp = math.ceil(H / self.patch_size)
            wp = math.ceil(W / self.patch_size)
            if hp % 2:
                hp += 1
            if wp % 2:
                wp += 1
            enc["image_grid_thw"] = torch.tensor([[1, hp, wp]], dtype=torch.long)
            enc["image_grid_idx"] = torch.tensor([[0]], dtype=torch.long)
        else:
            raise ValueError(f"Unexpected pixel_values shape: {tuple(pv.shape)}")

        # 最后做一次一致性断言
        assert enc["pixel_values"].dim() == 5, "pixel_values 必须为 [B,N,C,H,W]"
        assert "image_grid_thw" in enc and "image_grid_idx" in enc, "缺少 image_grid_* 元数据"

        return enc

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        from PIL import Image
        import math

        valid_encoded: List[Dict[str, Any]] = []

        for ex in examples:
            # 1) 解析 messages，收集图像与文本（你的数据始终是 system+user(image)+assistant(text)）
            convo = []
            images = []
            has_image = False

            for m in ex.get("messages", []):
                parts = []
                for p in m.get("content", []):
                    t = p.get("type")
                    if t == "image" and p.get("image"):
                        try:
                            img = Image.open(p["image"]).convert("RGB")
                            parts.append({"type": "image"})   # 模板中放入图片占位
                            images.append(img)
                            has_image = True
                        except Exception:
                            # 跳过坏图
                            pass
                    elif t == "text":
                        txt = p.get("text") or ""
                        if self.sanitize_user_image_token:
                            txt = txt.replace(self._user_image_literal, self._user_image_safe)
                        parts.append({"type": "text", "text": txt})
                if parts:
                    convo.append({"role": m.get("role", "user"), "content": parts})

            if not has_image:
                # 本样本无有效图像，跳过
                continue

            # 2) 渲染成 prompt（保持单样本）
            prompt = self.processor.apply_chat_template(
                convo,
                add_generation_prompt=self.add_generation_prompt,
                tokenize=False,
            )

            # 3) 尝试编码；若超过 max_length，则按短边递减缩放直到满足
            try_short = self.prefer_short_side or 896
            cur_imgs = images

            while True:
                enc = self.processor(
                    text=prompt,
                    images=cur_imgs,
                    padding=False,
                    truncation=False,
                    return_tensors="pt",
                )
                enc = self._normalize_enc(enc)

                L = enc["input_ids"].shape[1]
                if L <= self.max_length:
                    break

                if not self.auto_downscale_if_needed:
                    enc = None
                    break

                # 估算新的短边
                scale = math.sqrt(self.max_length / L) * 0.95
                new_short = max(self.downscale_floor, int(try_short * scale))
                if new_short >= try_short:
                    new_short = max(self.downscale_floor, try_short - self.downscale_step)

                # 到下限仍不变就放弃该样本
                if new_short < self.downscale_floor or new_short == try_short:
                    enc = None
                    break

                cur_imgs = [self._resize_keep_short(im, new_short) for im in cur_imgs]
                try_short = new_short

            if enc is not None:
                valid_encoded.append(enc)

        if not valid_encoded:
            # 返回空 batch，会让 Trainer 丢弃该 batch
            return {}

        # 4) 文本右侧 pad + labels
        max_len = max(e["input_ids"].shape[1] for e in valid_encoded)
        batch_input_ids, batch_attention_mask = [], []
        for enc in valid_encoded:
            ids = enc["input_ids"][0]
            am = enc["attention_mask"][0]
            pad_n = max_len - ids.shape[0]
            if pad_n > 0:
                ids = torch.nn.functional.pad(ids, (0, pad_n), value=self.pad_token_id)
                am = torch.nn.functional.pad(am, (0, pad_n), value=0)
            batch_input_ids.append(ids)
            batch_attention_mask.append(am)

        input_ids = torch.stack(batch_input_ids)          # [B, L]
        attention_mask = torch.stack(batch_attention_mask)
        labels = input_ids.clone()
        labels[attention_mask == 0] = self.label_pad_token_id

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # 5) 视觉对齐打包：按 batch 最大 N/H/W 右下零填充
        B = len(valid_encoded)
        num_tiles = [e["pixel_values"].shape[1] for e in valid_encoded]
        N_max = max(num_tiles)
        H_max = max(e["pixel_values"].shape[-2] for e in valid_encoded)
        W_max = max(e["pixel_values"].shape[-1] for e in valid_encoded)
        C = valid_encoded[0]["pixel_values"].shape[2]
        dtype = valid_encoded[0]["pixel_values"].dtype

        pv_batch = torch.zeros((B, N_max, C, H_max, W_max), dtype=dtype)
        for i, enc in enumerate(valid_encoded):
            pv = enc["pixel_values"][0]  # [N, C, H, W]
            N, _, H, W = pv.shape
            pv_batch[i, :N, :, :H, :W] = pv
        batch["pixel_values"] = pv_batch  # [B, N_max, C, H, W]

        # 6) 构建批级网格与索引
        #   - 将每个样本的 image_grid_thw（[N,3]）按样本顺序拼接为 flat；
        #   - image_grid_idx 为 [B, N_max]，每行引用 flat 中对应 tile 的全局索引，其余填 0；
        flat_grids = []
        idx_rows = torch.zeros((B, N_max), dtype=torch.long)
        offset = 0
        for i, enc in enumerate(valid_encoded):
            g = enc["image_grid_thw"]           # [N, 3]
            n = g.shape[0]
            flat_grids.append(g)
            # 本样本的全局索引范围
            if "image_grid_idx" in enc:
                # enc 自带 [1, N]（局部 0..N-1），加 offset 变全局
                local = enc["image_grid_idx"][0] + offset
            else:
                local = torch.arange(n, dtype=torch.long) + offset
            idx_rows[i, :n] = local
            offset += n

        image_grid_thw = torch.cat(flat_grids, dim=0)  # [sum_N, 3]
        batch["image_grid_thw"] = image_grid_thw
        batch["image_grid_idx"] = idx_rows
        batch["num_image_tiles"] = torch.tensor(num_tiles, dtype=torch.long)

        return batch


# ============== 主训练入口 ==============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_config", required=True, help="YAML 配置文件路径")
    ap.add_argument("--resume", action="store_true", help="断点续训")
    args = ap.parse_args()

    import yaml
    with open(args.train_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))

    # 数据集路径（使用你生成的 SFT jsonl）
    train_jsonl = cfg["data"]["train_jsonl"]
    val_jsonl = cfg["data"]["val_jsonl"]

    train_ds = load_dataset("json", data_files=train_jsonl, split="train")
    val_ds = load_dataset("json", data_files=val_jsonl, split="train")

    # 模型 & 处理器
    model, processor = load_model_and_processor(cfg)
    tune_image_processor_from_cfg(processor, cfg)
    model = apply_freeze_and_lora(model, cfg)

    # Collator
    max_seq_len = int(cfg.get("max_seq_len", 4096))
    images_cfg = cfg.get("images", {})
    collator = VLDataCollator(
        processor=processor,
        model_config=model.config,
        max_length=max_seq_len,
        add_generation_prompt=bool(cfg.get("add_generation_prompt", False)),
        sanitize_user_image_token=bool(images_cfg.get("sanitize_user_image_token", True)),
        auto_downscale_if_needed=bool(images_cfg.get("auto_downscale_if_needed", True)),
        prefer_short_side=int(images_cfg.get("prefer_short_side", 896)),
        downscale_floor=int(images_cfg.get("downscale_floor", 448)),
        downscale_step=int(images_cfg.get("downscale_step", 64)),
    )

    # 训练参数
    tr_args = cfg.get("trainer", {})
    ta = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["optim"]["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["optim"]["grad_accum"],
        learning_rate=cfg["optim"].get("lr_lm", 1e-4),
        num_train_epochs=cfg["optim"].get("epochs", 2),
        warmup_steps=cfg["optim"].get("warmup_steps", 200),
        weight_decay=cfg["optim"].get("weight_decay", 0.0),
        bf16=(cfg.get("precision", "bf16").lower() == "bf16"),
        fp16=(cfg.get("precision", "bf16").lower() == "fp16"),
        gradient_checkpointing=cfg["optim"].get("grad_checkpointing", False),
        evaluation_strategy=tr_args.get("evaluation_strategy", "steps"),  # ✅ 正确字段名
        eval_steps=tr_args.get("eval_steps", 200),
        save_strategy=tr_args.get("save_strategy", "steps"),
        save_steps=tr_args.get("save_steps", 200),
        save_total_limit=tr_args.get("save_total_limit", 3),
        logging_steps=tr_args.get("logging_steps", 50),
        load_best_model_at_end=tr_args.get("load_best_model_at_end", True),
        metric_for_best_model=tr_args.get("metric_for_best_model", "eval_loss"),
        greater_is_better=tr_args.get("greater_is_better", False),
        save_safetensors=tr_args.get("save_safetensors", True),
        report_to=tr_args.get("report_to", ["none"]),
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 2)),
        remove_unused_columns=False,  # 视觉任务一定要 False
        ddp_find_unused_parameters=False if int(os.environ.get("WORLD_SIZE", "1")) > 1 else None,
    )

    trainer = Trainer(
        model=model,
        args=ta,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train(resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()
