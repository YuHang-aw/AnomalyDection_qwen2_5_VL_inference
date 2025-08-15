#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, argparse
from pathlib import Path
import math
import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, set_seed

# 你现有的加载工具：保留
from src.modeling.load_qwen_vl import (
    load_model_and_processor,
    tune_image_processor_from_cfg,
    apply_freeze_and_lora,
)

# ========== 工具函数 ==========
def ceil_even(x: int) -> int:
    """向上取偶数，用于保证 2x2 spatial merge 不报错"""
    return x if x % 2 == 0 else x + 1

def find_user_only_messages(msgs):
    """
    从你的数据格式中抽取：只包含 system + user 的消息列表
    假设结构： [system?, user, assistant]
    """
    # 兼容多种写法：只要取到第一个 assistant 之前的内容即可
    out = []
    for m in msgs:
        out.append(m)
        if m.get("role") == "assistant":
            out.pop()     # 不要 assistant
            break
    return out

def collect_images_from_messages(msgs):
    """按 messages 中出现顺序收集所有图片路径（通常一张）"""
    paths = []
    for m in msgs:
        for c in m.get("content", []):
            if c.get("type") == "image" and c.get("image"):
                paths.append(c["image"])
    return paths


# ========== DataCollator ==========
class VLDataCollator:
    """
    v9 最小必要修复版：
    - 统一 pixel_values 到 [B, N, C, H, W]
    - 优先使用 processor 返回的 image_grid_*；缺省时兜底构造 (ceil & even)
    - 严格返回符合模型 forward 的纯 Tensor 字典
    - 仅对最后一段 assistant 文本计算 loss
    - 支持长文本自动下采样图像 (可选)
    """
    def __init__(
        self,
        processor,
        model_config,
        max_length=4096,
        add_generation_prompt=False,
        label_pad_token_id: int = -100,
        auto_downscale_if_needed=True,
        prefer_short_side=None,
        downscale_floor=448,
        downscale_step=64,
    ):
        self.processor = processor
        self.max_length = int(max_length)
        self.add_generation_prompt = bool(add_generation_prompt)
        self.label_pad_token_id = int(label_pad_token_id)

        self.auto_downscale_if_needed = bool(auto_downscale_if_needed)
        self.prefer_short_side = int(prefer_short_side) if prefer_short_side else None
        self.downscale_floor = int(downscale_floor)
        self.downscale_step  = int(downscale_step)

        tok = processor.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.pad_token_id = tok.pad_token_id

        # patch_size 用于兜底计算 grid
        self.patch_size = getattr(self.processor, "image_processor", getattr(self.processor, "image_processor", None))
        self.patch_size = getattr(self.processor.image_processor, "patch_size", 14)  # 默认给个常见值，避免取不到

    # --- 图像短边等比缩小 ---
    @staticmethod
    def _resize_keep_short(img, new_short):
        from PIL import Image
        w, h = img.size
        if min(w, h) <= new_short: return img
        if w < h:
            return img.resize((new_short, int(h * (new_short / w))), Image.BICUBIC)
        else:
            return img.resize((int(w * (new_short / h)), new_short), Image.BICUBIC)

    # --- 规范化返回，统一 [B, N, C, H, W] 并兜底 grid ---
    def _normalize_enc(self, enc):
        pv = enc["pixel_values"]

        if pv.dim() == 5:
            # [B, N, C, H, W]
            pass

        elif pv.dim() == 4 and "image_grid_thw" in enc and enc["image_grid_thw"].shape[0] == pv.shape[0]:
            # [N, C, H, W] （无 batch 维，有 tile）
            pv = pv.unsqueeze(0)  # -> [1, N, C, H, W]
            enc["pixel_values"] = pv
            if "image_grid_idx" not in enc:
                N = pv.shape[1]
                enc["image_grid_idx"] = torch.arange(N, dtype=torch.long).unsqueeze(0)

        elif pv.dim() == 4:
            # [B, C, H, W] （常见 B=1，无 tile）
            B, C, H, W = pv.shape
            pv = pv.unsqueeze(1)  # -> [B, 1, C, H, W]
            enc["pixel_values"] = pv
            if "image_grid_thw" not in enc:
                hp = ceil_even(math.ceil(H / self.patch_size))
                wp = ceil_even(math.ceil(W / self.patch_size))
                enc["image_grid_thw"] = torch.tensor([[1, hp, wp]], dtype=torch.long)
            if "image_grid_idx" not in enc:
                enc["image_grid_idx"] = torch.tensor([[0]], dtype=torch.long)

        elif pv.dim() == 3:
            # [C,H,W] 极少见
            C, H, W = pv.shape
            pv = pv.unsqueeze(0).unsqueeze(0)  # -> [1,1,C,H,W]
            enc["pixel_values"] = pv
            hp = ceil_even(math.ceil(H / self.patch_size))
            wp = ceil_even(math.ceil(W / self.patch_size))
            enc["image_grid_thw"] = torch.tensor([[1, hp, wp]], dtype=torch.long)
            enc["image_grid_idx"] = torch.tensor([[0]], dtype=torch.long)

        else:
            raise ValueError(f"Unexpected pixel_values shape: {pv.shape}")

        # 保证 dtype/类型正确
        if enc["pixel_values"].dtype not in (torch.float32, torch.bfloat16, torch.float16):
            enc["pixel_values"] = enc["pixel_values"].float()

        # 有些实现把 grid 放在其它键名上（比如 grid_thw），这里做一次归一化
        if "image_grid_thw" not in enc and "grid_thw" in enc:
            enc["image_grid_thw"] = enc.pop("grid_thw")

        return enc

    # --- 计算 labels：仅对最后一段 assistant 文本计损失 ---
    def _make_labels_last_assistant(self, prompt_full, prompt_user_only, images):
        # A: user-only
        enc_user = self.processor(
            text=prompt_user_only,
            images=images,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        len_user = enc_user["input_ids"].shape[-1]

        # B: full
        enc_full = self.processor(
            text=prompt_full,
            images=images,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        input_ids = enc_full["input_ids"]            # [1, L]
        attention_mask = enc_full["attention_mask"]  # [1, L]

        # mask：user 段及 padding 之外位置 = -100，仅 assistant 段计算损失
        labels = input_ids.clone()
        labels[:, :len_user] = self.label_pad_token_id
        labels[attention_mask == 0] = self.label_pad_token_id
        return enc_full, labels

    def __call__(self, examples):
        from PIL import Image

        valid = []

        for ex in examples:
            msgs = ex["messages"]

            # 1) 准备 prompt（全量 & 只有 user）
            # Qwen2.5-VL 常见调用：传入 [msgs]，取第 0 条结果字符串
            prompt_full = self.processor.apply_chat_template(
                [msgs], add_generation_prompt=False, tokenize=False
            )[0]

            msgs_user = find_user_only_messages(msgs)
            prompt_user = self.processor.apply_chat_template(
                [msgs_user], add_generation_prompt=False, tokenize=False
            )[0]

            # 2) 收集并加载图像（不做尺寸门槛过滤，交给 processor）
            img_paths = collect_images_from_messages(msgs)
            images = []
            for p in img_paths:
                try:
                    im = Image.open(p).convert("RGB")
                    images.append(im)
                except Exception:
                    pass

            if len(images) == 0:
                # 没图就跳过该样本
                continue

            # 3) 控制过长（先不缩放；必要时下方 while 中会自适应缩放）
            try_short = self.prefer_short_side or 1024
            cur_imgs = images

            while True:
                # 3.1) 得到 full 编码 + labels（只对 assistant 段）
                enc_full, labels = self._make_labels_last_assistant(
                    prompt_full, prompt_user, cur_imgs
                )

                # 3.2) 规范化视觉分支形状 & grid
                enc_full = self._normalize_enc(enc_full)

                L = enc_full["input_ids"].shape[-1]
                if L <= self.max_length:
                    break

                if not self.auto_downscale_if_needed:
                    enc_full = None
                    break

                # 简单按比例缩短
                scale = math.sqrt(self.max_length / L) * 0.95
                new_short = max(self.downscale_floor, int(try_short * scale))
                if new_short >= try_short:
                    new_short = max(self.downscale_floor, try_short - self.downscale_step)
                if new_short < self.downscale_floor or new_short == try_short:
                    enc_full = None
                    break

                cur_imgs = [self._resize_keep_short(im, new_short) for im in cur_imgs]
                try_short = new_short

            if enc_full is None:
                continue

            enc_full["labels"] = labels
            valid.append(enc_full)

        if not valid:
            # 明确失败，避免 Trainer 在空 batch 上走奇怪路径
            raise RuntimeError("DataCollator produced an empty batch: all samples invalid after processing.")

        # ===== 文本部分右填充 =====
        max_len = max(e["input_ids"].shape[-1] for e in valid)
        pad_id = self.pad_token_id

        input_ids, attention_mask, labels = [], [], []
        for e in valid:
            ids = e["input_ids"][0]
            am  = e["attention_mask"][0]
            lb  = e["labels"][0]
            pad_n = max_len - ids.shape[0]
            if pad_n > 0:
                ids = torch.nn.functional.pad(ids, (0, pad_n), value=pad_id)
                am  = torch.nn.functional.pad(am,  (0, pad_n), value=0)
                lb  = torch.nn.functional.pad(lb,  (0, pad_n), value=self.label_pad_token_id)
            input_ids.append(ids)
            attention_mask.append(am)
            labels.append(lb)

        batch = {
            "input_ids": torch.stack(input_ids).to(torch.long),
            "attention_mask": torch.stack(attention_mask).to(torch.long),
            "labels": torch.stack(labels).to(torch.long),
        }

        # ===== 视觉部分打包 =====
        B = len(valid)
        num_tiles = [e["pixel_values"].shape[1] if e["pixel_values"].dim() == 5 else 1 for e in valid]
        N_max = max(num_tiles)
        C = valid[0]["pixel_values"].shape[-3]
        H_max = max(e["pixel_values"].shape[-2] for e in valid)
        W_max = max(e["pixel_values"].shape[-1] for e in valid)

        dtype = valid[0]["pixel_values"].dtype
        pv_batch = torch.zeros((B, N_max, C, H_max, W_max), dtype=dtype)
        idx_batch = torch.zeros((B, N_max), dtype=torch.long)

        grid_list = []
        tile_offset = 0
        for i, e in enumerate(valid):
            pv = e["pixel_values"][0]  # [N,C,H,W]
            N, _, H, W = pv.shape
            pv_batch[i, :N, :, :H, :W] = pv

            # image_grid_idx: [B,N]
            if "image_grid_idx" in e:
                idx = e["image_grid_idx"][0]  # [N]
            else:
                idx = torch.arange(N, dtype=torch.long)
            idx_batch[i, :N] = idx + tile_offset

            # image_grid_thw: [sum_tiles, 3]
            if "image_grid_thw" in e:
                grid = e["image_grid_thw"]  # [N,3] 或 [1,3]（单 tile）
                if grid.dim() == 1:
                    grid = grid.unsqueeze(0)
            else:
                # 兜底：从 pv 的 H,W 推断
                hp = ceil_even(math.ceil(H / self.patch_size))
                wp = ceil_even(math.ceil(W / self.patch_size))
                grid = torch.tensor([[1, hp, wp]], dtype=torch.long)

            grid_list.append(grid)
            tile_offset += N

        image_grid_thw = torch.cat(grid_list, dim=0)  # [sum_N, 3]

        batch["pixel_values"] = pv_batch
        batch["image_grid_idx"] = idx_batch
        batch["image_grid_thw"] = image_grid_thw

        # ===== 强校验（调通后可注释）=====
        need = ["input_ids","attention_mask","pixel_values","image_grid_thw","image_grid_idx","labels"]
        for k in list(batch.keys()):
            if k not in need: batch.pop(k)

        assert isinstance(batch["pixel_values"], torch.Tensor) and batch["pixel_values"].dim() == 5
        assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["labels"].shape
        assert batch["image_grid_idx"].shape[0] == batch["input_ids"].shape[0]  # B
        assert batch["image_grid_thw"].dim() == 2 and batch["image_grid_thw"].shape[1] == 3

        # 2x2 merge 可整除性（可选）
        t, h, w = image_grid_thw[:,0], image_grid_thw[:,1], image_grid_thw[:,2]
        seq_vis = int((t*h*w).sum().item())
        if seq_vis % 4 != 0:
            # 不直接 raise；打印提示，模型里也可能处理 padding
            print(f"[Warn] visual tokens {seq_vis} not divisible by 4 (2x2 merge).")

        # dtypes
        batch["pixel_values"] = batch["pixel_values"].float()
        for k in ["input_ids","attention_mask","labels","image_grid_thw","image_grid_idx"]:
            batch[k] = batch[k].to(torch.long)

        return batch


# ========== 训练入口 ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_config", required=True)
    ap.add_argument("--resume", action="store_true", help="断点续训")
    args = ap.parse_args()

    import yaml
    with open(args.train_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))

    # 数据集（你的 jsonl 正是 messages 列表）
    train_ds = load_dataset("json", data_files=cfg["data"]["train_jsonl"], split="train")
    val_ds   = load_dataset("json", data_files=cfg["data"]["val_jsonl"],   split="train")

    # 模型+处理器（沿用你的封装）
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
        prefer_short_side=int(cfg.get("image_short_side", 1024)),
        auto_downscale_if_needed=True,
        downscale_floor=448,
        downscale_step=64,
    )

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
        bf16=(cfg.get("precision","bf16").lower()=="bf16"),
        fp16=(cfg.get("precision","bf16").lower()=="fp16"),
        gradient_checkpointing=cfg["optim"].get("grad_checkpointing", False),
        evaluation_strategy=tr_args.get("evaluation_strategy","steps"),
        eval_steps=tr_args.get("eval_steps",200),
        save_strategy=tr_args.get("save_strategy","steps"),
        save_steps=tr_args.get("save_steps",200),
        save_total_limit=tr_args.get("save_total_limit",3),
        logging_steps=tr_args.get("logging_steps",50),
        load_best_model_at_end=tr_args.get("load_best_model_at_end", True),
        metric_for_best_model=tr_args.get("metric_for_best_model","eval_loss"),
        greater_is_better=tr_args.get("greater_is_better", False),
        save_safetensors=tr_args.get("save_safetensors", True),
        report_to=["none"],
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 2)),
        remove_unused_columns=False,  # 很关键：别让 Trainer 丢掉 messages 字段
        ddp_find_unused_parameters=False if int(os.environ.get("WORLD_SIZE","1")) > 1 else None,
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
