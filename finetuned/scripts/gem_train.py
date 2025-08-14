#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, argparse
from pathlib import Path
import torch
from datasets import load_dataset, Features, Sequence, Value
from transformers import TrainingArguments, Trainer, set_seed

from src.modeling.load_qwen_vl import *

class VLDataCollator:
    """
    最终修正版：
    - 文本填充：必须使用右填充 (right-padding)，并确保 tokenizer 有效的 pad_token_id。
    - grid_thw：整个 batch 合并成一个扁平的 torch.LongTensor[TotalImages, 3]。
    - image_grid_idx：新增 LongTensor[B, N_max]，指示每个样本的 grid 在 grid_thw 中的行索引。
    - h/w：基于像素尺寸计算，向上补齐到 spatial_merge_size 的倍数，且 >= m。
    - pixel_values：5D Tensor [B, N_max, C, H_max, W_max] + pixel_values_mask。
    """
    def __init__(self, processor, model_config, max_length=4096,
                 add_generation_prompt=False, label_pad_token_id=-100,
                 sanitize_user_image_token=True,
                 auto_downscale_if_needed=True, prefer_short_side=None,
                 downscale_floor=448, downscale_step=64):
        self.processor = processor
        self.cfg = model_config
        self.max_length = int(max_length)
        self.add_generation_prompt = bool(add_generation_prompt)
        self.label_pad_token_id = int(label_pad_token_id)
        self.sanitize_user_image_token = bool(sanitize_user_image_token)

        self.auto_downscale_if_needed = bool(auto_downscale_if_needed)
        self.prefer_short_side = int(prefer_short_side) if prefer_short_side else None
        self.downscale_floor = int(downscale_floor)
        self.downscale_step  = int(downscale_step)

        tok = processor.tokenizer
        # 关键修正 1: 确保 pad_token 已设置，否则会导致 pad_id 为 None 或无效值
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.pad_token_id = tok.pad_token_id

        self.image_token_id  = getattr(model_config, "image_token_id", tok.convert_tokens_to_ids("<|image_pad|>"))
        self.vision_start_id = getattr(model_config, "vision_start_token_id", tok.convert_tokens_to_ids("<|vision_start|>"))
        self.vision_end_id   = getattr(model_config, "vision_end_token_id",   tok.convert_tokens_to_ids("<|vision_end|>"))

        vc = getattr(model_config, "vision_config", None)
        self.patch_size = int(getattr(vc, "patch_size", 14)) if vc is not None else 14
        self.spatial_merge_size = int(getattr(vc, "spatial_merge_size", 2)) if vc is not None else 2

        self._user_image_literal = "<image>"
        self._user_image_safe = "〈image〉"

    @staticmethod
    def _resize_keep_short(img, new_short):
        from PIL import Image
        w, h = img.size
        if min(w, h) <= new_short: return img
        if w < h:
            return img.resize((new_short, int(h * (new_short / w))), Image.BICUBIC)
        else:
            return img.resize((int(w * (new_short / h)), new_short), Image.BICUBIC)

    def _find_blocks(self, ids):
        starts, ends = [], []
        for i, t in enumerate(ids):
            if t == self.vision_start_id: starts.append(i)
            elif t == self.vision_end_id: ends.append(i)
        if not starts and not ends: return [], 0
        if len(starts) != len(ends): return [], -1
        blocks = []
        for s, e in zip(starts, ends):
            if s > e: return [], -1
            blocks.append((s, e))
        return blocks, 0 # a placeholder for total_img count

    def _extract_sizes_from_pv(self, pv):
        sizes = []
        if torch.is_tensor(pv):
            if pv.dim() == 4: sizes = [(int(pv.shape[2]), int(pv.shape[3]))] * int(pv.shape[0])
            elif pv.dim() == 3: sizes = [(int(pv.shape[1]), int(pv.shape[2]))]
            elif pv.dim() == 2: sizes = [(int(pv.shape[0]), int(pv.shape[1]))]
            else: raise ValueError(f"Unexpected pixel_values dims: {pv.dim()}")
        elif isinstance(pv, (list, tuple)):
            for t in pv:
                if not torch.is_tensor(t): raise ValueError("pv list elements must be tensors")
                if t.dim() == 4:
                    for u in t: sizes.append((int(u.shape[-2]), int(u.shape[-1])))
                else: sizes.append((int(t.shape[-2]), int(t.shape[-1])))
        else: raise ValueError("pixel_values must be Tensor or list of Tensors")
        return sizes

    def _sizes_to_grid_tensor(self, sizes):
        import math
        p = max(1, self.patch_size)
        m = max(1, self.spatial_merge_size)
        def up_to(x, k):
            x = max(k, int(x))
            return ((x + k - 1) // k) * k
        grid = []
        for (H, W) in sizes:
            H = max(1, int(H)); W = max(1, int(W))
            h = up_to(math.ceil(H / float(p)), m)
            w = up_to(math.ceil(W / float(p)), m)
            grid.append((1, h, w))
        return torch.as_tensor(grid, dtype=torch.long) if grid else torch.zeros((0,3), dtype=torch.long)

    def __call__(self, examples):
        from PIL import Image
        import torch, math

        # 1) 模板 + PIL
        chats_for_template, images_batch = [], []
        for ex in examples:
            mm_tpl, imgs = [], []
            for m in ex["messages"]:
                parts_tpl = []
                for p in m["content"]:
                    if p["type"] == "image" and p.get("image"):
                        parts_tpl.append({"type": "image"})
                        imgs.append(Image.open(p["image"]).convert("RGB"))
                    else:
                        txt = (p.get("text") or "")
                        if self.sanitize_user_image_token and self._user_image_literal in txt:
                            txt = txt.replace(self._user_image_literal, self._user_image_safe)
                        parts_tpl.append({"type": "text", "text": txt})
                mm_tpl.append({"role": m["role"], "content": parts_tpl})
            chats_for_template.append(mm_tpl)
            images_batch.append(imgs)

        prompts = self.processor.apply_chat_template(
            chats_for_template, add_generation_prompt=self.add_generation_prompt, tokenize=False
        )

        # 2) 逐样本 encode
        encoded = []
        for prompt, imgs in zip(prompts, images_batch):
            try_short = self.prefer_short_side or 896
            cur_imgs = imgs
            while True:
                enc = self.processor(text=prompt, images=cur_imgs, padding=False, truncation=False, return_tensors="pt")
                ids = enc["input_ids"][0]
                blocks, _ = self._find_blocks(ids.tolist())
                L = ids.shape[-1]
                if blocks:
                    min_keep, max_keep = min(s for s, _ in blocks), max(e for _, e in blocks) + 1
                    need = max_keep - min_keep
                else: need = 0

                if need <= self.max_length:
                    if L > self.max_length:
                         # 截断时，优先保留视觉部分和其后的文本
                        if blocks:
                            left = max(0, max_keep - self.max_length)
                        else: # 如果没有图像，从左边截断
                            left = L - self.max_length
                        enc["input_ids"] = ids[left:left+self.max_length].unsqueeze(0)
                        enc["attention_mask"] = enc["attention_mask"][0][left:left+self.max_length].unsqueeze(0)
                    break
                if not self.auto_downscale_if_needed: raise ValueError(f"Visual span ({need}) > max_length.")
                scale = math.sqrt(self.max_length / max(need,1)) * 0.95
                new_short = max(self.downscale_floor, int(try_short * scale))
                if new_short >= try_short: new_short = max(self.downscale_floor, try_short - self.downscale_step)
                if new_short < self.downscale_floor or new_short == try_short: raise ValueError("Downscaling failed.")
                cur_imgs = [self._resize_keep_short(im, new_short) for im in cur_imgs]
                try_short = new_short
            encoded.append(enc)

        # 3) 文本 pad + labels
        max_len = max(e["input_ids"].shape[-1] for e in encoded)
        ids_list, am_list = [], []
        for e in encoded:
            ids, am = e["input_ids"][0], e["attention_mask"][0]
            pad_n = max_len - ids.shape[-1]
            if pad_n > 0:
                # 关键修正 2: 必须使用右-padding (right-padding)
                ids = torch.nn.functional.pad(ids, (0, pad_n), value=self.pad_token_id)
                am = torch.nn.functional.pad(am, (0, pad_n), value=0)
            ids_list.append(ids); am_list.append(am)
        input_ids = torch.stack(ids_list, dim=0)
        attention_mask = torch.stack(am_list, dim=0)
        labels = input_ids.clone(); labels[attention_mask == 0] = self.label_pad_token_id
        batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        # 4) 视觉部分打包
        def _ensure_chw(t):
            if t.dim() == 2: t = t.unsqueeze(0)
            if t.shape[0] == 1: t = t.repeat(3,1,1)
            return t
        def _to_4d_per_sample(pv):
            if torch.is_tensor(pv):
                if pv.dim() == 4: return _ensure_chw(pv) if pv.shape[1] == 1 else pv
                elif pv.dim() in (2,3): return _ensure_chw(pv).unsqueeze(0)
            imgs = [_ensure_chw(t) for t in pv]
            sizes = [im.shape[-2:] for im in imgs]
            max_h, max_w = (max(s[0] for s in sizes), max(s[1] for s in sizes)) if sizes else (0,0)
            padded = [torch.nn.functional.pad(im, (0, max_w-im.shape[-1], 0, max_h-im.shape[-2])) for im in imgs]
            return torch.stack(padded, dim=0) if padded else torch.zeros((0,3,0,0))

        per_pv, per_grid = [], []
        for e in encoded:
            pv = _to_4d_per_sample(e["pixel_values"])
            sizes = self._extract_sizes_from_pv(pv)
            grid = self._sizes_to_grid_tensor(sizes)
            per_pv.append(pv); per_grid.append(grid)

        # 合并 pixel_values 到 5D
        N_max = max(pv.shape[0] for pv in per_pv) if per_pv else 0
        if N_max > 0:
            H_max = max(pv.shape[2] for pv in per_pv)
            W_max = max(pv.shape[3] for pv in per_pv)
            C = per_pv[0].shape[1]
            pv_batch = torch.zeros((len(examples), N_max, C, H_max, W_max), dtype=per_pv[0].dtype)
            pv_mask = torch.zeros((len(examples), N_max), dtype=torch.bool)
            for i, pv in enumerate(per_pv):
                N, _, H, W = pv.shape
                if N > 0:
                    pv_batch[i, :N, :, :H, :W] = pv
                    pv_mask[i, :N] = True
        else: # 处理没有图像的 batch
            pv_batch = torch.zeros((len(examples), 0, 3, 0, 0))
            pv_mask = torch.zeros((len(examples), 0), dtype=torch.bool)

        batch["pixel_values"] = pv_batch
        batch["pixel_values_mask"] = pv_mask

        # 合并 grid_thw 为扁平 Tensor，并创建 grid_idx
        grid_idx = torch.zeros((len(examples), N_max), dtype=torch.long)
        flat_grids = []
        current_idx = 0
        for i, grid in enumerate(per_grid):
            N = grid.shape[0]
            if N > 0:
                indices = torch.arange(current_idx, current_idx + N)
                grid_idx[i, :N] = indices
                flat_grids.append(grid)
                current_idx += N
        batch["grid_thw"] = torch.cat(flat_grids, dim=0) if flat_grids else torch.zeros((0,3), dtype=torch.long)
        batch["image_grid_idx"] = grid_idx
        batch["image_grid_thw"] = batch["grid_thw"] # 兼容别名

        if os.environ.get("DEBUG_VL", "0") == "1":
            print(f"--- VLDataCollator Batch ---")
            for k, v in batch.items():
                if torch.is_tensor(v):
                    print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
            print("--------------------------")

        return batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_config", required=True)
    ap.add_argument("--resume", action="store_true", help="断点续训")
    args = ap.parse_args()

    import yaml
    with open(args.train_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))

    train_ds = load_dataset("json", data_files=cfg["data"]["train_jsonl"], split="train")
    val_ds   = load_dataset("json", data_files=cfg["data"]["val_jsonl"],   split="train")

    model, processor = load_model_and_processor(cfg)
    tune_image_processor_from_cfg(processor, cfg)
    model = apply_freeze_and_lora(model, cfg)

    max_seq_len = int(cfg.get("max_seq_len", 4096))
    images_cfg = cfg.get("images", {})
    collator = VLDataCollator(
        processor=processor,
        model_config=model.config,
        max_length=max_seq_len,
        prefer_short_side=int(images_cfg.get("prefer_short_side", 896)),
        auto_downscale_if_needed=bool(images_cfg.get("auto_downscale_if_needed", True)),
        downscale_floor=int(images_cfg.get("downscale_floor", 448)),
        downscale_step=int(images_cfg.get("downscale_step", 64)),
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
        eval_strategy=tr_args.get("evaluation_strategy","steps"),
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
        remove_unused_columns=False,
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
