#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, set_seed

from src.modeling.load_qwen_vl import *

class VLDataCollator:
    """
    最终修正版 v5:
    - 修复了 `IndexError: tuple index out of range` 错误。
    - 在计算 padding 长度时，对 1D 张量使用 `ids.shape[0]` 而不是 `ids.shape[1]`。
    - 保留了 v4 的所有正确逻辑：信任 processor、正确打包视觉部分、右填充。
    """
    def __init__(self, processor, model_config, max_length=4096,
                 add_generation_prompt=False, label_pad_token_id=-100,
                 sanitize_user_image_token=True,
                 auto_downscale_if_needed=True, prefer_short_side=None,
                 downscale_floor=448, downscale_step=64):
        self.processor = processor
        self.max_length = int(max_length)
        self.add_generation_prompt = bool(add_generation_prompt)
        self.label_pad_token_id = int(label_pad_token_id)
        self.sanitize_user_image_token = bool(sanitize_user_image_token)

        self.auto_downscale_if_needed = bool(auto_downscale_if_needed)
        self.prefer_short_side = int(prefer_short_side) if prefer_short_side else None
        self.downscale_floor = int(downscale_floor)
        self.downscale_step  = int(downscale_step)

        tok = processor.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.pad_token_id = tok.pad_token_id

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
        encoded_samples = []
        for prompt, imgs in zip(prompts, images_batch):
            try_short = self.prefer_short_side or 896
            cur_imgs = imgs
            while True:
                enc = self.processor(text=prompt, images=cur_imgs, padding=False, truncation=False, return_tensors="pt")
                L = enc["input_ids"].shape[1]
                if L <= self.max_length:
                    break
                if not self.auto_downscale_if_needed:
                    raise ValueError(f"Sequence length ({L}) > max_length ({self.max_length}).")
                scale = math.sqrt(self.max_length / L) * 0.95
                new_short = max(self.downscale_floor, int(try_short * scale))
                if new_short >= try_short:
                    new_short = max(self.downscale_floor, try_short - self.downscale_step)
                if new_short < self.downscale_floor or new_short == try_short:
                    raise ValueError(f"Downscaling failed to reduce sequence length below {self.max_length}.")
                cur_imgs = [self._resize_keep_short(im, new_short) for im in cur_imgs]
                try_short = new_short
            encoded_samples.append(enc)

        # 3) 文本右填充 + labels
        max_len = max(e["input_ids"].shape[1] for e in encoded_samples)
        batch_input_ids, batch_attention_mask = [], []
        for enc in encoded_samples:
            ids, am = enc["input_ids"][0], enc["attention_mask"][0]
            
            # 关键修正: 对 1D 张量使用 shape[0]
            pad_n = max_len - ids.shape[0]
            
            if pad_n > 0:
                ids = torch.nn.functional.pad(ids, (0, pad_n), value=self.pad_token_id)
                am  = torch.nn.functional.pad(am,  (0, pad_n), value=0)
            batch_input_ids.append(ids)
            batch_attention_mask.append(am)
        
        input_ids = torch.stack(batch_input_ids)
        attention_mask = torch.stack(batch_attention_mask)
        labels = input_ids.clone()
        labels[attention_mask == 0] = self.label_pad_token_id
        batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        # 4) 视觉部分打包
        per_pv = [enc.get('pixel_values') for enc in encoded_samples]
        per_grid_thw = [enc.get('image_grid_thw') for enc in encoded_samples]
        per_grid_idx = [enc.get('image_grid_idx') for enc in encoded_samples]

        image_counts = [
            pv.shape[1] 
            for pv in per_pv 
            if pv is not None and pv.dim() == 5 and pv.numel() > 0
        ]
        N_max = max(image_counts) if image_counts else 0

        if N_max > 0:
            H_max = max(pv.shape[3] for pv in per_pv if pv is not None and pv.numel() > 0)
            W_max = max(pv.shape[4] for pv in per_pv if pv is not None and pv.numel() > 0)
            C = next(pv.shape[2] for pv in per_pv if pv is not None and pv.numel() > 0)
            
            pv_batch = torch.zeros((len(examples), N_max, C, H_max, W_max), dtype=torch.bfloat16 if self.processor.image_processor.torch_dtype == torch.bfloat16 else torch.float16)
            pv_mask = torch.zeros((len(examples), N_max), dtype=torch.bool)
            
            for i, pv in enumerate(per_pv):
                if pv is not None and pv.numel() > 0:
                    _, N, _, H, W = pv.shape
                    pv_batch[i, :N, :, :H, :W] = pv[0]
                    pv_mask[i, :N] = True
        else:
            pv_batch = torch.zeros((len(examples), 0, 3, 0, 0), dtype=torch.float32)
            pv_mask = torch.zeros((len(examples), 0), dtype=torch.bool)

        batch["pixel_values"] = pv_batch
        batch["pixel_values_mask"] = pv_mask

        # 合并 grid_thw 和 grid_idx
        flat_grids, flat_indices = [], []
        current_idx_offset = 0
        
        for i, (grid, idx) in enumerate(zip(per_grid_thw, per_grid_idx)):
            if grid is not None and idx is not None and grid.numel() > 0:
                num_images = grid.shape[0]
                flat_grids.append(grid)
                adjusted_idx = idx[0] + current_idx_offset
                flat_indices.append(adjusted_idx)
                current_idx_offset += num_images

        if flat_grids:
            batch["grid_thw"] = torch.cat(flat_grids, dim=0)
            idx_batch = torch.zeros((len(examples), N_max), dtype=torch.long)
            for i, idx in enumerate(flat_indices):
                num_images = len(idx)
                idx_batch[i, :num_images] = idx
            batch["image_grid_idx"] = idx_batch
        else:
            batch["grid_thw"] = torch.zeros((0, 3), dtype=torch.long)
            batch["image_grid_idx"] = torch.zeros((len(examples), N_max), dtype=torch.long)

        batch["image_grid_thw"] = batch["grid_thw"]

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
