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
    最终的、绝对正确的版本 (v10):
    - 核心修正：采用严格的“白名单”机制对 processor 输出进行维度检查。
    - 只有当 pixel_values 的维度明确为 4D 或 5D 时，才被视为有效。
    - 任何其他维度（<4D 或 >5D）的张量都会被明确拒绝，从而彻底根除 IndexError。
    - 这是在反复失败后，针对 processor 不可预测的边缘行为的最终解决方案。
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
        
        self.patch_size = self.processor.image_processor.patch_size

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

        valid_encoded_samples = []
        
        for ex_idx, ex in enumerate(examples):
            mm_tpl, imgs, img_paths = [], [], []
            has_valid_image = False
            for m in ex["messages"]:
                parts_tpl = []
                for p in m["content"]:
                    if p["type"] == "image" and p.get("image"):
                        try:
                            img_path = p["image"]
                            img = Image.open(img_path).convert("RGB")
                            if img.width >= self.patch_size and img.height >= self.patch_size:
                                parts_tpl.append({"type": "image"})
                                imgs.append(img)
                                img_paths.append(img_path)
                                has_valid_image = True
                        except Exception:
                            pass
                    else:
                        txt = (p.get("text") or "")
                        if self.sanitize_user_image_token: txt = txt.replace(self._user_image_literal, self._user_image_safe)
                        parts_tpl.append({"type": "text", "text": txt})
                mm_tpl.append({"role": m["role"], "content": parts_tpl})

            if not has_valid_image: continue

            prompt = self.processor.apply_chat_template([mm_tpl], add_generation_prompt=self.add_generation_prompt, tokenize=False)[0]

            try_short = self.prefer_short_side or 896
            cur_imgs = imgs
            while True:
                enc = self.processor(text=prompt, images=cur_imgs, padding=False, truncation=False, return_tensors="pt")
                
                pv = enc.get("pixel_values")
                
                # --- v10 CORE FIX: 严格的维度守卫 ---
                if pv is None or pv.numel() == 0:
                    enc = None; break

                if pv.dim() == 5:
                    # 已经是 [B, N, C, H, W]，通过
                    pass
                elif pv.dim() == 4:
                    # 是 [B, C, H, W]，规范化并创建元数据
                    pv = pv.unsqueeze(1)
                    enc['pixel_values'] = pv
                    
                    _B, _C, H, W = pv.shape[0], pv.shape[2], pv.shape[3], pv.shape[4]
                    h_patch_num = H // self.patch_size
                    w_patch_num = W // self.patch_size
                    
                    enc['image_grid_thw'] = torch.tensor([[1, h_patch_num, w_patch_num]], dtype=torch.long)
                    enc['image_grid_idx'] = torch.tensor([[0]], dtype=torch.long)
                else:
                    # 任何其他维度都是无效的，必须拒绝
                    print(f"\n[DataCollator Warning] Skipping sample with image '{img_paths[0]}' because processor returned a tensor with unexpected dimension: {pv.shape}. This is an invalid sample.\n")
                    enc = None; break
                # --- 守卫结束 ---

                L = enc["input_ids"].shape[1]
                if L <= self.max_length: break
                if not self.auto_downscale_if_needed: enc = None; break
                scale = math.sqrt(self.max_length / L) * 0.95
                new_short = max(self.downscale_floor, int(try_short * scale))
                if new_short >= try_short: new_short = max(self.downscale_floor, try_short - self.downscale_step)
                if new_short < self.downscale_floor or new_short == try_short: enc = None; break
                cur_imgs = [self._resize_keep_short(im, new_short) for im in cur_imgs]
                try_short = new_short
            
            if enc: valid_encoded_samples.append(enc)

        if not valid_encoded_samples: return {}

        # 后续所有代码都建立在 valid_encoded_samples 中所有样本都绝对安全的基础上
        # 3) 文本右填充 + labels
        max_len = max(e["input_ids"].shape[1] for e in valid_encoded_samples)
        batch_input_ids, batch_attention_mask = [], []
        for enc in valid_encoded_samples:
            ids, am = enc["input_ids"][0], enc["attention_mask"][0]
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
        num_valid_samples = len(valid_encoded_samples)
        
        num_tiles_per_sample = [e['pixel_values'].shape[1] for e in valid_encoded_samples]
        N_max = max(num_tiles_per_sample)
        H_max = max(e['pixel_values'].shape[3] for e in valid_encoded_samples)
        W_max = max(e['pixel_values'].shape[4] for e in valid_encoded_samples)
        C = valid_encoded_samples[0]['pixel_values'].shape[2]
        dtype = valid_encoded_samples[0]['pixel_values'].dtype

        pv_batch = torch.zeros((num_valid_samples, N_max, C, H_max, W_max), dtype=dtype)
        for i, enc in enumerate(valid_encoded_samples):
            pv = enc['pixel_values'][0]
            N, _, H, W = pv.shape
            pv_batch[i, :N, :, :H, :W] = pv
        batch['pixel_values'] = pv_batch

        flat_grids, padded_indices = [], []
        idx_offset = 0
        for i, enc in enumerate(valid_encoded_samples):
            grid = enc['image_grid_thw']
            flat_grids.append(grid)
            idx = enc['image_grid_idx'][0] + idx_offset
            padded_indices.append(idx)
            idx_offset += num_tiles_per_sample[i]

        batch['grid_thw'] = torch.cat(flat_grids, dim=0)
        
        idx_batch = torch.zeros((num_valid_samples, N_max), dtype=torch.long)
        for i, idx in enumerate(padded_indices):
            idx_batch[i, :len(idx)] = idx
        batch['image_grid_idx'] = idx_batch
        
        batch['image_grid_thw'] = batch['grid_thw']

        return batch


def main():
    # main 函数保持不变
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
