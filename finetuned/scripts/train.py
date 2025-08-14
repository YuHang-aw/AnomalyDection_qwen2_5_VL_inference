#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, argparse
from pathlib import Path
import torch
from datasets import load_dataset, Features, Sequence, Value
from transformers import TrainingArguments, Trainer, set_seed

from src.modeling.load_qwen_vl import *

class VLDataCollator:
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
        self.image_token_id  = getattr(self.cfg, "image_token_id", tok.convert_tokens_to_ids("<|image_pad|>"))
        self.vision_start_id = getattr(self.cfg, "vision_start_token_id", tok.convert_tokens_to_ids("<|vision_start|>"))
        self.vision_end_id   = getattr(self.cfg, "vision_end_token_id",   tok.convert_tokens_to_ids("<|vision_end|>"))

        vc = getattr(self.cfg, "vision_config", None)
        self.patch_size = int(getattr(vc, "patch_size", 14)) if vc is not None else 14

        self._user_image_literal = "<image>"
        self._user_image_safe = "〈image〉"

    @staticmethod
    def _resize_keep_short(img, new_short):
        from PIL import Image
        w, h = img.size
        if min(w, h) <= new_short:
            return img
        if w < h:
            nw = new_short; nh = int(h * (new_short / w))
        else:
            nh = new_short; nw = int(w * (new_short / h))
        return img.resize((nw, nh), Image.BICUBIC)

    def _find_blocks(self, ids):
        starts, ends = [], []
        for i, t in enumerate(ids):
            if t == self.vision_start_id: starts.append(i)
            elif t == self.vision_end_id: ends.append(i)
        if not starts and not ends: return [], 0
        if len(starts) != len(ends): return [], -1
        blocks, total_img = [], 0
        si = 0
        for ei in range(len(ends)):
            s, e = starts[si], ends[ei]
            if s > e: return [], -1
            blocks.append((s, e))
            total_img += sum(1 for t in ids[s:e+1] if t == self.image_token_id)
            si += 1
        return blocks, total_img

    def _extract_sizes_from_pv(self, pv):
        import torch
        sizes = []
        if torch.is_tensor(pv):
            if pv.dim() == 4:  # [N,C,H,W]
                _, _, H, W = pv.shape
                sizes = [(int(H), int(W))] * int(pv.shape[0])
            elif pv.dim() == 3:  # [C,H,W]
                _, H, W = pv.shape
                sizes = [(int(H), int(W))]
            elif pv.dim() == 2:  # [H,W]
                H, W = pv.shape
                sizes = [(int(H), int(W))]
            else:
                raise ValueError(f"Unexpected pixel_values dims: {pv.dim()}")
        elif isinstance(pv, (list, tuple)):
            for t in pv:
                if not hasattr(t, "shape"):
                    raise ValueError("pixel_values list elements must be tensors")
                if t.dim() == 4:
                    for u in t: sizes.append((int(u.shape[-2]), int(u.shape[-1])))
                elif t.dim() == 3:
                    sizes.append((int(t.shape[-2]), int(t.shape[-1])))
                elif t.dim() == 2:
                    sizes.append((int(t.shape[-2]), int(t.shape[-1])))
                else:
                    raise ValueError(f"Unexpected per-image dims: {t.dim()}")
        else:
            raise ValueError("pixel_values must be Tensor or list of Tensors")
        return sizes

    def _ensure_chw(self, t):
        import torch
        if t.dim() == 2:
            t = t.unsqueeze(0)
        elif t.dim() == 3:
            pass
        else:
            raise ValueError(f"Single image tensor must be 2D/3D, got {t.dim()}D.")
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        return t

    def _to_4d_per_sample(self, pv):
        import torch
        if torch.is_tensor(pv):
            if pv.dim() == 4:
                if pv.shape[1] == 1: pv = pv.repeat(1,3,1,1)
                return pv
            elif pv.dim() in (2, 3):
                chw = self._ensure_chw(pv)
                return chw.unsqueeze(0)
            else:
                raise ValueError(f"pixel_values tensor must be 2D/3D/4D, got {pv.dim()}D.")
        elif isinstance(pv, (list, tuple)):
            imgs = []
            for t in pv:
                if not torch.is_tensor(t): raise ValueError("pixel_values list elements must be tensor")
                if t.dim() == 4:
                    for u in t: imgs.append(self._ensure_chw(u))
                else:
                    imgs.append(self._ensure_chw(t))
            sizes = [im.shape[-2:] for im in imgs]
            max_h = max(h for h, w in sizes); max_w = max(w for h, w in sizes)
            padded = []
            for im in imgs:
                h, w = im.shape[-2:]
                pad = (0, max_w - w, 0, max_h - h)
                padded.append(torch.nn.functional.pad(im, pad))
            return torch.stack(padded, dim=0)
        else:
            raise ValueError("pixel_values must be Tensor or list/tuple of Tensors")

    def __call__(self, examples):
        from PIL import Image
        import torch, math
        tok = self.processor.tokenizer

        # 1) 组模板
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

        # 2) 逐样本 encode（必要时降分）
        encoded = []
        for prompt, imgs in zip(prompts, images_batch):
            try_short = self.prefer_short_side or 896
            cur_imgs = imgs
            while True:
                enc = self.processor(text=prompt, images=cur_imgs, padding=False, truncation=False, return_tensors="pt")
                ids = enc["input_ids"][0]; am = enc["attention_mask"][0]
                ids_list = ids.tolist()

                # 计算视觉块边界 & 真实 image_pad 数
                blocks, _img_tok_cnt_total = self._find_blocks(ids_list)
                if _img_tok_cnt_total < 0:
                    raise ValueError("Malformed vision token blocks in encoded input; check template & data.")

                # —— 先按像素估一个 grid（floor），随后与真实 token 数精对齐
                if "pixel_values" in enc:
                    sizes0 = self._extract_sizes_from_pv(enc["pixel_values"])  # 本样本的每张图 (H,W)
                else:
                    sizes0 = []

                # 通常你是单图样本；这里按“单块/单图”处理，不做嵌套列表
                # 真实 image_pad 数：把所有视觉块里的 <image_pad> 统计起来
                real_cnt = 0
                for s,e in (blocks or []):
                    real_cnt += sum(1 for t in ids_list[s:e+1] if t == self.image_token_id)

                patch = self.patch_size
                if sizes0:
                    H, W = sizes0[0]
                    h_guess = max(1, int(H) // patch)   # 注意：用 floor
                    w_guess = max(1, int(W) // patch)
                else:
                    # 没拿到尺寸时，用近似正方根分解
                    side = max(1, int(math.sqrt(max(real_cnt,1))))
                    h_guess, w_guess = side, max(1, real_cnt // max(1, side))

                # 与真实 token 数对齐（关键！）
                prod = h_guess * w_guess
                if real_cnt > 0 and prod != real_cnt:
                    # 先按宽高比微调
                    if sizes0 and (H>0 and W>0):
                        ratio = W / max(H,1)
                        h = max(1, int(round(math.sqrt(real_cnt / max(ratio,1e-6)))))
                        w = max(1, real_cnt // h)
                        if h*w != real_cnt:
                            h = real_cnt; w = 1
                    else:
                        h = real_cnt; w = 1
                else:
                    h, w = h_guess, w_guess

                enc["image_sizes"] = [(int(H), int(W))] if sizes0 else [(h*self.patch_size, w*self.patch_size)]
                enc["image_grid_thw"] = [(1, int(h), int(w))] if real_cnt>0 else []

                # 视觉块优先的文本截断
                L = ids.shape[-1]
                if blocks:
                    min_keep = min(s for s, _ in blocks)
                    max_keep = max(e for _, e in blocks) + 1
                    need = max_keep - min_keep
                else:
                    need = 0

                if need <= self.max_length:
                    if L > self.max_length and blocks:
                        left = max(0, L - self.max_length)
                        if left > min_keep: left = min_keep
                        if left + self.max_length < max_keep:
                            left = max(0, max_keep - self.max_length)
                        ids = ids[left:left+self.max_length]
                        am  = am[left:left+self.max_length]
                        enc["input_ids"] = ids.unsqueeze(0)
                        enc["attention_mask"] = am.unsqueeze(0)
                    break

                if not self.auto_downscale_if_needed:
                    raise ValueError(
                        f"Visual token span ({need}) > max_length ({self.max_length}). "
                        f"Set a smaller image size (e.g., image_short_side=896) or enable auto_downscale_if_needed."
                    )
                # 自动降分
                scale = math.sqrt(self.max_length / max(need,1)) * 0.95
                new_short = max(self.downscale_floor, int(try_short * scale))
                if new_short >= try_short:
                    new_short = max(self.downscale_floor, try_short - self.downscale_step)
                if new_short < self.downscale_floor or new_short == try_short:
                    raise ValueError(
                        f"Even after planned downscaling (floor={self.downscale_floor}), "
                        f"visual span ({need}) > max_length ({self.max_length})."
                    )
                cur_imgs = [self._resize_keep_short(im, new_short) for im in cur_imgs]
                try_short = new_short

            encoded.append(enc)

        # 3) 文本右对齐 + labels
        max_len = max(e["input_ids"].shape[-1] for e in encoded)
        pad_id = tok.pad_token_id
        input_ids_list, attn_list = [], []
        for e in encoded:
            ids = e["input_ids"][0]; am = e["attention_mask"][0]
            pad_n = max_len - ids.shape[-1]
            if pad_n > 0:
                ids = torch.nn.functional.pad(ids, (0, pad_n), value=pad_id)
                am  = torch.nn.functional.pad(am,  (0, pad_n), value=0)
            input_ids_list.append(ids); attn_list.append(am)

        input_ids = torch.stack(input_ids_list, dim=0)
        attention_mask = torch.stack(attn_list, dim=0)
        labels = input_ids.clone(); labels[attention_mask == 0] = self.label_pad_token_id
        batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        # 4) pixel_values 统一 + grid/sizes 打包
        per_sample_pv, per_sample_mask = [], []
        per_sample_grids, per_sample_sizes = [], []

        for e in encoded:
            pv = self._to_4d_per_sample(e["pixel_values"]) if "pixel_values" in e else None
            if pv is None:
                raise ValueError("processor did not return pixel_values")
            per_sample_pv.append(pv)
            per_sample_mask.append(torch.ones((pv.shape[0],), dtype=torch.bool, device=pv.device))

            grid = e.get("image_grid_thw", [])
            sizes = e.get("image_sizes", [])
            # 强校验：单样本时 grid 必须是 list[tuple]，且 ∑(t*h*w) 等于真实 image_pad 数（再验一次）
            assert isinstance(grid, (list, tuple)) and all(isinstance(x, (list,tuple)) and len(x)==3 for x in grid), \
                f"grid_thw malformed: {grid}"
            per_sample_grids.append([ (int(t),int(h),int(w)) for (t,h,w) in grid ])
            per_sample_sizes.append([ (int(H),int(W)) for (H,W) in sizes ])

        if len(encoded) == 1:
            batch["pixel_values"] = per_sample_pv[0]         # [N,C,H,W]
            batch["pixel_values_mask"] = per_sample_mask[0]  # [N]
            batch["image_grid_thw"] = per_sample_grids[0]    # list[tuple]
            batch["image_sizes"]    = per_sample_sizes[0]    # list[(H,W)]
        else:
            # 多样本：pixel_values 做 5D pad；grid/sizes 逐样本列表
            Ns = [pv.shape[0] for pv in per_sample_pv]
            Cs = [pv.shape[1] for pv in per_sample_pv]
            Hs = [pv.shape[2] for pv in per_sample_pv]
            Ws = [pv.shape[3] for pv in per_sample_pv]
            C = Cs[0]; assert all(c == C for c in Cs), f"Channel mismatch: {Cs}"
            Nmax, Hmax, Wmax = max(Ns), max(Hs), max(Ws)

            pv_batch, mask_batch = [], []
            for pv, m in zip(per_sample_pv, per_sample_mask):
                Ni, Ci, Hi, Wi = pv.shape
                if Hi != Hmax or Wi != Wmax:
                    pv = torch.nn.functional.pad(pv, (0, Wmax - Wi, 0, Hmax - Hi))
                if Ni < Nmax:
                    pad_imgs = torch.zeros((Nmax - Ni, C, Hmax, Wmax), dtype=pv.dtype, device=pv.device)
                    pv = torch.cat([pv, pad_imgs], dim=0)
                    m  = torch.cat([m, torch.zeros((Nmax - Ni,), dtype=torch.bool, device=m.device)], dim=0)
                pv_batch.append(pv.unsqueeze(0))
                mask_batch.append(m.unsqueeze(0))
            batch["pixel_values"] = torch.cat(pv_batch, dim=0)
            batch["pixel_values_mask"] = torch.cat(mask_batch, dim=0)
            batch["image_grid_thw"] = per_sample_grids
            batch["image_sizes"]    = per_sample_sizes

        batch.setdefault("images", batch["pixel_values"])
        batch.setdefault("grid_thw", batch["image_grid_thw"])

        # 可选调试：检查 grid 与真实 token 完全一致
        if os.environ.get("DEBUG_GRID", "0") == "1":
            ids0 = encoded[0]["input_ids"][0].tolist()
            blocks0, _ = self._find_blocks(ids0)
            real0 = sum(1 for s,e in blocks0 for t in ids0[s:e+1] if t == self.image_token_id)
            got0  = sum(t*h*w for (t,h,w) in (batch["image_grid_thw"] if isinstance(batch["image_grid_thw"], list) else []))
            print(f"[GRID DEBUG] real={real0}  grid_sum={got0}  grid={batch['image_grid_thw'][:3]}")

        return batch



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_config", required=True)
    ap.add_argument("--resume", action="store_true", help="断点续训")
    args = ap.parse_args()

    # 读 yaml 配置
    import yaml
    with open(args.train_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))

    # ===== 数据 =====
    train_ds = load_dataset("json",
        data_files=cfg["data"]["train_jsonl"],
        split="train",
    )
    val_ds   = load_dataset("json",
        data_files=cfg["data"]["val_jsonl"],
        split="train"
    )

    # ===== 模型 + 处理器 =====
    model, processor = load_model_and_processor(cfg)
    tune_image_processor_from_cfg(processor, cfg)
    model = apply_freeze_and_lora(model, cfg)

    # ===== Collator 实例 =====
    max_seq_len = int(cfg.get("max_seq_len", 4096))
    sanitize_image_literal = bool(cfg.get("data", {}).get("sanitize_image_literal", True))
    images_cfg = cfg.get("images", {})  # 可选区块，无则走默认
    auto_downscale = bool(images_cfg.get("auto_downscale_if_needed", True))
    downscale_floor = int(images_cfg.get("downscale_floor", 448))
    downscale_step  = int(images_cfg.get("downscale_step", 64))
    prefer_short    = int(cfg.get("image_short_side", 896))  # 建议 896

    collator = VLDataCollator(
        processor=processor,
        model_config=model.config,
        max_length=max_seq_len,
        add_generation_prompt=False,               # 训练固定 False
        label_pad_token_id=-100,
        sanitize_user_image_token=sanitize_image_literal,
        auto_downscale_if_needed=auto_downscale,
        prefer_short_side=prefer_short,
        downscale_floor=downscale_floor,
        downscale_step=downscale_step,
    )

    # ===== 训练参数 =====
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
        remove_unused_columns=False,                  # 关键：别裁掉 pixel_values 等键
        ddp_find_unused_parameters=False if int(os.environ.get("WORLD_SIZE", "1")) > 1 else None,
    )

    # ===== Trainer =====
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
