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
    - 先 pad 图像：把 H,W pad 到 patch_size*spatial_merge_size 的倍数（仅右/下边补齐）
    - 再计算 grid_thw：h = H_pad/patch, w = W_pad/patch（天然是 m 的倍数）
    - pixel_values：
        * 单样本 → [N,C,H,W]
        * 多样本 → [B,Nmax,C,Hmax,Wmax] + pixel_values_mask
    - grid_thw：
        * 单样本 → torch.LongTensor[N,3]（行=(1,h,w)）
        * 多样本 → list[Tensor[Ni,3]]
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
        # 保证有 pad_token
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.pad_token_id = tok.pad_token_id

        # 特殊 token（兜底）
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
            nw = new_short; nh = int(h * (new_short / w))
        else:
            nh = new_short; nw = int(w * (new_short / h))
        return img.resize((nw, nh), Image.BICUBIC)

    def _pad_to_grid_multiple(self, img):
        """把 PIL 图像右/下 pad 到 (patch_size*spatial_merge_size) 的倍数。"""
        from PIL import ImageOps
        p = max(1, self.patch_size)
        m = max(1, self.spatial_merge_size)
        unit = p * m  # 最小对齐单元（像素）
        W, H = img.size
        W_pad = ((W + unit - 1) // unit) * unit
        H_pad = ((H + unit - 1) // unit) * unit
        pad_r = W_pad - W
        pad_b = H_pad - H
        if pad_r == 0 and pad_b == 0:
            return img, H_pad, W_pad
        # 只在右、下方向 pad，保持内容不变
        img2 = ImageOps.expand(img, border=(0, 0, pad_r, pad_b), fill=0)
        return img2, H_pad, W_pad

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
        sizes = []
        if torch.is_tensor(pv):
            if pv.dim() == 4:  # [N,C,H,W]
                N, _, H, W = pv.shape
                sizes = [(int(H), int(W)) for _ in range(int(N))]
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

    def _sizes_to_grid_tensor_exact(self, sizes):
        """sizes 已经 pad 到 unit 的倍数：直接算 h=H/patch, w=W/patch（天然是 m 的倍数）"""
        import torch
        p = max(1, self.patch_size)
        grid = [(1, int(H//p), int(W//p)) for (H, W) in sizes]
        return torch.as_tensor(grid, dtype=torch.long) if grid else torch.zeros((0,3), dtype=torch.long)

    def __call__(self, examples):
        from PIL import Image
        import torch, math
        tok = self.processor.tokenizer

        # 1) 组模板 + 收集、预 pad PIL
        chats_for_template, images_batch, grids_from_pad = [], [], []
        p = self.patch_size; m = self.spatial_merge_size
        for ex in examples:
            mm_tpl, imgs, grids = [], [], []
            for mobj in ex["messages"]:
                parts_tpl = []
                for pobj in mobj["content"]:
                    if pobj["type"] == "image" and pobj.get("image"):
                        img = Image.open(pobj["image"]).convert("RGB")
                        # 可选：先做短边下采样再对齐 pad（避免超长序列）
                        if self.prefer_short_side:
                            img = self._resize_keep_short(img, self.prefer_short_side)
                        img, H_pad, W_pad = self._pad_to_grid_multiple(img)
                        imgs.append(img)
                        # 对齐后 h,w（天然是 m 的倍数、且不会为 0）
                        h = H_pad // p; w = W_pad // p
                        grids.append((1, h, w))
                        parts_tpl.append({"type": "image"})
                    else:
                        txt = (pobj.get("text") or "")
                        if self.sanitize_user_image_token and self._user_image_literal in txt:
                            txt = txt.replace(self._user_image_literal, self._user_image_safe)
                        parts_tpl.append({"type": "text", "text": txt})
                mm_tpl.append({"role": mobj["role"], "content": parts_tpl})
            chats_for_template.append(mm_tpl)
            images_batch.append(imgs)
            grids_from_pad.append(grids)

        prompts = self.processor.apply_chat_template(
            chats_for_template, add_generation_prompt=self.add_generation_prompt, tokenize=False
        )

        # 2) 逐样本 encode（不截断视觉块；必要时降分重新来一遍）
        encoded = []
        for prompt, imgs, grids in zip(prompts, images_batch, grids_from_pad):
            try_short = self.prefer_short_side or 896
            cur_imgs = imgs
            cur_grids = grids
            while True:
                enc = self.processor(text=prompt, images=cur_imgs, padding=False, truncation=False, return_tensors="pt")

                # 文本侧判断是否超长（视觉块不截）
                ids = enc["input_ids"][0]; am = enc["attention_mask"][0]
                ids_list = ids.tolist()
                blocks, _ = self._find_blocks(ids_list)
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
                        enc["input_ids"] = ids[left:left+self.max_length].unsqueeze(0)
                        enc["attention_mask"] = am[left:left+self.max_length].unsqueeze(0)
                    # 记录预 pad 得到的 grid（与像素一一对应）
                    enc["image_grid_thw"] = torch.as_tensor(cur_grids, dtype=torch.long) if cur_grids else torch.zeros((0,3), dtype=torch.long)
                    break

                if not self.auto_downscale_if_needed:
                    raise ValueError(
                        f"Visual token span ({need}) > max_length ({self.max_length}). "
                        f"Try smaller image_short_side or enable auto_downscale_if_needed."
                    )
                # 若必须降分：先把图片下采样，然后**再**走一次 pad 对齐
                scale = math.sqrt(self.max_length / max(need,1)) * 0.95
                new_short = max(self.downscale_floor, int(try_short * scale))
                if new_short >= try_short:
                    new_short = max(self.downscale_floor, try_short - self.downscale_step)
                if new_short < self.downscale_floor or new_short == try_short:
                    raise ValueError("Downscaling failed to reduce visual span.")
                new_imgs, new_grids = [], []
                for im in imgs:
                    im2 = self._resize_keep_short(im, new_short)
                    im2, H_pad, W_pad = self._pad_to_grid_multiple(im2)
                    new_imgs.append(im2)
                    new_grids.append((1, (H_pad//p), (W_pad//p)))
                cur_imgs, cur_grids = new_imgs, new_grids
                try_short = new_short

            encoded.append(enc)

        # 3) 文本右 pad + labels
        max_len = max(e["input_ids"].shape[-1] for e in encoded)
        ids_list, am_list = [], []
        for e in encoded:
            ids = e["input_ids"][0]; am = e["attention_mask"][0]
            pad_n = max_len - ids.shape[-1]
            if pad_n > 0:
                ids = torch.nn.functional.pad(ids, (0, pad_n), value=self.pad_token_id)  # 右 pad
                am  = torch.nn.functional.pad(am,  (0, pad_n), value=0)
            ids_list.append(ids); am_list.append(am)
        input_ids = torch.stack(ids_list, dim=0)
        attention_mask = torch.stack(am_list, dim=0)
        labels = input_ids.clone(); labels[attention_mask == 0] = self.label_pad_token_id
        batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        # 4) 视觉：把 processor 的输出改成期望的 4D/5D，并校验 grid 与特征长度一致
        def _ensure_chw(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 2: t = t.unsqueeze(0)
            elif t.dim() != 3: raise ValueError(f"image tensor must be 2D/3D, got {t.dim()}D")
            if t.shape[0] == 1: t = t.repeat(3,1,1)
            return t
        def _to_4d_per_sample(pv):
            if torch.is_tensor(pv):
                if pv.dim() == 4:
                    if pv.shape[1] == 1: pv = pv.repeat(1,3,1,1)
                    return pv
                elif pv.dim() in (2,3):
                    return _ensure_chw(pv).unsqueeze(0)
                else:
                    raise ValueError(f"pixel_values dim {pv.dim()}")
            else:
                imgs = []
                for t in pv:
                    if not torch.is_tensor(t): raise ValueError("pixel_values list elements must be tensor")
                    if t.dim() == 4:
                        for u in t: imgs.append(_ensure_chw(u))
                    else:
                        imgs.append(_ensure_chw(t))
                # 这些都是预 pad 后的图，尺寸应已一致
                Hs = [im.shape[-2] for im in imgs]; Ws = [im.shape[-1] for im in imgs]
                Hm, Wm = max(Hs), max(Ws)
                padded = [torch.nn.functional.pad(im, (0, Wm - im.shape[-1], 0, Hm - im.shape[-2])) for im in imgs]
                return torch.stack(padded, dim=0)

        per_pv, per_grid, per_sizes = [], [], []
        for e in encoded:
            pv = _to_4d_per_sample(e["pixel_values"])
            sizes = self._extract_sizes_from_pv(pv)  # 这就是预 pad 后的 (H_pad,W_pad)
            # grid 由预 pad 直接得出（与 sizes 一致）
            grid = self._sizes_to_grid_tensor_exact(sizes)
            # 与 encode 时记录的 grid 再次核对，如不一致，以真实 sizes 回写（防御性）
            grid_ref = e.get("image_grid_thw", None)
            if grid_ref is not None and grid_ref.numel():
                if grid.shape != grid_ref.shape or not torch.equal(grid, grid_ref.to(grid.dtype)):
                    grid = grid  # 以 sizes 推导的为准
            per_pv.append(pv); per_grid.append(grid); per_sizes.append(sizes)

            # 最关键的**一致性断言**：视觉特征 token 数 == ∑(h*w)
            # 对于每张图，patch 数 = (H_pad/patch)*(W_pad/patch) = h*w
            # 由于我们已 pad 到 unit，成立；若失败则说明有意外变形
            hws = grid[:,1:].prod(dim=1).long().sum().item() if grid.numel() else 0
            # 真实 patch 数（逐图相加）
            real = sum([(H//self.patch_size)*(W//self.patch_size) for (H,W) in sizes])
            assert hws == real, f"[FATAL] grid_sum({hws}) != real_patches({real}); sizes={sizes}, grid={grid.tolist()}"

        if len(encoded) == 1:
            batch["pixel_values"]      = per_pv[0]            # [N,C,H,W]
            batch["pixel_values_mask"] = torch.ones((per_pv[0].shape[0],), dtype=torch.bool)
            batch["image_grid_thw"]    = per_grid[0]          # LongTensor[N,3]
            batch["grid_thw"]          = batch["image_grid_thw"]
            batch["image_sizes"]       = per_sizes[0]
        else:
            # 多样本：pixel_values pad 到 5D；grid 逐样本 list[Tensor[Ni,3]]
            Ns = [pv.shape[0] for pv in per_pv]
            Cs = [pv.shape[1] for pv in per_pv]
            Hs = [pv.shape[2] for pv in per_pv]
            Ws = [pv.shape[3] for pv in per_pv]
            C = Cs[0]; assert all(c==C for c in Cs)
            Nmax, Hmax, Wmax = max(Ns), max(Hs), max(Ws)
            pv_batch, mask_batch = [], []
            for pv in per_pv:
                Ni, Ci, Hi, Wi = pv.shape
                if Hi != Hmax or Wi != Wmax:
                    pv = torch.nn.functional.pad(pv, (0, Wmax - Wi, 0, Hmax - Hi))
                if Ni < Nmax:
                    pad_imgs = torch.zeros((Nmax-Ni, C, Hmax, Wmax), dtype=pv.dtype, device=pv.device)
                    pv = torch.cat([pv, pad_imgs], dim=0)
                    msk = torch.cat([torch.ones((Ni,), dtype=torch.bool, device=pv.device),
                                     torch.zeros((Nmax-Ni,), dtype=torch.bool, device=pv.device)], dim=0)
                else:
                    msk = torch.ones((Ni,), dtype=torch.bool, device=pv.device)
                pv_batch.append(pv.unsqueeze(0)); mask_batch.append(msk.unsqueeze(0))
            batch["pixel_values"]      = torch.cat(pv_batch, dim=0)
            batch["pixel_values_mask"] = torch.cat(mask_batch, dim=0)
            batch["image_grid_thw"]    = per_grid
            batch["grid_thw"]          = batch["image_grid_thw"]
            batch["image_sizes"]       = per_sizes

        if os.environ.get("DEBUG_VL", "0") == "1":
            print("---- DEBUG_VL ----")
            if torch.is_tensor(batch["grid_thw"]):
                g = batch["grid_thw"]
                print("grid_thw:", tuple(g.shape), g.tolist()[:min(3, g.shape[0])])
            else:
                print("grid_thw(list):", [tuple(x.shape) for x in batch["grid_thw"]])
            pv = batch["pixel_values"]
            print("pixel_values:", tuple(pv.shape))
            print("------------------")

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
    images_cfg      = cfg.get("images", {})
    auto_downscale  = bool(images_cfg.get("auto_downscale_if_needed", True))
    downscale_floor = int(images_cfg.get("downscale_floor", 448))
    downscale_step  = int(images_cfg.get("downscale_step", 64))
    prefer_short    = int(cfg.get("image_short_side", 896))

    collator = VLDataCollator(
        processor=processor,
        model_config=model.config,
        max_length=max_seq_len,
        add_generation_prompt=False,
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
        remove_unused_columns=False,   # 保留 pixel_values / grid_thw
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
