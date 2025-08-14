#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, set_seed

from src.modeling.load_qwen_vl import *

# -----------------------------
# Collator (final, robust)
# -----------------------------
class VLDataCollator:
    """
    - 逐样本 encode（truncation=False）
    - “视觉块优先”：截断只裁文本，不切 <|vision_start|>...<|vision_end|>
    - pixel_values: 单样本 [N,C,H,W]；多样本 [B,Nmax,C,Hmax,Wmax] + pixel_values_mask
    - grid_thw（传入模型）：
        * 单样本：list[tuple(int,int,int)] → [(t,h,w), ...]，t=1
        * 多样本：list[list[tuple(...)] ]（逐样本一份）
      其中 h、w = ceil(H/patch)，再向上补齐为 spatial_merge_size 的倍数，且最小=spatial_merge_size
      之后用真实 <|image_pad|> 数对齐：若 ∑(t*h*w) != real，最后一个 (t,h,w) 以 m 步长微调 w 使其相等
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
        # 视觉相关 token id；若 config 没有则回退
        self.image_token_id  = getattr(model_config, "image_token_id", tok.convert_tokens_to_ids("<|image_pad|>"))
        self.vision_start_id = getattr(model_config, "vision_start_token_id", tok.convert_tokens_to_ids("<|vision_start|>"))
        self.vision_end_id   = getattr(model_config, "vision_end_token_id",   tok.convert_tokens_to_ids("<|vision_end|>"))

        vc = getattr(model_config, "vision_config", None)
        self.patch_size = int(getattr(vc, "patch_size", 14)) if vc is not None else 14
        self.spatial_merge_size = int(getattr(vc, "spatial_merge_size", 2)) if vc is not None else 2

        # 清洗 "<image>" 的替换
        self._user_image_literal = "<image>"
        self._user_image_safe = "〈image〉"

    # --- helpers ---
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
                if not torch.is_tensor(t):
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

    def _sizes_to_grid_list(self, sizes):
        """(H,W) → (t=1,h,w)，先 ceil(H/patch)、ceil(W/patch)，再向上补齐为 m 的倍数；最小 = m。"""
        import math
        p = max(1, self.patch_size)
        m = max(1, self.spatial_merge_size)
        def up_to(x, k):  # 向上补为 k 的倍数
            x = max(k, int(x))
            return ((x + k - 1) // k) * k
        grid = []
        for (H, W) in sizes:
            H = max(1, int(H)); W = max(1, int(W))
            h = up_to(math.ceil(H / float(p)), m)
            w = up_to(math.ceil(W / float(p)), m)
            grid.append((1, int(h), int(w)))
        return grid  # list[tuple]

    def _count_image_pad_tokens(self, ids, blocks):
        """统计真实 <|image_pad|> 数（遍历所有视觉块内的 image_token）。"""
        if not blocks: return 0
        return sum(1 for s, e in blocks for t in ids[s:e+1] if t == self.image_token_id)

    def _adjust_grid_to_match_tokens(self, grid, real_cnt):
        """
        grid: list[(1,h,w)]，m 为 spatial_merge_size。
        若 sum(t*h*w) != real_cnt，微调最后一项的 w（以 m 步长），直至相等；必要时也可调 h。
        """
        if real_cnt <= 0:
            return []  # 没有视觉 token，grid 置空
        m = max(1, self.spatial_merge_size)
        grid = [ (int(t), int(h), int(w)) for (t,h,w) in grid ]
        total = sum(t*h*w for (t,h,w) in grid)
        if total == real_cnt:
            return grid

        if not grid:
            # 极端兜底（理论上不会来这）：造一个 m×m 的块并按需要放大 w
            h = m
            w = max(m, (real_cnt + h - 1)//h)
            # 再向上对齐到 m
            w = ((w + m - 1)//m)*m
            return [(1, h, w)]

        # 微调最后一项
        t, h, w = grid[-1]
        h = max(m, (h + m - 1)//m * m)
        w = max(m, (w + m - 1)//m * m)
        delta = real_cnt - total
        if delta == 0:
            grid[-1] = (t, h, w)
            return grid

        # 以 m 步长调整 w（每次改动增加/减少 t*h*m 个 token）
        step = max(1, t*h*m)
        # 需要的“步数”
        k = (abs(delta) + step - 1)//step
        if delta > 0:
            w_new = w + k*m
        else:
            w_new = max(m, w - k*m)
        grid[-1] = (t, h, w_new)

        # 再次校验，不同步就最后再微调一次
        total2 = sum(tt*hh*ww for (tt,hh,ww) in grid)
        if total2 != real_cnt:
            # 二次微调：直接把 w 设为能匹配的最近值（仍按 m 对齐）
            need_last = real_cnt - sum(tt*hh*ww for (tt,hh,ww) in grid[:-1])
            # need_last 必须是 t*h*X 的形式 → 求 X
            x = max(m, (need_last + t*h - 1)//(t*h))  # 向上
            x = ((x + m - 1)//m) * m
            grid[-1] = (t, h, x)
        return grid

    # --- main ---
    def __call__(self, examples):
        from PIL import Image
        import math
        tok = self.processor.tokenizer

        # 1) 组模板 + 收集 PIL
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

                ids = enc["input_ids"][0]
                am  = enc["attention_mask"][0]
                ids_list = ids.tolist()

                # 视觉块 & 真实 image_pad 数
                blocks, _ = self._find_blocks(ids_list)
                real_cnt = self._count_image_pad_tokens(ids_list, blocks)

                # 用像素尺寸估 grid（对齐到 m，且最小=m），再和真实 image_pad 对齐
                if "pixel_values" in enc:
                    sizes0 = self._extract_sizes_from_pv(enc["pixel_values"])
                else:
                    sizes0 = []
                grid0 = self._sizes_to_grid_list(sizes0 if sizes0 else [(self.patch_size, self.patch_size)])
                grid  = self._adjust_grid_to_match_tokens(grid0, real_cnt)

                # 保存在 enc 里，稍后打包
                enc["image_sizes"] = sizes0
                enc["grid_thw_list"] = grid  # 关键：list[tuple]（单样本扁平）

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
                        left = max(0, L - self.max_length)  # 保尾
                        if left > min_keep: left = min_keep
                        if left + self.max_length < max_keep:
                            left = max(0, max_keep - self.max_length)
                        enc["input_ids"] = ids[left:left+self.max_length].unsqueeze(0)
                        enc["attention_mask"] = am[left:left+self.max_length].unsqueeze(0)
                    break

                if not self.auto_downscale_if_needed:
                    raise ValueError(
                        f"Visual token span ({need}) > max_length ({self.max_length}). "
                        f"Set a smaller image size (e.g., image_short_side=896) or enable auto_downscale_if_needed."
                    )

                # 自动降分：按比例估算短边
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

        # 4) 统一 pixel_values；grid 按“单样本扁平 list[tuple] / 多样本 list[list[tuple]]”返回
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
            elif isinstance(pv, (list, tuple)):
                imgs = []
                for t in pv:
                    if not torch.is_tensor(t): raise ValueError("pixel_values list elements must be tensor")
                    if t.dim() == 4:
                        for u in t: imgs.append(_ensure_chw(u))
                    else:
                        imgs.append(_ensure_chw(t))
                sizes = [im.shape[-2:] for im in imgs]
                max_h = max(h for h, w in sizes); max_w = max(w for h, w in sizes)
                padded = [torch.nn.functional.pad(im, (0, max_w - im.shape[-1], 0, max_h - im.shape[-2])) for im in imgs]
                return torch.stack(padded, dim=0)
            else:
                raise ValueError("pixel_values must be Tensor or list/tuple of Tensors")

        per_pv, per_mask, per_grid, per_sizes = [], [], [], []
        for e in encoded:
            pv = _to_4d_per_sample(e["pixel_values"])
            per_pv.append(pv)
            per_mask.append(torch.ones((pv.shape[0],), dtype=torch.bool, device=pv.device))

            grid = e.get("grid_thw_list", [])
            # 强校验：grid 必须是 list[tuple] 且每个 (t,h,w) 的 h/w >= m
            m = max(1, self.spatial_merge_size)
            assert isinstance(grid, (list, tuple)) and all(isinstance(x, (list,tuple)) and len(x)==3 for x in grid), \
                f"grid_thw malformed: {grid}"
            for i,(t,h,w) in enumerate(grid):
                h = int(h); w = int(w); t = int(t)
                if h < m: h = m
                if w < m: w = m
                # 再对齐到 m 的倍数
                if h % m != 0: h = ((h + m - 1)//m)*m
                if w % m != 0: w = ((w + m - 1)//m)*m
                grid[i] = (t,h,w)
            per_grid.append(grid)
            per_sizes.append(e.get("image_sizes", []))

        # 写 batch（单样本=扁平 list[tuple]；多样本=list[list[tuple]]）
        if len(encoded) == 1:
            batch["pixel_values"]      = per_pv[0]            # [N,C,H,W]
            batch["pixel_values_mask"] = per_mask[0]          # [N]
            batch["grid_thw"]          = per_grid[0]          # list[tuple]（单样本扁平！）
            batch["image_grid_thw"]    = batch["grid_thw"]    # 兼容别名
            batch["image_sizes"]       = per_sizes[0]         # list[(H,W)]
        else:
            # 多样本场景：pixel_values pad 到 5D；grid 保持逐样本列表
            Ns = [pv.shape[0] for pv in per_pv]
            Cs = [pv.shape[1] for pv in per_pv]
            Hs = [pv.shape[2] for pv in per_pv]
            Ws = [pv.shape[3] for pv in per_pv]
            C = Cs[0]; assert all(c==C for c in Cs)
            Nmax, Hmax, Wmax = max(Ns), max(Hs), max(Ws)
            pv_batch, mask_batch = [], []
            for pv, msk in zip(per_pv, per_mask):
                Ni, Ci, Hi, Wi = pv.shape
                if Hi != Hmax or Wi != Wmax:
                    pv = torch.nn.functional.pad(pv, (0, Wmax - Wi, 0, Hmax - Hi))
                if Ni < Nmax:
                    pad_imgs = torch.zeros((Nmax-Ni, C, Hmax, Wmax), dtype=pv.dtype, device=pv.device)
                    pv = torch.cat([pv, pad_imgs], dim=0)
                    msk = torch.cat([msk, torch.zeros((Nmax-Ni,), dtype=torch.bool, device=msk.device)], dim=0)
                pv_batch.append(pv.unsqueeze(0)); mask_batch.append(msk.unsqueeze(0))
            batch["pixel_values"]      = torch.cat(pv_batch, dim=0)
            batch["pixel_values_mask"] = torch.cat(mask_batch, dim=0)
            batch["grid_thw"]          = per_grid              # list[list[tuple]]
            batch["image_grid_thw"]    = batch["grid_thw"]
            batch["image_sizes"]       = per_sizes

        # 调试：确保不会出现 h/w==0；打印头两项
        if os.environ.get("DEBUG_GRID", "0") == "1":
            g = batch["grid_thw"]
            if isinstance(g, list) and g and isinstance(g[0], tuple):
                print("GRID(single):", len(g), "head:", g[:2])
            elif isinstance(g, list):
                print("GRID(batch):", [len(x) for x in g], "head0:", (g[0][:2] if g and g[0] else []))
            else:
                print("GRID type unexpected:", type(g))

        return batch

# -----------------------------
# train loop (unchanged)
# -----------------------------
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

    # 数据
    train_ds = load_dataset("json",
        data_files=cfg["data"]["train_jsonl"],
        split="train",
    )
    val_ds   = load_dataset("json",
        data_files=cfg["data"]["val_jsonl"],
        split="train"
    )

    # 模型 + 处理器
    model, processor = load_model_and_processor(cfg)
    tune_image_processor_from_cfg(processor, cfg)
    model = apply_freeze_and_lora(model, cfg)

    # Collator
    max_seq_len = int(cfg.get("max_seq_len", 4096))
    sanitize_image_literal = bool(cfg.get("data", {}).get("sanitize_image_literal", True))
    images_cfg = cfg.get("images", {})
    auto_downscale = bool(images_cfg.get("auto_downscale_if_needed", True))
    downscale_floor = int(images_cfg.get("downscale_floor", 448))
    downscale_step  = int(images_cfg.get("downscale_step", 64))
    prefer_short    = int(cfg.get("image_short_side", 896))  # 建议 896

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
