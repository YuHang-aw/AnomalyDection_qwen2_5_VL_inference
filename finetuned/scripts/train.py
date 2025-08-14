#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, argparse
from pathlib import Path
import torch
from datasets import load_dataset, Features, Sequence, Value
from transformers import TrainingArguments, Trainer, set_seed

from src.modeling.load_qwen_vl import *
# =========== Collator：模板→文本，占位→真实图片 ===========
class VLDataCollator:
    """
    最小不定修正版（接入 config）：
    - 逐样本 encode（truncation=False）
    - 优先完整保留视觉块 <|vision_start|> ... <|vision_end|>
    - 若超长：先做“整块保留式”文本截断；如视觉块本身超过 max_length：
        * auto_downscale_if_needed=True → 等比例下采样，直到能塞进为止
        * 否则报错，提示调小分辨率或调大 max_length
    - 可选清洗用户文本中的 "<image>" 字面量为 "〈image〉"
    - 所有关键阈值都可从 config 传入
    """
    def __init__(
        self,
        processor,
        model_config,
        max_length=4096,
        add_generation_prompt=False,
        label_pad_token_id=-100,
        sanitize_user_image_token=True,
        # ↓↓↓ 来自配置的图像控制项
        auto_downscale_if_needed=True,
        prefer_short_side=None,   # 起始“短边”尝试值（例如 cfg['image_short_side']）
        downscale_floor=448,      # 最低短边，防止无限缩
        downscale_step=64,        # 无法从比例法推小就按步长减
    ):
        self.processor = processor
        self.cfg = model_config
        self.max_length = int(max_length)
        self.add_generation_prompt = bool(add_generation_prompt)  # 训练建议 False
        self.label_pad_token_id = int(label_pad_token_id)
        self.sanitize_user_image_token = bool(sanitize_user_image_token)

        self.auto_downscale_if_needed = bool(auto_downscale_if_needed)
        self.prefer_short_side = int(prefer_short_side) if prefer_short_side else None
        self.downscale_floor = int(downscale_floor)
        self.downscale_step  = int(downscale_step)

        tok = processor.tokenizer
        # Qwen2.5-VL 的视觉相关 token id（若模型未提供则用常见文本反查）
        self.image_token_id = getattr(self.cfg, "image_token_id", tok.convert_tokens_to_ids("<|image_pad|>"))
        self.vision_start_id = getattr(self.cfg, "vision_start_token_id", tok.convert_tokens_to_ids("<|vision_start|>"))
        self.vision_end_id   = getattr(self.cfg, "vision_end_token_id",   tok.convert_tokens_to_ids("<|vision_end|>"))

        # 清洗用
        self._user_image_literal = "<image>"
        self._user_image_safe = "〈image〉"

    @staticmethod
    def _resize_keep_short(img, new_short):
        from PIL import Image
        w, h = img.size
        if min(w, h) <= new_short:
            return img
        if w < h:
            nw = new_short
            nh = int(h * (new_short / w))
        else:
            nh = new_short
            nw = int(w * (new_short / h))
        return img.resize((nw, nh), Image.BICUBIC)

    def _find_blocks(self, ids):
        """返回视觉块 [start, end] 闭区间列表，以及块内 image_token 计数。"""
        starts, ends = [], []
        for i, t in enumerate(ids):
            if t == self.vision_start_id: starts.append(i)
            elif t == self.vision_end_id: ends.append(i)
        if not starts and not ends:
            return [], 0
        if len(starts) != len(ends):
            return [], -1
        blocks, total_img = [], 0
        si = 0
        for ei in range(len(ends)):
            s, e = starts[si], ends[ei]
            if s > e:
                return [], -1
            blocks.append((s, e))
            total_img += sum(1 for t in ids[s:e+1] if t == self.image_token_id)
            si += 1
        return blocks, total_img

    def __call__(self, examples):
        from PIL import Image
        import torch, math
        tok = self.processor.tokenizer

        # 1) 组织模板输入：图片只放“占位”，真实 PIL 由 processor 处理；可选清洗 "<image>"
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

        # 2) 逐样本 encode（不截断）；必要时对该样本“按需降分辨率”
        encoded = []
        for prompt, imgs in zip(prompts, images_batch):
            try_short = self.prefer_short_side or 896  # 起步短边（可从 cfg['image_short_side'] 传入）
            cur_imgs = imgs

            while True:
                enc = self.processor(text=prompt, images=cur_imgs, padding=False, truncation=False, return_tensors="pt")
                ids = enc["input_ids"][0]
                am  = enc["attention_mask"][0]
                ids_list = ids.tolist()

                blocks, _img_tok_cnt = self._find_blocks(ids_list)
                if _img_tok_cnt < 0:
                    raise ValueError("Malformed vision token blocks in encoded input; check template & data.")

                L = ids.shape[-1]
                if blocks:
                    min_keep = min(s for s, _ in blocks)
                    max_keep = max(e for _, e in blocks) + 1
                    need = max_keep - min_keep
                else:
                    min_keep = 0
                    max_keep = 0
                    need = 0

                # 2.1 视觉块能放下 → 如整体仍超长，做“整块保留式”文本截断
                if need <= self.max_length:
                    if L > self.max_length and blocks:
                        left = max(0, L - self.max_length)  # 优先保尾
                        if left > min_keep: left = min_keep  # 不能切断第一块视觉
                        if left + self.max_length < max_keep:
                            left = max(0, max_keep - self.max_length)
                        ids = ids[left:left+self.max_length]
                        am  = am[left:left+self.max_length]
                        enc["input_ids"] = ids.unsqueeze(0)
                        enc["attention_mask"] = am.unsqueeze(0)
                    break  # 这条样本 OK

                # 2.2 视觉块本身 > max_length
                if not self.auto_downscale_if_needed:
                    raise ValueError(
                        f"Visual token span ({need}) > max_length ({self.max_length}). "
                        f"Set a smaller image size (e.g., image_short_side=896) or enable auto_downscale_if_needed."
                    )

                # 按需下采样：估算缩放比例，把 need 压到 95% budget
                scale = math.sqrt(self.max_length / need) * 0.95
                new_short = max(self.downscale_floor, int(try_short * scale))
                if new_short >= try_short:
                    new_short = max(self.downscale_floor, try_short - self.downscale_step)

                if new_short < self.downscale_floor or new_short == try_short:
                    raise ValueError(
                        f"Even after planned downscaling (floor={self.downscale_floor}), "
                        f"visual span ({need}) > max_length ({self.max_length}). "
                        f"建议：把 image_short_side 调到 896/672，或提高 max_length（若模型支持）。"
                    )

                # 真的缩图再重编一次
                cur_imgs = [self._resize_keep_short(im, new_short) for im in cur_imgs]
                try_short = new_short

            encoded.append(enc)

        # === 3) 右侧 padding → batch，labels 的 padding 置 -100（保持不变）===
        max_len = max(e["input_ids"].shape[-1] for e in encoded)
        pad_id = tok.pad_token_id
        input_ids_list, attn_list = [], []
        for e in encoded:
            ids = e["input_ids"][0]
            am  = e["attention_mask"][0]
            pad_n = max_len - ids.shape[-1]
            if pad_n > 0:
                ids = torch.nn.functional.pad(ids, (0, pad_n), value=pad_id)
                am  = torch.nn.functional.pad(am,  (0, pad_n), value=0)
            input_ids_list.append(ids)
            attn_list.append(am)

        input_ids = torch.stack(input_ids_list, dim=0)
        attention_mask = torch.stack(attn_list, dim=0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = self.label_pad_token_id

        batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        # === 4) pixel_values — 统一成 Tensor，覆盖 2D/3D/4D/list 的所有形态 ===
        def _ensure_chw(t: torch.Tensor) -> torch.Tensor:
            """
            把任意一张图规整成 [C,H,W]：
            - [H,W]  → [1,H,W]（随后会复制到 3 通道）
            - [C,H,W] 保持
            - 其他维度一律报错（不应该出现）
            """
            if t.dim() == 2:  # [H,W]
                t = t.unsqueeze(0)  # [1,H,W]
            elif t.dim() == 3:  # [C,H,W]
                pass
            else:
                raise ValueError(f"Single image tensor must be 2D/3D, got {t.dim()}D.")
            # 单通道 → 3 通道（RGB）
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            return t

        def _to_4d_per_sample(pv):
            """
            把“本样本”的 pixel_values 规整成 [Ni,C,H,W]：
            - Tensor: [H,W]/[C,H,W]/[N,C,H,W]
            - List/Tuple: [Tensor(...), ...]，元素可混合 2D/3D/4D（4D 会拆开）
            """
            if torch.is_tensor(pv):
                if pv.dim() == 4:           # [N,C,H,W]
                    # 也容错 N=1,C=H;W=... 等怪形态，但假设标准 4D
                    Ni, Ci, Hi, Wi = pv.shape
                    if Ci == 1:
                        pv = pv.repeat(1, 3, 1, 1)
                    return pv
                elif pv.dim() in (2, 3):    # [H,W] / [C,H,W]
                    chw = _ensure_chw(pv)   # -> [C,H,W]
                    return chw.unsqueeze(0) # -> [1,C,H,W]
                else:
                    raise ValueError(f"pixel_values tensor must be 2D/3D/4D, got {pv.dim()}D.")

            elif isinstance(pv, (list, tuple)):
                imgs = []
                for t in pv:
                    if not torch.is_tensor(t):
                        raise ValueError("pixel_values list elements must be torch.Tensor")
                    if t.dim() == 4:  # [N,C,H,W] → 拆到 3D
                        for u in t:
                            imgs.append(_ensure_chw(u))
                    else:
                        imgs.append(_ensure_chw(t))  # 2D/3D → 3D CHW
                # 对齐到本样本内统一 H/W 再 stack → [N,C,Hmax,Wmax]
                sizes = [im.shape[-2:] for im in imgs]
                max_h = max(h for h, w in sizes)
                max_w = max(w for h, w in sizes)
                padded = []
                for im in imgs:  # im: [C,H,W]
                    h, w = im.shape[-2:]
                    pad = (0, max_w - w, 0, max_h - h)  # W-left, W-right, H-top, H-bottom
                    padded.append(torch.nn.functional.pad(im, pad))
                return torch.stack(padded, dim=0)  # [N,C,max_h,max_w]

            else:
                raise ValueError("pixel_values must be Tensor or list/tuple of Tensors")

        if "pixel_values" in encoded[0]:
            per_sample_pv = []
            per_sample_mask = []

            # 每个样本规整到 [Ni,C,Hi,Wi]
            for e in encoded:
                pv = _to_4d_per_sample(e["pixel_values"])
                per_sample_pv.append(pv)
                per_sample_mask.append(torch.ones((pv.shape[0],), dtype=torch.bool, device=pv.device))

            if len(encoded) == 1:
                # 单样本：大多数 Qwen 实现接受 [N,C,H,W]
                batch["pixel_values"] = per_sample_pv[0]
                batch["pixel_values_mask"] = per_sample_mask[0]  # [N]
            else:
                # 多样本：pad 到统一 [B,Nmax,C,Hmax,Wmax]
                Ns = [pv.shape[0] for pv in per_sample_pv]
                Cs = [pv.shape[1] for pv in per_sample_pv]
                Hs = [pv.shape[2] for pv in per_sample_pv]
                Ws = [pv.shape[3] for pv in per_sample_pv]
                # 断言通道一致（前面已做 1→3 复制）
                C = Cs[0]
                assert all(c == C for c in Cs), f"Channel mismatch: {Cs}"
                Nmax, Hmax, Wmax = max(Ns), max(Hs), max(Ws)

                pv_batch = []
                mask_batch = []
                for pv, m in zip(per_sample_pv, per_sample_mask):
                    Ni, Ci, Hi, Wi = pv.shape
                    if Hi != Hmax or Wi != Wmax:
                        pv = torch.nn.functional.pad(pv, (0, Wmax - Wi, 0, Hmax - Hi))
                    if Ni < Nmax:
                        pad_imgs = torch.zeros((Nmax - Ni, C, Hmax, Wmax), dtype=pv.dtype, device=pv.device)
                        pv = torch.cat([pv, pad_imgs], dim=0)
                        m = torch.cat([m, torch.zeros((Nmax - Ni,), dtype=torch.bool, device=m.device)], dim=0)
                    pv_batch.append(pv.unsqueeze(0))   # [1,Nmax,C,Hmax,Wmax]
                    mask_batch.append(m.unsqueeze(0))  # [1,Nmax]
                batch["pixel_values"] = torch.cat(pv_batch, dim=0)        # [B,Nmax,C,Hmax,Wmax]
                batch["pixel_values_mask"] = torch.cat(mask_batch, dim=0)  # [B,Nmax]

            # 保险：同时给出 images 键（有些 forward 走 images）
            if "images" not in batch:
                batch["images"] = batch["pixel_values"]

            assert torch.is_tensor(batch["pixel_values"]) and batch["pixel_values"].dim() in (4, 5), \
                f"pixel_values must be 4D or 5D tensor, got {batch['pixel_values'].shape}"

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

    # ===== 模型 + 处理器（离线环境下请用本地路径 + local_files_only）=====
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
    prefer_short    = int(cfg.get("image_short_side", 896))  # 你 YAML 里当前是 1024

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
