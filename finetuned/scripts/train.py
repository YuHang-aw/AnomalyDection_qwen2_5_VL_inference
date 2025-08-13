#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, argparse
from pathlib import Path
import torch
from datasets import load_dataset, Features, Sequence, Value
from transformers import TrainingArguments, Trainer, set_seed

from src.modeling.load_qwen_vl import load_model_and_processor, apply_freeze_and_lora

# =========== 明确 schema，避免 Arrow 误判 ===========
features = Features({
    "messages": Sequence({
        "role": Value("string"),
        "content": Sequence({
            "type":  Value("string"),
            "text":  Value("string"),  # 若数据里有 null，建议先用修复脚本改成 ""
            "image": Value("string"),  # 图片路径或 ""（无图）
        })
    })
})

# =========== Collator：模板→文本，占位→真实图片 ===========
class VLDataCollator:
    """
    最小不定修正版：
    - 逐样本 encode（truncation=False）
    - 若超长：滑窗选取长度=max_length 的片段，**保证完整覆盖所有视觉块**
    - 截断后再次校验 image_token 计数是否不变
    - 文本里误出现 "<image>" 可选清洗为 "〈image〉"（避免干扰）
    """
    def __init__(self, processor, model_config, max_length=4096, add_generation_prompt=False, label_pad_token_id=-100, sanitize_user_image_token=True):
        self.processor = processor
        self.cfg = model_config
        self.max_length = int(max_length)
        self.add_generation_prompt = bool(add_generation_prompt)  # 训练阶段建议 False
        self.label_pad_token_id = int(label_pad_token_id)
        self.sanitize_user_image_token = bool(sanitize_user_image_token)

        tok = processor.tokenizer
        # 从 config 读取三类关键 token id（Qwen2/2.5-VL 标准字段）
        self.image_token_id = getattr(self.cfg, "image_token_id", tok.convert_tokens_to_ids("<|image_pad|>"))
        self.vision_start_id = getattr(self.cfg, "vision_start_token_id", tok.convert_tokens_to_ids("<|vision_start|>"))
        self.vision_end_id   = getattr(self.cfg, "vision_end_token_id",   tok.convert_tokens_to_ids("<|vision_end|>"))

        # 允许安全清洗用户文本里误写的 "<image>"
        self._user_image_literal = "<image>"
        self._user_image_safe = "〈image〉"

    def _find_blocks(self, ids):
        """
        返回所有视觉块 [start_idx, end_idx]（闭区间），并统计其中 image_token 数量。
        若出现起止不配对，返回空列表。
        """
        starts, ends = [], []
        for i, t in enumerate(ids):
            if t == self.vision_start_id: starts.append(i)
            elif t == self.vision_end_id: ends.append(i)
        if not starts and not ends:
            return [], 0  # 无视觉块
        if len(starts) != len(ends):
            return [], -1  # 异常

        blocks = []
        total_img_tokens = 0
        si = 0
        for ei in range(len(ends)):
            # 假设严格配对且顺序正确（Qwen 官方模板会保证）
            s = starts[si]
            e = ends[ei]
            if s > e:  # 不合法
                return [], -1
            blocks.append((s, e))
            total_img_tokens += sum(1 for t in ids[s:e+1] if t == self.image_token_id)
            si += 1
        return blocks, total_img_tokens

    def __call__(self, examples):
        from PIL import Image
        import torch
        tok = self.processor.tokenizer

        # 1) 组织模板输入：图片只放“占位”，真实 PIL 交给 processor；可选清洗用户文本里的 "<image>"
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

        # 2) 逐样本 encode（不截断）
        encoded = []
        for prompt, imgs in zip(prompts, images_batch):
            enc = self.processor(text=prompt, images=imgs, padding=False, truncation=False, return_tensors="pt")
            ids = enc["input_ids"][0]
            am  = enc["attention_mask"][0]
            ids_list = ids.tolist()

            # 2.1 定位视觉块并统计 image_token（原始计数）
            blocks, img_tok_cnt = self._find_blocks(ids_list)
            if img_tok_cnt < 0:
                raise ValueError("Malformed vision token blocks in encoded input; check template & data.")
            # —— 注意：这里**不再**用“文本里数占位符”的办法，因为 Qwen2.5-VL 一图 = 多个 image_token

            # 3) 超长则“视觉块优先保留”式截断
            L = ids.shape[-1]
            if L > self.max_length and blocks:
                # 需要一个窗口 [left, right) 长度为 max_length，完全覆盖所有视觉块
                min_keep = min(s for s, _ in blocks)
                max_keep = max(e for _, e in blocks) + 1  # 右开
                need = max_keep - min_keep
                if need > self.max_length:
                    # 单是视觉块就超过了 max_length —— 上游必须缩短文本或下调分辨率
                    raise ValueError(
                        f"Visual token span ({need}) > max_length ({self.max_length}). "
                        f"Reduce text or set smaller image pixels (min/max_pixels)."
                    )
                # 优先尽量保留尾部的对话；先给出一个候选左界
                left = max(0, L - self.max_length)
                # 确保窗口覆盖视觉块
                if left > min_keep:
                    left = min_keep
                if left + self.max_length < max_keep:
                    left = max(0, max_keep - self.max_length)
                right = left + self.max_length
                ids = ids[left:right]
                am  = am[left:right]
                ids_list = ids.tolist()

                # 3.1 截断后再次统计 image_token，必须与截断前一致
                _, img_tok_cnt_after = self._find_blocks(ids_list)
                if img_tok_cnt_after != img_tok_cnt:
                    raise ValueError(
                        "Unsafe truncation removed part of visual tokens. Increase max_length, "
                        "shorten text, or reduce image pixels."
                    )
                enc["input_ids"] = ids.unsqueeze(0)
                enc["attention_mask"] = am.unsqueeze(0)

            encoded.append(enc)

        # 4) 右侧 padding 成 batch，并生成 labels（pad→-100）
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

        # 5) pixel_values（若存在）直接带上
        if "pixel_values" in encoded[0]:
            if len(encoded) == 1:
                batch["pixel_values"] = encoded[0]["pixel_values"]
            else:
                try:
                    import torch
                    batch["pixel_values"] = torch.stack([e["pixel_values"] for e in encoded], dim=0)
                except Exception:
                    batch["pixel_values"] = [e["pixel_values"] for e in encoded]

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
        features=features
    )
    val_ds   = load_dataset("json",
        data_files=cfg["data"]["val_jsonl"],
        split="train",
        features=features
    )

    # ===== 模型 + 处理器（离线环境下请用本地路径 + local_files_only）=====
    model, processor = load_model_and_processor(cfg)
    model = apply_freeze_and_lora(model, cfg)

    # ===== Collator 实例 =====
    collator = VLDataCollator(
        processor=processor,
        model_config=model.config,
        max_length=4096,                # 保证 ≥ 文本tokens + 全部视觉tokens
        add_generation_prompt=False     # 训练阶段固定 False
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
        eva_strategy=tr_args.get("evaluation_strategy","steps"),
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
