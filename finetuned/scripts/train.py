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
    def __init__(self, processor, max_length=4096, add_generation_prompt=False, label_pad_token_id=-100):
        self.processor = processor
        self.max_length = int(max_length)
        self.add_generation_prompt = add_generation_prompt
        self.label_pad_token_id = int(label_pad_token_id)

        tok = processor.tokenizer
        # 探测模型使用的“图像占位符”文本（Qwen2-VL 通常为 "<image>"）
        self.image_token_text = getattr(processor, "image_token", None)
        if not self.image_token_text:
            for t in (tok.special_tokens_map.get("additional_special_tokens", []) or []):
                if isinstance(t, str) and "image" in t.lower():
                    self.image_token_text = t
                    break
        if not self.image_token_text:
            self.image_token_text = "<image>"
        # 该占位符对应的 token 序列（通常 1 个 id，但也兼容多 id）
        self.image_token_ids = tok.encode(self.image_token_text, add_special_tokens=False)

    @staticmethod
    def _find_subseq_positions(seq, subseq):
        if not subseq:
            return []
        L, M = len(seq), len(subseq)
        if M == 1:
            tid = subseq[0]
            return [i for i, x in enumerate(seq) if x == tid]
        pos = []
        for i in range(L - M + 1):
            if seq[i:i+M] == subseq:
                pos.append(i)
        return pos

    def __call__(self, examples):
        from PIL import Image
        tok = self.processor.tokenizer

        # 1) 组装模板输入：文本里清洗 "<image>"，图片只放“占位”，真实 PIL 进 images 列表
        chats_for_template, images_batch = [], []
        for ex in examples:
            mm_tpl, imgs = [], []
            for m in ex["messages"]:
                parts_tpl = []
                for p in m["content"]:
                    if p["type"] == "image" and p.get("image"):
                        parts_tpl.append({"type": "image"})  # 只占位
                        imgs.append(Image.open(p["image"]).convert("RGB"))
                    else:
                        txt = (p.get("text") or "")
                        # 清洗用户文本里误写的占位符，避免被 tokenizer 当作图片标记
                        if self.image_token_text in txt:
                            txt = txt.replace(self.image_token_text, "〈image〉")
                        parts_tpl.append({"type": "text", "text": txt})
                mm_tpl.append({"role": m["role"], "content": parts_tpl})
            chats_for_template.append(mm_tpl)
            images_batch.append(imgs)

        # 2) 用 chat template 得到最终 prompt 文本（模板会按图片数量自动插入占位符）
        prompts = self.processor.apply_chat_template(
            chats_for_template, add_generation_prompt=self.add_generation_prompt, tokenize=False
        )

        # 2.1 快速字符串级一致性校验（文本中占位符个数应等于图片数）
        for i, (prompt, imgs) in enumerate(zip(prompts, images_batch)):
            if prompt.count(self.image_token_text) != len(imgs):
                raise ValueError(
                    f"[image-token-count-mismatch] sample#{i}: text has "
                    f"{prompt.count(self.image_token_text)} image markers but {len(imgs)} images."
                )

        # 3) 逐样本编码（不截断），如超长再做“安全截断”
        encoded_items = []
        for prompt, imgs in zip(prompts, images_batch):
            enc = self.processor(
                text=prompt, images=imgs,
                padding=False, truncation=False, return_tensors="pt"
            )

            ids = enc["input_ids"][0]
            am  = enc["attention_mask"][0]
            ids_list = ids.tolist()

            # token 级再次校验占位符数量
            img_pos = self._find_subseq_positions(ids_list, self.image_token_ids)
            if len(img_pos) != len(imgs):
                raise ValueError("Encoded input_ids image markers != images; check data/template.")

            L = ids.shape[-1]
            if L > self.max_length:
                first_img = img_pos[0]
                last_img_end = img_pos[-1] + len(self.image_token_ids)

                # 若仅包含所有图片标记的区间都 > max_length，则此样本无法安全截断
                if (last_img_end - first_img) > self.max_length:
                    raise ValueError(
                        "Sequence too long to keep all image tokens within max_length; "
                        "shorten upstream text or increase max_length."
                    )

                # 优先保留序列尾部；若会切掉第一枚图片标记，则左界对齐到第一枚图片标记
                left = L - self.max_length
                if left > first_img:
                    left = first_img
                right = left + self.max_length  # <= L（或等于）
                ids = ids[left:right]
                am  = am[left:right]

                # 再确认图片标记仍完整保留
                ids_after = ids.tolist()
                img_pos_after = self._find_subseq_positions(ids_after, self.image_token_ids)
                if len(img_pos_after) != len(imgs):
                    raise ValueError("Unsafe truncation removed image tokens; adjust max_length or truncate earlier.")

                enc["input_ids"] = ids.unsqueeze(0)
                enc["attention_mask"] = am.unsqueeze(0)

            encoded_items.append(enc)

        # 4) 手动右侧 padding → batch，对 labels 的 padding 置 -100
        max_len = max(e["input_ids"].shape[-1] for e in encoded_items)
        pad_id = tok.pad_token_id
        input_ids_list, attn_list = [], []
        for e in encoded_items:
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

        # 5) 像素张量组织（你的 batch_size=1 时无需关心多样本对齐）
        if "pixel_values" in encoded_items[0]:
            if len(encoded_items) == 1:
                batch["pixel_values"] = encoded_items[0]["pixel_values"]
            else:
                # 多样本情况下尝试堆叠；若每样本图片数不同，可改为列表喂给模型（取决于具体实现）
                pvs = [e["pixel_values"] for e in encoded_items]
                try:
                    batch["pixel_values"] = torch.stack(pvs, dim=0)
                except Exception:
                    # 回退：保留为列表，模型需支持
                    batch["pixel_values"] = pvs

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
        max_length=int(cfg.get("max_seq_len", 4096)),
        add_gen_prompt=False,
        label_pad_token_id=-100
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
