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
    def __init__(self, processor, max_length=4096, add_gen_prompt=False, label_pad_token_id=-100):
        self.processor = processor
        self.max_length = max_length
        self.add_gen_prompt = add_gen_prompt
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, examples):
        from PIL import Image

        chats_for_template = []   # 传给 apply_chat_template（不带真实图片）
        images_batch = []         # 每条样本的图片列表（真实 PIL）
        for ex in examples:
            mm_msgs_tpl = []
            imgs = []
            for m in ex["messages"]:
                parts_tpl = []
                for p in m["content"]:
                    if p["type"] == "image" and p.get("image"):
                        # 模板里只要占位，不传大对象
                        parts_tpl.append({"type": "image"})
                        # 真实图片进 images
                        img = Image.open(p["image"]).convert("RGB")
                        imgs.append(img)
                    else:
                        parts_tpl.append({"type": "text", "text": (p.get("text") or "")})
                mm_msgs_tpl.append({"role": m["role"], "content": parts_tpl})
            chats_for_template.append(mm_msgs_tpl)
            images_batch.append(imgs)

        # 1) 先用模板生成文本（含 <image> 占位）
        prompts = self.processor.apply_chat_template(
            chats_for_template,
            add_generation_prompt=self.add_gen_prompt,
            tokenize=False
        )

        # 2) 再把文本 + 真实图片一起编码成张量
        enc = self.processor(
            text=prompts,
            images=images_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 3) 监督学习标签：等于 input_ids，padding 处置 -100
        labels = enc["input_ids"].clone()
        if "attention_mask" in enc:
            labels[enc["attention_mask"] == 0] = self.label_pad_token_id
        enc["labels"] = labels
        return enc


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
