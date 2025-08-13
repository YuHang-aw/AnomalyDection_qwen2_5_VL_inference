#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, argparse, math
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import EvalPrediction

from src.modeling.load_qwen_vl import load_model_and_processor, apply_freeze_and_lora
from datasets import load_dataset, Features, Sequence, Value

features = Features({
    "messages": Sequence({
        "role": Value("string"),
        "content": Sequence({
            "type":  Value("string"),
            "text":  Value("string"),  # 允许为 None
            "image": Value("string"),  # 允许为 None（仅保存路径）
        })
    })
})


# 简易collator：每步用processor把 messages+image 打成模型需要的 tensors
class VLDataCollator:
    def __init__(self, processor, max_length=4096):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples):
        chats = []
        from PIL import Image

        for ex in examples:
            mm_msgs = []
            for m in ex["messages"]:
                parts_out = []
                for p in m["content"]:
                    if p["type"] == "image":
                        # 把路径 -> PIL，供 processor 使用
                        img = Image.open(p["image"]).convert("RGB")
                        parts_out.append({"type":"image","image": img, "text": None})
                    else:
                        parts_out.append({"type":"text","text": p.get("text") or "", "image": None})
                mm_msgs.append({"role": m["role"], "content": parts_out})
            chats.append(mm_msgs)

        batch = self.processor.apply_chat_template(
            chats, add_generation_prompt=False, tokenize=True, return_tensors="pt"
        )
        return batch



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_config", required=True)
    args = ap.parse_args()

    # 读yaml
    import yaml
    with open(args.train_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 数据
    train_ds = load_dataset("json", data_files=cfg["data"]["train_jsonl"], split="train", features=features)
    val_ds   = load_dataset("json", data_files=cfg["data"]["val_jsonl"],   split="train", features=features)

    # 模型+处理器
    model, processor = load_model_and_processor(cfg)
    model = apply_freeze_and_lora(model, cfg)

    tr_args = cfg.get("trainer", {})
    ta = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["optim"]["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["optim"]["grad_accum"],
        learning_rate=cfg["optim"].get("lr_lm",1e-4),  # 统一lr；如需分组lr请见下方注
        num_train_epochs=cfg["optim"].get("epochs",2),
        warmup_steps=cfg["optim"].get("warmup_steps",200),
        weight_decay=cfg["optim"].get("weight_decay",0.0),
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
    )

    # 你也可以用 --resume 开启断点续训
    # python scripts/train.py --train_config configs/train_phase1.yaml --resume
    import sys
    resume = ("--resume" in sys.argv)

    trainer = Trainer(
        model=model,
        args=ta,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train(resume_from_checkpoint=resume)

if __name__ == "__main__":
    main()
