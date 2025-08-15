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
    整体完善的、回归标准的版本 (v12):
    - 彻底摒弃在循环中单次调用 processor 的错误模式。
    - 遵循 Hugging Face 标准实践：先收集批次中所有的文本和图像，然后进行“单次批处理调用”。
    - 充分利用 processor 的强大功能，让其自动处理批次内的 padding、truncation 和视觉特征对齐。
    - 删除了所有不必要的、易错的手动打包、维度检查和元数据拼接逻辑。
    - 代码更简洁、效率更高，并且从根本上避免了因误用 processor 导致的各种维度和对齐错误。
    """
    def __init__(self, processor, max_length=4096, label_pad_token_id=-100):
        self.processor = processor
        self.max_length = int(max_length)
        self.label_pad_token_id = int(label_pad_token_id)

    def __call__(self, examples):
        from PIL import Image
        
        # 1. 准备数据：将所有样本的文本和图像收集到列表中
        batch_prompts = []
        batch_images = []
        
        for ex in examples:
            # 简化逻辑：我们假设每个样本都符合 '一图一问' 的格式
            try:
                prompt = self.processor.apply_chat_template(
                    ex["messages"], add_generation_prompt=False, tokenize=False
                )
                
                # 找到并加载图像
                img_path = None
                for msg in ex["messages"]:
                    for content in msg["content"]:
                        if content["type"] == "image":
                            img_path = content.get("image")
                            break
                    if img_path:
                        break
                
                if prompt and img_path:
                    image = Image.open(img_path).convert("RGB")
                    batch_prompts.append(prompt)
                    batch_images.append(image)
            except Exception as e:
                # 如果某个样本格式有问题或图片无法加载，打印警告并跳过
                print(f"\n[DataCollator Warning] Skipping a malformed sample. Error: {e}\n")
                continue

        if not batch_prompts or not batch_images:
            return {}

        # 2. 单次批处理调用：让 processor 处理整个批次
        try:
            inputs = self.processor(
                text=batch_prompts,
                images=batch_images,
                return_tensors="pt",
                padding="longest", # 自动填充到批次中的最大长度
                truncation=True,
                max_length=self.max_length,
            )
        except Exception as e:
            print(f"\n[DataCollator Error] Processor failed to handle the batch. Error: {e}\n")
            return {}

        # 3. 创建 labels
        # processor 已经为我们处理好了 padding，我们只需复制 input_ids 并替换 padding token
        labels = inputs.input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.label_pad_token_id
        inputs["labels"] = labels

        return inputs


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
    collator = VLDataCollator(
        processor=processor,
        max_length=max_seq_len,
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
        # remove_unused_columns=False, # 当 collator 返回标准 Hugging Face 输出时，可以不设置此项
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
