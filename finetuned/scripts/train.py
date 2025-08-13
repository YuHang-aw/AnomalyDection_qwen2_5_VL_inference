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

        # 找到 image 占位符在文本里的写法与其 token 序列
        # Qwen2-VL 通常是 "<image>"，但稳妥起见做自动探测
        tok = processor.tokenizer
        self.image_token_text = getattr(processor, "image_token", None)
        if not self.image_token_text:
            # 从附加特殊 token 里找包含 "image" 的那个
            for t in (tok.special_tokens_map.get("additional_special_tokens", []) or []):
                if "image" in t.lower():
                    self.image_token_text = t
                    break
        if not self.image_token_text:
            # 最后兜底：直接用常见写法
            self.image_token_text = "<image>"
        # 该占位符对应的 token 序列（一般是 1 个 id；万一是多 id 也支持）
        self.image_token_ids = tok.encode(self.image_token_text, add_special_tokens=False)

    @staticmethod
    def _find_subseq_positions(seq, subseq):
        """返回 subseq 在 seq 中的起始索引列表（允许 subseq 长度为 1）。"""
        if not subseq:
            return []
        L, M = len(seq), len(subseq)
        if M == 1:
            tid = subseq[0]
            return [i for i,x in enumerate(seq) if x == tid]
        pos = []
        for i in range(L - M + 1):
            if seq[i:i+M] == subseq:
                pos.append(i)
        return pos

    def __call__(self, examples):
        from PIL import Image
        tok = self.processor.tokenizer

        # 1) 先把多模态消息 -> 模板文本（只放占位符，不放真实图片）
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
                        parts_tpl.append({"type": "text", "text": p.get("text") or ""})
                mm_tpl.append({"role": m["role"], "content": parts_tpl})
            chats_for_template.append(mm_tpl)
            images_batch.append(imgs)

        prompts = self.processor.apply_chat_template(
            chats_for_template, add_generation_prompt=self.add_gen_prompt, tokenize=False
        )

        # 2) 逐样本编码（不截断），如超长则做“安全左截断”，确保不丢任何 <image> 标记
        encoded_items = []
        for prompt, imgs in zip(prompts, images_batch):
            # 编码：绝对不要在这里 truncation=True
            enc = self.processor(
                text=prompt, images=imgs,
                padding=False, truncation=False, return_tensors="pt"
            )
            input_ids = enc["input_ids"][0]
            attn = enc["attention_mask"][0]
            ids_list = input_ids.tolist()

            # 统计该样本里出现了多少个 <image> 标记（按 token 序列找）
            img_pos = self._find_subseq_positions(ids_list, self.image_token_ids)
            # 一致性快速自检：文本里图片占位符个数应与 imgs 数量一致
            if len(img_pos) != len(imgs):
                # 有些模板会在系统/assistant里也放占位，这里给个更友好的报错
                raise ValueError(f"[image-token-count-mismatch] text has {len(img_pos)} image markers "
                                 f"but got {len(imgs)} images. Prompt snippet: {prompt[:160]} ...")

            L = input_ids.shape[-1]
            if L > self.max_length:
                # 只从左侧截断；且必须保证“第一枚 <image> 标记仍被保留”
                # 目标起点：右对齐剪裁
                start = L - self.max_length
                if img_pos:  # 确保不会把最早的 image token 切没了
                    first_img = img_pos[0]
                    start = min(start, first_img)  # 不能越过第一枚图像 token
                    if start > first_img:
                        start = first_img

                # 真正裁剪
                input_ids = input_ids[start:]
                attn = attn[start:]

                # （可选）如果你希望进一步严格：再次确认裁剪后 image token 还在
                ids_after = input_ids.tolist()
                img_pos_after = self._find_subseq_positions(ids_after, self.image_token_ids)
                if len(img_pos_after) != len(imgs):
                    # 若出现这一行，说明样本太长且 image token 很靠左。你可以选择丢弃这个样本，
                    # 或者先在上游把文本（如 <regions>）做更强的截断。
                    raise ValueError("Unsafe truncation removed image tokens; increase max_length or pre-truncate text.")

                enc["input_ids"] = input_ids.unsqueeze(0)
                enc["attention_mask"] = attn.unsqueeze(0)
                # 像 pixel_values / pixel_attention（若有）不需要改，它们与图片个数相关，不跟 token 位移相关

            encoded_items.append(enc)

        # 3) 手动对齐成 batch（右侧 padding）
        # 文本部分
        max_len = max(e["input_ids"].shape[-1] for e in encoded_items)
        pad_id = tok.pad_token_id
        input_ids_list, attn_list, labels_list = [], [], []

        for e in encoded_items:
            ids = e["input_ids"][0]
            am = e["attention_mask"][0]
            pad_needed = max_len - ids.shape[-1]
            if pad_needed > 0:
                ids = torch.nn.functional.pad(ids, (0, pad_needed), value=pad_id)
                am  = torch.nn.functional.pad(am,  (0, pad_needed), value=0)
            input_ids_list.append(ids)
            attn_list.append(am)

        input_ids = torch.stack(input_ids_list, dim=0)
        attention_mask = torch.stack(attn_list, dim=0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = self.label_pad_token_id

        # 图像部分：按 processor 的返回组织（通常是 [B, N_img, 3, H, W] 或 list）
        # 这里兼容几种返回风格
        if "pixel_values" in encoded_items[0]:
            # 统一堆叠到 batch 维
            pix = []
            for e in encoded_items:
                # e["pixel_values"] 可能是 [N, 3, H, W] 或 list([3,H,W], ...)
                pv = e["pixel_values"]
                if isinstance(pv, torch.Tensor):
                    pix.append(pv)        # [N, 3, H, W]
                else:
                    pix.append(torch.stack(pv, dim=0))
            # 尽量不做 padding；大多数情况下 N_img 一致（你的数据一般 1）
            pixel_values = torch.stack(pix, dim=0)  # [B, N, 3, H, W]
        else:
            pixel_values = None

        batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values
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
