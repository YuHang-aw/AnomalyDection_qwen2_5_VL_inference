# src/modeling/load_qwen_vl.py
import os, torch
from typing import Tuple
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_model_and_processor(cfg) -> Tuple[torch.nn.Module, object]:
    name = cfg["model_name"]
    trust = True
    local_only = bool(int(os.environ.get("HF_HUB_OFFLINE", "0"))) or os.path.isdir(name)

    # --- 处理器 ---
    processor = AutoProcessor.from_pretrained(
        name, trust_remote_code=trust, local_files_only=local_only
    )

    # --- QLoRA: 4-bit 量化配置 ---
    peft_cfg = cfg.get("peft", {})
    use_qlora = bool(peft_cfg.get("qlora", False))
    bnb_config = None
    torch_dtype = torch.bfloat16 if str(cfg.get("precision","bf16")).lower()=="bf16" else torch.float16
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # --- 模型加载（优先 Vision2Seq，失败降级到 CausalLM） ---
    common_kwargs = dict(trust_remote_code=trust, local_files_only=local_only)
    try:
        if use_qlora:
            model = AutoModelForCausalLM.from_pretrained(
                name, quantization_config=bnb_config, **common_kwargs
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch_dtype, device_map="auto", **common_kwargs
            )
    except Exception:
        if use_qlora:
            model = AutoModelForVision2Seq.from_pretrained(
                name, quantization_config=bnb_config, **common_kwargs
            )
        else:
            model = AutoModelForVision2Seq.from_pretrained(
                name, torch_dtype=torch_dtype, device_map="auto", **common_kwargs
            )

    # 训练相关的小开关
    model.config.use_cache = False  # 配合梯度检查点
    return model, processor


def apply_freeze_and_lora(model, cfg):
    peft_cfg = cfg.get("peft", {})
    freeze_cfg = cfg.get("freeze", {})
    use_qlora = bool(peft_cfg.get("qlora", False))

    # === 0) QLoRA 前置准备（必须在 LoRA 前执行） ===
    if use_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg["optim"].get("grad_checkpointing", False),
        )
        # 很多架构需要这句来给量化层输入建图
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # === 1) 先把全模型冻结（LoRA 模块随后插入，默认可训练） ===
    for p in model.parameters():
        p.requires_grad = False

    # === 2) 插入 LoRA（按你的 target_modules） ===
    target = peft_cfg.get("target_modules", [
        "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"
    ])
    lconf = LoraConfig(
        r=int(peft_cfg.get("r", 16)),
        lora_alpha=int(peft_cfg.get("alpha", 32)),
        lora_dropout=float(peft_cfg.get("dropout", 0.05)),
        target_modules=target,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lconf)

    # === 3) 按 freeze 配置解/冻结 ===
    # 3.1 视觉塔
    if bool(freeze_cfg.get("vision_tower", True)):
        # 已经全量冻结，无需动作；这里留空只是表义
        pass
    else:
        for n, p in model.named_parameters():
            if any(k in n.lower() for k in ["vision_tower","vision_model","visual","clip"]):
                p.requires_grad = True

    # 3.2 projector
    if not bool(freeze_cfg.get("projector", False)):
        for n, p in model.named_parameters():
            if "mm_projector" in n or "multi_modal_projector" in n:
                p.requires_grad = True

    # 3.3 语言侧：lora_only / full / none
    lang_mode = str(freeze_cfg.get("language", "lora_only")).lower()
    if lang_mode == "full":
        # 语言模型全部可训练（包括 embedding/MLP 等）
        for n, p in model.named_parameters():
            if any(k in n for k in ["model.embed_tokens", "model.layers", "lm_head"]):
                p.requires_grad = True
    elif lang_mode == "none":
        # 什么都不解冻（只靠 projector 或视觉等）
        pass
    else:
        # lora_only：默认状态（全冻 + LoRA 可训练），无需额外动作
        pass

    # === 4) 梯度检查点（放最后更稳） ===
    if cfg["optim"].get("grad_checkpointing", False):
        model.gradient_checkpointing_enable()

    # === 5) 安全断言 & 打印 ===
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trn == 0:
        raise RuntimeError("No trainable parameters. Check freeze.* / peft.target_modules / QLoRA prepare order.")
    print(f"[PEFT] Trainable params: {trn/1e6:.2f}M / {tot/1e6:.2f}M")
    for n, p in model.named_parameters():
        if p.requires_grad and ("lora" in n or "mm_projector" in n):
            print("  ✓", n)
            break
    return model
