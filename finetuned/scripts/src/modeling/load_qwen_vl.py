from typing import Dict, Any
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor, AutoModelForVision2Seq

def load_model_and_processor(cfg):
    name = cfg["model_name"]  # 本地路径
    proc = AutoProcessor.from_pretrained(
        name, trust_remote_code=True, local_files_only=True
    )
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            name,
            torch_dtype="bfloat16",
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto",
        )
    except Exception:
        # 少数版本用 CausalLM 也能跑
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype="bfloat16",
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto",
        )
    return model, proc


# 冻结/LoRA注入（按phase配置）

def apply_freeze_and_lora(model, cfg):
    peft_cfg = cfg.get("peft", {})
    qlora = bool(peft_cfg.get("qlora", False))
    r = int(peft_cfg.get("r", 16))
    alpha = int(peft_cfg.get("alpha", 32))
    dropout = float(peft_cfg.get("dropout", 0.05))

    # 1) QLoRA 前置准备（4-bit/8-bit 才需要）
    if qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg["optim"].get("grad_checkpointing", False)
        )

    # 2) 先把全模型冻结（LoRA 是新加的、默认可训练）
    for n, p in model.named_parameters():
        p.requires_grad = False

    # 3) 插入 LoRA —— 只命中语言侧线性层；需要再训 projector 的话额外解冻
    target = peft_cfg.get("target_modules", [
        "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"
    ])
    lconf = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target,
        bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lconf)

    # 4) （可选）解冻多模态投影/视觉侧
    if cfg.get("train_projector", False):
        for n, p in model.named_parameters():
            if "mm_projector" in n:
                p.requires_grad = True

    # 5) 开启梯度检查点（若需要）
    if cfg["optim"].get("grad_checkpointing", False):
        model.gradient_checkpointing_enable()

    # 6) 断言：至少有若干参数在训练
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError("No trainable parameters after applying LoRA. Check target_modules / freezing order.")
    return model
