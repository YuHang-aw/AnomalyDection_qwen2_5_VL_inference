from typing import Dict, Any
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor, AutoModelForVision2Seq

# 说明：不同版本的Qwen2.5-VL在HF类名可能略有差异，如遇问题可切换到AutoModelForCausalLM或官方类。

def load_model_and_processor(cfg: Dict[str, Any]):
    name = cfg["model_name"]
    proc = AutoProcessor.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        name,
        torch_dtype=("bf16"),
        trust_remote_code=True,
        device_map="auto"
    )
    return model, proc

# 冻结/LoRA注入（按phase配置）

def apply_freeze_and_lora(model, cfg: Dict[str, Any]):
    # 冻结视觉塔
    if cfg.get("freeze",{}).get("vision_tower", False) is True:
        if hasattr(model, "vision_tower"):
            for p in model.vision_tower.parameters():
                p.requires_grad = False
    elif cfg.get("freeze",{}).get("vision_tower") == "partial_unfreeze":
        # 仅示例：顶层block可通过名字匹配（需根据实际命名调整）
        top_n = int(cfg.get("vision",{}).get("top_blocks_unfreeze", 0))
        if hasattr(model, "vision_tower") and top_n>0:
            # 假设 model.vision_tower.blocks 是列表（具体按模型实现适配）
            blocks = getattr(model.vision_tower, "blocks", [])
            keep = set(id(b) for b in blocks[-top_n:])
            for b in blocks:
                req = (id(b) in keep)
                for p in b.parameters(): p.requires_grad = req

    # Projector 冻结与否
    if cfg.get("freeze",{}).get("projector", False):
        if hasattr(model, "mm_projector"):
            for p in model.mm_projector.parameters(): p.requires_grad = False

    # 语言侧控制："lora_only" 表示仅LoRA权重训练
    # LoRA 配置
    pconf = cfg.get("peft", {})
    if pconf.get("enable", False):
        if pconf.get("qlora", False):
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.get("optim",{}).get("grad_checkpointing", False))
        target = pconf.get("target_modules", [])
        lconf = LoraConfig(
            r=int(pconf.get("r",16)),
            lora_alpha=int(pconf.get("alpha",32)),
            lora_dropout=float(pconf.get("dropout",0.05)),
            target_modules=target,
            bias="none"
        )
        model = get_peft_model(model, lconf)
    return model
