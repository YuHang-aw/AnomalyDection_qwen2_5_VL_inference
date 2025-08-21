import torch
import torch.nn as nn

try:
    from torchvision.ops.misc import FrozenBatchNorm2d
except Exception:
    class FrozenBatchNorm2d(nn.Module): pass  # 兜底

def _module_device_dtype(m: nn.Module):
    for t in list(m.parameters()) + list(m.buffers()):
        return t.device, t.dtype
    return torch.device("cpu"), torch.float32

@torch.no_grad()
def convert_frozen_bn_to_bn(module: nn.Module):
    """
    递归把 FrozenBatchNorm2d 换成 BatchNorm2d，并把“新 BN”放到旧层相同的 device/dtype。
    """
    for name, child in list(module.named_children()):
        if isinstance(child, FrozenBatchNorm2d):
            dev, dtype = _module_device_dtype(child)
            bn = nn.BatchNorm2d(
                num_features=child.num_features,
                eps=getattr(child, "eps", 1e-5),
                momentum=0.1, affine=True, track_running_stats=True
            ).to(device=dev, dtype=dtype)

            for dst, src in [("weight","weight"),("bias","bias"),
                             ("running_mean","running_mean"),("running_var","running_var")]:
                if hasattr(child, src) and hasattr(bn, dst):
                    getattr(bn, dst).data.copy_(getattr(child, src).data.to(device=dev, dtype=dtype))

            if hasattr(bn, "num_batches_tracked"):
                bn.num_batches_tracked.zero_()
            bn.eval()
            setattr(module, name, bn)
        else:
            convert_frozen_bn_to_bn(child)



import math

def _is_depthwise(m: nn.Conv2d):
    return isinstance(m, nn.Conv2d) and m.groups == m.in_channels == m.out_channels

def lcm(a, b): return abs(a*b) // math.gcd(a, b) if a and b else max(a, b)

def detect_align_multiple(model: nn.Module, base_round=8):
    """
    返回一个“全局对齐倍数”，保证通道数永远是：
      L = lcm( base_round, 所有 Conv2d.groups(>1), 所有 PixelShuffle(r)^2 )
    """
    L = base_round
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.groups > 1:
            L = lcm(L, m.groups)
        if isinstance(m, nn.PixelShuffle):
            L = lcm(L, m.upscale_factor * m.upscale_factor)
    return max(L, 1)

def collect_ignored_layers_keep_backbone_convs(full: nn.Module):
    """
    允许：backbone 里的 Conv2d（包括分组/深度可分离）被联动裁剪
    忽略：检测头/transformer/cls/bbox 等，以及 out=1/3 的输出卷积
    —— 注意：不再因为 groups>1 就忽略。让依赖修复能触达这些层。
    """
    allowed = set()
    if hasattr(full, "backbone"):
        for m in full.backbone.modules():
            if isinstance(m, nn.Conv2d):
                allowed.add(m)

    ignored = []
    ban_keywords = ("transformer", "encoder", "decoder", "attn", "attention",
                    "head", "bbox", "cls", "query", "dn", "matcher", "postprocessor")
    for name, m in full.named_modules():
        if isinstance(m, nn.Conv2d):
            low = name.lower()
            if (m.out_channels in (1, 3)) or any(k in low for k in ban_keywords):
                if m not in ignored:
                    ignored.append(m)
    # 注意：不要把 groups>1 的 conv 放进 ignored！
    return ignored


import copy
import torch
import torch.nn as nn
import torch_pruning as tp

class BackboneOnly(nn.Module):
    def __init__(self, full):
        super().__init__()
        self.full = full
    def forward(self, x):
        return self.full.backbone(x)

def val_prune(self, need_json=False):
    self.eval()
    device = self.device
    full = self.ema.module if self.ema else self.model

    # 深拷贝“完整模型”并放到 CUDA
    pruned_model = copy.deepcopy(full).to(device).eval()

    # 仅转换 backbone 里的 FrozenBN（保持 device/dtype）
    if hasattr(pruned_model, "backbone"):
        convert_frozen_bn_to_bn(pruned_model.backbone)

    # 构造 4D CUDA dummy（只给 backbone 用；不需要 labels）
    H, W = getattr(self.cfg, "val_input_size", (640, 640))
    example_inputs = torch.randn(1, 3, H, W, device=device).float()

    # 只忽略检测头/Transformer 等；允许分组卷积被联动裁剪
    ignored_layers = collect_ignored_layers_keep_backbone_convs(pruned_model)

    # 全局对齐倍数：包含所有 groups 与 PixelShuffle 约束
    round_to = detect_align_multiple(
        pruned_model,
        base_round=getattr(self.cfg, "val_prune_round_to", 8)
    )

    pruning_ratio = getattr(self.cfg, "val_prune_ratio", 0.30)
    imp = tp.importance.GroupMagnitudeImportance(p=2)

    # 让 pruner 在构图阶段只跑 backbone(x)，但“建图对象”仍是完整模型（这样 encoder/decoder 属性健在）
    def forward_fn_runs_backbone(m: nn.Module, x: torch.Tensor):
        return m.backbone(x)

    pruned_model.float()  # 构图用 fp32 更稳
    pruner = tp.pruner.MetaPruner(
        pruned_model,                       # 传完整模型！
        example_inputs=example_inputs,
        importance=imp,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,      # 只忽略头/transformer/out=1/3
        global_pruning=True,
        isomorphic=True,
        round_to=round_to,                  # 关键：含所有分组因子
        forward_fn=forward_fn_runs_backbone,
    )

    # （你的 tp 版本不支持 forward_fn 参数统计）——用包装器来统计 backbone 的 MAC/参数
    backbone_view = BackboneOnly(pruned_model).to(device).eval().float()
    base_macs, base_params = tp.utils.count_ops_and_params(backbone_view, example_inputs)
    print(f"[VAL-PRUNE][BASE] {base_macs/1e9:.3f}G MACs, {base_params/1e6:.3f}M params")

    pruner.step()  # ★ 不要包 no_grad

    macs, params = tp.utils.count_ops_and_params(backbone_view, example_inputs)
    print(f"[VAL-PRUNE][PRUNED] {macs/1e9:.3f}G ({macs/base_macs:.3f}x), {params/1e6:.3f}M")

    # 健康检查：只跑 backbone 一次
    with torch.no_grad():
        _ = backbone_view(example_inputs)

    # 用“剪过的完整模型”评估（need_json 保留）
    test_stats, coco_evaluator = evaluate(
        pruned_model,
        self.criterion,
        self.postprocessor,
        self.val_dataloader,
        self.evaluator,
        device,
        epoch=-1,
        use_wandb=False,
        need_json=need_json,
        output_dir=self.output_dir,
    )

    if self.output_dir:
        dist_utils.save_on_master(
            coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
        )
    return
