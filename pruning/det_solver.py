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


def collect_ignored_layers_keep_backbone_convs(full: nn.Module):
    """
    允许：backbone 里的所有 Conv2d（含分组/深度可分离）参与联动裁剪
    忽略：检测头/transformer/cls/bbox/query 等；以及 out=1/3 的输出卷积（常见最后输出）
    """
    ignored = []
    ban_keywords = ("transformer", "encoder", "decoder", "attn", "attention",
                    "head", "bbox", "cls", "query", "dn", "matcher", "postprocessor")
    for name, m in full.named_modules():
        if isinstance(m, nn.Conv2d):
            low = name.lower()
            if any(k in low for k in ban_keywords) or (m.out_channels in (1, 3)):
                ignored.append(m)
    return ignored

import copy
import torch
import torch.nn as nn
import torch_pruning as tp

class BackboneOnly(nn.Module):
    def __init__(self, full): super().__init__(); self.full = full
    def forward(self, x): return self.full.backbone(x)

def val_prune(self, need_json=False):
    self.eval()
    device = self.device
    full = self.ema.module if self.ema else self.model

    # 1) 深拷贝完整模型，放到 CUDA
    pruned_model = copy.deepcopy(full).to(device).eval()

    # 2) 只转换 backbone 的 FrozenBN -> BN（保持 device/dtype）
    if hasattr(pruned_model, "backbone"):
        convert_frozen_bn_to_bn(pruned_model.backbone)

    # 3) 构造 4D CUDA dummy（只用于构图/剪枝）
    H, W = getattr(self.cfg, "val_input_size", (640, 640))
    example_inputs = torch.randn(1, 3, H, W, device=device).float()

    # 4) 只忽略头/transformer；允许分组/深度可分离卷积联动
    ignored_layers = collect_ignored_layers_keep_backbone_convs(pruned_model)

    # 5) 放宽对齐，确保“剪得动”；后续再按需调大（8/16）
    round_to = getattr(self.cfg, "val_prune_round_to", 1)
    pruning_ratio = getattr(self.cfg, "val_prune_ratio", 0.30)

    # 6) 用 BNScaleImportance；构图只跑 backbone
    imp = tp.importance.BNScaleImportance() if any(isinstance(m, nn.BatchNorm2d) for m in pruned_model.modules()) \
          else tp.importance.GroupMagnitudeImportance(p=2)

    def fwd_backbone(m: nn.Module, x: torch.Tensor):
        return m.backbone(x)

    pruned_model.float()  # 构图更稳
    pruner = tp.pruner.MetaPruner(
        pruned_model,                   # 传“完整模型”！
        example_inputs=example_inputs,
        importance=imp,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        global_pruning=True,
        isomorphic=True,               # 有残差分支时建议 True
        round_to=round_to,             # 先 1，确认能剪，再加约束
        forward_fn=fwd_backbone,
    )

    # —— 剪前统计（不传 forward_fn；用包装器）——
    backbone_view = BackboneOnly(pruned_model).to(device).eval().float()
    base_macs, base_params = tp.utils.count_ops_and_params(backbone_view, example_inputs)

    # 记录可剪层以便确认是否“剪得动”
    try:
        prunable = getattr(pruner, "prunable_layers", None)
    except Exception:
        prunable = None
    if prunable is not None:
        print(f"[VAL-PRUNE] prunable conv layers: {len(prunable)}")
        if len(prunable) == 0:
            print("[VAL-PRUNE] WARNING: no prunable layers detected in backbone. Check ignored_layers/forward_fn.")

    # 真正执行剪枝（不要包 no_grad）
    pruner.step()

    macs, params = tp.utils.count_ops_and_params(backbone_view, example_inputs)
    print(f"[VAL-PRUNE] MACs {base_macs/1e9:.3f}G -> {macs/1e9:.3f}G "
          f"({macs/max(base_macs,1):.3f}x), Params {base_params/1e6:.3f}M -> {params/1e6:.3f}M")

    # 粗略统计：哪些层发生了 out_channels 变化（确认“确实剪到”）
    changed = []
    for name, m in pruned_model.backbone.named_modules():
        if isinstance(m, nn.Conv2d):
            # 用 module 原始拷贝对比
            m0 = dict(full.backbone.named_modules()).get(name, None)
            if isinstance(m0, nn.Conv2d) and m0.out_channels != m.out_channels:
                changed.append((name, m0.out_channels, m.out_channels, getattr(m, "groups", 1)))
    print(f"[VAL-PRUNE] pruned convs: {len(changed)}")
    for n, c0, c1, g in changed[:10]:
        print(f"  - {n}: {c0}->{c1}, groups={g}")
    if not changed:
        print("[VAL-PRUNE] WARNING: no conv actually pruned. Try smaller round_to (1) / 更小 ignored / 更大 pruning_ratio.")

    # 健康检查
    with torch.no_grad():
        _ = backbone_view(example_inputs)

    # 7) 评估：用“剪过的完整模型”，need_json 保留；签名做兼容
    try:
        test_stats, coco_evaluator = evaluate(
            pruned_model,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            need_json=need_json,
        )
    except TypeError:
        # 兼容你们老的 evaluate 签名（没有 need_json）
        test_stats, coco_evaluator = evaluate(
            pruned_model,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
        )

    if self.output_dir:
        dist_utils.save_on_master(
            coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
        )
    return
