import torch
import torch.nn as nn

# 如果你们的 FrozenBN 类在别的命名空间，按需导入
try:
    from torchvision.ops.misc import FrozenBatchNorm2d
except Exception:
    class FrozenBatchNorm2d(nn.Module): pass  # 占位，避免导入失败

@torch.no_grad()
def convert_frozen_bn_to_bn(module: nn.Module):
    """
    把所有 FrozenBatchNorm2d 替换成可学的 BatchNorm2d，并拷贝权重/均值/方差。
    保持 eval() 状态，避免影响推理统计。
    """
    for name, child in list(module.named_children()):
        if isinstance(child, FrozenBatchNorm2d):
            bn = nn.BatchNorm2d(
                num_features=child.num_features,
                eps=child.eps if hasattr(child, "eps") else 1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,   # 注意拼写
            )
            # 兼容 buffer/param 两种情形
            for attr_bn, attr_fbn in [
                ("weight", "weight"),
                ("bias", "bias"),
                ("running_mean", "running_mean"),
                ("running_var", "running_var"),
            ]:
                if hasattr(child, attr_fbn) and hasattr(bn, attr_bn):
                    getattr(bn, attr_bn).data.copy_(getattr(child, attr_fbn).data)
            # 一般 BN 会有这个 buffer
            if hasattr(bn, "num_batches_tracked"):
                bn.num_batches_tracked.zero_()
            bn.eval()  # 只用于推理/构图，不更新统计量
            setattr(module, name, bn)
        else:
            convert_frozen_bn_to_bn(child)


import copy
import torch
import torch.nn as nn
import torch_pruning as tp

def _is_depthwise(m: nn.Conv2d):
    return isinstance(m, nn.Conv2d) and m.groups == m.in_channels == m.out_channels

def _infer_example_inputs_from_val_loader(val_loader, device, fallback_hw=(640, 640)):
    try:
        batch = next(iter(val_loader))
        x = batch[0] if isinstance(batch, (list, tuple)) else (
            batch.get("images", None) if isinstance(batch, dict) else batch
        )
        if isinstance(x, (list, tuple)):
            x = x[0]
        if isinstance(x, torch.Tensor):
            c, h, w = x.shape[-3:]
            return torch.randn(1, c, h, w, device=device)
    except Exception:
        pass
    h, w = fallback_hw
    return torch.randn(1, 3, h, w, device=device)

def _round_to_from_model(model: nn.Module, base_round=8):
    r2 = 1
    for m in model.modules():
        if isinstance(m, nn.PixelShuffle):
            r2 = max(r2, m.upscale_factor * m.upscale_factor)
    return max(base_round, r2)

def _collect_ignored_layers_only_keep_backbone_convs(full_model: nn.Module):
    """
    允许：backbone 里的普通 Conv2d（非 depthwise、非分组、非 out=1/3）
    忽略：其它所有 Conv2d（包括 heads/transformer）、以及不满足条件的 backbone conv
    """
    allowed = set()
    if hasattr(full_model, "backbone"):
        for m in full_model.backbone.modules():
            if isinstance(m, nn.Conv2d):
                if not _is_depthwise(m) and m.groups == 1 and m.out_channels not in (1, 3):
                    allowed.add(m)
    ignored = []
    for m in full_model.modules():
        if isinstance(m, nn.Conv2d) and m not in allowed:
            ignored.append(m)
    return ignored

def val_prune(self, need_json=False):
    self.eval()

    # === 1) 拿“完整模型”（去掉 DDP 外壳）
    full = self.ema.module if self.ema else self.model
    device = self.device

    # === 2) 深拷贝整模（别改 self.model/EMA）
    pruned_model = copy.deepcopy(full).to(device).eval()

    # === 3) 仅把 backbone 里的 FrozenBN 换成 BN（更利于依赖图与重要性）
    if hasattr(pruned_model, "backbone"):
        convert_frozen_bn_to_bn(pruned_model.backbone)

    # === 4) 构例子输入（不要放在 no_grad 下）
    example_inputs = _infer_example_inputs_from_val_loader(
        self.val_dataloader, device,
        fallback_hw=getattr(self.cfg, "val_input_size", (640, 640))
    )

    # === 5) 只剪 backbone 的 Conv2d：其余全部忽略
    ignored_layers = _collect_ignored_layers_only_keep_backbone_convs(pruned_model)
    round_to = _round_to_from_model(pruned_model, base_round=getattr(self.cfg, "val_prune_round_to", 8))
    pruning_ratio = getattr(self.cfg, "val_prune_ratio", 0.30)

    # 重要性：组 L2；也可换 BNScaleImportance
    imp = tp.importance.GroupMagnitudeImportance(p=2)

    # === 6) 用 MetaPruner 对“整模”建图，但仅会动我们没忽略的层（即 backbone convs）
    pruner = tp.pruner.MetaPruner(
        pruned_model,
        example_inputs=example_inputs,
        importance=imp,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        global_pruning=True,
        isomorphic=True,
        round_to=round_to,
    )

    # (可选) 打印前后 MAC/参数
    base_macs, base_params = tp.utils.count_ops_and_params(pruned_model, example_inputs)
    print(f"[VAL-PRUNE][BASE] {base_macs/1e9:.3f}G MACs, {base_params/1e6:.3f}M params")

    pruner.step()   # 关键：不要包 no_grad

    macs, params = tp.utils.count_ops_and_params(pruned_model, example_inputs)
    print(f"[VAL-PRUNE][PRUNED] {macs/1e9:.3f}G MACs ({macs/base_macs:.3f}x), {params/1e6:.3f}M params")

    # 健康检查：前向一遍
    with torch.no_grad():
        _ = pruned_model(example_inputs)

    # === 7) 用“剪过的完整模型”去做 evaluate（need_json 保留）
    test_stats, coco_evaluator = evaluate(
        pruned_model,
        self.criterion,
        self.postprocessor,
        self.val_dataloader,
        self.evaluator,
        self.device,
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

