# ====== 1) FrozenBN -> BN（递归，保持 eval）======
import torch
import torch.nn as nn
import copy
import torch_pruning as tp

try:
    from torchvision.ops.misc import FrozenBatchNorm2d
except Exception:
    class FrozenBatchNorm2d(nn.Module): pass  # 兜底

@torch.no_grad()
def convert_frozen_bn_to_bn(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, FrozenBatchNorm2d):
            bn = nn.BatchNorm2d(
                num_features=child.num_features,
                eps=getattr(child, "eps", 1e-5),
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )
            # copy params/buffers
            for a_bn, a_fbn in [
                ("weight", "weight"),
                ("bias", "bias"),
                ("running_mean", "running_mean"),
                ("running_var", "running_var"),
            ]:
                if hasattr(child, a_fbn) and hasattr(bn, a_bn):
                    getattr(bn, a_bn).data.copy_(getattr(child, a_fbn).data)
            if hasattr(bn, "num_batches_tracked"):
                bn.num_batches_tracked.zero_()
            bn.eval()
            setattr(module, name, bn)
        else:
            convert_frozen_bn_to_bn(child)

# ====== 2) 只保留 backbone 里的“普通 Conv2d”为可剪，其余全部忽略 ======
def _is_depthwise(m: nn.Conv2d):
    return isinstance(m, nn.Conv2d) and m.groups == m.in_channels == m.out_channels

def collect_ignored_layers_only_keep_backbone_convs(full_model: nn.Module):
    allowed = set()
    if hasattr(full_model, "backbone"):
        for m in full_model.backbone.modules():
            if isinstance(m, nn.Conv2d) and m.groups == 1 and not _is_depthwise(m) and m.out_channels not in (1, 3):
                allowed.add(m)
    ignored = []
    for m in full_model.modules():
        if isinstance(m, nn.Conv2d) and m not in allowed:
            ignored.append(m)
    return ignored

def round_to_from_model(model: nn.Module, base_round=8):
    r2 = 1
    for m in model.modules():
        if isinstance(m, nn.PixelShuffle):
            r2 = max(r2, m.upscale_factor * m.upscale_factor)
    return max(base_round, r2)

# ====== 3) val_prune（最小改动）======
def val_prune(self, need_json=False):
    self.eval()
    device = self.device
    full = self.ema.module if self.ema else self.model

    # 3.1 深拷贝“完整模型”，别改原模型/EMA
    pruned_model = copy.deepcopy(full).to(device).eval()

    # 3.2 仅转换 backbone 里的 FrozenBN
    if hasattr(pruned_model, "backbone"):
        convert_frozen_bn_to_bn(pruned_model.backbone)

    # 3.3 构造一个简单的 4D dummy 输入（与 backbone 接口匹配）
    H, W = getattr(self.cfg, "val_input_size", (640, 640))
    example_inputs = torch.randn(1, 3, H, W, device=device)  # 只要 Tensor；不需要 labels

    # 3.4 只剪 backbone.Conv2d
    ignored_layers = collect_ignored_layers_only_keep_backbone_convs(pruned_model)
    round_to = round_to_from_model(pruned_model, base_round=getattr(self.cfg, "val_prune_round_to", 8))
    pruning_ratio = getattr(self.cfg, "val_prune_ratio", 0.30)

    # 3.5 用 forward_fn 让 pruner 只跑 backbone(images)
    def forward_fn_whole_model_runs_backbone(m: nn.Module, x: torch.Tensor):
        # x 已在 CUDA；不要改 dtype/device
        return m.backbone(x)

    imp = tp.importance.GroupMagnitudeImportance(p=2)
    pruner = tp.pruner.MetaPruner(
        pruned_model,                       # 注意：传“完整模型”！
        example_inputs=example_inputs,      # 一个 4D Tensor 即可
        importance=imp,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,      # 限定只剪 backbone convs
        global_pruning=True,
        isomorphic=True,
        round_to=round_to,
        forward_fn=forward_fn_whole_model_runs_backbone,  # 关键！
    )

    # 可选：打印前后 MAC/参数
    base_macs, base_params = tp.utils.count_ops_and_params(
        pruned_model, example_inputs, forward_fn=forward_fn_whole_model_runs_backbone
    )
    print(f"[VAL-PRUNE][BASE] {base_macs/1e9:.3f}G MACs, {base_params/1e6:.3f}M params")

    # ★ 不要包 no_grad：需要 AutoGrad 构图
    pruner.step()

    macs, params = tp.utils.count_ops_and_params(
        pruned_model, example_inputs, forward_fn=forward_fn_whole_model_runs_backbone
    )
    print(f"[VAL-PRUNE][PRUNED] {macs/1e9:.3f}G ({macs/base_macs:.3f}x), {params/1e6:.3f}M")

    # 健康检查：完整模型 + 真正的 dataloader 输入路径
    with torch.no_grad():
        _ = forward_fn_whole_model_runs_backbone(pruned_model, example_inputs)

    # 3.6 用“剪过的完整模型”评估（need_json 保留）
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
