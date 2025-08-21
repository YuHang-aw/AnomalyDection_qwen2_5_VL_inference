import torch
import torch.nn as nn

try:
    from torchvision.ops.misc import FrozenBatchNorm2d
except Exception:
    class FrozenBatchNorm2d(nn.Module): pass  # 兜底

def _module_device_dtype(m: nn.Module):
    # 找到任意一个 param/buffer 来确定 device/dtype
    for t in list(m.parameters()) + list(m.buffers()):
        return t.device, t.dtype
    # 实在找不到就返默认
    return torch.device("cpu"), torch.float32

@torch.no_grad()
def convert_frozen_bn_to_bn(module: nn.Module):
    """
    递归把 FrozenBatchNorm2d 换成 BatchNorm2d，
    且把“新 BN”放到与旧层一致的 device/dtype，上来就 eval()。
    """
    for name, child in list(module.named_children()):
        if isinstance(child, FrozenBatchNorm2d):
            dev, dtype = _module_device_dtype(child)

            bn = nn.BatchNorm2d(
                num_features=child.num_features,
                eps=getattr(child, "eps", 1e-5),
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ).to(device=dev, dtype=dtype)

            # 安全拷贝（权重/偏置/均值/方差在 FrozenBN 里是 buffer）
            mapping = [
                ("weight", "weight"),
                ("bias", "bias"),
                ("running_mean", "running_mean"),
                ("running_var", "running_var"),
            ]
            for dst, src in mapping:
                if hasattr(child, src) and hasattr(bn, dst):
                    getattr(bn, dst).data.copy_(getattr(child, src).data.to(dtype=dtype, device=dev))
            if hasattr(bn, "num_batches_tracked"):
                bn.num_batches_tracked.zero_()
            bn.eval()  # 只用于推理/构图
            setattr(module, name, bn)
        else:
            convert_frozen_bn_to_bn(child)


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


import copy
import torch
import torch.nn as nn
import torch_pruning as tp

def val_prune(self, need_json=False):
    self.eval()
    device = self.device
    full = self.ema.module if self.ema else self.model

    # === 深拷贝整模，别改训练用模型/EMA ===
    pruned_model = copy.deepcopy(full).to(device).eval()

    # === 只转换 backbone 里的 FrozenBN，且保持 device/dtype 一致 ===
    if hasattr(pruned_model, "backbone"):
        convert_frozen_bn_to_bn(pruned_model.backbone)

    # === 构造 4D CUDA dummy（只给 backbone 用；数值不重要，形状重要）===
    H, W = getattr(self.cfg, "val_input_size", (640, 640))
    example_inputs = torch.randn(1, 3, H, W, device=device)

    # === 只剪 backbone.Conv2d，其余全部忽略 ===
    ignored_layers = collect_ignored_layers_only_keep_backbone_convs(pruned_model)
    round_to = round_to_from_model(pruned_model, base_round=getattr(self.cfg, "val_prune_round_to", 8))
    pruning_ratio = getattr(self.cfg, "val_prune_ratio", 0.30)

    # === forward_fn：让 pruner 在构图时只跑 backbone(x) ===
    def forward_fn_runs_backbone(m: nn.Module, x: torch.Tensor):
        return m.backbone(x)

    imp = tp.importance.GroupMagnitudeImportance(p=2)

    # 混精时避免 dtype 冲突：构图/剪枝阶段强制用 fp32
    was_training = pruned_model.training
    pruned_model.eval()
    pruned_model.float()
    example_inputs = example_inputs.float()

    pruner = tp.pruner.MetaPruner(
        pruned_model,                   # 传“整模”
        example_inputs=example_inputs,  # 只要一个 Tensor(1,3,H,W)
        importance=imp,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        global_pruning=True,
        isomorphic=True,
        round_to=round_to,
        forward_fn=forward_fn_runs_backbone,  # 关键！
    )

    # 可选：剪前统计
    base_macs, base_params = tp.utils.count_ops_and_params(
        pruned_model, example_inputs, forward_fn=forward_fn_runs_backbone
    )
    print(f"[VAL-PRUNE][BASE] {base_macs/1e9:.3f}G MACs, {base_params/1e6:.3f}M params")

    # ★ 不要包 no_grad（需要 Autograd 构图）
    pruner.step()

    macs, params = tp.utils.count_ops_and_params(
        pruned_model, example_inputs, forward_fn=forward_fn_runs_backbone
    )
    print(f"[VAL-PRUNE][PRUNED] {macs/1e9:.3f}G ({macs/base_macs:.3f}x), {params/1e6:.3f}M")

    # 健康检查：backbone 前向一遍
    with torch.no_grad():
        _ = forward_fn_runs_backbone(pruned_model, example_inputs)

    # 恢复精度设置（若你们 evaluate 里会用 AMP，可在 evaluate 内自行 autocast）
    if was_training:
        pruned_model.train(False)

    # === 用“剪过的完整模型”跑 evaluate；need_json 保留 ===
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
