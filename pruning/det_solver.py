# ===== FrozenBN -> BN（保证 device/dtype 一致）=====
import torch, copy
import torch.nn as nn
import torch_pruning as tp

try:
    from torchvision.ops.misc import FrozenBatchNorm2d
except Exception:
    class FrozenBatchNorm2d(nn.Module): pass

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
            if hasattr(bn, "num_batches_tracked"): bn.num_batches_tracked.zero_()
            bn.eval()
            setattr(module, name, bn)
        else:
            convert_frozen_bn_to_bn(child)

# ===== 只允许剪 backbone 的普通 Conv2d =====
def _is_depthwise(m: nn.Conv2d):
    return isinstance(m, nn.Conv2d) and m.groups == m.in_channels == m.out_channels

def collect_ignored_layers_only_keep_backbone_convs(full: nn.Module):
    allowed = set()
    if hasattr(full, "backbone"):
        for m in full.backbone.modules():
            if isinstance(m, nn.Conv2d) and m.groups == 1 and not _is_depthwise(m) and m.out_channels not in (1,3):
                allowed.add(m)
    ignored = []
    for m in full.modules():
        if isinstance(m, nn.Conv2d) and m not in allowed:
            ignored.append(m)
    return ignored

def round_to_from_model(model: nn.Module, base_round=8):
    r2 = 1
    for m in model.modules():
        if isinstance(m, nn.PixelShuffle):
            r2 = max(r2, m.upscale_factor*m.upscale_factor)
    return max(base_round, r2)

# ===== 仅用于统计的包装器：只跑 backbone =====
class BackboneOnly(nn.Module):
    def __init__(self, full):
        super().__init__()
        self.full = full
    def forward(self, x):
        return self.full.backbone(x)

# ===== DetSolver 内的 val_prune（或在 val 里加开关）=====
def val_prune(self, need_json=False):
    self.eval()
    device = self.device
    full = self.ema.module if self.ema else self.model

    # 深拷贝完整模型 → 先放到 CUDA，再转换 BN，防止设备不一致
    pruned_model = copy.deepcopy(full).to(device).eval()
    if hasattr(pruned_model, "backbone"):
        convert_frozen_bn_to_bn(pruned_model.backbone)

    # 构造 4D CUDA dummy（只用于构图/剪枝）
    H, W = getattr(self.cfg, "val_input_size", (640, 640))
    example_inputs = torch.randn(1, 3, H, W, device=device).float()

    # 只剪 backbone.Conv2d
    ignored_layers = collect_ignored_layers_only_keep_backbone_convs(pruned_model)
    round_to = round_to_from_model(pruned_model, base_round=getattr(self.cfg, "val_prune_round_to", 8))
    pruning_ratio = getattr(self.cfg, "val_prune_ratio", 0.30)

    # 重要性
    imp = tp.importance.GroupMagnitudeImportance(p=2)

    # 让 MetaPruner 的构图只跑 backbone(x)
    def forward_fn_runs_backbone(m: nn.Module, x: torch.Tensor):
        return m.backbone(x)

    # 用 float 做构图更稳（避免 AMP 干扰）
    pruned_model.float()

    pruner = tp.pruner.MetaPruner(
        pruned_model,                       # 传完整模型
        example_inputs=example_inputs,
        importance=imp,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        global_pruning=True, isomorphic=True,
        round_to=round_to,
        forward_fn=forward_fn_runs_backbone,
    )

    # —— 统计前后 MAC/Params（不再传 forward_fn；用包装器）——
    backbone_view = BackboneOnly(pruned_model).to(device).eval().float()
    base_macs, base_params = tp.utils.count_ops_and_params(backbone_view, example_inputs)
    print(f"[VAL-PRUNE][BASE] {base_macs/1e9:.3f}G MACs, {base_params/1e6:.3f}M params")

    pruner.step()   # 注意：不要包 no_grad

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
