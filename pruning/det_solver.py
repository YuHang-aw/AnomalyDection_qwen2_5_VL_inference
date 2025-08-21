# -*- coding: utf-8 -*-
import copy, math
import torch
import torch.nn as nn
import torch_pruning as tp

# ====== FrozenBN -> BN（保证 device/dtype 一致）======
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
            if hasattr(bn, "num_batches_tracked"):
                bn.num_batches_tracked.zero_()
            bn.eval()
            setattr(module, name, bn)
        else:
            convert_frozen_bn_to_bn(child)

# ====== 统计专用：只跑 backbone 的“视图” ======
class BackboneOnly(nn.Module):
    def __init__(self, full): super().__init__(); self.full = full
    def forward(self, x): return self.full.backbone(x)

def lcm(a, b): 
    return abs(a*b) // math.gcd(a, b) if a and b else max(a, b)

def _conv_out_importance_L1(conv: nn.Conv2d):
    # 每个输出通道的重要性：卷积核 L1/均值，稳定又无需 BN
    with torch.no_grad():
        w = conv.weight.detach().abs()
        # [Cout, Cin, Kh, Kw] -> [Cout]
        return w.mean(dim=(1,2,3))

def val_prune(self, need_json=False):
    self.eval()
    device = self.device
    full = self.ema.module if self.ema else self.model

    # 1) 深拷贝完整模型，放 CUDA，冻结统计
    pruned = copy.deepcopy(full).to(device).eval()

    # 2) 只转换 backbone 的 FrozenBN -> BN（保持 device/dtype）
    if hasattr(pruned, "backbone"):
        convert_frozen_bn_to_bn(pruned.backbone)

    # 3) 4D CUDA dummy（只用于构图/剪枝），不要 labels
    H, W = getattr(self.cfg, "val_input_size", (640, 640))
    example = torch.randn(1, 3, H, W, device=device).float()

    # 4) 依赖图：构图只跑 backbone(x)，但我们后面会执行“全模型”的联动计划
    def fwd_backbone(m: nn.Module, x: torch.Tensor):
        return m.backbone(x)

    DG = tp.DependencyGraph().build(pruned, example_inputs=example, forward_fn=fwd_backbone)

    # 5) 全局/默认参数
    target_ratio = getattr(self.cfg, "val_prune_ratio", 0.30)  # 期望的全局比例
    base_round  = getattr(self.cfg, "val_prune_round_to", 1)   # 先放开，后续按局部自适应对齐

    # 6) 逐层“自适应安全裁剪”：只选 backbone 里 groups==1 的产出卷积
    prunable_convs = []
    for name, m in pruned.backbone.named_modules():
        if isinstance(m, nn.Conv2d) and m.groups == 1 and m.out_channels > 4:
            prunable_convs.append((name, m))

    pruned_summary = []

    for name, conv in prunable_convs:
        Cout = conv.out_channels
        want = int(Cout * target_ratio)
        if want <= 0: 
            continue

        # —— 找“联动会影响到的下游分组因子”的 LCM（自适应对齐）——
        # 用一个“预计划”探路：idxs=[0] 足够让 DG 给出依赖的下游操作集合
        probe_plan = DG.get_pruning_plan(conv, tp.prune_conv_out_channels, idxs=[0])
        M_local = base_round
        for op in probe_plan:  # PruningPlan 可迭代；不同 tp 版本字段名略有差异
            mod = getattr(op, "target", None) or getattr(op, "module", None)
            fn  = getattr(op, "handler", None) or getattr(op, "func", None)
            if isinstance(mod, nn.Conv2d):
                # 只对“修 in_channels”的依赖层收集分组因子
                if fn is tp.prune_conv_in_channels and mod.groups > 1:
                    M_local = lcm(M_local, mod.groups)
            if isinstance(mod, nn.PixelShuffle):
                M_local = lcm(M_local, mod.upscale_factor * mod.upscale_factor)

        # 把想裁数量收敛到 M_local 的倍数（保证整除）
        k = (want // max(M_local,1)) * max(M_local,1)
        if k <= 0:
            continue

        # —— 选通道（重要性）——
        score = _conv_out_importance_L1(conv)  # [Cout]
        idx_all = torch.argsort(score).tolist()  # 从小到大
        # 自适应 + 失败回退：从 k 开始，不行就减去 M_local 继续试
        ok = False
        while k > 0 and not ok:
            idx = idx_all[:k]
            plan = DG.get_pruning_plan(conv, tp.prune_conv_out_channels, idxs=idx)
            try:
                plan.exec()
                ok = True
            except RuntimeError as e:
                # 如果仍然触发 groups 不可整除等问题，就往下回退一个对齐步长
                k -= max(M_local,1)
                if k <= 0:
                    break
        if ok:
            pruned_summary.append((name, Cout, conv.out_channels, M_local))
        # 如果不 ok，跳过本层，继续下一层

    # 7) 统计（只看 backbone 的 MAC/Params）
    backbone_view = BackboneOnly(pruned).to(device).eval().float()
    macs, params = tp.utils.count_ops_and_params(backbone_view, example)
    macs0, params0 = tp.utils.count_ops_and_params(BackboneOnly(full).to(device).eval().float(), example)
    print(f"[VAL-PRUNE] Backbone MACs {macs0/1e9:.3f}G -> {macs/1e9:.3f}G  "
          f"Params {params0/1e6:.3f}M -> {params/1e6:.3f}M")
    print(f"[VAL-PRUNE] 实际被剪的卷积层数: {len(pruned_summary)}")
    for n, c0, c1, M in pruned_summary[:12]:
        print(f"  - {n}: {c0}->{c1}  (对齐步长 M={M})")

    # 8) 健康检查：前向一遍 backbone
    with torch.no_grad():
        _ = pruned.backbone(example)

    # 9) 评估（你的 evaluate 没有 epoch/use_wandb/output_dir 就用这个签名）
    try:
        test_stats, coco_evaluator = evaluate(
            pruned, self.criterion, self.postprocessor,
            self.val_dataloader, self.evaluator, self.device,
            need_json=need_json,
        )
    except TypeError:
        test_stats, coco_evaluator = evaluate(
            pruned, self.criterion, self.postprocessor,
            self.val_dataloader, self.evaluator, self.device,
        )

    return
