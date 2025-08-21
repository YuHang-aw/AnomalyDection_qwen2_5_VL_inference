import copy, math, torch
import torch.nn as nn
import torch.fx as fx
import torch_pruning as tp

# ---------- 1) FrozenBN -> BN（保证 device/dtype 一致） ----------
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
            bn = nn.BatchNorm2d(child.num_features, eps=getattr(child,"eps",1e-5),
                                momentum=0.1, affine=True, track_running_stats=True
                               ).to(device=dev, dtype=dtype)
            for dst, src in [("weight","weight"),("bias","bias"),
                             ("running_mean","running_mean"),("running_var","running_var")]:
                if hasattr(child, src) and hasattr(bn, dst):
                    getattr(bn, dst).data.copy_(getattr(child, src).data.to(dev, dtype))
            if hasattr(bn,"num_batches_tracked"): bn.num_batches_tracked.zero_()
            bn.eval()
            setattr(module, name, bn)
        else:
            convert_frozen_bn_to_bn(child)

# ---------- 2) FX 跟踪，找出“风险层”：所有 groups>1 的 Conv2d 及其直接上游供给层 ----------
def _build_module_map(root: nn.Module):
    # name -> module
    return dict(root.named_modules())

def _fx_find_grouped_and_producers(backbone: nn.Module, example_x: torch.Tensor):
    """
    返回需要忽略的模块集合：所有分组/深度可分离 Conv2d 以及它们的直接上游供给 Conv2d/Linear（若有）。
    """
    gm = fx.symbolic_trace(backbone)
    name_to_mod = _build_module_map(backbone)
    to_ignore = set()

    # node.target 可能是模块名（call_module）或函数（call_function/call_method）
    # 我们只处理 call_module 且是 Conv2d
    # 再往前找它的第一个上游 call_module 且是（Conv2d/Linear）的 node 作为“供给层”
    for n in gm.graph.nodes:
        if n.op == "call_module":
            mod = name_to_mod.get(n.target, None)
            if isinstance(mod, nn.Conv2d) and mod.groups > 1:
                # 本层：必须忽略
                to_ignore.add(mod)
                # 往前找供给层
                producer = None
                for inp in n.all_input_nodes:
                    if inp.op == "call_module":
                        pm = name_to_mod.get(inp.target, None)
                        if isinstance(pm, (nn.Conv2d, nn.Linear)):
                            producer = pm
                            break
                if producer is not None:
                    to_ignore.add(producer)
    return to_ignore

# ---------- 3) 评估里需要忽略的非 backbone 模块（头/transformer 等） ----------
def _ignore_non_backbone_heads(full: nn.Module):
    ignored = []
    ban_keywords = ("transformer","encoder","decoder","attn","attention","head",
                    "bbox","cls","query","dn","matcher","postprocessor")
    for name, m in full.named_modules():
        if isinstance(m, nn.Conv2d) and (m.out_channels in (1,3) or any(k in name.lower() for k in ban_keywords)):
            ignored.append(m)
    return set(ignored)

# ---------- 4) 仅用于统计的包装器（count_ops 不支持 forward_fn） ----------
class BackboneOnly(nn.Module):
    def __init__(self, full): super().__init__(); self.full = full
    def forward(self, x): return self.full.backbone(x)

# ---------- 5) DetSolver 内的 val_prune ----------
def val_prune(self, need_json=False):
    self.eval()
    device = self.device
    full = self.ema.module if self.ema else self.model

    # 深拷贝完整模型并放到 CUDA
    pruned_model = copy.deepcopy(full).to(device).eval()

    # 只把 backbone 里的 FrozenBN 转成 BN（保持 device/dtype）
    if hasattr(pruned_model, "backbone"):
        convert_frozen_bn_to_bn(pruned_model.backbone)

    # CUDA dummy 输入（构图用；不需要 label）
    H, W = getattr(self.cfg, "val_input_size", (640, 640))
    example_x = torch.randn(1, 3, H, W, device=device).float()

    # 用 FX 找出分组/深度可分离 Conv 及其直接上游供给层 —— 全部忽略
    risk_ignores = _fx_find_grouped_and_producers(pruned_model.backbone, example_x)
    # 再忽略检测头/transformer 等
    head_ignores = _ignore_non_backbone_heads(pruned_model)
    ignored_layers = list(risk_ignores.union(head_ignores))

    # round_to 先用 8/16（别用巨大 LCM，不然很难动）；剪枝比例适中
    round_to = getattr(self.cfg, "val_prune_round_to", 8)
    pruning_ratio = getattr(self.cfg, "val_prune_ratio", 0.30)

    # 重要性：有 BN 用 BNScale，效果更稳
    has_bn = any(isinstance(m, nn.BatchNorm2d) for m in pruned_model.modules())
    importance = tp.importance.BNScaleImportance() if has_bn else tp.importance.GroupMagnitudeImportance(p=2)

    # 只在构图阶段跑 backbone(x)，但“建图对象”仍是完整模型（避免 encoder/decoder 丢失）
    def fwd_backbone(m: nn.Module, x: torch.Tensor):
        return m.backbone(x)

    pruned_model.float()  # 构图阶段用 fp32 稳定
    pruner = tp.pruner.MetaPruner(
        pruned_model,
        example_inputs=example_x,
        importance=importance,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        global_pruning=True,
        isomorphic=True,
        round_to=round_to,
        forward_fn=fwd_backbone,
    )

    # 剪前统计
    backbone_view = BackboneOnly(pruned_model).to(device).eval().float()
    base_macs, base_params = tp.utils.count_ops_and_params(backbone_view, example_x)

    # 真正执行剪枝（不要加 no_grad）
    pruner.step()

    macs, params = tp.utils.count_ops_and_params(backbone_view, example_x)
    print(f"[VAL-PRUNE] MACs {base_macs/1e9:.3f}G -> {macs/1e9:.3f}G, "
          f"Params {base_params/1e6:.3f}M -> {params/1e6:.3f}M")

    # 健康检查：backbone 前向一遍
    with torch.no_grad():
        _ = backbone_view(example_x)

    # 评估（按你自己的 evaluate 签名来；这里给一个兼容调用）
    try:
        test_stats, coco_evaluator = evaluate(
            pruned_model,
            self.criterion, self.postprocessor, self.val_dataloader,
            self.evaluator, self.device, need_json=need_json
        )
    except TypeError:
        test_stats, coco_evaluator = evaluate(
            pruned_model,
            self.criterion, self.postprocessor, self.val_dataloader,
            self.evaluator, self.device
        )
    return
