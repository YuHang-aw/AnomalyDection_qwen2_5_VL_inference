# ======= 依赖 =======
import copy, json, time, datetime, math
import torch
import torch.nn as nn
import torch.fx as fx
import torch_pruning as tp

# -------- FrozenBN -> BN（保持 device/dtype）--------
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
                child.num_features,
                eps=getattr(child, "eps", 1e-5),
                momentum=0.1, affine=True, track_running_stats=True
            ).to(device=dev, dtype=dtype)
            for dst, src in [("weight","weight"),("bias","bias"),
                             ("running_mean","running_mean"),("running_var","running_var")]:
                if hasattr(child, src) and hasattr(bn, dst):
                    getattr(bn, dst).data.copy_(getattr(child, src).data.to(dev, dtype))
            if hasattr(bn, "num_batches_tracked"):
                bn.num_batches_tracked.zero_()
            bn.eval()
            setattr(module, name, bn)
        else:
            convert_frozen_bn_to_bn(child)

# -------- 统计用包装器（count_ops 不支持 forward_fn）--------
class BackboneOnly(nn.Module):
    def __init__(self, full): super().__init__(); self.full = full
    def forward(self, x): return self.full.backbone(x)

# -------- 工具：命名工具 --------
def _named_module_dict(root: nn.Module):
    return dict(root.named_modules())

def _is_depthwise(m: nn.Module):
    return isinstance(m, nn.Conv2d) and m.groups == m.in_channels == m.out_channels

# -------- FX: 找高风险层（分组/DW及其供给层 + 参与add/cat的生产者）--------
def _fx_collect_risky_producers(backbone: nn.Module):
    risky = set()
    name2mod = _named_module_dict(backbone)
    gm = fx.symbolic_trace(backbone)

    # 1) 所有 groups>1 Conv 及其直接供给层
    for n in gm.graph.nodes:
        if n.op == "call_module":
            mod = name2mod.get(n.target, None)
            if isinstance(mod, nn.Conv2d) and getattr(mod, "groups", 1) > 1:
                risky.add(mod)
                # 找第一个上游的 conv/linear
                for inp in n.all_input_nodes:
                    if inp.op == "call_module":
                        pm = name2mod.get(inp.target, None)
                        if isinstance(pm, (nn.Conv2d, nn.Linear)):
                            risky.add(pm)
                            break

    # 2) 参与 add/cat(dim=1) 的生产者（最近的 Conv/Linear）
    def _closest_producers(node):
        prod = []
        for inp in node.all_input_nodes:
            target = None
            q = [inp]
            seen = set()
            while q:
                t = q.pop()
                if t in seen: continue
                seen.add(t)
                if t.op == "call_module":
                    m = name2mod.get(t.target, None)
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        target = m; break
                q.extend(t.all_input_nodes)
            if target is not None:
                prod.append(target)
        return prod

    aten_add = {torch.add, torch.ops.aten.add.Tensor}
    aten_cat = {torch.cat, torch.ops.aten.cat.default}

    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target in aten_add:
            for p in _closest_producers(n):
                risky.add(p)
        if n.op == "call_function" and n.target in aten_cat:
            # 只关心 dim=1 的拼接
            dim = None
            if isinstance(n.args, tuple) and len(n.args) >= 2:
                dim = n.args[1]
            if dim in (None, 1):   # None 时很多实现默认按 dim=0；这里更保守一些
                for p in _closest_producers(n):
                    risky.add(p)
    return risky

# -------- 允许剪谁？（严格白名单）--------
def _collect_allowed_backbone_convs(full: nn.Module):
    allowed = set()
    if not hasattr(full, "backbone"): return allowed
    for name, m in full.backbone.named_modules():
        if isinstance(m, nn.Conv2d):
            # 只允许普通 conv（groups==1）、且不是极小输出（1/3 通常是输出层）
            if m.groups == 1 and not _is_depthwise(m) and m.out_channels not in (1, 3):
                # 屏蔽常见“阶段/桥接/下采样/shortcut/expand/transition”等关键层
                low = name.lower()
                if any(k in low for k in ("downsample","shortcut","proj","expand","transition","stem_out")):
                    continue
                allowed.add(m)
    return allowed

# -------- 把 allowed 再减去“高风险”层，得到最终可剪集合；其余全忽略 --------
def _build_ignore_and_stage_groups(full: nn.Module):
    allowed = _collect_allowed_backbone_convs(full)

    risky = set()
    if hasattr(full, "backbone"):
        try:
            risky = _fx_collect_risky_producers(full.backbone)
        except Exception:
            pass

    final_allowed = {m for m in allowed if m not in risky}

    # stage 分组（按 'backbone.<stage>.' 的第2段名分组）
    name2mod = dict(full.backbone.named_modules())
    mod2name = {v:k for k,v in name2mod.items()}
    groups = {}
    for m in final_allowed:
        name = mod2name.get(m, "")
        stage = name.split(".", 1)[0] if "." in name else name
        groups.setdefault(stage, []).append(m)

    # ignored_layers = 全部 Conv2d - final_allowed（整模范围）
    all_convs = [m for m in full.modules() if isinstance(m, nn.Conv2d)]
    ignored_layers = [m for m in all_convs if m not in final_allowed]
    return ignored_layers, groups

# -------- 让 pruner 看见“整模”依赖（只传图像 Tensor即可）--------
def _full_forward(m: nn.Module, x: torch.Tensor):
    return m(x)

# ================= DetSolver: 新版 val_prune（分阶段试剪 + 自动回滚） =================
def val_prune(self, need_json=False):
    """
    更稳健的验证剪枝：
      - 仅主动剪 backbone 内的安全 Conv2d（groups==1），其余全部忽略主动剪
      - FX 识别 add/cat、分组/DW 的高风险生产者，将其排除主动剪
      - 按 stage 分组，逐组尝试剪枝；每组剪完立刻整模 forward 验证，不通过则回滚跳过
      - pruner 运行整模前向（只喂图像），自动修正桥接/颈部/in_proj 的 in_channels
    """
    self.eval()
    device = self.device
    full = self.ema.module if self.ema else self.model

    # 1) 深拷贝完整模型 → CUDA
    pruned_model = copy.deepcopy(full).to(device).eval()

    # 2) 仅在 backbone 内把 FrozenBN -> BN（保持 device/dtype）
    if hasattr(pruned_model, "backbone"):
        convert_frozen_bn_to_bn(pruned_model.backbone)

    # 3) dummy 输入（不要 labels）
    H, W = getattr(self.cfg, "val_input_size", (640, 640))
    example_x = torch.randn(1, 3, H, W, device=device).float()

    # 4) 构造忽略集合 + stage 分组
    ignored_layers_all, stage_groups = _build_ignore_and_stage_groups(pruned_model)

    # 5) 基础参数
    round_to = getattr(self.cfg, "val_prune_round_to", 8)    # 先 8，稳定后再提 16
    global_ratio = getattr(self.cfg, "val_prune_ratio", 0.30)
    has_bn = any(isinstance(m, nn.BatchNorm2d) for m in pruned_model.modules())
    importance = tp.importance.BNScaleImportance() if has_bn else tp.importance.GroupMagnitudeImportance(p=2)

    # 6) 统计剪前（只看 backbone）
    try:
        backbone_view = BackboneOnly(pruned_model).to(device).eval().float()
        base_macs, base_params = tp.utils.count_ops_and_params(backbone_view, example_x)
        print(f"[VAL-PRUNE] Before: {base_macs/1e9:.3f}G MACs, {base_params/1e6:.3f}M params")
    except Exception:
        base_macs, base_params = None, None

    # 7) 逐 stage 试剪 + 健康检查；失败则自动回滚
    stages = sorted(stage_groups.keys())
    success_stages = []
    for si, stage in enumerate(stages):
        allow_this_stage = set(stage_groups[stage])
        if not allow_this_stage:
            continue

        # 本轮的“忽略层” = 全忽略 ∪（其它 stage 的 allowed）
        ignored_layers = list(set(ignored_layers_all) | {m for s,mods in stage_groups.items() if s != stage for m in mods})

        # 复制当前模型做试剪
        trial = copy.deepcopy(pruned_model).to(device).eval().float()

        pruner = tp.pruner.MetaPruner(
            trial,
            example_inputs=example_x,
            importance=importance,
            pruning_ratio=global_ratio,       # 每个 stage 用同一比例；需要可以改成 per-stage
            ignored_layers=ignored_layers,    # 只允许本 stage 的安全层被“主动剪”
            global_pruning=True,
            isomorphic=True,
            round_to=round_to,
            forward_fn=_full_forward,         # 跑“整模”前向，自动修正所有依赖 in_channels
        )

        try:
            pruner.step()                     # 不要 no_grad
            with torch.no_grad():
                _ = trial(example_x)          # 整模健康检查
        except Exception as e:
            print(f"[VAL-PRUNE][{stage}] failed, skip. Reason: {repr(e)}")
            continue  # 回滚：不替换 pruned_model，跳过该 stage

        # 通过健康检查，接受本次修改
        pruned_model = trial
        success_stages.append(stage)
        print(f"[VAL-PRUNE][{stage}] pruned OK.")

    if not success_stages:
        print("[VAL-PRUNE] No stage was pruned safely. Returning original model for evaluation.")

    # 8) 剪后统计
    try:
        backbone_view = BackboneOnly(pruned_model).to(device).eval().float()
        macs, params = tp.utils.count_ops_and_params(backbone_view, example_x)
        if base_macs is not None:
            print(f"[VAL-PRUNE] After : {macs/1e9:.3f}G MACs ({macs/max(base_macs,1):.3f}x), "
                  f"Params {params/1e6:.3f}M")
    except Exception:
        pass

    # 9) 最终 sanity：整除性 & 快速前向
    for name, m in pruned_model.named_modules():
        if isinstance(m, nn.Conv2d):
            g = int(getattr(m, "groups", 1))
            assert m.in_channels % g == 0, f"[groups] {name}: in={m.in_channels}, groups={g}"
    with torch.no_grad():
        _ = pruned_model(example_x)

    # 10) 评估（适配不同 evaluate 签名）
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
        try:
            test_stats, coco_evaluator = evaluate(
                pruned_model,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                epoch=-1,
                use_wandb=False,
            )
        except TypeError:
            test_stats, coco_evaluator = evaluate(
                pruned_model,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                -1,
                False,
                output_dir=getattr(self, "output_dir", None),
            )

    if self.output_dir:
        dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
    return
