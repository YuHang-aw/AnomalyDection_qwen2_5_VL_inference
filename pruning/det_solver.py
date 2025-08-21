# --- A) forward_fn：让 pruner 穿过 backbone → neck/input_proj ----
import torch.nn as nn

def _as_list_feats(feats):
    if isinstance(feats, (list, tuple)):
        return list(feats)
    if isinstance(feats, dict):
        # 保序取 values
        return list(feats.values())
    return [feats]

def forward_fn_backbone_to_bridge(m: nn.Module, x):
    # 1) backbone
    feats = m.backbone(x)
    feats = _as_list_feats(feats)

    # 2) 颈部/多尺度聚合（如果有）
    if hasattr(m, "neck"):
        feats = m.neck(feats)  # 常见实现会接受 list[Tensor]
        feats = _as_list_feats(feats)

    # 3) input_proj / in_proj / 输入投影到 transformer 维度（尽量覆盖常见命名）
    for attr in ("input_proj", "in_proj", "proj", "input_proj_list"):
        if hasattr(m, attr):
            proj = getattr(m, attr)
            if isinstance(proj, nn.ModuleList):
                L = min(len(proj), len(feats))
                feats = [proj[i](feats[i]) for i in range(L)]
            elif isinstance(proj, nn.ModuleDict):
                # 以相同顺序映射
                keys = list(proj.keys())
                L = min(len(keys), len(feats))
                feats = [proj[keys[i]](feats[i]) for i in range(L)]
            elif isinstance(proj, nn.Module):
                # 单分支情形
                if len(feats) == 1:
                    feats = [proj(feats[0])]
                else:
                    # 如果是单模块但多尺度，至少让它看见第一个
                    feats = [proj(feats[0])] + feats[1:]
            break

    # 返回 tuple/list 都可以；用 tuple 覆盖所有分支，保证依赖能连到每条路
    return tuple(feats)

# pruner 初始化时改为：
pruner = tp.pruner.MetaPruner(
    pruned_model,
    example_inputs=example_inputs,
    importance=importance,
    pruning_ratio=pruning_ratio,
    ignored_layers=ignored_layers,   # 见下面 B）
    global_pruning=True,
    isomorphic=True,
    round_to=round_to,
    forward_fn=forward_fn_backbone_to_bridge,  # ← 关键：让它“看见”桥接层
)
# --- B) ignored_layers：不要把桥接层从图里“藏掉” ---
def collect_ignored_layers_keep_backbone_only(full: nn.Module):
    """
    目标：只剪 backbone 的 Conv2d；其它模块“可以被动跟裁 in_channels”，
    但不要作为主动裁剪目标（尤其是 transformer 里的 proj / head）。
    注意：桥接层（neck / input_proj）不能从计算图里消失，否则 in_channels 不会被修。
    """
    ignored = []
    ban_keywords = (
        "transformer", "encoder", "decoder", "attn", "attention",
        "head", "bbox", "cls", "query", "dn", "matcher", "postprocessor"
    )
    bridge_keywords_allowlist = ("neck", "input_proj", "in_proj", "proj")  # ← 不屏蔽这些

    for name, m in full.named_modules():
        if isinstance(m, nn.Conv2d):
            low = name.lower()
            # 如果是桥接层，保留在图里（不加入忽略）
            if any(k in low for k in bridge_keywords_allowlist):
                continue
            # 其它“非 backbone”/检测头/transformer 相关 → 忽略掉（不作为主动裁剪目标）
            if any(k in low for k in ban_keywords) or (m.out_channels in (1, 3)):
                ignored.append(m)
    return ignored
