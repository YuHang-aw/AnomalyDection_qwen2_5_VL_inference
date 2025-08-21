"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import datetime
import json
import time
import copy
import math

import torch
import torch.nn as nn
import torch.fx as fx

# 依赖: torch-pruning v2.x
import torch_pruning as tp

from ..misc import dist_utils, stats
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


# -------------------------- FrozenBN -> BN（保持 device/dtype） --------------------------
try:
    from torchvision.ops.misc import FrozenBatchNorm2d
except Exception:
    class FrozenBatchNorm2d(nn.Module):  # 兜底
        pass

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
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ).to(device=dev, dtype=dtype)

            for dst, src in [
                ("weight", "weight"),
                ("bias", "bias"),
                ("running_mean", "running_mean"),
                ("running_var", "running_var"),
            ]:
                if hasattr(child, src) and hasattr(bn, dst):
                    getattr(bn, dst).data.copy_(getattr(child, src).data.to(device=dev, dtype=dtype))
            if hasattr(bn, "num_batches_tracked"):
                bn.num_batches_tracked.zero_()
            bn.eval()
            setattr(module, name, bn)
        else:
            convert_frozen_bn_to_bn(child)


# -------------------------- FX: 找风险层（分组/DW 及其直接供给层） --------------------------
def _build_module_map(root: nn.Module):
    return dict(root.named_modules())

def _fx_find_grouped_and_producers(backbone: nn.Module, example_x: torch.Tensor):
    """
    返回需要忽略“主动剪”的模块集合：
      - 所有 groups>1 的 Conv2d（含 depthwise）
      - 以及它们的“直接上游供给层”（一般是 Conv2d/Linear）
    这些层会保留在图里（用于依赖修正），但不作为主动剪枝目标，避免整除/对齐崩溃。
    """
    gm = fx.symbolic_trace(backbone)
    name_to_mod = _build_module_map(backbone)
    to_ignore = set()

    for n in gm.graph.nodes:
        if n.op == "call_module":
            mod = name_to_mod.get(n.target, None)
            if isinstance(mod, nn.Conv2d) and getattr(mod, "groups", 1) > 1:
                to_ignore.add(mod)
                # 往前找第一个生产该输入特征的 conv/linear
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


# -------------------------- 忽略检测头/Transformer；保留桥接层参与依赖修正 --------------------------
def _ignore_non_backbone_heads(full: nn.Module):
    """
    忽略检测头/transformer/输出卷积（out=1或3），以免它们被“主动剪”；
    但桥接层（neck / input_proj / in_proj / proj）不要忽略，让它们在依赖里被自动修 in_channels。
    """
    ignored = []
    ban_keywords = (
        "transformer", "encoder", "decoder", "attn", "attention",
        "head", "bbox", "cls", "query", "dn", "matcher", "postprocessor"
    )
    allow_bridge = ("neck", "input_proj", "in_proj", "proj", "input_proj_list")

    for name, m in full.named_modules():
        if isinstance(m, nn.Conv2d):
            low = name.lower()
            if any(k in low for k in allow_bridge):
                continue
            if any(k in low for k in ban_keywords) or (m.out_channels in (1, 3)):
                ignored.append(m)
    return set(ignored)


# -------------------------- 统计用包装器（count_ops 不支持 forward_fn） --------------------------
class BackboneOnly(nn.Module):
    def __init__(self, full): super().__init__(); self.full = full
    def forward(self, x): return self.full.backbone(x)


# -------------------------- forward_fn：让 pruner 看见桥接层 --------------------------
def _as_list_feats(feats):
    if isinstance(feats, (list, tuple)):
        return list(feats)
    if isinstance(feats, dict):
        return list(feats.values())
    return [feats]

def forward_fn_backbone_to_bridge(m: nn.Module, x):
    feats = m.backbone(x)
    feats = _as_list_feats(feats)

    # neck（如有）
    if hasattr(m, "neck"):
        try:
            feats = m.neck(feats)
            feats = _as_list_feats(feats)
        except Exception:
            # 有些实现 neck(x) 接受 dict 或单 Tensor；尽量兼容
            try:
                feats = _as_list_feats(m.neck(feats))
            except Exception:
                pass

    # input_proj / in_proj / proj / input_proj_list（尽可能覆盖多分支）
    for attr in ("input_proj", "in_proj", "proj", "input_proj_list"):
        if hasattr(m, attr):
            proj = getattr(m, attr)
            if isinstance(proj, nn.ModuleList):
                L = min(len(proj), len(feats))
                feats = [proj[i](feats[i]) for i in range(L)]
            elif isinstance(proj, nn.ModuleDict):
                keys = list(proj.keys())
                L = min(len(keys), len(feats))
                feats = [proj[keys[i]](feats[i]) for i in range(L)]
            elif isinstance(proj, nn.Module):
                if len(feats) == 1:
                    feats = [proj(feats[0])]
                else:
                    feats = [proj(feats[0])] + feats[1:]
            break
    # 返回 tuple，确保依赖能连到每条分支
    return tuple(feats)


# =================================== Solver ===================================
class DetSolver(BaseSolver):
    def fit(self):
        self.train()
        args = self.cfg
        metric_names = ["AP50:95", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]

        if self.use_wandb:
            import wandb
            wandb.init(
                project=args.yaml_cfg["project_name"],
                name=args.yaml_cfg["exp_name"],
                config=args.yaml_cfg,
            )
            wandb.watch(self.model)

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)
        top1 = 0
        best_stat = {
            "epoch": -1,
        }
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                self.last_epoch,
                self.use_wandb
            )
            for k in test_stats:
                best_stat["epoch"] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f"best_stat: {best_stat}")

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epochs):
            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                if self.ema:
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                epochs=args.epochs,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                use_wandb=self.use_wandb,
                output_dir=self.output_dir,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / "last.pth"]
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                epoch,
                self.use_wandb,
                output_dir=self.output_dir,
            )

            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"Test/{k}_{i}".format(k), v, epoch)

                if k in best_stat:
                    best_stat["epoch"] = epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print["epoch"] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg2.pth")
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg1.pth")

                best_stat_print[k] = max(best_stat[k], top1)
                print(f"best_stat: {best_stat_print}")

                if best_stat["epoch"] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg2.pth")
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg1.pth")

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {"epoch": -1}
                    if self.ema:
                        self.ema.decay -= 0.0001
                        self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                        print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.use_wandb:
                wandb_logs = {}
                for idx, metric_name in enumerate(metric_names):
                    wandb_logs[f"metrics/{metric_name}"] = test_stats["coco_eval_bbox"][idx]
                wandb_logs["epoch"] = epoch
                import wandb as _wandb
                _wandb.log(wandb_logs)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(self):
        self.eval()
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            epoch=-1,
            use_wandb=False,
        )
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        return

    # ------------------------------ 这里是你要的 val_prune ------------------------------
    def val_prune(self, need_json=False):
        """
        仅“主动剪” backbone.Conv2d；让 pruner 经过 neck/input_proj，从而自动修正桥接层 in_channels。
        - 避免 in_channels % groups != 0：分组/DW 及其直接供给层不会作为主动剪目标
        - 避免 1432 vs 2048：forward_fn 穿过桥接层，使其 in_channels 随上游自动对齐
        """
        self.eval()
        device = self.device
        full = self.ema.module if self.ema else self.model

        # 1) 深拷贝完整模型 → CUDA → eval
        pruned_model = copy.deepcopy(full).to(device).eval()

        # 2) 仅在 backbone 内把 FrozenBN -> BN（保持 device/dtype）
        if hasattr(pruned_model, "backbone"):
            convert_frozen_bn_to_bn(pruned_model.backbone)

        # 3) 构造 4D CUDA dummy（只给图用；不要 labels）
        H, W = getattr(self.cfg, "val_input_size", (640, 640))
        example_x = torch.randn(1, 3, H, W, device=device).float()

        # 4) 计算忽略集合：风险层（DW/分组及其供给层） + 检测头/Transformer；保留桥接层
        risk_ignores = set()
        if hasattr(pruned_model, "backbone"):
            try:
                risk_ignores = _fx_find_grouped_and_producers(pruned_model.backbone, example_x)
            except Exception as _:
                pass  # FX 失败也没关系，只是少了一层保护
        head_ignores = _ignore_non_backbone_heads(pruned_model)
        ignored_layers = list(risk_ignores.union(head_ignores))

        # 5) round_to & ratio
        round_to = getattr(self.cfg, "val_prune_round_to", 8)   # 建议: 先 8，再按需要调 16
        pruning_ratio = getattr(self.cfg, "val_prune_ratio", 0.30)

        # 6) 重要性：若存在 BN，用 BNScale 更稳；否则 GroupMagnitude
        has_bn = any(isinstance(m, nn.BatchNorm2d) for m in pruned_model.modules())
        importance = tp.importance.BNScaleImportance() if has_bn else tp.importance.GroupMagnitudeImportance(p=2)

        # 7) pruner：构图只跑 backbone→neck→input_proj，但“建图对象”是完整模型
        pruned_model.float()  # 构图用 fp32 更稳
        pruner = tp.pruner.MetaPruner(
            pruned_model,
            example_inputs=example_x,
            importance=importance,
            pruning_ratio=pruning_ratio,
            ignored_layers=ignored_layers,     # 这些层不会作为“主动剪”对象，但仍会在依赖里被修 in_channels
            global_pruning=True,
            isomorphic=True,
            round_to=round_to,
            forward_fn=forward_fn_backbone_to_bridge,
        )

        # 8) （可选）统计剪前后 MAC/Params，仅统计 backbone（避免你版本的 forward_fn 限制）
        try:
            backbone_view = BackboneOnly(pruned_model).to(device).eval().float()
            base_macs, base_params = tp.utils.count_ops_and_params(backbone_view, example_x)
        except Exception:
            base_macs, base_params = None, None

        # 9) 真正执行剪枝（不要包 no_grad）
        pruner.step()

        # 10) 基本体检：保证所有 Conv2d 的整除关系
        for name, m in pruned_model.named_modules():
            if isinstance(m, nn.Conv2d):
                g = int(getattr(m, "groups", 1))
                if g > 0:
                    assert m.in_channels % g == 0, f"[groups] {name}: in={m.in_channels}, groups={g}"

        # 11) （可选）统计剪后
        try:
            macs, params = tp.utils.count_ops_and_params(backbone_view, example_x)
            if base_macs is not None:
                print(f"[VAL-PRUNE] MACs {base_macs/1e9:.3f}G -> {macs/1e9:.3f}G, "
                      f"Params {base_params/1e6:.3f}M -> {params/1e6:.3f}M")
        except Exception:
            pass

        # 12) 小规模前向健康检查（整模，仅图像 Tensor）
        with torch.no_grad():
            _ = pruned_model(example_x)

        # 13) 用“剪过的完整模型”评估；对 evaluate 的不同签名做兼容
        try:
            test_stats, coco_evaluator = evaluate(
                pruned_model,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                device,
                need_json=need_json,
            )
        except TypeError:
            # 可能需要 epoch/use_wandb/output_dir 参数（兼容你 fit/val 的两种调用）
            try:
                test_stats, coco_evaluator = evaluate(
                    pruned_model,
                    self.criterion,
                    self.postprocessor,
                    self.val_dataloader,
                    self.evaluator,
                    device,
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
                    device,
                    -1,
                    False,
                    output_dir=getattr(self, "output_dir", None),
                )

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        return
