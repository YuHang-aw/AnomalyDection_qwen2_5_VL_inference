import datetime
import json
import time
import torch
import torch.nn as nn
import torch_pruning as tp

from ..misc import dist_utils, stats
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch

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
                    best_stat["epoch"] = (
                        epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                    )
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print["epoch"] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                        else:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg1.pth"
                            )

                best_stat_print[k] = max(best_stat[k], top1)
                print(f"best_stat: {best_stat_print}")  # global best

                if best_stat["epoch"] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(
                            self.state_dict(), self.output_dir / "best_stg1.pth"
                        )

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {
                        "epoch": -1,
                    }
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
                wandb.log(wandb_logs)

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
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

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
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return

    def val_prune(self, need_json=False):
        """
        这个方法执行剪枝的操作，处理剪枝后的 BN 参数，并在剪枝后修复所有 BN 的参数长度。
        """

        self.eval()
        device = self.device
        full = self.ema.module if self.ema else self.model

        # 深拷贝完整模型并放到 CUDA
        pruned_model = copy.deepcopy(full).to(device).eval()

        # 只转换 backbone 里的 FrozenBN -> BN（保持 device/dtype）
        if hasattr(pruned_model, "backbone"):
            convert_frozen_bn_to_bn(pruned_model.backbone)

        # 构造 4D CUDA dummy（只给 backbone 用；不需要 label）
        H, W = getattr(self.cfg, "val_input_size", (640, 640))
        example_inputs = torch.randn(1, 3, H, W, device=device).float()

        # 用 FX 找出分组/深度可分离 Conv 及其直接上游供给层 —— 全部忽略
        risk_ignores = _fx_find_grouped_and_producers(pruned_model.backbone, example_inputs)
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
            example_inputs=example_inputs,
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
        base_macs, base_params = tp.utils.count_ops_and_params(backbone_view, example_inputs)

        # 真正执行剪枝（不要加 no_grad）
        pruner.step()

        macs, params = tp.utils.count_ops_and_params(backbone_view, example_inputs)
        print(f"[VAL-PRUNE] MACs {base_macs/1e9:.3f}G -> {macs/1e9:.3f}G, "
              f"Params {base_params/1e6:.3f}M -> {params/1e6:.3f}M")

        # 健康检查：backbone 前向一遍
        with torch.no_grad():
            _ = backbone_view(example_inputs)

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

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )
        return
