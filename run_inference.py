# run_inference.py
# -*- coding: utf-8 -*-
import argparse
import os
import json
import yaml
from tqdm import tqdm
import time
import tempfile
import traceback
import random

from inference_core import IntegratedInferenceCore, ApiClientInferenceCore
from utils.dataset_loader import load_dataset
from utils.image_processor import slice_image, add_bbox_to_image, generate_random_windows
from utils.region_hints import build_region_hints_text

from prompts import build_text_prompt

# 允许的加载参数（避免 HfArgumentParser 报错）
_ALLOWED_LOADING_KEYS = {
    "model_name_or_path", "template", "infer_backend",
    "adapter_name_or_path", "finetuning_type",
}
_OPTIONAL_TRY_KEYS = {
    "quantization_bit", "trust_remote_code", "flash_attn", "rope_scaling",
    "max_model_len", "torch_dtype", "quantization_device_map",
}

# ----------------------------
# 配置健壮解析
# ----------------------------
def _cfg_bool(d: dict, key: str, default: bool) -> bool:
    v = (d or {}).get(key, default)
    if isinstance(v, bool): return v
    s = str(v).strip().lower()
    if s in {"1","true","yes","y","on"}: return True
    if s in {"0","false","no","n","off"}: return False
    return default

def _cfg_int(d: dict, key: str, default: int) -> int:
    """容忍 '8,000' / '8_000' / '0x2000' / 8000.0 / True 等。"""
    v = (d or {}).get(key, default)
    if isinstance(v, bool):
        print(f"[WARN] {key} 是布尔值 {v!r}，回退 {default}")
        return default
    if isinstance(v, (int, float)):
        return int(v)
    try:
        s = str(v).strip().replace(",", "").replace("_", "")
        return int(s, 0)
    except Exception:
        print(f"[WARN] {key} 值不合法: {v!r}，回退 {default}")
        return default

def _normalize_backend(val: str) -> str:
    if not val: return val
    v = str(val).strip().lower()
    alias = {"hf":"huggingface","hugging_face":"huggingface","transformers":"huggingface","hf_transformers":"huggingface"}
    return alias.get(v, v)

def _normalize_template(t: str) -> str:
    if not t: return t
    t = str(t).strip().lower()
    alias = {"qwen2.5_vl":"qwen2_vl","qwen-2.5-vl":"qwen2_vl","qwen2-vl":"qwen2_vl"}
    return alias.get(t, t)

def _sanitize_loading_args(d: dict) -> dict:
    cfg = dict(d or {})
    if "infer_backend" in cfg:
        cfg["infer_backend"] = _normalize_backend(cfg["infer_backend"])
        if cfg["infer_backend"] not in {"huggingface", "vllm", "sglang"}:
            print(f"[WARN] 无效 infer_backend={cfg['infer_backend']}，已回退默认")
            cfg.pop("infer_backend", None)
    if "template" in cfg:
        new_t = _normalize_template(cfg["template"])
        if new_t != cfg["template"]:
            print(f"[INFO] 规范 template: '{cfg['template']}' -> '{new_t}'")
            cfg["template"] = new_t
    allowed = _ALLOWED_LOADING_KEYS | _OPTIONAL_TRY_KEYS
    clean = {k: v for k, v in cfg.items() if k in allowed}
    dropped = sorted(set(cfg) - set(clean))
    if dropped:
        print(f"[WARN] 忽略未支持的加载参数：{dropped}")
    return clean

def _atomic_write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path), encoding="utf-8") as tmp:
        json.dump(obj, tmp, indent=2, ensure_ascii=False)
        tmp.flush(); os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)

def _maybe_attach_prompt(result_dict, text_prompt, images_for_prompt, inference_cfg, manifest=None):
    if not _cfg_bool(inference_cfg, "save_prompt_text", False):
        return
    max_chars = _cfg_int(inference_cfg, "save_prompt_max_chars", 8000)
    txt = text_prompt or ""
    result_dict["prompt_text"] = txt[:max_chars]
    result_dict["prompt_text_truncated"] = (len(txt) > max_chars)
    if _cfg_bool(inference_cfg, "save_prompt_images", True):
        # 兼容旧字段
        result_dict["prompt_images"] = list(images_for_prompt or [])
        # 新字段：结构化清单（更易复现/排错）
        if manifest:
            result_dict["prompt_image_manifest"] = manifest


def _get_crops_cfg(config: dict) -> dict:
    """
    兼容多种命名：优先 'crops'，其次 'slicing_random'，最后 'slicing'。
    这样即使 YAML 曾有重复 'slicing' 也不致冲突。
    """
    for key in ("crops", "slicing_random", "slicing"):
        v = config.get(key)
        if isinstance(v, dict) and any(k in v for k in ("random_sizes","max_windows_per_round","include_global")):
            return v
    return {}

# ----------------------------
# 主逻辑
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Qwen-VL 衬砌板异常检测（集成/API双模式）")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    # 1) 读取配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # 2) 初始化推理核心
    execution_cfg = config.get("execution", {})
    model_cfg = config.get("model", {})
    mode = (execution_cfg.get("mode") or "integrated").strip().lower()

    if mode == "integrated":
        print("执行模式: [集成模式]")
        loading_args = dict(model_cfg.get("loading_args") or {})
        loading_args["model_name_or_path"] = model_cfg.get("path")
        loading_args.setdefault("template", model_cfg.get("template", "qwen2_vl"))
        loading_args = _sanitize_loading_args(loading_args)
        core = IntegratedInferenceCore(loading_args)
    elif mode == "api_client":
        print("执行模式: [API 客户端]")
        core = ApiClientInferenceCore(execution_cfg.get("api_client_settings", {}))
    else:
        raise ValueError(f"未知执行模式: {mode}")

    # 3) 加载数据
    data_cfg = config.get("data", {})
    dataset_dir = data_cfg.get("data_dir")
    if not dataset_dir:
        raise ValueError("data.data_dir 未设置")
    per_cat_limit = data_cfg.get("limit_samples_per_category") or None
    dataset = load_dataset(dataset_dir, per_cat_limit)
    if not dataset:
        print("数据集为空，退出。"); core.cleanup(); return

    # 3.1 随机/限额
    if _cfg_bool(data_cfg, "shuffle", False):
        rng = random.Random(_cfg_int(data_cfg, "seed", 42))
        rng.shuffle(dataset)
    max_total = args.max_samples if args.max_samples is not None else data_cfg.get("max_samples_total")
    if max_total and int(max_total) > 0:
        dataset = dataset[:int(max_total)]
    print(f"[INFO] 将推理 {len(dataset)} 张样本。")

    # 4) 检查点 & 输出路径
    output_dir = data_cfg.get("output_dir") or "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    now_tag = time.strftime("%Y%m%d-%H%M%S")
    ckpt_name = data_cfg.get("checkpoint_filename", "auto")
    if ckpt_name == "auto":
        ckpt_name = f"checkpoint-{now_tag}.json"
    checkpoint_path = os.path.join(output_dir, ckpt_name)
    print(f"[INFO] 进度检查点：{checkpoint_path}")
    results = []

    def save_checkpoint(extra: dict | None = None):
        meta = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "dataset_dir": dataset_dir,
            "processed": len(results),
        }
        if extra: meta.update(extra)
        _atomic_write_json(checkpoint_path, {"meta": meta, "results_so_far": results})

    # —— 小工具：构建图片清单（few-shot: S#i；eval: #k）
    def _build_manifest(few_shot_examples, eval_paths):
        manifest = []
        for i, (p, _lb) in enumerate(few_shot_examples or [], 1):
            manifest.append({"path": p, "role": "fewshot", "index": i, "tag": f"S#{i}"})
        for k, p in enumerate(eval_paths or [], 1):
            manifest.append({"path": p, "role": "eval", "index": k, "tag": f"#{k}"})
        return manifest

    # 5) few-shot
    inference_cfg = config.get("inference", {}) or {}
    use_fewshot = _cfg_bool(inference_cfg, "use_fewshot", False)
    few_shot_examples = []
    if use_fewshot:
        few_dir = inference_cfg.get("fewshot_examples_dir")
        if not few_dir:
            raise ValueError("启用 use_fewshot 时必须设置 inference.fewshot_examples_dir")
        few_limit = inference_cfg.get("fewshot_limit_per_class")
        few_shot_examples = load_dataset(few_dir, few_limit)
        if not few_shot_examples:
            print("[WARN] few-shot 目录为空，已忽略 few-shot。")
            use_fewshot = False
        elif _cfg_bool(inference_cfg, "exclude_fewshot_from_eval", True):
            fs_paths = {p for p, _ in few_shot_examples}
            before = len(dataset)
            dataset = [(p, l) for p, l in dataset if p not in fs_paths]
            removed = before - len(dataset)
            if removed > 0:
                print(f"[INFO] 已从评测集中剔除 few-shot 样本：{removed} 张")

    # 6) 模式选择
    inference_mode = (inference_cfg.get("mode") or "full_image").strip()
    valid_modes = {"full_image", "sliced_image", "random_crops"}
    if inference_mode not in valid_modes:
        raise ValueError(f"未知的 inference.mode: {inference_mode}")

    # 7) 模板选择
    prompt_templates = config.get("prompts", {}) or {}
    def _get_template(key: str, fallback: str = "zero_shot") -> str:
        tpl = prompt_templates.get(key) or prompt_templates.get(fallback)
        if not tpl:
            raise ValueError(f"缺少 prompts.{key} / prompts.{fallback} 模板")
        return tpl

    try:
        if inference_mode == "full_image":
            print("\n任务: [大图完整推理]")
            use_bbox = _cfg_bool(inference_cfg, "use_bbox", False)

            # 决定模板
            if use_fewshot and use_bbox:
                prompt_key = "few_shot_with_bbox_base"; prompt_type = "few_shot_bbox"
            elif use_bbox:
                prompt_key = "bbox"; prompt_type = "bbox"
            elif use_fewshot:
                prompt_key = "few_shot_base"; prompt_type = "few_shot"
            else:
                prompt_key = "zero_shot"; prompt_type = "zero_shot"
            prompt_template = _get_template(prompt_key)

            for image_path, true_label in tqdm(dataset, desc="处理大图中"):
                # 1) BBox 可视化（可选）
                image_to_process = image_path
                if use_bbox:
                    temp_dir = os.path.join(output_dir, "temp_bbox_images")
                    os.makedirs(temp_dir, exist_ok=True)
                    out_img = os.path.join(temp_dir, os.path.basename(image_path))
                    image_to_process = add_bbox_to_image(image_path, out_img, config.get("bbox_settings", {}))

                # 2) 文本 prompt（只文字）
                num_few = len(few_shot_examples) if use_fewshot else 0
                if num_few:
                    order_hint = f"图片顺序：示例 {num_few} 张（S#1…S#{num_few}）在前；待判定 1 张（#1）在后。"
                else:
                    order_hint = "将提供 1 张待判定图（#1）。"

                # ---- 构造 placeholders，并在需要时注入 region_hints ----
                placeholders = {"image_order_hint": order_hint}

                region_hints_cfg = (inference_cfg.get("region_hints") or {})
                if bool(region_hints_cfg.get("enabled", False)):
                    try:
                        # 使用“原图”的尺寸与坐标系生成 hints 文本（与是否画红框无关）
                        placeholders["region_hints"] = build_region_hints_text(
                            image_path=image_path,
                            cfg=region_hints_cfg
                        )
                    except Exception as e:
                        print(f"[WARN] 生成 region_hints 失败（{os.path.basename(image_path)}）：{e}")
                        placeholders["region_hints"] = ""
                else:
                    placeholders["region_hints"] = ""

                text_prompt = build_text_prompt(
                    prompt_template=prompt_template,
                    few_shot_examples=few_shot_examples if use_fewshot else None,
                    show_fewshot_label=_cfg_bool(inference_cfg, "fewshot_label_hint", True),
                    placeholders=placeholders   # ← 关键：把占位字典传进去
                )


                # 3) 图片列表（few-shot 在前，待测在后）
                eval_list = [image_to_process]
                images_for_prompt = ([ex[0] for ex in few_shot_examples] if use_fewshot else []) + eval_list
                manifest = _build_manifest(few_shot_examples if use_fewshot else [], eval_list)

                # 4) 推理
                try:
                    response = core.predict(text_prompt, images_for_prompt)
                    item = {
                        "image_path": image_path,
                        "true_label": true_label,
                        "prompt_type": prompt_type,
                        "model_response": response,
                        "prompt_image_manifest": manifest,
                    }
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    print(f"[ERR] 推理失败：{image_path} -> {err}")
                    item = {
                        "image_path": image_path,
                        "true_label": true_label,
                        "prompt_type": prompt_type,
                        "error": err,
                        "traceback": traceback.format_exc(),
                        "prompt_image_manifest": manifest,
                    }

                _maybe_attach_prompt(item, text_prompt, images_for_prompt, inference_cfg)
                results.append(item)
                save_checkpoint({"last_image": os.path.basename(image_path)})

        elif inference_mode == "random_crops":
            print("\n任务: [随机裁剪 - 全局+局部]")
            crops_cfg = _get_crops_cfg(config)
            multi_image = _cfg_bool(inference_cfg, "multi_image_per_round", True)
            max_per_round = _cfg_int(crops_cfg, "max_windows_per_round", 6)
            base_images = [ex[0] for ex in few_shot_examples] if use_fewshot else []

            # 模板：多图一轮推荐 multi_crop_base；否则回退 zero_shot / few_shot_base
            if use_fewshot and multi_image:
                prompt_template = _get_template("few_shot_multi_crop", "multi_crop_base")
            elif use_fewshot and not multi_image:
                prompt_template = _get_template("few_shot_base", "zero_shot")
            elif not use_fewshot and multi_image:
                prompt_template = _get_template("multi_crop_base", "zero_shot")
            else:
                prompt_template = _get_template("zero_shot")

            for image_path, true_label in tqdm(dataset, desc="处理大图中"):
                temp_dir = os.path.join(output_dir, "temp_random_crops")
                windows = generate_random_windows(image_path, crops_cfg, temp_dir)
                win_paths = [w["path"] for w in windows][:max_per_round]

                if multi_image:
                    k = len(win_paths)
                    order_hint = f"本轮共有 {k} 张待判定图，按输入顺序编号为 #1…#{k}；若提供示例（S#1…），仅供参考。"
                    text_prompt = build_text_prompt(
                        prompt_template=prompt_template,
                        few_shot_examples=few_shot_examples if use_fewshot else None,
                        show_fewshot_label=_cfg_bool(inference_cfg, "fewshot_label_hint", True),
                        placeholders={"image_order_hint": order_hint}
                    )
                    eval_list = win_paths
                    images_for_prompt = base_images + eval_list
                    manifest = _build_manifest(few_shot_examples if use_fewshot else [], eval_list)

                    response = core.predict(text_prompt, images_for_prompt)
                    item = {
                        "image_path": image_path,
                        "true_label": true_label,
                        "prompt_type": "random_crops_multi",
                        "windows": windows,
                        "model_response": response,
                        "prompt_image_manifest": manifest,
                    }
                    _maybe_attach_prompt(item, text_prompt, images_for_prompt, inference_cfg)
                    results.append(item)

                else:
                    # 单窗多轮：每轮只有 #1
                    order_hint = "本轮按单窗逐次判定；每次仅 1 张待判定图（#1）。若提供示例（S#1…），仅供参考。"
                    text_prompt = build_text_prompt(
                        prompt_template=prompt_template,
                        few_shot_examples=few_shot_examples if use_fewshot else None,
                        show_fewshot_label=_cfg_bool(inference_cfg, "fewshot_label_hint", True),
                        placeholders={"image_order_hint": order_hint}
                    )
                    perwin = []
                    abnormal = False
                    for pth in win_paths:
                        images_for_prompt = base_images + [pth]
                        resp = core.predict(text_prompt, images_for_prompt)
                        perwin.append({"crop_path": pth, "response": resp})
                        if "【判断】: 异常" in str(resp):
                            abnormal = True

                    # 统一清单（把所有窗记为 #1…#K）
                    manifest = _build_manifest(few_shot_examples if use_fewshot else [], win_paths)
                    item = {
                        "image_path": image_path,
                        "true_label": true_label,
                        "prompt_type": "random_crops_single",
                        "windows": windows,
                        "final_decision": "异常" if abnormal else "正常",
                        "per_window": perwin,
                        "prompt_image_manifest": manifest,
                    }
                    # 在整图层记录一次 prompt（避免过大）
                    _maybe_attach_prompt(item, text_prompt, base_images + ["<random-crop>"], inference_cfg)
                    results.append(item)

                save_checkpoint({"last_image": os.path.basename(image_path)})

        elif inference_mode == "sliced_image":
            print("\n任务: [网格切片后推理]")
            slicing_cfg = config.get("slicing", {}) or {}
            slice_size = _cfg_int(slicing_cfg, "slice_size", 768)
            slice_overlap = _cfg_int(slicing_cfg, "slice_overlap", 128)

            # 模板（不含切片数，因每张大图不同，稍后逐图注入）
            base_template = _get_template("few_shot_base" if use_fewshot else "zero_shot")
            base_images = [ex[0] for ex in few_shot_examples] if use_fewshot else []
            temp_dir = os.path.join(output_dir, "temp_sliced_images")

            for image_path, true_label in tqdm(dataset, desc="处理大图中"):
                sliced_paths = slice_image(image_path, slice_size, slice_overlap, temp_dir)
                k = len(sliced_paths)
                order_hint = f"该大图被切成 {k} 张切片（#1…#{k}）；若提供示例（S#1…），仅供参考。"
                text_prompt = build_text_prompt(
                    prompt_template=base_template,
                    few_shot_examples=few_shot_examples if use_fewshot else None,
                    show_fewshot_label=_cfg_bool(inference_cfg, "fewshot_label_hint", True),
                    placeholders={"image_order_hint": order_hint}
                )

                abnormal_slice_found = False
                slice_responses = []
                for slice_path in tqdm(sliced_paths, desc="  -> 推理切片", leave=False):
                    images_for_prompt = base_images + [slice_path]
                    try:
                        response = core.predict(text_prompt, images_for_prompt)
                        slice_responses.append({"slice": os.path.basename(slice_path), "response": response})
                        if "【判断】: 异常" in str(response):
                            abnormal_slice_found = True
                    except Exception as e:
                        err = f"{type(e).__name__}: {e}"
                        print(f"[ERR] 切片推理失败：{slice_path} -> {err}")
                        slice_responses.append({"slice": os.path.basename(slice_path), "error": err})

                manifest = _build_manifest(few_shot_examples if use_fewshot else [], sliced_paths)
                final_decision = "异常" if abnormal_slice_found else "正常"
                item = {
                    "image_path": image_path,
                    "true_label": true_label,
                    "prompt_type": "sliced_image",
                    "final_decision": final_decision,
                    "slice_details": slice_responses,
                    "prompt_image_manifest": manifest,
                }
                # 记录统一的 text_prompt 与“基准图片清单”占位，避免爆文件
                _maybe_attach_prompt(item, text_prompt, base_images + ["<slice>"], inference_cfg)
                results.append(item)
                save_checkpoint({"last_image": os.path.basename(image_path)})

        else:
            raise ValueError(f"未知的 inference.mode: {inference_mode}")

    finally:
        try: core.cleanup()
        except Exception: pass

    # 8) 保存最终结果
    out_name = config.get("data", {}).get("results_filename", "results.json")
    if out_name == "auto":
        tags = []
        if use_fewshot: tags.append("fs")
        if _cfg_bool(inference_cfg, "use_bbox", False): tags.append("bbox")
        tag = ("-" + "-".join(tags)) if tags else ""
        out_name = f"{inference_mode}{tag}-{now_tag}.json"
    out_path = os.path.join(output_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print(f"所有任务完成，结果已保存：{out_path}")
    print(f"进度检查点保留：{checkpoint_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()
