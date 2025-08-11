# -*- coding: utf-8 -*-
import argparse
import os
import json
import yaml
from tqdm import tqdm
import time
import tempfile
import traceback

# 核心模块：根据配置选择不同的推理引擎
from inference_core import IntegratedInferenceCore, ApiClientInferenceCore

# 工具模块：数据加载和图像处理（注意：你的 utils 里 load_dataset 要支持 (data_dir, limit) 两参）
from utils.dataset_loader import load_dataset
from utils.image_processor import slice_image, add_bbox_to_image

# Prompt 构建模块（新版：只产出“文本”，图片分开传）
from prompts import build_text_prompt


# ----------------------------
# 仅保留 ChatModel 在推理阶段常用、且不会引发 HfArgumentParser 报错的键
# （如果你的 YAML 里还有别的键，建议放到启动 API 的 YAML 里用。）
# ----------------------------
_ALLOWED_LOADING_KEYS = {
    # 文档明确：推理必需/可选键
    "model_name_or_path",
    "template",
    "infer_backend",          # "huggingface" / "vllm" / "sglang"（通常只用前两者）
    "adapter_name_or_path",   # 若使用 LoRA/适配器
    "finetuning_type",        # lora / qlora / full ...
}

_OPTIONAL_TRY_KEYS = {
    "quantization_bit",
    "trust_remote_code",
    "flash_attn",
    "rope_scaling",
    "max_model_len",
    "torch_dtype",
    "quantization_device_map",
}


def _normalize_backend(val: str) -> str:
    if not val:
        return val
    v = str(val).strip().lower()
    alias = {
        "hf": "huggingface",
        "hugging_face": "huggingface",
        "transformers": "huggingface",
        "hf_transformers": "huggingface",
    }
    return alias.get(v, v)


def _normalize_template(t: str) -> str:
    if not t:
        return t
    t = str(t).strip().lower()
    alias = {
        "qwen2.5_vl": "qwen2_vl",
        "qwen-2.5-vl": "qwen2_vl",
        "qwen2-vl": "qwen2_vl",
    }
    return alias.get(t, t)


def _sanitize_loading_args(d: dict) -> dict:
    cfg = dict(d or {})
    # 轻量规范化
    if "infer_backend" in cfg:
        cfg["infer_backend"] = _normalize_backend(cfg["infer_backend"])
        if cfg["infer_backend"] not in {"huggingface", "vllm", "sglang"}:
            print(f"[WARN] 无效 infer_backend={cfg['infer_backend']}，已回退默认。")
            cfg.pop("infer_backend", None)
    if "template" in cfg:
        new_t = _normalize_template(cfg["template"])
        if new_t != cfg["template"]:
            print(f"[INFO] 将 template '{cfg['template']}' 规范为 '{new_t}'")
            cfg["template"] = new_t

    allowed = _ALLOWED_LOADING_KEYS | _OPTIONAL_TRY_KEYS
    clean = {k: v for k, v in cfg.items() if k in allowed}
    dropped = sorted(set(cfg) - set(clean))
    if dropped:
        print(f"[WARN] 已忽略未在允许列表中的参数键：{dropped}")
    return clean


def _atomic_write_json(path: str, obj) -> None:
    """原子写入，防止中途损坏。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path), encoding="utf-8") as tmp:
        json.dump(obj, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def main():
    # --- 1) 读取配置 ---
    parser = argparse.ArgumentParser(description="Qwen-VL 衬砌板异常检测（支持集成/API双模式）")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误：找不到配置文件 '{args.config}'")
        return

    # --- 2) 初始化推理核心 ---
    execution_cfg = config.get("execution", {})
    model_cfg = config.get("model", {})
    core = None

    print("=" * 50)
    mode = execution_cfg.get("mode")
    if mode == "integrated":
        print("执行模式: [集成模式] - 直接加载本地模型")
        loading_args = dict(model_cfg.get("loading_args") or {})
        # 兜底：把 path 写回 ChatModel 需要的键名
        loading_args["model_name_or_path"] = model_cfg.get("path")
        # 若未指定模板，给出 Qwen2-VL 常用模板兜底值
        loading_args.setdefault("template", model_cfg.get("template", "qwen2_vl"))
        # 过滤未知键，避免 HfArgumentParser 报错
        loading_args = _sanitize_loading_args(loading_args)
        core = IntegratedInferenceCore(loading_args)

    elif mode == "api_client":
        print("执行模式: [API 客户端] - 请求已启动的 LLaMA-Factory OpenAI 兼容服务")
        api_settings = execution_cfg.get("api_client_settings", {})
        core = ApiClientInferenceCore(api_settings)
    else:
        raise ValueError(f"未知或未指定的执行模式: {mode}")
    print("=" * 50)

    # --- 3) 加载数据集 ---
    data_cfg = config.get("data", {})
    limit = data_cfg.get("limit_samples_per_category") or None
    dataset_dir = data_cfg.get("data_dir")
    if not dataset_dir:
        raise ValueError("data.data_dir 未设置")
    dataset = load_dataset(dataset_dir, limit)
    if not dataset:
        print("数据集为空，已退出。")
        core.cleanup()
        return

    output_dir = data_cfg.get("output_dir") or "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    results = []

    # --- 3.1) 检查点文件名（带时间戳） ---
    now_tag = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_filename = data_cfg.get("checkpoint_filename", "auto")
    if checkpoint_filename == "auto":
        checkpoint_filename = f"checkpoint-{now_tag}.json"
    checkpoint_path = os.path.join(output_dir, checkpoint_filename)
    print(f"[INFO] 进度检查点：{checkpoint_path}")

    # --- 4) 预加载 few-shot 示例（如启用） ---
    inference_cfg = config.get("inference", {})
    few_shot_examples = []
    if inference_cfg.get("use_fewshot"):
        few_dir = inference_cfg.get("fewshot_examples_dir")
        if not few_dir:
            raise ValueError("启用 use_fewshot 时必须提供 fewshot_examples_dir")
        few_limit = inference_cfg.get("fewshot_limit_per_class")  # 可选参数
        few_shot_examples = load_dataset(few_dir, few_limit)
        if not few_shot_examples:
            print("[WARN] few-shot 目录为空，已忽略 few-shot。")
        else:
            print(f"已加载 {len(few_shot_examples)} 个 few-shot 示例。")

        # ☆（可选）把 few-shot 样本从评测集里剔除，避免同图既当示例又当测试
        if inference_cfg.get("exclude_fewshot_from_eval", True):
            fs_set = {p for p, _ in few_shot_examples}
            before = len(dataset)
            dataset = [(p, l) for p, l in dataset if p not in fs_set]
            removed = before - len(dataset)
            if removed > 0:
                print(f"[INFO] 已从评测集中剔除 few-shot 样本：{removed} 张")

    # --- 5) 推理循环 ---
    inference_mode = inference_cfg.get("mode")
    if inference_mode not in {"full_image", "sliced_image"}:
        raise ValueError(f"未知的 inference.mode: {inference_mode}")

    prompt_templates = config.get("prompts", {})
    system_prompt = prompt_templates.get("system", "")  # 如需 system，可在 core 内部扩展支持

    # 统一的检查点保存器：每处理一张图就写一次，防止中途断电丢数据
    def save_checkpoint(extra_meta: dict | None = None):
        meta = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "inference_mode": inference_mode,
            "dataset_dir": dataset_dir,
            "processed": len(results),
        }
        if extra_meta:
            meta.update(extra_meta)
        payload = {"meta": meta, "results_so_far": results}
        _atomic_write_json(checkpoint_path, payload)

    try:
        if inference_mode == "full_image":
            print("\n任务: [大图完整推理]")
            use_fewshot = inference_cfg.get("use_fewshot", False)
            use_bbox = inference_cfg.get("use_bbox", False)

            prompt_type = "zero_shot"
            prompt_template_key = "zero_shot"

            if use_fewshot and use_bbox:
                prompt_type = "few_shot_with_bbox"
                prompt_template_key = "few_shot_with_bbox_base"
                print("增强: [Few-shot] + [BBox]")
            elif use_bbox:
                prompt_type = "bbox"
                prompt_template_key = "bbox"
                print("增强: [BBox]")
            elif use_fewshot:
                prompt_type = "few_shot"
                prompt_template_key = "few_shot_base"
                print("增强: [Few-shot]")

            prompt_template = prompt_templates.get(prompt_template_key)
            if not prompt_template:
                raise ValueError(f"prompts.{prompt_template_key} 未在配置中定义")

            for image_path, true_label in tqdm(dataset, desc="处理大图中"):
                image_to_process = image_path
                images_for_prompt = []

                # BBox 预处理
                if use_bbox:
                    temp_bbox_dir = os.path.join(output_dir, "temp_bbox_images")
                    os.makedirs(temp_bbox_dir, exist_ok=True)
                    bbox_image_path = os.path.join(temp_bbox_dir, os.path.basename(image_path))
                    image_to_process = add_bbox_to_image(
                        image_path, bbox_image_path, config.get("bbox_settings", {})
                    )

                # 1) 文本提示词：只生成文本，不含 <image>
                text_prompt = build_text_prompt(
                    prompt_template=prompt_template,
                    few_shot_examples=few_shot_examples if use_fewshot else None
                )

                # 2) 图片：作为独立列表传给引擎（顺序：few-shot 在前，待测在后）
                if use_fewshot:
                    images_for_prompt = [ex[0] for ex in few_shot_examples]
                images_for_prompt.append(image_to_process)

                # 推理
                try:
                    response = core.predict(text_prompt, images_for_prompt)
                    result_item = {
                        "image_path": image_path,
                        "true_label": true_label,
                        "model_response": response,
                        "prompt_type": prompt_type
                    }
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    print(f"[ERR] 推理失败：{image_path} -> {err}")
                    result_item = {
                        "image_path": image_path,
                        "true_label": true_label,
                        "error": err,
                        "traceback": traceback.format_exc(),
                        "prompt_type": prompt_type
                    }

                results.append(result_item)
                # 写检查点
                save_checkpoint({"last_image": os.path.basename(image_path)})

        elif inference_mode == "sliced_image":
            print("\n任务: [大图切割后推理]")
            slicing_cfg = config.get("slicing", {})
            use_fewshot = inference_cfg.get("use_fewshot", False)

            prompt_type = "sliced_zero_shot"
            prompt_template_key = "zero_shot"
            if use_fewshot:
                prompt_type = "sliced_with_few_shot"
                prompt_template_key = "few_shot_base"
                print("增强: [Few-shot on Slices]")

            prompt_template = prompt_templates.get(prompt_template_key)
            if not prompt_template:
                raise ValueError(f"prompts.{prompt_template_key} 未在配置中定义")

            temp_slice_dir = os.path.join(output_dir, "temp_sliced_images")

            # 先构建一次文本 prompt（对所有切片通用）
            text_prompt = build_text_prompt(
                prompt_template=prompt_template,
                few_shot_examples=few_shot_examples if use_fewshot else None
            )
            base_images = [ex[0] for ex in few_shot_examples] if use_fewshot else []

            for image_path, true_label in tqdm(dataset, desc="处理大图中"):
                sliced_paths = slice_image(
                    image_path,
                    slicing_cfg.get("slice_size", 768),
                    slicing_cfg.get("slice_overlap", 128),
                    temp_slice_dir
                )

                abnormal_slice_found = False
                slice_responses = []

                for slice_path in tqdm(sliced_paths, desc="  -> 推理切片", leave=False):
                    images_for_prompt = base_images + [slice_path]
                    try:
                        response = core.predict(text_prompt, images_for_prompt)
                        slice_responses.append({
                            "slice": os.path.basename(slice_path),
                            "response": response
                        })
                        if "【判断】: 异常" in str(response):
                            abnormal_slice_found = True
                    except Exception as e:
                        err = f"{type(e).__name__}: {e}"
                        print(f"[ERR] 切片推理失败：{slice_path} -> {err}")
                        slice_responses.append({
                            "slice": os.path.basename(slice_path),
                            "error": err
                        })

                final_decision = "异常" if abnormal_slice_found else "正常"
                results.append({
                    "image_path": image_path,
                    "true_label": true_label,
                    "final_decision": final_decision,
                    "prompt_type": prompt_type,
                    "slice_details": slice_responses
                })

                # 写检查点
                save_checkpoint({"last_image": os.path.basename(image_path)})

        else:
            raise ValueError(f"未知的 inference.mode: {inference_mode}")

    finally:
        # 无论是否异常，最后都清理
        try:
            core.cleanup()
        except Exception:
            pass

    # --- 6) 保存最终结果 ---
    output_filename = data_cfg.get("results_filename", "results.json")
    if output_filename == "auto":
        timestamp = now_tag
        tags = []
        if inference_cfg.get("use_fewshot"): tags.append("fs")
        if inference_cfg.get("use_bbox"): tags.append("bbox")
        tag_str = ("-" + "-".join(tags)) if tags else ""
        output_filename = f"{inference_mode}{tag_str}-{timestamp}.json"

    out_path = os.path.join(output_dir, output_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 50)
    print(f"所有任务完成，结果已保存：{out_path}")
    print(f"进度检查点保留：{checkpoint_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
