import argparse
import os
import json
import yaml
from tqdm import tqdm
import time

# 核心模块：根据配置选择不同的推理引擎
from inference_core import IntegratedInferenceCore, ApiClientInferenceCore

# 工具模块：数据加载和图像处理
from utils.dataset_loader import load_dataset
from utils.image_processor import slice_image, add_bbox_to_image

# Prompt构建模块
from prompts import get_final_prompt

def main():
    # --- 1. 加载配置 ---
    # 使用 argparse 仅用于指定配置文件路径，使脚本更具通用性
    parser = argparse.ArgumentParser(description="Qwen-VL 衬砌板异常检测框架 (双模式)")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件的路径。")
    args = parser.parse_args()

    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误：找不到配置文件 '{args.config}'。请确保文件存在或通过 --config 指定正确路径。")
        return

    # --- 2. 根据配置初始化推理核心 (Core) ---
    execution_cfg = config.get('execution', {})
    model_cfg = config.get('model', {})
    core = None

    print("="*50)
    if execution_cfg.get('mode') == 'integrated':
        print("执行模式: [集成模式 (Integrated)] - 脚本将直接加载模型。")
        loading_args = model_cfg.get('loading_args', {})
        loading_args['model_name_or_path'] = model_cfg.get('path')
        core = IntegratedInferenceCore(loading_args)

    elif execution_cfg.get('mode') == 'api_client':
        print("执行模式: [API客户端模式 (API Client)] - 脚本将请求远程API服务。")
        api_settings = execution_cfg.get('api_client_settings', {})
        core = ApiClientInferenceCore(api_settings)
    else:
        raise ValueError(f"配置文件中的执行模式未知或未指定: '{execution_cfg.get('mode')}'")
    print("="*50)

    # --- 3. 加载数据集和准备 ---
    data_cfg = config.get('data', {})
    limit = data_cfg.get('limit_samples_per_category') or None
    dataset = load_dataset(data_cfg.get('data_dir'), limit)
    if not dataset:
        return

    os.makedirs(data_cfg.get('output_dir'), exist_ok=True)
    results = []

    # --- 4. 预加载 Few-shot 示例 (如果需要) ---
    inference_cfg = config.get('inference', {})
    few_shot_examples = []
    if inference_cfg.get('use_fewshot'):
        fewshot_dir = inference_cfg.get('fewshot_examples_dir')
        if not fewshot_dir:
            raise ValueError("配置中启用了 'use_fewshot'，但未提供 'fewshot_examples_dir'。")
        few_shot_examples = load_dataset(fewshot_dir)
        print(f"已加载 {len(few_shot_examples)} 个 Few-shot 示例。")

    # --- 5. 执行推理循环 ---
    inference_mode = inference_cfg.get('mode')
    prompt_templates = config.get('prompts', {})

    if inference_mode == "full_image":
        # ... full_image 推理逻辑 ...
        print("\n推理任务: [大图完整推理 (Full Image)]")
        use_fewshot = inference_cfg.get('use_fewshot', False)
        use_bbox = inference_cfg.get('use_bbox', False)
        
        prompt_type = "zero_shot"
        prompt_template = prompt_templates.get('zero_shot')

        if use_fewshot and use_bbox:
            prompt_type = "few_shot_with_bbox"
            prompt_template = prompt_templates.get('few_shot_with_bbox_base')
            print("增强功能: [Few-shot] + [BBox]")
        elif use_bbox:
            prompt_type = "bbox"
            prompt_template = prompt_templates.get('bbox')
            print("增强功能: [BBox]")
        elif use_fewshot:
            prompt_type = "few_shot"
            prompt_template = prompt_templates.get('few_shot_base')
            print("增强功能: [Few-shot]")

        for image_path, true_label in tqdm(dataset, desc="处理大图中"):
            image_to_process = image_path
            images_for_prompt = []

            # 如果使用BBox，处理图片并构建Prompt
            if use_bbox:
                temp_bbox_dir = os.path.join(data_cfg.get('output_dir'), 'temp_bbox_images')
                os.makedirs(temp_bbox_dir, exist_ok=True)
                bbox_image_path = os.path.join(temp_bbox_dir, os.path.basename(image_path))
                image_to_process = add_bbox_to_image(image_path, bbox_image_path, config.get('bbox_settings', {}))
            
            # 构建最终的Prompt和图片列表
            prompt = get_final_prompt(prompt_template, few_shot_examples)
            if use_fewshot:
                images_for_prompt = [ex[0] for ex in few_shot_examples]
            images_for_prompt.append(image_to_process)
            
            response = core.predict(prompt, images_for_prompt)
            results.append({
                "image_path": image_path,
                "true_label": true_label,
                "model_response": response,
                "prompt_type": prompt_type
            })

    elif inference_mode == "sliced_image":
        # ... sliced_image 推理逻辑 ...
        print("\n推理任务: [大图切割后推理 (Sliced Image)]")
        slicing_cfg = config.get('slicing', {})
        use_fewshot = inference_cfg.get('use_fewshot', False)
        prompt_type = "sliced_zero_shot"
        prompt_template = prompt_templates.get('zero_shot')

        if use_fewshot:
            prompt_type = "sliced_with_few_shot"
            prompt_template = prompt_templates.get('few_shot_base')
            print("增强功能: [Few-shot on Slices]")

        temp_slice_dir = os.path.join(data_cfg.get('output_dir'), 'temp_sliced_images')
        
        for image_path, true_label in tqdm(dataset, desc="处理大图中"):
            sliced_paths = slice_image(
                image_path,
                slicing_cfg.get('slice_size', 768),
                slicing_cfg.get('slice_overlap', 128),
                temp_slice_dir
            )
            
            abnormal_slice_found = False
            slice_responses = []
            
            prompt = get_final_prompt(prompt_template, few_shot_examples)
            base_images = [ex[0] for ex in few_shot_examples] if use_fewshot else []

            for slice_path in tqdm(sliced_paths, desc=f"  -> 推理切片", leave=False):
                images_for_prompt = base_images + [slice_path]
                response = core.predict(prompt, images_for_prompt)
                slice_responses.append({"slice": os.path.basename(slice_path), "response": response})
                if "【判断】: 异常" in response:
                    abnormal_slice_found = True
            
            final_decision = "异常" if abnormal_slice_found else "正常"
            results.append({
                "image_path": image_path,
                "true_label": true_label,
                "final_decision": final_decision,
                "prompt_type": prompt_type,
                "slice_details": slice_responses
            })

    # --- 6. 保存结果 ---
    output_filename = data_cfg.get('results_filename', 'results.json')
    if output_filename == 'auto':
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        enhancements = []
        if inference_cfg.get('use_fewshot'): enhancements.append('fs')
        if inference_cfg.get('use_bbox'): enhancements.append('bbox')
        enh_str = ('-' + '-'.join(enhancements)) if enhancements else ''
        output_filename = f"{inference_mode}{enh_str}-{timestamp}.json"
    
    output_file_path = os.path.join(data_cfg.get('output_dir'), output_filename)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print("\n" + "="*50)
    print(f"所有任务完成！结果已保存至: {output_file_path}")
    print("="*50)
    
    # --- 7. 清理资源 ---
    core.cleanup()

if __name__ == "__main__":
    main()

