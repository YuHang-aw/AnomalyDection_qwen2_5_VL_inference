import argparse
import os
import json
import yaml
from tqdm import tqdm

from inference_core import VLModelInference
# 导入新的prompt函数
from prompts import get_zero_shot_prompt, get_few_shot_prompt, get_bbox_prompt, get_few_shot_with_bbox_prompt
from utils.dataset_loader import load_dataset
from utils.image_processor import slice_image, add_bbox_to_image

def main():
    # --- 1. 解析参数 (与上一版相同) ---
    parser = argparse.ArgumentParser(description="使用Qwen-VL对衬砌板进行异常检测")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件的路径。")
    args = parser.parse_args()
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误：找不到配置文件 {args.config}。")
        return
    
    flat_config = {}
    for section, params in config.items():
        for key, value in params.items():
            flat_config[f"{section}_{key}"] = value

    for key, value in flat_config.items():
        arg_name = '--' + key.replace('_', '-')
        parser.add_argument(arg_name, type=type(value), default=value)
    
    args = parser.parse_args()

    # --- 参数逻辑校验 (更新) ---
    if args.inference_mode == "sliced_image" and args.inference_use_bbox:
        # 警告并强制关闭bbox，因为在切片模式下无意义
        print("警告：在 'sliced_image' 模式下，'use_bbox' 参数无效并将被忽略。")
        args.inference_use_bbox = False

    # --- 2. 初始化模型和加载数据集 (不变) ---
    model = VLModelInference(args.model_path)
    dataset = load_dataset(args.data_data_dir)
    if not dataset: return
    os.makedirs(args.data_output_dir, exist_ok=True)
    results = []

    # 准备Few-shot示例（如果任何模式需要）
    few_shot_examples = []
    if args.inference_use_fewshot:
        if not args.inference_fewshot_examples_dir:
            raise ValueError("启用few-shot模式必须提供 'fewshot_examples_dir'")
        few_shot_examples = load_dataset(args.inference_fewshot_examples_dir)
        print(f"已加载 {len(few_shot_examples)} 个Few-shot示例。")

    # --- 3. 根据模式执行推理 (逻辑重构) ---
    if args.inference_mode == "full_image":
        print("模式: [大图完整推理]")
        output_file = os.path.join(args.data_output_dir, 'full_image_results.json')

        for image_path, true_label in tqdm(dataset, desc="处理大图中"):
            image_to_process = image_path
            prompt_type = ""
            
            # 准备BBox（如果启用）
            if args.inference_use_bbox:
                temp_bbox_dir = os.path.join(args.data_output_dir, 'temp_bbox_images')
                os.makedirs(temp_bbox_dir, exist_ok=True)
                bbox_image_path = os.path.join(temp_bbox_dir, os.path.basename(image_path))
                image_to_process = add_bbox_to_image(image_path, bbox_image_path)

            # 决定使用哪个Prompt和图片列表
            if args.inference_use_fewshot and args.inference_use_bbox:
                prompt = get_few_shot_with_bbox_prompt(few_shot_examples)
                images_for_prompt = [ex[0] for ex in few_shot_examples] + [image_to_process]
                prompt_type = "few_shot_with_bbox"
            elif args.inference_use_bbox:
                prompt = get_bbox_prompt()
                images_for_prompt = [image_to_process]
                prompt_type = "bbox"
            elif args.inference_use_fewshot:
                prompt = get_few_shot_prompt(few_shot_examples)
                images_for_prompt = [ex[0] for ex in few_shot_examples] + [image_to_process]
                prompt_type = "few_shot"
            else: # Zero-shot
                prompt = get_zero_shot_prompt()
                images_for_prompt = [image_to_process]
                prompt_type = "zero_shot"

            response = model.predict(prompt, images_for_prompt)
            results.append({
                "image_path": image_path,
                "true_label": true_label,
                "model_response": response,
                "prompt_type": prompt_type
            })

    elif args.inference_mode == "sliced_image":
        print("模式: [大图切割后推理]")
        if args.inference_use_fewshot:
            print("增强功能: [Few-shot已启用] (注意: 性能开销较大)")
        output_file = os.path.join(args.data_output_dir, 'sliced_image_results.json')
        temp_slice_dir = os.path.join(args.data_output_dir, 'temp_sliced_images')
        
        for image_path, true_label in tqdm(dataset, desc="处理切割图中"):
            sliced_paths = slice_image(
                image_path, args.slicing_slice_size, args.slicing_slice_overlap, temp_slice_dir
            )
            
            abnormal_slice_found = False
            slice_responses = []

            # 为切片准备prompt和图片列表
            if args.inference_use_fewshot:
                prompt = get_few_shot_prompt(few_shot_examples)
                # 示例图片需要为每个切片都传入
                base_images_for_prompt = [ex[0] for ex in few_shot_examples]
            else:
                prompt = get_zero_shot_prompt()
                base_images_for_prompt = []

            for slice_path in tqdm(sliced_paths, desc=f"推理切片", leave=False):
                images_for_prompt = base_images_for_prompt + [slice_path]
                response = model.predict(prompt, images_for_prompt)
                slice_responses.append({"slice": slice_path, "response": response})
                if "【判断】: 异常" in response:
                    abnormal_slice_found = True
            
            final_decision = "异常" if abnormal_slice_found else "正常"
            results.append({
                "image_path": image_path,
                "true_label": true_label,
                "final_decision": final_decision,
                "prompt_type": "sliced_with_few_shot" if args.inference_use_fewshot else "sliced_zero_shot",
                "slice_details": slice_responses
            })

    # --- 4. 保存和清理 (不变) ---
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"所有任务完成！结果已保存至: {output_file}")
    model.cleanup()

if __name__ == "__main__":
    main()