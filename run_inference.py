import argparse
import os
import json
import yaml  # 导入yaml库
from tqdm import tqdm

from inference_core import VLModelInference
from prompts import get_zero_shot_prompt, get_few_shot_prompt, get_bbox_prompt
from utils.dataset_loader import load_dataset
from utils.image_processor import slice_image, add_bbox_to_image

def main():
    # --- 1. 解析参数：优先命令行，其次配置文件 ---
    parser = argparse.ArgumentParser(description="使用Qwen-VL对衬砌板进行异常检测")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件的路径。")
    # 添加一个空的 'args' 命名空间，用于接收来自命令行的覆盖参数
    args = parser.parse_args()

    # 加载配置文件
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误：找不到配置文件 {args.config}。请确保文件存在或通过 --config 指定正确路径。")
        return
    
    # 将YAML中的嵌套字典扁平化，方便后续使用
    # 例如 config['model']['path'] -> config['model_path']
    flat_config = {}
    for section, params in config.items():
        for key, value in params.items():
            flat_config[f"{section}_{key}"] = value

    # 更新 argparse，让它也能解析配置文件中的所有参数
    # 这样命令行就可以覆盖配置文件中的值
    for key, value in flat_config.items():
        # 将 a_b -> --a-b
        arg_name = '--' + key.replace('_', '-')
        parser.add_argument(arg_name, type=type(value), default=value)
    
    # 再次解析，这次会包含所有参数
    args = parser.parse_args()

    # --- 参数逻辑校验 ---
    if args.inference_use_fewshot and args.inference_use_bbox:
        parser.error("参数冲突：use_fewshot 和 use_bbox 不能同时为 true。请检查配置文件或命令行参数。")
    if args.inference_mode == "sliced_image" and (args.inference_use_fewshot or args.inference_use_bbox):
        print("警告：在 'sliced_image' 模式下，use_fewshot 和 use_bbox 参数将被忽略。")

    # --- 2. 初始化模型 ---
    model = VLModelInference(args.model_path)

    # --- 3. 加载数据集 ---
    dataset = load_dataset(args.data_data_dir)
    if not dataset:
        return

    # --- 4. 准备输出目录和结果文件 ---
    os.makedirs(args.data_output_dir, exist_ok=True)
    results = []

    # --- 5. 根据模式执行推理 (与之前版本基本相同，只是参数来源变为args) ---
    if args.inference_mode == "full_image":
        print("模式: [大图完整推理]")
        output_file = os.path.join(args.data_output_dir, 'full_image_results.json')
        
        few_shot_examples = []
        if args.inference_use_fewshot:
            if not args.inference_fewshot_examples_dir:
                raise ValueError("启用few-shot模式必须在配置文件或命令行中提供 'fewshot_examples_dir'")
            few_shot_examples = load_dataset(args.inference_fewshot_examples_dir)
            print(f"已加载 {len(few_shot_examples)} 个Few-shot示例。")

        for image_path, true_label in tqdm(dataset, desc="处理大图中"):
            image_to_process = image_path
            
            if args.inference_use_bbox:
                temp_bbox_dir = os.path.join(args.data_output_dir, 'temp_bbox_images')
                os.makedirs(temp_bbox_dir, exist_ok=True)
                bbox_image_path = os.path.join(temp_bbox_dir, os.path.basename(image_path))
                image_to_process = add_bbox_to_image(image_path, bbox_image_path)
                prompt = get_bbox_prompt()
                images_for_prompt = [image_to_process]
            elif args.inference_use_fewshot:
                prompt = get_few_shot_prompt(few_shot_examples)
                images_for_prompt = [ex[0] for ex in few_shot_examples] + [image_to_process]
            else:
                prompt = get_zero_shot_prompt()
                images_for_prompt = [image_to_process]

            response = model.predict(prompt, images_for_prompt)
            results.append({
                "image_path": image_path,
                "true_label": true_label,
                "model_response": response,
                "prompt_type": "bbox" if args.inference_use_bbox else ("few_shot" if args.inference_use_fewshot else "zero_shot")
            })

    elif args.inference_mode == "sliced_image":
        print("模式: [大图切割后推理]")
        output_file = os.path.join(args.data_output_dir, 'sliced_image_results.json')
        temp_slice_dir = os.path.join(args.data_output_dir, 'temp_sliced_images')
        os.makedirs(temp_slice_dir, exist_ok=True)
        
        prompt = get_zero_shot_prompt()

        for image_path, true_label in tqdm(dataset, desc="处理切割图中"):
            sliced_paths = slice_image(
                image_path,
                args.slicing_slice_size,
                args.slicing_slice_overlap,
                temp_slice_dir
            )
            
            abnormal_slice_found = False
            slice_responses = []
            for slice_path in tqdm(sliced_paths, desc=f"推理切片", leave=False):
                response = model.predict(prompt, [slice_path])
                slice_responses.append({"slice": slice_path, "response": response})
                if "【判断】: 异常" in response:
                    abnormal_slice_found = True
            
            final_decision = "异常" if abnormal_slice_found else "正常"
            results.append({
                "image_path": image_path,
                "true_label": true_label,
                "final_decision": final_decision,
                "slice_details": slice_responses
            })

    # --- 6. 保存结果 ---
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"所有任务完成！结果已保存至: {output_file}")
    
    # --- 7. 清理 ---
    model.cleanup()


if __name__ == "__main__":
    main()
