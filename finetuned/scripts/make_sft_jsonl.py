#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 新的、简化的对话模板
SYSTEM_PROMPT = (
    "你是一位专业的缺陷判读专家。请根据提供的图像，判断其中是否存在缺陷。"
)
USER_PROMPT = "请判断这张图片中的区域是否存在异常，并给出结论。"

def main():
    ap = argparse.ArgumentParser(description="从 regions.jsonl 为每个 ROI 创建裁切图和 SFT 记录。")
    ap.add_argument("--regions_jsonl", required=True, help="输入的 regions.jsonl 文件路径。")
    ap.add_argument("--out_jsonl", required=True, help="输出的 SFT 格式 jsonl 文件路径。")
    ap.add_argument("--crops_dir", required=True, help="存放所有裁切后 ROI 图像的目录。")
    args = ap.parse_args()

    src_path = Path(args.regions_jsonl)
    dst_path = Path(args.out_jsonl)
    crops_path = Path(args.crops_dir)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    crops_path.mkdir(parents=True, exist_ok=True)

    print(f"源文件: {src_path}")
    print(f"输出 SFT: {dst_path}")
    print(f"裁切图目录: {crops_path}")

    with open(src_path, "r", encoding="utf-8") as f_in, open(dst_path, "w", encoding="utf-8") as f_out:
        # 使用 tqdm 显示进度条
        for line in tqdm(f_in, desc="处理 Regions"):
            try:
                it = json.loads(line)
                base_img_path = Path(it["image"])
                if not base_img_path.exists():
                    print(f"警告: 图片不存在，跳过 {base_img_path}")
                    continue
                
                base_img = Image.open(base_img_path).convert("RGB")
                pos_ids = set(it.get("positive_region_ids", []))

                # 为每一个 region 创建一条训练数据
                for r in it["regions"]:
                    rid = r["id"]
                    x1, y1, x2, y2 = r['x1'], r['y1'], r['x2'], r['y2']

                    # 裁切 ROI
                    crop_img = base_img.crop((x1, y1, x2, y2))
                    
                    # 保存裁切图
                    crop_filename = f"{base_img_path.stem}_region_{rid}.png"
                    crop_save_path = crops_path / crop_filename
                    crop_img.save(crop_save_path, "PNG")

                    # 判断该 ROI 是正样本还是负样本
                    is_positive = (rid in pos_ids)
                    
                    # 构建 Assistant 的回答
                    if is_positive:
                        assistant_content = {
                            "decision": "异常",
                            "type": r.get("label", "未知类型") # 如果有标签，也一并提供
                        }
                    else:
                        assistant_content = {
                            "decision": "正常"
                        }
                    assistant_text = json.dumps(assistant_content, ensure_ascii=False)

                    # 构建最终的 SFT 记录
                    sft_record = {
                        "messages": [
                            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": USER_PROMPT},
                                    # 关键：这里引用的是裁切后的小图路径
                                    {"type": "image", "image": str(crop_save_path)}
                                ]
                            },
                            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
                        ]
                    }
                    
                    f_out.write(json.dumps(sft_record, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"处理行失败: {line.strip()}，错误: {e}")

    print(f"成功写入 SFT 数据到 → {dst_path}")
    print(f"所有裁切图已保存到 → {crops_path}")

if __name__ == "__main__":
    main()
