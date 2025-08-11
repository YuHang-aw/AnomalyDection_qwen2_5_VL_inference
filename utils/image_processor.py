from PIL import Image, ImageDraw
import os

# 禁用Pillow的解压炸弹检查，因为我们要处理大图
Image.MAX_IMAGE_PIXELS = None

def slice_image(image_path: str, slice_size: int, overlap: int, output_dir: str) -> list[str]:
    """
    将大图切割成带重叠的小图。

    Args:
        image_path (str): 大图的路径。
        slice_size (int): 切割后小图的尺寸 (正方形)。
        overlap (int): 小图之间的重叠像素。
        output_dir (str): 存放小图的临时目录。

    Returns:
        list[str]: 所有生成的小图路径列表。
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"错误: 找不到图片文件 {image_path}")
        return []

    img_w, img_h = img.size
    stride = slice_size - overlap
    sliced_image_paths = []

    # 创建独立的子目录存放每个大图的切片
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    slice_specific_dir = os.path.join(output_dir, base_name)
    os.makedirs(slice_specific_dir, exist_ok=True)

    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            # 确保不会切出边界
            box = (x, y, min(x + slice_size, img_w), min(y + slice_size, img_h))
            
            # 如果切片太小，可以跳过
            if (box[2] - box[0]) < overlap or (box[3] - box[1]) < overlap:
                continue

            slice_img = img.crop(box)
            
            slice_filename = f"slice_{y}_{x}.png"
            slice_path = os.path.join(slice_specific_dir, slice_filename)
            slice_img.save(slice_path)
            sliced_image_paths.append(slice_path)
            
    return sliced_image_paths

def add_bbox_to_image(image_path: str, output_path: str):
    """
    在图片中心添加一个示例BBox并保存。
    在实际应用中，BBox坐标应该来自检测算法或标注数据。
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # 示例：在中心绘制一个占图片1/4大小的红色方框
    w, h = img.size
    box_w, box_h = w // 2, h // 2
    x1, y1 = (w - box_w) // 2, (h - box_h) // 2
    x2, y2 = x1 + box_w, y1 + box_h
    
    draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
    
    img.save(output_path)
    return output_path
