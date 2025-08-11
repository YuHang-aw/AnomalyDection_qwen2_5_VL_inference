# -*- coding: utf-8 -*-
import os
from typing import List
from PIL import Image, ImageDraw

def slice_image(image_path: str, slice_size: int, slice_overlap: int, out_dir: str) -> List[str]:
    """
    把大图切成固定大小的小图（方块），有重叠。
    返回小图路径列表。
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            W, H = im.size
            step = max(1, slice_size - max(0, slice_overlap))
            paths = []
            basename = os.path.splitext(os.path.basename(image_path))[0]

            y = 0
            while y < H:
                x = 0
                y2 = min(H, y + slice_size)
                if y2 - y < slice_size and y != 0:
                    y = max(0, H - slice_size)
                    y2 = H
                while x < W:
                    x2 = min(W, x + slice_size)
                    if x2 - x < slice_size and x != 0:
                        x = max(0, W - slice_size)
                        x2 = W
                    crop = im.crop((x, y, x2, y2))
                    # 填补边缘不足的块（右下角处），统一输出尺寸
                    if crop.size != (slice_size, slice_size):
                        pad = Image.new("RGB", (slice_size, slice_size), (0, 0, 0))
                        pad.paste(crop, (0, 0))
                        crop = pad

                    out_name = f"{basename}_y{y}_x{x}.png"
                    out_path = os.path.join(out_dir, out_name)
                    crop.save(out_path, format="PNG")
                    paths.append(out_path)

                    if x2 == W:
                        break
                    x += step
                if y2 == H:
                    break
                y += step

            return paths
    except Exception as e:
        print(f"[ERR] 切图失败：{image_path} -> {e}")
        return []

def add_bbox_to_image(image_path: str, out_path: str, cfg: dict) -> str:
    """
    在图像中心画一个相对尺寸的方框。失败则回退返回原图路径。
    """
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            W, H = im.size
            rel = float(cfg.get("relative_size", 0.5))
            rel = max(0.05, min(rel, 0.95))
            side = int(min(W, H) * rel)
            cx, cy = W // 2, H // 2
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(W - 1, x1 + side)
            y2 = min(H - 1, y1 + side)

            color = cfg.get("outline_color", "red")
            width = int(cfg.get("outline_width", 5))

            draw = ImageDraw.Draw(im)
            for i in range(width):
                draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            im.save(out_path)
            return out_path
    except Exception as e:
        print(f"[WARN] 画框失败，使用原图：{image_path} -> {e}")
        return image_path
