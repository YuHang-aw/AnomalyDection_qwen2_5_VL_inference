# -*- coding: utf-8 -*-
import os, math, random
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _basename_wo_ext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def make_global_view(image_path: str,
                     out_dir: str,
                     max_side: Optional[int] = None) -> str:
    """
    生成全局视图文件：
    - 若 max_side 为空或图像较小：直接返回原图路径（不复制）
    - 否则生成一份等比缩放副本，最长边=max_side
    """
    try:
        if not max_side or max_side <= 0:
            return image_path
        _ensure_dir(out_dir)
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            W, H = im.size
            side = max(W, H)
            if side <= max_side:
                return image_path
            scale = max_side / float(side)
            newW, newH = max(1, int(W * scale)), max(1, int(H * scale))
            im = im.resize((newW, newH), Image.BICUBIC)
            out_path = os.path.join(out_dir, f"{_basename_wo_ext(image_path)}_global_{max_side}.png")
            im.save(out_path)
            return out_path
    except Exception as e:
        print(f"[WARN] make_global_view 失败：{image_path} -> {e}")
        return image_path

def random_crops(image_path: str,
                 sizes: List[int],
                 per_size: int,
                 jitter: float,
                 out_dir: str,
                 seed: int = 42,
                 max_total: Optional[int] = None) -> List[Dict]:
    """
    随机裁剪若干窗口：
    - sizes: 例如 [512, 768]
    - per_size: 每个尺寸采样多少个窗
    - jitter: 尺寸抖动比例，如 0.15 表示在 [size*(1-0.15), size*(1+0.15)] 内浮动
    - 返回: [{path, x1,y1,x2,y2, size}, ...]
    """
    _ensure_dir(out_dir)
    results: List[Dict] = []
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            W, H = im.size

            rnd = random.Random(seed)
            def _rand_int(a, b):  # 包含端点
                return rnd.randint(a, b) if b >= a else a

            for base_size in sizes:
                for _ in range(max(0, per_size)):
                    # 抖动后的尺寸
                    low = max(32, int(base_size * (1 - jitter)))
                    high = int(base_size * (1 + jitter))
                    s = _rand_int(low, high)

                    # 若窗口大于图像边，仍允许，但需贴边裁
                    s = min(s, max(W, H))  # 不至于溢出太夸张

                    # 计算合法范围（使窗口完全落在图内）
                    x_max = max(0, W - s)
                    y_max = max(0, H - s)
                    x1 = _rand_int(0, x_max) if x_max > 0 else 0
                    y1 = _rand_int(0, y_max) if y_max > 0 else 0
                    x2, y2 = x1 + s, y1 + s
                    x2 = min(W, x2)
                    y2 = min(H, y2)

                    crop = im.crop((x1, y1, x2, y2))
                    # 统一输出为正方形尺寸 s×s（边界不足时补黑）
                    if crop.size != (s, s):
                        pad = Image.new("RGB", (s, s), (0, 0, 0))
                        pad.paste(crop, (0, 0))
                        crop = pad

                    out_name = f"{_basename_wo_ext(image_path)}_rand_{s}_y{y1}_x{ x1 }.png"
                    out_path = os.path.join(out_dir, out_name)
                    crop.save(out_path, format="PNG")

                    results.append({
                        "path": out_path,
                        "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                        "size": int(s), "type": "random"
                    })

                    if max_total and len(results) >= max_total:
                        return results
    except Exception as e:
        print(f"[ERR] random_crops 失败：{image_path} -> {e}")
    return results

def generate_random_windows(image_path: str, cfg: dict, out_dir: str) -> List[Dict]:
    """
    生成「全局 + 随机局部」窗口，数量控制在 ≤ max_windows_per_round
    cfg 字段（建议）：
      include_global: true
      global_downscale_max_side: 1408
      random_sizes: [512, 768]
      random_per_size: 3
      random_jitter: 0.15
      max_windows_per_round: 6
      seed: 42
    """
    _ensure_dir(out_dir)
    include_global = bool(cfg.get("include_global", True))
    g_max = cfg.get("global_downscale_max_side", None)
    sizes = list(cfg.get("random_sizes", [512, 768]))
    per_size = int(cfg.get("random_per_size", 3))
    jitter = float(cfg.get("random_jitter", 0.15))
    max_windows = int(cfg.get("max_windows_per_round", 6))
    seed = int(cfg.get("seed", 42))

    windows: List[Dict] = []

    # 1) 全局视图（可缩放）
    if include_global:
        global_dir = os.path.join(out_dir, "global")
        g_path = make_global_view(image_path, global_dir, g_max)
        windows.append({"path": g_path, "type": "global", "size": "global"})

    # 2) 随机局部
    remain = max(0, max_windows - len(windows))
    if remain > 0:
        rand_dir = os.path.join(out_dir, "rand")
        rand = random_crops(image_path, sizes, per_size, jitter, rand_dir, seed=seed, max_total=remain)
        windows.extend(rand)

    # 保障不超过上限
    if len(windows) > max_windows:
        windows = windows[:max_windows]
    return windows

# ======= 旧接口保留（网格切片）=======
def slice_image(image_path: str, slice_size: int, slice_overlap: int, out_dir: str) -> List[str]:
    """
    仍保留：规则网格切片（原有逻辑）
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

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _load_sidecar_json(json_path: str):
    try:
        import json
        if os.path.isfile(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 兼容 { "boxes": [...] } 或直接 [ ... ]
            if isinstance(data, dict) and "boxes" in data:
                return data["boxes"]
            if isinstance(data, list):
                return data
    except Exception as e:
        print(f"[WARN] 读取 JSON 失败：{json_path} -> {e}")
    return None

def _load_yolo_txt(txt_path: str, W: int, H: int, normalized: bool = True):
    """
    解析 YOLO 格式：每行 'cls cx cy w h'（通常归一化到 0~1）
    返回 list[dict(x1,y1,x2,y2,label)]
    """
    boxes = []
    if not os.path.isfile(txt_path):
        return boxes
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) < 5:
                    continue
                cls = parts[0]
                cx, cy, ww, hh = map(float, parts[1:5])
                if normalized:
                    cx *= W; cy *= H; ww *= W; hh *= H
                x1 = cx - ww / 2.0
                y1 = cy - hh / 2.0
                x2 = cx + ww / 2.0
                y2 = cy + hh / 2.0
                x1 = _clamp(int(round(x1)), 0, W - 1)
                y1 = _clamp(int(round(y1)), 0, H - 1)
                x2 = _clamp(int(round(x2)), 0, W - 1)
                y2 = _clamp(int(round(y2)), 0, H - 1)
                if x2 > x1 and y2 > y1:
                    boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": str(cls)})
    except Exception as e:
        print(f"[WARN] 读取 YOLO 标注失败：{txt_path} -> {e}")
    return boxes

def _normalize_boxes(raw_boxes, W: int, H: int, assume_normalized: Optional[bool]) -> list:
    """
    接受以下任意形态：
      - [ [x1,y1,x2,y2], ... ]
      - [ {"x1":..,"y1":..,"x2":..,"y2":..,"label"?:..}, ... ]
      - 允许 0~1 归一化；若未显式指明，则做一个“<=1 判定”的保守推断
    """
    out = []
    if not raw_boxes:
        return out
    def _is_norm_candidate(vals):
        try:
            return all(0.0 <= float(v) <= 1.0 for v in vals)
        except Exception:
            return False

    for b in raw_boxes:
        if isinstance(b, dict):
            x1, y1, x2, y2 = b.get("x1"), b.get("y1"), b.get("x2"), b.get("y2")
            label = b.get("label")
        else:
            x1, y1, x2, y2 = b[:4]
            label = None

        # 判定是否归一化
        is_norm = False
        if assume_normalized is not None:
            is_norm = bool(assume_normalized)
        else:
            try:
                is_norm = _is_norm_candidate([x1, y1, x2, y2])
            except Exception:
                is_norm = False

        try:
            x1 = float(x1); y1 = float(y1); x2 = float(x2); y2 = float(y2)
            if is_norm:
                x1 *= W; y1 *= H; x2 *= W; y2 *= H
            x1 = _clamp(int(round(x1)), 0, W - 1)
            y1 = _clamp(int(round(y1)), 0, H - 1)
            x2 = _clamp(int(round(x2)), 0, W - 1)
            y2 = _clamp(int(round(y2)), 0, H - 1)
            if x2 > x1 and y2 > y1:
                out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": label})
        except Exception:
            continue
    return out

def add_bbox_to_image(
    image_path: str,
    output_path: str,
    bbox_settings: Optional[dict] = None
) -> str:
    """
    在图上把候选框画出来，并保存到 output_path；失败时回退返回原图路径。
    bbox_settings 可用字段（都可选）：
      # 1) 直接给框（像素或 0~1 归一化）
      boxes: [ [x1,y1,x2,y2], ... ] 或 [ {x1,y1,x2,y2,label?}, ... ]
      boxes_normalized: true/false   # 若省略则自动判断是否归一化

      # 2) YOLO 旁文件
      labels_dir: "/path/to/labels"  # 查找 basename(image).txt
      yolo_normalized: true          # 默认 true

      # 3) 简单 JSON 旁文件
      annotations_dir: "/path/to/jsons"  # 查找 basename(image).bboxes.json

      # 可视化样式
      color: "red"           # 或 "#RRGGBB"
      line_width: 3
      show_label: true
      fill_alpha: 0          # 0=不填充；(0,255] 越大越不透明
      label_bg_alpha: 160    # 标签底色透明度
    """
    try:
        settings = dict(bbox_settings or {})
        with Image.open(image_path) as im:
            im = im.convert("RGBA")
            W, H = im.size

            # 1) 聚合框源
            boxes = []
            # 1.1 inline boxes
            if "boxes" in settings and settings["boxes"]:
                boxes = _normalize_boxes(settings["boxes"], W, H, settings.get("boxes_normalized"))

            # 1.2 YOLO
            if not boxes and settings.get("labels_dir"):
                stem = os.path.splitext(os.path.basename(image_path))[0]
                yolo_path = os.path.join(settings["labels_dir"], f"{stem}.txt")
                boxes = _load_yolo_txt(yolo_path, W, H, normalized=bool(settings.get("yolo_normalized", True)))

            # 1.3 简单 JSON
            if not boxes and settings.get("annotations_dir"):
                stem = os.path.splitext(os.path.basename(image_path))[0]
                json_path = os.path.join(settings["annotations_dir"], f"{stem}.bboxes.json")
                raw = _load_sidecar_json(json_path)
                if raw:
                    boxes = _normalize_boxes(raw, W, H, settings.get("boxes_normalized"))

            if not boxes:
                # 没有框就直接把原图拷贝/另存，返回原图也行
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                im.convert("RGB").save(output_path)
                return output_path

            # 2) 画框
            color = settings.get("color", "red")
            lw = int(settings.get("line_width", 3))
            show_label = bool(settings.get("show_label", True))
            fill_alpha = int(settings.get("fill_alpha", 0))
            label_bg_alpha = int(settings.get("label_bg_alpha", 160))

            overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # 字体尝试：找不到系统字体时，用默认
            try:
                font = ImageFont.truetype("arial.ttf", size=14)
            except Exception:
                font = ImageFont.load_default()

            for b in boxes:
                x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
                # 可选填充
                if fill_alpha > 0:
                    draw.rectangle([(x1, y1), (x2, y2)], fill=(255, 0, 0, _clamp(fill_alpha, 0, 255)))
                # 轮廓
                for k in range(max(1, lw)):
                    draw.rectangle([(x1 - k, y1 - k), (x2 + k, y2 + k)], outline=color)
                # 标签
                if show_label and b.get("label"):
                    label = str(b["label"])
                    tw, th = draw.textsize(label, font=font)
                    pad = 2
                    bx2 = _clamp(x1 + tw + 2 * pad, 0, W)
                    by2 = _clamp(y1 + th + 2 * pad, 0, H)
                    # 底色
                    draw.rectangle([(x1, y1), (bx2, by2)], fill=(0, 0, 0, _clamp(label_bg_alpha, 0, 255)))
                    draw.text((x1 + pad, y1 + pad), label, fill=(255, 255, 255, 255), font=font)

            out = Image.alpha_composite(im, overlay).convert("RGB")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            out.save(output_path)
            return output_path
    except Exception as e:
        print(f"[ERR] add_bbox_to_image 失败：{image_path} -> {e}")
        # 回退：用原图
        return image_path
