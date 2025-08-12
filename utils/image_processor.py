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
