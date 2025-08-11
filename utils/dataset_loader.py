# -*- coding: utf-8 -*-
import os
from typing import List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _is_img(name: str) -> bool:
    return os.path.splitext(name.lower())[1] in IMAGE_EXTS

def _scan_dir(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    files = [os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path)) if _is_img(f)]
    return files

def load_dataset(data_dir: str, limit_per_class: int | None = None) -> List[Tuple[str, str]]:
    """
    返回 [(image_path, label)] 列表。
    - 若 data_dir 下存在 'n' 与 'p' 子目录，则分别作为负/正样本。
    - 否则将 data_dir 视作平铺目录，label 记为 'unknown'。
    - limit_per_class: 每类（n/p）最多取多少张；<=0 或 None 表示不限制。
    """
    if not data_dir or not os.path.isdir(data_dir):
        print(f"[WARN] 数据目录不存在：{data_dir}")
        return []

    samples: List[Tuple[str, str]] = []
    has_np = False
    for lab in ("n", "p"):
        sub = os.path.join(data_dir, lab)
        if os.path.isdir(sub):
            has_np = True
            imgs = _scan_dir(sub)
            if limit_per_class and limit_per_class > 0:
                imgs = imgs[:limit_per_class]
            samples.extend((fp, lab) for fp in imgs)

    if has_np:
        return samples

    # 平铺目录（无 n/p）
    imgs = _scan_dir(data_dir)
    if limit_per_class and limit_per_class > 0:
        imgs = imgs[:limit_per_class]
    return [(fp, "unknown") for fp in imgs]
