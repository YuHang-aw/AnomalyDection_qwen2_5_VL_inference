#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, random, argparse, re
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple
import yaml
random.seed(42)

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---- 工具：从一行里切出“三列”：<filename>;<coords_field>;<label> ----
def _split_three_fields(line: str, col_delim: str = ";"):
    """用首个与最后一个分隔符定位三列，防止第二列里也有分号"""
    s = line.strip()
    if not s: return None
    # 统一全角分号到半角，便于定位列分隔
    s_std = s.replace("；", ";")
    L = s_std.find(col_delim)
    R = s_std.rfind(col_delim)
    if L == -1 or R == -1 or R == L:
        # 兜底：没有三列结构，返回 None 让上层用其他规则（比如每行就一个框）
        return None
    c1 = s_std[:L].strip()
    c2 = s_std[L+1:R].strip()
    c3 = s_std[R+1:].strip()
    return c1, c2, c3

# ---- 工具：把 coords_field 解析成若干 xyxy 框 ----
_NUM_SPLIT = re.compile(r"[；;,，\s]+")

def _parse_multi_boxes(coords_field: str, coords_are_xyxy: bool = True) -> List[List[int]]:
    """
    coords_field 可能长这样：
    "1190；206；1712；377 1813；555；1866；636"
    我们把所有分隔符统一，然后每 4 个数字一组 → 框。
    若将来是 cx,cy,w,h，可在这里做转换。
    """
    s = coords_field.strip()
    if not s:
        return []
    # 全角分号等都视为分隔
    nums = [t for t in _NUM_SPLIT.split(s) if t]
    vals = []
    for t in nums:
        try:
            vals.append(float(t))
        except:
            # 丢掉非数字 token
            pass
    boxes = []
    for i in range(0, len(vals), 4):
        if i+3 >= len(vals): break
        x1,y1,x2,y2 = vals[i], vals[i+1], vals[i+2], vals[i+3]
        if not coords_are_xyxy:
            # 若是 cx,cy,w,h → 转 xyxy
            cx, cy, w, h = x1, y1, x2, y2
            x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
        x1,y1,x2,y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        boxes.append([x1,y1,x2,y2])
    return boxes

def _random_decoys(W: int, H: int, k: int, min_size: int = 24):
    out = []
    for _ in range(k):
        w = random.randint(min_size, max(min(W//4, W-1), min_size))
        h = random.randint(min_size, max(min(H//4, H-1), min_size))
        x1 = random.randint(0, max(W - w - 1, 0))
        y1 = random.randint(0, max(H - h - 1, 0))
        x2 = x1 + w
        y2 = y1 + h
        out.append([x1,y1,x2,y2])
    return out

def _clip_box(x1,y1,x2,y2, W,H):
    x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
    y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
    if x2<=x1 or y2<=y1: return None
    return [x1,y1,x2,y2]

def _gather_images(root: Path):
    imgs = []
    for p in root.rglob("*.*"):
        if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]:
            imgs.append(p)
    return imgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/data.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    ab_dir = Path(cfg["abnormal_dir"]).expanduser()
    nm_dir = Path(cfg["normal_dir"]).expanduser()
    out_dir = Path(cfg["out_dir"]).expanduser(); out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed",42)); random.seed(seed)
    min_box = int(cfg.get("min_box_size",24))
    max_boxes = int(cfg.get("max_boxes_per_image",8))
    neg_per_pos = int(cfg.get("abnormal_neg_per_pos",1))
    normal_k = int(cfg.get("normal_decoys_per_image",6))

    # 解析相关
    col_delim = cfg.get("column_delim",";")
    coords_are_xyxy = bool(cfg.get("coords_are_xyxy", True))
    label_map = cfg.get("label_map", {}) or {}

    # 收集
    items = []

    # --- 异常图：从同名txt读取 ---
    ab_imgs = _gather_images(ab_dir)
    for imgp in ab_imgs:
        txtp = imgp.with_suffix(".txt")
        if not txtp.exists():
            continue
        try:
            W,H = Image.open(imgp).size
        except Exception:
            continue

        pos = []  # [(xyxy, label)]
        with open(txtp, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line: 
                    continue
                # 优先尝试“三列”模式：<filename>;<coords_field>;<label>
                tri = _split_three_fields(line, col_delim=col_delim)
                if tri:
                    _fname, coord_field, lab = tri
                    boxes = _parse_multi_boxes(coord_field, coords_are_xyxy=coords_are_xyxy)
                    if lab in label_map: lab = label_map[lab]
                    for b in boxes:
                        x1,y1,x2,y2 = b
                        if (x2-x1)<min_box or (y2-y1)<min_box: 
                            continue
                        cb = _clip_box(x1,y1,x2,y2,W,H)
                        if cb: pos.append((cb, lab))
                else:
                    # 兜底：这一行就只是一组坐标（或多组），可选末尾有 ;label
                    # 尝试末尾 ;label
                    s = line.replace("；",";")
                    if ";" in s:
                        # 可能 coords...;label 或恰好就是坐标里全是分号（无label）
                        # 先尝试从最后一个 ; 提取 label（如果label看起来像词）
                        parts = s.rsplit(";", 1)
                        coord_field = parts[0]
                        lab = parts[1].strip() if len(parts)==2 and not re.match(r"^\s*[\d\.;,，\s]+\s*$", parts[1]) else ""
                    else:
                        coord_field = line
                        lab = ""
                    if lab in label_map: lab = label_map[lab]
                    boxes = _parse_multi_boxes(coord_field, coords_are_xyxy=coords_are_xyxy)
                    for b in boxes:
                        x1,y1,x2,y2 = b
                        if (x2-x1)<min_box or (y2-y1)<min_box: 
                            continue
                        cb = _clip_box(x1,y1,x2,y2,W,H)
                        if cb: pos.append((cb, lab if lab else None))

        if not pos:
            continue

        # 负ROI：按 1:1 配
        neg_need = min(max_boxes - len(pos), len(pos)*neg_per_pos)
        negs = _random_decoys(W,H,max(0,neg_need),min_box)

        regions = []
        rid = 1
        pos_ids = []
        for (x1,y1,x2,y2), lab in pos:
            regions.append({"id":rid,"x1":x1,"y1":y1,"x2":x2,"y2":y2,"label":lab})
            pos_ids.append(rid); rid += 1
        for (x1,y1,x2,y2) in negs:
            regions.append({"id":rid,"x1":x1,"y1":y1,"x2":x2,"y2":y2,"label":None})
            rid += 1

        regions = regions[:max_boxes]
        items.append({
            "image": str(imgp),
            "W": W, "H": H,
            "regions": regions,
            "positive_region_ids": pos_ids,
            "image_level": "异常"
        })

    # --- 正常图：只生成干扰ROI ---
    nm_imgs = _gather_images(nm_dir)
    for imgp in nm_imgs:
        try:
            W,H = Image.open(imgp).size
        except Exception:
            continue
        k = min(normal_k, max_boxes)
        decoys = _random_decoys(W,H,k,min_box)
        regions = []
        for i,(x1,y1,x2,y2) in enumerate(decoys, start=1):
            regions.append({"id":i,"x1":x1,"y1":y1,"x2":x2,"y2":y2,"label":None})
        items.append({
            "image": str(imgp),
            "W": W, "H": H,
            "regions": regions,
            "positive_region_ids": [],
            "image_level": "正常"
        })

    # --- 划分：train / val / test ---
    # 1) 先处理 holdout：把指定文件名强制进 test
    holdout = set(cfg.get("holdout_list", []) or [])
    test_items, remain = [], []
    for it in items:
        name = os.path.basename(it["image"])
        (test_items if name in holdout else remain).append(it)

    # 2) 剩余集合再按 parent 分组或直接随机
    def parent_key(p): return str(Path(p).parent)
    if bool(cfg.get("split_by_parent", True)):
        buckets = {}
        for it in remain:
            k = parent_key(it["image"])
            buckets.setdefault(k, []).append(it)
        groups = list(buckets.values())
        random.shuffle(groups)
        flat = []
        for g in groups: flat.append(g)
        # 逐组填充
        remain_flat = [it for g in flat for it in g]
    else:
        remain_flat = remain[:]
        random.shuffle(remain_flat)

    n = len(remain_flat)
    tr_target = int(n * cfg.get("train_ratio", 0.8))
    va_target = int(n * cfg.get("val_ratio",   0.1))
    # test 剩下的
    train_items = remain_flat[:tr_target]
    val_items   = remain_flat[tr_target:tr_target+va_target]
    test_items += remain_flat[tr_target+va_target:]

    # --- 写出 ---
    out_all = Path(cfg["out_dir"]) / "regions_all.jsonl"
    out_tr  = Path(cfg["out_dir"]) / "regions_train.jsonl"
    out_va  = Path(cfg["out_dir"]) / "regions_val.jsonl"
    out_te  = Path(cfg["out_dir"]) / "regions_test.jsonl"
    for path, data in [(out_all, items),(out_tr, train_items),(out_va, val_items),(out_te, test_items)]:
        with open(path, "w", encoding="utf-8") as f:
            for it in data:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Train: {len(train_items)}, Val: {len(val_items)}, Test: {len(test_items)}")
    print(f"→ {out_tr}\n→ {out_va}\n→ {out_te}")

if __name__ == "__main__":
    main()
