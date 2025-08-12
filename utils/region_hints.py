# utils/region_hints.py
# -*- coding: utf-8 -*-
import os, json, math
from typing import List, Dict, Optional
from PIL import Image

def _clamp(v, lo, hi): return max(lo, min(hi, v))

def _bbox_from_points(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]

def _round(v, nd): 
    return float(f"{v:.{nd}f}") if nd is not None else float(v)

def _load_labelme(json_path: str) -> Optional[dict]:
    if not os.path.isfile(json_path): 
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] 读取 LabelMe 失败：{json_path} -> {e}")
        return None

def _shape_to_bbox(shape: dict) -> Dict:
    st = (shape.get("shape_type") or "").lower()
    pts = shape.get("points") or []
    if not pts: return {}
    if st == "rectangle" and len(pts) >= 2:
        (x1,y1),(x2,y2) = pts[0], pts[1]
        return {"x1":min(x1,x2), "y1":min(y1,y2), "x2":max(x1,x2), "y2":max(y1,y2)}
    if st == "circle" and len(pts) >= 2:
        (cx,cy),(px,py) = pts[0], pts[1]
        r = math.hypot(px-cx, py-cy)
        return {"x1":cx-r, "y1":cy-r, "x2":cx+r, "y2":cy+r}
    # polygon / polyline / line / linestrip / point 等：统一取外接框
    x1,y1,x2,y2 = _bbox_from_points(pts)
    # point 给一个小半径框（10px）
    if st == "point":
        r = 10
        return {"x1":x1-r, "y1":y1-r, "x2":x1+r, "y2":y1+r}
    return {"x1":x1, "y1":y1, "x2":x2, "y2":y2}

def _norm_xyxy(b, W, H):
    x1 = _clamp(b["x1"], 0, W-1); y1 = _clamp(b["y1"], 0, H-1)
    x2 = _clamp(b["x2"], 0, W-1); y2 = _clamp(b["y2"], 0, H-1)
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return {"x1":x1, "y1":y1, "x2":x2, "y2":y2}

def _format_regions_text(regions: List[Dict], W:int, H:int, cfg:dict) -> str:
    """
    生成给模型看的统一标注文本：
    <regions>
    <box id=1 x1=.. y1=.. x2=.. y2=.. label="crack"/>
    ...
    </regions>
    """
    if not regions: return ""
    fmt = (cfg.get("coordinate_format") or "xyxy").lower()  # xyxy | xywh_norm
    nd = cfg.get("round")  # 小数位（仅 norm 生效）
    include_label = bool(cfg.get("include_label", True))
    pre  = cfg.get("prefix", "<regions>")
    post = cfg.get("suffix", "</regions>")
    lines = [pre]
    for i, r in enumerate(regions, 1):
        if fmt == "xywh_norm":
            x = (r["x1"] + r["x2"]) / 2.0 / W
            y = (r["y1"] + r["y2"]) / 2.0 / H
            w = (r["x2"] - r["x1"]) / W
            h = (r["y2"] - r["y1"]) / H
            x,y,w,h = [_round(v, nd or 4) for v in (x,y,w,h)]
            lab = f' label="{r.get("label","")}"' if include_label and r.get("label") else ""
            lines.append(f'<box id={i} fmt="xywh_norm" x={x} y={y} w={w} h={h}{lab}/>')
        else:
            lab = f' label="{r.get("label","")}"' if include_label and r.get("label") else ""
            x1,y1,x2,y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
            lines.append(f"<box id={i} x1={x1} y1={y1} x2={x2} y2={y2}{lab}/>")
    lines.append(post)
    return "\n".join(lines)

def build_region_hints_text(image_path: str, cfg: dict) -> str:
    """
    从 LabelMe 旁文件生成 regions 文本。
    cfg:
      enabled: true
      json_dir: "./labelme_json"
      max_regions: 5
      include_label: true
      coordinate_format: "xyxy"  # 或 "xywh_norm"
      round: 4                   # 仅 xywh_norm 时生效
      prefix: "<regions>"
      suffix: "</regions>"
      allow_nohit: true          # 找不到就返回空字符串
    """
    if not cfg or not cfg.get("enabled", False):
        return ""
    jd = cfg.get("json_dir")
    if not jd:
        return ""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(jd, f"{stem}.json")
    meta = _load_labelme(json_path)
    if not meta:
        return "" if cfg.get("allow_nohit", True) else f"<regions />"

    W = int(meta.get("imageWidth") or 0)
    H = int(meta.get("imageHeight") or 0)
    # 若 LabelMe 不含尺寸，尝试读图
    if not W or not H:
        try:
            from PIL import Image
            with Image.open(image_path) as im:
                W, H = im.size
        except Exception:
            pass
    shapes = meta.get("shapes") or []
    regions = []
    for s in shapes:
        b = _shape_to_bbox(s)
        if not b: continue
        # 限制在图内
        b = _norm_xyxy(b, W, H)
        lab = s.get("label")
        b["label"] = lab
        regions.append(b)

    # 限制数量
    k = int(cfg.get("max_regions", 5))
    if k > 0 and len(regions) > k:
        regions = regions[:k]

    return _format_regions_text(regions, W, H, cfg)
