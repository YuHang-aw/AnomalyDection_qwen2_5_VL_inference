#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, argparse
from pathlib import Path

SYSTEM_PROMPT = (
    "你是堤坝衬砌板缺陷判读专家。规则：\n"
    "- 只依据图像纹理/结构和候选区域进行判断；不得依据‘是否有提示/是否有框/颜色’下结论。\n"
    "- 回答中不得出现‘红框/颜色/矩形/框住’等描述；如需引用，请使用‘区域 #id’或坐标。\n"
    "- 输出JSON，键包含：per_region(数组)、image_level(正常/异常)。每个per_region项含 id, decision(正常/异常), type(可选), confidence(0~1), rationale(证据)。\n"
)

def regions_to_text(regions, include_label=False):
    lines = ["<regions>"]
    for r in regions:
        lab = (f" label=\"{r.get('label','')}\"" if include_label and r.get('label') else "")
        lines.append(f"<box id={r['id']} x1={r['x1']} y1={r['y1']} x2={r['x2']} y2={r['y2']}{lab}/>")
    lines.append("</regions>")
    return "\n".join(lines)

def _part_text(s: str):
    return {"type":"text","text": s, "image": None}

def _part_image(path: str):
    # 这里只存路径，真正的 PIL 加载在 collator 里做
    return {"type":"image","text": None, "image": path}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--include_label", action="store_true", help="是否在<regions>里带label（一般关闭以免泄漏）")
    args = ap.parse_args()

    src = Path(args.regions_jsonl)
    dst = Path(args.out_jsonl); dst.parent.mkdir(parents=True, exist_ok=True)

    with open(src, "r", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            it = json.loads(line)
            regions = it["regions"]
            pos_ids = set(it.get("positive_region_ids", []))

            reg_text = regions_to_text(regions, include_label=args.include_label)
            user_content = [
                _part_text("请在候选区域内判断是否存在异常，并返回结构化JSON。"),
                _part_text(reg_text),
                _part_image(it["image"])
            ]

            per_region = []
            for r in regions:
                rid = r["id"]
                is_pos = (rid in pos_ids)
                item = {
                    "id": rid,
                    "decision": "异常" if is_pos else "正常",
                    "confidence": 0.85 if is_pos else 0.9,
                    "rationale": "依据纹理形态/边缘/连续性等特征做出判断"
                }
                if is_pos and r.get("label"):
                    item["type"] = r["label"]
                per_region.append(item)
            assistant_json = {"per_region": per_region, "image_level": it.get("image_level","正常")}
            assistant_text = json.dumps(assistant_json, ensure_ascii=False)

            rec = {
                "messages": [
                    {"role":"system",   "content":[_part_text(SYSTEM_PROMPT)]},
                    {"role":"user",     "content": user_content},
                    {"role":"assistant","content":[_part_text(assistant_text)]}
                ]
            }
            g.write(json.dumps(rec, ensure_ascii=False)+"\n")
    print(f"Wrote → {dst}")

if __name__ == "__main__":
    main()
