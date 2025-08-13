#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json, argparse
from pathlib import Path

def to_part(obj):
    # 把任意对象转成 {type,text,image}
    if isinstance(obj, dict):
        t = obj.get("type")
        if t == "image" and isinstance(obj.get("image"), str):
            return {"type":"image","text": None, "image": obj["image"]}
        # 其他情况一律当文本
        txt = obj.get("text", None)
        if txt is None and "image" in obj and isinstance(obj["image"], str):
            # 有些旧结构用 {"image": "/path"} 表示图片
            return {"type":"image","text": None, "image": obj["image"]}
        return {"type":"text","text": "" if txt is None else str(txt), "image": None}
    elif isinstance(obj, str):
        return {"type":"text","text": obj, "image": None}
    elif obj is None:
        return {"type":"text","text": "", "image": None}
    else:
        return {"type":"text","text": str(obj), "image": None}

def norm_content(content):
    # 目标：list of parts
    if isinstance(content, list):
        return [to_part(p) for p in content]
    else:
        return [to_part(content)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    src, dst = Path(args.in_jsonl), Path(args.out_jsonl)
    dst.parent.mkdir(parents=True, exist_ok=True)

    n, fixed = 0, 0
    with open(src, "r", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            n += 1
            rec = json.loads(line)
            msgs = rec.get("messages", [])
            new_msgs = []
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                nc = norm_content(content)
                if nc != content:
                    fixed += 1
                new_msgs.append({"role": role, "content": nc})
            rec["messages"] = new_msgs
            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"normalized {fixed} messages in {n} records → {dst}")

if __name__ == "__main__":
    main()
