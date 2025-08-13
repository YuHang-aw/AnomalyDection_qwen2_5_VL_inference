# scripts/repair_sft_jsonl.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json, argparse
from pathlib import Path

def _wrap_part(x):
    # 把字符串或旧格式统一包成 [{type,text,image}]
    if isinstance(x, list):
        # 旧样本里 user 已经是 list，但元素可能缺键；补齐键
        out = []
        for p in x:
            if isinstance(p, dict) and "type" in p:
                out.append({"type": p["type"],
                            "text": p.get("text", None),
                            "image": p.get("image", None)})
            else:
                out.append({"type":"text","text": str(p), "image": None})
        return out
    else:
        return [{"type":"text","text": str(x), "image": None}]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    src = Path(args.in_jsonl); dst = Path(args.out_jsonl)
    dst.parent.mkdir(parents=True, exist_ok=True)

    n, fixed = 0, 0
    with open(src, "r", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            n += 1
            rec = json.loads(line)
            msgs = rec.get("messages", [])
            new_msgs = []
            for m in msgs:
                role = m.get("role")
                content = m.get("content")
                wrapped = _wrap_part(content)
                if wrapped != content:
                    fixed += 1
                new_msgs.append({"role": role, "content": wrapped})
            rec["messages"] = new_msgs
            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Processed {n} lines, fixed {fixed} messages. → {dst}")

if __name__ == "__main__":
    main()
