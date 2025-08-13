#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, re, json
from pathlib import Path
from collections import Counter, defaultdict

# 列分隔：优先用“首分号/末分号”切三列，避免第二列里也有分号
def split_three_fields(line: str, col_delim: str = ";"):
    s = line.strip()
    if not s:
        return None
    s_std = s.replace("；", ";")
    L = s_std.find(col_delim)
    R = s_std.rfind(col_delim)
    if L == -1 or R == -1 or R == L:
        return None
    return s_std[:L].strip(), s_std[L+1:R].strip(), s_std[R+1:].strip()

# 是否“看起来像坐标/数字串”（用来区分没有标签的情况）
_NUM_LIKE = re.compile(r"^[\d\.\s;,，；]+$")

def normalize_label(s: str):
    # 归一化：去首尾、全角->半角分隔、转小写、去空白和下划线
    t = (s or "").strip()
    t = t.replace("；",";").replace("，",",")
    t = t.lower()
    t = re.sub(r"[\s_]+", "", t)
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abnormal_dir", required=True, help="异常图目录（每图旁边同名 .txt）")
    ap.add_argument("--column_delim", default=";", help="三列之间的分隔符，默认 ;")
    ap.add_argument("--out", default="./data/prepared/label_stats.json", help="统计结果 JSON 输出路径")
    args = ap.parse_args()

    abdir = Path(args.abnormal_dir)
    txts = sorted(abdir.rglob("*.txt"))

    raw_counter = Counter()
    norm_groups = defaultdict(Counter)
    per_file = {}

    for txtp in txts:
        per_file[str(txtp)] = Counter()
        with open(txtp, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                tri = split_three_fields(line, col_delim=args.column_delim)
                if tri:
                    _, _, lab = tri
                else:
                    # 兜底：尝试把最后一个分号后的内容当作标签；若仍然像数字串，则视为无标签
                    s = line.replace("；",";")
                    parts = s.rsplit(";", 1)
                    lab = parts[1].strip() if len(parts)==2 and not _NUM_LIKE.match(parts[1]) else ""
                if not lab:
                    lab = "(空/缺失)"
                raw_counter[lab] += 1
                per_file[str(txtp)][lab] += 1
                norm = normalize_label(lab)
                norm_groups[norm][lab] += 1

    # 选择每个归一化组的“代表标签”（出现频次最高者）
    canonical = {}
    for norm, cnts in norm_groups.items():
        lab, _ = cnts.most_common(1)[0]
        canonical[norm] = lab

    # 生成 label_map 建议：把同组的变体都映射到代表标签
    label_map_suggest = {}
    for norm, cnts in norm_groups.items():
        canon = canonical[norm]
        for variant in cnts:
            if variant != canon:
                label_map_suggest[variant] = canon

    out = {
        "total_txt_files": len(txts),
        "raw_counts": raw_counter.most_common(),
        "normalized_groups": {k: dict(v) for k,v in norm_groups.items()},
        "canonical_by_norm": canonical,
        "label_map_suggest": label_map_suggest,
        "per_file_samples": {k: dict(v) for k, v in list(per_file.items())[:5]}  # 只放前5个文件做参考
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as g:
        json.dump(out, g, ensure_ascii=False, indent=2)

    # 终端友好打印
    print("\n=== 标签频次（原样） ===")
    for lab, n in raw_counter.most_common():
        print(f"{lab}: {n}")

    print("\n=== data.yaml 的 label_map 建议（可直接粘贴） ===")
    if label_map_suggest:
        print("label_map:")
        for k,v in label_map_suggest.items():
            print(f"  {k}: {v}")
    else:
        print("# 你的标签已经很统一，可不必配置 label_map")

    print(f"\n已写出统计文件：{args.out}")

if __name__ == "__main__":
    main()
