#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    args = ap.parse_args()
    bad = 0
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"[JSON ERROR] line {i}: {e}")
                bad += 1; continue
            msgs = rec.get("messages")
            if not isinstance(msgs, list):
                print(f"[SCHEMA] line {i}: messages is {type(msgs).__name__}, expect list"); bad += 1; continue
            for j, m in enumerate(msgs):
                c = m.get("content")
                if not isinstance(c, list):
                    print(f"[SCHEMA] line {i} msg#{j}: content is {type(c).__name__}, expect list")
                    bad += 1
    if bad == 0:
        print("OK: all messages[].content are lists.")
    else:
        print(f"Found {bad} problems.")
if __name__ == "__main__":
    main()
