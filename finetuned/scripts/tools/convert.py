# convert_to_sharegpt.py
import json, sys, pathlib
def pick_text(block):
    if isinstance(block, list):
        # 取第一个 text
        for it in block:
            if it.get("type")=="text":
                return it.get("text","")
    return ""
def first_image(block):
    if isinstance(block, list):
        for it in block:
            if it.get("type")=="image" and it.get("image"):
                return it["image"]
    return None

def convert_line(line):
    it = json.loads(line)
    sys_msgs = [m for m in it["messages"] if m["role"]=="system"]
    usr_msgs = [m for m in it["messages"] if m["role"]=="user"]
    asst_msgs= [m for m in it["messages"] if m["role"]=="assistant"]
    system = pick_text(sys_msgs[0]["content"]) if sys_msgs else None
    user_t = pick_text(usr_msgs[0]["content"]) if usr_msgs else ""
    img_p  = first_image(usr_msgs[0]["content"]) if usr_msgs else None
    asst_t = pick_text(asst_msgs[0]["content"]) if asst_msgs else ""

    if not img_p:
        return None  # 没图就跳过

    sample = {
      "conversations":[
        {"from":"system","value": system or ""},
        {"from":"human","value": f"{user_t}\n<image>"},
        {"from":"gpt","value": asst_t}
      ],
      "images":[str(img_p)]
    }
    return sample

def main(src, dst):
    with open(src, "r", encoding="utf-8") as fin, open(dst,"w",encoding="utf-8") as fout:
        for ln in fin:
            ex = convert_line(ln)
            if ex: fout.write(json.dumps(ex, ensure_ascii=False)+"\n")

if __name__=="__main__":
    # 用法：python convert_to_sharegpt.py data/prepared/sft_train.jsonl data/lf/roi_train.jsonl
    main(sys.argv[1], sys.argv[2])
