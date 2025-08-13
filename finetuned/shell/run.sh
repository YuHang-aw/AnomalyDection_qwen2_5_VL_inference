export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
# 可选：指定缓存根目录（如果不想写入默认家目录）
export HF_HOME=/mnt/.hf

# 0) 激活虚拟环境（已离线装好依赖）
source .venv/bin/activate

# 1) 切分 & 生成候选
python scripts/prepare_regions.py --config configs/data.yaml

# 2) 构建多模态SFT样本
python scripts/build_sft_jsonl.py --regions_jsonl ./data/prepared/regions_train.jsonl --out_jsonl ./data/prepared/sft_train.jsonl
python scripts/build_sft_jsonl.py --regions_jsonl ./data/prepared/regions_val.jsonl   --out_jsonl ./data/prepared/sft_val.jsonl
python scripts/build_sft_jsonl.py --regions_jsonl ./data/prepared/regions_test.jsonl  --out_jsonl ./data/prepared/sft_test.jsonl  # 留作盲测

# 3) 训练（Phase-1：冻结视觉塔 + LoRA/QLoRA）
python scripts/train.py --train_config configs/train_phase1.yaml

# (中断续训)
python scripts/train.py --train_config configs/train_phase1.yaml --resume
