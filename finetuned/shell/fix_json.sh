python scripts/repair_sft_jsonl.py --in_jsonl ./data/prepared/sft_train.jsonl --out_jsonl ./data/prepared/sft_train_fixed.jsonl
python scripts/repair_sft_jsonl.py --in_jsonl ./data/prepared/sft_val.jsonl   --out_jsonl ./data/prepared/sft_val_fixed.jsonl

# 修改 configs/train_phase1.yaml 指向 *_fixed.jsonl 后训练
python scripts/train.py --train_config configs/train_phase1.yaml
