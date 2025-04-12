#!/bin/bash

if [ $# -lt 2 ]; then
  echo "❌ Usage: $0 <dataset> <input_file>(without .pt)"
  exit 1
fi

INFILE="/cluster/work/axelle/Datasets-embedded/$1/$2.pt"
OUTDIR="/cluster/work/axelle/Falcon-results/$1/$2/$(date +%F_%H-%M-%S)"

NEIGHBOR_FILE="$OUTDIR/FakeNews_neighbors.pth"
CHECKPOINT="$OUTDIR/run1/models/model_10.pth"
FINAL_DATASET="$OUTDIR/falconed_fakenews_data.pt"
FINAL_CSV="$OUTDIR/falconed_fakenews_labels.csv"

mkdir -p "$OUTDIR"

# === Step 2: Find neighbors ===
if [ ! -f "$NEIGHBOR_FILE" ]; then
  echo "🔍 Finding neighbors..."
  python find_neighbors.py \
    --cfg_file configs/fake_news_bert/coarse2fine/base.yaml \
    --override_cfg DATASET.NAME FakeNews DATASET.DATAROOT "$INFILE" OUTPUT_DIR "$OUTDIR" \
    --output_dir "$OUTDIR" \
    --use_faiss \
    --use_raw_input
else
  echo "✅ Neighbors already exist."
fi

# === Step 3: Train model with FALCON ===
python main.py \
  --cfg_file configs/fake_news_bert/coarse2fine/base.yaml \
  --override_cfg DATASET.NAME FakeNews DATASET.DATAROOT "$INFILE" OUTPUT_DIR "$OUTDIR/run1" NEIGHBORS "$NEIGHBOR_FILE" \
  --port 8080


# === Step 4: Infer fine-grained labels and export final dataset ===
python infer_fine_labels.py \
  --cfg_file configs/fake_news_bert/coarse2fine/base.yaml \
  --override_cfg DATASET.NAME FakeNews DATASET.DATAROOT "$INFILE" MODEL.NUM_CLASSES 7 \
  --model_path "$CHECKPOINT" \
  --output_pt "$FINAL_DATASET" \
  --output_csv "$FINAL_CSV"
  

