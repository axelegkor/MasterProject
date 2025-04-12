import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

from datasets import FakeNewsBertData
from models import create_backbone_factory, construct_classifier
from config import load_config, override_config

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--output_pt", type=str, required=True)
parser.add_argument("--output_csv", type=str, required=False)
parser.add_argument("--override_cfg", type=str, nargs="+", required=False)
args = parser.parse_args()

# === Load config and model ===
cfg = load_config(args.cfg_file)
cfg = override_config(cfg, args.override_cfg) if args.override_cfg else cfg

model_base, embed_dim = create_backbone_factory(cfg)(cfg.MODEL.PRETRAINED)
model = construct_classifier(
    cfg.MODEL.HEAD_TYPE, model_base, embed_dim, cfg.MODEL.NUM_CLASSES
)

checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint["model"])
model.eval().cuda()

# === Load dataset ===
data = FakeNewsBertData(cfg.DATASET.DATAROOT)
loader = DataLoader(data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False)

all_preds = []
all_ids = []
all_labels = []
all_embeddings = []

# === Inference ===
with torch.no_grad():
    for batch in tqdm(loader, desc="Running inference for fine labels"):
        x = batch["inputs"].cuda()
        logits = model(x)
        preds = logits.argmax(dim=1).cpu()

        all_preds.extend(preds.tolist())
        all_ids.extend(batch["index"].tolist())
        all_labels.extend(batch["coarse_label"].tolist())
        all_embeddings.extend(x.cpu().tolist())

# === 📊 Fine label distribution ===
print("📊 Fine label distribution:")
print(pd.Series(all_preds).value_counts().sort_index())

# === 🧠 Fine label grouping by coarse label ===
fine_tensor = torch.tensor(all_preds)
coarse_tensor = torch.tensor(all_labels)

print("\n🧠 Fine label distribution by coarse class:")
for c in [0, 1]:
    selected = fine_tensor[coarse_tensor == c]
    unique_fine = sorted(set(selected.tolist()))
    print(f"  Coarse class {c}: fine labels → {unique_fine}")

# === Save as .pt ===
torch.save(
    {
        "ids": [data.ids[i] for i in all_ids],
        "embeddings": torch.tensor(all_embeddings),
        "coarse_labels": torch.tensor(all_labels),
        "fine_labels": fine_tensor,
    },
    args.output_pt,
)

print(f"✅ Saved enriched dataset with fine labels to {args.output_pt}")

# === Optional CSV output ===
if args.output_csv:
    df = pd.DataFrame(
        {
            "id": [data.ids[i] for i in all_ids],
            "binaryLabel": all_labels,
            "fineLabel": all_preds,
        }
    )
    df.to_csv(args.output_csv, index=False)
    print(f"📝 Saved preview CSV to {args.output_csv}")
