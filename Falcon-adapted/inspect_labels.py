import torch
from collections import defaultdict

# === Change to your actual output path ===
data_path = "/cluster/work/axelle/Falcon-results/KaggleFakeNews/slices-512-256-v1/2025-04-09_03-44-48/falconed_fakenews_data.pt"

print(f"🔍 Loading fine-grained dataset from:\n{data_path}")
data = torch.load(data_path)

fine_labels = data['fine_labels']
coarse_labels = data['coarse_labels']

groups = defaultdict(set)

for coarse, fine in zip(coarse_labels, fine_labels):
    groups[coarse.item()].add(fine.item())

print("\n🧠 Fine label distribution by coarse class:\n")
for coarse_class in sorted(groups):
    fine_set = sorted(groups[coarse_class])
    print(f"  Coarse class {coarse_class}: fine labels → {fine_set}")
