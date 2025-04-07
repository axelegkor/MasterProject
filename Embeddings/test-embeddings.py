import pandas
from sentence_transformers import SentenceTransformer

datasetPath = "../../../work/axelle/Datasets-original/KaggleFakeNews/train.csv"

# === Load CSV ===
df = pandas.read_csv(datasetPath)
sample = df.head(5).copy()

# === Combine title + text into one string ===
sample["full_text"] = (
    sample["title"].fillna("") + " " + sample["text"].fillna("")
).str.strip()

# === Load Sentence-BERT model ===
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# === Encode ===
embeddings = model.encode(sample["full_text"].tolist(), convert_to_numpy=True)

# === Print Results ===
for i, (text, emb) in enumerate(zip(sample["full_text"], embeddings)):
    print(
        f"\nExample {i+1}:\nText: {text[:100]}...\nEmbedding shape: {emb.shape}\nEmbedding (first 5 values): {emb[:5]}"
    )
