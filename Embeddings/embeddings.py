import pandas
import torch
from transformers import AutoTokenizer
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

datasetPath = "../../../work/axelle/Datasets-original/KaggleFakeNews/train.csv"
outputPath = (
    "../../../work/axelle/Datasets-embedded/KaggleFakeNews/slices-512-256-v1.pt"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# Function that returns the embedding of a full article, regardless of length
# Strides smaller than window_size ensures overlapping chunks
def embed(text, window_size=512, stride=256):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunk_embeddings = []
    weights = []

    # To avoid the first part of the text being represented only once, while the other parts are represented multiple times
    # Works best if stride is half of window_size
    if (
        len(tokens) > stride
    ):  # Not strictly necessary, but avoids unnecessary computation
        first_tokens = tokens[0:stride]
        first_text = tokenizer.decode(first_tokens, skip_special_tokens=True)
        first_embedding = model.encode(first_text)
        chunk_embeddings.append(first_embedding)
        weights.append(len(first_tokens))  # Weights.append(stride)

    for start in range(0, len(tokens), stride):
        end = start + window_size
        chunk_tokens = tokens[start:end]

        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunk_embedding = model.encode(chunk_text)

        chunk_embeddings.append(chunk_embedding)
        weights.append(len(chunk_tokens))

    # To avoid the last part of the text being represented only once, while the other parts are represented multiple times
    # Works best if stride is half of window_size
    remainder_start = len(tokens) - (len(tokens) % stride)
    if (
        remainder_start < len(tokens) and len(tokens) > stride
    ):  # Not strictly necessary, but avoids unnecessary computation
        last_tokens = tokens[
            remainder_start:
        ]  # The tokens that have only been represented once
        last_text = tokenizer.decode(last_tokens, skip_special_tokens=True)
        last_embedding = model.encode(last_text)
        chunk_embeddings.append(last_embedding)
        weights.append(len(last_tokens))  # Weights.append(stride)

    if not chunk_embeddings:
        return None  # np.zeros(model.get_sentence_embedding_dimension())

    chunk_embeddings = np.array(chunk_embeddings)
    weights = np.array(weights, dtype=np.float32)

    # Ensure a smaller chunk (the last one) is given an appropriate weight
    weighted_embedding = np.average(chunk_embeddings, axis=0, weights=weights)
    return weighted_embedding


df = pandas.read_csv(datasetPath)
df["full_text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()

ids = []
labels = []
embeddings = []

# Loop through each article and compute its embedding
for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding articles"):
    article_id = row["id"]
    label = row["label"]
    full_text = row["full_text"]

    # Compute embedding for the article
    embedding = embed(full_text)

    # Store only if embedding was successfully created
    if embedding is not None:
        ids.append(article_id)
        embeddings.append(embedding)
        labels.append(label)

output = {
    "ids": ids,
    "embeddings": torch.tensor(np.array(embeddings), dtype=torch.float32),
    "labels": torch.tensor(labels),
}

torch.save(output, outputPath)
