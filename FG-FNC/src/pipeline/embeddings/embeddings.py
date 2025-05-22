import pandas
import torch
from transformers import AutoTokenizer
from typing import Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import config

DATASET = config.DATASET
FILENAME = config.FILENAME
WHITENING = config.WHITENING
DIMENSIONALITY_REDUCTION = config.DIMENSIONALITY_REDUCTION

datasetPath = (
    "/cluster/work/axelle/Datasets-preprocessed/" + DATASET + "/" + FILENAME + ".csv"
)

nw = ""
if not WHITENING:
    nw = "_nw"
    
outputPath = (
    "/cluster/work/axelle/Datasets-embedded/" + DATASET + "/" + FILENAME + "_" + str(DIMENSIONALITY_REDUCTION) + nw + ".pt"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# Function that returns the embedding of a full article, regardless of length
# Strides smaller than window_size ensures overlapping chunks
def embed(text, window_size=512, stride=256) -> np.ndarray:
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]

    if len(tokens) < window_size:
        # If the text is smaller than the window size, return the embedding of the whole text
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        embedding = model.encode(text)
        return embedding

    else:
        # To avoid the first part of the text being represented only once, while the other parts are represented multiple times
        # Works best if stride is half of window_size
        chunk_embeddings = []
        weights = []
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
        if remainder_start < len(
            tokens
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

def compute_whitening(embeddings: np.ndarray, k: int = None, eps: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(embeddings, axis=0)
    cov = np.cov(embeddings - mu, rowvar=False, bias=True)
    U, S, _ = np.linalg.svd(cov)
    if k is not None:
        U = U[:, :k]
        S = S[:k]
    w= np.dot(U, np.diag(1.0 / np.sqrt(S + eps)))
    whitened = np.dot(embeddings - mu, w)
    return whitened, mu, w

df = pandas.read_csv(datasetPath)

ids = []
embeddings = []
labels = []


# Loop through each article and compute its embedding
for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding articles"):
    article_id = row["id"]
    text = row["text"]
    label = row["label"]

    # Compute embedding for the article
    embedding = embed(text)

    # Store only if embedding was successfully created
    if embedding is not None:
        ids.append(article_id)
        embeddings.append(embedding)
        labels.append(label)

if WHITENING:        
    whitened, mu, w = compute_whitening(np.array(embeddings), k=DIMENSIONALITY_REDUCTION)
    embeddings = whitened


output = {
    "id": ids,
    "embedding": torch.tensor(np.array(embeddings), dtype=torch.float32),
    "label": torch.tensor(labels),
}

if WHITENING:
    output["mu"] = torch.tensor(mu, dtype=torch.float32)
    output["w"] = torch.tensor(w, dtype=torch.float32)


torch.save(output, outputPath)
