import torch
import numpy as np
from core.antibody import Antibody
from core.antigen import Antigen
from typing import List, Tuple

def load_model(path: str) -> Tuple[List[Antibody], np.ndarray, np.ndarray]:
    data = torch.load(path)
    mu = data.get("mu", None)
    w = data.get("w", None)
    if mu is not None:
        mu = mu.numpy()
    if w is not None:
        w = w.numpy()
        
    antibodies: List[Antibody] = []
    for i in range(len(data["id"])):
        id = data["id"][i]
        center = data["embedding"][i].numpy()
        radii = data["radii"][i].item()
        multiplier = data["multiplier"][i].numpy()
        label = data["label"][i].item()
        accuracy = data["accuracy"][i].item()
        antibody = Antibody(id, center, radii, multiplier, label, accuracy)
        antibodies.append(antibody)
    return antibodies, mu, w

def load_antigens(path: str) -> List[Antigen]:
    data = torch.load(file_path)

    ids = data["id"]
    embeddings = data["embedding"]
    labels = data["label"]

    antigens = []
    for i in range(len(ids)):
        antigen = Antigen(
            id=ids[i],
            embedding=embeddings[i],
            label=labels[i],
        )
        antigens.append(antigen)
        
    return antigens