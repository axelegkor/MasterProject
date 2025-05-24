from dataclasses import dataclass
import torch

@dataclass
class Antigen:
    id: str
    embedding: torch.Tensor
    label: int
