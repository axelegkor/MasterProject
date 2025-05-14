from dataclasses import dataclass
import numpy as np

@dataclass
class Antigen:
    id: str
    embedding: np.ndarray
    label: int
