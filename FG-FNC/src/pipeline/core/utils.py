from typing import List
from antibody import Antibody
from antigen import Antigen
import torch


def batch_is_recognized(
    antibodies: List[Antibody], antigens: List[Antigen]
) -> torch.Tensor:
    C = torch.stack([antibody.center for antibody in antibodies])
    M = torch.stack([antibody.multiplier for antibody in antibodies])
    R = torch.tensor([antibody.radii for antibody in antibodies], device=C.device)
    E = torch.stack([antigen.embedding for antigen in antigens])
    diff = (E[None, :, :] - C[:, None, :]) / M[:, None, :]
    distances = torch.linalg.vector_norm(diff, dim=2)
    return distances <= R[:, None]
