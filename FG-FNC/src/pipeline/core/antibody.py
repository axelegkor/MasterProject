import torch
from dataclasses import dataclass

import torch.linalg
from antigen import Antigen


@dataclass
class Antibody:
    id: str
    center: (
        torch.Tensor
    )  # A vector that represents the antibody's position in the embedding space
    radii: float  # The hyperspherical radius of the antibody's recognition region
    multiplier: (
        torch.Tensor
    )  # A vector that represents the scaling of the antibody's recognition region
    label: int  # The classification label of the antibody
    accuracy: float = None  # The accuracy of the antibody in recognizing antigens

    # Function to calculate the vector distance between the antibody's center and an antigen
    def vector_distance(self, antigen: Antigen) -> torch.Tensor:
        return antigen.embedding - self.center

    # Function to calculate the scaled down vector distance with the antibody's radii multiplier
    def scaled_vector_distance(self, antigen: Antigen) -> torch.Tensor:
        return self.vector_distance(antigen) / self.multiplier

    # Function to calculate the scaled euclidean distance between the antibody's center and an antigen
    def scaled_distance(self, antigen: Antigen) -> float:
        return torch.linalg.vector_norm(self.scaled_vector_distance(antigen)).item()

    # Function to check if an antigen is recognized by the antibody
    def is_recognized(self, antigen: Antigen) -> bool:
        return self.scaled_distance(antigen) <= self.radii

    # Functions for antibody mutation
    def center_mutation(self, movement: torch.Tensor) -> None:
        self.center += movement

    def radii_mutation(self, change: float) -> None:
        self.radii *= change

    def multiplier_mutation(self, change: torch.Tensor) -> None:
        self.multiplier *= change
        