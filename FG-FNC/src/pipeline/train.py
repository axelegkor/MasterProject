import numpy as np
from typing import List, Dict, Tuple
from core.antigen import Antigen
from core.antibody import Antibody
import uuid
import copy

# Hyperparameters
DATA_PATH = "data/training_data.csv"
NUM_CLASSES = 5
EMBEDDING_DIM = 384
INITIALISATION_METHOD = "random"  # Options: "random", "informed"
POPULATION_SIZE = 100
NUM_GENERATIONS = 1000
REPLACEMENT_RATION_MAX = 0.6  # The replacement ratio in the first generation
REPLACEMENT_RATION_MIN = 0.01  # The replacement ratio in the last generation
ERROR_SCALING = (
    2  # Scaling factor for how much a false positive affects the correctness score
)
CORRECTNESS_WEIGHT = 2.5  # Weight of the correctness part of the fitness score
COVERAGE_WEIGHT = 1.0  # Weight of the coverage part of the fitness score
CORRECTNESS_TYPE = "continuous"  # Options: "continuous", "binary"
CORRECTNESS_EXPONENT = 0.5  # Exponent for the calculation of continuous correctness
UNIQUENESS_WEIGHT = 1.2  # Weight of the uniqueness part of the fitness score
MIN_FITNESS_SCALING = 0.2  # Minimum scaling factor for the fitness of an antibody
MAX_FITNESS_SCALING = 2  # Maximum scaling factor for the fitness of an antibody
NUM_CLONES = 10  # Number of clones to create for each antibody
CENTER_MUTATION_RATE = 1  # Probability of mutating the center of an antibody
RADII_MUTATION_RATE = 1  # Probability of mutating the radii of an antibody
MULTIPLIER_MUTATION_RATE = 1  # Probability of mutating the multiplier of an antibody
CENTER_MUTATION_INTENSITY = 0.02  # Intensity of the center mutations, relative to the mean norm of the population
CENTER_MUTATION_DIMENSIONS = (
    0.1  # Portion of dimensions to mutate for the center mutations
)
RADII_MUTATION_INTENSITY = 1  # Intensity of the radii mutations
MIN_RADII_MUTATION = (
    0.2  # Minimum scaling factor for the radii of an antibody, must be > 0
)
MAX_RADII_MUTATION = 5  # Maximum scaling factor for the radii of an antibody
MULTIPLIER_MUTATION_INTENSITY = 0.2  # Intensity of the multiplier mutations
MULTIPLIER_MUTATION_DIMENSIONS = (
    0.1  # Portion of dimensions to mutate for the multiplier mutations
)


# Function to load the training data into a list of Antigen objects, and return the average embedding
def load_training_data(file_path: str) -> Tuple[List[Antigen], float]:
    # TODO: Implement the function to load training data from a file

    return [], 0.0  # Placeholder return value


# Helper function to calculate the shortest distance between a point in an antigen with a different label
def max_radius(point: np.ndarray, antigens: List[Antigen], label: int) -> float:
    distances = [
        np.linalg.norm(point - antigen.embedding)
        for antigen in antigens
        if antigen.label != label
    ]
    return max(distances)


# Helper function to calculate the center of mass of a population of antigens
def center_mean(antigens: List[Antigen]) -> np.ndarray:
    return np.mean([antigen.embedding for antigen in antigens], axis=0)


def center_std(antigens: List[Antigen]) -> float:
    return np.std([antigen.embedding for antigen in antigens])


# Function to randomly initialise a population of antibodies
def random_initialisation(
    antigens: List[Antigen], population_size: int
) -> List[Antibody]:
    rng = np.random.default_rng()
    mean = center_mean(antigens)
    std = center_std(antigens)
    antibodies = []
    for _ in range(population_size):
        center = rng.normal(mean, std, size=EMBEDDING_DIM)
        label = rng.integers(0, NUM_CLASSES)
        antibody = Antibody(
            id=str(uuid.uuid4()),
            center=center,
            radii=max_radius(center, antigens, label),
            multiplier=np.ones(EMBEDDING_DIM),
            label=label,
        )
        antibodies.append(antibody)


def informed_initialisation(
    antigens: List[Antigen], population_size: int
) -> List[Antibody]:
    antibodies = []
    coverage = np.ones(len(antigens))
    rng = np.random.default_rng()
    for _ in range(population_size):
        probabilities = 1 / coverage
        probabilities /= np.sum(probabilities)
        index = rng.choice(len(antigens), p=probabilities)
        antigen = antigens[index]
        antibody = Antibody(
            id=str(uuid.uuid4()),
            center=antigen.embedding,
            radii=max_radius(antigen.embedding, antigens, antigen.label),
            multiplier=np.ones(EMBEDDING_DIM),
            label=antigen.label,
        )
        antibodies.append(antibody)
        for i in range(len(antigens)):
            if antibody.is_recognized(antigens[i]):
                coverage[i] += 1
    return antibodies


def initialise_population(
    antigens: List[Antigen], population_size: int
) -> List[Antibody]:
    if INITIALISATION_METHOD == "random":
        return random_initialisation(antigens, population_size)
    elif INITIALISATION_METHOD == "informed":
        return informed_initialisation(antigens, population_size)
    else:
        raise ValueError(f"Unknown initialisation method: {INITIALISATION_METHOD}")


def shared_count(antibodies: List[Antibody], antigens: List[Antigen]) -> Dict[str, int]:
    shared_counts = {}
    for antigen in antigens:
        shared_count = 0
        for antibody in antibodies:
            if antibody.is_recognized(antigen):
                shared_count += 1
        shared_counts[antigen.id] = shared_count
    return shared_counts


# Function to calculate the correctness of antibodies
def fitness_of_antibodies_continuous_correctness(
    antibodies: List[Antibody], antigens: List[Antigen]
) -> Dict[str, float]:
    fitness_scores = {}
    shared_counts = shared_count(antibodies, antigens)
    for antibody in antibodies:
        correctness_count = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        shared_affinity = 0
        for antigen in antigens:
            if antibody.is_recognized(antigen):
                if antibody.label == antigen.label:
                    true_positives += 1
                else:
                    false_positives += 1
                shared_affinity += 1 / shared_counts[antigen.id]
                correctness_count += (
                    1
                    - (abs(antigen.label - antibody.label) / (NUM_CLASSES - 1))
                    ** CORRECTNESS_EXPONENT
                )
            else:
                if antibody.label == antigen.label:
                    false_negatives += 1
        correctness = (
            correctness_count / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        coverage = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        uniqueness = shared_affinity / true_positives if true_positives > 0 else 0
        fitness = (
            (CORRECTNESS_WEIGHT * correctness)
            + (COVERAGE_WEIGHT * coverage)
            + (UNIQUENESS_WEIGHT * uniqueness)
        )
        fitness_scores[antibody.id] = fitness
    return fitness_scores


# Function to calculate the correctness of antibodies
def fitness_of_antibodies_binary_correctness(
    antibodies: List[Antibody], antigens: List[Antigen]
) -> Dict[str, float]:
    fitness_scores = {}
    shared_counts = shared_count(antibodies, antigens)
    for antibody in antibodies:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        shared_affinity = 0
        for antigen in antigens:
            if antibody.is_recognized(antigen):
                if antibody.label == antigen.label:
                    true_positives += 1
                else:
                    false_positives += 1
                shared_affinity += 1 / shared_counts[antigen.id]
            else:
                if antibody.label == antigen.label:
                    false_negatives += 1
        correctness = (
            (true_positives - (false_positives * ERROR_SCALING))
            / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        coverage = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        uniqueness = shared_affinity / true_positives if true_positives > 0 else 0
        fitness = (
            (CORRECTNESS_WEIGHT * correctness)
            + (COVERAGE_WEIGHT * coverage)
            + (UNIQUENESS_WEIGHT * uniqueness)
        )
        fitness_scores[antibody.id] = fitness
    return fitness_scores


def fitness_of_antibodies(
    antibodies: List[Antibody], antigens: List[Antigen]
) -> Dict[str, float]:
    if CORRECTNESS_TYPE == "continuous":
        return fitness_of_antibodies_continuous_correctness(antibodies, antigens)
    elif CORRECTNESS_TYPE == "binary":
        return fitness_of_antibodies_binary_correctness(antibodies, antigens)
    else:
        raise ValueError(f"Unknown correctness type: {CORRECTNESS_TYPE}")


# Function that performs center mutation on an antibody
def center_mutation(
    antibody: Antibody,
    fitness_scaling: float,
    mean_norm: float,
    rng: np.random.Generator,
) -> None:
    intensity = fitness_scaling * CENTER_MUTATION_INTENSITY * mean_norm
    movement = np.zeros(EMBEDDING_DIM)
    dim_mask = rng.random(EMBEDDING_DIM) < CENTER_MUTATION_DIMENSIONS
    movement[dim_mask] = rng.normal(0, intensity, size=np.sum(dim_mask))
    antibody.center_mutation(movement)


# Function that performs radii mutation on an antibody
def radii_mutation(
    antibody: Antibody, fitness_scaling: float, rng: np.random.Generator
) -> None:
    intensity = fitness_scaling * RADII_MUTATION_INTENSITY
    change = np.clip(rng.normal(1, intensity), MIN_RADII_MUTATION, MAX_RADII_MUTATION)
    antibody.radii_mutation(change)


# Function that performs multiplier mutation on an antibody
def multiplier_mutation(
    antibody: Antibody,
    fitness_scaling: float,
    rng: np.random.Generator,
) -> None:
    intensity = fitness_scaling * MULTIPLIER_MUTATION_INTENSITY
    change = np.ones(EMBEDDING_DIM)
    dim_mask = rng.random(EMBEDDING_DIM) < MULTIPLIER_MUTATION_DIMENSIONS
    change[dim_mask] = rng.normal(1, intensity, size=np.sum(dim_mask))
    antibody.multiplier_mutation(change)


# Function using crowding to clone and mutate antibodies
def crowding(
    antibodies: List[Antibody],
    replacement_ratio: float,
    fitness_scores: Dict[str, float],
    mean_norm: float,
    antigens: List[Antigen],
) -> List[Antibody]:
    antibodies_eligible = antibodies.copy()
    updated_antibodies = antibodies.copy()
    rng = np.random.default_rng()
    num_replacements = int(len(antibodies) * replacement_ratio)
    avg_fitness = np.mean(list(fitness_scores.values()))
    for _ in range(num_replacements):
        index = rng.integers(0, len(antibodies_eligible))
        antibody = antibodies_eligible[index]
        antibodies_eligible.pop(index)
        fitness = fitness_scores[antibody.id]
        fitness_scaling = np.clip(
            avg_fitness / (fitness + 1e-8), MIN_FITNESS_SCALING, MAX_FITNESS_SCALING
        )
        clones = [antibody]
        for i in range(NUM_CLONES):
            clone = copy.deepcopy(antibody)
            clone.id = str(uuid.uuid4())
            probabilities = [
                CENTER_MUTATION_RATE,
                RADII_MUTATION_RATE,
                MULTIPLIER_MUTATION_RATE,
            ] / np.sum(
                [CENTER_MUTATION_RATE, RADII_MUTATION_RATE, MULTIPLIER_MUTATION_RATE]
            )
            mutation_type = rng.choice(
                ["center", "radii", "multiplier"], p=probabilities
            )
            if mutation_type == "center":
                center_mutation(clone, fitness_scaling, mean_norm, rng)
            elif mutation_type == "radii":
                radii_mutation(clone, fitness_scaling, rng)
            elif mutation_type == "multiplier":
                multiplier_mutation(clone, fitness_scaling, rng)

            clones.append(clone)
        clone_fitness_scores = fitness_of_antibodies(clones, antigens)
        best_clone_id = max(clone_fitness_scores, key=clone_fitness_scores.get)
        best_clone = next(clone for clone in clones if clone.id == best_clone_id)
        updated_antibodies[index] = best_clone


# Main function to run the training process
def main():
    # Load training data
    antigens = load_training_data(DATA_PATH)

    # Initialise population of antibodies
    population = initialise_population(antigens, POPULATION_SIZE, INITIALISATION_METHOD)
