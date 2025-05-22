import numpy as np
from typing import List, Dict, Tuple
from core.antigen import Antigen
from core.antibody import Antibody
import uuid
import copy
import torch
import config

# Hyperparameters
DATASET = "Liar"
FILENAME = "train"
DIMENSIONALITY_REDUCTION = 128
WHITENING = True

NUM_CLASSES = 5
INITIALISATION_METHOD = "random"  # Options: "random", "informed"
POPULATION_SIZE = (
    0.01  # Initial population size as a fraction of the training data size
)
NUM_GENERATIONS = 1000
REPLACEMENT_RATIO_MAX = 0.6  # The replacement ratio in the first generation
REPLACEMENT_RATIO_MIN = 0.01  # The replacement ratio in the last generation
LEAKING_FREQUENCY = 10  # Frequency of leaking
TOTAL_LEAKING = 0.5  # Total leaking ratio compared to the population size
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

nw = ""
if not WHITENING:
    nw = "_nw"

input_file_path = (
    "/cluster/work/axelle/Datasets-embedded/"
    + DATASET
    + "/"
    + FILENAME
    + "_"
    + str(DIMENSIONALITY_REDUCTION)
    + nw
    + ".pt"
)


# Function to load the training data into a list of Antigen objects
def load_training_data(
    file_path: str,
) -> Tuple[List[Antigen], float, np.ndarray, np.ndarray]:
    data = torch.load(file_path)

    ids = data["id"]
    embeddings = data["embedding"]
    labels = data["label"]
    mu = data.get("mu", None)
    w = data.get("w", None)

    antigens = []
    for i in range(len(ids)):
        antigen = Antigen(
            id=ids[i],
            embedding=embeddings[i],
            label=labels[i],
        )
        antigens.append(antigen)

    return antigens, embeddings.shape[1], mu, w


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
    antigens: List[Antigen],
    population_size: int,
    mean: np.ndarray,
    std: float,
    embedding_dim: int,
) -> List[Antibody]:
    rng = np.random.default_rng()
    antibodies = []
    for _ in range(population_size):
        center = rng.normal(mean, std, size=embedding_dim)
        label = rng.integers(0, NUM_CLASSES)
        antibody = Antibody(
            id=str(uuid.uuid4()),
            center=center,
            radii=max_radius(center, antigens, label),
            multiplier=np.ones(embedding_dim),
            label=label,
        )
        antibodies.append(antibody)


def informed_initialisation(
    antigens: List[Antigen],
    population_size: int,
    coverage: np.ndarray,
    embedding_dim: int,
) -> List[Antibody]:
    antibodies = []
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
            multiplier=np.ones(embedding_dim),
            label=antigen.label,
        )
        antibodies.append(antibody)
        for i in range(len(antigens)):
            if antibody.is_recognized(antigens[i]):
                coverage[i] += 1
    return antibodies


def initialise_population(
    antigens: List[Antigen],
    population_size: int,
    initialisation_method: str,
    coverage: np.ndarray,
    mean: np.ndarray,
    std: float,
    embedding_dim: int,
) -> List[Antibody]:
    if initialisation_method == "random":
        return random_initialisation(
            antigens, population_size, mean, std, embedding_dim
        )
    elif initialisation_method == "informed":
        return informed_initialisation(
            antigens, population_size, coverage, embedding_dim
        )
    else:
        raise ValueError(f"Unknown initialisation method: {initialisation_method}")


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
    antibodies: List[Antibody], antigens: List[Antigen], correctness_type: str
) -> Dict[str, float]:
    if correctness_type == "continuous":
        return fitness_of_antibodies_continuous_correctness(antibodies, antigens)
    elif correctness_type == "binary":
        return fitness_of_antibodies_binary_correctness(antibodies, antigens)
    else:
        raise ValueError(f"Unknown correctness type: {correctness_type}")


# Function that performs center mutation on an antibody
def center_mutation(
    antibody: Antibody,
    fitness_scaling: float,
    center_std: float,
    rng: np.random.Generator,
    embedding_dim: int,
) -> None:
    intensity = fitness_scaling * CENTER_MUTATION_INTENSITY * center_std
    movement = np.zeros(embedding_dim)
    dim_mask = rng.random(embedding_dim) < CENTER_MUTATION_DIMENSIONS
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
    embedding_dim: int,
) -> None:
    intensity = fitness_scaling * MULTIPLIER_MUTATION_INTENSITY
    change = np.ones(embedding_dim)
    dim_mask = rng.random(embedding_dim) < MULTIPLIER_MUTATION_DIMENSIONS
    change[dim_mask] = rng.normal(1, intensity, size=np.sum(dim_mask))
    antibody.multiplier_mutation(change)


# Function using crowding to clone and mutate antibodies
def crowding(
    antibodies: List[Antibody],
    replacement_ratio: float,
    fitness_scores: Dict[str, float],
    center_std: float,
    antigens: List[Antigen],
    embedding_dim: int,
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
                center_mutation(clone, fitness_scaling, center_std, rng, embedding_dim)
            elif mutation_type == "radii":
                radii_mutation(clone, fitness_scaling, rng)
            elif mutation_type == "multiplier":
                multiplier_mutation(clone, fitness_scaling, rng, embedding_dim)

            clones.append(clone)
        clone_fitness_scores = fitness_of_antibodies(clones, antigens)
        best_clone_id = max(clone_fitness_scores, key=clone_fitness_scores.get)
        best_clone = next(clone for clone in clones if clone.id == best_clone_id)
        updated_antibodies[index] = best_clone

    return updated_antibodies


def coverage(antibodies: List[Antibody], antigens: List[Antigen]) -> np.ndarray:
    coverage = np.ones(len(antigens))
    for antibody in antibodies:
        for i in range(len(antigens)):
            if antibody.is_recognized(antigens[i]):
                coverage[i] += 1
    return coverage

def accuracy(antibody: Antibody, antigens: List[Antigen]) -> float:
    positives = 0
    true_positives = 0
    for antigen in antigens:
        if antibody.is_recognized(antigen):
            positives += 1
            if antibody.label == antigen.label:
                true_positives += 1
    return true_positives / positives if positives > 0 else 0


# Main function to run the training process
def main():
    # Load training data
    antigens, embedding_dim, mu, w = load_training_data(input_file_path)

    # Calculate the mean and standard deviation of the antigens
    mean = center_mean(antigens)
    std = center_std(antigens)

    # Initialise population of antibodies
    population = initialise_population(
        antigens,
        POPULATION_SIZE,
        INITIALISATION_METHOD,
        np.ones(len(antigens)),
        mean,
        std,
        embedding_dim,
    )

    total_leaking_amount = int(round(POPULATION_SIZE * TOTAL_LEAKING))

    leaking_per_generation = int(
        round(total_leaking_amount / round((NUM_GENERATIONS / LEAKING_FREQUENCY)))
    )

    for generation in range(NUM_GENERATIONS):

        replacement_ratio = (
            REPLACEMENT_RATIO_MAX
            - (REPLACEMENT_RATIO_MAX - REPLACEMENT_RATIO_MIN)
            * generation
            / NUM_GENERATIONS
        )
        fitness_scores = fitness_of_antibodies(
            antibodies=population, antigens=antigens, correctness_type=CORRECTNESS_TYPE
        )
        population = crowding(
            antibodies=population,
            replacement_ratio=replacement_ratio,
            fitness_scores=fitness_scores,
            center_std=std,
            antigens=antigens,
        )

        # Leaking mechanism
        if TOTAL_LEAKING != 0 and generation % LEAKING_FREQUENCY == 0:
            leaking_population = initialise_population(
                antigens,
                leaking_per_generation,
                INITIALISATION_METHOD,
                coverage(population, antigens),
            )
            population.extend(leaking_population)
            
    population_with_accuracy = []
    for antibody in population:
        antibody.accuracy = accuracy(antibody, antigens)
        population_with_accuracy.append(antibody)
    population = population_with_accuracy

    # Save the final population
    output = {
        "id": [antibody.id for antibody in population],
        "center": torch.tensor(
            [antibody.center for antibody in population], dtype=torch.float32
        ),
        "radii": torch.tensor(
            [antibody.radii for antibody in population], dtype=torch.float32
        ),
        "multiplier": torch.tensor(
            [antibody.multiplier for antibody in population], dtype=torch.float32
        ),
        "label": torch.tensor(
            [antibody.label for antibody in population], dtype=torch.int
        ),
        "accuracy": torch.tensor(
            [antibody.accuracy for antibody in population], dtype=torch.float32
        ),
    }
    
    if mu is not None:
        output["mu"] = torch.tensor(mu, dtype=torch.float32)
    if w is not None:
        output["w"] = torch.tensor(w, dtype=torch.float32)

    output_file_path = "x"

    torch.save(output, output_file_path)
