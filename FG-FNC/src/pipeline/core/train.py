import numpy as np
from typing import List, Dict, Tuple
from classes.antigen import Antigen
from classes.antibody import Antibody
import uuid
import copy
import torch


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
    num_classes: int,
) -> List[Antibody]:
    rng = np.random.default_rng()
    antibodies = []
    for _ in range(population_size):
        center = rng.normal(mean, std, size=embedding_dim)
        label = rng.integers(0, num_classes)
        antibody = Antibody(
            id=str(uuid.uuid4()),
            center=center,
            radii=max_radius(center, antigens, label),
            multiplier=np.ones(embedding_dim),
            label=label,
        )
        antibodies.append(antibody)

    return antibodies


def antigen_based_initialisation(
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
    num_classes: int,
) -> List[Antibody]:
    if initialisation_method == "random":
        return random_initialisation(
            antigens=antigens,
            population_size=population_size,
            mean=mean,
            std=std,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
        )
    elif initialisation_method == "antigen_based":
        return antigen_based_initialisation(
            antigens=antigens,
            population_size=population_size,
            coverage=coverage,
            embedding_dim=embedding_dim,
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
    antibodies: List[Antibody],
    antigens: List[Antigen],
    num_classes: int,
    correctness_exponent: float,
    correctness_weight: float,
    coverage_weight: float,
    uniqueness_weight: float,
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
                    - (abs(antigen.label - antibody.label) / (num_classes - 1))
                    ** correctness_exponent
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
            (correctness_weight * correctness)
            + (coverage_weight * coverage)
            + (uniqueness_weight * uniqueness)
        )
        fitness_scores[antibody.id] = fitness
    return fitness_scores


# Function to calculate the correctness of antibodies
def fitness_of_antibodies_binary_correctness(
    antibodies: List[Antibody],
    antigens: List[Antigen],
    error_scaling: float,
    correctness_weight: float,
    coverage_weight: float,
    uniqueness_weight: float,
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
            (true_positives - (false_positives * error_scaling))
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
            (correctness_weight * correctness)
            + (coverage_weight * coverage)
            + (uniqueness_weight * uniqueness)
        )
        fitness_scores[antibody.id] = fitness
    return fitness_scores


def fitness_of_antibodies(
    antibodies: List[Antibody],
    antigens: List[Antigen],
    correctness_type: str,
    correctness_weight: float,
    coverage_weight: float,
    uniqueness_weight: float,
    num_classes: int,
    correctness_exponent: float,
    error_scaling: float,
) -> Dict[str, float]:
    if correctness_type == "continuous":
        return fitness_of_antibodies_continuous_correctness(
            antibodies,
            antigens,
            num_classes=num_classes,
            correctness_exponent=correctness_exponent,
            correctness_weight=correctness_weight,
            coverage_weight=coverage_weight,
            uniqueness_weight=uniqueness_weight,
        )
    elif correctness_type == "binary":
        return fitness_of_antibodies_binary_correctness(
            antibodies,
            antigens,
            error_scaling=error_scaling,
            correctness_weight=correctness_weight,
            coverage_weight=coverage_weight,
            uniqueness_weight=uniqueness_weight,
        )
    else:
        raise ValueError(f"Unknown correctness type: {correctness_type}")


# Function that performs center mutation on an antibody
def center_mutation(
    antibody: Antibody,
    fitness_scaling: float,
    center_std: float,
    rng: np.random.Generator,
    embedding_dim: int,
    center_mutation_intensity: float,
    center_mutation_dimensions: float,
) -> None:
    intensity = fitness_scaling * center_mutation_intensity * center_std
    movement = np.zeros(embedding_dim)
    dim_mask = rng.random(embedding_dim) < center_mutation_dimensions
    movement[dim_mask] = rng.normal(0, intensity, size=np.sum(dim_mask))
    antibody.center_mutation(movement)


# Function that performs radii mutation on an antibody
def radii_mutation(
    antibody: Antibody,
    fitness_scaling: float,
    rng: np.random.Generator,
    radii_mutation_intensity: float,
    min_radii_mutation: float,
    max_radii_mutation: float,
) -> None:
    intensity = fitness_scaling * radii_mutation_intensity
    change = np.clip(rng.normal(1, intensity), min_radii_mutation, max_radii_mutation)
    antibody.radii_mutation(change)


# Function that performs multiplier mutation on an antibody
def multiplier_mutation(
    antibody: Antibody,
    fitness_scaling: float,
    rng: np.random.Generator,
    embedding_dim: int,
    multiplier_mutation_intensity: float,
    multiplier_mutation_dimensions: float,
) -> None:
    intensity = fitness_scaling * multiplier_mutation_intensity
    change = np.ones(embedding_dim)
    dim_mask = rng.random(embedding_dim) < multiplier_mutation_dimensions
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
    center_mutation_intensity: float,
    center_mutation_dimensions: float,
    radii_mutation_intensity: float,
    min_radii_mutation: float,
    max_radii_mutation: float,
    multiplier_mutation_intensity: float,
    multiplier_mutation_dimensions: float,
    min_fitness_scaling: float,
    max_fitness_scaling: float,
    num_clones: int,
    center_mutation_rate: float,
    radii_mutation_rate: float,
    multiplier_mutation_rate: float,
    correctness_type: str,
    correctness_weight: float,
    coverage_weight: float,
    uniqueness_weight: float,
    num_classes: int,
    correctness_exponent: float,
    error_scaling: float,
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
            avg_fitness / (fitness + 1e-8), min_fitness_scaling, max_fitness_scaling
        )
        clones = [antibody]
        for _ in range(num_clones):
            clone = copy.deepcopy(antibody)
            clone.id = str(uuid.uuid4())
            probabilities = [
                center_mutation_rate,
                radii_mutation_rate,
                multiplier_mutation_rate,
            ] / np.sum(
                [center_mutation_rate, radii_mutation_rate, multiplier_mutation_rate]
            )
            mutation_type = rng.choice(
                ["center", "radii", "multiplier"], p=probabilities
            )
            if mutation_type == "center":
                center_mutation(
                    antibody=clone,
                    fitness_scaling=fitness_scaling,
                    center_std=center_std,
                    rng=rng,
                    embedding_dim=embedding_dim,
                    center_mutation_intensity=center_mutation_intensity,
                    center_mutation_dimensions=center_mutation_dimensions,
                )
            elif mutation_type == "radii":
                radii_mutation(
                    antibody=clone,
                    fitness_scaling=fitness_scaling,
                    rng=rng,
                    radii_mutation_intensity=radii_mutation_intensity,
                    min_radii_mutation=min_radii_mutation,
                    max_radii_mutation=max_radii_mutation,
                )
            elif mutation_type == "multiplier":
                multiplier_mutation(
                    antibody=clone,
                    fitness_scaling=fitness_scaling,
                    rng=rng,
                    embedding_dim=embedding_dim,
                    multiplier_mutation_intensity=multiplier_mutation_intensity,
                    multiplier_mutation_dimensions=multiplier_mutation_dimensions,
                )

            clones.append(clone)
        clone_fitness_scores = fitness_of_antibodies(
            antibodies=clones,
            antigens=antigens,
            correctness_type=correctness_type,
            correctness_weight=correctness_weight,
            coverage_weight=coverage_weight,
            uniqueness_weight=uniqueness_weight,
            num_classes=num_classes,
            correctness_exponent=correctness_exponent,
            error_scaling=error_scaling,
        )
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
def train(config) -> Tuple[List[Antibody], np.ndarray, np.ndarray]:
    if not config.WHITENING:
        whitening_str = "nw"
    else:
        whitening_str = str(config.DIMENSIONALITY_REDUCTION)

    input_file_path = (
        "/cluster/work/axelle/Datasets-embedded/"
        + config.DATASET
        + "/"
        + whitening_str
        + "/"
        + "train"
        + ".pt"
    )
    # Load training data
    print(f"📥 Loading training data from: {input_file_path}")
    antigens, embedding_dim, mu, w = load_training_data(input_file_path)
    print(
        f"✅ Loaded {len(antigens)} antigens with embedding dimension {embedding_dim}"
    )

    # Calculate the mean and standard deviation of the antigens
    mean = center_mean(antigens)
    std = center_std(antigens)

    # Initialise population of antibodies
    print(f"⚙️ Initialising population using method: {config.INITIALISATION_METHOD}")
    population = initialise_population(
        antigens=antigens,
        population_size=config.POPULATION_SIZE * len(antigens),
        initialisation_method=config.INITIALISATION_METHOD,
        coverage=np.ones(len(antigens)),
        mean=mean,
        std=std,
        embedding_dim=embedding_dim,
        num_classes=config.NUM_CLASSES,
    )

    print(f"👾 Initial population size: {len(population)}")

    total_leaking_amount = int(round(config.POPULATION_SIZE * config.TOTAL_LEAKING))

    leaking_per_generation = int(
        round(
            total_leaking_amount
            / round((config.NUM_GENERATIONS / config.LEAKING_FREQUENCY))
        )
    )

    for generation in range(config.NUM_GENERATIONS):
        print(f"🔄 Generation {generation + 1}/{config.NUM_GENERATIONS}")
        replacement_ratio = (
            config.REPLACEMENT_RATIO_MAX
            - (config.REPLACEMENT_RATIO_MAX - config.REPLACEMENT_RATIO_MIN)
            * generation
            / config.NUM_GENERATIONS
        )
        fitness_scores = fitness_of_antibodies(
            antibodies=population,
            antigens=antigens,
            correctness_type=config.CORRECTNESS_TYPE,
            correctness_weight=config.CORRECTNESS_WEIGHT,
            coverage_weight=config.COVERAGE_WEIGHT,
            uniqueness_weight=config.UNIQUENESS_WEIGHT,
            num_classes=config.NUM_CLASSES,
            correctness_exponent=config.CORRECTNESS_EXPONENT,
            error_scaling=config.ERROR_SCALING,
        )
        population = crowding(
            antibodies=population,
            replacement_ratio=replacement_ratio,
            fitness_scores=fitness_scores,
            center_std=std,
            antigens=antigens,
            embedding_dim=embedding_dim,
            center_mutation_intensity=config.CENTER_MUTATION_INTENSITY,
            center_mutation_dimensions=config.CENTER_MUTATION_DIMENSIONS,
            radii_mutation_intensity=config.RADII_MUTATION_INTENSITY,
            min_radii_mutation=config.MIN_RADII_MUTATION,
            max_radii_mutation=config.MAX_RADII_MUTATION,
            multiplier_mutation_intensity=config.MULTIPLIER_MUTATION_INTENSITY,
            multiplier_mutation_dimensions=config.MULTIPLIER_MUTATION_DIMENSIONS,
            min_fitness_scaling=config.MIN_FITNESS_SCALING,
            max_fitness_scaling=config.MAX_FITNESS_SCALING,
            num_clones=config.NUM_CLONES,
            center_mutation_rate=config.CENTER_MUTATION_RATE,
            radii_mutation_rate=config.RADII_MUTATION_RATE,
            multiplier_mutation_rate=config.MULTIPLIER_MUTATION_RATE,
            correctness_type=config.CORRECTNESS_TYPE,
            correctness_weight=config.CORRECTNESS_WEIGHT,
            coverage_weight=config.COVERAGE_WEIGHT,
            uniqueness_weight=config.UNIQUENESS_WEIGHT,
            num_classes=config.NUM_CLASSES,
            correctness_exponent=config.CORRECTNESS_EXPONENT,
            error_scaling=config.ERROR_SCALING,
        )

        # Leaking mechanism
        if config.TOTAL_LEAKING != 0 and generation % config.LEAKING_FREQUENCY == 0:
            leaking_population = initialise_population(
                antigens=antigens,
                population_size=leaking_per_generation,
                initialisation_method=config.INITIALISATION_METHOD,
                coverage=coverage(population, antigens),
                mean=mean,
                std=std,
                embedding_dim=embedding_dim,
                num_classes=config.NUM_CLASSES,
            )
            population.extend(leaking_population)

    population_with_accuracy = []
    for antibody in population:
        antibody.accuracy = accuracy(antibody, antigens)
        population_with_accuracy.append(antibody)
    population = population_with_accuracy

    print(f"🏁 Training complete")
    return population, mu, w
