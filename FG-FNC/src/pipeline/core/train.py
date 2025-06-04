from typing import List, Dict, Tuple
from antibody import Antibody
from antigen import Antigen
from utils import batch_is_recognized
import uuid
import copy
import torch


# Function to load the training data into a list of Antigen objects
def load_training_data(
    file_path: str,
    max_dataset_size: int,
) -> Tuple[List[Antigen], float]:
    data = torch.load(file_path)

    ids = data["id"]
    embeddings = data["embedding"]
    labels = data["label"]

    num_samples = min(max_dataset_size, len(ids))
    indices = torch.randperm(len(ids))[:num_samples]

    antigens = []
    for i in indices:
        antigen = Antigen(
            id=ids[i],
            embedding=embeddings[i].to("cuda"),
            label=labels[i],
        )
        antigens.append(antigen)

    return antigens, embeddings.shape[1]


# Helper function to determine the number of classes based on the dataset
def get_num_classes(dataset: str) -> int:
    if dataset == "Liar":
        return 6
    elif dataset == "Politifact":
        return 6
    elif dataset == "Averitec":
        return 3
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# # Function to calculate the maximum radius for an antibody based on a point and a list of antigens
# def max_radius(point: torch.Tensor, antigens: List[Antigen]) -> float:
#     # Stack all antigen embeddings
#     all_embeddings = torch.stack([antigen.embedding for antigen in antigens])

#     # Compute distances from the point to all antigens
#     distances = torch.linalg.vector_norm(all_embeddings - point, dim=1)

#     # Determine the n-th nearest (0.1% of total antigens, at least 1)
#     n = max(5, int(0.001 * len(antigens)))
#     sorted_distances, _ = torch.sort(distances)

#     return sorted_distances[n - 1].item()


# Helper function to calculate the shortest distance between a point and an antigen with a different label
def max_radius(point: torch.Tensor, antigens: List[Antigen], label: int) -> float:
    filtered = torch.stack(
        [antigen.embedding for antigen in antigens if antigen.label != label]
    )
    distances = torch.linalg.vector_norm(filtered - point, dim=1)
    return torch.min(distances).item()


# Helper function to calculate the center of mass of a population of antigens
def center_mean(antigens: List[Antigen]) -> torch.Tensor:
    return torch.stack([antigen.embedding for antigen in antigens]).mean(dim=0)


def center_std(antigens: List[Antigen]) -> float:
    return torch.stack([antigen.embedding for antigen in antigens]).std().item()


# Function to randomly initialise a population of antibodies
def random_initialisation(
    antigens: List[Antigen],
    population_size: int,
    mean: torch.Tensor,
    std: float,
    embedding_dim: int,
    num_classes: int,
) -> List[Antibody]:
    antibodies = []
    for i in range(population_size):
        print(f"Initialising: {i + 1}/{population_size}", flush=True)
        center = torch.normal(mean=mean, std=std).to("cuda")
        label = torch.randint(0, num_classes, (1,)).item()
        print(f"Label: {label}", flush=True)
        print(f"Name : {i + 1}", flush=True)
        antibody = Antibody(
            id=str(uuid.uuid4()),
            center=center,
            radii=(max_radius(center, antigens, label)),
            multiplier=torch.ones(embedding_dim, device="cuda"),
            label=label,
            name=i + 1,  # Optional name for the antibody
        )
        antibodies.append(antibody)

    return antibodies


def antigen_based_initialisation(
    antigens: List[Antigen],
    population_size: int,
    coverage: torch.Tensor,
    embedding_dim: int,
) -> List[Antibody]:
    antibodies = []
    for i in range(population_size):
        print(f"Initialising: {i + 1}/{population_size}", flush=True)
        probabilities = 1 / coverage
        probabilities /= probabilities.sum()

        index = torch.multinomial(probabilities, 1).item()
        antigen = antigens[index]
        antibody = Antibody(
            id=str(uuid.uuid4()),
            center=antigen.embedding.clone(),
            radii=max_radius(antigen.embedding, antigens, antigen.label),
            multiplier=torch.ones(embedding_dim, device="cuda"),
            label=antigen.label,
            name=i + 1,  # Optional name for the antibody
        )
        antibodies.append(antibody)
        for i in range(len(antigens)):
            if antibody.is_recognized(antigens[i]):
                coverage[i] += 1
        print(f"Name : {antibody.name}", flush=True)
        print(f"Label: {antibody.label}", flush=True)
        print(f"Radii: {antibody.radii}", flush=True)

    return antibodies


def initialise_population(
    antigens: List[Antigen],
    population_size: int,
    initialisation_method: str,
    coverage: torch.Tensor,
    mean: torch.Tensor,
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


# def shared_count(antibodies: List[Antibody], antigens: List[Antigen]) -> Dict[str, int]:
#     shared_counts = {}
#     counter = 0
#     for antigen in antigens:
#         counter += 1
#         shared_count = 0
#         for antibody in antibodies:
#             if antibody.is_recognized(antigen):
#                 shared_count += 1
#         shared_counts[antigen.id] = shared_count
#     return shared_counts


def shared_count(antibodies: List[Antibody], antigens: List[Antigen]) -> Dict[str, int]:
    recognized = batch_is_recognized(antibodies, antigens)[0]
    shared_counts = recognized.sum(dim=0)
    return {antigens[i].id: shared_counts[i].item() for i in range(len(antigens))}


# Function to calculate the correctness of antibodies
# def fitness_of_antibodies_continuous_correctness(
#     antibodies: List[Antibody],
#     antigens: List[Antigen],
#     num_classes: int,
#     correctness_exponent: float,
#     correctness_weight: float,
#     coverage_weight: float,
#     uniqueness_weight: float,
# ) -> Dict[str, float]:
#     fitness_scores = {}
#     shared_counts = shared_count(antibodies, antigens)
#     counter = 0
#     for antibody in antibodies:
#         counter += 1
#         print(
#             f"Calculating fitness for antibody {counter}/{len(antibodies)}", flush=True
#         )
#         correctness_count = 0
#         true_positives = 0
#         false_positives = 0
#         false_negatives = 0
#         shared_affinity = 0
#         for antigen in antigens:
#             if antibody.is_recognized(antigen):
#                 if antibody.label == antigen.label:
#                     true_positives += 1
#                 else:
#                     false_positives += 1
#                 shared_affinity += 1 / shared_counts[antigen.id]
#                 correctness_count += (
#                     1
#                     - (abs(antigen.label - antibody.label) / (num_classes - 1))
#                     ** correctness_exponent
#                 )
#             else:
#                 if antibody.label == antigen.label:
#                     false_negatives += 1
#         correctness = (
#             correctness_count / (true_positives + false_positives)
#             if (true_positives + false_positives) > 0
#             else 0
#         )
#         coverage = (
#             true_positives / (true_positives + false_negatives)
#             if (true_positives + false_negatives) > 0
#             else 0
#         )
#         uniqueness = shared_affinity / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
#         fitness = (
#             (correctness_weight * correctness)
#             + (coverage_weight * coverage)
#             + (uniqueness_weight * uniqueness)
#         )
#         fitness_scores[antibody.id] = fitness
#     return fitness_scores


# Function to calculate the correctness of antibodies
# def fitness_of_antibodies_binary_correctness(
#     antibodies: List[Antibody],
#     antigens: List[Antigen],
#     error_scaling: float,
#     correctness_weight: float,
#     coverage_weight: float,
#     uniqueness_weight: float,
# ) -> Dict[str, float]:
#     fitness_scores = {}
#     shared_counts = shared_count(antibodies, antigens)
#     for antibody in antibodies:
#         true_positives = 0
#         false_positives = 0
#         false_negatives = 0
#         shared_affinity = 0
#         for antigen in antigens:
#             if antibody.is_recognized(antigen):
#                 if antibody.label == antigen.label:
#                     true_positives += 1
#                 else:
#                     false_positives += 1
#                 shared_affinity += 1 / shared_counts[antigen.id]
#             else:
#                 if antibody.label == antigen.label:
#                     false_negatives += 1
#         correctness = (
#             (true_positives - (false_positives * error_scaling))
#             / (true_positives + false_positives)
#             if (true_positives + false_positives) > 0
#             else 0
#         )
#         coverage = (
#             true_positives / (true_positives + false_negatives)
#             if (true_positives + false_negatives) > 0
#             else 0
#         )
#         uniqueness = shared_affinity / true_positives if true_positives > 0 else 0
#         fitness = (
#             (correctness_weight * correctness)
#             + (coverage_weight * coverage)
#             + (uniqueness_weight * uniqueness)
#         )
#         fitness_scores[antibody.id] = fitness
#     return fitness_scores


# def fitness_of_antibodies_binary_correctness(
#     antibodies: List[Antibody],
#     antigens: List[Antigen],
#     error_scaling: float,
#     correctness_weight: float,
#     coverage_weight: float,
#     uniqueness_weight: float,
#     average_radii_approximation_weight: float,
#     avg_distance: float,
# ) -> Dict[str, float]:

#     # Precompute recognition matrix
#     recognized = batch_is_recognized(antibodies, antigens)[0]
#     shared_counts = recognized.sum(dim=0)  # [N]
#     antigen_labels = torch.tensor([a.label for a in antigens], device=recognized.device)

#     fitness_scores = {}

#     for i, antibody in enumerate(antibodies):
#         mask = recognized[i]
#         correct_mask = antigen_labels == antibody.label
#         true_positives = (mask & correct_mask).sum().item()
#         false_positives = (mask & ~correct_mask).sum().item()
#         false_negatives = (~mask & correct_mask).sum().item()

#         shared_affinity = (1 / shared_counts[mask]).sum().item() if mask.any() else 0

#         denom = true_positives + false_positives
#         correctness = (true_positives) / denom if denom > 0 else 0
#         if correctness < 0.15:
#             correctness = 0.15
#         coverage = (
#             true_positives / (true_positives + false_negatives)
#             if (true_positives + false_negatives) > 0
#             else 0
#         )
#         uniqueness = shared_affinity / denom if denom > 0 else 0
#         average_radii_approximation = 1 - (
#             (abs(antibody.radii - (avg_distance - (0.1 * avg_distance)))) / avg_distance
#         )
#         fitness = (
#             correctness_weight * correctness
#             + coverage_weight * coverage
#             + uniqueness_weight * uniqueness
#             + average_radii_approximation_weight * average_radii_approximation
#         )
#         # print("Correctness:", correctness_weight * correctness, flush=True)
#         # print("Coverage:", coverage_weight * coverage, flush=True)
#         # print("Uniqueness:", uniqueness_weight * uniqueness, flush=True)
#         fitness_scores[antibody.id] = fitness

#     return fitness_scores

# def fitness_of_antibodies_continuous_correctness(
#     antibodies: List[Antibody],
#     antigens: List[Antigen],
#     num_classes: int,
#     correctness_exponent: float,
#     correctness_weight: float,
#     coverage_weight: float,
#     uniqueness_weight: float,
# ) -> Dict[str, float]:

#     recognized = batch_is_recognized(antibodies, antigens)[0]

#     shared_counts = recognized.sum(dim=0)

#     antigen_labels = torch.tensor([a.label for a in antigens], device=recognized.device)
#     fitness_scores = {}

#     for i, antibody in enumerate(antibodies):

#         mask = recognized[i]
#         correct_mask = antigen_labels == antibody.label

#         true_positives = (mask & correct_mask).sum().item()
#         false_positives = (mask & ~correct_mask).sum().item()
#         false_negatives = (~mask & correct_mask).sum().item()


#         coverage = (
#             true_positives / (true_positives + false_negatives)
#             if (true_positives + false_negatives) > 0
#             else 0
#         )
#         uniqueness = shared_affinity / denom if denom > 0 else 0

#         fitness = (
#             correctness_weight * correctness
#             + coverage_weight * coverage
#             + uniqueness_weight * uniqueness
#         )
#         fitness_scores[antibody.id] = fitness

#     return fitness_scores


def fitness_of_antibodies(
    antibodies: List[Antibody],
    antigens: List[Antigen],
    num_classes: int,
    correctness_weight: float,
    correctness_type: str,
    correctness_exponent: float,
    coverage_weight: float,
    uniqueness_weight: float,
    avg_radii_approximation_weight: float,
    avg_offset: float,
    avg_distance: float,
) -> Dict[str, float]:

    # Precompute recognition matrix
    recognized = batch_is_recognized(antibodies, antigens)[0]
    shared_counts = recognized.sum(dim=0)  # [N]
    antigen_labels = torch.tensor([a.label for a in antigens], device=recognized.device)

    fitness_scores = {}

    for i, antibody in enumerate(antibodies):
        mask = recognized[i]
        correct_mask = antigen_labels == antibody.label
        true_positives = (mask & correct_mask).sum().item()
        false_positives = (mask & ~correct_mask).sum().item()
        false_negatives = (~mask & correct_mask).sum().item()

        shared_affinity = (1 / shared_counts[mask]).sum().item() if mask.any() else 0

        denom = true_positives + false_positives

        if correctness_type == "continuous":
            distances = 1 - (
                (torch.abs(antigen_labels - antibody.label) / (num_classes - 1))
                ** correctness_exponent
            )
            correctness_count = distances[mask].sum().item()
            correctness = correctness_count / denom if denom > 0 else 0

        elif correctness_type == "binary":
            correctness = (true_positives) / denom if denom > 0 else 0
        else:
            raise ValueError(f"Unknown correctness type: {correctness_type}")

        if correctness < 0.15:
            correctness = 0.15

        coverage = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        uniqueness = shared_affinity / denom if denom > 0 else 0
        avg_radii_approximation = 1 - (
            (abs(antibody.radii - (avg_distance - (avg_offset * avg_distance))))
            / avg_distance
        )
        fitness = (
            correctness_weight * correctness
            + coverage_weight * coverage
            + uniqueness_weight * uniqueness
            + avg_radii_approximation_weight * avg_radii_approximation
        )
        # print("Correctness:", correctness_weight * correctness, flush=True)
        # print("Coverage:", coverage_weight * coverage, flush=True)
        # print("Uniqueness:", uniqueness_weight * uniqueness, flush=True)
        fitness_scores[antibody.id] = fitness

    return fitness_scores


# Function that performs center mutation on an antibody
def center_mutation(
    antibody: Antibody,
    fitness_scaling: float,
    center_std: float,
    embedding_dim: int,
    center_mutation_intensity: float,
    center_mutation_dimensions: float,
) -> None:
    intensity = fitness_scaling * center_mutation_intensity * center_std
    movement = torch.zeros(embedding_dim, device="cuda")
    dim_mask = torch.rand(embedding_dim, device="cuda") < center_mutation_dimensions
    movement[dim_mask] = torch.normal(
        mean=0, std=intensity, size=(int(dim_mask.sum().item()),), device="cuda"
    )
    # print(f"Center mutation movement length: {torch.linalg.vector_norm(movement)}", flush=True)
    antibody.center_mutation(movement)


# Function that performs radii mutation on an antibody
def radii_mutation(
    antibody: Antibody,
    fitness_scaling: float,
    radii_mutation_intensity: float,
    min_radii_mutation: float,
    max_radii_mutation: float,
    center_std: float,
) -> None:
    intensity = fitness_scaling * radii_mutation_intensity * center_std
    change = (
        torch.normal(mean=1, std=intensity, size=(1,), device="cuda")
        .clamp(min=min_radii_mutation, max=max_radii_mutation)
        .item()
    )
    # print(f"Radii intensity: {intensity}", flush=True)
    # print(f"Radii mutation change: {change}", flush=True)
    antibody.radii_mutation(change)


# Function that performs multiplier mutation on an antibody
def multiplier_mutation(
    antibody: Antibody,
    fitness_scaling: float,
    multiplier_mutation_intensity: float,
    multiplier_mutation_dimensions: float,
    min_radii_mutation: float,
    max_radii_mutation: float,
) -> None:
    embedding_dim = antibody.multiplier.shape[0]
    intensity = fitness_scaling * multiplier_mutation_intensity
    change = torch.ones(embedding_dim, device="cuda")
    dim_mask = torch.rand(embedding_dim, device="cuda") < multiplier_mutation_dimensions
    change[dim_mask] = torch.normal(
        mean=1.0,
        std=float(intensity),
        size=(int(dim_mask.sum().item()),),
        device="cuda",
    ).clamp(min=min_radii_mutation, max=max_radii_mutation)
    # print(f"Multiplier change: {change}", flush=True)
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
    avg_radii_approximation_weight: float,
    avg_offset: float,
    avg_distance: float,
) -> List[Antibody]:
    print(
        f"Performing crowding with replacement ratio: {replacement_ratio}", flush=True
    )

    population = antibodies.copy()
    indexed_population = list(enumerate(population))
    num_replacements = int(len(population) * replacement_ratio)
    avg_fitness = (
        torch.tensor(list(fitness_scores.values()), device="cuda").mean().item()
    )

    for _ in range(num_replacements):
        rand_idx = torch.randint(0, len(indexed_population), (1,)).item()
        true_index, parent = indexed_population.pop(rand_idx)

        parent_fitness = fitness_scores[parent.id]
        scaling = avg_fitness / (parent_fitness + 1e-8)
        scaling = max(min(scaling, max_fitness_scaling), min_fitness_scaling)

        clones = [parent]
        for _ in range(num_clones):
            clone = Antibody(
                id=str(uuid.uuid4()),
                center=parent.center.clone(),
                radii=parent.radii,
                multiplier=parent.multiplier.clone(),
                label=parent.label,
                name=parent.name,
            )

            mutation_probs = torch.tensor(
                [center_mutation_rate, radii_mutation_rate, multiplier_mutation_rate],
                device="cuda",
                dtype=torch.float32,
            )
            mutation_probs /= mutation_probs.sum()
            mutation_type = torch.multinomial(mutation_probs, 1).item()

            if mutation_type == 0:
                center_mutation(
                    antibody=clone,
                    fitness_scaling=scaling,
                    center_std=center_std,
                    embedding_dim=embedding_dim,
                    center_mutation_intensity=center_mutation_intensity,
                    center_mutation_dimensions=center_mutation_dimensions,
                )
            elif mutation_type == 1:
                radii_mutation(
                    antibody=clone,
                    fitness_scaling=scaling,
                    radii_mutation_intensity=radii_mutation_intensity,
                    min_radii_mutation=min_radii_mutation,
                    max_radii_mutation=max_radii_mutation,
                    center_std=center_std,
                )
            elif mutation_type == 2:
                multiplier_mutation(
                    antibody=clone,
                    fitness_scaling=scaling,
                    multiplier_mutation_intensity=multiplier_mutation_intensity,
                    multiplier_mutation_dimensions=multiplier_mutation_dimensions,
                    min_radii_mutation=min_radii_mutation,
                    max_radii_mutation=max_radii_mutation,
                )

            clones.append(clone)

        clone_scores = fitness_of_antibodies(
            antibodies=clones,
            antigens=antigens,
            num_classes=num_classes,
            correctness_weight=correctness_weight,
            correctness_type=correctness_type,
            correctness_exponent=correctness_exponent,
            coverage_weight=coverage_weight,
            uniqueness_weight=uniqueness_weight,
            avg_radii_approximation_weight=avg_radii_approximation_weight,
            avg_offset=avg_offset,
            avg_distance=avg_distance,
        )
        best_id = max(clone_scores, key=clone_scores.get)
        best_clone = next(clone for clone in clones if clone.id == best_id)

        # print(
        #     f"Replacing antibody {parent.name} (label {parent.label}) → clone {best_clone.name} (label {best_clone.label})",
        #     flush=True,
        # )

        population[true_index] = best_clone

    print("Crowding finished", flush=True)
    return population


# def coverage(antibodies: List[Antibody], antigens: List[Antigen]) -> torch.Tensor:
#     coverage = torch.ones(len(antigens), device="cuda")
#     for antibody in antibodies:
#         for i in range(len(antigens)):
#             if antibody.is_recognized(antigens[i]):
#                 coverage[i] += 1
#     return coverage


def coverage(antibodies: List[Antibody], antigens: List[Antigen]) -> torch.Tensor:
    recognized = batch_is_recognized(antibodies, antigens)[0]
    return recognized.sum(dim=0).float() + 1


# def accuracy(antibody: Antibody, antigens: List[Antigen]) -> float:
#     positives = 0
#     true_positives = 0
#     for antigen in antigens:
#         if antibody.is_recognized(antigen):
#             positives += 1
#             if antibody.label == antigen.label:
#                 true_positives += 1
#     return true_positives / positives if positives > 0 else 0


def accuracies(antibodies: List[Antibody], antigens: List[Antigen]) -> List[float]:
    recognized = batch_is_recognized(antibodies, antigens)[0]
    antigen_labels = torch.tensor(
        [antigen.label for antigen in antigens], device=recognized.device
    )
    accuracies = []
    for i, antibody in enumerate(antibodies):
        mask = recognized[i]
        if mask.sum() == 0:
            accuracies.append(0.0)
        else:
            correct = (antigen_labels[mask] == antibody.label).sum().item()
            accuracies.append(correct / mask.sum().item())
    return accuracies


# Main function to run the training process
def train(config) -> List[Antibody]:

    num_classes = get_num_classes(config["DATASET"])
    if not config["WHITENING"]:
        whitening_str = "nw"
    else:
        whitening_str = str(config["DIMENSIONALITY_REDUCTION"])

    input_file_path = (
        "/cluster/work/axelle/Datasets-embedded/"
        + config["DATASET"]
        + "/"
        + whitening_str
        + "/"
        + "train"
        + ".pt"
    )
    # Load training data
    print(f"📥 Loading training data from: {input_file_path}")
    antigens, embedding_dim = load_training_data(
        input_file_path, config["MAX_DATASET_SIZE"]
    )

    print(
        f"✅ Loaded {len(antigens)} antigens with embedding dimension {embedding_dim}"
    )

    # Calculate the mean and standard deviation of the antigens
    mean = center_mean(antigens)
    std = center_std(antigens)

    # Initialise population of antibodies
    print(f"⚙️ Initialising population using method: {config['INITIALISATION_METHOD']}")
    population = initialise_population(
        antigens=antigens,
        population_size=int(round(config["POPULATION_SIZE"] * len(antigens))),
        initialisation_method=config["INITIALISATION_METHOD"],
        coverage=torch.ones(len(antigens), device="cuda"),
        mean=mean,
        std=std,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
    )
    print("center std", std)
    distances = batch_is_recognized(population, antigens)[1]
    avg_distance = distances.mean().item()
    print(f"📏 Average distance between antibodies and antigens: {avg_distance:.4f}")

    print(f"👾 Initial population size: {len(population)}")

    total_leaking_amount = int(
        round(len(antigens) * config["POPULATION_SIZE"] * config["TOTAL_LEAKING"])
    )

    leaking_per_generation = int(
        round(
            total_leaking_amount
            / ((config["NUM_GENERATIONS"] / config["LEAKING_FREQUENCY"]))
        )
    )

    for generation in range(config["NUM_GENERATIONS"]):
        print(f"🔄 Generation {generation + 1}/{config['NUM_GENERATIONS']}", flush=True)
        replacement_ratio = (
            config["REPLACEMENT_RATIO_MAX"]
            - (config["REPLACEMENT_RATIO_MAX"] - config["REPLACEMENT_RATIO_MIN"])
            * generation
            / config["NUM_GENERATIONS"]
        )
        fitness_scores = fitness_of_antibodies(
            antibodies=population,
            antigens=antigens,
            num_classes=num_classes,
            correctness_weight=config["CORRECTNESS_WEIGHT"],
            correctness_type=config["CORRECTNESS_TYPE"],
            correctness_exponent=config["CORRECTNESS_EXPONENT"],
            coverage_weight=config["COVERAGE_WEIGHT"],
            uniqueness_weight=config["UNIQUENESS_WEIGHT"],
            avg_radii_approximation_weight=config["AVERAGE_RADII_APPROXIMATION_WEIGHT"],
            avg_offset=config["AVERAGE_OFFSET"],
            avg_distance=avg_distance,
        )
        population = crowding(
            antibodies=population,
            replacement_ratio=replacement_ratio,
            fitness_scores=fitness_scores,
            center_std=std,
            antigens=antigens,
            embedding_dim=embedding_dim,
            center_mutation_intensity=config["CENTER_MUTATION_INTENSITY"],
            center_mutation_dimensions=config["CENTER_MUTATION_DIMENSIONS"],
            radii_mutation_intensity=config["RADII_MUTATION_INTENSITY"],
            min_radii_mutation=config["MIN_RADII_MUTATION"],
            max_radii_mutation=config["MAX_RADII_MUTATION"],
            multiplier_mutation_intensity=config["MULTIPLIER_MUTATION_INTENSITY"],
            multiplier_mutation_dimensions=config["MULTIPLIER_MUTATION_DIMENSIONS"],
            min_fitness_scaling=config["MIN_FITNESS_SCALING"],
            max_fitness_scaling=config["MAX_FITNESS_SCALING"],
            num_clones=config["NUM_CLONES"],
            center_mutation_rate=config["CENTER_MUTATION_RATE"],
            radii_mutation_rate=config["RADII_MUTATION_RATE"],
            multiplier_mutation_rate=config["MULTIPLIER_MUTATION_RATE"],
            correctness_type=config["CORRECTNESS_TYPE"],
            correctness_weight=config["CORRECTNESS_WEIGHT"],
            coverage_weight=config["COVERAGE_WEIGHT"],
            uniqueness_weight=config["UNIQUENESS_WEIGHT"],
            num_classes=num_classes,
            correctness_exponent=config["CORRECTNESS_EXPONENT"],
            avg_radii_approximation_weight=config["AVERAGE_RADII_APPROXIMATION_WEIGHT"],
            avg_offset=config["AVERAGE_OFFSET"],
            avg_distance=avg_distance,
        )

        # Leaking mechanism
        if (
            config["TOTAL_LEAKING"] != 0
            and generation % config["LEAKING_FREQUENCY"] == 0
        ):
            print(
                f"💧 Leaking {leaking_per_generation} antibodies into the population",
            )
            leaking_population = initialise_population(
                antigens=antigens,
                population_size=leaking_per_generation,
                initialisation_method=config["INITIALISATION_METHOD"],
                coverage=coverage(population, antigens),
                mean=mean,
                std=std,
                embedding_dim=embedding_dim,
                num_classes=num_classes,
            )
            population.extend(leaking_population)
    print("calculating accuracies", flush=True)
    accuracies_list = accuracies(population, antigens)
    for antibody, accuracy in zip(population, accuracies_list):
        antibody.accuracy = accuracy

    print(f"🏁 Training complete", flush=True)
    return population
