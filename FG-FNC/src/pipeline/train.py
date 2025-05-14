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
ERROR_SCALING = 2 # Scaling factor for how much a false positive affects the correctness score
CORRECTNESS_WEIGHT = 2.5 # Weight of the correctness part of the fitness score
COVERAGE_WEIGHT = 1.0 # Weight of the coverage part of the fitness score
UNIQUENESS_WEIGHT = 1.2 # Weight of the uniqueness part of the fitness score
NUM_CLONES = 10 # Number of clones to create for each antibody
CENTER_MUTATION_RATE = 1 # Probability of mutating the center of an antibody
RADII_MUTATION_RATE = 1 # Probability of mutating the radii of an antibody
MULTIPLIER_MUTATION_RATE = 1 # Probability of mutating the multiplier of an antibody
CENTER_MUTATION_INTENSITY = 0.02 # Intensity of the center mutations, relative to the mean norm of the population



# Function to load the training data into a list of Antigen objects, and return the average embedding
def load_training_data(file_path: str) -> Tuple[List[Antigen], float]:
    #TODO: Implement the function to load training data from a file
    
    return [], 0.0  # Placeholder return value


# Helper function to calculate the shortest distance between a point in an antigen with a different label
def max_radius(point: np.ndarray, antigens: List[Antigen], label: int) -> float:
    distances = [np.linalg.norm(point - antigen.embedding) for antigen in antigens if antigen.label != label]
    return max(distances)

# Function to randomly initialise a population of antibodies
def random_initialisation(antigens: List[Antigen], population_size: int) -> List[Antibody]:
    #TODO: Implement random initialisation of antibodies
    return []

def informed_initialisation(antigens: List[Antigen], population_size: int) -> List[Antibody]:
    antibodies = []
    coverage = np.ones(len(antigens))
    rng = np.random.default_rng()
    for _ in range(population_size):
        probabilities = 1 / coverage
        probabilities /= np.sum(probabilities)
        index = rng.choice(len(antigens), p=probabilities)
        antigen = antigens[index]
        antibody = Antibody(id=str(uuid.uuid4()), center=antigen.embedding, radii=max_radius(antigen.embedding, antigens, antigen.label), multiplier=np.ones(EMBEDDING_DIM), label=antigen.label)
        antibodies.append(antibody)
        for i in range(len(antigens)):
            if antibody.is_recognized(antigens[i]):
                coverage[i] += 1
    return antibodies
                
def initialise_population(antigens: List[Antigen], population_size: int, method: str) -> List[Antibody]:
    if method == "random":
        return random_initialisation(antigens, population_size)
    elif method == "informed":
        return informed_initialisation(antigens, population_size)
    else:
        raise ValueError(f"Unknown initialisation method: {method}")
    
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
def fitness_of_antibodies(antibodies: List[Antibody], antigens: List[Antigen]) -> Dict[str, float]:
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
        correctness = (true_positives - (false_positives * ERROR_SCALING)) / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        coverage = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        uniqueness = shared_affinity / true_positives if true_positives > 0 else 0
        fitness = (CORRECTNESS_WEIGHT * correctness) + (COVERAGE_WEIGHT * coverage) + (UNIQUENESS_WEIGHT * uniqueness)
        fitness_scores[antibody.id] = fitness
    return fitness_scores
        

# Function using crowding to clone and mutate antibodies
def crowding(antibodies: List[Antibody], replacement_ratio: float, fitness_scores: Dict[str, float], mean_norm: float) -> List[Antibody]:
    antibodies_eligible = antibodies.copy()
    updated_antibodies = antibodies.copy()
    rng = np.random.default_rng()
    num_replacements = int(len(antibodies) * replacement_ratio)
    for _ in range(num_replacements):
        index = rng.integers(0, len(antibodies_eligible))
        antibody = antibodies_eligible[index]
        antibodies_eligible.pop(index)
        clones = [antibody]
        for i in range (NUM_CLONES):
            clone = copy.deepcopy(antibody)
            probabilities = [CENTER_MUTATION_RATE, RADII_MUTATION_RATE, MULTIPLIER_MUTATION_RATE] / np.sum([CENTER_MUTATION_RATE, RADII_MUTATION_RATE, MULTIPLIER_MUTATION_RATE])
            mutation_type = rng.choice(["center", "radii", "multiplier"], p=probabilities)
            if mutation_type == "center":
        
        
        
# Main function to run the training process
def main():
    # Load training data
    antigens = load_training_data(DATA_PATH)
    
    # Initialise population of antibodies
    population = initialise_population(antigens, POPULATION_SIZE, INITIALISATION_METHOD)
    
    