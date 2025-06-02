NAME = "continuous"
EXPERIMENT = "accuracy"

# Varying hyperparameters
DATASET = "Liar"  # Options: "Liar", "PolitiFact", "Averitec"

CORRECTNESS_TYPE = "binary"  # Options: "continuous", "binary"

VOTING_METHOD = "binary"  # Options: "continuous", "binary"

INITIALISATION_METHOD = "antigen_based"  # Options: "random", "antigen_based"

POPULATION_SIZE = 0.1  # Initial population size as a fraction of the training data size

TOTAL_LEAKING = 0.2  # Total leaking ratio compared to the population size

FORCED_COVERAGE = False  # Whether to force coverage or not

WHITENING = True  # Whether to use whitening or not
DIMENSIONALITY_REDUCTION = 128  # Dimensionality reduction for the embeddings


# Permanent hyperparameters
NUM_GENERATIONS = 100
REPLACEMENT_RATIO_MAX = 0.6  # The replacement ratio in the first generation
REPLACEMENT_RATIO_MIN = 0.01  # The replacement ratio in the last generation
LEAKING_FREQUENCY = 10  # Frequency of leaking

ERROR_SCALING = (
    2  # Scaling factor for how much a false positive affects the correctness score
)
CORRECTNESS_EXPONENT = 0.5  # Exponent for the calculation of continuous correctness
CORRECTNESS_WEIGHT = 1  # Weight of the correctness part of the fitness score
COVERAGE_WEIGHT = 1  # Weight of the coverage part of the fitness score
UNIQUENESS_WEIGHT = 1  # Weight of the uniqueness part of the fitness score


MIN_FITNESS_SCALING = 0.5  # Minimum scaling factor for the fitness of an antibody
MAX_FITNESS_SCALING = 2  # Maximum scaling factor for the fitness of an antibody


NUM_CLONES = 10  # Number of clones to create for each antibody
CENTER_MUTATION_RATE = 1  # Probability of mutating the center of an antibody
RADII_MUTATION_RATE = 1  # Probability of mutating the radii of an antibody
MULTIPLIER_MUTATION_RATE = 1  # Probability of mutating the multiplier of an antibody
CENTER_MUTATION_INTENSITY = 0.1  # 0.005  # 0.02  # Intensity of the center mutations, relative to the mean norm of the population
CENTER_MUTATION_DIMENSIONS = (
    0.2  # Portion of dimensions to mutate for the center mutations
)
RADII_MUTATION_INTENSITY = 0.2  # 0.2  # 1  # Intensity of the radii mutations
MIN_RADII_MUTATION = (
    0.7  # Minimum scaling factor for the radii of an antibody, must be > 0
)
MAX_RADII_MUTATION = 1.4  # Maximum scaling factor for the radii of an antibody
MULTIPLIER_MUTATION_INTENSITY = (
    0.1  # 0.1  # 0.2  # Intensity of the multiplier mutations
)
MULTIPLIER_MUTATION_DIMENSIONS = (
    0.2  # Portion of dimensions to mutate for the multiplier mutations
)
