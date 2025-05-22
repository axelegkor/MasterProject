import config
import torch
import os
import time
import numpy as np
from typing import List
from train import train
from infer import classify_antigen
from classes.antigen import Antigen
from classes.antibody import Antibody


def load_test_data() -> List[Antigen]:
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
        + "test"
        + ".pt"
    )

    data = torch.load(input_file_path)

    ids = data["id"]
    embeddings = data["embedding"]
    labels = data["label"]

    antigens = []
    for i in range(len(ids)):
        antigen = Antigen(
            id=ids[i],
            embedding=embeddings[i],
            label=labels[i],
        )
        antigens.append(antigen)
    return antigens


def evaluate():
    
    antigens = load_test_data()
    accuracies = []
    avg_misses = []
    coverages = []
    time_consumptions = []
    for i in range(20):
        
        print("Evaluating run", i + 1)
        antibodies, mu, w = train(config=config)
        start_time = time.time()
        evaluated = 0
        correct = 0
        misses = 0
        unclassified = 0

        for antigen in antigens:
            predicted = classify_antigen(
                voting_method=config.VOTING_METHOD,
                antigen=antigen,
                antibodies=antibodies,
                whitening=config.WHITENING,
                mu=mu,
                w=w,
                forced_coverage=config.FORCED_COVERAGE,
            )
            if predicted == -1:
                unclassified += 1
            elif predicted == antigen.label:
                correct += 1
                evaluated += 1
            else:
                misses += abs(predicted - antigen.label)
                evaluated += 1

        time_consumptions.append(time.time() - start_time)
        accuracies.append(correct / evaluated if evaluated > 0 else 0)
        avg_misses.append(misses / evaluated if evaluated > 0 else 0)
        coverages.append(
            evaluated / (evaluated + unclassified)
            if evaluated + unclassified > 0
            else 0
        )

    # Write results to file
    output_dir = (
        f"/cluster/work/axelle/Evaluation_results/{config.EXPERIMENT}/{config.DATASET}"
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{config.NAME}.txt", "w") as f:
        f.write("=== Summary of 20 Evaluation Runs ===\n\n")

        f.write("Accuracy Statistics:\n")
        f.write(f"{'Mean':<20}{np.mean(accuracies):.4f}\n")
        f.write(f"{'Max':<20}{np.max(accuracies):.4f}\n")
        f.write(f"{'Min':<20}{np.min(accuracies):.4f}\n")
        f.write(f"{'Std Dev':<20}{np.std(accuracies):.4f}\n\n")

        f.write("Average Misses Statistics:\n")
        f.write(f"{'Mean':<20}{np.mean(avg_misses):.4f}\n")
        f.write(f"{'Max':<20}{np.max(avg_misses):.4f}\n")
        f.write(f"{'Min':<20}{np.min(avg_misses):.4f}\n")
        f.write(f"{'Std Dev':<20}{np.std(avg_misses):.4f}\n\n")

        f.write("Coverage Statistics:\n")
        f.write(f"{'Mean':<20}{np.mean(coverages):.4f}\n")
        f.write(f"{'Max':<20}{np.max(coverages):.4f}\n")
        f.write(f"{'Min':<20}{np.min(coverages):.4f}\n")
        f.write(f"{'Std Dev':<20}{np.std(coverages):.4f}\n\n")

        f.write("Time Consumption Statistics:\n")
        f.write(f"{'Mean':<20}{np.mean(time_consumptions):.4f}\n")
        f.write(f"{'Max':<20}{np.max(time_consumptions):.4f}\n")
        f.write(f"{'Min':<20}{np.min(time_consumptions):.4f}\n")
        f.write(f"{'Std Dev':<20}{np.std(time_consumptions):.4f}\n\n")

        f.write("=== Parameter values used for the evaluation ===\n\n")
        f.write(f"Dataset: {config.DATASET}\n")
        f.write(f"Number of Classes: {config.NUM_CLASSES}\n")
        f.write(f"Correctness Type: {config.CORRECTNESS_TYPE}\n")
        f.write(f"Voting Method: {config.VOTING_METHOD}\n")
        f.write(f"Initialisation Method: {config.INITIALISATION_METHOD}\n")
        f.write(f"Population Size: {config.POPULATION_SIZE}\n")
        f.write(f"Total Leaking: {config.TOTAL_LEAKING}\n")
        f.write(f"Forced Coverage: {config.FORCED_COVERAGE}\n")
        f.write(f"Whitening: {config.WHITENING}\n")
        f.write(f"Dimensionality Reduction: {config.DIMENSIONALITY_REDUCTION}\n")

        f.write("=== Per-Run Evaluation Results ===\n\n")
        f.write(
            f"{'Run':<5}{'Accuracy':>12}{'Avg Misses':>15}{'Coverage':>18}{'Time (s)':>14}\n"
        )
        f.write("-" * 65 + "\n")
        for i in range(20):
            f.write(
                f"{i+1:<5}{accuracies[i]:>12.4f}{avg_misses[i]:>15.4f}{coverages[i]:>18.4f}{time_consumptions[i]:>14.4f}\n"
            )


evaluate()
