import runpy
import torch
import os
import time
import numpy as np
from typing import List
from train import train, get_num_classes
from infer import classify_antigens
from antigen import Antigen
from antibody import Antibody


print("🌱 Script started", flush=True)
print("🔍 CONFIG_PATH =", os.environ.get("CONFIG_PATH"), flush=True)
config = runpy.run_path(os.environ["CONFIG_PATH"])
print("✅ Config loaded", flush=True)


def load_test_data() -> List[Antigen]:
    print("Loading test data...", flush=True)
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
        + "test"
        + ".pt"
    )

    data = torch.load(input_file_path)

    ids = data["id"]
    embeddings = data["embedding"].to("cuda")
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


def write_stat_block(f, title: str, values: List[float]):
    t = torch.tensor(values, dtype=torch.float32)  # ✅ force float dtype
    f.write(f"{title}:\n")
    f.write(f"{'Mean':<20}{t.mean().item():.4f}\n")
    f.write(f"{'Max':<20}{t.max().item():.4f}\n")
    f.write(f"{'Min':<20}{t.min().item():.4f}\n")
    f.write(f"{'Std Dev':<20}{t.std(unbiased=False).item():.4f}\n\n")


def evaluate():
    antigens = load_test_data()
    accuracies = []
    avg_misses = []
    coverages = []
    time_consumptions = []
    all_pred_counts = []
    for i in range(3):
        start_time = time.time()
        print("Evaluating run " + str(i + 1) + "/20", flush=True)
        antibodies = train(config=config)

        print("Classifying antigens", flush=True)
        predictions = classify_antigens(
            voting_method=config["VOTING_METHOD"],
            antigens=antigens,
            antibodies=antibodies,
            forced_coverage=config["FORCED_COVERAGE"],
            num_classes=get_num_classes(config["DATASET"]),
        )
        from collections import Counter

        pred_counts = Counter(predictions.tolist())
        all_pred_counts.append(pred_counts)

        correct = 0
        misses = 0
        evaluated = 0
        unclassified = 0
        for j, antigen in enumerate(antigens):
            predicted = predictions[j].item()
            if predicted == -1:
                unclassified += 1
            elif predicted == antigen.label:
                correct += 1
                evaluated += 1
            else:
                misses += abs(predicted - antigen.label)
                evaluated += 1
        # for antigen in antigens:
        #     counter += 1
        #     print(
        #         "Evaluating antigen " + str(counter) + "/" + str(len(antigens)),
        #         flush=True,
        #     )
        #     predicted = classify_antigen(
        #         voting_method=config["VOTING_METHOD"],
        #         antigen=antigen,
        #         antibodies=antibodies,
        #         forced_coverage=config["FORCED_COVERAGE"],
        #     )
        #     if predicted == -1:
        #         unclassified += 1
        #     elif predicted == antigen.label:
        #         correct += 1
        #         evaluated += 1
        #     else:
        #         misses += abs(predicted - antigen.label)
        #         evaluated += 1

        time_consumptions.append(time.time() - start_time)
        accuracies.append(correct / evaluated if evaluated > 0 else 0)
        avg_misses.append(misses / evaluated if evaluated > 0 else 0)
        coverages.append(
            evaluated / (evaluated + unclassified)
            if evaluated + unclassified > 0
            else 0
        )

    # Write results to file
    output_dir = f"/cluster/work/axelle/Evaluation_results/{config['EXPERIMENT']}/{config['DATASET']}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{config['NAME']}.txt", "w") as f:
        f.write("=== Summary of 20 Evaluation Runs ===\n\n")

        write_stat_block(f, "Accuracy Statistics", accuracies)
        write_stat_block(f, "Average Misses Statistics", avg_misses)
        write_stat_block(f, "Coverage Statistics", coverages)
        write_stat_block(f, "Time Consumption Statistics", time_consumptions)

        f.write("=== Parameter values used for the evaluation ===\n\n")
        f.write(f"Dataset: {config['DATASET']}\n")
        f.write(f"Correctness Type: {config['CORRECTNESS_TYPE']}\n")
        f.write(f"Voting Method: {config['VOTING_METHOD']}\n")
        f.write(f"Initialisation Method: {config['INITIALISATION_METHOD']}\n")
        f.write(f"Population Size: {config['POPULATION_SIZE']}\n")
        f.write(f"Total Leaking: {config['TOTAL_LEAKING']}\n")
        f.write(f"Forced Coverage: {config['FORCED_COVERAGE']}\n")
        f.write(f"Whitening: {config['WHITENING']}\n")
        f.write(f"Dimensionality Reduction: {config['DIMENSIONALITY_REDUCTION']}\n")

        f.write("=== Per-Run Evaluation Results ===\n\n")

        f.write(
            f"{'Run':<5}{'Accuracy':>12}{'Avg Misses':>15}{'Coverage':>18}{'Time (s)':>14}\n"
        )
        f.write("-" * 65 + "\n")
        for i in range(len(accuracies)):
            f.write(
                f"{i+1:<5}{accuracies[i]:>12.4f}{avg_misses[i]:>15.4f}{coverages[i]:>18.4f}{time_consumptions[i]:>14.4f}\n"
            )
        f.write("\n=== Prediction Counts ===\n\n")
        for i, pred_counts in enumerate(all_pred_counts):
            f.write(f"Run {i + 1}:\n")
            for label, count in sorted(pred_counts.items()):
                f.write(f"Label {label}: {count}\n")
            f.write("\n")


evaluate()
