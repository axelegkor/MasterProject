import torch
import numpy as np
from antibody import Antibody
from antigen import Antigen
from typing import List, Tuple
from utils import batch_is_recognized


# def classify_antigen(
#     voting_method: str,
#     antigen: Antigen,
#     antibodies: List[Antibody],
#     forced_coverage: bool,
# ) -> int:
#     votes = {}
#     for antibody in antibodies:
#         if antibody.is_recognized(antigen):
#             votes[antibody.label] = votes.get(antibody.label, 0) + antibody.accuracy
#     if not votes:
#         if forced_coverage:
#             closest = (None, 0)
#             for antibody in antibodies:
#                 if closest[0] is None or antibody.scaled_distance(antigen) < closest[1]:
#                     closest = (antibody, antibody.scaled_distance(antigen))
#             return closest[0].label
#         else:
#             return -1
#     if voting_method == "binary":
#         return max(votes, key=votes.get)
#     elif voting_method == "continuous":
#         total_votes = sum(votes.values())
#         weighted_sum = sum(label * (votes[label] / total_votes) for label in votes)
#         return int(round(weighted_sum))
#     else:
#         raise ValueError(f"Unknown voting method: {voting_method}")


def classify_antigens(
    voting_method: str,
    antigens: List[Antigen],
    antibodies: List[Antibody],
    forced_coverage: bool,
    num_classes: int,
) -> torch.Tensor:
    # print("Antibody labels:", [antibody.label for antibody in antibodies])
    A = len(antibodies)
    N = len(antigens)
    recognized, distances, _ = batch_is_recognized(antibodies, antigens)
    # print(
    #     "Radiis:", batch_is_recognized(antibodies, antigens)[2]
    # )  # Radii of antibodies
    # for i in range(A):
    #     print(recognized[i][:10])
    #     print(distances[i][:10])
    # print(
    #     "Number of antibodies recognizing each antigen",
    #     batch_is_recognized(antibodies, antigens)[0].sum(dim=0),
    # )  # Number of antibodies recognizing each antigen

    accuracy_tensor = torch.tensor(
        [antibody.accuracy for antibody in antibodies], device="cuda"
    )
    labels = torch.tensor([antibody.label for antibody in antibodies], device="cuda")
    votes = torch.zeros((N, num_classes), device="cuda")

    for c in range(num_classes):
        label_mask = labels == c
        relevant_recognitions = recognized[label_mask]  # (A_c, N)
        relevant_accuracies = accuracy_tensor[label_mask]  # (A_c,)

        num_recognizers = relevant_recognitions.sum(dim=0)  # (N,)
        total_accuracy = (relevant_recognitions.T * relevant_accuracies).sum(
            dim=1
        )  # (N,)
        avg_accuracy = total_accuracy / (num_recognizers + 1e-8)

        # alpha = 0.5  # can experiment with 0.3–0.7
        # votes[:, c] = num_recognizers.float().pow(alpha) * avg_accuracy
        votes[:, c] = num_recognizers.float() * avg_accuracy

    no_votes = votes.sum(dim=1) == 0

    if voting_method == "binary":
        preds = votes.argmax(dim=1)
    elif voting_method == "continuous":
        class_indices = torch.arange(num_classes, device="cuda")
        weighted_sum = (votes * class_indices).sum(dim=1) / (votes.sum(dim=1) + 1e-8)
        preds = weighted_sum.round().long()
    else:
        raise ValueError(f"Unknown voting method: {voting_method}")

    # ✅ Print prediction summary
    print("🧠 First 5 predictions:", preds[:5])

    print("🔍 First 5 antigens:")
    for i in range(5):
        antigen = antigens[i]
        print(
            f"Antigen ID: {antigen.id}, Label: {antigen.label}, Embedding: {antigen.embedding[:5].cpu().numpy()}"
        )

    print("🔍 All antibodies:")
    for i, ab in enumerate(antibodies):
        print(
            f"Antibody Label: {ab.label}, Center: {ab.center[:5].cpu().numpy()}, Radii: {ab.radii}, Multiplier: {ab.multiplier[:5].cpu().numpy()}"
        )
    print("🧪 Inspecting votes for antigens:")
    for i in range(10):
        for j, ab in enumerate(antibodies):
            print(
                f"Antibody Name: {ab.name}, Antibody Label: {ab.label}, Recognizes antigen {i}: {recognized[j, i].item()}, Accuracy: {ab.accuracy:.2f}, Radii: {ab.radii:.2f}, Distance: {distances[j, i].item():.2f}"
            )
        print(
            "--------------------------------------------------------------------------"
        )
    from collections import Counter

    # if forced_coverage and no_votes.any():
    #     unclassified = torch.nonzero(no_votes, as_tuple=True)[0]
    #     for i in unclassified:
    #         antigen = antigens[i]
    #         min_distance = float("inf")
    #         closest_label = -1
    #         for antibody in antibodies:
    #             distance = antibody.scaled_distance(antigen)
    #             if distance < min_distance:
    #                 min_distance = distance
    #                 closest_label = antibody.label
    #         preds[i] = closest_label
    if forced_coverage and no_votes.any():
        unclassified = torch.nonzero(no_votes, as_tuple=True)[0]  # shape: (U,)

        # Get index of minimum distance antibody for each unclassified antigen
        closest_indices = distances[:, unclassified].argmin(dim=0)  # shape: (U,)

        # Convert antibody index to label
        closest_labels = torch.tensor(
            [antibodies[i].label for i in closest_indices.tolist()], device="cuda"
        )

        # Assign forced labels
        preds[unclassified] = closest_labels

    elif not forced_coverage and no_votes.any():
        preds[no_votes] = -1
    pred_counts = Counter(preds.tolist())
    print("📊 Prediction distribution:", pred_counts)
    return preds

    # for antibody in antibodies:
    #     if antibody.is_recognized(antigen):
    #         votes[antibody.label] = votes.get(antibody.label, 0) + antibody.accuracy
    # if not votes:
    #     if forced_coverage:
    #         closest = (None, 0)
    #         for antibody in antibodies:
    #             if closest[0] is None or antibody.scaled_distance(antigen) < closest[1]:
    #                 closest = (antibody, antibody.scaled_distance(antigen))
    #         return closest[0].label
    #     else:
    #         return -1
    # if voting_method == "binary":
    #     return max(votes, key=votes.get)
    # elif voting_method == "continuous":
    #     total_votes = sum(votes.values())
    #     weighted_sum = sum(label * (votes[label] / total_votes) for label in votes)
    #     return int(round(weighted_sum))
    # else:
    #     raise ValueError(f"Unknown voting method: {voting_method}")
