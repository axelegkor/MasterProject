import pandas as pd
import json
import runpy
import os

config = runpy.run_path(os.environ["CONFIG_PATH"])

DATASET = config["DATASET"]
FILENAME = config["FILENAME"]


if DATASET == "Liar":
    INPUT_FILE_TYPE = "tsv"
elif DATASET == "Politifact":
    INPUT_FILE_TYPE = "jsonl"
elif DATASET == "Averitec":
    INPUT_FILE_TYPE = "json"

input_path = (
    "/cluster/work/axelle/Datasets-original/"
    + DATASET
    + "/"
    + FILENAME
    + "."
    + INPUT_FILE_TYPE
)
output_path = (
    "/cluster/work/axelle/Datasets-preprocessed/" + DATASET + "/" + FILENAME + ".csv"
)


def preprocess_liar():
    df = pd.read_csv(input_path, sep="\t", header=None)
    df = df[[2, 1]]
    df.columns = ["text", "label"]

    df.insert(0, "id", range(len(df)))

    # Clean data
    df["text"] = df["text"].str.strip()
    df["label"] = df["label"].str.strip()

    # Map labels to integers
    label_map = {
        "pants-fire": 0,
        "false": 1,
        "barely-true": 2,
        "half-true": 3,
        "mostly-true": 4,
        "true": 5,
    }

    df = df[df["text"].notna() & df["text"].str.strip().astype(bool)]
    df = df[df["label"].isin(label_map)]

    df["label"] = df["label"].map(label_map)

    df.to_csv(output_path, index=False)


def preprocess_politifact():
    with open(input_path, "r") as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)

    df = df[["statement", "verdict"]]
    df.columns = ["text", "label"]

    df.insert(0, "id", range(len(df)))

    # Clean data
    df["text"] = df["text"].str.strip().str.replace('\\"', "", regex=False)
    df["label"] = df["label"].str.strip()

    # Map labels to integers
    label_map = {
        "pants-fire": 0,
        "false": 1,
        "mostly-false": 2,
        "half-true": 3,
        "mostly-true": 4,
        "true": 5,
    }
    df = df[df["text"].notna() & df["text"].str.strip().astype(bool)]
    df = df[df["label"].isin(label_map)]
    df["label"] = df["label"].map(label_map)
    df.to_csv(output_path, index=False)


def preprocess_averitec():
    with open(input_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    df = df[["claim", "label"]]
    df.columns = ["text", "label"]

    df.insert(0, "id", range(len(df)))

    # Clean data
    df["text"] = df["text"].str.strip()
    df["label"] = df["label"].str.strip()

    # Map labels to integers
    label_map = {
        "Refuted": 0,
        "Conflicting Evidence/Cherrypicking": 1,
        "Supported": 2,
    }

    df = df[df["text"].notna() & df["text"].str.strip().astype(bool)]
    df = df[df["label"].isin(label_map)]

    df["label"] = df["label"].map(label_map)

    df.to_csv(output_path, index=False)


def main():
    if DATASET == "Liar":
        preprocess_liar()
    elif DATASET == "Politifact":
        preprocess_politifact()
    elif DATASET == "Averitec":
        preprocess_averitec()
    else:
        raise ValueError(f"Unknown dataset: {DATASET}")
    print(f"Preprocessing {DATASET} dataset...")
    print(f"Preprocessed dataset saved to {output_path}")


main()
