import pandas as pd
import config

DATASET = config.DATASET
FILENAME = config.FILENAME
INPUT_FILE_TYPE = config.INPUT_FILE_TYPE

input_path = "/cluster/work/axelle/Datasets-original/" + DATASET + "/" + FILENAME + "." + INPUT_FILE_TYPE
output_path = "/cluster/work/axelle/Datasets-preprocessed/" + DATASET + "/" + FILENAME + ".csv"

def preprocess_liar():
    df = pd.read_csv(input_path, sep="\t", header=None)
    df = df[[1, 2]]
    df.columns = ["label", "text"]
    
    df.insert(0, "id", range(len(df)))
    
    # Map labels to integers
    label_map = {
        "pants-fire": 0,
        "false": 1,
        "barely-true": 2,
        "half-true": 3,
        "mostly-true": 4,
        "true": 5,
    }
    df["label"] = df["label"].map(label_map)

    df.to_csv(output_path, index=False)


def main():
    if DATASET == "Liar":
        preprocess_liar()
        
main()