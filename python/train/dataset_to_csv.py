import json
import numpy as np
import pandas as pd
from pathlib import Path
import pathlib
import os

# Get the current directory
DATASETS_PATH=pathlib.Path(".").resolve()

def loadFile(filePath: Path, shuffle: bool = True):
    data_out = []
    accuracy_out = []
    with open(filePath) as f:
        data = json.load(f)
        for entry in data["data"]:
            data_out.append([entry["x"], entry["y"], entry["a"]])
            accuracy_out.append(entry["detection_accuracy"])

    if shuffle:
        index = np.arange(data_out.shape[0])
        np.random.shuffle(index)
        data_out = data_out[index]
        accuracy_out = accuracy_out[index]

    return np.array(data_out), np.array(accuracy_out)


def generateHandDataset(labels, dataset_path):
    data = {"label": [], "hand": [], "accuracy": []}
    for i in range(21):
        data.update({"x{}".format(i): [], "y{}".format(i): []})

    for label in labels:
        for hand in [0, 1]:
            fileName = label + ["_left", "_right"][hand] + "_hand.json"
            filePath = dataset_path / fileName

            if filePath.is_file():
                list_data, list_accuracy = loadFile(filePath, False)
                data["label"] += [label] * list_data.shape[0]
                data["hand"] += ["left" if hand == 0 else "right"] * list_data.shape[0]
                data["accuracy"] += list(list_accuracy)

                for i in range(21):
                    data["x{}".format(i)] += list(list_data[:, 0, i])
                    data["y{}".format(i)] += list(list_data[:, 1, i])

                print(fileName, "imported")
            else:
                print(fileName, "not found")
    return data

def run():
    handLabels = [
        'fist',
        'four',
        'ok',
        'palm',
        'tick',
        'v',
        'other',
    ]
    print(DATASETS_PATH / "Hand")
    datasetHands = pd.DataFrame(generateHandDataset(handLabels, DATASETS_PATH / "Hand"))
    datasetHands.to_csv(DATASETS_PATH / "HandPose7_Dataset.csv", index=False)

if __name__ == "__main__":
    run()
