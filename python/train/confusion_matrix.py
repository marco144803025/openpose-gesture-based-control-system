from tensorflow import keras
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os
import tensorflow
from tensorflow import keras
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from keras.models import load_model

def handDataset( 
    testSplit: float = 0.1,
    shuffle: bool = True,
    handID: int = 0,
    local_import: bool = False,
):
    """Return the dataset of hand keypoints (see pose_classification_kit/datasets/HandPose_Dataset.csv)
    as numpy arrays.

    Args:
        testSplit (float, optional): Percent of the dataset reserved for testing. Defaults to 0.15. Must be between 0.0 and 1.0.
        shuffle (bool, optional): Shuffle the whole dataset. Defaults to True.
        handID (int, optional): Select hand side - 0:left, 1:right. Default to 0.
        local_import (bool, optional): Choose to use local dataset or fetch online dataset (global repository). Default False.

    Returns:
        dict: {
        'x_train': training keypoints,
        'y_train': training labels,
        'y_train_onehot': training labels one-hot encoded,
        'x_test': testing keypoints,
        'y_test': testing labels,
        'y_test_onehot': testing labels one-hot encoded,
        'labels': list of labels
    }
    """
    assert 0.0 <= testSplit <= 1.0
    
    datasetPath = pathlib.Path(".").resolve() /"train"/ "HandPose7_Dataset.csv"
    print(datasetPath)
    dataset_df = pd.read_csv(datasetPath)


    hand_label = "right" if handID else "left"
    handLabels_df = {
        hand_i: dataset_df.loc[dataset_df["hand"] == hand_i].groupby("label")
        for hand_i in ["left", "right"]
    }
    labels = list(dataset_df.label.unique())
    print(labels)
    
    # Find the minimum number of samples accross categories to uniformly distributed sample sets
    total_size_cat = handLabels_df[hand_label].size().min()
    test_size_cat = int(total_size_cat * testSplit)
    train_size_cat = total_size_cat - test_size_cat

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    # Iterate over each labeled group
    for label, group in handLabels_df[hand_label]:
        # remove irrelevant columns
        group_array = group.drop(["label", "hand", "accuracy"], axis=1).to_numpy()
        np.random.shuffle(group_array)

        x_train.append(group_array[:train_size_cat])
        y_train.append([label] * train_size_cat)
        x_test.append(group_array[train_size_cat : train_size_cat + test_size_cat])
        y_test.append([label] * test_size_cat)

    # Concatenate sample sets as numpy arrays
    x_train = np.concatenate(x_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # Shuffle in unison
    if shuffle:
        shuffler_test = np.random.permutation(test_size_cat * len(labels))
        shuffler_train = np.random.permutation(train_size_cat * len(labels))
        x_train = x_train[shuffler_train]
        x_test = x_test[shuffler_test]
        y_train = y_train[shuffler_train]
        y_test = y_test[shuffler_test]

    # One-hot encoding
    y_train_onehot = get_one_hot(
        np.array([labels.index(sample) for sample in y_train]), len(labels)
    )
    y_test_onehot = get_one_hot(
        np.array([labels.index(sample) for sample in y_test]), len(labels)
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "y_train_onehot": y_train_onehot,
        "x_test": x_test,
        "y_test": y_test,
        "y_test_onehot": y_test_onehot,
        "labels": np.array(labels),
    }    
    
def get_one_hot(targets: np.ndarray, nb_classes: int):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])

dataset = handDataset(testSplit=.2, shuffle=True, handID=0)
x_train = dataset['x_train']
y_train = dataset['y_train_onehot']
labels = dataset['labels']

# Load the saved model
model_path = pathlib.Path(".").resolve() /"train/model/aug_7_left.h5"
with tf.device('/cpu:0'):
    model = load_model(model_path)
model = keras.models.load_model(model_path)
start_time = time.time()
model.evaluate(x=dataset['x_test'], y=dataset['y_test_onehot'])
print(dataset['x_test'].shape)

labels_predict = model.predict(dataset['x_test'])
end_time = time.time()
total_time=(end_time-start_time)*1000
inference_time_per_sample =total_time/dataset['x_test'].shape[0]
print('Total inference time is:',total_time)
print("number of frame is:", dataset['x_test'].shape[0])
print('Average inference time is:',inference_time_per_sample)
confusion_matrix = np.array(
    tensorflow.math.confusion_matrix(
        np.argmax(labels_predict,axis=1),
        np.argmax(dataset['y_test_onehot'],axis=1)
))

fig, ax = plt.subplots(figsize=(10,10), dpi=100)

ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
tick_marks = np.arange(len(dataset['labels']))
ax.set_xticks(tick_marks)
ax.set_xticklabels(dataset['labels'], rotation=40, ha='right')
ax.set_yticks(tick_marks)
ax.set_yticklabels(dataset['labels'])

thresh = np.max(confusion_matrix) / 2.
for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
    plt.text(j, i, confusion_matrix[i, j],
        horizontalalignment="center",
        color="white" if confusion_matrix[i, j] > thresh else "black")

fig.tight_layout()
ax.set_ylabel('True label', fontsize=12)
ax.set_xlabel('Predicted label', fontsize=12)
plt.show()