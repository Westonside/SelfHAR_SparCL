from typing import Sequence

import numpy as np
import preprocess
from avalanche.benchmarks.utils import AvalancheTensorDataset, AvalancheDataset, make_tensor_classification_dataset

import torch
from torch.utils.data import TensorDataset


def get_task_labels(labels, tasks):
    last_task = 0
    task_assignments = np.zeros(len(labels), dtype=np.int8)
    for i, task_value in enumerate(tasks):
        task_range = np.arange(last_task, task_value)
        task_assignments[np.where(np.isin(labels, task_range))] = i
        last_task = task_value
    return task_assignments


def create_avalanche_dataset(dataset, classes_per_task=2, classes_in_first=2):
    loaded_data = dataset

    train_data_x = torch.from_numpy(loaded_data.train)
    train_data_x = train_data_x.float()
    train_data_y = loaded_data.train_label
    if len(train_data_y.shape) > 1: # if it is one hot encoded
        train_data_y = torch.from_numpy(np.argmax(train_data_y, axis=1))
    else:
        train_data_y = torch.from_numpy(train_data_y)
    test_data_x = torch.from_numpy(loaded_data.test)
    test_data_x = test_data_x.float()
    if len(loaded_data.test_label.shape) > 1:
        test_data_y = torch.from_numpy(np.argmax(loaded_data.test_label, axis=1))
    else:
        test_data_y = torch.from_numpy(loaded_data.test_label)

    num_classes = torch.unique(train_data_y).tolist()
    tasks = [x for x in np.arange(classes_in_first, len(num_classes), classes_per_task)]
    tasks.append(len(num_classes))  # adding in the last tastasks
    train_task_assignments = get_task_labels(train_data_y, tasks)
    test_task_assignments = get_task_labels(test_data_y, tasks)

    train_dataset = make_tensor_classification_dataset(train_data_x, train_data_y, task_labels=train_task_assignments)
    test_dataset = make_tensor_classification_dataset(test_data_x, test_data_y, task_labels=test_task_assignments)

    return train_dataset, test_dataset, list(num_classes), (train_data_x, train_data_y), (test_data_x, test_data_y)


if __name__ == '__main__':
    create_avalanche_dataset("../../../SensorBasedTransformerTorch/datasets/processed", ["SHL"])
