import numpy as np
import preprocess
from avalanche.benchmarks.utils import make_classification_dataset, AvalancheTensorDataset
from torch.utils.data.dataset import TensorDataset
from avalanche.benchmarks.utils import AvalancheDataset
from preprocess.dataset_loading import load_datasets
import torch


def get_task_labels(labels, tasks):
    last_task = 0
    task_assignments = np.zeros(len(labels), dtype=np.int8)
    for i, task_value in enumerate(tasks):
        task_range = np.arange(last_task, task_value)
        task_assignments[np.where(np.isin(labels, task_range))] = i
        last_task = task_value
    return task_assignments


def create_avalanche_dataset(data_dir, data_files, classes_per_task=2, classes_in_first=2):
    loaded_data = load_datasets(data_files, data_dir)



    data_X = torch.from_numpy(loaded_data.train)
    data_X = data_X.float()
    labels = loaded_data.train_label
    labels = torch.from_numpy(np.argmax(labels, axis=1))

    data_test_X = torch.from_numpy(loaded_data.test)
    data_test_X = data_test_X.float()
    data_test_labels = torch.from_numpy(np.argmax(loaded_data.test_label, axis=1))


    tensor_ds = TensorDataset(data_X, labels)
    test_tensor_ds = TensorDataset(data_test_X, data_test_labels)


    num_classes = torch.unique(labels)
    tasks = [x for x in np.arange(classes_in_first, len(num_classes), classes_per_task)]
    tasks.append(len(num_classes)) # adding in the last tastasks
    train_task_assignments = get_task_labels(labels, tasks)
    test_task_assignments = get_task_labels(data_test_labels, tasks)

    task_ds = make_classification_dataset(tensor_ds, task_labels=train_task_assignments)
    test_task_ds = make_classification_dataset(test_tensor_ds, task_labels=test_task_assignments)
    return task_ds, test_task_ds, list(num_classes)







if __name__ == '__main__':
    create_avalanche_dataset("../../../SensorBasedTransformerTorch/datasets/processed", ["SHL"])
