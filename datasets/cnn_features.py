from argparse import Namespace
from typing import Tuple
from preprocess import dataset_loading
import numpy as np
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
import hickle as hkl
from datasets.utils.continual_dataset import get_previous_train_loader, store_masked_loaders
import torchvision.transforms as transforms

import sklearn.model_selection
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader

from typing import Tuple
from torch.utils.data import Dataset, DataLoader

class MyCNNDataset(Dataset):
    def __init__(self, data, classes_to_idx: list):
        self.data, self.targets = data
        self.classes_to_idx = classes_to_idx

    def __getitem__(self, index: int) -> Tuple:
        point, target = torch.tensor(self.data[index]), torch.tensor(self.targets[index])

        if hasattr(self, 'logits'):
            return point, target, point, torch.tensor(self.logits[index])

        return point, target, point

    def __len__(self) -> int:
        return len(self.data)


class SequentialCNNDataset(ContinualDataset):
    NAME = "signal_dataset"
    N_CLASSES_PER_TASK = 2
    TOTAL_CLASSES = 8
    SETTING = 'class-il'
    TRANSFORM = None
    N_TASKS = 4
    def __init__(self, args: Namespace, train,test):
        #load the file
        # print(data)
        self.train_X, self.train_y = train
        # self.train_y = np.argmax(self.train_y, axis=1)
        self.test_X, self.test_y = test
        # self.test_y = np.argmax(self.test_y,axis=1)

        self.classes = []

        # if args.validation:
        #     validation_labels = np.argmax(data.validation_label, axis=1)
        #     self.validation = (data.validation, validation_labels)
        super().__init__(args)


    def get_data_loaders(self, return_dataset=False) -> Tuple[DataLoader, DataLoader]:
        train_dataset = MyCNNDataset((self.train_X, self.train_y), classes_to_idx=self.classes)
        # print(self.train_X.shape, self.train_y.shape)
        test_dataset = MyCNNDataset((self.test_X, self.test_y), classes_to_idx=self.classes)
        # print(self.test_X.shape , self.test_y.shape)
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        if not return_dataset:
            return train, test
        return train, test, train_dataset, test_dataset


    def not_aug_dataloader(self, batch_size: int) -> DataLoader:
        # i am not applying the transformation pipeline seen below:
        # transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])
        train_dataset = MyCNNDataset(self.train_X,idx_to_class=self.classes, classes=self.classes)
        train_loader = get_previous_train_loader(train_dataset, batch_size)
        return train_loader


    @staticmethod
    def get_transform() -> transforms:
        # return the identity function
        return transforms.Lambda(lambda x: x)

    @staticmethod
    def get_loss() -> nn.functional:
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform() -> transforms:
        return transforms.Lambda(lambda x: x)

    @staticmethod
    def get_denormalization_transform() -> transforms:
        return transforms.Lambda(lambda x: x)
