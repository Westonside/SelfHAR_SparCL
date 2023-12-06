from argparse import Namespace

import hickle
import numpy as np
import sklearn.model_selection
import torch
import torchvision.transforms as transforms
# from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn as nn
import pickle
from datasets.utils import base_path
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
from torch.utils.data import Dataset, DataLoader

"""
 you use batching because there is not enough storage to load the whole thing into memory
 a batch is a chunk of the dataset
 so if you had a batch size of 1 you will give your model one example at a time
 an example of the maximum batch size would you give your model all data entries you won't have the memory to load straight into memroy
 also model learns better in batches
 
 you will want to chunk your dataset [0,len(data), chunk_size] go through data in chunks
"""




"""
    Augmented data and non augmented are the same in each
    so set them to be the same exact thing 
    STORE THE SAME DATA TO THE BUFFER 
"""

class MyHHAR(Dataset):
    def __init__(
            self,
            packed_input: Tuple, # packed input will be a tuple of data to labels
            classes: list,
            idx_to_class: list,
            task_classes=None
    ):

        print('testing ')
        self.data = packed_input[0]
        self.targets = packed_input[1]
        self.classes = classes

        self.class_to_idx = idx_to_class

    def __getitem__(self, index: int) -> Tuple:
        point, target = torch.tensor(self.data[index]), torch.tensor(self.targets[index])

        if hasattr(self,'logits'):
            return point, target, point, torch.tensor(self.logits[index])

        return point, target,point

    def __len__(self) -> int:
       return len(self.data)


def shuffle_data(data, labels):
    shuffle_indicies  = np.random.permutation(len(data))
    shuffle_data = data[shuffle_indicies]
    shuffle_labels = labels[shuffle_indicies]
    return shuffle_data, shuffle_labels


class SequentialHHAR(ContinualDataset):
    NAME = 'hhar_features'
    GYRO = 'hhar_features_gyro'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 4
    TOTAL_CLASSES = 8
    TRANSFORM = None

    def __init__(self, args: Namespace):
        # load the data
        data = hickle.load(args.modal_file)
        print('testing')
        self.train = data['train_features'], data['train_labels']
        self.test = data['test_features'], data['test_labels']
        if args.validation:
            train_X, train_y = self.train
            train_data, val_data, train_labels, val_labels = train_test_split(train_X, train_y, test_size=0.1,
                             random_state=42)

            self.train = train_data, train_labels
            self.validation = val_data, val_labels
        self.classes = []
        self.label_map = []
        # try:
        #     try:
        #         file = args.modal_file
        #     except:
        #         file = 'HHAR/hhar_features.pkl'
        #
        #     with open(base_path() + file, 'rb') as f:
        #         self.t = 0
        #         data = pickle.load(f)
        #         # shuffle the data
        #         if args.shuffle:
        #             shuf_data, shuf_labels = shuffle_data(data['features'], data['labels'])
        #             data['features'] = shuf_data
        #             data['labels'] = shuf_labels
        #         self.data = data['features']
        #         self.labels = data['labels']
        #         self.label_map = data['label_map']
        #         if args.validation:
        #             X_train, X_temp, y_train, y_temp = sklearn.model_selection.train_test_split(self.data, self.labels, test_size=0.2)
        #             X_test, X_val, y_test, y_val = sklearn.model_selection.train_test_split(X_temp, y_temp, test_size=0.5)
        #             self.train = (X_train, y_train)
        #             self.validation = (X_val, y_val)
        #             self.test = (X_test, y_test)
        #         else:
        #             X_train, X_test, y_train, y_test =  sklearn.model_selection.train_test_split(self.data, self.labels, train_size=0.8)
        #             self.train = (X_train, y_train)
        #             self.test = (X_test, y_test)
        #         self.classes = [x for x in self.label_map]
        # except Exception:
        #     print("Invalid data path!")
        #     exit(1)
        super().__init__(args)

    def get_data_loaders(self, return_dataset=False) -> Tuple[DataLoader, DataLoader]:
        # i want to open the file and then split the data
        # train_dataset = MyHHAR(data)

        train_dataset = MyHHAR(self.train, classes= self.classes, idx_to_class= self.label_map)
        test_dataset = MyHHAR(self.test, classes= self.classes, idx_to_class=self.label_map)
        train, test = store_masked_loaders(train_dataset,test_dataset,self)
        if not return_dataset:
            return train,test
        return train, test,train_dataset, test_dataset

    def not_aug_dataloader(self, batch_size: int) -> DataLoader:
        # i am not applying the transformation pipeline seen below:
        # transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyHHAR(self.train,idx_to_class=self.label_map, classes=self.label_map.keys())
        train_loader = get_previous_train_loader(train_dataset, batch_size)
        return train_loader

    # @staticmethod
    # def get_backbone() -> nn.Module:
    #     pass

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
