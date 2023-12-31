# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from datasets.SequentialSignal import SequentialSignalDataset
from datasets.multimodal_features import SequentialMultiModalFeatures
from datasets.seq_mnist import SequentialMNIST
from datasets.seq_cifar10 import SequentialCIFAR10
from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.hhar_features import SequentialHHAR
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace

NAMES = {
    SequentialMNIST.NAME: SequentialMNIST,
    SequentialCIFAR10.NAME: SequentialCIFAR10,
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    SequentialHHAR.NAME: SequentialHHAR,
    SequentialHHAR.GYRO: SequentialHHAR,
    SequentialMultiModalFeatures.NAME: SequentialMultiModalFeatures,
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    if args.dataset in ["UCI", "SHL", "WISDM", "MotionSense"]:
        return SequentialSignalDataset(args)
    return NAMES[args.dataset](args)