import json
import os
import numpy as np
from hickle import hickle
from preprocess.dataset_loading import load_datasets

from datasets import SequentialMultiModalFeatures, SequentialSignalDataset
from datasets.cnn_features import SequentialCNNDataset
from datasets.multimodal_features import create_multimodal_data
from models.in_out_model import InOut
from models.super_special_model import HartClassificationModelSparCL


def load_config(config_file: str):
    with open(config_file) as f:
        configuration = json.load(f)

    return configuration


def load_data_model(configuration, args, features=512, dropout=0.3):
    args.modal_file = configuration["files"]

    if configuration.get("epochs") is not None:
        args.epoch = configuration["epochs"]
    else:
        args.epoch = 60

    if configuration.get("batch_size") is not None:
        args.batch_size = configuration["batch_size"]
    else:
        args.batch_size = 64

    if configuration.get("use_cl_mask") is not None:
        args.use_cl_mask = configuration["use_cl_mask"]
    else:
        args.use_cl_mask = False

    if configuration.get("lr") is not None:
        args.lr = configuration["lr"]

    else:
        args.lr =  0.03

    if configuration.get("patience") is not None:
        args.patience = configuration["patience"]
    else:
        args.patience = 15

    if configuration.get("sparsity_profile") is not None:
        args.sp_config_file = configuration["sparsity_profile"]
    else:
        raise Exception(f"Please provide a sparsity profile for config {configuration}")

    if configuration.get("validation") is not None:
        args.validation = configuration["validation"]
    else:
        args.validation = False

    if configuration.get("loss") is not None:
        args.loss = configuration["loss"]
    else:
        args.loss = "ce"

    dataset = load_dataset(configuration["files"], configuration["type"], args)
    if configuration.get('balance_training') and configuration['balance_training']:
        balance_dataset(dataset)
    model = load_model(configuration["model"], configuration["input_shape"], dataset.TOTAL_CLASSES, features=features)

    return model, dataset


def balance_dataset(ds):
    un = np.unique(ds.train_y)
    smallest = min([np.count_nonzero(ds.train_y == x) for x in un])
    balanced = np.hstack([np.where(ds.train_y == x)[0][0:smallest] for x in un])
    ds.train_X = ds.train_X[balanced]
    ds.train_y = ds.train_y[balanced]
    print('dataset has been balanced')


def load_dataset(file, model_type, args):
    args.modal_file = file
    if 'features' in model_type:
        if model_type == 'cnn_features':
            path = file[:file.rfind('/')+1]
            ds_name = file[file.rfind('/')+1:]
            files = [x for x in os.listdir(path) if ds_name in x]
            accel = hickle.load(os.path.join(path,files[0]))
            gyro = hickle.load(os.path.join(path,files[1]))
            train = np.hstack([accel['train_features'], gyro['train_features']]), accel['train_labels']
            test =  np.hstack([accel['test_features'], gyro['test_features']]), accel['test_labels']
            data = SequentialCNNDataset(args, train,test)
            print('testing') # here is where you have to load the data and then combine the data

        elif model_type == "multi_modal_clustering_features":
            load = create_multimodal_data(args)
            data = SequentialMultiModalFeatures(args,*load)
            print('testing') # here you just load the data and will pass it to the dataset
        else:
            raise Exception("Invalid feature type provided! Options include cnn_features or multi_modal_clustering_features")
    elif model_type == 'baseline_hart':
        data = SequentialSignalDataset(args)

    return data


def load_model(model_type, input_shape, num_classes:int, features=128, dropout=0.2):
    model = models[model_type]
    if model_type == 'simple':
        model = model(input_shape, num_classes, features=features,dropout_rate=dropout)
    else:
        model = model(num_classes)
    return model



models = {
    "simple": InOut,
    "hart": HartClassificationModelSparCL
}