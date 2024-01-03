import json
from preprocess.dataset_loading import load_datasets
import model_impl.HART
import numpy as np
import torch
from avalanche.evaluation.metrics import timing_metrics, forgetting_metrics, accuracy_metrics, loss_metrics, \
    class_accuracy_metrics, confusion_matrix_metrics

from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin, EWCPlugin, BiCPlugin, gss_greedy, GSS_greedyPlugin, \
    GenerativeReplayPlugin, RWalkPlugin
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.training.templates import SupervisedTemplate
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from avalanche.training import ICaRL, AGEM
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import sys

from baselines.datasets.sensor_dataset import create_avalanche_dataset

sys.path.append('..')
from models.super_special_model import HartClassificationModelSparCL

global_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
    This is the code used to run all baselines  
"""
def create_confuse(predictions, labels, cl_type):
    labels = labels.numpy()

    cm = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    display = ConfusionMatrixDisplay(cm)
    ax.set(title=f'Confusion Matrix for {cl_type}')
    display.plot(ax=ax)
    fig.savefig(f'confusion_matrix_{cl_type}.png', format='png', bbox_inches='tight')
    plt.close(fig)


input_shape = (128, 3)


def get_plugin(scenario):
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger()],
        benchmark=scenario,
        strict_checks=False
    )


plugin_matcher = {
    "ewc": EWCPlugin,
    "icarl": ICaRL,
    "agem": AGEM,
    "bic": BiCPlugin,
    "rwalk": RWalkPlugin
}


def match_plugin(plugin_type: str, model, optim, crit, **kwargs):
    if plugin_type == 'ewc' or plugin_type == 'bic':
        return plugin_matcher[plugin_type](**kwargs)
    return plugin_matcher[plugin_type](model=model, optimizer=optim, criterion=crit, **kwargs)


def match_model(model_type: str):
    if model_type == 'hart':
        return HartClassificationModelSparCL


def baseline_fn(data, cl_type: str, model: str, log_file: str, epochs=64, batch_size=64, **kwargs):
    train_set, test_set, classes, train_ds, test_ds = create_avalanche_dataset(data)
    text_logger = TextLogger(open(log_file, "a"))
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        loss_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        class_accuracy_metrics(
            epoch=True, stream=True, classes=classes
        ),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(num_classes=len(classes), stream=True),
        loggers=[text_logger, interactive_logger],
        collect_all=True
    )

    crit = nn.CrossEntropyLoss().to(global_device)
    network = match_model(model)(len(classes))
    optimizer = torch.optim.Adam(network.parameters(), 0.03)

    if cl_type != 'icarl':
        plugin = match_plugin(cl_type, model, optimizer, crit, **kwargs)
        strategy = SupervisedTemplate(
            network, optimizer, crit, plugins=[plugin], device=global_device, train_epochs=epochs,
            train_mb_size=batch_size,
            evaluator=eval_plugin,
        )
    else:
        strategy = ICaRL(nn.Identity(), network, optimizer, memory_size=kwargs["mem"], buffer_transform=nn.Identity(),
                         train_epochs=epochs,
                         train_mb_size=batch_size, eval_mb_size=batch_size, device=global_device, fixed_memory=True,
                         evaluator=eval_plugin)
    classes_per_task = {x: 2 for x in range(4)}
    fixed_class_order = classes
    my_benchmark = nc_benchmark(
        train_dataset=train_set,
        test_dataset=test_set,
        fixed_class_order=classes,
        # per_exp_classes=classes_per_task,
        n_experiences=4,
        task_labels=True,
        shuffle=True
    )

    network.to(global_device)
    try:
        print('*' * 50, "Starting the Training Loop", "*" * 50)
        results = []
        strategy.is_training = True
        for experience in my_benchmark.train_stream:
            print("*" * 20, f"starting experience for {cl_type}", "*" * 20)
            print("classes", experience.current_experience, "with", experience.classes_in_this_experience)
            res = strategy.train(experience)
            results.append(res)
            print("*" * 20, "Experience End", "*" * 20)
        #
        # print("training metrics: \n", results)
        test_stream = my_benchmark.test_stream
        #
        #
        cls2idx = {a: b for a, b in zip(test_stream.benchmark.classes_order, test_stream.benchmark.class_mapping)}
        y = torch.tensor(np.vectorize(cls2idx.get)(test_ds[1].numpy())).type(torch.int)
        network.eval()

        with torch.no_grad():
            test_data, test_labels = test_ds
            test_data = test_data.to(global_device)
            # test_labels = test_labels.to(global_device)
            predicted = []
            for i in range(0, test_data.shape[0], batch_size):
                outputs = network(test_data[i:i + batch_size])
                preds = torch.argmax(outputs, 1).cpu().numpy()
                predicted.append(preds)
        labels = y.cpu().numpy()
        predicted = np.hstack(predicted)
        network.train()  # old habit
        micro = f1_score(labels, predicted, average="micro", zero_division=0)
        macro = f1_score(labels, predicted, average="macro", zero_division=0)
        print(f'f1 report for ')
        print(f'f1_micro: {micro}')
        print(f'f1_macro: {macro}')
        print(classification_report(labels, predicted, zero_division=0))
        with open(f"f1_report_{cl_type}.txt", "w") as file:
            file.write(f"micro: {micro}, macro: {macro}")

        create_confuse(predicted, y, cl_type)
    except Exception as e:
        print('exception for model ', cl_type)
        print('Exception: ', e)


def load_config(config_file: str):
    with open(config_file) as f:
        configuration = json.load(f)

    return configuration

def run_baseline(dataset, configuration):
    configs = load_config(f'./baselines/configs/{configuration}')
    for config in configs['configs']:
        baseline_fn(dataset,**config)



if __name__ == '__main__':
    configuration = load_config('./configs/base_config.json')
    file_configuration = configuration["baseline_files"]
    dataset = load_datasets(file_configuration['files'], path=file_configuration['data_dir'])
    for config in configuration["configs"]:
        baseline_fn(dataset, **config)
