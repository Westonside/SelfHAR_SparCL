import torch
from avalanche.benchmarks.utils import AvalancheTensorDataset, make_tensor_classification_dataset
from avalanche.evaluation.metrics import loss_metrics, accuracy_metrics, timing_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from pytorchcv.model_provider import get_model as ptcv_get_model

from avalanche.benchmarks.generators import nc_benchmark

from models.in_out_model import InOut
from avalanche.training import ICaRL, GEM, GDumb, JointTraining
import torch.optim as optim
import torch.nn as nn
global_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_gdumb(model, optim, crit, mem_size, train_mb_size, epochs, eval_mb_size, plugins, evaluator,  **kwargs):
    return GDumb(model, optim, crit, mem_size=mem_size, train_epochs=epochs, train_mb_size=train_mb_size, eval_mb_size=eval_mb_size, device=global_device, plugins=plugins, evaluator=evaluator)


def setup_normal_train(model,optim,crit, epochs, train_batch_size=64,eval_batch=64, **kwargs):
    return JointTraining(model,optim,crit,train_mb_size=train_batch_size, train_epochs=epochs, eval_mb_size=eval_batch, device=global_device)

strategy_fns = {
    "gdumb": setup_gdumb,
    "icarl": None,
    "normal": setup_normal_train,
}

def get_scenario(train,test,n_tasks, n_classes):
    return nc_benchmark(
        train_dataset=train,
        test_dataset=test,
        n_experiences=n_tasks,
        shuffle=True,
        task_labels=False,
        fixed_class_order=list(range(n_classes))
    )

def get_strategy(config):
    strat = list(config.keys())[0]
    return strategy_fns[strat]


input_shape = (128,3)
def buffer_model_training(data, n_classes, configuration):
    print('testing')
    n_tasks = n_classes / 2
    model = ptcv_get_model("quartz")
    lr = 0.03
    mom = None
    wd = None
    optimizer = optim.SGD(model.parameters(), lr=lr,momentum=mom,weight_decay=wd)
    strategy = get_strategy(configuration)(**configuration)
    crit = nn.CrossEntropyLoss()

    train_data, train_labels = data['train_data']
    test_data, test_labels = data['testing_data']

    train_dataset = make_tensor_classification_dataset((train_data, train_labels))
    test_dataset = make_tensor_classification_dataset((test_data, test_labels))


    # avalanche.benchmarks.make_tensor_classification_dataset
    scenario = nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_experiences=1,
        shuffle=True,
        task_labels=False
    )

    train_stream = scenario.train_stream
    test_stream = scenario.test_stream
    strat = get_strategy(configuration)(**configuration)
    results = []
    strat.train(train_stream)
    results.append(strat.eval(test_stream))
    print(results)
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


run_configurations = {
    "normal":{
        "mem_size": None,
        "epochs": 80,
        "batch_size": 64,
        "train_mb_size": None,
        "eval_mb_size": None,
        "n_tasks": 1,
    },
    "gdumb": {
        "mem_size": None,
        "epochs": 80,
        "batch_size": 64,
        "train_mb_size": None,
        "eval_mb_size": None,
        "n_tasks": 4,
    }
}


if __name__ == '__main__':
    datasets = {
        "SHL": None
    }
    for configuration in run_configurations:
        for dataset in datasets:
            buffer_model_training(None, None, configuration)



