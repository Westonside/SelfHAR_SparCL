import torch
from avalanche.evaluation.metrics import loss_metrics, accuracy_metrics, timing_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


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

def get_strategy(config):
    strat = list(config.keys())[0]
    return strategy_fns[strat]


input_shape = (128,3)
def buffer_model_training(data, n_classes, configuration):
    print('testing')
    n_tasks = n_classes / 2
    model = InOut(128, n_classes)
    lr = 0.03
    mom = None
    wd = None
    optimizer = optim.SGD(model.parameters(), lr=lr,momentum=mom,weight_decay=wd)
    strategy = get_strategy(configuration)(**configuration)
    crit = nn.CrossEntropyLoss()


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
    "gdumb": {
        "mem_size": None,
        "epochs": 80,
        "batch_size": 64,
        "train_mb_size": None,
        "eval_mb_size": None,
    }
}


if __name__ == '__main__':
    datasets = {
        "SHL": None
    }
    for configuration in run_configurations:
        for dataset in datasets:
            buffer_model_training(None, None, configuration)



