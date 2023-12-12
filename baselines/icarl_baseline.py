import model_impl.HART
import torch
from avalanche.benchmarks import ni_benchmark
from avalanche.benchmarks.utils import AvalancheTensorDataset, make_tensor_classification_dataset
from avalanche.evaluation.metrics import loss_metrics, accuracy_metrics, timing_metrics, forgetting_metrics, \
    class_accuracy_metrics, confusion_matrix_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin, EWCPlugin
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.training.templates import SupervisedTemplate

from baselines.datasets.sensor_dataset import create_avalanche_dataset
from models.in_out_model import InOut
from avalanche.training import ICaRL, GEM, GDumb, JointTraining, Naive
import torch.optim as optim
import torch.nn as nn

from utils.configuration_util import load_config

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
# def buffer_model_training(data, n_classes, configuration):
#     print('testing')
#     n_tasks = n_classes / 2
#     model = ptcv_get_model("quartz")
#     lr = 0.03
#     mom = None
#     wd = None
#     optimizer = optim.SGD(model.parameters(), lr=lr,momentum=mom,weight_decay=wd)
#     strategy = get_strategy(configuration)(**configuration)
#     crit = nn.CrossEntropyLoss()
#
#     train_data, train_labels = data['train_data']
#     test_data, test_labels = data['testing_data']
#
#     train_dataset = make_tensor_classification_dataset((train_data, train_labels))
#     test_dataset = make_tensor_classification_dataset((test_data, test_labels))
#
#
#     # avalanche.benchmarks.make_tensor_classification_dataset
#     scenario = nc_benchmark(
#         train_dataset=train_dataset,
#         test_dataset=test_dataset,
#         n_experiences=1,
#         shuffle=True,
#         task_labels=False
#     )
#
#     train_stream = scenario.train_stream
#     test_stream = scenario.test_stream
#     strat = get_strategy(configuration)(**configuration)
#     results = []
#     strat.train(train_stream)
#     results.append(strat.eval(test_stream))
#     print(results)
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
    "ewc": EWCPlugin(ewc_lambda=0.001) #TODO: Grid search ewc lambda value
}


def match_plugin(plugin_type: str, model, optim, **kwargs):
    # first check if the type is a plugin or other todo later
    return plugin_matcher[plugin_type]


def match_model(model_type: str):
    if model_type == 'hart':
        return model_impl.HART.HartClassificationModel




def baseline_fn(cl_type: str, model: str, files_dir: str, files: str, log_file: str, epochs=64, batch_size=64, **kwargs):
    train_set, test_set,  classes = create_avalanche_dataset(files_dir, files)

    network = match_model(model)(len(classes))
    optimizer = torch.optim.Adam(network.parameters(),0.03)
    plugin = match_plugin(cl_type, network, optimizer, **kwargs)
    crit = nn.CrossEntropyLoss()



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
    strategy = SupervisedTemplate(
        network, optimizer, crit, plugins=[plugin], device=global_device,train_epochs=epochs,train_mb_size=batch_size,
        evaluator=eval_plugin,
    )
    classes_per_task = {x: 2 for x in range(4)}
    my_benchmark = nc_benchmark(
        train_dataset=train_set,
        test_dataset=test_set,
        per_exp_classes=classes_per_task,
        n_experiences=4,
        task_labels=False
    )

    print('*' * 50, "Starting the Training Loop", "*" * 50)
    results = []
    strategy.is_training = True
    for experience in my_benchmark.train_stream:
        print("*" * 20, "starting experience" , "*"*20)
        res = strategy.train(experience)
        results.append(res)
        print("*"*20, "Experience End", "*"*20)


if __name__ == '__main__':
    configuration = load_config('./configs/base_config.json')
    for config in configuration["configs"]:
        baseline_fn(**config)



