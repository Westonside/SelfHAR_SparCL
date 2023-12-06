import json
import os
import argparse
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pickle
import copy

import utils.model_utils
from datasets.hhar_features import SequentialHHAR
import time

from datasets.multi_modal_clustering_dataset import MultiModalClusteringDataset
from models.in_out_model import InOut
# from models.resnet32_cifar10_grasp import resnet32
# from models.vgg_grasp import vgg19, vgg16
# from models.resnet20_cifar import resnet20
from models.resnet18_cifar import resnet18
from torch.optim.lr_scheduler import _LRScheduler

from testers import *

import numpy as np
import numpy.random as npr

import random
from prune_utils import *

# CL dataset and buffer library
from datasets import get_dataset, SequentialMultiModalFeatures
from utils import dynamic_architecture_util
from utils.buffer import Buffer
from utils.stats import calculate_f1_scores

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--arch', type=str, default=None,
                    help='[vgg, resnet, convnet, alexnet]')
parser.add_argument('--depth', default=None, type=int,
                    help='depth of the neural network, 16,19 for vgg; 18, 50 for resnet')
# parser.add_argument('--dataset', type=str, default="cifar10",
#                     help='[cifar10, cifar100]')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='for multi-gpu training')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--optmzr', type=str, default='adam', metavar='OPTMZR',
                    help='optimizer used (default: adam)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr-decay', type=int, default=60, metavar='LR_decay',
                    help='how many every epoch before lr drop (default: 30)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disabls CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--lr-scheduler', type=str, default='default',
                    help='define lr scheduler')
parser.add_argument('--warmup', action='store_true', default=False,
                    help='warm-up scheduler')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='M',
                    help='warmup-lr, smaller than original lr')
parser.add_argument('--warmup-epochs', type=int, default=0, metavar='M',
                    help='number of epochs for lr warmup')
parser.add_argument('--mixup', action='store_true', default=False,
                    help='ce mixup')
parser.add_argument('--alpha', type=float, default=0.3, metavar='M',
                    help='for mixup training, lambda = Beta(alpha, alpha) distribution. Set to 0.0 to disable')
parser.add_argument('--smooth', action='store_true', default=False,
                    help='lable smooth')
parser.add_argument('--smooth-eps', type=float, default=0.0, metavar='M',
                    help='smoothing rate [0.0, 1.0], set to 0.0 to disable')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--rho', type=float, default=0.0001,
                    help="Just for initialization")
parser.add_argument('--pretrain-epochs', type=int, default=0, metavar='M',
                    help='number of epochs for pretrain')
parser.add_argument('--pruning-epochs', type=int, default=0, metavar='M',
                    help='number of epochs for pruning')
parser.add_argument('--remark', type=str, default=None,
                    help='optimizer used (default: adam)')
parser.add_argument('--save-model', type=str, default='model/',
                    help='optimizer used (default: adam)')
parser.add_argument('--sparsity-type', type=str, default='random-pattern',
                    help="define sparsity_type: [irregular,column,filter,pattern]")
parser.add_argument('--config-file', type=str, default='config_vgg16',
                    help="config file name")

# ------- argments for CL setup ----------
parser.add_argument('--use_cl_mask', action='store_true', default=False, help='use CL mask or not')
parser.add_argument('--buffer-size', type=int, default=500, metavar='N',
                    help='buffer size for class incremental training (default: 100)')
parser.add_argument('--buffer_weight', type=float, default=1.0, help="weight of ce loss of buffered samples")
parser.add_argument('--buffer_weight_beta', type=float, default=1.0,
                    help="weight of ce loss of buffered samples in DERPP")
parser.add_argument('--dataset', type=str, default="seq-cifar10",
                    help='[seq-cifar10, seq-cifar100]')
parser.add_argument('--validation', action='store_true', default=False,
                    help='CL validation T of F')
parser.add_argument('--test_epoch_interval', type=int, default=1, metavar='how often we do test',
                    help='buffer size for class incremental training (default: 100)')
parser.add_argument('--evaluate_mode', action='store_true', default=False,
                    help='if we want to evaluate the checkpoints')
parser.add_argument("--eval_checkpoint", default=None, type=str, metavar="PATH",
                    help="path to evalute checkpoint (default: none)")
parser.add_argument('--gradient_efficient', action='store_true', default=False,
                    help='add gradient efficiency')
parser.add_argument('--gradient_efficient_mix', action='store_true', default=False,
                    help='add gradient efficiency (mix method)')
parser.add_argument('--gradient_remove', type=float, default=0.1, help="extra removal for gradient efficiency")
parser.add_argument('--gradient_sparse', type=float, default=0.75,
                    help="total gradient_sparse for training")
parser.add_argument('--sample_frequency', type=int, default=30, help="sample frequency for gradient mask")
parser.add_argument('--replay_method', type=str, default='er', help='replay method to use')

parser.add_argument('--patternNum', type=int, default=8, metavar='M',
                    help='number of epochs for lr warmup')
parser.add_argument('--rand-seed', action='store_true', default=False,
                    help='use random seed')
parser.add_argument("--log-filename", default=None, type=str, help='log filename, will override self naming')
parser.add_argument("--resume", default=None, type=str, metavar="PATH",
                    help="path to latest checkpoint (default: none)")
parser.add_argument('--save-mask-model', action='store_true', default=False,
                    help='save a sparse model indicating pruning mask')
parser.add_argument('--mask-sparsity', type=str, default=None, help='dir and file name for mask models')

parser.add_argument('--output-dir', required=True, help='directory where to save results')
parser.add_argument('--output-name', type=str, required=True)
parser.add_argument('--remove-data-epoch', type=int, default=200,
                    help='the epoch to remove partial training dataset')
parser.add_argument('--data-augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--remove-n', type=int, default=0,
                    help='number of sorted examples to remove from training')
parser.add_argument('--keep-lowest-n', type=int, default=0,
                    help='number of sorted examples to keep that have the lowest score, equivalent to start index of removal, if a negative number given, remove random draw of examples')
parser.add_argument('--sorting-file', type=str, default=None, help='input file name for sorted pkl file')
parser.add_argument('--input-dir', type=str, default=".", help='input dir for sorted pkl file')
parser.add_argument('--is-two-dim', type=bool, default=False, help='indicate if data is 2D or not')

prune_parse_arguments(parser)
# args = parser.parse_args()
total_epochs = 20
test_har = True
herding = False
dynamic = False
early_stop = True
args = argparse.Namespace(
    arch='simple' if test_har else 'resnet',
    arch_type = 'dynamic' if dynamic else 'static',
    patience=15,
    # arch='resnet',
    shuffle=False if test_har else False,
    # modal_file='HHAR/gyro_motion_hhar.pkl',
    modal_file='HHAR/accel_motion_hhar.pkl',
    buffer_mode='herding' if  herding else 'reservoir',
    depth=18,
    workers=4,
    multi_gpu=False,
    s=0.0001,
    batch_size=128,
    test_batch_size=256,
    epochs=500,
    optmzr='sgd',
    lr=0.03,
    lr_decay=60,
    momentum=0.9,
    weight_decay=0.0001,
    no_cuda=False,
    seed=888,
    lr_scheduler='cosine',
    warmup=False,
    warmup_lr=0.0001,
    warmup_epochs=0,
    mixup=False,
    alpha=0.3,
    smooth=False,
    smooth_eps=0.0,
    log_interval=10,
    rho=0.0001,
    pretrain_epochs=0,
    pruning_epochs=0,
    remark='irr_0.75_mut',
    save_model='checkpoints/resnet18/paper/gradient_effi/mutate_irr/seq-cifar10/buffer_500/',
    sparsity_type='random-pattern',
    config_file='config_vgg16',
    use_cl_mask=False,
    buffer_size=800,
    buffer_weight=0.1,
    buffer_weight_beta=0.5,
    # dataset='seq-cifar10',
    dataset='multi_modal_features',
    # dataset='hhar_features' if test_har else 'seq-cifar10',
    validation=True,
    test_epoch_interval=1,
    evaluate_mode=False,
    eval_checkpoint=None,
    gradient_efficient=False,
    gradient_efficient_mix=True,
    gradient_remove=0.1,
    gradient_sparse=0.8,
    sample_frequency=30,
    replay_method='derpp',
    patternNum=8,
    rand_seed=False,
    log_filename='checkpoints/resnet18/paper/gradient_effi/mutate_irr/seq-cifar10/buffer_500//seed_888_75_derpp_0.80.txt',
    resume=None,
    save_mask_model=False,
    mask_sparsity=None,
    output_dir='checkpoints/resnet18/paper/gradient_effi/mutate_irr/seq-cifar10/buffer_500/',
    output_name='irr_0.75_mut_RM_3000_20',
    remove_data_epoch=20,
    data_augmentation=False,
    remove_n=3000,
    keep_lowest_n=0,
    sorting_file=None,
    input_dir='.',
    sp_retrain=True,
    sp_config_file='./profiles/resnet18_cifar/irr/in_out_0.75.yaml' if test_har else './profiles/resnet18_cifar/irr/resnet18_0.75.yaml',
    sp_no_harden=False,
    sp_admm_sparsity_type='irregular',
    sp_load_frozen_weights=None,
    retrain_mask_pattern='weight',
    sp_update_init_method='zero',
    sp_mask_update_freq=5,
    sp_lmd=0.5,
    retrain_mask_sparsity=-1.0,
    retrain_mask_seed=None,
    sp_prune_before_retrain=True,
    output_compressed_format=False,
    sp_grad_update=False,
    sp_grad_decay=0.98,
    sp_grad_restore_threshold=-1,
    sp_global_magnitude=False,
    sp_pre_defined_mask_dir=None,
    upper_bound='0.74-0.75-0.75',
    lower_bound='0.75-0.76-0.75',
    mask_update_decay_epoch='5-45',
    cuda=True
)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

args.cuda = not args.no_cuda and torch.cuda.is_available()  # see if the cuda core is available

if args.rand_seed:  # if there is a random seed passed
    seed = random.randint(1, 999)  # generate one
    print("Using random seed:", seed)
else:
    seed = args.seed  # other wise use a manual seed
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)  # set the cuda manual seed
    print("Using manual seed:", seed)

if not os.path.exists(args.save_model):  # if there is not a save model path
    os.makedirs(args.save_model)  # create one

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
is_two_dim = False


class CrossEntropyLossMaybeSmooth(nn.CrossEntropyLoss):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    def __init__(self, smooth_eps=0.0):
        super(CrossEntropyLossMaybeSmooth, self).__init__()
        self.smooth_eps = smooth_eps
        # the smooth_eps represents the label smoothign factor determining how much smoothing is applied to one hot encoded labels
        # make the one hot encoding more smooth by reducing the one hot encoding labeles of: [0, 1, 0, 0] -> [0.05, 0.9, 0.05, 0.05]

    def forward(self, output, target, smooth=False):
        if not smooth:
            return F.cross_entropy(output, target)  # if there is no smoothing then just return the cross entropy loss

        target = target.contiguous().view(-1)
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smooth_eps) + (1 - one_hot) * self.smooth_eps / (n_class - 1)
        log_prb = F.log_softmax(output, dim=1)
        loss = -(smooth_one_hot * log_prb).sum(dim=1).mean()
        return loss


def mixup_data(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)  # draw random samples from a beta distribution
    else:
        lam = 1.0

    batch_size = x.size()[0]  # the size of the input
    index = torch.randperm(
        batch_size).cuda()  # create a random permutation of the batch this will create a random sequence of numbers 0-batchsuize-1 then moves the tensor to gpu memory
    # a tensor can have n dimensions so a vector is just a 1d tensor
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, smooth):
    return lam * criterion(pred, y_a, smooth=smooth) + \
        (1 - lam) * criterion(pred, y_b, smooth=smooth)


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_iter: target learning rate is reached at total_iter, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_iter, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_iter = total_iter
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_iter:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_iter + 1.) for base_lr in
                self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def train(model, trainset, criterion, scheduler, optimizer, epoch, t, buffer, dataset,
          # t is the task id in the dataset
          example_stats_train, train_indx, maskretrain, masks, cl_mask=None, task_dict=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_loss = 0.
    correct = 0.
    total = 0.
    # switch to train mode
    model.train()

    valid = {}

    # Get permutation to shuffle trainset
    trainset_permutation_inds = npr.permutation(
        np.arange(len(trainset.targets)))  # numpy random permutation
    batch_size = args.batch_size
    end = time.time()
    for batch_idx, batch_start_ind in enumerate(  # go through the training set using the bath size and batch index
            range(0, len(trainset.targets), batch_size)):
        data_time.update(time.time() - end)

        # prune_update_learning_rate(optimizer, epoch, args)

        # Get trainset indices for batch
        batch_inds = trainset_permutation_inds[
                     batch_start_ind:  # get the indicies of the batch that have been randomly permuted
                     batch_start_ind + batch_size]
        if len(batch_inds) < args.batch_size:
            continue
        # Get batch inputs and targets, transform them appropriately
        transformed_trainset = []
        not_transformed_trainset = []
        for ind in batch_inds:  # go through the batch indicies
            transformed_trainset.append(trainset.__getitem__(ind)[0])
            not_transformed_trainset.append(trainset.__getitem__(ind)[2])
        inputs = torch.stack(
            transformed_trainset)  # this stacks the transformed data in all batches you will have a list of 32 indicies that point to somehwhere in the data converts it to a tensor of 32x96
        not_transformed_inputs = torch.stack(
            not_transformed_trainset)  # this stacks the tranformed and non trasnformed samples per batch
        targets = torch.LongTensor(np.array(trainset.targets)[batch_inds].tolist())  # turn targets to a tensor

        # Map to available device
        inputs = inputs.cuda(non_blocking=True)  # move to gpu
        targets = targets.cuda(non_blocking=True)  # move to gpu
        if args.mixup:
            inputs, target_a, target_b, lam = mixup_data(inputs, targets, args.alpha)

        # Forward propagation, compute loss, get predictions
        # add buffer here
        # not giving the 96 features to the resnset
        # print(model)
        # for name, param in model.named_parameters():
        #     print(param.requires_grad, name)
        # exit(1)
        if (not buffer is None) and (
        not buffer.is_empty()) and t > 0:  # if the buffer is not empty or not the first task
            if args.replay_method == "er":  # if you have "er" replay not sure what that is (we use derpp)
                buf_inputs, buf_labels = buffer.get_data(
                    args.batch_size, transform=dataset.get_transform())
                if not args.merge_batch or (t == 0):
                    # compute output
                    outputs = model(inputs)  # get the outputs for the model
                    # add CL per task mask
                    if cl_mask is not None:
                        mask_add_on = torch.zeros_like(outputs)  # create a mask of zeros like the outputs tensor
                        mask_add_on[:, cl_mask] = float(
                            '-inf')  # change the mask up to cl_mask which may be a list of indicies or a single index to be negatve infinity
                        cl_masked_output = outputs + mask_add_on  # apply the mask to the outputs, setting certain elements values to negative infinity
                        ce_loss = criterion(cl_masked_output,
                                            targets)  # the loss is set to the critear applied on the masked outputs using the targets
                    else:  # if there is no masking
                        ce_loss = criterion(outputs, targets)  # if no mask then just claculate the cross entropy loss
                    # do an additional forward
                    # print("Buffer training!")
                    buf_output = model(buf_inputs)
                    buf_ce_loss = criterion(buf_output, buf_labels)
                    # ce_loss = ce_loss.mean()
                    # buf_ce_loss = buf_ce_loss.mean()
                    ce_loss += args.buffer_weight * buf_ce_loss
                else:
                    assert buffer is not None, "merge batch is not available when buffer is None!"
                    cat_inputs = torch.cat([inputs, buf_inputs], dim=0)  # combine the inputs
                    cat_targets = torch.cat([targets, buf_labels])  # combine the targets and the buffer labels
                    # compute output
                    cat_outputs = model(cat_inputs)  # compute output
                    # make sure only count non-buffer data
                    outputs = cat_outputs[:args.batch_size]
                    # add CL per task mask
                    if cl_mask is not None:  # this takes the outputs and then masks up to the batch size
                        mask_add_on = torch.zeros_like(cat_outputs)
                        # only add mask for the first half of batch
                        mask_add_on[:args.batch_size, cl_mask] = float('-inf')
                        cl_masked_output = cat_outputs + mask_add_on
                        ce_loss = criterion(cl_masked_output, cat_targets)
                    else:
                        ce_loss = criterion(cat_outputs, cat_targets)


            else:  # if using der or derpp
                # compute output
                outputs = model(
                    inputs)  # predict on  the passed in inputs will have probability distribution for prediction
                # add CL per task mask
                if cl_mask is not None:  # if you have a cl mask then mask the other classes not used in this prediction
                    mask_add_on = torch.zeros_like(outputs)  # make a tensor of 0s of len outputs
                    mask_add_on[:, cl_mask] = float(
                        '-inf')  # set the classes not used in the taks to be negative infinity
                    cl_masked_output = outputs + mask_add_on  # apply the mask
                    ce_loss = criterion(cl_masked_output,
                                        targets)  # calculate the loss for the classes in the current task
                else:
                    ce_loss = criterion(outputs, targets)

                # print(inputs.shape)

                if args.replay_method == "der":  # if you are using der
                    buf_inputs, buf_logits = buffer.get_data(
                        args.batch_size, transform=dataset.get_transform())
                    buf_output = model(buf_inputs)
                    buf_mse_loss = F.mse_loss(buf_output, buf_logits, reduction="none")
                    buf_mse_loss = torch.mean(buf_mse_loss, axis=-1)
                    # ce_loss = ce_loss.mean()
                    # buf_mse_loss = buf_mse_loss.mean()
                    ce_loss += args.buffer_weight * buf_mse_loss

                elif args.replay_method == "derpp":  # if you are using der++ this is our case you classify and get loss baesd off past predictions compared to current and then try to clasiify past tasks
                    buf_inputs, _, buf_logits = buffer.get_data(
                        # get the input data and the past logits from the buffer
                        args.batch_size,
                        transform=dataset.get_transform())  # you will try to get get your logits as close as possible to the logits in the buffer (regularizes) want to predict similarly to past predictions to preserve prediction
                    # you will need to transform the logits if you have a dynamic architecture

                    # if dynamic:
                    #     buf_logits = buf_logits.cpu()
                    #     imbalanced_logits = len(buf_logits[0]) != model.layer2.out_features
                    #
                    #     # Convert the list of tensors on the CPU
                    #     buf_logits = dynamic_architecture_util.extend_array(buf_logits, model.layer2.out_features).cuda()


                        # Move buf_logits back to the GPU

                    buf_output = model(buf_inputs)  # predict on the past inputs in the buffer
                    # print(buf_inputs.shape)
                    buf_mse_loss = F.mse_loss(buf_output, buf_logits,
                                              reduction="none")  # caculate the loss based on the difference between the two sets of logits
                    buf_mse_loss = torch.mean(buf_mse_loss, axis=-1)  # calculate the mean loss for the predictions
                    # print(ce_loss.shape, buf_mse_loss.shape)

                    # ce_loss = ce_loss.mean()
                    # buf_mse_loss = buf_mse_loss.mean()
                    ce_loss += args.buffer_weight * buf_mse_loss  # now multiply the loss by a value that we set that indicates how much we care about predicting close to past predictions in our case 0.1

                    buf_inputs, buf_labels, _ = buffer.get_data(
                        # now  you will get new buffer input values and the class labels
                        args.batch_size, transform=dataset.get_transform())
                    # print(buf_inputs.shape)
                    buf_output = model(buf_inputs)  # you will predict on this input
                    buf_ce_loss = criterion(buf_output,
                                            buf_labels)  # now you will calculate the loss by comparing the output to the class labels
                    # print(ce_loss.shape, buf_ce_loss.shape)
                    # exit(0)
                    # buf_ce_loss = buf_ce_loss.mean()
                    ce_loss += args.buffer_weight_beta * buf_ce_loss  # now you will caculate the loss using the beta value to specify how much you care about correct class predictions on past exmaples

        else:  # no replay
            # compute output
            outputs = model(inputs)  # get the ouput of the model
            # add CL per task mask
            if cl_mask is not None:  # this will mask the outputs for the classes not involved in the current tasks
                mask_add_on = torch.zeros_like(outputs)  # creates zero tensor of size output
                mask_add_on[:, cl_mask] = float('-inf')  # mask the predictions for the classes not in the task
                cl_masked_output = outputs + mask_add_on  # now perform the masking of the outputs for the other classes
                ce_loss = criterion(cl_masked_output, targets)  # calculate the loss
            else:
                ce_loss = criterion(outputs, targets)
        loss = ce_loss  # set the loss
        # your loss will be the loss per item in the batch

        # loss = criterion(outputs, targets)
        predicted = torch.argmax(outputs,1)

        # Update statistics and loss
        acc = predicted == targets  # get the predictions and see where the prediction was correct

        for j, index in enumerate(
                batch_inds):  # go through the batch using batch and batch index batch inds being the position in the data where for the batch permuatation
            # go through each batch and find the most incorrectly predicted in the batch
            # Get index in original dataset (not sorted by forgetting) YOU ARE GOING THROUGH SAMPLES
            index_in_original_dataset = train_indx[index]  # get the index in the non permuted dataset
            # INDEX WILL BE AN ARR containing indicies to samples in the data for the current batch
            # Compute missclassification margin
            output_correct_class = outputs.data[j, targets[j].item()]  # get all the correct classes for the batch
            sorted_output, _ = torch.sort(outputs.data[j, :])  # sort the outputs for a given batch
            if acc[j]:  # if correct prediction
                # Example classified correctly, highest incorrect class is 2nd largest output
                output_highest_incorrect_class = sorted_output[
                    -2]  # if the prediction was correct for this, it will be the most predicted so get the second most predicted
            else:  # incorrect prediction
                # Example misclassified, highest incorrect class is max output
                output_highest_incorrect_class = sorted_output[-1]
            margin = output_correct_class.item(  # calcualte the margin between the correct predictions and incorrect
            ) - output_highest_incorrect_class.item()

            # Add the statistics of the current training example to dictionary
            index_stats = example_stats_train.get(index_in_original_dataset,  # add the stats in the
                                                  [[], [], []])

            index_stats[0].append(loss[j].item())  # add the loss for the item to the first arr
            index_stats[1].append(acc[j].sum().item())  # add the accuracy summed to teh second
            index_stats[2].append(margin)  # add the tird to the final j
            example_stats_train[
                index_in_original_dataset] = index_stats  # then set the stats for this sample within the current batch to the index in the original dataset
        # Update loss, backward propagate, update optimizer
        # print('inside len(example_stats_train)',len(example_stats_train))

        # losses.update(loss.item(), inputs.size(0))

        loss = loss.mean()  # get the mean of the loss for all batches
        train_loss += loss.item()  # add the loss to the training loss
        total += targets.size(0)  # add batch size to the total this represents the total seen inputs
        correct += predicted.eq(
            targets.data).cpu().sum()  # add the total correct predictions for this batch to the counter
        loss.backward()  # backpropagate the loss
        if args.gradient_efficient:  # if grad efficient
            prune_apply_masks_on_grads_efficient()
        elif args.gradient_efficient_mix:  # if mixed grad efficient
            if batch_idx % args.sample_frequency == 0:  # if the batch idx is divisble by the sample frequency
                prune_apply_masks_on_grads_mix()  # this will mask gradients for masked weights then it will mask gradients below the 80th percentile and create a gradient mask that is used in apply gradients efficient to reduce computation
            else:
                prune_apply_masks_on_grads_efficient()  # this will apply the gradient mask made in the prune apply masks on grads mix
        else:
            prune_apply_masks_on_grads()
        optimizer.step()  # step the learning rate optimizer

        if batch_idx != (len(trainset) // batch_size) - 1:  # if the current batch is this value
            optimizer.zero_grad()  # set the optimizer to be zero grade
        prune_apply_masks()  # now prune apply the masks

        batch_time.update(time.time() - end)  # get the batch time
        end = time.time()  # get the end

        # Add training accuracy to dict
        index_stats = example_stats_train.get('train', [[], []])
        index_stats[1].append(100. * correct.item() / float(total))
        example_stats_train['train'] = index_stats

        # add data to buffer at the very end of the training iteration
        # Note that the datas are already transformed
        if args.replay_method == 'er':
            buffer.add_data(examples=not_transformed_inputs, labels=targets)
        elif args.replay_method == 'der':
            buffer.add_data(examples=not_transformed_inputs, logits=outputs.data)
        elif (args.replay_method == 'derpp'):
            # print('running derpp')
              # and args.buffer_mode != 'herding'):
            # if dynamic and  t > 0 and imbalanced_logits: # if the first epoch and not the first task
            #     #here you will have to clear the buffer and add in the larger logits values
            #     _in, _lab, logits = buffer.get_all_data()
            #     buffer.empty()
            #     logits  = logits.cpu()
            #     extended_buff = dynamic_architecture_util.extend_array(logits, model.layer2.out_features).cuda() # now that you have extended the
            #     buffer.add_data(examples=_in, labels=_lab, logits=extended_buff)
            buffer.add_data(examples=not_transformed_inputs, labels=targets, logits=outputs.data)

        if batch_idx % 10 == 0:  # if the batch index is divisble by 10
            for param_group in optimizer.param_groups:  # set the param group
                current_lr = param_group['lr']

            # at the end of each batch you can run your validation function

            print('Epoch: [{0}][{1}/{2}]\t'
                  'LR: {3:.5f}\t'
                  'Loss {4:.4f}\t'
                  'Acc@1 {5:.3f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            .format(
                epoch, batch_idx, (len(trainset) // batch_size) + 1,
                current_lr,
                loss.item(), 100. * correct.item() / total,
                batch_time=batch_time,
                data_time=data_time
            ))
    # at the end of epoch you will perform your validation


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mask_classes(outputs, dataset, k):
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


'''
    Impl Notes: In order to monitor forgetting I am conducting my validation on all tasks 
'''


def validation(model, dataset, epoch, task, task_dict):
    model.eval()  # turn on evaluate model

    if task not in task_dict:
        task_dict[task] = {}
    task_dict[task][epoch] = {"individual": []}  # in the dict at the task value is a value for the epoch
    with torch.no_grad():
        total_correct = 0
        total_loss = 0
        for t in range(task + 1):  # go up to that task
            cur_classes = np.arange(t * dataset.N_CLASSES_PER_TASK, (t + 1) * dataset.N_CLASSES_PER_TASK)
            # task_valid_loss = 0
            correct = 0
            # add the data if class belongs to the current class
            task_class_vals = np.where(
                np.isin(dataset.validation[1], cur_classes))  # get the indicies where they are in the task
            data, label = torch.tensor(dataset.validation[0][task_class_vals]).cuda(), torch.tensor(
                dataset.validation[1][task_class_vals]).cuda()
            output = model(data)
            criterion = nn.CrossEntropyLoss()

            task_valid_loss = criterion(output, label).item()
            total_loss += task_valid_loss

            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

            correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()
            total_correct += correct  # add the total correct to the total

            task_acc = (float(correct) / len(data)) * 100.
            assert (task_acc < 100.)
            cur_task_epoch_task = {
                "task": t,
                "task_validation_loss": task_valid_loss,
                "task_accuracy": task_acc,
                "correct": correct,
            }

            task_dict[task][epoch]["individual"].append(cur_task_epoch_task)

        total_acc = float(total_correct) / float(len(dataset.validation[1])) * 100.
        task_dict[task][epoch]["overall"] = {
            "total_validation_accuracy": total_acc,
            "total_loss": total_loss,
            "total_correct": total_correct,
        }
    print("=" * 110, f"Total validation loss: {total_loss}\n")

    model.train()  # set back to training mode
    return total_loss


def test(model, dataset):
    model.eval()
    acc_list = np.zeros((dataset.N_TASKS,))
    til_acc_list = np.zeros((dataset.N_TASKS,))
    confusion = np.zeros((dataset.N_CLASSES_PER_TASK, dataset.N_CLASSES_PER_TASK), dtype=np.uint64)
    with torch.no_grad():
        for task, test_loader in enumerate(dataset.test_loaders):
            predictions = {
                'predictions': [],
                'labels': [],
            }
            test_loss = 0
            correct = 0
            til_correct = 0
            for data in test_loader:
                if is_two_dim:
                    img, target, _ = data
                else:
                    img, target = data
                # print(f"\tTest classes"+str(np.unique(target)))
                if args.cuda:
                    img, target = img.cuda(), target.cuda()
                img, target = Variable(img, volatile=True), Variable(target)
                output = model(img)
                criterion = nn.CrossEntropyLoss()
                test_loss = criterion(output, target)
                # test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                predictions['predictions'].extend(pred.tolist())
                predictions['labels'].extend(target.tolist())


                # pred for task incremental
                mask_classes(output, dataset, task)
                til_pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                til_correct += til_pred.eq(target.data.view_as(til_pred)).cpu().sum()

            test_loss /= len(test_loader.dataset)
            acc = float(100. * correct) / float(len(test_loader.dataset))
            til_acc = float(100. * til_correct) / float(len(test_loader.dataset))
            acc_list[task] = acc
            til_acc_list[task] = til_acc
            print(
                    f"Task {task}, Average loss {test_loss:.4f}, Class inc Accuracy {acc:.3f}, Task inc Accuracy {til_acc:.3f}")
    micro_f1, macro_f1 = calculate_f1_scores(confusion)
    print('Micro F1 Score: ', micro_f1)
    print('Macro F1 Score: ', macro_f1)

    return acc_list, til_acc_list, micro_f1, macro_f1


def evaluate(model, dataset, last=False, test=False):
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    model.eval()

    accs = np.zeros((dataset.N_TASKS,))
    accs_mask_classes = np.zeros((dataset.N_TASKS,))
    confusion = np.zeros((dataset.TOTAL_CLASSES, dataset.TOTAL_CLASSES), dtype=np.uint64)

    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1: # the test loader is batching the test data into 158 batches of 128
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        predictions = {
            'predictions': [],
            'labels': [],
        }

        # print(test_loader.data)
        for data in test_loader: # the batched data in the loader
            # print('testing ', data)
            with torch.no_grad():
                if len(data) == 3:
                    inputs, labels, _ = data
                else:
                    inputs, labels = data #the data

                predictions['labels'].extend(labels.tolist())
                inputs, labels = inputs.cuda(), labels.cuda()

                outputs = model(inputs)  # predict on the input

                pred = torch.argmax(outputs, dim=1)
                predictions['predictions'].extend(pred.tolist())
                correct += torch.sum(pred == labels).item()  # get th ecorrect items


                total += labels.shape[0]  # increment the total inputs

                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()
        if total == 0.0:
            continue
        accs[k] = correct / total * 100
        accs_mask_classes[k] = correct_mask_classes / total * 100
        # here you will update the confusion matrix at that index
        for i, answer in enumerate(predictions['labels']):
            # go through the answer and mark if it was a false positive or not
            confusion[answer][predictions['predictions'][i]] += 1



    print("*" * 50, "Confusion Matrix", "*" * 50)
    print('\n\n', confusion, '\n')
    return accs, accs_mask_classes


def compute_forgetting_statistics(diag_stats, npresentations):
    presentations_needed_to_learn = {}
    unlearned_per_presentation = {}
    margins_per_presentation = {}
    first_learned = {}
    print('len(diag_stats.items())', len(diag_stats.items()))

    for example_id, example_stats in diag_stats.items():

        # Skip 'train' and 'test' keys of diag_stats
        if not isinstance(example_id, str):

            # Forgetting event is a transition in accuracy from 1 to 0
            presentation_acc = np.array(example_stats[1][:npresentations])
            transitions = presentation_acc[1:] - presentation_acc[:-1]

            # Find all presentations when forgetting occurs
            if len(np.where(transitions == -1)[0]) > 0:
                unlearned_per_presentation[example_id] = np.where(
                    transitions == -1)[0] + 2
            else:
                unlearned_per_presentation[example_id] = []

            # Find number of presentations needed to learn example, 
            # e.g. last presentation when acc is 0
            if len(np.where(presentation_acc == 0)[0]) > 0:
                presentations_needed_to_learn[example_id] = np.where(
                    presentation_acc == 0)[0][-1] + 1
            else:
                presentations_needed_to_learn[example_id] = 0

            # Find the misclassication margin for each presentation of the example
            margins_per_presentation = np.array(
                example_stats[2][:npresentations])

            # Find the presentation at which the example was first learned, 
            # e.g. first presentation when acc is 1
            if len(np.where(presentation_acc == 1)[0]) > 0:
                first_learned[example_id] = np.where(
                    presentation_acc == 1)[0][0]
            else:
                first_learned[example_id] = np.nan

    return presentations_needed_to_learn, unlearned_per_presentation, margins_per_presentation, first_learned


def sort_examples_by_forgetting(unlearned_per_presentation_all,
                                first_learned_all, npresentations):
    # Initialize lists
    example_original_order = []
    example_stats = []

    for example_id in unlearned_per_presentation_all[0].keys():

        # Add current example to lists
        example_original_order.append(example_id)
        example_stats.append(0)

        # Iterate over all training runs to calculate the total forgetting count for current example (presentation is continuous learning where you add in new ones)
        for i in range(len(unlearned_per_presentation_all)):

            # Get all presentations when current example was forgotten during current training run
            stats = unlearned_per_presentation_all[i][example_id]

            # If example was never learned during current training run, add max forgetting counts
            if np.isnan(first_learned_all[i][example_id]):
                example_stats[-1] += npresentations
            else:
                example_stats[-1] += len(stats)

    num_unforget = len(np.where(np.array(example_stats) == 0)[0])
    print('Number of unforgettable examples: {}'.format(
        len(np.where(np.array(example_stats) == 0)[0])))
    return np.array(example_original_order)[np.argsort(
        example_stats)], np.sort(example_stats), num_unforget


def check_filename(fname, args_list):
    # If no arguments are specified to filter by, pass filename
    if args_list is None:
        return 1

    for arg_ind in np.arange(0, len(args_list), 2):
        arg = args_list[arg_ind]
        arg_value = args_list[arg_ind + 1]

        # Check if filename matches the current arg and arg value
        if arg + '_' + arg_value + '__' not in fname:
            print('skipping file: ' + fname)
            return 0

    return 1


# Format time for printing purposes
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


def sparCL(args, dataset_name, data_location=None, run_num=None, epoch_info=None):
    print('starting with ', data_location)
    args.dataset=dataset_name
    if args.cuda:
        if args.arch == "vgg":
            if args.depth == 19:
                model = vgg19(dataset=args.dataset)
            elif args.depth == 16:
                model = vgg16(dataset=args.dataset)
            else:
                sys.exit("vgg doesn't have those depth!")
        elif args.arch == "resnet":
            if args.depth == 18:
                model = resnet18(dataset=args.dataset)
            elif args.depth == 20:
                model = resnet20(dataset=args.dataset)
            elif args.depth == 32:
                model = resnet32(depth=32, dataset=args.dataset)
            else:
                sys.exit("resnet doesn't implement those depth!")
        elif args.arch == "simple":
            # if dynamic:
            #     model = InOut(96, 2) # start with 2 classes
            if dataset_name == 'multi_modal_features':
                model = InOut(2048,8)
            else:
                model = InOut(96,6)
        else:
            sys.exit("wrong arch!")

        if args.multi_gpu:
            model = torch.nn.DataParallel(model)
        model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    criterion.__init__(reduce=False)
    early_stopping = utils.model_utils.EarlyStop(patience=args.patience, testing=False)


    # ----------- load checkpoint ---------------------
    model_state = None
    current_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if 'state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                model_state = checkpoint['state_dict']
                current_epoch = checkpoint['current_epoch']
            else:
                model_state = checkpoint

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            # time.sleep(1)
            model_state = None
        time.sleep(2)

    if not model_state is None:
        model.load_state_dict(model_state)

    if data_location is not None:
        args.modal_file = data_location

    log_filename = args.log_filename
    print(log_filename)

    log_filename_dir_str = log_filename.split('/')
    log_filename_dir = "/".join(log_filename_dir_str[:-1])
    if not os.path.exists(log_filename_dir):
        os.system('mkdir -p ' + log_filename_dir)
        print("New folder {} created...".format(log_filename_dir))

    with open(log_filename, 'a') as f:
        for arg in sorted(vars(args)):
            f.write("{}:".format(arg))
            f.write("{}".format(getattr(args, arg)))
            f.write("\n")

    # ------------- pre training ---------------------
    print("==============pre training=================")

    prune_init(args, model)
    prune_apply_masks()  # if wanted to make sure the mask is applied in retrain
    prune_print_sparsity(model)

    _, total_sparsity = test_sparsity(model, column=False, channel=False, filter=False, kernel=False)

    # CL buffer and dataset setup
    dataset = get_dataset(args)
    global is_two_dim, total_epochs
    is_two_dim = args.dataset in [SequentialHHAR.NAME, SequentialMultiModalFeatures.NAME]
    print('*' * 100)
    print('dataset', args)

    print("*" * 10 + f"Inspecting {args.dataset}" + "*" * 10)
    print("*" * 10 + "Initializing buffer" + "*" * 10)

    if args.buffer_size > 0:  # if there is a specified buffer size
        buffer = Buffer(args.buffer_size,
                        torch.device('cuda' if torch.cuda.is_available() else 'cpu'), mode=args.buffer_mode)  # create a buffer
    else:
        buffer = None
    # acc matric will be 3x3 if there are 3 tasks
    acc_matrix = np.zeros(
        (dataset.N_TASKS, dataset.N_TASKS))  # create an accuracy matrix of initialized with 0s for each of the tasks

    # Initialize dictionary to save statistics for every example presentation
    # example_stats_train = {}  # change name because fogetting function also have example_stats
    task_valid_info = {}


    for t in range(dataset.N_TASKS):  # for each copntinaul learning task set out
        example_stats_train = {}

        optimizer_init_lr = args.warmup_lr if args.warmup else args.lr  # the learning rate to be used

        optimizer = None
        if (args.optmzr == 'sgd'):  # the optimizers
            # optimizer = torch.optim.SGD(model.parameters(), optimizer_init_lr, momentum=0.9, weight_decay=1e-4)
            optimizer = torch.optim.SGD(model.parameters(),
                                        optimizer_init_lr)  # CL no momentum and wd set the optimizer
        elif (args.optmzr == 'adam'):
            optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)

        scheduler = None

        # initialize training dataset and full dataset here

        _, _, train_dataset, test_dat = dataset.get_data_loaders(return_dataset=True)  # get the training dataset
        full_dataset = copy.deepcopy(train_dataset)  # create a copy of the training dataset
        if args.buffer_mode == 'herding':
            # this is where you will check if the buffer is empty or not
            if buffer is not None and not buffer.is_empty():  # if the buffer is not empty
                # then you will combine the buffer inputs and labels with the training set
                print('buffer is not empty')
                buff_val = buffer.get_all_data()
                full_dataset.data = np.concatenate((full_dataset.data.cpu(), buff_val[0]), axis=0)
                full_dataset.targets  = np.concatenate((full_dataset.targets.cpu(), buff_val[1]), axis=0)
                full_dataset.classes = np.unique(np.concatenate((full_dataset.classes, np.unique(buff_val[1], axis=1))))

                # full_dataset.classes = full_dataset.classes.cuda()
                # full_dataset.targets = full_dataset.targets.cuda()
                # full_dataset.data = full_dataset.data.cuda()
        if args.sorting_file == None:
            train_indx = np.array(range(len(full_dataset.targets)))  # create an array from [0-> len(training_set)]
        else:
            try:
                with open(
                        os.path.join(args.input_dir, args.sorting_file) + '.pkl',
                        'rb') as fin:
                    ordered_indx = pickle.load(fin)['indices']
            except IOError:
                with open(os.path.join(args.input_dir, args.sorting_file),
                          'rb') as fin:
                    ordered_indx = pickle.load(fin)['indices']

            # Get the indices to remove from training
            elements_to_remove = np.array(ordered_indx)[
                                 -1:-1 + args.remove_n]  # get the indicies that will go the testing (i think)
            print('elements_to_remove', len(elements_to_remove))

            # Remove the corresponding elements
            train_indx = np.setdiff1d(range(len(train_dataset.targets)), elements_to_remove)
            print('train_indx', len(train_indx))
        # adam = torch.optim.Adam()
        # Reassign train data and labels and save the removed data
        train_dataset.data = full_dataset.data[train_indx, :, :, :] if not is_two_dim else full_dataset.data[train_indx,
                                                                                           :]  # this will set the data of the training set to be all values on all dimensions will not do anything for the first run
        print(train_dataset.data.shape)  # (35000, 32, 32, 3)
        # TODO this may not be mutating the overall architecture (may need to do a dataset.training = full_dataset.data[train_inx,:]
        train_dataset.targets = np.array(full_dataset.targets)[
            train_indx].tolist()  # set the dastaset targets to be the modified datatset get training sset labels
        print('len(train_dataset.targets)', len(train_dataset.targets))

        # for the purpose of masking outputs for other classes in the classification output
        if args.use_cl_mask:  # if you are to use a continuous learning mask then generate one
            cur_classes = np.arange(t * dataset.N_CLASSES_PER_TASK, (
                        t + 1) * dataset.N_CLASSES_PER_TASK)  # this creates an array of values that represent the claseses present for the current task being learned ex: first iter -> [0,1] number of classes in this task
            # print(cur_classes, "first part of the mask") # below will get the classes not in current task
            # if dynamic:
            #     cl_mask = np.setdiff1d(np.arange(model.layer2.out_features), # this will now get the classes not in the current task but allows a dynamic architecture
            #                        cur_classes)  # this will find the difference betweeen the two arrays so this will find difference between [0,...num_classes] and [0,1] (the classes in the current task) this returns the values present in the first not in the second
            # else:
            cl_mask = np.setdiff1d(np.arange(dataset.TOTAL_CLASSES), cur_classes)
            # print(cl_mask, "second part of the mask ") #the first iter should be [2,3,4,...,9] because those are the values present in all the classes not in the first
            # creates way to mask  the other classes out from the output
            # f.close()
        else:
            cl_mask = None

        # STARTING TRAINING for the task
        total_epochs = int(args.epochs / dataset.N_TASKS)
        if epoch_info is not None and epoch_info.get(t) is not None: # if custom epochs have been set
            total_epochs = epoch_info[t]

        for epoch in range(
                total_epochs):  # for each epoch (Btw batch is a subset of training used in one iteration update model in batches) epoch is one full pass through the whole dataset
            prune_update(
                epoch)  # prune and grow with mask updates this calls update mask in the retrain  does growing and pruning every 5 epoch
            optimizer.zero_grad()  # set the optimizer to be 0 (need to do this for training)

            #########remove data at 25 epoch, update dataset ######
            if epoch > 0 and epoch % args.sp_mask_update_freq == 0 and epoch <= args.remove_data_epoch:
                if args.sorting_file == None:
                    print('epoch', epoch)

                    unlearned_per_presentation_all, first_learned_all = [], []
                    # calcualte the number of tasks forgotten
                    _, unlearned_per_presentation, _, first_learned = compute_forgetting_statistics(example_stats_train,
                                                                                                    int(args.epochs / dataset.N_TASKS))
                    print('unlearned_per_presentation', len(unlearned_per_presentation))
                    print('first_learned', len(first_learned))

                    unlearned_per_presentation_all.append(unlearned_per_presentation)
                    first_learned_all.append(first_learned)

                    print('unlearned_per_presentation_all', len(unlearned_per_presentation_all))
                    print('first_learned_all', len(first_learned_all))

                    # print('epoch before sort ordered_examples len',len(ordered_examples))

                    # Sort examples by forgetting counts in ascending order, over one or more training runs
                    ordered_examples, ordered_values, num_unforget = sort_examples_by_forgetting(
                        unlearned_per_presentation_all, first_learned_all, int(args.epochs / dataset.N_TASKS))

                    # Save sorted output
                    if args.output_name.endswith('.pkl'):
                        with open(os.path.join(args.output_dir,
                                               args.output_name + "_task_" + str(t) + "_unforget_" + str(num_unforget)),
                                  'wb') as fout:
                            pickle.dump({
                                'indices': ordered_examples,
                                'forgetting counts': ordered_values
                            }, fout)
                    else:
                        with open(
                                os.path.join(args.output_dir, args.output_name + "_task_" + str(t) + "_unforget_" + str(
                                    num_unforget) + '.pkl'),
                                'wb') as fout:
                            pickle.dump({
                                'indices': ordered_examples,
                                'forgetting counts': ordered_values
                            }, fout)

                    # Get the indices to remove from training
                    print('epoch before ordered_examples len', len(ordered_examples))
                    print('epoch before len(train_dataset.targets)', len(train_dataset.targets))
                    elements_to_remove = np.array(
                        ordered_examples)[args.keep_lowest_n:args.keep_lowest_n + (
                        int(args.remove_n / (int(args.remove_data_epoch) / args.sp_mask_update_freq)))]
                    # Remove the corresponding elements
                    print('elements_to_remove', len(elements_to_remove))

                    train_indx = np.setdiff1d(
                        # range(len(train_dataset.targets)), elements_to_remove)
                        train_indx, elements_to_remove)
                    print('removed train_indx', len(train_indx))

                    # Reassign train data and labels
                    if is_two_dim:
                        train_dataset.data = full_dataset.data[train_indx]
                        train_dataset.targets = np.array(
                            full_dataset.targets)[train_indx].tolist()
                    else:
                        train_dataset.data = full_dataset.data[train_indx, :, :, :] if not is_two_dim else full_dataset[
                            train_indx]  # TODO SEE WHATS HAPPENING HERE
                        train_dataset.targets = np.array(
                            full_dataset.targets)[train_indx].tolist()

                    print('shape', train_dataset.data.shape)
                    print('len(train_dataset.targets)', len(train_dataset.targets))

                    # print('epoch after random ordered_examples len', len(ordered_examples))
                    #####empty example_stats_train!!! Because in original, forget process come before the whole training process
                    example_stats_train = {}

                ##########

            print('Training on ' + str(len(train_dataset.targets)) + ' examples')

            train(model, train_dataset, criterion, scheduler, optimizer, epoch, t, buffer, dataset,
                  # train the model, divide into batches and then perform
                  example_stats_train, train_indx, maskretrain=False, masks={}, cl_mask=cl_mask,
                  task_dict=task_valid_info)
            if args.validation:
                print("=" * 120, 'validation')
                total_loss = validation(model, dataset, epoch, t, task_valid_info)
                print("Total Validation Loss: ", total_loss)
                if run_num is not None and run_num==0 and early_stop: # you can perform early stopping only on the first run
                    if early_stopping.check(total_loss):
                        # set the epoch information so that the next run knows
                        if epoch_info is None:
                            epoch_info = {}
                        epoch_info[t] = epoch+1
                        print('Early stopping at epoch: ', epoch)
                        acc_list, til_acc_list = evaluate(model, dataset, test=t==dataset.N_TASKS-1 and epoch == (total_epochs-1)) # run if the last task
                        prec1 = sum(acc_list) / (t + 1)
                        til_prec1 = sum(til_acc_list) / (t + 1)
                        acc_matrix[t] = acc_list
                        forgetting = np.mean((np.max(acc_matrix, axis=0) - acc_list)[:t]) if t > 0 else 0.0
                        learning_acc = np.mean(np.diag(acc_matrix)[:t + 1])

                        lr = optimizer.param_groups[0]['lr']
                        log_line = 'Training on ' + str(len(train_dataset.targets)) + ' examples\n'
                        log_line += f"Task: {t}, Epoch:{epoch}, Average Acc:[{prec1:.3f}], , Task Inc Acc:[{til_prec1:.3f}], Learning Acc:[{learning_acc:.3f}], Forgetting:[{forgetting:.3f}], LR:{lr}\n"
                        log_line += "\t"
                        for i in range(t + 1):
                            log_line += f"Acc@T{i}: {acc_list[i]:.3f}\t"
                        log_line += "\n"
                        log_line += "\t"
                        for i in range(t + 1):
                            log_line += f"Til-Acc@T{i}: {til_acc_list[i]:.3f}\t"
                        log_line += "\n"
                        print(log_line)
                        with open(log_filename, 'a') as f:
                            f.write(log_line)
                            f.write("\n")

                        if args.evaluate_mode and args.eval_checkpoint is not None:
                            break
                        early_stopping.reset()
                        break # stop the training for the task

            prune_print_sparsity(model)  # at the end prune and grow
            if args.gradient_efficient or args.gradient_efficient_mix:  # show the sparsity of the mask
                show_mask_sparsity()

            if epoch % args.test_epoch_interval == 10 or epoch == (total_epochs - 1):
            #if epoch % args.test_epoch_interval == 10 or epoch == (int(args.epochs / dataset.N_TASKS) - 1):
                print(t, 'running the eval stage')
                acc_list, til_acc_list = evaluate(model, dataset, test=t==dataset.N_TASKS-1 and epoch == (total_epochs-1)) # run if the last task

                if nums_run == 0:
                    if epoch_info is None:
                        epoch_info = {}
                    epoch_info[t] = epoch
                prec1 = sum(acc_list) / (t + 1)
                til_prec1 = sum(til_acc_list) / (t + 1)
                acc_matrix[t] = acc_list
                forgetting = np.mean((np.max(acc_matrix, axis=0) - acc_list)[:t]) if t > 0 else 0.0
                learning_acc = np.mean(np.diag(acc_matrix)[:t + 1])

                lr = optimizer.param_groups[0]['lr']
                log_line = 'Training on ' + str(len(train_dataset.targets)) + ' examples\n'
                log_line += f"Task: {t}, Epoch:{epoch}, Average Acc:[{prec1:.3f}], , Task Inc Acc:[{til_prec1:.3f}], Learning Acc:[{learning_acc:.3f}], Forgetting:[{forgetting:.3f}], LR:{lr}\n"
                log_line += "\t"
                for i in range(t + 1):
                    log_line += f"Acc@T{i}: {acc_list[i]:.3f}\t"
                log_line += "\n"
                log_line += "\t"
                for i in range(t + 1):
                    log_line += f"Til-Acc@T{i}: {til_acc_list[i]:.3f}\t"
                log_line += "\n"
                print(log_line)
                with open(log_filename, 'a') as f:
                    f.write(log_line)
                    f.write("\n")

                if args.evaluate_mode and args.eval_checkpoint is not None:
                    break


        # save model checkpoint after every task
        filename = "./{}seed{}_{}_{}{}_{}_acc_{:.3f}_fgt_{:.3f}_{}_lr{}_{}_sp{:.3f}_task_{}.pt".format(args.save_model,
                                                                                                       seed,
                                                                                                       args.remark,
                                                                                                       args.arch,
                                                                                                       args.depth,
                                                                                                       args.dataset,
                                                                                                       prec1,
                                                                                                       forgetting,
                                                                                                       args.optmzr,
                                                                                                       args.lr,
                                                                                                       args.lr_scheduler,
                                                                                                       total_sparsity,
                                                                                                       t)
        torch.save(model.state_dict(), filename)
        # at the end of the training of the task fill the buffer with examples
        # if args.buffer_mode == 'herding':
        #     print('herding')
        #     buffer.fill_buffer(model, dataset,t)

        # at the end of the task you will extend the model
        # if dynamic:
        #     print('running dynamic')
        #     model.extend_fc_layer(dataset.N_CLASSES_PER_TASK) # it will add n classes to the prediction
        # buffer_class_percentage(buffer,t  )


    _,_,f1_micro, f1_macro = test(model, dataset)
    with open('results_file.txt', 'a')as f:
        val = f"{args.modal_file}: {f1_micro, f1_macro}"
        f.write(val)
    # dump the validation data

    run_file = 'HHAR/hhar_features.pkl' if args.modal_file is None else 'gyro'
    return task_valid_info, epoch_info
    # with open(f"{run_file}_validation_{args.arch}_validation.pkl", 'wb') as f:
    #     pickle.dump(task_valid_info, f)


def process_validations(valids, modal_type):

    first = valids[0]

    # go through each of the keys
    for task in first.keys():
        print(task)
        for epoch in first[task].keys():  # go through each epoch
            for i, epoch_task in enumerate(
                    first[task][epoch]['individual']):  # these are the stats at each epoch for each task
                # get all the accuracies for all runs at this point
                for j, stat in enumerate(epoch_task.keys()):
                    print(stat, 'this is the stat being run on')
                    # collect the stat across all runs
                    all_run_stat = np.array([x[task][epoch]['individual'][i][stat] for x in valids])
                    stat_mean = np.mean(all_run_stat)
                    print(all_run_stat, stat_mean)
                    print("\n")
                    # at the end set the first
                    first[task][epoch]['individual'][i][stat] = stat_mean

    with open(f"{modal_type}_fiverun_mean_validation_simple.pkl", 'wb') as f:
        pickle.dump(first, f)


if __name__ == '__main__':
    nums_run = 1

    # run_files = {
    #     'gyro': 'HHAR/20231103-182025_hhar_features_gyro.pkl',
    #     'accel': 'HHAR/20231104-162132_hhar_features_accel.pkl'
    # }
    cnn_features = False
    # run_files = {
    #     'both': '../SensorBasedTransformerTorch/extracted_features/multi_modal_extracted_features.hkl'
    # }



    # run_files = {
    #     'accel': '../selfhar_testing/',
    #     'gyro': '../selfhar_testing/20231204-023349_SHL_features_gyro.hkl'
    #
    # }
    cnn_dir = '../selfhar_testing/features_dir/'
    clustering_dir = '../SensorBasedTransformerTorch/extracted_features/'
    # get the cnn features
    use_cnn = True
    use_clustering = True
    cnn = {f"cnn_accel{i}" if "accel" in x else f"cnn_gyro{i}": os.path.join(cnn_dir,x) for i, x in enumerate(os.listdir(cnn_dir))} if use_cnn else {}
    clustering = {f"clustering{i}": os.path.join(clustering_dir,x) for i,x in enumerate(os.listdir(clustering_dir))} if use_clustering else {}


    run_files = {**cnn, **clustering}

    for dataset in run_files.keys():
        validations = []
        dataset_name = 'multi_modal_features' if "clustering" in dataset else 'hhar_features'
        # epochs = 500
        epoch_information = None
        for i in range(nums_run): # this will run n times to be averaged
            valid_info, epoch_info = sparCL(args, dataset_name, run_files[dataset],run_num=i, epoch_info=epoch_information)
            if i == 0:
                epoch_information = epoch_info
            validations.append(valid_info)
            torch.cuda.empty_cache()  # clear the cache after each run
        process_validations(validations, dataset)

    print('done')
