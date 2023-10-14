from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import sys
import pickle
import collections
from numpy import linalg as LA
import copy
import yaml
import numpy as np

import datetime
import operator
import random

def prune_parse_arguments(parser):
    admm_args = parser.add_argument_group('Multi level admm arguments')
    admm_args.add_argument('--sp-load-frozen-weights',
        type=str, help='the weights that are frozen '
        'throughout the pruning process')


def canonical_name(name):
    # if the model is running in parallel, the name may start
    # with "module.", but if hte model is running in a single
    # GPU, it may not, we always filter the name to be the version
    # without "module.",
    # names in the config should not start with "module."
    if "module." in name:
        return name.replace("module.", "")
    else:
        return name


def _collect_dir_keys(configs, dir):
    if not isinstance(configs, dict):
        return

    for name in configs:
        if name not in dir:
            dir[name] = []
        dir[name].append(configs)
    for name in configs:
        _collect_dir_keys(configs[name], dir)


def _canonicalize_names(configs, model, logger):
    dir = {}
    collected_keys = _collect_dir_keys(configs, dir)
    for name in model.state_dict():
        cname = canonical_name(name)
        if cname == name:
            continue
        if name in dir:
            assert cname not in dir
            for parent in dir[name]:
                assert cname not in parent
                parent[cname] = parent[name]
                del parent[name]
            print("Updating parameter from {} to {}".format(name, cname))


def load_configs(model, filename, logger):
    assert filename is not None, \
            "Config file must be specified"

    with open(filename, "r") as stream:
        try:
            configs = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    _canonicalize_names(configs, model, logger)

    if "prune_ratios" in configs:
        config_prune_ratios = configs["prune_ratios"]

        count = 0
        prune_ratios = {}
        for name in model.state_dict():
            W = model.state_dict()[name]
            cname = canonical_name(name)

            if cname not in config_prune_ratios:
                continue
            count = count + 1
            prune_ratios[name] = config_prune_ratios[cname]
            if name != cname:
                print("Map weight config name from {} to {}".\
                    format(cname, name))

        if len(prune_ratios) != len(config_prune_ratios):
            extra_weights = set(config_prune_ratios) - set(prune_ratios)
            for name in extra_weights:
                print("{} in config file cannot be found".\
                    format(name))

    return configs, prune_ratios


def weight_pruning(args, configs, name, w, prune_ratio, mask_fixed_params=None):
    """
    weight pruning [irregular,column,filter]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_ratio (float between 0-1): target sparsity of weights

    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero
    This will prune weights by using the prune ratio which lists the percentile the weights should be equal to or over to not be pruned, if there are gradients for the weights, then it will include the gradient into the weight value
    """

    weight = w.detach().clone().cpu().numpy()  # convert cpu tensor to numpy vector
    weight_ori = copy.copy(weight) #copy the original weight

    w_grad = None
    if not w.grad is None and args.sp_lmd: # if there is gradient and the sparsity lmd is set
        grad_copy = copy.copy(w.grad) # copy the gradient
        w_grad = grad_copy.detach().clone().cpu().numpy() # move the gradient to numpy arr so that it can be modified

    if mask_fixed_params is not None: # if there are fixed mask parameters in the mask
        mask_fixed_params = mask_fixed_params.detach().cpu().numpy()

    percent = prune_ratio * 100 #convert the prune ratio to a percentage this was the lower bound value originally
    if (args.sp_admm_sparsity_type == "irregular") or (args.sp_admm_sparsity_type == "irregular_global"): #if you have irregular sparsity types (CURRENTLY MY situation does)

        weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values so that you can capture the magnitude of each weigth value
        if not w_grad is None: # if the weight gradient is set
            grad_temp = np.abs(w_grad) #get the absolute values of the gradient so that you can capture the magnitude of the gradient
            imp_temp = weight_temp + (args.sp_lmd * grad_temp) # calculate the temporary weight values to include the weight and the gradient but modifying the gradient value but the sp_lmd
        else:
            imp_temp = weight_temp # if no weight gradient then set the imp temp to be the absolute value of the weight temporary value
        percentile = np.percentile(imp_temp, percent)  # get a value that is in the 75th percentile  of magnitude this is in absolute value (in my case this will get the vlaues in the 75th percentile)
        under_threshold = imp_temp < percentile # get the values that are below this percentile  this will be a boolean list
        above_threshold = imp_temp > percentile # get the values above the 75th percentile (bool list)
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion this will be 1 to indicate if the value is above the listed percentile and 0 if not to be masked
        # weight[under_threshold] = 0
        ww = weight * above_threshold #multiply the weights by the weights the bool array containing 1 indicating if the weight is over the threshold, this will remove all values under the threshold and keep the ones over the threshold the ones under will be multiplied by 0
        return torch.from_numpy(above_threshold), torch.from_numpy(ww) # return two tensors the first containing the bool converted to int list containing values over 75th percentile(mask) and the second value is the weights above or equal to the 75th percentile

    elif (args.sp_admm_sparsity_type == "random_irregular"):

        non_zeros = np.zeros(weight.shape).flatten()
        print("percent:", prune_ratio)
        print("non-zeros.size:", non_zeros.size)
        print("non-zeros index:", int(non_zeros.size * (1 - prune_ratio)))
        non_zeros[:int(non_zeros.size * (1 - prune_ratio))] = 1

        np.random.shuffle(non_zeros)

        non_zeros = np.reshape(non_zeros, weight.shape)
        non_zeros = non_zeros.astype(np.float32)
        # zero_mask = torch.from_numpy(non_zeros).cuda()
        # weight *= non_zeros
        ww = weight_ori * non_zeros

        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(ww).cuda()

    elif (args.sp_admm_sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        row_l2_norm = LA.norm(weight2d, 2, axis=1)
        percentile = np.percentile(row_l2_norm, percent)
        under_threshold = row_l2_norm < percentile
        above_threshold = row_l2_norm > percentile
        weight2d[under_threshold, :] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = above_threshold[i]
        weight = weight2d.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        ww = weight_ori * expand_above_threshold
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(ww).cuda()

    elif (args.sp_admm_sparsity_type == "pattern"):
        print("pattern pruning...", weight.shape)
        shape = weight.shape

        if shape[2] != 3:
            non_zeros = weight != 0
            non_zeros = non_zeros.astype(np.float32)
            return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()

        pattern1 = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]]  # 3
        pattern2 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]  # 12
        pattern3 = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]]  # 65
        pattern4 = [[0, 2], [1, 2], [2, 0], [2, 1], [2, 2]]  # 120

        pattern5 = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]  # 1
        pattern6 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 2]]  # 14
        pattern7 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 2]]  # 44
        pattern8 = [[0, 0], [0, 2], [1, 2], [2, 0], [2, 2]]  # 53

        pattern9 = [[1, 0], [1, 2], [2, 0], [2, 1], [2, 2]]  # 125
        pattern10 = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2]]  # 6
        pattern11 = [[1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]  # 126
        pattern12 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 0]]  # 10

        patterns_dict = {1: pattern1,
                         2: pattern2,
                         3: pattern3,
                         4: pattern4,
                         5: pattern5,
                         6: pattern6,
                         7: pattern7,
                         8: pattern8
                         }

        for i in range(shape[0]):
            for j in range(shape[1]):
                current_kernel = weight[i, j, :, :].copy()
                temp_dict = {} # store each pattern's norm value on the same weight kernel
                for key, pattern in patterns_dict.items():
                    temp_kernel = current_kernel.copy()
                    for index in pattern:
                        temp_kernel[index[0], index[1]] = 0
                    current_norm = LA.norm(temp_kernel)
                    temp_dict[key] = current_norm
                best_pattern = max(temp_dict.items(), key=operator.itemgetter(1))[0]
                # print(best_pattern)
                for index in patterns_dict[best_pattern]:
                    weight[i, j, index[0], index[1]] = 0
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        # zeros = weight == 0
        # zeros = zeros.astype(np.float32)
        ww = weight_ori * non_zeros
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(ww).cuda()
    elif (args.sp_admm_sparsity_type == "random_pattern"):
        print("pattern pruning...", weight.shape)
        shape = weight.shape

        if shape[2] != 3:
            non_zeros = weight != 0
            non_zeros = non_zeros.astype(np.float32)
            return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()

        pattern1 = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]]  # 3
        pattern2 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]  # 12
        pattern3 = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]]  # 65
        pattern4 = [[0, 2], [1, 2], [2, 0], [2, 1], [2, 2]]  # 120

        pattern5 = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]  # 1
        pattern6 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 2]]  # 14
        pattern7 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 2]]  # 44
        pattern8 = [[0, 0], [0, 2], [1, 2], [2, 0], [2, 2]]  # 53

        pattern9 = [[1, 0], [1, 2], [2, 0], [2, 1], [2, 2]]  # 125
        pattern10 = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2]]  # 6
        pattern11 = [[1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]  # 126
        pattern12 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 0]]  # 10

        patterns_dict = {1 : pattern1,
                         2 : pattern2,
                         3 : pattern3,
                         4 : pattern4,
                         5 : pattern5,
                         6 : pattern6,
                         7 : pattern7,
                         8 : pattern8
                         }
        for i in range(shape[0]):
            for j in range(shape[1]):
                current_kernel = weight[i, j, :, :].copy()
                pick_pattern = random.choice(list(patterns_dict.values()))
                for index in pick_pattern:
                    weight[i, j, index[0], index[1]] = 0
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        # zeros = weight == 0
        # zeros = zeros.astype(np.float32)
        ww = weight_ori * non_zeros
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(ww).cuda()

    elif (args.sp_admm_sparsity_type == "filter_balance"):

        def ratio_convert(overall_ratio):
            return 1 - ((1 - overall_ratio) * 9 / 4)

        print("pruning filter with balanced outputs")

        shape = weight.shape
        if shape[2] == 3:
            prune_ratio = ratio_convert(prune_ratio)

        kth_smallest = int(shape[1] * prune_ratio)  # the percent from script is used to represent k-th smallest l2-norm kernel will be pruned in each filter

        weight3d = weight.reshape(shape[0], shape[1], -1)
        for i in range(shape[0]):
            kernel_l2norm_list = LA.norm(weight3d[i,:,:], 2, axis=1)
            partial_sorted_index = np.argpartition(kernel_l2norm_list, kth_smallest)  # list of all indices, but partially sorted
            kth_smallest_index = partial_sorted_index[:kth_smallest]  # indices of k-th smallest l2-norm
            for idx in kth_smallest_index:
                weight3d[i, idx, :] = 0
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)

        ww = weight_ori * non_zeros

        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(ww).cuda()



    elif (args.sp_admm_sparsity_type == "free_block_prune_column_4"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape

        length_f = 4
        if shape2d[0] % length_f != 0:
            print("the layer size is not divisible")
            raise SyntaxError("block_size error")

        cross_f = int(shape2d[0] / length_f)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)

        mat = None
        mat_above = None

        for f in range(cross_f):
            # print("f={}/{}".format(f, crossbar_num_f))
            frag = weight2d[f * length_f:(f + 1) * length_f, :]
            frag_above = expand_above_threshold[f * length_f:(f + 1) * length_f, :]
            if mat is None:
                mat = frag
                mat_above = frag_above
            else:
                mat = np.hstack((mat, frag))
                mat_above = np.hstack((mat_above, frag_above))

        row_l2_norm = LA.norm(mat, 2, axis=0)
        percentile = np.percentile(row_l2_norm, percent)
        under_threshold = row_l2_norm <= percentile
        above_threshold = row_l2_norm > percentile
        mat[:, under_threshold] = 0
        # weight2d[weight2d < 1e-40] = 0
        above_threshold = above_threshold.astype(np.float32)


        for i in range(shape2d[1] * cross_f):
            mat_above[:, i] = above_threshold[i]

        for f in range(cross_f):
            # print("f={}/{}".format(f, crossbar_num_f))
            weight2d[f * length_f:(f + 1) * length_f, :] = mat[:, f * shape2d[1]: (f + 1) * shape2d[1]]
            expand_above_threshold[f * length_f:(f + 1) * length_f, :] = mat_above[:, f * shape2d[1]: (f + 1) * shape2d[1]]

            # change frag will change weight2d as well

        weight = weight.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)

        ww = weight_ori * expand_above_threshold

        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(ww).cuda()


    raise SyntaxError("Unknown sparsity type: {}".format(args.sp_admm_sparsity_type))


def weight_growing(args, name, pruned_weight_np, lower_bound_value, upper_bound_value, update_init_method, mask_fixed_params=None):
    shape = None
    weight1d = None

    if mask_fixed_params is not None:
        mask_fixed_params = mask_fixed_params.detach().cpu().numpy()

    if upper_bound_value == 0: # if the upper bound of growing is 0
        print("==> GROW: {}: to DENSE despite the sparsity type is \n".format(name))
        np_updated_mask = np.ones_like(pruned_weight_np, dtype=np.float32)
        updated_mask = torch.from_numpy(np_updated_mask).cuda()
        return updated_mask

    if upper_bound_value == lower_bound_value:
        print("==> GROW: {}: no grow, keep the mask and do finetune \n".format(name))
        non_zeros_updated = pruned_weight_np != 0
        non_zeros_updated = non_zeros_updated.astype(np.float32)
        np_updated_mask = non_zeros_updated
        updated_mask = torch.from_numpy(np_updated_mask).cuda()
        return updated_mask

    if (args.sp_admm_sparsity_type == "irregular"): # if there is irregualr sparsity
        # randomly select and set zero weights to non-zero to restore sparsity
        non_zeros_prune = pruned_weight_np != 0 # get the values that are not zero from the pruned weight

        shape = pruned_weight_np.shape
        weight1d = pruned_weight_np.reshape(1, -1)[0] # reshape the pruned weight this will change the shappe to have 1 row and as many columns the -1 says automatically calculate the # of cols based on total elements in array and get the one array from the 2d array this makes the array 1d
        zeros_indices = np.where(weight1d == 0)[0] # get the index of values that are 0
        if args.sp_global_magnitude: # if there is global sparsity magnitude
            num_added_zeros = int((lower_bound_value - upper_bound_value) * np.size(weight1d))
        else: # the upper bound is the desired sparsity (in our case: 0.74) so this will get the diff between the current sparsity and the desired sparsity in the total weight
            num_added_zeros = int(np.size(zeros_indices) - upper_bound_value * np.size(weight1d)) # substract the size of zero values by the upper bound value * the number of values in the weights ex: 36864 - (0.74 * 49512) this will get this will not add more zeros than are present in the weight
        num_added_zeros = num_added_zeros if num_added_zeros < np.size(zeros_indices) else np.size(zeros_indices) # if the number of zero values is greater than the  number of added zeros then the number of number of added zeros otherwise return the number of zero values
        num_added_zeros = num_added_zeros if num_added_zeros > 0 else 0 # if the number of zeros is greater than zero then return the value else 0
        target_sparsity = 1 - (np.count_nonzero(non_zeros_prune) + num_added_zeros) * 1.0 / np.size(pruned_weight_np) # get the sparsity by 1- the number of nonzeros in the pruned weight  + the nubmer of added zeros /  size of the weight  1- 12271(non zero values in pruned tensor) + 508(zeros added) /49152(total values)
        indices = np.random.choice(zeros_indices, #take the indicies of the zero values in the weight and choose num_added_zeros number of indicies that are to be selected to be grown
                                   num_added_zeros,
                                   replace=False)
        print("==> CALCULATE: all zeros: {}, need grow {} zeros, selected zeros: {} ".format(len(zeros_indices),
                                                                                             num_added_zeros,
                                                                                             len(indices)))
        #this says there are that many zeros and you need to introduce that many zeros and you selected the following
        # initialize selected weights
        if update_init_method == "weight":
            pass
            # current_nozero = weight1d[np.nonzero(weight1d)]
            # current_mean = np.mean(current_nozero)
            # current_std = np.std(current_nozero)
            # weight1d[indices] = np.random.normal(loc=current_mean, scale=current_std, size=np.size(indices))
            #
            # weight = weight1d.reshape(shape)
            #
            # print("==> double check sparsity after updating mask...")
            # non_zeros_updated = weight != 0
            # non_zeros_updated = non_zeros_updated.astype(np.float32)
            # num_nonzeros_updated = np.count_nonzero(non_zeros_updated)
            # sparsity_updated = 1 - (num_nonzeros_updated * 1.0) / total_num
            # print(("{}: {}, {}, {}\n".format(name, str(num_nonzeros_updated), str(total_num), str(sparsity_updated))))
            #
            # # update mask
            # # zero_mask = torch.from_numpy(non_zeros_updated).cuda()
            # np_updated_zero_one_mask = non_zeros_updated
            #
            # # write updated weights back to model
            # model.state_dict()[name].data.copy_(torch.from_numpy(weight))
        elif update_init_method == "zero": # the update init method will set the selected zero values to be "grown back" to be set to -1
            # set selected weights to -1 to get corrrect updated masks
            weight1d[indices] = -1 # set the randomly selected inactive weight  values to -1
            weight = weight1d.reshape(shape) # reshape back to the oringinal tensor
            non_zeros_updated = weight != 0 # get the values where the weight is not 0 of the now updated values
            non_zeros_updated = non_zeros_updated.astype(np.float32) # convert true -> 1 false -> 0 this allows for the values that are -1 to be shown as 1 to be activated in order to update the mask
            print("==> GROW: {}: revise sparse mask to sparsity {}\n".format(name, target_sparsity))
            #what is happening above is that the selected inactive values will be set to be -1 and a new mask will be made from all the values that are not 0 (includes -1) so  that you know the new mask that should be active
            # update mask
            # zero_mask = torch.from_numpy(non_zeros_updated).cuda()
            np_updated_zero_one_mask = non_zeros_updated # set the new mask to be the grown mask

            # assign 0 to -1 weight
            weight1d[indices] = 0
            weight = weight1d.reshape(shape)

            # write updated weights back to model
            # self.model.state_dict()[name].data.copy_(torch.from_numpy(weight))
        elif update_init_method == "kaiming":
            assert (False)

        np_updated_mask = np_updated_zero_one_mask # set the new mask to be the mask with the added growth
        updated_mask = torch.from_numpy(np_updated_mask).cuda() # convert this to a torch tensor and move to gpu

        return updated_mask





    elif "pattern" in args.sp_admm_sparsity_type:

        def ratio_convert(overall_ratio):
            return 1 - ((1 - overall_ratio) * 9 / 4)

        pattern1 = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]]  # 3
        pattern2 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]  # 12
        pattern3 = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]]  # 65
        pattern4 = [[0, 2], [1, 2], [2, 0], [2, 1], [2, 2]]  # 120

        pattern5 = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]  # 1
        pattern6 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 2]]  # 14
        pattern7 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 2]]  # 44
        pattern8 = [[0, 0], [0, 2], [1, 2], [2, 0], [2, 2]]  # 53

        patterns_dict = {1: pattern1,
                         2: pattern2,
                         3: pattern3,
                         4: pattern4,
                         5: pattern5,
                         6: pattern6,
                         7: pattern7,
                         8: pattern8
                         }

        shape = pruned_weight_np.shape

        if shape[2] == 3:
            upper_bound_value = ratio_convert(upper_bound_value)

        conv_kernel_indicate = np.sum(pruned_weight_np, axis=(2, 3))
        conv_kernel_indicate = conv_kernel_indicate != 0
        conv_kernel_indicate = conv_kernel_indicate.astype(np.float32)

        conv_kernel_sum = np.sum(conv_kernel_indicate)

        if (conv_kernel_sum == (shape[0] * shape[1])): # np empty kernel exist
            non_zeros_updated = pruned_weight_np != 0
            non_zeros_updated = non_zeros_updated.astype(np.float32)
            updated_mask = torch.from_numpy(non_zeros_updated).cuda()
            return updated_mask
        else:
            for i in range(shape[0]):
                zeros_indices_per_filter = np.where(conv_kernel_indicate[i, :] == 0)[0] # find empty kernel per filter
                num_added_kernel_per_filter = int(np.size(zeros_indices_per_filter) - upper_bound_value * shape[1]) # calculate how many empty kernels per filter shoule be grown
                indices = np.random.choice(zeros_indices_per_filter,
                                           num_added_kernel_per_filter,
                                           replace=False)
                conv_kernel_indicate[i, :][indices] = -1

            c = np.where(conv_kernel_indicate == -1) # empty kernel indices of the ones need to be grow

            print("==> CALCULATE: all kernels: {}, need grow {} kernels ".format(np.size(conv_kernel_indicate),
                                                                                 len(c[0])))

            for idx in range(len(c[0])):
                target_kernel = pruned_weight_np[c[0][idx], c[1][idx], :, :] # find the empty kernel in weight
                target_kernel = np.ones_like(target_kernel)

                if shape[2] == 3:
                    pick_pattern = random.choice(list(patterns_dict.values()))
                    for index in pick_pattern:
                        target_kernel[index[0], index[1]] = 0
                pruned_weight_np[c[0][idx], c[1][idx], :, :] = target_kernel

            non_zeros_updated = pruned_weight_np != 0
            non_zeros_updated = non_zeros_updated.astype(np.float32)
            np_updated_mask = non_zeros_updated
            updated_mask = torch.from_numpy(np_updated_mask).cuda()

            mask_sparsity = 1 - (np.count_nonzero(np_updated_mask)) * 1.0 / np.size(pruned_weight_np)

            print("==> GROW: {}: revise sparse mask to sparsity {}\n".format(name, mask_sparsity))

            return updated_mask

    elif (args.sp_admm_sparsity_type == "free_block_prune_column_4"):
        shape = pruned_weight_np.shape
        weight2d = pruned_weight_np.reshape(shape[0], -1)
        shape2d = weight2d.shape

        mat = None
        length_f = 4
        cross_f = int(shape2d[0] / length_f)

        if shape2d[0] % length_f != 0:
            print("the layer size is not divisible")
            raise SyntaxError("block_size error")

        for f in range(cross_f):
            frag = weight2d[f * length_f:(f + 1) * length_f, :]
            if mat is None:
                mat = frag
            else:
                mat = np.hstack((mat, frag))

        block_indicate = np.sum(mat, axis=0)
        zeros_indices_per_block = np.where(block_indicate[:] == 0)[0] # find empty block per layer
        num_added_block = int(np.size(zeros_indices_per_block) - upper_bound_value * mat.shape[1])
        indices = np.random.choice(zeros_indices_per_block,
                                   num_added_block,
                                   replace=False)
        print("==> CALCULATE: all blocks: {}, need grow {} blocks ".format(np.size(block_indicate), len(indices)))

        mat[:, indices] = -1

        for f in range(cross_f):
            weight2d[f * length_f:(f + 1) * length_f, :] = mat[:, f * shape2d[1]: (f + 1) * shape2d[1]]

        pruned_weight_np = weight2d.reshape(shape)

        non_zeros_updated = pruned_weight_np != 0
        non_zeros_updated = non_zeros_updated.astype(np.float32)
        np_updated_mask = non_zeros_updated
        updated_mask = torch.from_numpy(np_updated_mask).cuda()

        mask_sparsity = 1 - (np.count_nonzero(np_updated_mask)) * 1.0 / np.size(pruned_weight_np)

        print("==> GROW: {}: revise sparse mask to sparsity {}\n".format(name, mask_sparsity))

        return updated_mask