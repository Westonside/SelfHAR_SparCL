
import torch
import logging
import sys
import os
import numpy as np
import argparse
import time
import random
import copy
from . import utils_pr
# from .admm import weight_growing, weight_pruning, ADMM
from .utils_pr import weight_pruning, weight_growing

def prune_parse_arguments(parser):
    parser.add_argument('--retrain-mask-pattern', type=str, default='weight',
                    help="retrain mask pattern")
    parser.add_argument('--sp-update-init-method', type=str, default='zero',
                        help="mask update initialization method")
    parser.add_argument('--sp-mask-update-freq', type=int, default=5,
                        help="how many epochs to update sparse mask")
    parser.add_argument('--sp-lmd', type=float, default=0.5,
                        help="importance coefficient lambda")
    parser.add_argument('--retrain-mask-sparsity', type=float, default=-1.0,
                    help="sparsity of a retrain mask, used when retrain-mask-pattern is set to NOT being 'weight' ")
    parser.add_argument('--retrain-mask-seed', type=int, default=None,
                    help="seed to generate a random mask")
    parser.add_argument('--sp-prune-before-retrain', action='store_true',
                        help="Prune the loaded model before retrain, in case of loading a dense model")
    parser.add_argument('--output-compressed-format', action='store_true',
                        help="output compressed format")
    parser.add_argument("--sp-grad-update", action="store_true",
                        help="enable grad update when training in random GaP")
    parser.add_argument("--sp-grad-decay", type=float, default=0.98,
                        help="The decay number for gradient")
    parser.add_argument("--sp-grad-restore-threshold", type=float, default=-1,
                        help="When the decay")
    parser.add_argument("--sp-global-magnitude", action="store_true",
                        help="Use global magnitude to prune models")
    parser.add_argument('--sp-pre-defined-mask-dir', type=str, default=None,
                        help="using another sparse model to init sparse mask")

    parser.add_argument('--upper-bound', type=str, default=None,
                        help="using another sparse model to init sparse mask")
    parser.add_argument('--lower-bound', type=str, default=None,
                        help="using another sparse model to init sparse mask")
    parser.add_argument('--mask-update-decay-epoch', type=str, default=None,
                        help="using another sparse model to init sparse mask")


class SparseTraining(object):
    def __init__(self, args, model, logger=None, pre_defined_mask=None, seed=None):
        self.args = args
        # we assume the model does not change during execution
        self.model = model
        self.pattern = self.args.retrain_mask_pattern
        self.pre_defined_mask = pre_defined_mask # as model's state_dict
        self.sparsity = self.args.retrain_mask_sparsity
        self.seed = self.args.retrain_mask_seed
        self.sp_mask_update_freq = self.args.sp_mask_update_freq #how frequently the mask is updated
        self.update_init_method = self.args.sp_update_init_method # how the mask is inialized
        self.seq_gap_layer_indices = None

        if logger is None:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
            self.logger = logging.getLogger("pruning")
        else:
            self.logger = logger

        self.logger.info("Command line:")
        self.logger.info(' '.join(sys.argv))
        self.logger.info("Args:")
        self.logger.info(args)

        self.masks = {}
        self.gradient_masks = {}
        self.masked_layers = {}
        self.configs, self.prune_ratios = utils_pr.load_configs(model, args.sp_config_file, self.logger)

        if "masked_layers" in self.configs:
            self.masked_layers = self.configs['masked_layers']
        else:
            for name, W in (self.model.named_parameters()):
                self.masked_layers[utils_pr.canonical_name(name)] = None


        if "fixed_layers" in self.configs:
            self.fixed_layers = self.configs['fixed_layers']
        else:
            self.fixed_layers = None
        self.fixed_layers_save = {}

        if self.args.upper_bound != None:
            self.upper_bound = self.args.upper_bound
            print("!!!!! upper_bound", self.upper_bound)
        else:
            self.upper_bound = None

        if self.args.lower_bound != None:
            self.lower_bound = self.args.lower_bound
            print("!!!!! lower_bound", self.lower_bound)
        else:
            self.lower_bound = None

        if self.args.mask_update_decay_epoch != None:
            self.mask_update_decay_epoch = self.args.mask_update_decay_epoch
        else:
            self.mask_update_decay_epoch = None

        # if "upper_bound" in self.configs:
        #     self.upper_bound = self.configs['upper_bound']
        # else:
        #     self.upper_bound = None
        # if "lower_bound" in self.configs:
        #     self.lower_bound = self.configs['lower_bound']
        # else:
        #     self.lower_bound = None
        # if "mask_update_decay_epoch" in self.configs:
        #     self.mask_update_decay_epoch = self.configs['mask_update_decay_epoch']
        # else:
        #     self.mask_update_decay_epoch = None



        self.init()

    def init(self):

        self.generate_mask(self.pre_defined_mask)


    def apply_masks(self): #this will go through the named masked 0 out some of the weights based on the mask
        with torch.no_grad(): #disabled gradient calculation
            for name, W in (self.model.named_parameters()): # for each name and weight
                if name in self.masks: # if the name is in the weight is in the masked weights
                    dtype = W.dtype # get the datatype of the weight (value win the self.masks[name] will be the weight in the mask after training
                    W.mul_((self.masks[name] != 0).type(dtype))  #this will then multiply the weight elements wise by the binary mask which is created by comparing the value to 0 to 0 out certain values specified in the mask
                    # W.data = (W * (self.masks[name] != 0).type(dtype)).type(dtype)
                    pass

    """
        The gradient of a weight tells how much the loss fn would change if the specific weight were to change         
    """
    def apply_masks_on_grads(self): # this will 0 out some of the gradients of the weights the gradient is the change in all weight in regard in change to error
        with torch.no_grad():
            for name, W in (self.model.named_parameters()): # for  each weight
                if name in self.masks: # if the wieght is in the mask
                    dtype = W.dtype # get teh datatype of the mask
                    (W.grad).mul_((self.masks[name] != 0).type(dtype)) # this will 0 out the gradients  that have a 0 value in the mask
                    pass

    def apply_masks_on_grads_efficient(self): #this will 0 out gradients of weights that are in the gradient mask
        with torch.no_grad():
            for name, W in (self.model.named_parameters()):
                if name in self.gradient_masks:
                    dtype = W.dtype
                    (W.grad).mul_((self.gradient_masks[name] != 0).type(dtype)) #0 out if it is not a 0 in the gradient mask, indicating that it is to be masked out
                    pass

    """
    This function will go through weights and 0 out the gradients that are 0 in the mask in self.maks
    then this will get thet percentile to which gradients should be removed, it will then 0 out gradients that are below the ratio and then finally 
    will create a new mask of the values that are above threshold, meaning that they are important to the current task and save it in the mask dictionary
    """
    def apply_masks_on_grads_mix(self):
        skip = []
        with torch.no_grad(): # set no gradient claculation
            for name, W in (self.model.named_parameters()): # go through the weights and biases for layers and zero out gradietns for masked weights
                if 'norm' in name and name in self.masks:
                    self.masks.pop(name)
                if name in self.masks: # if in masked
                    dtype = W.dtype # get the data type of the weight
                    if(W.grad is None):
                        skip.append(name)
                        continue
                    (W.grad).mul_((self.masks[name] != 0).type(dtype)) # mask out the gradient if not 1 in the mask
            
            for name, W in (self.model.named_parameters()): # for each weight, get the sparsity ratio get the weights and see if the weight meets
                if name not in self.masks or name in skip and 'norm' not in name:  # if not in mask
                    continue
                # cuda_pruned_weights = None
                percent = self.args.gradient_sparse * 100  # the gradient sparsity ratio
                weight_temp = np.abs(W.grad.cpu().detach().numpy( # this will move the gradient tensor to cpu and creates a nertw tensory that is converted to numpy and then gets abs value
                ))  # a buffer that holds weights with absolute values
                percentile = np.percentile( # finds a va;ie where a certain percentage of the gradients are
                    weight_temp,
                    percent)  # get a value for this percentitle
                under_threshold = weight_temp < percentile # check if the weight meets the threshold and generates a boolean array
                above_threshold = weight_temp > percentile # check if above threshold generates a boolean array of those above threshold
                above_threshold = above_threshold.astype(
                    np.float32
                )  # has to convert bool to float32 for numpy-tensor conversion
                # print(name)
                under_threshold = torch.from_numpy(under_threshold)
                W.grad[under_threshold] = 0 # this will zero out the gradients that are under the threshold for gradient sparsity THIS IS WHERE YOU REMOVE SMALLER GRADIENTS

                # gradient = W.grad.data
                # above_threshold, cuda_pruned_gradient = admm.weight_pruning(args, name, gradient, args.gradient_sparse)  # get sparse model in cuda
                # W.grad.data = cuda_pruned_gradient  # replace the data field in variable

                gradient = W.grad.cpu().detach().numpy() # get the gradients put it on cpu and then convert numpy
                non_zeros = gradient != 0 #get a boolean array of non zero gradients
                non_zeros = non_zeros.astype(np.float32) # turn the boolean array to zeros [true, false] -> [1.0,0.0]
                zero_mask = torch.from_numpy(non_zeros).cuda() # move the values indicating that a gradient is activated to gpu this will be the mask for the grads
                self.gradient_masks[name] = zero_mask #set the gradient mask to be the mask indicating that the 1.0 values in the mask are important to the task and should not be masked

    def test_mask_sparsity(self, column=False, channel=False, filter=False, kernel=False):
        
        # --------------------- total sparsity --------------------
        # comp_ratio_list = []

        total_zeros = 0
        total_nonzeros = 0
        layer_cont = 1
        mask = self.gradient_masks
        for name, weight in mask.items(): #go through all the weigts and names in the mask
            zeros = np.sum(weight.cpu().detach().numpy() == 0) #get the number of values in the weight that are 0
            total_zeros += zeros # add to the total zeros
            non_zeros = np.sum(weight.cpu().detach().numpy() != 0) #get thee number of 0s in the weight
            total_nonzeros += non_zeros # add the number of non zeros to the running total
            print("(empty/total) masks of {}({}) is: ({}/{}). irregular sparsity is: {:.4f}".format(
                name, layer_cont, zeros, zeros+non_zeros, zeros / (zeros+non_zeros)))

            layer_cont += 1
        if total_nonzeros == 0:
            comp_ratio = 0.
            total_sparsity = 0
            print("---------------------------------------------------------------------------")
            print("layer does not have values!! skipping")
        else:
            comp_ratio = float((total_zeros + total_nonzeros)) / float(total_nonzeros) if layer_cont > 5 else 0.#get the ratio of non zero to 0
            total_sparsity = total_zeros / (total_zeros + total_nonzeros) if layer_cont > 5 else 0.# get the sparsity ratio

            print("---------------------------------------------------------------------------")
            print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
                total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros))if layer_cont > 5 else 0.)
            print("only consider conv layers, compression rate is: {:.4f}".format(
                (total_zeros + total_nonzeros) / total_nonzeros)) if layer_cont > 5 else 0.
            print("===========================================================================\n\n")

        return comp_ratio, total_sparsity #return the two ratios
        

    def show_masks(self, debug=False):
        with torch.no_grad():
            if debug: # if debugging
                name = 'module.layer1.0.conv1.weight'
                np_mask = self.masks[name].cpu().numpy() # get the masks as a np arraay
                np.set_printoptions(threshold=sys.maxsize) # set print options
                print(np.squeeze(np_mask)[0], name) #squeeze the mask and print hte name
                return
            for name, W in self.model.named_parameters(): # go through the weights
                if name in self.masks: # if the name is in the mask
                    np_mask = self.masks[name].cpu().numpy() # get the mask for the weight
                    np.set_printoptions(threshold=sys.maxsize) # set printing options
                    print(np.squeeze(np_mask)[0], name) #squeeze the mask



    def update_mask(self, epoch, batch_idx): #updating the mask this wil udpate the params of the mask
        # a hacky way to differenate random GaP and others
        if not self.mask_update_decay_epoch: # if there is not a mask update decay epoch we setting this to '5-45'
            return # don't update
        if batch_idx != 0: # if not the first batch
            return

        freq = self.sp_mask_update_freq # get the frequency of sparsity mask frequency (we are using 5)

        bound_index = 0 # starting bound index

        try: # if mask_update_decay_epoch has only one entry
            int(self.mask_update_decay_epoch) #this will fail for '5-45' goto exception
            freq_decay_epoch = int(self.mask_update_decay_epoch)
            try: # if upper/lower bound have only one entry
                float(self.upper_bound)
                float(self.lower_bound)
                upper_bound = [str(self.upper_bound)] #set the upper bound to be one value in an array
                lower_bound = [str(self.lower_bound)] #set the lower bound in an array in itself
                bound_index = 0 # set the bound index
            except ValueError: # if upper/lower bound have multiple entries and cant be converted to float
                upper_bound = self.upper_bound.split('-')  # grow-to sparsity
                lower_bound = self.lower_bound.split('-')  # prune-to sparsity
                if epoch >= freq_decay_epoch: # if the epoch is greater than or equal to the frequencydecay epoch
                    freq *= 1 #dont change the frequency
                    bound_index += 1 #increment the bound_index upper todo bound will say the upper bound to how much update the mask update
        except ValueError: # if mask_update_decay_epoch has multiple entries meaning
            freq_decay_epoch = self.mask_update_decay_epoch.split('-') # split the values
            for i in range(len(freq_decay_epoch)):  # for each frequency decay epoch ex: '5-45' -> 5,45
                freq_decay_epoch[i] = int(freq_decay_epoch[i]) # add the freq decay epoch todo unsure what these values are

            try:
                float(self.upper_bound) #ex: '0.74-0.75-0.75' multiple floats goto exception
                float(self.lower_bound) #'0.75-0.76-0.75'
                upper_bound = [str(self.upper_bound)]
                lower_bound = [str(self.lower_bound)]
                bound_index = 0
            except ValueError:
                upper_bound = self.upper_bound.split('-')  # grow-to sparsity
                lower_bound = self.lower_bound.split('-')  # prune-to sparsity

                if len(freq_decay_epoch) + 1 <= len(upper_bound): # upper/lower bound num entries enough for all update if there are more upper boudns than frequency decay values
                    for decay in freq_decay_epoch:
                        if epoch >= decay: # if the decay is less than or equal to the current epoch
                            freq *= 1
                            bound_index += 1
                else: # upper/lower bound num entries less than update needs, use the last entry to do rest updates
                    for idx, _ in enumerate(upper_bound):
                        if epoch >= freq_decay_epoch[idx] and idx != len(upper_bound) - 1:
                            freq *= 1
                            bound_index += 1

        lower_bound_value = float(lower_bound[bound_index])  # you will now have a lower and upper bound
        upper_bound_value = float(upper_bound[bound_index])

        if epoch % freq == 0: #this will run for the first epoch epoch % 5 ==0?
            '''
            calculate prune_part and grow_part for sequential GaP, if no seq_gap_layer_indices specified in yaml file,
            set prune_part and grow_part to all layer specified in yaml file as random GaP do.
            
            '''
            prune_part, grow_part = self.seq_gap_partition() # get the partition that can be grown and pruned

            with torch.no_grad():
                sorted_to_prune = None
                if self.args.sp_global_magnitude: # false in our case
                    total_size = 0
                    for name, W in (self.model.named_parameters()):
                        if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) \
                                and (name not in self.prune_ratios.keys()):
                            continue
                        total_size += W.data.numel()
                    to_prune = np.zeros(total_size)
                    index = 0
                    for name, W in (self.model.named_parameters()):
                        if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) \
                                and (name not in self.prune_ratios.keys()):
                            continue
                        size = W.data.numel()
                        to_prune[index:(index+size)] = W.data.clone().cpu().view(-1).abs().numpy()
                        index += size
                    sorted_to_prune = np.sort(to_prune)

                # import pdb; pdb.set_trace()
                for name, W in (self.model.named_parameters()): # go through the weights
                    if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) and (name not in self.prune_ratios.keys()): # if the weight is in the prune keys
                        continue

                    weight = W.cpu().detach().numpy() # if so put into cpu
                    weight_current_copy = copy.copy(weight) # copy the weight


                    non_zeros = weight != 0 # set the non zero values to where they are not 0 bool array
                    non_zeros = non_zeros.astype(np.float32) #convert the non_zero values to an array of floats so 1 if true 0 if false
                    num_nonzeros = np.count_nonzero(non_zeros) # get the count of non zero vlaues
                    total_num = non_zeros.size # get the number of values in the weight
                    sparsity = 1 - (num_nonzeros * 1.0) / total_num # use this to calculate sparsity (percentage of weights that are non zero)
                    np_orig_mask = self.masks[name].cpu().detach().numpy() # take the original mask

                    print(("\n==> BEFORE UPDATE: {}: {}, {}, {}".format(name,
                                                                    str(num_nonzeros),
                                                                    str(total_num),
                                                                    str(sparsity))))

                    ############## pruning #############
                    pruned_weight_np = None
                    if name in prune_part: # if the current weight is in the to prune partition
                        sp_admm_sparsity_type_copy = copy.copy(self.args.sp_admm_sparsity_type) #create a copy of the sparsity type we are using 'irregular'
                        sparsity_type_list = (self.args.sp_admm_sparsity_type).split("+") #split on the + if you have multiple types
                        for i in range(len(sparsity_type_list)):
                            sparsity_type = sparsity_type_list[i] # in our case only have one sparsity type of irregular
                            print("* sparsity type {} is {}".format(i, sparsity_type))
                            self.args.sp_admm_sparsity_type = sparsity_type # set the sparsity weight
                            # this will get a pruned mask that indicate values over and under the prune ratio andd the pruned weight
                            pruned_mask, pruned_weight = weight_pruning(self.args, # this will prune the weight using the lower bound value and the sparsity type
                                                                        self.configs, #prune the weight
                                                                        name,
                                                                        W,
                                                                        lower_bound_value) # the lower bound value is used to inform the percentile that every weight must be equal to or above to not be pruned
                            self.args.sp_admm_sparsity_type = sp_admm_sparsity_type_copy # set the sparsity type to be the copy of the copy
                            # pruned_mask_np = pruned_mask.cpu().detach().numpy()
                            pruned_weight_np = pruned_weight.cpu().detach().numpy() # detach the pruned weight that was returned

                            W.mul_(pruned_mask.cuda()) #multiply the weight so that the values above the percentile (0.75) are kept and the others under are removed


                            non_zeros_prune = pruned_weight_np != 0 # get the number of values that are not pruned
                            num_nonzeros_prune = np.count_nonzero(non_zeros_prune.astype(np.float32)) # count the number of non zero values
                            print(("==> PRUNE: {}: {}, {}, {}".format(name, # print the sparsity
                                                             str(num_nonzeros_prune),
                                                             str(total_num),
                                                             str(1 - (num_nonzeros_prune * 1.0) / total_num))))

                            self.masks[name] = pruned_mask.cuda() # udpate the mask to the one that was returned


                            if self.args.gradient_efficient: # in our case this is not used
                                new_lower_bound_value = lower_bound_value + self.args.gradient_remove
                                pruned_mask, pruned_weight = weight_pruning(self.args,
                                                                            self.configs,
                                                                            name,
                                                                            W,
                                                                            new_lower_bound_value)
                                self.gradient_masks[name] = pruned_mask.cuda()


                    ############## growing #############
                    if name in grow_part: # if the weight is in the growing part
                        if pruned_weight_np is None: # use in seq gap we have a pruned_weight_np in epoch 0
                            pruned_weight_np = weight_current_copy # set a pruned weight if no pruning happened for the weight

                        updated_mask = weight_growing(self.args,
                                                      name,
                                                      pruned_weight_np,
                                                      lower_bound_value,
                                                      upper_bound_value,
                                                      self.update_init_method)
                        self.masks[name] = updated_mask # set the updated mask to be new mask with the grown indicies added in
                        pass



    def cut_all_partitions(self, all_update_layer_name):
        # calculate the number of partitions and range
        temp1 = str(self.seq_gap_layer_indices)
        temp1 = (temp1).split('-')
        num_partition = len(temp1) + 1
        head = 0
        end = len(all_update_layer_name)
        all_range = []

        for i, indice in enumerate(temp1):
            assert int(indice) < end, "\n\n * Error, seq_gap_layer_indices must within range [0, {}]".format(end - 1)
        assert len(temp1) == len(set(temp1)), "\n\n * Error, seq_gap_layer_indices can not have duplicate element"

        for i in range(0, num_partition):
            if i == 0:
                range_i = (head, int(temp1[i]))
            elif i == num_partition - 1:
                range_i = (int(temp1[i - 1]), end)
            else:
                range_i = (int(temp1[i - 1]), int(temp1[i]))
            print(range_i)
            all_range.append(range_i)

        for j in range(num_partition):
            range_j = all_range[j]
            self.all_part_name_list.append(all_update_layer_name[range_j[0]:range_j[1]])

    def seq_gap_partition(self):
        prune_part = []
        grow_part = []

        if self.seq_gap_layer_indices is None: # Random Gap: add all layer name in prune part and grow part list
            for name, _ in self.model.named_parameters(): # go through the name parameters
                if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) and (name not in self.prune_ratios.keys()): # if the name is not in the prune keys then you will skip it
                    continue # this will see if the value is in the prune values which is specified in the yaml file
                prune_part.append(name) # if the name is in the prune keyts add to prune
                grow_part.append(name)  # add to the grow partition as well
        else: # Sequential gap One-run: partition model
            all_update_layer_name = []
            for name, _ in self.model.named_parameters(): # go thorugh all values j
                if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) and (name not in self.prune_ratios.keys()):
                    continue
                all_update_layer_name.append(name)
            if not self.all_part_name_list:
                self.cut_all_partitions(all_update_layer_name) # get all partitions by name in self.all_part_name_list

            to_grow = (self.all_part_name_list).pop(0)
            to_prune = self.all_part_name_list

            for layer in to_grow:
                grow_part.append(layer)
            for part in to_prune:
                for layer in part:
                    prune_part.append(layer)

            (self.all_part_name_list).append(to_grow)

        return prune_part, grow_part # this will return the layers that can be prune and grown



    def generate_mask(self, pre_defined_mask=None):
        masks = {}
        # import pdb; pdb.set_trace()
        if self.pattern == 'weight': # if the pattern is off weight


            with torch.no_grad(): #disasble gradients
                for name, W in (self.model.named_parameters()): # for each weight

                    if (utils_pr.canonical_name(name) not in self.masked_layers) and (name not in self.masked_layers): # skip of tthe name is not in the masked layeres
                        continue

                    weight = W.cpu().detach().numpy() # get the weight tensor
                    non_zeros = weight != 0 # gett the number of non-zero values in the weight
                    non_zeros = non_zeros.astype(np.float32)  # turn the boolean array to a float array
                    num_nonzeros = np.count_nonzero(non_zeros) # get the count of non zero values int the float array
                    total_num = non_zeros.size # get the number of zeros
                    sparsity = 1 - (num_nonzeros * 1.0) / total_num # calcualte the sparsity
                    #self.logger.info("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))
                    print(("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity))))
                    if sparsity < 0.1:
                        #self.logger.info("{}: sparsity too low, skip".format(name))
                        print("{}: sparsity too low, skip".format(name))
                        continue
                    zero_mask = torch.from_numpy(non_zeros).cuda() # set the non zeros to gpu

                    self.masks[name] = zero_mask # set the mask for the weight 

            #for name in masks:
            #    print("Current mask includes:", name)
                    #if 'weight' in name:
                    #    print(name, (np.sum(non_zeros) + 0.0) / np.size(non_zeros) )
                #exit()



        elif self.pattern == 'random':
            if self.seed is not None:
                print("Setting the random mask seed as {}".format(self.seed))
                np.random.seed(self.seed)

            with torch.no_grad():
                # self.sparsity (args.retrain_mask_sparsity) will override prune ratio config file
                if self.sparsity > 0:
                    sparsity = self.sparsity

                    for name, W in (self.model.named_parameters()):
                        if 'weight' in name and 'bn' not in name:
                            non_zeros = np.zeros(W.data.shape).flatten()
                            non_zeros[:int(non_zeros.size*(1-sparsity))] = 1

                            np.random.shuffle(non_zeros)

                            non_zeros = np.reshape(non_zeros, W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()
                        else:
                            non_zeros = np.ones(W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()
                        self.masks[name] = zero_mask

                else: #self.sparsity < 0

                    for name, W in (self.model.named_parameters()):
                        if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) \
                                and (name not in self.prune_ratios.keys()):
                            continue
                        if name in self.prune_ratios:
                            # Use prune_ratio[] to indicate which layers to random masked
                            sparsity = self.prune_ratios[name]
                            '''
                            if sparsity < 0.001:
                                continue
                            '''
                            non_zeros = np.zeros(W.data.shape).flatten()
                            non_zeros[:int(non_zeros.size*(1-sparsity))] = 1

                            np.random.shuffle(non_zeros)

                            non_zeros = np.reshape(non_zeros, W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()
                        else:
                            non_zeros = np.ones(W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()

                        self.masks[name] = zero_mask

                # # DEBUG:
                DEBUG = False
                if DEBUG:
                    for name, W in (self.model.named_parameters()):
                        m = self.masks[name].detach().cpu().numpy()
                        total_ones = np.sum(m)
                        total_size = np.size(m)
                        print( name, m.shape, (total_ones+0.0)/total_size)

                #exit()
        # TO DO
        elif self.pattern == 'regular':
            with torch.no_grad():
                for name, W in self.model.named_parameters():
                    if 'weight' in name and 'bn' not in name:

                        ouputSize, inputSize = W.data.shape[0], W.data.shape[1]
                        non_zeros = np.zeros(W.data.shape)
                        non_zeros = np.squeeze(non_zeros)

                        if 'sa1.conv_blocks.0.0.weight' in name or 'sa1.conv_blocks.1.0.weight' in name or 'sa1.conv_blocks.2.0.weight' in name:
                            non_zeros[::self.args.mask_sample_rate,::] = 1

                        else:
                            non_zeros[::self.args.mask_sample_rate,::self.args.mask_sample_rate] = 1

                        non_zeros = np.reshape(non_zeros, W.data.shape)
                        non_zeros = non_zeros.astype(np.float32)
                        zero_mask = torch.from_numpy(non_zeros).cuda()

                    else:
                        non_zeros = 1 - np.zeros(W.data.shape)
                        non_zeros = non_zeros.astype(np.float32)
                        zero_mask = torch.from_numpy(non_zeros).cuda()
                    self.masks[name] = zero_mask
        elif self.pattern == 'global_weight':
            with torch.no_grad():
                all_w = []
                all_name = []
                print('Concatenating all weights...')
                for name, W in self.model.named_parameters():
                    if (utils_pr.canonical_name(name) not in self.prune_ratios) and (name not in self.prune_ratios):
                        continue
                    all_w.append(W.detach().cpu().numpy().flatten())
                    all_name.append(name)
                np_w = all_w[0]
                for i in range(1,len(all_w)):
                    np_w = np.append(np_w, all_w[i])

                #print(np_w.shape)
                print("All weights concatenated!")
                print("Start sorting all the weights...")
                np_w = np.sort(np.abs(np_w))
                print("Sort done!")
                L = len(np_w)
                #print(np_w)
                if self.args.retrain_mask_sparsity >= 0.0:
                    thr = np_w[int(L * self.args.retrain_mask_sparsity)]

                    for name, W in self.model.named_parameters():
                        if (utils_pr.canonical_name(name) not in self.prune_ratios) and (name not in self.prune_ratios):
                            continue


                        np_mask = np.abs(W.detach().cpu().numpy())  > thr
                        print(name, np.size(np_mask), np.sum(np_mask), float(np.sum(np_mask))/np.size(np_mask) )

                        self.masks[name] = torch.from_numpy(np_mask).cuda()

                    total_non_zero = 0
                    total_size = 0
                    with open('gw_sparsity.txt','w') as f:
                        for name, W in sorted(self.model.named_parameters()):
                            if (utils_pr.canonical_name(name) not in self.prune_ratios) and (name not in self.prune_ratios):
                                continue
                            np_mask = self.masks[name].detach().cpu().numpy()
                            sparsity = 1.0 - float(np.sum(np_mask))/np.size(np_mask)
                            if sparsity < 0.5:
                                sparsity = 0.0

                            if sparsity < 0.5:
                                total_non_zero += np.size(np_mask)
                            else:
                                total_non_zero += np.sum(np_mask)
                            total_size += np.size(np_mask)

                            f.write("{}: {}\n".format(name,sparsity))
                    print("Thr:{}".format(thr))
                    print("{},{},{}".format(total_non_zero, total_size, float(total_non_zero)/total_size))
                    exit()



        elif self.pattern == 'none':
            with torch.no_grad():
                for name, W in self.model.named_parameters():
                    non_zeros = np.ones(W.data.shape)
                    non_zeros = non_zeros.astype(np.float32)
                    zero_mask = torch.from_numpy(non_zeros).cuda()
            self.masks[name] = zero_mask

        elif self.pattern == "pre_defined":
            assert pre_defined_mask is not None, "\n\n * Error, pre_defined sparse mask model must be declared!"
            with torch.no_grad():
                for name, W in pre_defined_mask.items():
                    if (utils_pr.canonical_name(name) not in self.masked_layers) and (name not in self.masked_layers):
                        continue

                    weight = W.cpu().detach().numpy()
                    non_zeros = weight != 0
                    non_zeros = non_zeros.astype(np.float32)
                    num_nonzeros = np.count_nonzero(non_zeros)
                    total_num = non_zeros.size
                    sparsity = 1 - (num_nonzeros * 1.0) / total_num
                    #self.logger.info("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))
                    print(("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity))))
                    if sparsity < 0.1:
                        #self.logger.info("{}: sparsity too low, skip".format(name))
                        print("{}: sparsity too low, skip".format(name))
                        continue
                    zero_mask = torch.from_numpy(non_zeros).cuda()

                    self.masks[name] = zero_mask

        else:
            print("mask pattern not recognized!")
            exit()

        return self.masks
