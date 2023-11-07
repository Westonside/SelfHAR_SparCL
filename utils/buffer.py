# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from typing import Tuple
from torchvision import transforms


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size: # if the buffer is not full it will incrementally add values into the buffer
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1) # get an index in between 0 and sthe seen examples
    if rand < buffer_size:
        return rand
    else:
        return -1 # return -1 if this is above the buffer size


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size



class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir', 'herding'] #ensure one of the correct buffer modes
        self.buffer_size = buffer_size #set the buffer size
        self.device = device #set the dvice
        self.num_seen_examples = 0 #default the seen examples
        if mode == 'herding':
            self.functional_index = None
        else:
            self.functional_index = eval(mode) #this evaluates an expression if it is legal python then it will execute the python code passed in

        if mode == 'ring': #if ring mode j
            assert n_tasks is not None #assert there are a certain nubmer of tasks
            self.task_number = n_tasks #set the number of tasks
            self.buffer_portion_size = buffer_size // n_tasks #divide the buffer up into tasks
        if mode == 'herding': #if herding mode
            self.seen_classes = 0 #set the seen classes to 0

        self.attributes = ['examples', 'labels', 'logits', 'task_labels'] # set the attributes of the buffer

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes: # for all the attributes
            attr = eval(attr_str) #this will set attribute to the string passed in and run as python
            if attr is not None and not hasattr(self, attr_str): # if the attribute has a value and the bugffer has the attribute
                typ = torch.int64 if attr_str.endswith('els') else torch.float32 # create a multi-dimensional(tensor) of either int 64 or float 32
                setattr(self, attr_str, torch.zeros((self.buffer_size, #set the attribute of the buffer to be aatribute = a buffer of 0s with size buffer size and a shape of the attribute shape (skipping the baetch size)
                        *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'): # if the buffer does not the attribute of storing eamples
            self.init_tensors(examples, labels, logits, task_labels) #create a new tensor for the examles

        for i in range(examples.shape[0]):  # go through all examples that were in the batch so go through the 32 examples
            index = reservoir(self.num_seen_examples, self.buffer_size)  # get the index using the resevoir algo
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)



    def get_data(self, size: int, transform: transforms=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]): #if the size(batch) is greater than the number of inputs seen or inputs in the buffer
            size = min(self.num_seen_examples, self.examples.shape[0]) # set the size to be what is in the buffer

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]), # you will then choose random values in the buffer
                                  size=size, replace=False)
        if transform is None: transform = lambda x: x # if there is not a transformation is the identity function
        ret_tuple = (torch.stack([transform(ee.cpu()) # this will then stack all examples to a list and apply the transformation
                            for ee in self.examples[choice]]).to(self.device),) # gets the selected sampels
        for attr_str in self.attributes[1:]: # this will add teh desired attributes that you want in our buffer so in our case ['labels', 'logits', 'task_labels']
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0


    def fill_buffer(self, model, dataset, task: int):
        model.eval()

        samples_per_class = self.buffer_size // (dataset.N_CLASSES_PER_TASK*(task+1)) # get the number of samples per class
        # if you are in the first task then you will fill the buffer with the first task only
        if task > 0:
            buf_x, buf_y, buf_l = self.get_all_data(None) # get all data from the buffer in format (input, label, logits)
            self.empty() # empty the buffer
            for _y in buf_y.unique(): # for all the unique classes in the buffer
               # get the indicies of where the label is equal to the current label
               idx = (buf_y == _y)
               _y_x, _y_y, _y_l = buf_x[idx], buf_y[idx], buf_l[idx] # get the inputs, labels and logits
               self.add_data(
                    examples=_y_x[:samples_per_class],
                    labels=_y_y[:samples_per_class],
                    logits=_y_l[:samples_per_class]
               )

            # 2) Then, fill with current tasks
        loader = dataset.train_loader  # get the training loader
        norm_trans = dataset.get_normalization_transform()  # get the normalization transform
        if norm_trans is None:
            def norm_trans(x): return x  # identity function if no normalization transform
        classes_start, classes_end = task* dataset.N_CLASSES_PER_TASK, (
                    task+ 1) * dataset.N_CLASSES_PER_TASK  # get the start and end of the classes

        # 2.1 Extract all features
        a_x, a_y, a_f, a_l = [], [], [], []  # inputs labels features and logits
        for x, y, not_norm_x in loader:  # for the input, label and non normalized input in the loader
            mask = (y >= classes_start) & (y < classes_end)  # get the mask of the labels that are in the current task
            x, y, not_norm_x = x[mask], y[mask], not_norm_x[
                mask]  # get the inputs, labels and non normalized inputs that are in the current task
            if not x.size(0):
                continue
            x, y, not_norm_x = (a.to(self.device) for a in
                                (x, y, not_norm_x))  # move the input label and non normalized input to the device
            a_x.append(not_norm_x.to('cpu'))  # append the non normalized input to the cpu
            a_y.append(y.to('cpu'))  # append the label to the cpu
            feats = x
            with torch.no_grad():
                outs = model(feats)
            a_f.append(feats.cpu())  # append the features to the cpu
            a_l.append(torch.sigmoid(outs).cpu())  # append the outputs to the cpu andn apply sigmoid
        a_x, a_y, a_f, a_l = torch.cat(a_x), torch.cat(a_y), torch.cat(a_f), torch.cat(
            a_l)  # concatenate all the inputs, labels, features and outputs

        #     feats = self.net(norm_trans(not_norm_x), returnt='features')  # get the features from the network
        #     outs = self.net.classifier(feats)  # get the outputs from the network
        #     a_f.append(feats.cpu())  # append the features to the cpu
        #     a_l.append(torch.sigmoid(outs).cpu())  # append the outputs to the cpu andn apply sigmoid
        # a_x, a_y, a_f, a_l = torch.cat(a_x), torch.cat(a_y), torch.cat(a_f), torch.cat(
        #     a_l)  # concatenate all the inputs, labels, features and outputs
        # skip this stage because have pre extracted features

        # 2.2 Compute class means for each class in the labels
        for _y in a_y.unique():  # this will look to find the most representative examples for each class
            idx = (a_y == _y)  # get the indicies of the labels that are equal to the current label
            _x, _y, _l = a_x[idx], a_y[idx], a_l[idx]  # get the inputs, labels and logits for the current class
            feats = a_f[idx]  # get the features
            mean_feat = feats.mean(0, keepdim=True)  # get the mean of the features

            running_sum = torch.zeros_like(
                mean_feat)  # create a running sum of zeros with the same shape as the mean features
            i = 0
            while i < samples_per_class and i < feats.shape[
                0]:  # while the number of samples per class is less than the number of features
                cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, # calculate the difference the mean features and all features and get the smallest diff
                                                                          1)  # calculate the cost which is the norm of the mean features minus the features plus the running sum divided by i+1

                idx_min = cost.argmin().item()  # get the index of the minimum cost

                self.add_data(
                    examples=_x[idx_min:idx_min + 1].to(self.device),
                    labels=_y[idx_min:idx_min + 1].to(self.device),
                    logits=_l[idx_min:idx_min + 1].to(self.device)
                )

                running_sum += feats[idx_min:idx_min + 1]
                feats[idx_min] = feats[idx_min] + 1e6
                i += 1

        # # assert len(mem_buffer.examples) <= mem_buffer.buffer_size
        # # assert mem_buffer.num_seen_examples <= mem_buffer.buffer_size
        #
        # self.net.train(mode)
