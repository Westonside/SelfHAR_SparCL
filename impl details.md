#Design Description

## Pruning and Growing with mask updates in Epoch
The weights will have a lower and upper bound that is used when selecting weights to be pruned or grown back
For every layer this function will be run
- Weights will be added to a grow and prune array
- To start the function will get the before update sparsity which will be calcualted by getting the number non zero values / total number of values to get the start
- Pruning will use a sparsity type(irregular in our case) the weight will be taken and a prune ratio which is the lower bound variable

In this pruning approach, if the weight has a gradient, then it will also get the gradient. If using the irregular sparsity a new tensor will be calculated to
be either the absolute values of the weights or the absolute values + the gradient * a args.sp_ldm which tells how much factor the gradient should have

When the abs of the weights is taken (+ grad)? then the weights that are over the prune ratio will be kept and the values under will be zeroed out and the mask will be updated to contain
the values that are zero in the mask

example run of the prune: tensor a = [1,2,3,45,0,0,0.2], upper_value = 0.75 this will get the values in the 75th percentile and zero out everything below that value
it will then return: the mask containing values above the threshold (new pruned mask)  new tensor with mask applied representing the mask

Why the absolute value of the weights? This allows the values of the weights to be captured by magnitude

- Then the weight will be grown since it was added into the grow partition

In the grow function this will take in the upper and lower bound values along with the now pruned weight
if there is irregular sparsity (true in our case) this will
make the weight 1D and then it will calcualte the nubmer of zeros that should be grown back to reach the desired sparsity
this is calculated by: (number of zero values) - upper bound * total number of values
This will get the difference between the total zeros and the number of zeros that are wanted and the difference will tell you how many values should not be 0
if the value is negative, meaning that you have less zeros than the target then 0 will be added.

Then it will select the number of values that are 0 in the tensor specified by  (number of zero values) - upper bound(sparsity value) * total number of values
which represents the number of values that are needed to reach the desired sparsity value. That many values will be selected at random
and set them to be -1 and then update the mask to have all values that are not 0 and then the -1 values will be set back to 1.
The purpose of that was to update the mask and then the updated grown mask will be returned

What does the value of upper_bound_value represent? The desired sparsity level where 0.74 means 74% of all weights should be zero

## Training
- The function will divide the data randomly into permutations that will be used to be the batches, the batches will be of the format 32x96 32 being batch size
- Then the function will check if the buffer is not empty and if it is not the first task (because there will be nothing in the buffer except its own data)
- This will then predict on the inputs and mask out the classes not used in the current task using the cl_mask and then it will predict the cross entropy loss
- Check the replay method, we are using derpp(der++)
This is where the continual learning mechanism comes in which is through the use of DER++ that uses a replay mechanism to achieve the continual learning
### DERPP Mechanism

How does the buffer get values? It will get batch size values by randomly selecting size samples(32 in our case) in the buffer indices for buffer samples
the buffer will then apply a transformation on the values. Then the desired attributes will be added to the stored buffer input values ['labels', 'logits', 'task_labels']
this tells that it wants labels, logits and task labels that is characterized by DER++.

You will get the past inputs and logits, the goal will be to have similar logits to past predictions for the same input. It is important to note that the
buffer will apply transformations to the input data to encourage better feature extraction.

This logit similarity is then put into a loss function and the mean loss for all the buffer input and output loss will be calculated. This
encourages the model to predict similarly to past tasks allowing to fine tune to past tasks. The mean loss will be multiplied by an arg buffer wight value
this buffer weight values how much we should care about accuractely predicting close to past predictions in our case we use 0.1.

Then you will get more data from the buffer, this time you will get the buffer input values and the class labels to go with it, still applying transformations.
This is characteristic of derpp because it uses this step. You will then predict on the fetched buffer examples and then calcualte the loss between the outputs and
the classification of the past input. You will then multiply the classification loss by a beta value you can set to specify how much you care about correct classifications on
past tasks (in our case 0.5).

These losses will be added to the ce_loss
order of training goes :
Predict(inputs) -> get loss using outputs -> go in buffer -> get past inputs and logits -> predict on buffer inps -> loss on diff(out_logits, bufferlogits) * ∂ ->
get inputs and class labels from buff -> predict on input -> loss based on(class_pred, buf_labels) * ß

It is important to note that the loss will then be formatted as the loss per item in the batch

- Now you will get the accurate predictions and you will go through each batch, getting the input value from the batch and finding it in the original dataset
- Get the correct class and if you had correct prediciton or not then you will sort the values to get the most incorrect class in your predictions
- Caculate the margin between correct predictions and incorrect

for a given value in the batch you will sort the outputs and then you will get the most incorrectly predicted class. In our case this would mean
that if the prediction was correct for the sample then you would get the second most predicted, if you were wrong then the most incorrect class for the
batch would be the last item since its sorted.

Note for stats saving: This is all done by going through all the samples in the current batch. You will calculate the most
incorrectly predicted class per sample in the current batch

- Once you have done all these steps for a given batch
- Take the mean of loss for the batches
- Add the correct preds
- Back propagate the loss
- Now the model that is being used is gradient efficient mixed
- If the batch id(0-# num batches) % sample freq(which is set 30 in our case) == 0 then you will prune and apply masks on gradient on mask
- Else if the batch id % sample freq != 0 then you will apply masks on grads efficient

### prune apply masks on grad mix  case 1
Goal: Remove the gradients for weights that have been masked also mask off gradients that are below a given percentile(our case 80th)
this reduces the computation cost by keeping only the most important gradients per layer's weights, achieving gradient sparsity. This performed per layer and then a new mask will be returned after all layers have been
done the mask will be indicating non zero gradients at that layer.

You will go through the weights and if the weight is to be masked you will then get all the values in the mask that have been selected to be turned off
this is done by checking if the value in the mask is 0 or 1 and then multiply the weight's gradients by this this 0 or 1 tensor indicating if the weight
is masked or not, zeroing off the gradients. The values that are not 0 will then be the updated gradient mask that will be set in teh dictionary for that layer


Every 5 batches this will grad mix function will be applied, updating the gradient mask
What is the purpose of the mask?



### prune apply mask on grad efficient
apply_masks_on_grads_efficient will use the gradient mask obtained from apply mask grad mix and then zero out the gradietns that were masked in that function


How this Pruning works: The apply mask grad mix will zero out gradients on disabled weights and then disable the gradietns below the 80th percentile this will then form the new gradient mask which indicates that non zero gradients after applying prune.
Then in the next 5 batches, the prune apply mask grad efficient will use the gradient mask made in the apply mask grad mix and zero out those gradients.


- Then the prune apply masks will be called where it will apply the mask to the weights and zero out masks that are not in the mask


- then data will be addd to the buffer using derpp stratgegy
### buffer adding
it will taek the non transformed data, the labels and the logits for the current batch
Using resevoir sampling it will select an index to get inde (uses random selection with unknown len)
if the index is over buffer size nothing will be added but if it does get something from the batch then it will add the labels, logits and class label to the buffer
this means nothing could be added from the batch too


summary: this will go through the entries and using resevoir sampling will add to the buffer if it can generate a random number that
falls within the buffer size randint(0, seen_examples +1 ) if that is in buffer size then the input would go there with logits, label and input.

- then if the batch is divisible by 10 then set param group lr
- print the epoch stats


### Training summary
For each batch you will predict on the current batch and if it is not task 0 then you will predict on past input. There will be loss trying to match
past logits with current logits on a buffered past class. There will also be loss trying to correctly predict past classes. Then for each item in batch will find
the most incorrectly predicted class using batch indicies to pointer to original dataset and then you will save this misprediction stats and most incorrect class to a dict.
ex: train -> loss -> backprop -> repeat



NOTE for DERPP: Reference the DERPP screenshot and labelled pdf to learn about the specifics
- printing the sparsity for all parameters
-test the mask sparsity which
- every args.test_epoch_interval or every few epochs the accuracy for task incremental learning will be tested


## Testing
The reason that the testing accuracy was the same for the step before is because the training will run an evaluation function that uses
the testing seat at the last task
# Impl Notes
## Training
- For every step of training you have to zero_grad() your optimizer - learn why
- After you calculate the loss you run loss.backward()
- Pytorch has a scheduler that is used to find the best learning rate, the scheduler will look at the learning rate throughout training, updating at a specified number of steps)
- A step in the case of a scheduler is an iteration or an epoch in the training process
- The validation set is used to validate the model during the training stage, gives information on how to best update hyper params
- During training model will also be using validation, weights will not be use


## stats
training with gyro test:
Task 0, Average loss 0.0004, Class inc Accuracy 85.269, Task inc Accuracy 86.407
Task 1, Average loss 0.0000, Class inc Accuracy 90.249, Task inc Accuracy 98.853
Task 2, Average loss 0.0001, Class inc Accuracy 79.869, Task inc Accuracy 99.509
much worse than the accel

here is with the accel:

Task 0, Average loss 0.0000, Class inc Accuracy 98.131, Task inc Accuracy 99.291
Task 1, Average loss 0.0001, Class inc Accuracy 93.805, Task inc Accuracy 98.686
Task 2, Average loss 0.0001, Class inc Accuracy 87.263, Task inc Accuracy 99.621


final accel:

Task 0, Average loss 0.0001, Class inc Accuracy 99.076, Task inc Accuracy 99.569
Task 1, Average loss 0.0000, Class inc Accuracy 94.192, Task inc Accuracy 98.232
Task 2, Average loss 0.0003, Class inc Accuracy 86.360, Task inc Accuracy 99.841
