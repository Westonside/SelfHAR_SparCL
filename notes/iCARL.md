# ICARL
The goal of iCARL is to provide a means of incremental learning where tasks can progressively be added over time without having to have all 
your data in one place


### Mean of exemplars
### Herding
Items are added to the memory buffer at the end of each task. The buffer is configured by Buffer Size / (seen classes) for each.
These items are added at the end of the task by going through the training set and calculating the mean of exemplars for 
each class that was in the task. The items that are closest to the class mean are stored and the buffer will be ordered by importance.
It is important to note that before classes are removed by choosing only choosing the buffer size/ seen classes and adding the new tasks in, the first m classes were be picked in reduce exemplar set




### Classification