import numpy as np
from matplotlib import pyplot as plt


def visualize_training(progress_dict: dict):
    # get the training from a dictionary
    training = progress_dict['training']
    for task in training.keys():
        # get the progress for each task
        progress = training[task]
        # plot the progress
        plt.plot(progress, label='task {}'.format(task))
    plt.legend()
    plt.savefig('training_progress.png')
    plt.show()





if __name__ == '__main__':
    # load the progress dictionary
    progress_dict = {}
    for i in range(3):
        progress_dict[i] = np.arange(0,90,10) / (i+1)
    dict_value = {'training': progress_dict}
    visualize_training(dict_value)