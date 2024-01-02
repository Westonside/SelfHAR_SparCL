
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
def main():
    # load the data
    validation_folder = 'validations'
    validation_file = '../SHL_fiverun_mean_validation_simple.pkl'
    validation_path = validation_file
    with open(validation_path, 'rb') as f:
        data = pickle.load(f)
    fig, axes = plt.subplots(len(data.keys()), 2, figsize=(10, 5 * len(data.keys())))
    # set the title to be Loss and Accuracy for the whole plot
    fig.suptitle('Loss and Accuracy for each task and overall')

    valid_acc = []
    for task in data.keys():
        # go through epoch
        # print(data[task][0]['overall'])

        # get the validation loss for each task
        valid_loss = []
        task_valid_acc = []
        for validation_task in range(len(data[task][0]['individual'])):
            print(validation_task)
            # this is getting all the values for one first which makes sense
            print([data[task] for i in data[task].keys()])
            all_valid_task_loss = [data[task][i]['individual'][validation_task]['task_validation_loss'] for i in
                                   data[task].keys()]
            all_valid_task_acc = [data[task][i]['individual'][validation_task]['task_accuracy'] for i in
                                  data[task].keys()]
            # overall accuracy
            total_valid_task_loss = [data[task][i]['overall']['total_validation_accuracy'] for i in data[task].keys()]
            # overall loss
            total_valid_task_acc = [data[task][i]['overall']['total_loss'] for i in data[task].keys()]
            axes[task, 0].plot(range(len(data[task].keys())), all_valid_task_acc,
                               label=f"Accuracy for t:{validation_task} in t:{task}")
            axes[task, 0].plot(range(len(data[task].keys())), total_valid_task_loss,
                               label=f"Overall Accuracy task {validation_task} in t:{task}")
            axes[task, 0].plot()
            axes[task, 0].legend()

            axes[task, 1].plot(range(len(data[task].keys())), all_valid_task_loss,
                               label=f"Loss t:{validation_task} in t:{task}")
            axes[task, 1].plot(range(len(data[task].keys())), total_valid_task_acc,
                               label=f"Overall Loss t:{validation_task} in t:{task}")
            axes[task, 1].legend()
            # move the legend so it does not obscure the plot to the top
            # axes[task,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
            #   ncol=3, fancybox=True, shadow=True)

            # add gridlines to both
            axes[task, 0].grid()
            axes[task, 1].grid()
            # add x label of epoch
            axes[task, 0].set_xlabel('Epoch')
            axes[task, 1].set_xlabel('Epoch')
            # add y label of accuracy for the first one
            axes[task, 0].set_ylabel('Accuracy')
            axes[task, 1].set_ylabel('Loss')

            task_valid_acc.append(all_valid_task_loss)

        valid_acc.append(task_valid_acc)

    # print(valid_acc)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    output_dir = "../plots"
    os.makedirs(f"{output_dir}", exist_ok=True)
    output_path = f"{output_dir}/plot_{validation_file[validation_file.find('/')+1:validation_file.find('_')]}.png"
    print("Saving plot to:", output_path)

    try:
        plt.savefig(output_path)
        print("Save successful!")
    except Exception as e:
        print("Error:", e)


    plt.show()
    # save the plot as an image to the folder plots and have the name be in the format plot_<FileName>.png
    # plt.savefig(f"plots/plot_{validation_file.split('.')[0]}.png")


if __name__ == '__main__':
    main()