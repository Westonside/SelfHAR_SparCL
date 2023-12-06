import numpy as np

"""
 This function calculates the f1 macro and micro scores for a given confusion matrix
 The confusion matrix must be a nxn matrix where n is the number of classes
"""
def calculate_f1_scores(confusion_matrix):
    # the true positives are the diagonal elements
    true_positives = np.diag(confusion_matrix) # get the correct predictions
    # the false positives are when the model predicted the class was the class so sum the rows
    false_positives = np.sum(confusion_matrix, axis=0) - true_positives # get the false positives
    # the false negatives are when the model predicted the class was not the class so sum the columns to show guessing other classes
    false_negatives = np.sum(confusion_matrix, axis=1) - true_positives # get the false negatives

    # the precision is the true positives / true positives + false positives
    precision = np.nan_to_num(true_positives / (true_positives + false_positives))
    # the recall is the true positives / true positives + false negatives
    recall = np.nan_to_num(true_positives / (true_positives + false_negatives))
    # the f1 score is the harmonic mean of the precision and recall
    f1 = 2 * precision * recall / (precision + recall)

    # the micro f1 score is the sum of the true positives / sum of the true positives + false positives
    micro_precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))
    # the micro recall is the sum of the true positives / sum of the true positives + false negatives
    micro_recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))
    # the micro f1 score is the harmonic mean of the precision and recall
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    #the macro f1 score is the average of the f1 scores across all class
    macro_f1 = np.nanmean(f1)

    return micro_f1, macro_f1


def testing_f1():
    confusion = np.eye(3, dtype=np.uint64) * 20
    for col in range(len(confusion)):
        for row in range(confusion.shape[1]):
            if col != row:
                confusion[col][row] = 4

    print(confusion)
    print(calculate_f1_scores(confusion))

if __name__ == '__main__':
    testing_f1()