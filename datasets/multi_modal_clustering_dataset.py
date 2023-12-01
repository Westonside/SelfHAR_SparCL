from datasets import ContinualDataset
import hickle as hkl

class MultiModalClusteringDataset(ContinualDataset):
    NAME = 'multi_modal_clustering_dataset'
    N_CLASSES_PER_TASK = 2
    TOTAL_CLASSES = None
    def __init__(self, path):
        data = hkl.load(path)
        self.train_data, self.train_labels = data['train_data']
        self.test_data, self.test_labels = data['testing_data']