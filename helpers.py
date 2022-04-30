from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import gzip
import numpy as np


class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.shape[0] / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        
        s = StratifiedShuffleSplit(n_splits = self.n_splits, test_size = 0.5)
        X = torch.randn(self.class_vector.shape[0],2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)

    
class MnistDataset(Dataset):
    
    def __init__(self, image_path, label_path):
    
        # Read images and labels
        X = MnistDataset.read_images(path = image_path) / 255
        y = MnistDataset.read_labels(path = label_path)
        
        self.images = torch.tensor(X).float()
        self.labels = torch.tensor(y)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx, :, :], self.labels[idx]
    
    @staticmethod
    def read_images(path):
        ''' Read all images '''
        # Credits: https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python

        with gzip.open(path, 'r') as f:

            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')

            # second 4 bytes is the number of images
            image_count = int.from_bytes(f.read(4), 'big')

            # third 4 bytes is the row count
            row_count = int.from_bytes(f.read(4), 'big')

            # fourth 4 bytes is the column count
            column_count = int.from_bytes(f.read(4), 'big')

            # rest is the image pixel data, each pixel is stored as an unsigned byte
            # pixel values are 0 to 255
            image_data = f.read()
            images     = np.frombuffer(image_data, dtype = np.uint8).reshape((image_count, row_count, column_count))

            return images
    
    @staticmethod
    def read_labels(path):
        ''' Read all labels '''
        # Credits: https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python

        with gzip.open(path, 'r') as f:

            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')

            # second 4 bytes is the number of labels
            label_count = int.from_bytes(f.read(4), 'big')

            # rest is the label data, each label is stored as unsigned byte
            # label values are 0 to 9
            label_data = f.read()
            labels     = np.frombuffer(label_data, dtype = np.uint8)

            return labels