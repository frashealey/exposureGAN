import os
import numpy as np
from skimage import io
from torch.utils.data import Dataset


class ExposureDataset(Dataset):
    def __init__(self, input_dir, truth_dir, transform_input=None, transform_truth=None):
        # defines transforms and directories for inputs and ground truths
        self.transform_input = transform_input
        self.transform_truth = transform_truth

        # creates array of paired inputs and ground truths
        inputs = np.array(os.listdir(input_dir))
        # (each ground truth repeated 5x to match with inputs)
        truths = np.repeat(np.array(os.listdir(truth_dir)), 5)
        self.filenames = np.stack((np.char.add(np.full(inputs.shape, input_dir), inputs), np.char.add(np.full(truths.shape, truth_dir), truths)), axis=-1)

    def __len__(self):
        return self.filenames.shape[0]

    def __getitem__(self, index):
        # reads contents of input image and its respective GT (using skimage)
        selected_input = io.imread(self.filenames[index][0])
        selected_truth = io.imread(self.filenames[index][1])

        # applies specified transform(s)
        if self.transform_input is not None:
            selected_input = self.transform_input(selected_input)
        if self.transform_truth is not None:
            selected_truth = self.transform_truth(selected_truth)

        # returns (paired) input image and GT
        return selected_input, selected_truth
