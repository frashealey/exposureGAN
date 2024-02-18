import os
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset


# dataset with more "ordinary" lazy loading
class ExposureDataset(Dataset):
    def __init__(self, input_dir, truth_dir, transform_input=None, transform_truth=None):
        # defines transforms and directories for inputs and ground truths
        self.transform_input = transform_input
        self.transform_truth = transform_truth
        # creates arrays of inputs and ground truths
        self.inputs = np.array(os.listdir(input_dir))
        self.input_dir = input_dir
        self.truths = np.array(os.listdir(truth_dir))
        self.truth_dir = truth_dir

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        # reads contents of input image and its respective GT (using skimage)
        selected_input = read_image(os.path.join(self.input_dir, self.inputs[index]))
        selected_truth = read_image(os.path.join(self.truth_dir, self.truths[index // 5]))

        # applies specified transform(s)
        if self.transform_input is not None:
            selected_input = self.transform_input(selected_input)
        if self.transform_truth is not None:
            selected_truth = self.transform_truth(selected_truth)

        # returns (paired) input image and GT
        return selected_input, selected_truth
