import os
import numpy as np
from skimage import io
from torch.utils.data import Dataset
from tqdm import tqdm
import torch


# class that allows all data to be preloaded to gpu
class ExposureDatasetGPU(Dataset):
    def __init__(self, input_dir, truth_dir, transform_input=None, transform_truth=None):

        # create list of input and gt paths
        input_dirs = np.array(os.listdir(input_dir))
        input_dirs = np.char.add(np.full(input_dirs.shape, input_dir), input_dirs)
        truth_dirs = np.array(os.listdir(truth_dir))
        truth_dirs = np.char.add(np.full(truth_dirs.shape, truth_dir), truth_dirs)
        # read and transform each image my_tensor = torch.cat([a, b], dim=2)
        self.input_imgs = torch.stack([transform_input(io.imread(input_dirs[i])) for i in tqdm(range(0, input_dirs.shape[0]))])
        self.truth_imgs = torch.stack([transform_truth(io.imread(truth_dirs[i])) for i in tqdm(range(0, truth_dirs.shape[0]))])

    def __len__(self):
        return self.input_imgs.shape[0]

    def __getitem__(self, index):
        # returns (paired) input image and GT
        return self.input_imgs[index], self.truth_imgs[index // 5]


# dataset with more "ordinary" lazy loading
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
