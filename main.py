"""
Exposure Correction GAN
Code by Fras Healey

(all training and testing completed on 12-thread CPU (Ryzen 5 7600),
12GB VRAM GPU (RTX 4070), 32GB DRAM Windows 11 PC)
"""

from enum import Enum
import numpy as np
from skimage import io
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class ModelMode(Enum):
    train = 1
    evaluate = 2
    test = 3

# configures cuda (if available)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")

# choose whether to train, evaluate, or test model
model_mode = ModelMode.train

# hyper-parameters
batch_size = 32
num_workers = 4
num_epochs = 200
lr_gen = 0.0004
lr_dis = 0.0001
adv_loss = nn.BCEWithLogitsLoss()
recon_loss = nn.L1Loss()
recon_lambda = 100

# file (dataset, results) directories
train_dir = "./dataset/training"
test_dir = "./dataset/testing"    # (expert c used)
pretrain_dir = "./pretrains"
