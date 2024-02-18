"""
Exposure Correction GAN
Code by Fras Healey

(all training and testing completed on 6 core 12-thread CPU,
(Ryzen 5 7600) 12GB VRAM GPU (RTX 4070), 32GB DRAM Windows 11 PC)
"""

import os
import re
from enum import Enum
import tkinter as tk
from tkinter import filedialog
import numpy as np
from natsort import natsorted
from skimage import io
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from dataset import ExposureDataset, ExposureDatasetGPU
from models import Generator, Discriminator


class ModelMode(Enum):
    train = 1
    evaluate = 2
    test = 3

# choose whether to train, evaluate, or test model
model_mode = ModelMode.train

# configures cuda (if available)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")

# hyper-parameters
batch_size = 64
num_workers = 6
num_epochs = 100
lr_gen = 0.0002
lr_dis = 0.0002
adv_loss = nn.BCEWithLogitsLoss()
recon_loss = nn.L1Loss()
recon_lambda = 100

# file (dataset, results) directories
train_dir = "./dataset/training/"
test_dir = "./dataset/testing/"    # (expert c used)
pretrain_dir = "./pretrains/"

def backprop(scaler, optim, loss):
    # uses gradient scaler to backpropagate and update weights
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

def trainer(gen, dis, train_loader, gen_optim, dis_optim, gen_scaler, dis_scaler, recon_loss, adv_loss):
    # defines tqdm (progress bar)
    looper = tqdm(train_loader)

    # sets each model to training mode
    # (so layers like dropout, batchnorm, etc. infer correctly)
    gen.train()
    dis.train()

    # iterates through dataloader
    for i, (x, y) in enumerate(looper):
        # x = x.to(device)
        # y = y.to(device)

        # trains discriminator
        dis_optim.zero_grad()
        # uses mixed precision to decrease training time,
        # (by automatically allowing certain operations to use float16 over float32)
        with torch.cuda.amp.autocast():
            # generator creates a fake image
            fake_img = gen(x)
            # discriminator output for real image
            dis_real = dis(x, y)
            # discriminator output for fake image
            # (using .detach() to allow using the same fake_img during generator training)
            dis_fake = dis(x, fake_img.detach())
            # calculates total discriminator loss as average of losses
            # of discriminator being correct and being incorrect
            loss_dis = (adv_loss(dis_real, torch.ones_like(dis_real))) + (adv_loss(dis_fake, torch.zeros_like(dis_fake))) * 0.5

        # updates discriminator weights through backpropagation
        backprop(dis_scaler, dis_optim, loss_dis)

        # trains generator
        gen_optim.zero_grad()
        # uses mixed precision (see above)
        with torch.cuda.amp.autocast():
            # discriminator output for fake image
            dis_fake = dis(x, fake_img)
            # calculates total generator loss as sum of losses of
            # discriminator being correct and L1 loss of fake and GT images
            # (multiplied by lambda to negate visual artefacts)
            loss_gen = (adv_loss(dis_fake, torch.ones_like(dis_fake))) + (recon_loss(fake_img, y) * recon_lambda)

        # updates generator weights through backpropagation
        backprop(gen_scaler, gen_optim, loss_gen)

    # returns generator and discriminator losses
    return loss_gen, loss_dis

def tester(gen, test_loader, denorm):
    gen.eval()

    # defines arrays to store PSNR and SSIM
    # values for each channel
    psnr_arr = []
    ssim_arr = []
    # defines tqdm looper (display loading bar)
    looper = tqdm(test_loader)
    # iterates for every image in expert's testing set
    for i, (x, y) in enumerate(looper):
        x = x.to(device)
        y = y.to(device)

        # generator creates fake image
        with torch.no_grad():
            fake_img = gen(x)

        # transforming tensors to computable images
        real_img = denorm(y).cpu()[0].permute(1, 2, 0).numpy()
        fake_img = denorm(fake_img).cpu()[0].permute(1, 2, 0).numpy()

        # splitting by channel, as PSNR and SSIM can only
        # be calculated on a single channel
        real1 = real_img[:, :, 0]
        real2 = real_img[:, :, 1]
        real3 = real_img[:, :, 2]
        fake1 = fake_img[:, :, 0]
        fake2 = fake_img[:, :, 1]
        fake3 = fake_img[:, :, 2]
        data_range1 = fake1.max() - fake1.min()
        data_range2 = fake2.max() - fake2.min()
        data_range3 = fake3.max() - fake3.min()

        # calculates PSNR and SSIM for each channel
        # (using skimage)
        psnr1 = psnr(real1, fake1, data_range=data_range1)
        psnr2 = psnr(real2, fake2, data_range=data_range2)
        psnr3 = psnr(real3, fake3, data_range=data_range3)
        ssim1 = ssim(real1, fake1, data_range=data_range1)
        ssim2 = ssim(real2, fake2, data_range=data_range2)
        ssim3 = ssim(real3, fake3, data_range=data_range3)
        # calculates the mean average of the 3 channels
        psnr_arr.append((psnr1 + psnr2 + psnr3) / 3)
        ssim_arr.append((ssim1 + ssim2 + ssim3) / 3)

    # calculates and outputs the mean averages of PSNR and SSIM
    # for all images in expert's testing set
    psnr_mean = np.mean(np.array(psnr_arr))
    ssim_mean = np.mean(np.array(ssim_arr))

    return psnr_mean, ssim_mean

def evaluator(gen, image_path, transform, denorm):
    gen.eval()
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # read and transform image
    # (adding the batch dimension artifically using 'unsqueeze')
    image = transform(io.imread(image_path)).to(device).unsqueeze(0)

    # generator creates fake image
    with torch.no_grad():
        # generate fake image
        fake_image = gen(image)

    # input
    ax1.imshow(denorm(image).cpu()[0].permute(1, 2, 0))
    ax1.axis("off")
    ax1.set_title("Input")
    # result
    ax2.imshow(denorm(fake_image).cpu()[0].permute(1, 2, 0))
    ax2.axis("off")
    ax2.set_title("Result")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    # defines objects for generator and discriminator and inits weights
    gen = Generator().to(device)
    dis = Discriminator().to(device)
    # defining Adam optimisers for both networks betas=(0.5, 0.999)
    gen_optim = optim.Adam(gen.parameters(), lr=lr_gen, betas=(0.9, 0.999))
    dis_optim = optim.Adam(dis.parameters(), lr=lr_dis, betas=(0.9, 0.999))
    # list of checkpoints in given pretrain_dir
    pretrains = natsorted(
        list(
            filter(
                re.compile("^[0-9]+\_gen\.pth$|^[0-9]+\_dis\.pth$").match,
                os.listdir(pretrain_dir)
            )
        )
    )

    print(f"Initalised models on device: {device}")

    # training mode selected
    if model_mode == ModelMode.train:
        # defines dataset and dataloader
        print("Loading inputs/truths into VRAM:")
        train_dataset = ExposureDatasetGPU(
            os.path.join(train_dir, "INPUT_IMAGES/"),
            os.path.join(train_dir, "GT_IMAGES/"),
            transform_input=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor()
            ]),
            transform_truth=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor()
            ])
        )
        train_dataset = train_dataset.to(device)
        print(train_dataset.device)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True
        )

        # defines gradient scalers (prevents gradient values getting flushed to 0)
        gen_scaler = torch.cuda.amp.GradScaler()
        dis_scaler = torch.cuda.amp.GradScaler()

        # if pretrained model is present and not exceeding num_epochs,
        # then resume from where last stopped
        start_epoch = 0
        if len(pretrains) >= 2 and "gen.pth" in pretrains[-1] and "dis.pth" in pretrains[-2]:
            gen_epochs = int(pretrains[-1].split("_")[0]) + 1
            dis_epochs = int(pretrains[-2].split("_")[0]) + 1
            if gen_epochs < num_epochs and dis_epochs < num_epochs:
                # loads generator properties from saved state dictionaries
                gen_checkpoint = torch.load(
                    os.path.join(
                        pretrain_dir,
                        pretrains[-1]
                    ),
                    # (prevents CUDA out of memory error when loading checkpoint)
                    map_location="cpu"
                )
                gen.load_state_dict(gen_checkpoint["model_state_dict"])
                gen_optim.load_state_dict(gen_checkpoint["optimizer_state_dict"])

                # loads discriminator properties from saved state dictionaries
                dis_checkpoint = torch.load(
                    os.path.join(
                        pretrain_dir,
                        pretrains[-2]
                    ),
                    # (prevents CUDA out of memory error when loading checkpoint)
                    map_location="cpu"
                )
                dis.load_state_dict(dis_checkpoint["model_state_dict"])
                dis_optim.load_state_dict(dis_checkpoint["optimizer_state_dict"])

                # sets start_epoch to epoch after where last finished
                start_epoch = min(gen_checkpoint["epoch"], dis_checkpoint["epoch"]) + 1

        # iterates for every epoch in num_epochs
        for epoch in range(start_epoch, num_epochs):
            print(f"Epoch {epoch}")
            loss_gen, loss_dis = trainer(
                gen,
                dis,
                train_loader,
                gen_optim,
                dis_optim,
                gen_scaler,
                dis_scaler,
                recon_loss,
                adv_loss
            )
            print(f"Gen. loss: {loss_gen}")
            print(f"Dis. loss: {loss_dis}")

            # saves generator and discriminator checkpoints every epoch
            # (to ensure no progress is lost)
            torch.save({
                "epoch": epoch,
                "model_state_dict": gen.state_dict(),
                "optimizer_state_dict": gen_optim.state_dict(),
                "loss": loss_gen
            }, f"{pretrain_dir}/{epoch}_gen.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": dis.state_dict(),
                "optimizer_state_dict": dis_optim.state_dict(),
                "loss": loss_dis
            }, f"{pretrain_dir}/{epoch}_dis.pth")

    # testing mode selected
    elif model_mode == ModelMode.test:
        # checks model weights exist
        # (only need to check for generator)
        if len(pretrains) >= 1 and "gen.pth" in pretrains[-1]:
            # loads generator from saved state dictionaries
            gen_checkpoint = torch.load(
                os.path.join(
                    pretrain_dir,
                    pretrains[-1]
                )
            )
            gen.load_state_dict(gen_checkpoint["model_state_dict"])
            gen_optim.load_state_dict(gen_checkpoint["optimizer_state_dict"])

            # defines dataset and dataloader
            test_dataset = ExposureDataset(
                os.path.join(test_dir, "INPUT_IMAGES/"),
                os.path.join(test_dir, "GT_IMAGES/"),
                transform_input=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(256),
                    transforms.CenterCrop(256),
                    transforms.ToTensor()
                ]),
                transform_truth=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(256),
                    transforms.CenterCrop(256),
                    transforms.ToTensor()
                ])
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                num_workers=num_workers,
                shuffle=False,
                pin_memory=True
            )

            # runs psnr and ssim tests
            psnr_mean, ssim_mean = tester(
                gen,
                test_loader,
                transforms.Compose([
                ])
            )

            print(f"PSNR: {psnr_mean}")
            print(f"SSIM: {ssim_mean}")

        else:
            print("No pretrained model found - please ensure pretrain_dir is assigned correctly")

    # evaluation mode selected
    elif model_mode == ModelMode.evaluate:
        # checks model weights exist
        # (only need to check for generator)
        if len(pretrains) >= 1 and "gen.pth" in pretrains[-1]:
            # loads generator from saved state dictionaries
            gen_checkpoint = torch.load(
                os.path.join(
                    pretrain_dir,
                    pretrains[-1]
                )
            )
            gen.load_state_dict(gen_checkpoint["model_state_dict"])
            gen_optim.load_state_dict(gen_checkpoint["optimizer_state_dict"])

            # file prompt
            root = tk.Tk()
            # root.withdraw() doesn't work, so using this hacky workaround 
            root.attributes("-topmost", True, "-alpha", 0)
            selected_image = filedialog.askopenfilename(
                initialdir="./Data/derain/ALIGNED_PAIRS_TEST/REAL_DROPLETS",
                filetypes=[("JPG/JPEG files", ".jpg .jpeg")],
                title="Select image"
            )
            root.destroy()

            # file selected
            if len(selected_image) > 0:
                evaluator(
                    gen,
                    selected_image,
                    transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(256),
                        transforms.CenterCrop(256),
                        transforms.ToTensor()
                    ]),
                    transforms.Compose([
                    ])
                )

        else:
            print("No pretrained model found - please ensure pretrain_dir is assigned correctly")
