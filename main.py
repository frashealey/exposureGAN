"""
Exposure Correction GAN
Code by Fras Healey

(all training and testing completed on 12-thread CPU (Ryzen 5 7600),
12GB VRAM GPU (RTX 4070), 32GB DRAM Windows 11 PC)
"""

import os
import re
from enum import Enum
import tkinter as tk
from tkinter import filedialog
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

from dataset import ExposureDataset
from models import Generator, Discriminator


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
batch_size = 16
num_workers = 4
num_epochs = 200
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
    # defines tqdm (for progress bar)
    looper = tqdm(train_loader)

    # sets each model to training mode
    # (so layers like dropout, batchnorm, etc. infer correctly)
    gen.train()
    dis.train()

    # iterates through dataloader
    for i, (x, y) in enumerate(looper):
        x = x.to(device)
        y = y.to(device)

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
            # calculates total discriminator loss as average of losses of discriminator being correct and being incorrect
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

def evaluator(gen, image_path, transform, denorm):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    gen.eval()

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
        train_dataset = ExposureDataset(
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
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True
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
                    map_location="cpu"      # prevents CUDA out of memory error when loading checkpoint
                )
                gen.load_state_dict(gen_checkpoint["model_state_dict"])
                gen_optim.load_state_dict(gen_checkpoint["optimizer_state_dict"])

                # loads discriminator properties from saved state dictionaries
                dis_checkpoint = torch.load(
                    os.path.join(
                        pretrain_dir,
                        pretrains[-2]
                    ),
                    map_location="cpu"      # prevents CUDA out of memory error when loading checkpoint
                )
                dis.load_state_dict(dis_checkpoint["model_state_dict"])
                dis_optim.load_state_dict(dis_checkpoint["optimizer_state_dict"])

                # sets start_epoch to epoch after where last finished
                start_epoch = min(gen_checkpoint["epoch"], dis_checkpoint["epoch"]) + 1

        # iterates for every epoch in num_epochs
        for epoch in range(start_epoch, num_epochs):
            print(f"Epoch {epoch}")
            loss_gen, loss_dis = trainer(gen, dis, train_loader, gen_optim, dis_optim, gen_scaler, dis_scaler, recon_loss, adv_loss)
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

            # file is selected
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
