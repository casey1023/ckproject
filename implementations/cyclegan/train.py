# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import random
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=-1, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=10, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--gpu_device", type=str, default=0, help="set up which gpu you gonna use")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/test_image/%s" % opt.dataset_name, exist_ok=True)

if opt.epoch == -1:
    print("Please input the epoch to load in models!")
    exit()

if opt.dataset_name == "monet2photo":
    print("Please input the dataset you're gonna use!")
    exit()


# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

#cuda = torch.cuda.is_available()
cuda = False


input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = Generator(input_shape, opt.num_residual_blocks)
G_BA = Generator(input_shape, opt.num_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    torch.cuda.set_device(opt.gpu_device)
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    if torch.cuda.is_available():
        G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
        G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
        D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
        D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))
    else:
        print("no gpu")
        exit()

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

#Load in data
dataloader = DataLoader(
    ImageDataset("../../data/test_image_data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=False, mode="train"),
    batch_size=1,
    shuffle=False
)

wordlist = []

for (dirpath, dirnames, filenames) in os.walk("../../data/test_image_data/%s" % opt.dataset_name):
    wordlist.append(filenames)

'''
for i in range(len(wordlist)):
    print(wordlist[i])
'''

data_amount = len(dataloader)
cmp_num = int(input("how many pic do you want to compare?"))

for k in range(cmp_num):
    q = ''
    selfpick = False
    num_list = []
    jumpout = False

    while not jumpout:
        q = str(input("would you like to pick it yourself?"))
        if q != 'y' and q != 'n':
            continue
        else:
            jumpout = True
            if q == 'y':
                selfpick = True
            else:
                selfpick = False

    jumpout = False

    if selfpick == True:
        print('number is between 0 ~ %s' % str(data_amount - 1))
        while not jumpout:
            q = str(input("please pick 5 number(ex.2 10 6 3 56)"))
            temp = q.split(" ")
            jumpout = True
            
            if len(temp) != 5:
                jumpout = False
                continue
            
            for i in range(5):
                num_list.append(int(temp[i]))
            
            check = False
            for i in range(5):
                if (num_list[i] < 0) or (num_list[i] > (data_amount - 1)):
                    check = True
            
            if not(check):
                jumpout = True

    elif selfpick == False:
        for i in range(5):
            num_list.append(random.randint(0, data_amount))

    #print(num_list)

    num_list.sort()
    real_A = []
    real_B = []
    fake_A = []
    fake_B = []
    numcnt = 0

        

    for (i, batch) in enumerate(dataloader):
        if numcnt >= 5:
            break
        if i == num_list[numcnt]:
            numcnt += 1
            A = Variable(batch["A"].type(Tensor))
            B = Variable(batch["B"].type(Tensor))
            
            real_A.append(A[0])
            real_B.append(B[0])

            G_AB.eval()
            G_BA.eval()

            f_A = G_BA(B)
            f_B = G_AB(A)

            fake_A.append(f_A[0])
            fake_B.append(f_B[0])

    real_A = make_grid(real_A, nrow=5, normalize=False)
    real_B = make_grid(real_B, nrow=5, normalize=False)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)

    name = ""
    for i in range(5):
        name += str(num_list[i]) + "_"

    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)

    os.makedirs("images/test_image/%s" % opt.dataset_name, exist_ok=True)

    save_image(image_grid, "images/test_image/%s/%s.jpg" % (opt.dataset_name, name), normalize=False)


print("end")