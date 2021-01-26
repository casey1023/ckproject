import random
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
for i in range(10):
    print(random.sample(range(5),5))
a = np.zeros((3,4))
a[1][2] = 1
print(Tensor(a))