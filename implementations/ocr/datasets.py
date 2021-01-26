import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files = []
        for i in range(1, 1000 + 1):
            self.files_temp = sorted(glob.glob(os.path.join(root, "%s/%s" % mode % str(i)) + "/*.*"))
            self.files.append(self.files_temp)

    def __getitem__(self, index):
        image = []
        for i in range(1, 1000 + 1):
            image_temp = Image.open(self.files[i][random.randint(0, len(self.files[i]) - 1)])
            image.append(image_temp)

        item = []
        for i in range(1, 1000 + 1):
            item_temp = self.transform(image[i])
            item.append(item_temp)
        return {i: item[i] for i in range(1, 1000 + 1)}

    def __len__(self):
        return max(len(self.files[i]) for i in range(1, 1000 + 1))
