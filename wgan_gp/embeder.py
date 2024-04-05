import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from setting import opt

os.makedirs("vectors", exist_ok=True)

cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


img_shape = (opt.channels, opt.img_size, opt.img_size)

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # 加载图像为灰度
            image = Image.open(img_path).convert('L')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个与您的网络输入尺寸相匹配的黑色占位灰度图像
            image = torch.zeros((1, opt.img_size, opt.img_size), dtype=torch.float32)
        return image, 0

# 定义数据预处理步骤
transform = transforms.Compose([
    transforms.Resize((opt.img_size, opt.img_size)),  # 或者任何您需要的尺寸
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


if opt.type_of_data == "train": 
    image_dir = f"train_set/label{opt.data_label}"
    # 创建自定义数据集实例
    custom_dataset = CustomDataset(
        image_dir= image_dir,
        transform=transform
    )

    # 使用DataLoader加载数据
    # 在Windows系统中，较低的num_workers数或为0可以提高稳定性，但可能影响加载速度
    dataloader = DataLoader(
        custom_dataset,
        # batch_size = the number of samples in the folder
        batch_size=custom_dataset.__len__(), 
        shuffle=True,
        num_workers=0  # Windows环境建议设置为0或较低的数值
    )



    # save all images to a numpy file
    for i, (imgs, _) in enumerate(dataloader):
        real_imgs = Variable(imgs.type(Tensor))
        # save all images to a numpy file
        np.save(f"vectors/label{opt.data_label}_{opt.type_of_data}.npy", real_imgs.cpu().detach().numpy())

else:
    # save all images to a numpy file
    for epoch in range(0, opt.n_epochs, 10):
        image_dir = f"gen_images/label{opt.data_label}/epoch{epoch}"
         # 创建自定义数据集实例
        custom_dataset = CustomDataset(
            image_dir= image_dir,
            transform=transform
        )

        # 使用DataLoader加载数据
        # 在Windows系统中，较低的num_workers数或为0可以提高稳定性，但可能影响加载速度
        dataloader = DataLoader(
            custom_dataset,
            # batch_size = the number of samples in the folder
            batch_size=custom_dataset.__len__(), 
            shuffle=True,
            num_workers=0  # Windows环境建议设置为0或较低的数值
        )



        # save all images to a numpy file
        for i, (imgs, _) in enumerate(dataloader):
            real_imgs = Variable(imgs.type(Tensor))
            # save all images to a numpy file
            os.makedirs(f"vectors/epoch{epoch}", exist_ok=True)
            np.save(f"vectors/epoch{epoch}/label{opt.data_label}_{opt.type_of_data}_epoch{epoch}.npy", real_imgs.cpu().detach().numpy())