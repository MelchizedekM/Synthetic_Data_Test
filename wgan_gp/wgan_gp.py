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

os.makedirs("images", exist_ok=True)



img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # 第一个线性层到512维特征的层
        self.layer1 = nn.Sequential(
            nn.Linear(opt.img_size * opt.img_size * opt.channels, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 第二个线性层到256维特征的层
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 最终到1维输出的层
        self.classifier = nn.Sequential(
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.layer1(img_flat)
        x = self.layer2(x)
        validity = self.classifier(x)
        return validity

    def get_512_features(self, img):
        img_flat = img.view(img.shape[0], -1)
        x_512 = self.layer1(img_flat)  # 获取512维特征向量
        return x_512

    def get_256_features(self, img):
        img_flat = img.view(img.shape[0], -1)
        x_512 = self.layer1(img_flat)
        x_256 = self.layer2(x_512)    # 获取256维特征向量
        return x_256

# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
#os.makedirs("../../data/mnist", exist_ok=True)
#dataloader = torch.utils.data.DataLoader(
#    datasets.MNIST(
#        "../../data/mnist",
#        train=True,
#        download=True,
#        transform=transforms.Compose(
#            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#        ),
#    ),
#    batch_size=opt.batch_size,
#    shuffle=True,
#)
    

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

# 创建自定义数据集实例
custom_dataset = CustomDataset(
    image_dir= os.path.join("train_set", f"label{opt.data_label}"),
    transform=transform
)

# 使用DataLoader加载数据
# 在Windows系统中，较低的num_workers数或为0可以提高稳定性，但可能影响加载速度
dataloader = DataLoader(
    custom_dataset,
    batch_size=opt.batch_size,  # 或者任何您需要的批处理大小
    shuffle=True,
    num_workers=0  # Windows环境建议设置为0或较低的数值
)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            if batches_done % opt.sample_interval == 0:
                save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            batches_done += opt.n_critic

    # Save model
    if epoch % 10 == 0:
        torch.save(generator.state_dict(), "saved_models/label{}/generator_{}_{}.pth".format(opt.data_label,opt.data_label, epoch)),
        torch.save(discriminator.state_dict(), "saved_models/label{}/discriminator_{}_{}.pth".format(opt.data_label, opt.data_label, epoch))