import argparse
import os
import numpy as np
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from setting import opt


img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Directory for saving images
image_dir = f"gen_images/label{opt.data_label}"
os.makedirs(image_dir, exist_ok=True)

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

# Load the model once if the same model is used for all generations
model = Generator()
if cuda:
    model.cuda()

# Assuming the model file naming follows a specific pattern
model_path_pattern = "saved_models/label{}/generator_{}_{}.pth"


# Generate images for every 10 epochs
for epoch in range(0, opt.n_epochs, 10):
    model_path = model_path_pattern.format(opt.data_label, opt.data_label, epoch)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"Model file {model_path} not found.")
        continue  # Skip this epoch if the model file is not found

    for i in range(opt.num_gen):
        # Correcting the dimensions of the noise vector to (batch_size, latent_dim)
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

        # Generate a batch of images
        imgs = model(z)

        # Save the generated images
        for j, img in enumerate(imgs):
            save_path = f"{image_dir}/epoch{epoch}/{epoch}_{i}_{j}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_image(img.data, save_path, normalize=True)
