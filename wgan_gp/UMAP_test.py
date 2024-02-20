# using umap to visualize the vector space of the pictures
import umap
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import argparse
from torchvision import datasets
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from setting import opt
import time
from sklearn.datasets import load_digits
from mpl_toolkits.mplot3d import Axes3D


# os.makedirs("images", exist_ok=True)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

save_path = f'./umap/label{opt.data_label}'

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load the npy file
if opt.type_of_data == "train": 
    data = np.load(f'./vectors/label{opt.data_label}_train.npy')
    n_samples, x, y, z = data.shape
    data_flattened = data.reshape((n_samples, x*y*z))

    # UMAP
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data_flattened)
    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.savefig(f'{save_path}/train.png')

elif opt.type_of_data == "gen":
    data = list()
    for epoch in range(0, opt.n_epochs, 10):
        data.append(np.load(f'./vectors/epoch{epoch}/label{opt.data_label}_gen_epoch{epoch}.npy'))
    
    for i in range(len(data)):
        if i == 0:
            used_data = data[0]
        else:
            used_data = np.concatenate(data[0:i], axis=0)
        n_samples, x, y, z = used_data.shape
        data_flattened = used_data.reshape((n_samples, x*y*z))

        # UMAP
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(data_flattened)
        plt.scatter(embedding[:, 0], embedding[:, 1])
        plt.savefig(f'{save_path}/epoch{i}.png')

    data.append(np.load(f'./vectors/label{opt.data_label}_train.npy'))
    used_data = np.concatenate(data, axis=0)
    n_samples, x, y, z = used_data.shape
    data_flattened = used_data.reshape((n_samples, x*y*z))

    # UMAP
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data_flattened)
    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.savefig(f'{save_path}/all.png')

elif opt.type_of_data == "epoch":
    save_path = f'./umap/change'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data = list()
    for label in range(0, 10):
        data.append(np.load(f'./vectors/epoch{opt.n_epochs-10}/label{label}_gen_epoch{opt.n_epochs-10}.npy'))
    
    for i in range(len(data)):
        if i == 0:
            used_data = data[0]
        else:
            used_data = np.concatenate(data[0:i], axis=0)
        n_samples, x, y, z = used_data.shape
        data_flattened = used_data.reshape((n_samples, x*y*z))

        # UMAP
        reducer = umap.UMAP(n_components = max(opt.axe1, opt.axe2, opt.axe3) + 1)
        embedding = reducer.fit_transform(data_flattened)
        if opt.axe3 == 0:
            plt.scatter(embedding[:, opt.axe1], embedding[:, opt.axe2])
            plt.savefig(f'{save_path}/epoch{opt.n_epochs-10}_step{i}_{opt.axe1}vs{opt.axe2}.png')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X = embedding[:, opt.axe1]
            Y = embedding[:, opt.axe2]
            Z = embedding[:, opt.axe3]
            ax.scatter(X, Y, Z)
            plt.savefig(f'{save_path}/epoch{opt.n_epochs-10}_step{i}_{opt.axe1}vs{opt.axe2}vs{opt.axe3}.png')

else:
    print("Invalid type_of_data")
