import torch
from torch import nn
import torch.nn.functional as F

# from spectral_normalization import SpectralNorm
import numpy as np
from torch.nn.utils import spectral_norm
from config import opt



channels = opt.channels
bias = opt.bias
img_size = opt.img_size
img_shape = (opt.channels, opt.img_size, opt.img_size)
DIM_EMBED=128
GEN_SIZE =64

# condition setting
condition_size = opt.condition_size
condition_shape = (opt.channels, opt.condition_size, opt.condition_size)

# generator setting
dim_embed = DIM_EMBED
gen_size = 16

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, dim_embed):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)

        # self.embed = nn.Linear(dim_embed, num_features * 2, bias=False)
        # self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        # self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        # # self.embed = spectral_norm(self.embed) #seems not work

        self.embed_gamma = nn.Linear(dim_embed, num_features, bias=False)
        self.embed_beta = nn.Linear(dim_embed, num_features, bias=False)

    def forward(self, x, y):
        out = self.bn(x)
        # gamma, beta = self.embed(y).chunk(2, 1)
        # out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)

        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        out = out + out*gamma + beta

        return out



#########################################################
# genearator
    

class cond_generator(nn.Module):
    def __init__(self):
        super(cond_generator, self).__init__()

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
            nn.Linear(1024, int(np.prod(condition_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        condition = self.model(z)
        condition = condition.view(condition.shape[0], *condition_shape)
        return condition



class Cc_generator(nn.Module):
    def __init__(self, nz=16, dim_embed=100, ngf=GEN_SIZE, gen_size=gen_size, condition_size=condition_size, channels=channels):
        super(Cc_generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.dim_embed = dim_embed
        self.gen_size = gen_size
        self.condition_size = condition_size

        self.num_upsamples = int(np.log2(gen_size) - 2)
        
        self.linear_size = 4 * 4 * self.ngf * 2 ** self.num_upsamples
        self.linear = nn.Linear(nz + condition_size*condition_size, self.linear_size) 

        # Dynamic upsampling layers
        self.upsample_blocks = nn.ModuleList()
        in_channels = self.ngf * 2 ** self.num_upsamples
        for i in range(self.num_upsamples):
            out_channels = in_channels // 2
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    ConditionalBatchNorm2d(out_channels, dim_embed),
                    nn.ReLU()
                )
            )
            in_channels = out_channels

        self.final_conv = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            nn.Conv2d(ngf, channels, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.Tanh()
        )

    def forward(self, z, condition):
        condition = torch.squeeze(condition, 1)
        
        z = z.view(-1, self.nz)

        out = self.linear(z)
        out = out.view(-1, self.ngf * self.gen_size, 4, 4)

        for block in self.upsample_blocks:
            out = block(out, condition)

        out = self.final_conv(out)
        return out
    

