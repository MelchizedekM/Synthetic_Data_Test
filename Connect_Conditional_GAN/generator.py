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

#########################################################
# ConditionalBatchNorm2d

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, dim_embed):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)

        self.embed_gamma = nn.Linear(dim_embed, num_features, bias=False)
        self.embed_beta = nn.Linear(dim_embed, num_features, bias=False)

    def forward(self, x, y):
        out = self.bn(x)

        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        out = out + out*gamma + beta

        return out


#########################################################
# genearator

class cond_generator(nn.Module):
    def __init__(self, condition_shape = condition_shape):
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
    def __init__(self, noise_size=4, dim_embed=DIM_EMBED, ngf=GEN_SIZE, gen_size=gen_size, condition_size=condition_size, channels=channels):
        super(Cc_generator, self).__init__()
        self.noise_size = noise_size
        self.ngf = ngf
        self.dim_embed = dim_embed
        self.gen_size = gen_size
        self.condition_size = condition_size

        self.num_upsamples = int(np.log2(gen_size) - np.log2(noise_size))
        
        self.linear_size = noise_size * noise_size * self.ngf * 2 ** self.num_upsamples
        self.linear = nn.Linear(noise_size*noise_size, self.linear_size) 

        # Dynamic upsampling layers
        self.ConvTranspose2d_blocks = nn.ModuleList()
        self.ConditionalBatchNorm2d_blocks = nn.ModuleList()
        self.ReLU_blocks = nn.ModuleList()
        in_channels = self.ngf * 2 ** self.num_upsamples
        for i in range(self.num_upsamples):
            out_channels = in_channels // 2
            
            self.ConvTranspose2d_blocks.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=bias)
            )

            self.ConditionalBatchNorm2d_blocks.append(
                ConditionalBatchNorm2d(out_channels, dim_embed)
            )

            self.ReLU_blocks.append(
                nn.ReLU()
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
        condition = condition.view(-1, self.condition_size*self.condition_size)
        
        z = z.view(-1, self.noise_size*self.noise_size)

        out = self.linear(z)

        out = out.view(-1, self.ngf * 2 ** self.num_upsamples, self.noise_size, self.noise_size)

        for i in range(self.num_upsamples):
            out = self.ConvTranspose2d_blocks[i](out)
            out = self.ConditionalBatchNorm2d_blocks[i](out, condition)
            out = self.ReLU_blocks[i](out)

        out = self.final_conv(out)
        return out


class Final_G(nn.Module):
    def __init__(self, noise_size=4, dim_embed=DIM_EMBED, ngf=GEN_SIZE, gen_size=gen_size, condition_size=condition_size, channels=channels, condition_shape = condition_shape, Cc_generator=Cc_generator, cond_generator=cond_generator):
        super(Final_G, self).__init__()
        self.noise_size = noise_size
        self.ngf = ngf
        self.dim_embed = dim_embed
        self.gen_size = gen_size
        self.condition_size = condition_size
        self.channels = channels
        self.condition_shape = condition_shape
        self.Cc_generator = Cc_generator
        self.cond_generator = cond_generator


        self.cond_G = cond_generator(condition_shape = condition_shape)
        self.Cc_G_1 = Cc_generator(noise_size= noise_size // 2 , dim_embed=dim_embed // 2, ngf=ngf // 2, gen_size=condition_size * 2, condition_size=condition_size, channels=1)
        self.Cc_G_2 = Cc_generator(noise_size= noise_size, dim_embed=dim_embed, ngf=ngf, gen_size=gen_size, condition_size=condition_size * 2, channels=channels)

    def forward(self, z_0, z_1, z_2, sigma=0.001):
        condition = self.cond_G(z_0)
        noise_1 = torch.randn_like(condition) * sigma
        condition = condition + noise_1  
        condition = self.Cc_G_1(z_1, condition)
        noise_2 = torch.randn_like(condition) * sigma
        condition = condition + noise_2  
        out = self.Cc_G_2(z_2, condition)
        return out