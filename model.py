import torch
import torch.nn as nn
from spectral import SpectralNorm
import numpy as np


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.in_channels = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class Generator(nn.Module):
    """Generator."""

    def __init__(self, image_size=64, z_dim=128, conv_dim=64):
        super(Generator, self).__init__()
        # self.image_size = image_size

        repeat_num = int(np.log2(image_size)) - 3   # 3
        mult = 2 ** repeat_num  # 8

        # Input: (N, z_dim)

        self.l1 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)),
            nn.BatchNorm2d(conv_dim * mult),
            nn.ReLU(),
        )

        curr_dim = conv_dim * mult
        self.l2 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)),
            nn.BatchNorm2d(curr_dim // 2),
            nn.ReLU(),
        )

        curr_dim = curr_dim // 2
        self.l3 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)),
            nn.BatchNorm2d(curr_dim // 2),
            nn.ReLU(),
        )

        curr_dim = curr_dim // 2
        self.l4 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)),
            nn.BatchNorm2d(curr_dim // 2),
            nn.ReLU(),
        )

        curr_dim = curr_dim // 2
        self.last = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1),
            nn.Tanh()
        )

        self.attn1 = Self_Attn(128, 'relu')
        self.attn2 = Self_Attn(64, 'relu')

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        print(z.shape)
        out = self.l1(z)
        print(out.shape)
        out = self.l2(out)
        print(out.shape)
        out = self.l3(out)
        print(out.shape)
        out, p1 = self.attn1(out)
        print(out.shape)
        out = self.l4(out)
        print(out.shape)
        out, p2 = self.attn2(out)
        print(out.shape)
        out = self.last(out)
        print(out.shape)

        return out


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, im_chan=10, conv_dim=64):
        super(Discriminator, self).__init__()

        self.l1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(im_chan, conv_dim, 4, 2, 1)),
            nn.LeakyReLU(0.1),
        )

        curr_dim = conv_dim
        self.l2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1),
        )

        curr_dim = curr_dim * 2
        self.l3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1),
        )

        curr_dim = curr_dim * 2
        self.l4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1),
        )

        curr_dim = curr_dim * 2
        self.last = nn.Sequential(
            nn.Conv2d(curr_dim, 1, 4),
        )

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out, p1 = self.attn1(out)
        out = self.l4(out)
        out, p2 = self.attn2(out)
        out = self.last(out)

        return out.squeeze()


if __name__ == '__main__':
    image_size = 64
    z_dim = 128
    g_conv_dim = 64
    d_conv_dim = 64
    batch_size = 64
    n_classes = 10
    im_chan = 3
    image_path = '../resized_emoji_challenge'

    from data_loader import Data_Loader
    data_loader = Data_Loader(image_path, image_size, batch_size, True).loader()
    data_iter = iter(data_loader)
    real_images, labels = next(data_iter)

    from utils import *
    z = tensor2var(torch.randn(batch_size, z_dim))
    one_hot_labels = get_one_hot_labels(labels, n_classes).cuda()
    image_one_hot_labels = one_hot_labels[:, :, None, None]
    image_one_hot_labels = image_one_hot_labels.repeat(1, 1, image_size, image_size)

    noise_and_labels = combine_vectors(z, one_hot_labels)   # (batch_size, z_dim + n_classes)

    G = Generator(image_size, z_dim + n_classes, g_conv_dim)
    # print(G)
    D = Discriminator(im_chan + n_classes, d_conv_dim)
    print(D)

    noise_and_labels = noise_and_labels.to('cpu')
    print(f'noise_and_labels.shape: {noise_and_labels.shape}')
    fake_images, _, _ = G(noise_and_labels)
    # print(G(fixed_z))

    print('FAKE')
    fake_image_and_labels = combine_vectors(fake_images.detach().to('cpu'), image_one_hot_labels.to('cpu'))
    print(f'fake_image_and_labels.shape {fake_image_and_labels.shape}')
    d_out_fake, _, _ = D(fake_image_and_labels)
    print(d_out_fake.shape)

    print("REAL")
    real_image_and_labels = combine_vectors(real_images, image_one_hot_labels.to('cpu'))
    d_out_real, _, _ = D(fake_image_and_labels)
    print(d_out_real.shape)
