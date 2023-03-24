import math
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, scale_factor):
        # -----------------------------------#
        #   获得进行上采用的次数
        # -----------------------------------#
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()

        # --------------------------------------------------------#
        #   第一部分，低分辨率图像进入后会经过一个卷积+PRELU函数
        # --------------------------------------------------------#

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),#卷积神经网络
            nn.PReLU()#激活函数，CNN中常用。对正数原样输出，负数直接置零。但是这里的a不是固定下来的，而是可学习的。
        )
        # -------------------------------------------------------------#
        #   第二部分，经过num_residual个残差网络结构。
        #   每个残差网络内部包含两个卷积+标准化+PRELU，还有一个残差边。
        # -------------------------------------------------------------#
        self.block2 = ResidualBlock(64)#调用ResidualBloc函数进行残差网络
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )#添加残差边，有助于学习
        # -------------------------------------------------------------#
        #   第三部分，上采样部分，将长宽进行放大。
        #   两次上采样后，变为原来的4倍，实现提高分辨率。
        # -------------------------------------------------------------#
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]#调用UpsampleBLock函数上采样
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))#通道数为3的卷积，变成图片格式，理论上输出超分辨率后的图片
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()#一开始输入的是高分辨率图片
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),#第一步：64通道卷积+LeakyReLU
            nn.LeakyReLU(0.2),#激活函数，让负数区域不在饱和死掉，这里的斜率都是确定的。

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),#步长2的卷积+LeakyReLU+标准化
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),#步长1的卷积+LeakyReLU+标准化
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),#步长2的卷积+LeakyReLU+标准化
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),#步长1的卷积+LeakyReLU+标准化
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),#步长2的卷积+LeakyReLU+标准化
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),#步长1的卷积+LeakyReLU+标准化
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),#步长2的卷积+LeakyReLU+标准化，通道数512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            #长宽变为1/16
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)#调用库函数全连接层+LeakyReLU+全连接层+判别网络
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))#将输出限制为0和1


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
