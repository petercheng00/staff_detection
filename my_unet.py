import torch
from torch import nn

def convBlock(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3),
        nn.ReLU(),
    )

def cropAndConcat(a, b):
    margin2 = (a.shape[2] - b.shape[2]) // 2
    margin3 = (a.shape[3] - b.shape[3]) // 2
    a_cropped = a[:, :, margin2 : margin2 + b.shape[2], margin3 : margin3 + b.shape[3]]
    return torch.cat([a_cropped, b], 1)

class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.down_conv1 = convBlock(1, 64)
        self.down_conv2 = convBlock(64, 128)
        self.down_conv3 = convBlock(128, 256)
        self.down_conv4 = convBlock(256, 512)
        self.down_conv5 = convBlock(512, 1024)

        self.up_conv1 = convBlock(1024, 512)
        self.up_conv2 = convBlock(512, 256)
        self.up_conv3 = convBlock(256, 128)
        self.up_conv4 = convBlock(128, 64)

        self.final_conv = nn.Conv2d(64, 2, 1)

        self.max_pool = nn.MaxPool2d(2)
        self.tp_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.tp_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.tp_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.tp_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)


    def forward(self, x):
        features1 = self.down_conv1(x)
        features2 = self.down_conv2(self.max_pool(features1))
        features3 = self.down_conv3(self.max_pool(features2))
        features4 = self.down_conv4(self.max_pool(features3))
        bottleneck = self.down_conv5(self.max_pool(features4))
        x = self.up_conv1(cropAndConcat(features4, self.tp_conv1(bottleneck)))
        x = self.up_conv2(cropAndConcat(features3, self.tp_conv2(x)))
        x = self.up_conv3(cropAndConcat(features2, self.tp_conv3(x)))
        x = self.up_conv4(cropAndConcat(features1, self.tp_conv4(x)))
        return self.final_conv(x)

if __name__ == '__main__':
    unet = UNet()
    x = torch.rand(1, 1, 572, 572)
    y = unet(x)
    print(y.shape)
