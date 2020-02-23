import torch
from torch import nn

# UNet is composed of blocks which consist of 2 conv2ds and ReLUs
def convBlock(in_channels, out_channels, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=padding),
        nn.ReLU(),
    )

# Skip connections are concatenated, cropping if size changed due to no padding
def cropAndConcat(a, b):
    if (a.shape == b.shape):
        return torch.cat([a, b], 1)

    margin2 = (a.shape[2] - b.shape[2]) // 2
    margin3 = (a.shape[3] - b.shape[3]) // 2
    a_cropped = a[:, :, margin2 : margin2 + b.shape[2], margin3 : margin3 + b.shape[3]]
    return torch.cat([a_cropped, b], 1)

class UNet(nn.Module):

    # Depth includes the bottleneck block. So total number of blocks is depth * 2 - 1
    # Unexpected output sizes or num channels can occur if parameters aren't nice
    # powers of 2
    def __init__(self,
                 input_channels=1,
                 output_channels=2,
                 depth=5,
                 num_initial_channels=64,
                 conv_padding=0
                 ):
        super().__init__()

        # Going down, each conv block doubles in number of feature channels
        self.down_convs = nn.ModuleList()
        in_channels = input_channels
        out_channels = num_initial_channels
        for _ in range(depth-1):
            self.down_convs.append(convBlock(in_channels, out_channels, conv_padding))
            in_channels = out_channels
            out_channels *= 2

        self.bottleneck = convBlock(in_channels, out_channels, conv_padding)

        # On the way back up, feature channels decreases.
        # We also have transpose convolutions for upsampling
        self.up_convs = nn.ModuleList()
        self.tp_convs = nn.ModuleList()
        in_channels = out_channels
        out_channels = in_channels // 2
        for _ in range(depth-1):
            self.up_convs.append(convBlock(in_channels, out_channels, conv_padding))
            self.tp_convs.append(nn.ConvTranspose2d(in_channels, out_channels,
                                                    kernel_size=2, stride=2))
            # self.tp_convs.append(nn.Sequential(
                # nn.Upsample(mode='bilinear', scale_factor=2),
                # nn.Conv2d(in_channels, out_channels, kernel_size=1)))

            in_channels = out_channels
            out_channels //= 2

        # final layer is 1x1 convolution, don't need padding here
        self.final_conv = nn.Conv2d(in_channels, output_channels, 1)

        # max pooling gets applied in a couple places. It has no
        # trainable parameters, so we just make one and reuse it.
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        for down_conv in self.down_convs:
            features.append(down_conv(x))
            x = self.max_pool(features[-1])

        x = self.bottleneck(x)

        for up_conv, tp_conv, feature in zip(self.up_convs, self.tp_convs, reversed(features)):
            x = up_conv(cropAndConcat(feature, tp_conv(x)))

        return self.final_conv(x)

if __name__ == '__main__':
    unet = UNet(1, 2, 3, 32, 1)
    # unet = UNet()
    x = torch.rand(1, 1, 512, 512)
    y = unet(x)
    print(y.shape)
