import torch


class Conv3(torch.nn.Module):
    def __init__(self, c_in, c_out):
        super(Conv3, self).__init__()
        self.conv = torch.nn.Conv2d(c_in, c_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        y = self.conv(x)
        return y


class ConvDown(torch.nn.Module):
    def __init__(self, c_in, c_out):
        super(ConvDown, self).__init__()
        self.conv = torch.nn.Conv2d(c_in, c_out, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

    def forward(self, x):
        y = self.conv(x)
        return y


class ConvUp(torch.nn.Module):
    def __init__(self, c_in, c_out):
        super(ConvUp, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(c_in, c_out, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

    def forward(self, x):
        y = self.conv(x)
        return y


class ResNetBlock(torch.nn.Module):
    def __init__(self, c):
        super(ResNetBlock, self).__init__()

        self.network = torch.nn.Sequential(
            Conv3(c, c),
            torch.nn.InstanceNorm2d(c),
            torch.nn.ReLU(inplace=True),
            Conv3(c, c),
            torch.nn.InstanceNorm2d(c),
        )

    def forward(self, x):
        y = x + self.network(x)
        return y


class EncoderDecoder(torch.nn.Module):
    def __init__(self, c, c_mid=64):
        super(EncoderDecoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Sequential(
                Conv3(c, c_mid),
                torch.nn.InstanceNorm2d(c_mid),
                torch.nn.ReLU(inplace=True),
            ),
            torch.nn.Sequential(
                ConvDown(c_mid, c_mid * 2),
                torch.nn.InstanceNorm2d(c_mid * 2),
                torch.nn.ReLU(inplace=True),
            ),
            torch.nn.Sequential(
                ConvDown(c_mid * 2, c_mid * 4),
                torch.nn.InstanceNorm2d(c_mid * 4),
                torch.nn.ReLU(inplace=True),
            ),
            torch.nn.Sequential(
                ConvDown(c_mid * 4, c_mid * 8),
                torch.nn.InstanceNorm2d(c_mid * 8),
                torch.nn.ReLU(inplace=True),
            ),
        )

        self.resblock = torch.nn.Sequential(
            torch.nn.Sequential(
                Conv3(c_mid * 8 + c, c_mid * 8),
                torch.nn.InstanceNorm2d(c_mid * 8),
                torch.nn.ReLU(inplace=True),
            ),
            ResNetBlock(c_mid * 8),
            ResNetBlock(c_mid * 8),
            ResNetBlock(c_mid * 8),
            ResNetBlock(c_mid * 8),
            ResNetBlock(c_mid * 8),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Sequential(
                ConvUp(c_mid * 8, c_mid * 4),
                torch.nn.InstanceNorm2d(c_mid * 4),
                torch.nn.ReLU(inplace=True),
            ),
            torch.nn.Sequential(
                ConvUp(c_mid * 4, c_mid * 2),
                torch.nn.InstanceNorm2d(c_mid * 2),
                torch.nn.ReLU(inplace=True),
            ),
            torch.nn.Sequential(
                ConvUp(c_mid * 2, c_mid),
                torch.nn.InstanceNorm2d(c_mid),
                torch.nn.ReLU(inplace=True),
            ),
        )

    def forward(self, x, pose):
        # x.shape: B x C x S x S
        # pose.shape: B x k
        a0 = x
        S = a0.shape[-1]

        a1 = pose
        a2 = torch.tile(a1.unsqueeze(-1).unsqueeze(-1), (1, 1, S // 8, S // 8))

        b3 = self.encoder(a2)
        c0 = torch.cat((b3, a2), dim=1)
        c6 = self.resblock(c0)
        d_out = self.decoder(c6)
