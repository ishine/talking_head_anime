import torch
from torch import nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, c):
        super(ResnetBlock, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(c),
        )

    def forward(self, x):
        y = x + self.network(x)
        return y


class CommonBackbone(nn.Module):
    def __init__(self, c_mid=64):
        super(CommonBackbone, self).__init__()

        self.network = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(7, c_mid, kernel_size=7, stride=1, padding=3),
                nn.InstanceNorm2d(c_mid),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(c_mid, c_mid * 2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(c_mid * 2),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(c_mid * 2, c_mid * 4, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(c_mid * 4),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(c_mid * 4, c_mid * 8, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(c_mid * 8),
                nn.ReLU(inplace=True),
            ),

            ResnetBlock(c_mid * 8),
            ResnetBlock(c_mid * 8),
            ResnetBlock(c_mid * 8),
            ResnetBlock(c_mid * 8),
            ResnetBlock(c_mid * 8),
            ResnetBlock(c_mid * 8),

            nn.Sequential(
                nn.ConvTranspose2d(c_mid * 8, c_mid * 4, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(c_mid * 4),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(c_mid * 4, c_mid * 2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(c_mid * 2),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(c_mid * 2, c_mid, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(c_mid),
                nn.ReLU(inplace=True),
            ),
        ])

    def forward(self, x):
        for f in self.network:
            x = f(x)
        return x


class FaceMorpher(nn.Module):
    def __init__(self, conf):
        super(FaceMorpher, self).__init__()
        self.conf = conf
        c_mid = 64

        self.backbone = CommonBackbone(c_mid)

        self.change_image = nn.Sequential(
            nn.Conv2d(c_mid, 4, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
        )
        self.alpha_mask = nn.Sequential(
            nn.Conv2d(c_mid, 4, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, input_image, shape):
        """

        Args:
            input_image: input image. torch.Tensor of shape B x C(4) x H x W
            shape: shape vector. torch.Tensor of shape B x n(3)

        Returns:

        """
        B, n = shape.shape
        a0 = input_image  # B x 4 X H x W
        a1 = shape.reshape(B, n, 1, 1).repeat(1, 1, a0.shape[-2], a0.shape[-1])  # B x 3 x H x W
        a2 = torch.cat((a0, a1), dim=1)  # channel-wise concat

        d2 = self.backbone(a2)

        e0 = self.change_image(d2)
        e1 = self.alpha_mask(d2)
        # e2 = a0 * e0 + e1 * (1 - e0)
        e2 = e1 * e0 + (1 - e1) * a0

        result = {
            'a0': a0,
            'a1': a1,
            'e0': e0,
            'e1': e1,
            'e2': e2,
        }

        return result


class TwoAlgorithmFaceRotator(nn.Module):
    def __init__(self, conf):
        super(TwoAlgorithmFaceRotator, self).__init__()
        self.conf = conf
        c_mid = 64

        self.backbone = CommonBackbone(c_mid)

        self.change_image = nn.Sequential(
            nn.Conv2d(c_mid, 4, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
        )
        self.alpha_mask = nn.Sequential(
            nn.Conv2d(c_mid, 4, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid(),
        )

        self.appearance_flow = nn.Conv2d(c_mid, 2, kernel_size=7, stride=1, padding=3)

    def forward(self, input_image, pose):
        """

        Args:
            input_image: input image. torch.Tensor of shape B x C(4) x H x W
            pose: pose vector. torch.Tensor of shape B x n(3)

        Returns:

        """
        B, n = pose.shape
        a0 = input_image  # B x 4 X H x W
        a1 = pose.reshape(B, n, 1, 1).repeat(1, 1, a0.shape[-2], a0.shape[-1])  # B x 3 x H x W
        a2 = torch.cat((a0, a1), dim=1)  # channel-wise concat

        d2 = self.backbone(a2)

        e0 = self.change_image(d2)
        e1 = self.alpha_mask(d2)
        # e2 = a0 * e0 + e1 * (1 - e0)
        e2 = e1 * a0 + (1 - e1) * e0

        xs = torch.linspace(-1, 1, steps=a0.shape[-1])
        ys = torch.linspace(-1, 1, steps=a0.shape[-2])
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
        identity_appearance_flow = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0)  # 1 x 2 x H x W
        e3 = self.appearance_flow(d2) + identity_appearance_flow
        e4 = F.grid_sample(a0, e3.permute((0, 2, 3, 1)))

        result = {
            'a0': a0,
            'a1': a1,
            'e0': e0,
            'e1': e1,
            'e2': e2,
            'e3': e3,
            'e4': e4,
        }

        return result


class Combiner(nn.Module):
    def __init__(self, conf):
        super(Combiner, self).__init__()
        self.conf = conf
        c_mid = 64

        self.downs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(11, c_mid, kernel_size=7, stride=1, padding=3),
                nn.InstanceNorm2d(c_mid),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(c_mid, c_mid * 2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(c_mid * 2),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(c_mid * 2, c_mid * 4, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(c_mid * 4),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(c_mid * 4, c_mid * 8, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(c_mid * 8),
                nn.ReLU(inplace=True),
            )
        ])

        self.res = nn.Sequential(
            ResnetBlock(c_mid * 8),
            ResnetBlock(c_mid * 8),
            ResnetBlock(c_mid * 8),
            ResnetBlock(c_mid * 8),
            ResnetBlock(c_mid * 8),
            ResnetBlock(c_mid * 8),
        )

        self.ups = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(c_mid * 8, c_mid * 4, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(c_mid * 4),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(c_mid * 8, c_mid * 2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(c_mid * 2),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(c_mid * 4, c_mid, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(c_mid),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                # nn.ConvTranspose2d(c1 * 2, c1, kernel_size=4, stride=2, padding=1),
                # TODO diffrernt with original implementation
                nn.Conv2d(c_mid * 2, c_mid, kernel_size=7, stride=1, padding=3),
                nn.InstanceNorm2d(c_mid),
                nn.ReLU(inplace=True),
            ),
        ])

        self.combine_alpha_mask = nn.Sequential(
            nn.Conv2d(c_mid, 4, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid(),
        )
        self.change_for_retouch = nn.Sequential(
            nn.Conv2d(c_mid, 4, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
        )
        self.retouch_alpha_mask = nn.Sequential(
            nn.Conv2d(c_mid, 4, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, a0, a1, a2):
        """

        Args:
            a0: torch.Tensor of shape B x C(4) x H(256) x W(256)
            a1: torch.Tensor of shape B x C(4) x H(256) x W(256)
            a2: torch.Tensor of shape B x C(3) x H(256) x W(256)

        Returns:

        """
        a3 = torch.cat((a0, a1, a2), dim=1)  # channel-wise concat

        bs = []
        b = a3
        for down in self.downs:
            b = down(b)
            bs.append(b)

        b3 = b

        c5 = self.res(b3)

        d = c5
        for idx, up in enumerate(self.ups):
            d = up(d)
            if idx < 3:
                d = torch.cat((d, bs[-idx - 2]), dim=1)

        d6 = d

        e0 = self.combine_alpha_mask(d6)
        e1 = self.change_for_retouch(d6)
        e2 = self.retouch_alpha_mask(d6)

        e3 = e0 * a0 + (1 - e0) * a1
        e4 = e2 * e3 + (1 - e2) * e1

        result = {
            'a0': a0,
            'a1': a1,
            'e0': e0,
            'e1': e1,
            'e2': e2,
            'e3': e3,
            'e4': e4,
        }

        return result


class FaceRotator(nn.Module):
    def __init__(self):
        super(FaceRotator, self).__init__()


class THA1(nn.Module):
    def __init__(self, conf):
        super(THA1, self).__init__()
        self.conf = conf

        self.face_morpher = FaceMorpher(None)
        self.face_rotator = TwoAlgorithmFaceRotator(None)
        self.combiner = Combiner(None)

    def forward(self, x, shape, pose):
        """

        Args:
            x: input image. torch.Tensor of shape B x C(4) x H x W
            shape: shape vector. torch.Tensor of shape B x n(3)
            pose: pose vector. torch.Tensor of shape B x n(3)

        Returns:

        """
        morphed_face = self.face_morpher(x, shape)
        rotated_face1, rotated_face2 = self.face_rotator(morphed_face, pose)
        final_face = self.combiner(rotated_face1, rotated_face2, pose)
        return final_face


if __name__ == '__main__':
    a0 = torch.randn(1, 4, 256, 256)  # input image
    a2 = torch.randn(1, 3, 256, 256)  # pose image

    f1 = FaceMorpher()
    y = f1(a0, a2)
    print(y.shape)

    f2 = TwoAlgorithmFaceRotator()
    y1, y2 = f2(a0, a2)
    print(y1.shape)
    print(y2.shape)

    f3 = Combiner()
    y = f3(y1, y2, a2)
    print(y.shape)
