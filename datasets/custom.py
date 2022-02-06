import os
import random

import cv2
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm, trange

from datasets.base import BaseDataset


class ImageDataset(BaseDataset):
    def __init__(self, conf):
        super(ImageDataset, self).__init__(conf)

        self.dirs = [os.path.join(self.conf.path['root'], path)
                     for path in os.listdir(self.conf.path['root'])
                     if os.path.isdir(os.path.join(self.conf.path['root'], path))]

        train_split_idx = int(len(self.dirs) * 0.9)
        if self.conf.mode == 'train':
            self.dirs = self.dirs[:train_split_idx]
        elif self.conf.mode == 'eval':
            self.dirs = self.dirs[train_split_idx:]
        elif self.conf.mode == 'all':
            pass
        else:
            raise NotImplementedError

        self.data = []
        for path_dir in self.dirs:
            files = [os.path.join(path_dir, file) for file in os.listdir(path_dir)
                     if '_' in file]
            self.data.extend(files)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def np_img_to_torch(img):
        return torch.from_numpy(img).permute((2, 0, 1)) / 255.

    def augmentation(self, images):
        brightness_scale = 0.5
        brightness = random.uniform(1 - brightness_scale, 1 + brightness_scale)
        contrast_scale = 0.5
        contrast = random.uniform(1 - contrast_scale, 1 + contrast_scale)
        saturation_scale = 0.5
        saturation = random.uniform(1 - saturation_scale, 1 + saturation_scale)
        hue_scale = 0.3
        hue = random.uniform(-hue_scale, hue_scale)

        new_images = []
        for image in images:
            alpha = image[-1].clone().unsqueeze(0)
            new_image = image[:-1].clone()
            new_image = TF.adjust_brightness(new_image, brightness)
            new_image = TF.adjust_contrast(new_image, contrast)
            new_image = TF.adjust_saturation(new_image, saturation)
            new_image = torch.cat((new_image, alpha), dim=0)
            # new_image = TF.adjust_hue(new_image, hue)
            new_images.append(new_image)

        return new_images

    def getitem(self, idx):
        return_data = {}

        path_pose = self.data[idx]
        path_base = os.path.join(os.path.dirname(path_pose), os.path.basename(path_pose).split('_')[0] + '.png')

        img_base_np = cv2.imread(path_base, cv2.IMREAD_UNCHANGED)
        try:
            assert img_base_np.shape == (512, 512, 4), f'{path_base}, {img_base_np.shape}'
        except:
            print(path_pose, path_base)
        img_base_np = cv2.resize(img_base_np, (256, 256))
        img_base_np = cv2.cvtColor(img_base_np, cv2.COLOR_BGRA2RGBA)
        img_base = self.np_img_to_torch(img_base_np)

        img_target_np = cv2.imread(path_pose, cv2.IMREAD_UNCHANGED)
        assert img_target_np.shape == (512, 512, 4), f'{path_pose}, {img_target_np.shape}'
        img_target_np = cv2.resize(img_target_np, (256, 256))
        img_target_np = cv2.cvtColor(img_target_np, cv2.COLOR_BGRA2RGBA)
        img_target = self.np_img_to_torch(img_target_np)

        img_base, img_target = self.augmentation((img_base, img_target))

        pose = path_pose.rsplit('.', 1)[0].rsplit('_', 3)[-3:]
        pose = [float(val) for val in pose]
        assert len(pose) == 3, path_pose

        return_data['img_base'] = img_base
        return_data['img_target'] = img_target
        return_data['pose'] = torch.FloatTensor(pose)

        return return_data


class PlaceholderDataset(BaseDataset):
    def __init__(self, conf):
        super(PlaceholderDataset, self).__init__(conf)

    def __len__(self):
        return 10000

    def getitem(self, idx):
        return_data = {}

        return_data['img_base'] = torch.randn((4, 256, 256))
        return_data['img_target'] = torch.randn((4, 256, 256))
        return_data['pose'] = torch.Tensor([random.random(), random.random(), random.random()])

        return return_data


if __name__ == '__main__':
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from utils.util import cycle

    import sys

    code_root = '/root/talking_head_anime'
    os.chdir(code_root)
    sys.path.append(os.getcwd())

    conf = OmegaConf.load('configs/datasets/custom.yaml')
    d = ImageDataset(conf)
    loader = DataLoader(d, batch_size=4, num_workers=4)
    it = cycle(loader)

    from tqdm import trange

    for i in trange(20):
        item = next(it)
