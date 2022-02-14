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

        with open(self.conf.path['metadata'], 'r', encoding='utf-8') as f:
            data = f.readlines()

        self.dirs = []
        for idx, line in enumerate(data):
            model, label = line.strip().split('|')
            if label == 'L' or label == 'R':
                self.dirs.append({
                    'model': model,
                    'idx': idx,
                    'label': label
                })

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
        for model_data in self.dirs:
            dir_imgs = os.path.join(self.conf.path['root'], str(model_data['idx']))
            for file in os.listdir(dir_imgs):
                if file.startswith('pose'):
                    path_img = os.path.join(dir_imgs, file)
                    self.data.append((path_img, model_data))

    def __len__(self):
        return len(self.data)

    @staticmethod
    def read_img(path_img, imsize=256):
        img = cv2.imread(path_img, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (imsize, imsize))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        torch_img = torch.from_numpy(img).permute((2, 0, 1)) / 255.
        return torch_img

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

        path_pose, model_data = self.data[idx]  # example: pose_0.8972_0.5354_0.3292_6_21_5.png
        dir_model = os.path.dirname(path_pose)

        feature = os.path.basename(path_pose)[:-4].split('_')[1:]

        shape = feature[:3]
        pose = feature[3:]

        path_base = os.path.join(dir_model, 'base.png')
        path_shape = os.path.join(dir_model, f'shape_{shape[0]}_{shape[1]}_{shape[2]}.png')

        img_base = self.read_img(path_base, imsize=self.conf.imsize)
        img_shape = self.read_img(path_shape, imsize=self.conf.imsize)
        img_pose = self.read_img(path_pose, imsize=self.conf.imsize)

        img_base, img_shape, img_pose = self.augmentation((img_base, img_shape, img_pose))

        return_data['img_base'] = img_base
        return_data['img_shape'] = img_shape
        return_data['img_pose'] = img_pose

        if model_data['label'] == 'L':
            return_data['shape'] = torch.FloatTensor([float(shape[0]), float(shape[1]), float(shape[2])])
        else:  # left-right eye changed
            return_data['shape'] = torch.FloatTensor([float(shape[0]), float(shape[2]), float(shape[1])])

        return_data['pose'] = torch.FloatTensor([float(val) for val in pose])

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

    conf = OmegaConf.load('configs/datasets/custom.yaml')
    conf.mode = 'all'
    d = LRLabeledDataset(conf)
    loader = DataLoader(d, batch_size=4, num_workers=4)
    it = cycle(loader)

    from tqdm import trange

    for i in trange(20):
        item = next(it)
