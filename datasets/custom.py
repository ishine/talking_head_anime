import os
import random
import subprocess

import cv2
import torch

from datasets.base import BaseDataset


class SubprocessDataset(BaseDataset):
    def __init__(self, conf):
        super(SubprocessDataset, self).__init__(conf)
        with open(self.conf.path['metadata'], 'r', encoding='utf-8') as f:
            valid_models = f.readlines()

        self.data = [os.path.join(self.conf.path['root'], line.strip()) for line in valid_models]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def np_img_to_torch(img):
        return torch.from_numpy(img).permute((2, 0, 1)) / 255.

    def __getitem__(self, idx):
        commands = [
            'python', '-m', 'datasets.script2',
            f'{self.conf.path["metadata"]}',
            f'{idx}',
        ]
        command = ' '.join(commands)
        subprocess.call(command, shell=True, stdout=open(os.devnull, 'wb'))

        return_data = {}

        tmp_dir = self.conf.path['tmp']
        tmp_path = os.path.join(tmp_dir, f'{idx}.png')
        while not os.path.exists(tmp_path):
            time.sleep(0.5)
            print('waiting', tmp_path)
        img_base_np = cv2.imread(tmp_path, cv2.IMREAD_UNCHANGED)
        os.remove(tmp_path)
        return_data['img_base'] = self.np_img_to_torch(img_base_np)

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
    import time
    from torch.utils.data import DataLoader
    from utils.util import cycle

    import sys

    code_root = '/root/talking_head_anime'
    os.chdir(code_root)
    sys.path.append(os.getcwd())

    conf = OmegaConf.load('configs/datasets/custom.yaml')
    d = SubprocessDataset(conf)
    loader = DataLoader(d, batch_size=4, num_workers=4)
    it = cycle(loader)

    from tqdm import trange

    for i in trange(10):
        item = next(it)
