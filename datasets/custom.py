import os
import random

import torch

from datasets.base import BaseDataset
from datasets.render import Renderer
from datasets.utils.filter import find_model_in_dir


class WituGUIDataset(BaseDataset):
    def __init__(self, conf):
        super(WituGUIDataset, self).__init__(conf)
        self.renderer = Renderer(make_display=self.conf.make_display)

        with open(self.conf.path['metadata'], 'r', encoding='utf-8') as f:
            valid_models = f.readlines()

        self.data = [os.path.join(self.conf.path['root'], line.strip())
                     for line in valid_models]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def np_img_to_torch(img):
        return torch.from_numpy(img).permute((2, 0, 1)) / 255.

    def getitem(self, idx):
        return_data = {}

        dir_model = self.data[idx]
        _, path_model = find_model_in_dir(dir_model)
        self.renderer.import_model(path_model)
        self.renderer.set_camera_position()

        # self.renderer.augmentation()

        img_base_np = self.renderer.render_to_numpy_array()

        pose_mouth = random.random()
        pose_left_eye = random.random()
        pose_right_eye = random.random()

        key_mouth = 'あ'
        key_left_eye = 'ウィンク'
        key_right_eye = 'ウィンク右'

        # pose_head_x = (random.random() - 0.5) * 2 * 45 # [-45, 45], in degrees
        # pose_head_y = (random.random() - 0.5) * 2 * 45  # [-45, 45], in degrees
        # pose_head_z = (random.random() - 0.5) * 2 * 45  # [-45, 45], in degrees

        self.renderer.change_shapekey(key_mouth, pose_mouth)
        self.renderer.change_shapekey(key_left_eye, pose_left_eye)
        self.renderer.change_shapekey(key_right_eye, pose_right_eye)

        img_target = self.renderer.render_to_numpy_array()

        return_data['img_base'] = self.np_img_to_torch(img_base_np)
        return_data['pose'] = torch.Tensor([pose_mouth, pose_left_eye, pose_right_eye])
        return_data['img_target'] = self.np_img_to_torch(img_target)

        self.renderer.clean_blender()

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
