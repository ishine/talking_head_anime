import os
import random
import subprocess
import time
import sys

import cv2
import torch
import torchvision.transforms.functional as TF

from datasets.base import BaseDataset
from datasets.render import Renderer
from utils.data.filter import find_model_in_dir
from utils.util import suppress_stdout

import bpy


class BlendDataset(BaseDataset):
    def __init__(self, conf):
        super(BlendDataset, self).__init__(conf)

        with open(self.conf.path['metadata'], 'r', encoding='utf-8') as f:
            metadata = f.readlines()

        self.train_split_idx = int(len(metadata) * 0.9)
        if self.conf.mode == 'train':
            self.data = list(range(0, self.train_split_idx))
        elif self.conf.mode == 'eval':
            self.data = list(range(self.train_split_idx, len(metadata)))
        elif self.conf.mode == 'all':
            pass
        else:
            raise NotImplementedError

        self.MAX_ROTATION_ANGLE = 45

    @staticmethod
    def find_all_blends(dir_model):
        blends_in_dir = []

        for root, subdirs, files in os.walk(dir_model):
            for file in files:
                path_model = os.path.join(root, file)
                extension = file.rsplit('.', 1)[-1].lower()
                if extension == 'blend':
                    if path_model not in blends_in_dir:
                        blends_in_dir.append(path_model)

        return blends_in_dir

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

    def generate_random_shapekeys(self):
        return_dict = {}

        return_dict['あ'] = random.random()
        return_dict['ウィンク'] = random.random()
        return_dict['ウィンク右'] = random.random()

        return return_dict

    def generate_random_poses(self):
        return_dict = {}

        return_dict['X'] = (random.random() * 2 - 1) * self.MAX_ROTATION_ANGLE
        return_dict['Y'] = (random.random() * 2 - 1) * self.MAX_ROTATION_ANGLE
        return_dict['Z'] = (random.random() * 2 - 1) * self.MAX_ROTATION_ANGLE

        return return_dict

    def getitem(self, idx):
        return_data = {}

        dir_temp = './data/tmp'
        path_metadata = self.conf.path.metadata
        internal_idx = self.data[idx]

        # posed image
        random_shapekeys = self.generate_random_shapekeys()
        random_poses = self.generate_random_poses()

        commands = [
            'python', '-m', 'datasets.blends',
            f'{path_metadata}', f'{internal_idx}',
        ]
        shapekey_text = ''
        for key, value in random_shapekeys.items():
            shapekey_text += f'{key}___{value}___'
        shapekey_text = shapekey_text[:-3]
        commands.append(shapekey_text)

        pose_text = ''
        for key, value in random_poses.items():
            pose_text += f'{key}___{value}___'
        pose_text = pose_text[:-3]
        commands.append(pose_text)

        command = ' '.join(commands)
        subprocess.call(command, shell=True, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))

        # base image
        path_base = os.path.join(dir_temp, f'{internal_idx}_base.png')
        img_base_np = cv2.imread(path_base, cv2.IMREAD_UNCHANGED)
        img_base_np = cv2.cvtColor(img_base_np, cv2.COLOR_BGRA2RGBA)
        return_data['img_base'] = self.np_img_to_torch(img_base_np)

        path_target = os.path.join(dir_temp, f'{internal_idx}_pose.png')
        img_target = cv2.imread(path_target, cv2.IMREAD_UNCHANGED)
        img_target = cv2.cvtColor(img_target, cv2.COLOR_BGRA2RGBA)
        os.remove(path_target)
        return_data['img_target'] = self.np_img_to_torch(img_target)

        return_data['metadata'] = {
            'shapekey': random_shapekeys,
            'pose': random_poses,
            'model_path': self.data,
        }

        shape = torch.FloatTensor(list(random_shapekeys.values()))
        pose = torch.FloatTensor(list(random_poses.values()))
        return_data['shape'] = shape
        return_data['pose'] = pose

        return return_data


@suppress_stdout
def getitem_blends():
    code_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(code_root)
    sys.path.append(os.getcwd())

    print('#############', sys.argv)

    path_metadata = sys.argv[1]
    with open(path_metadata, 'r', encoding='utf-8') as f:
        metadata = f.readlines()

    idx = int(sys.argv[2])
    path_blend = metadata[idx].strip()

    shapekeys = sys.argv[3].strip().split('___')
    poses = sys.argv[4].strip().split('___')

    dir_temp = './data/tmp'
    os.makedirs(dir_temp, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=path_blend)

    camera_name = "Camera.001"
    light_name = "Light.001"
    Renderer.init_camera(camera_name=camera_name)
    Renderer.init_light(light_name=light_name)
    Renderer.set_camera_position(camera_name=camera_name, light_name=light_name)
    Renderer.fix_model()

    # base image
    path_base = os.path.join(dir_temp, f'{idx}_base.png')
    Renderer.set_output_path(path_base)
    Renderer.render()

    for i in range(len(shapekeys) // 2):
        shape_key = shapekeys[2 * i]
        shape_value = float(shapekeys[2 * i + 1])
        try:
            Renderer.change_shapekey(shape_key, shape_value)
        except Exception as e:
            pass

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects

    for obj_key, obj in bpy.data.objects.items():
        if obj_key == camera_name or obj_key == light_name:
            continue

        # object with poses
        if hasattr(obj, 'pose') and hasattr(obj.pose, 'bones') and obj.pose.bones is not None:
            # rotate head
            for i in range(len(poses) // 2):
                pose_key = poses[2 * i]
                pose_value = float(poses[2 * i + 1])
                Renderer.rotate_bone(obj, [('頭', pose_key, pose_value)])
            bpy.ops.object.mode_set(mode='OBJECT')
            obj.select_set(False)

    path_target = os.path.join(dir_temp, f'{idx}_pose.png')
    Renderer.set_output_path(path_target)
    Renderer.render()


if __name__ == '__main__':
    getitem_blends()
    # example: python -m datasets.blends data/3d_models/all_blends.txt 1 あ___1 X___30
