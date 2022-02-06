import os
import random
import subprocess
import time
import sys

import cv2
import torch

from datasets.base import BaseDataset
from datasets.render import Renderer
from utils.data.filter import find_model_in_dir
from utils.util import suppress_stdout


class SubprocessDataset(BaseDataset):
    def __init__(self, conf):
        super(SubprocessDataset, self).__init__(conf)
        with open(self.conf.path['metadata'], 'r', encoding='utf-8') as f:
            valid_models = f.readlines()

        data = [os.path.join(self.conf.path['root'], line.strip()) for line in valid_models]
        self.data = list(range(len(data)))

        train_split_idx = int(len(self.data) * 0.9)
        if self.conf.mode == 'train':
            self.data = self.data[:train_split_idx]
        elif self.conf.mode == 'eval':
            self.data = self.data[train_split_idx:]
        elif self.conf.mode == 'all':
            pass
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.data)

    @staticmethod
    def np_img_to_torch(img):
        return torch.from_numpy(img).permute((2, 0, 1)) / 255.

    def __getitem__(self, idx):
        return_data = {}
        model_idx = self.data[idx]

        key_mouth = 'あ'
        val_mouth = random.random()
        val_mouth = float(f'{val_mouth:.04f}')

        key_left_eye = 'ウィンク'
        val_left_eye = random.random()
        val_left_eye = float(f'{val_left_eye:.04f}')

        key_right_eye = 'ウィンク右'
        val_right_eye = random.random()
        val_right_eye = float(f'{val_right_eye:.04f}')

        return_data['pose'] = torch.FloatTensor([val_mouth, val_left_eye, val_right_eye])

        commands = [
            'python', '-m', 'datasets.script',
            f'{self.conf.path["metadata"]}', f'{model_idx}',
            f'{key_mouth}___{val_mouth}',
            f'{key_left_eye}___{val_left_eye}',
            f'{key_right_eye}___{val_right_eye}',
        ]
        command = ' '.join(commands)
        subprocess.call(command, shell=True, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))

        tmp_dir = self.conf.path['tmp']

        tmp_path = os.path.join(tmp_dir, f'{model_idx}_{val_mouth}_{val_left_eye}_{val_right_eye}.png')
        while not os.path.exists(tmp_path):
            # time.sleep(0.5)
            print('waiting', tmp_path)
            raise TimeoutError
        img_target = cv2.imread(tmp_path, cv2.IMREAD_UNCHANGED)
        img_target = cv2.cvtColor(img_target, cv2.COLOR_BGRA2RGBA)
        os.remove(tmp_path)
        return_data['img_target'] = self.np_img_to_torch(img_target)

        tmp_path = os.path.join(tmp_dir, f'{model_idx}.png')
        img_base_np = cv2.imread(tmp_path, cv2.IMREAD_UNCHANGED)
        img_base_np = cv2.cvtColor(img_base_np, cv2.COLOR_BGRA2RGBA)
        os.remove(tmp_path)
        return_data['img_base'] = self.np_img_to_torch(img_base_np)

        return return_data


@suppress_stdout
def main():
    """example script call:
    `python -m datasets.script2 ./data/3d_models/filtered_idxs.txt 1 あ___0.7632 ウィンク___0.4486 ウィンク右___0.315`
    renders base image and pose-moved image

    Returns:

    """
    code_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(code_root)
    sys.path.append(os.getcwd())

    path_metadata = sys.argv[1]
    with open(path_metadata, 'r', encoding='utf-8') as f:
        metadata = f.readlines()

    internal_idx = int(sys.argv[2])
    model_idx = metadata[internal_idx].strip()
    model_dir = os.path.join('./data/3d_models', model_idx)
    _, model_path = find_model_in_dir(model_dir)

    # display is needed when (not using cycles engine) and using non-gui environment
    # import pyvirtualdisplay
    # with pyvirtualdisplay.Display(visible=False, size=(1,1)) as disp:
    if True:
        # set renderer
        r = Renderer(make_display=False)
        r.set_addons()
        r.import_model(model_path)
        r.set_camera_position()

        # render rest-pose image
        # tmp_dir = './data/3d_models/tmp'
        tmp_dir = os.path.join(code_root, 'data', '3d_models', 'tmp')
        temp_path = os.path.join(tmp_dir, f'{internal_idx}.png')
        if not os.path.exists(temp_path):
            r.set_output_path(temp_path)
            r.render()

        # change pose
        temp_path = os.path.join(tmp_dir, f'{internal_idx}')
        for item in sys.argv[3:]:
            key, value = item.strip().split('___')
            value = float(value)
            r.change_shapekey(key, value)
            temp_path += f'_{value}'

        # render moved image
        temp_path += '.png'
        r.set_output_path(temp_path)
        r.render()

        # disp.stop()

    r.exit()


if __name__ == '__main__':
    main()
