import math
import os
import random
import sys

import bpy

from datasets.render import Renderer
from utils.data.filter import find_model_in_dir
from utils.util import suppress_stdout


def inverse_cdf(val):
    return math.sqrt(10 / 7 * val + 9 / 28 / 7) - 3 / 14


@suppress_stdout
def main2():
    code_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # os.chdir(code_root)
    sys.path.append(code_root)

    path_metadata = sys.argv[1]
    with open(path_metadata, 'r', encoding='utf-8') as f:
        metadata = f.readlines()

    internal_idx = int(sys.argv[2])
    print(metadata[internal_idx])
    path_blend, label = metadata[internal_idx].strip().split('|')

    if label.lower() != 'l' and label != 'r':
        return

    tmp_dir = './data/3d_models/imgset'
    os.makedirs(tmp_dir, exist_ok=True)
    blend_dir = os.path.join(tmp_dir, str(internal_idx))
    os.makedirs(blend_dir, exist_ok=True)

    for i in range(50):
        bpy.ops.wm.open_mainfile(filepath=path_blend)

        camera_name = "Camera.001"
        light_name = "Light.001"
        Renderer.init_camera(camera_name=camera_name)
        Renderer.init_light(light_name=light_name)
        Renderer.set_camera_position(camera_name=camera_name, light_name=light_name)
        Renderer.fix_model()
        bpy.data.lights[light_name].energy = 200

        # render rest-pose image
        path_base = os.path.join(blend_dir, 'base.png')
        Renderer.set_output_path(path_base)
        if i == 0:
            Renderer.render()

        shapekeys = {}
        shapekeys['あ'] = float(f'{random.random():.04f}')
        shapekeys['ウィンク'] = float(f'{random.random():.04f}')
        shapekeys['ウィンク右'] = float(f'{random.random():.04f}')

        poses = {}
        MAX_ROTATION_ANGLE = 30
        poses['X'] = int((random.random() * 2 - 1) * MAX_ROTATION_ANGLE)
        poses['Y'] = int((random.random() * 2 - 1) * MAX_ROTATION_ANGLE)
        poses['Z'] = int((random.random() * 2 - 1) * MAX_ROTATION_ANGLE)

        # set shape
        path_shape = os.path.join(blend_dir, 'shape')
        path_pose = os.path.join(blend_dir, 'pose')
        for shape_key, shape_value in shapekeys.items():
            Renderer.change_shapekey(shape_key, shape_value)
            path_shape += f'_{shape_value}'
            path_pose += f'_{shape_value}'
        path_shape += '.png'
        Renderer.set_output_path(path_shape)
        Renderer.render()

        # set pose
        for pose_key, pose_value in poses.items():
            Renderer.change_pose('頭', pose_key, pose_value)
            path_pose += f'_{pose_value}'
        path_pose += '.png'
        Renderer.set_output_path(path_pose)
        Renderer.render()

        bpy.ops.wm.quit_blender()


if __name__ == '__main__':
    main2()
