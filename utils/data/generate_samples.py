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
    path_blend = metadata[internal_idx].strip()

    tmp_dir = './data/3d_models/samples'
    os.makedirs(tmp_dir, exist_ok=True)
    blend_dir = os.path.join(tmp_dir, str(internal_idx))
    os.makedirs(blend_dir, exist_ok=True)

    for i in range(1):
        bpy.ops.wm.open_mainfile(filepath=path_blend)

        camera_name = "Camera.001"
        light_name = "Light.001"
        Renderer.init_camera(camera_name=camera_name)
        Renderer.init_light(light_name=light_name)
        Renderer.set_camera_position(camera_name=camera_name, light_name=light_name)
        Renderer.fix_model()
        bpy.data.lights[light_name].energy = 200

        MAX_ROTATION_ANGLE = 20

        # render rest-pose image
        path_base = os.path.join(blend_dir, '0.png')
        Renderer.set_output_path(path_base)
        Renderer.render()

        shapekeys = {}
        poses = {}

        # render a image
        Renderer.change_shapekey('あ', 1)
        Renderer.change_pose('頭', 'X', -MAX_ROTATION_ANGLE)

        path_base = os.path.join(blend_dir, '1.png')
        Renderer.set_output_path(path_base)
        Renderer.render()

        Renderer.change_shapekey('ウィンク', 1)
        Renderer.change_pose('頭', 'Y', MAX_ROTATION_ANGLE)
        path_base = os.path.join(blend_dir, '2.png')
        Renderer.set_output_path(path_base)
        Renderer.render()

        Renderer.change_shapekey('ウィンク右', 1)
        Renderer.change_pose('頭', 'Z', MAX_ROTATION_ANGLE)
        path_base = os.path.join(blend_dir, '3.png')
        Renderer.set_output_path(path_base)
        Renderer.render()

        bpy.ops.wm.quit_blender()


if __name__ == '__main__':
    main2()
