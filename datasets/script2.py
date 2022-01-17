import os
import sys

import bpy
import pyvirtualdisplay

from datasets.render import Renderer
from datasets.utils.filter import find_model_in_dir
from utils.util import suppress_stdout


def test_render(model_path: str, dir_temp: str = './result_temp'):
    """

    Args:
        model_path: path to model. recommended to give as absolute path
        dir_temp: temporary dir to save rendered images. recommended to give as absolute path

    Returns:

    """
    model_path = os.path.abspath(model_path)
    dir_temp = os.path.abspath(dir_temp)
    print(model_path)
    print(dir_temp)
    os.makedirs(dir_temp, exist_ok=True)
    r = Renderer()
    r.import_model(model_path)

    # set camera position

    # base image # TODO check if rest pose
    r.set_output_path(os.path.join(dir_temp, 'base.png'))
    r.render()

    # find bpy object with shape_keys
    for key, obj in bpy.data.objects.items():
        # set camera position
        if key == 'camera' or key == 'light':
            continue
        print(key, obj)
        # direction = -y -> +y

        # select single object
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # object with shape_keys
        if hasattr(obj.data, 'shape_keys') and obj.data.shape_keys is not None:
            for shape_key in ['あ', 'ウィンク', 'ウィンク右']:
                try:
                    r.change_shapekey(shape_key, 1.0)
                    r.set_output_path(os.path.join(dir_temp, f'{key}.{shape_key}.png'))
                    r.render()
                    r.change_shapekey(shape_key, 0.0)
                except Exception as e:
                    print(e)
                    pass

        # object with poses
        print(obj)
        if hasattr(obj, 'pose') and hasattr(obj.pose, 'bones') and obj.pose.bones is not None:
            # rotate head
            r.poseRig(
                obj,
                [
                    ('頭', 'Y', 60)
                ]
            )
            r.set_output_path(os.path.join(dir_temp, f'{key}.Y60.png'))
            r.render()

            r.poseRig(
                obj,
                [
                    ('頭', 'Y', -60)
                ]
            )
        obj.select_set(False)

    r.exit()


@suppress_stdout
def main():
    code_root = '/root/talking_head_anime'
    os.chdir(code_root)
    sys.path.append(os.getcwd())

    path_metadata = sys.argv[1]
    with open(path_metadata, 'r', encoding='utf-8') as f:
        metadata = f.readlines()

    internal_idx = int(sys.argv[2])
    model_idx = metadata[internal_idx].strip()
    model_dir = os.path.join('/raid/vision/dhchoi/data/3d_models', model_idx)
    _, model_path = find_model_in_dir(model_dir)

    # with pyvirtualdisplay.Display(visible=False, size=(1,1)) as disp:
    if True:
        r = Renderer(make_display=False)
        r.set_addons()
        r.import_model(model_path)
        r.set_camera_position()

        tmp_dir = '/raid/vision/dhchoi/data/3d_models/tmp'
        temp_path = os.path.join(tmp_dir, f'{internal_idx}.png')
        if not os.path.exists(temp_path):
            r.set_output_path(temp_path)
            r.render()

        # disp.stop()

    # temp_path = os.path.join(tmp_dir, f'{internal_idx}')
    # for item in sys.argv[2:]:
    #     key, value = item.strip().split('___')
    #     value = float(value)
    #     r.change_shapekey(key, value)
    #     temp_path += f'_{value}'
    #
    # # temp_path = sys.argv[-1]
    # temp_path += '.png'
    # r.set_output_path(temp_path)
    # r.render()
    r.exit()


if __name__ == '__main__':
    main()
