import os
import sys

import bpy
import pyvirtualdisplay

from datasets.render import Renderer
from datasets.utils.filter import find_model_in_dir
from utils.util import suppress_stdout


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
        # set renderer
        r = Renderer(make_display=False)
        r.set_addons()
        r.import_model(model_path)
        r.set_camera_position()

        # render rest-pose image
        tmp_dir = '/raid/vision/dhchoi/data/3d_models/tmp'
        temp_path = os.path.join(tmp_dir, f'{internal_idx}.png')
        if not os.path.exists(temp_path):
            r.set_output_path(temp_path)
            r.render()

        # render moved image
        temp_path = os.path.join(tmp_dir, f'{internal_idx}')
        for item in sys.argv[3:]:
            key, value = item.strip().split('___')
            value = float(value)
            r.change_shapekey(key, value)
            temp_path += f'_{value}'

        # temp_path = sys.argv[-1]
        temp_path += '.png'
        r.set_output_path(temp_path)
        r.render()

        # disp.stop()

    r.exit()


if __name__ == '__main__':
    main()
