import os
import sys
import subprocess

import bpy

from datasets.render import Renderer


def get_shapekeys(camera_name: str = "Camera", light_name: str = "Light"):
    shape_keys = []
    for obj_key, obj in bpy.data.objects.items():
        if obj_key == camera_name or obj_key == light_name:
            continue

        if hasattr(obj, 'data') and hasattr(obj.data, 'shape_keys'):
            # shape_keys = obj.data.shape_keys
            shape_keys = obj.data.shape_keys.key_blocks.keys()
            return shape_keys

    return shape_keys


def get_poses(camera_name: str = "Camera", light_name: str = "Light"):
    poses = None
    for obj_key, obj in bpy.data.objects.items():
        if obj_key == camera_name or obj_key == light_name:
            continue

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
        obj.select_set(True)  # Select the cube

        if hasattr(obj, 'data') and hasattr(obj.data, 'edit_bones'):
            bpy.ops.object.mode_set(mode='EDIT')
            poses = obj.data.edit_bones.keys()
            bpy.ops.object.mode_set(mode='OBJECT')
            break

    bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
    return poses


def save_one_blend(path_model, path_blend):
    Renderer.clean_blender()
    Renderer.set_addons()
    Renderer.init_camera()
    Renderer.init_light()
    Renderer.set_configs()
    Renderer._import_model(path_model)
    Renderer.fix_model()

    base_path = path_blend[:-6]  # path_blend endswith .blend
    os.makedirs(os.path.dirname(path_blend), exist_ok=True)

    shapekeys = get_shapekeys()
    shape_path = base_path + '.shape.txt'
    with open(shape_path, 'w', encoding='utf-8') as f:
        shapekeys_write = [item + '\n' for item in shapekeys]
        f.writelines(shapekeys_write)

    poses = get_poses()
    pose_path = base_path + '.pose.txt'
    with open(pose_path, 'w', encoding='utf-8') as f:
        poses_write = [item + '\n' for item in poses]
        f.writelines(poses_write)

    if os.path.exists(path_blend):
        os.remove(path_blend)
    bpy.ops.wm.save_as_mainfile(filepath=path_blend)

    bpy.ops.wm.quit_blender()

    return path_blend


def save(params):
    path_model, path_blend = params
    command = f'python -m utils.data.save_to_blends "{path_model}" "{path_blend}"'
    subprocess.call(
        command, shell=True,
        stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb')
    )


if __name__ == '__main__':
    path_model = sys.argv[-2]
    path_blend = sys.argv[-1]
    save_one_blend(path_model, path_blend)
