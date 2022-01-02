from collections import Counter, defaultdict
import json
import os
import shutil
import subprocess

import bpy
import rarfile
from tqdm import tqdm, trange
import zipfile

from datasets.render import Renderer


# region old

def see_extensions(dir_root):
    extensions = [file.rsplit('.', 1)[-1].lower() for file in os.listdir(dir_root)]
    print(Counter(extensions))


def remove_non_archives(dir_root):
    models = [os.path.join(dir_root, file) for file in os.listdir(dir_root)
              if os.path.isfile(os.path.join(dir_root, file))]
    for path_model in tqdm(models):
        model_name, extension = path_model.rsplit('.', 1)
        extension = extension.lower()
        if extension in ['wav', 'avi', 'jpg', 'txt', 'png', 'mp4', 'mp3', 'flv', 'bmp', 'gif']:
            os.remove(path_model)


# endregion

def find_model_in_dir(dir_model: str):
    """ searches if dir_model has loadable file

    Args:
        dir_model: root dir to find loadable file

    Returns:
        result: Boolean. True if exists, else False
        path_model: str. if result is true, valid model path is given. Else returns ''

    """
    result = False
    result_path = ''
    for root, subdirs, files in os.walk(dir_model):
        for file in files:
            path_model = os.path.join(root, file)
            if file.endswith('.pmx'):
                result = True
                result_path = path_model
                break
            elif file.endswith('.pmd'):
                result = True
                result_path = path_model
                break
            elif file.endswith('.vrm'):
                result = True
                result_path = path_model
                break

        if result:
            break

    return result, result_path


def find_valid_dirs(dir_root: str, path_save: str):
    """ finds subdir names which contains loadable files (.pmx, .pmd, .vrm)

    Args:
        dir_root: root dir to search
        path_save: path to save indices

    Returns:

    """
    models = sorted([os.path.join(dir_root, file) for file in os.listdir(dir_root)
                     if os.path.isdir(os.path.join(dir_root, file))])

    pmxs = []
    pmds = []
    vrms = []

    for dir_model in models:
        result_bool, result_path = find_model_in_dir(dir_model)
        if result_bool:
            if result_path.endswith('.pmx'):
                pmxs.append(dir_model)
            elif result_path.endswith('.pmd'):
                pmds.append(dir_model)
            elif result_path.endswith('.vrm'):
                vrms.append(dir_model)

    print(len(models), len(pmxs), len(pmds), len(vrms))

    valid_list = sorted(pmxs + pmds + vrms)
    valid_list = [os.path.basename(dirname) + '\n' for dirname in valid_list]
    with open(path_save, 'w', encoding='utf-8') as f:
        f.writelines(valid_list)


def remove_unsupported_dirs(dir_root: str, path_idxs: str):
    """remove dirs recursively with no supported files

    Args:
        dir_root: str, root dir
        path_idxs: str. path to metadata file which contains idx of dir which contains loadable file

    Returns:

    """
    models = os.listdir(dir_root)

    with open(path_idxs, 'r', encoding='utf-8') as f:
        valid_list = f.readlines()
    valid_list = [line.strip() for line in valid_list]

    for idx, model in enumerate(tqdm(models)):
        if model not in valid_list:
            dir_model = os.path.join(dir_root, model)
            try:
                if os.path.exists(dir_model) and os.path.isdir(dir_model):
                    shutil.rmtree(dir_model)
            except Exception as e:
                print(idx, e)


def get_metadata(path_model: str):
    """get keys from single model

    Args:
        path_model:

    Returns:
        data: dict

    """

    # set env
    r = Renderer()
    r.import_model(path_model=path_model)

    data = {key: {} for key in bpy.data.objects.keys()}
    for key, obj in bpy.data.objects.items():
        # select single object
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # find and get shape keys
        if hasattr(obj.data, 'shape_keys') and obj.data.shape_keys is not None:
            data[key]['shape_keys'] = obj.data.shape_keys.key_blocks.keys()

        # find and get pose(bones)
        if hasattr(obj, 'pose') and hasattr(obj.pose, 'bones') and obj.pose.bones is not None:
            bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.mode_set(mode='POSE')
            data[key]['bones'] = obj.pose.bones.keys()
        bpy.ops.object.mode_set(mode='OBJECT')

        # job finished
        obj.select_set(False)

    r.clear()
    return data


def extract_shapekeys(dir_model: str, path_save: str):
    """ find and save shape_keys and bones(poses)

    Args:
        dir_model: dir of model
        path_save: path to save metadata

    Returns:

    """
    # find loadable model

    _, path_model = find_model_in_dir(dir_model)

    metadata = get_metadata(path_model)

    with open(path_save, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)


def count_keys(dir_root: str, path_save: str):
    """ count keys in metadata

    Args:
        dir_root: dir of saved metadata files
        path_save: path to save

    Returns:

    """

    data = defaultdict(list)

    for file in sorted(os.listdir(dir_root)):
        with open(os.path.join(dir_root, file), 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            for object_key, object_value in metadata.items():
                for key in object_value:
                    data[key].append(object_key)

    counts = [f'{key}, {len(value)}\n' for key, value in data.items()]
    counts = sorted(counts, key=lambda x: int(x.strip().rsplit(', ')[-1]), reverse=True)
    with open(path_save, 'w', encoding='utf-8') as f:
        f.writelines(counts)


if __name__ == '__main__':
    dir_root = '/raid/vision/dhchoi/data/3d_models/models'
    dir_metadata = '/root/talking_head_anime/metadata/models'
    os.makedirs(dir_metadata, exist_ok=True)
    path_idxs = os.path.join(dir_metadata, 'idxs.txt')
    find_valid_dirs(dir_root, path_save=path_idxs)
    remove_unsupported_dirs(dir_root, path_idxs)

    # batch-extract shape_keys and poses
    with open(path_idxs, 'r', encoding='utf-8') as f:
        model_idxs = f.readlines()

    dir_shapekeys = os.path.join(dir_metadata, 'shape_keys')
    os.makedirs(dir_shapekeys, exist_ok=True)
    for idx, model_name in enumerate(model_idxs):
        model_name = model_name.strip()
        dir_model = os.path.join(dir_root, model_name)
        if os.path.isdir(dir_model):
            try:
                path_shape_keys = os.path.join(dir_shapekeys, f'{model_name}.json')
                extract_shapekeys(dir_model, path_shape_keys)
            except Exception as e:
                print(e)
                break
                pass

    path_shape_keys = os.path.join(dir_metadata, 'shape_keys.txt')
    count_keys(dir_shapekeys, path_save=path_shape_keys)

    print('###########finished############')
