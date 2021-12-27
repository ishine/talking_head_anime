from collections import Counter
import os
import shutil
import subprocess

import rarfile
from tqdm import tqdm, trange
import zipfile


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


def valid_dirs(dir_root, path_save):
    models = [os.path.join(dir_root, file) for file in os.listdir(dir_root)
              if os.path.isdir(os.path.join(dir_root, file))]

    pmxs = []
    pmds = []
    vrms = []

    for dir_model in tqdm(models):
        for root, subdirs, files in os.walk(dir_model):
            break_flag = False
            for file in files:
                if file.endswith('.pmx'):
                    pmxs.append(dir_model)
                    break_flag = True
                    break
                elif file.endswith('.pmd'):
                    pmds.append(dir_model)
                    break_flag = True
                    break
                elif file.endswith('.vrm'):
                    vrms.append(dir_model)
                    break_flag = True
                    break

            if break_flag:
                break

    valid_list = list(set(sorted(pmxs + pmds + vrms)))
    valid_list = [os.path.basename(dirname) + '\n' for dirname in valid_list]
    print(len(models), len(pmxs), len(pmds), len(vrms), len(valid_list))
    with open(path_save, 'w', encoding='utf-8') as f:
        f.writelines(valid_list)


def remove_unsupported_files(dir_root, path_idxs):
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


if __name__ == '__main__':
    dir_root = '/raid/vision/dhchoi/data/3d_models/models'
    # dir_root = 'D:\\media\\3d_models\\models'

    # remove_non_archives(dir_root)
    path1 = '../../metadata/nicovideo.txt'
    valid_dirs(dir_root, path_save=path1)

    remove_unsupported_files(dir_root, path1)
