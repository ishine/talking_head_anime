import os

import cv2
import numpy as np
from tqdm import tqdm, trange


def show_and_get(img, name):
    try:
        cv2.namedWindow(name)
        cv2.moveWindow(name, 0, 0)
        cv2.imshow(name, img)
        ret = chr(cv2.waitKey(0))
        cv2.destroyWindow(name)
    except Exception as e:
        raise e

    return ret


def main():
    path_metadata = 'data/3d_models/all_valid_blends.txt'
    with open(path_metadata, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    names = [line.strip() for line in lines]

    path_samples = 'data/3d_models/samples/'

    rets = []
    for i in trange(len(names)):
        name = names[i].split('/', 7)[-2]

        imgs = []
        for j in range(4):
            path_im = os.path.join(path_samples, str(i), f'{j}.png')
            im = cv2.imread(path_im, cv2.IMREAD_UNCHANGED)[...]
            imgs.append(im)
        imgs = np.concatenate(imgs, axis=1)
        ret = show_and_get(imgs, name)
        if ret == 'q':
            return
        rets.append(f'{names[i]}|{ret}\n')

    path_save = 'data/3d_models/filtered_idxs.txt'
    with open(path_save, 'w', encoding='utf-8') as f:
        f.writelines(rets)


if __name__ == '__main__':
    main()
