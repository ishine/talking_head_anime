import os

import cv2
import numpy as np
from tqdm import tqdm

def main(dir_root, path_save):
    dir_root = os.path.abspath(dir_root)
    dirnames = sorted(os.listdir(dir_root))

    rets = []

    for dirname in tqdm(dirnames):
        dir_model = os.path.join(dir_root, dirname)
        try:
            img = show_image(dir_model)
            windowname = dirname#.split('__')[1]
            cv2.namedWindow(windowname)
            cv2.moveWindow(windowname, 10, 10)
            cv2.imshow(windowname, img)
            ret = cv2.waitKey(0)
            if chr(ret) == 'q':
                break
            rets.append(f'{dirname}_____{chr(ret)}\n')
            cv2.destroyWindow(windowname)
        except:
            rets.append(f'{dirname}_____x\n')

    with open(path_save, 'w', encoding='utf-8') as f:
        f.writelines(rets)


def show_image(dir_model):
    imgs = sorted([os.path.join(dir_model, file) for file in os.listdir(dir_model) if file.endswith('.png')])
    if len(imgs) != 5:
        raise AssertionError
    imgs = [cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED) for path in
            imgs]
    imgs.append(np.zeros_like(imgs[0]))
    img1 = np.concatenate(imgs[:3], axis=-2)
    img2 = np.concatenate(imgs[3:], axis=-2)
    img = np.concatenate((img1, img2), axis=-3)
    return img


if __name__ == '__main__':
    main(r'metadata/test_images_models', 'metadata/filtered_models.txt')
