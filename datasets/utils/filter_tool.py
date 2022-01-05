import os

import cv2
import numpy as np


def main(dir_root, path_save):
    dir_root = os.path.abspath(dir_root)
    dirnames = sorted(os.listdir(dir_root))

    rets = []

    for dirname in dirnames:
        dir_model = os.path.join(dir_root, dirname)
        try:
            img = show_image(dir_model)
            cv2.imshow(dirname, img)
            ret = cv2.waitKey(0)
            if chr(ret) == 'q':
                break
            rets.append(f'{dirname}_____{chr(ret)}\n')
            cv2.destroyWindow(dirname)
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
    img = np.concatenate(imgs, axis=-2)
    return img


if __name__ == '__main__':
    main(r'metadata/test_images', 'metadata/hand_filtered.txt')
