import argparse
import glob
from multiprocessing import Pool
import os
import subprocess

from tqdm import tqdm, trange


def save(params):
    # remove core files generated from script
    error_files = glob.glob('utils/core.*')
    for f in error_files:
        os.remove(f)

    # run script
    path_model, path_blend = params
    command = f'python -m utils.data.save_to_blends "{path_model}" "{path_blend}"'
    subprocess.call(
        command, shell=True,
        stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb')
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, default=1,
                        help='number of processes to use')

    parser.add_argument('--models_text', type=str, default='data/3d_models/all_models.txt',
                        help='path to file containing list of available models')

    parser.add_argument('--models_dir', type=str, default='data/3d_models/models',
                        help='dir where models belong, will be replaced to args.blends_dir')
    parser.add_argument('--blends_dir', type=str, default='data/3d_models/blends',
                        help='dir where blend files will be saved, replaced from args.models_dir')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    with open(args.models_text, 'r', encoding='utf-8') as f:
        data = f.readlines()
    path_models = [item.strip() for item in data]

    path_blends = [item.replace(args.models_dir, args.blends_dir) + '.blend'
                   for item in path_models]

    pool = Pool(processes=args.processes)
    for _ in tqdm(pool.imap_unordered(save, zip(
            path_models, path_blends))):
        pass
