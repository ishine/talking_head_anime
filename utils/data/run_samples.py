import argparse
import glob
from itertools import cycle
import os
import subprocess
from multiprocessing import Pool

from tqdm import tqdm, trange


def run(params):
    # remove core files generated from script
    error_files = glob.glob('utils/core.*')
    for f in error_files:
        os.remove(f)

    # run script
    idx = params[0]
    path_metadata = params[1]
    command = f'python -m utils.data.generate_samples {path_metadata} {idx}'
    subprocess.call(
        command, shell=True,
        stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb')
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, default=1,
                        help='number of processes to use')

    parser.add_argument('--blends_text', type=str, default='data/3d_models/all_valid_blends.txt',
                        help='path to file containing list of available models')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    path_metadata = os.path.abspath(args.blends_text)
    with open(path_metadata, 'r', encoding='utf-8') as f:
        metadata = f.readlines()
    metadata = metadata

    print(len(metadata))
    pool = Pool(processes=args.processes)
    for _ in tqdm(pool.imap_unordered(run, zip(
            list(range(len(metadata))), cycle([path_metadata])
    ))):
        pass
