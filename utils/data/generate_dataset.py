from itertools import cycle
import os
import subprocess
from multiprocessing import Pool

from tqdm import tqdm, trange


def run2(params):
    idx = params[0]
    path_metadata = params[1]
    command = f'python -m utils.data.script3 {path_metadata} {idx}'
    subprocess.call(
        command, shell=True,
        stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb')
    )

if __name__ == '__main__':
    path_metadata = './data/3d_models/filtered_idxs.txt'
    path_metadata = os.path.abspath(path_metadata)
    with open(path_metadata, 'r', encoding='utf-8') as f:
        metadata = f.readlines()
    metadata = metadata[:10]

    print(len(metadata))
    pool = Pool(processes=1)
    for _ in tqdm(pool.imap_unordered(run2, zip(
            list(range(len(metadata))), cycle([path_metadata])
    ))):
        pass
