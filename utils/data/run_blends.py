from multiprocessing import Pool

from tqdm import tqdm, trange

from utils.data.save_to_blends import save

if __name__ == '__main__':
    metadata_models = 'data/3d_models/all_models.txt'

    with open(metadata_models, 'r', encoding='utf-8') as f:
        data = f.readlines()
    path_models = [item.strip() for item in data]
    path_blends = [item.replace('data/3d_models/models', 'data/3d_models/blends_220222') + '.blend'
                   for item in path_models]
    pool = Pool(processes=1)
    for _ in tqdm(pool.imap_unordered(save, zip(
            path_models, path_blends))):
        pass
