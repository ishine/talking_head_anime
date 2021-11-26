import argparse
import glob
import os

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from utils.util import save_files, build_models_from_config, build_datasets_from_config
from utils.logging import get_logger


class Trainer:
    def __init__(self, conf):
        super(Trainer, self).__init__()
        self.conf = conf
        self.device = self.conf.logging.device
        self.save_files()

        self.build_datasets()
        self.build_models()

        self.set_logger()

    # region init
    def save_files(self):
        savefiles = []
        for glob_path in self.conf.logging['savefiles']:
            savefiles += glob.glob(glob_path)
        save_files(self.conf.logging['log_dir'], savefiles)

    def build_datasets(self):
        datasets, loaders, iterators = build_datasets_from_config(self.conf.datasets)
        self.datasets = datasets
        self.loaders = loaders
        self.iterators = iterators

    def build_models(self):
        models, optims = build_models_from_config(self.conf.models)

        for key, value in models:
            models[key] = value.to(self.device)
        self.models = models
        self.optims = optims

    def set_logger(self):
        logger = get_logger(self.conf.logging)
        self.logger = logger
        self.global_step = 0
        self.global_epoch = 0

    # endregion

    def run(self):
        pass

    def training_step(self, batch):
        loss = {}
        logs = {}
        logs.update(batch)

        # input image: resting input image
        # pose1: pose image with left eye, right eye and mouth
        # pose2: pose image with neck tip x-rotation, neck tip y-rotation, and neck root z-rotation
        gt_rest_img = batch['gt_rest_img'].to(self.device)
        pose1 = batch['pose1'].to(self.device)

        gt_morphed_img = batch['gt_morphed_img']

        gen_morphed_img = self.models['FaceMorpher'](gt_rest_img, pose1)
        logs['gen_morphed_img'] = gen_morphed_img.detach().cpu()

        loss['l1_morph'] = F.l1_loss(gen_morphed_img, gt_morphed_img)
        loss['backward'] = loss['l1_morph']

        self.optims['FaceMorpher'].zero_grad()
        loss['backward'].backward()
        self.optims['FaceMorpher'].step()

        if self.global_step % self.conf.logging.freq['log'] == 0:
            self.logger.write_log(logs, mode='train', step=self.global_step)
            self.logger.write_loss(loss, mode='train', step=self.global_step)


@torch.no_grad()
def inference():
    pass


def main():
    args = parse_args()
    conf = OmegaConf.load(args.config)

    conf.logging['log_dir'] = os.path.join(conf.logging['log_dir'], str(conf.logging['seed']))
    os.makedirs(conf.logging['log_dir'], exist_ok=True)

    if args.train:
        trainer = Trainer(conf)
        trainer.run()

    if args.infer:
        inference()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_tha.yaml',
                        help='config file path')

    parser.add_argument('--train', action='store_true',
                        help='')
    parser.add_argument('--infer', action='store_true',
                        help='')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
