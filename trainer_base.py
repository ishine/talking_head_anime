import argparse
import importlib
import glob
import os

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from utils.util import save_files, build_models_from_config, build_datasets_from_config, cycle
from utils.logging import get_logger


class BaseTrainer:
    def __init__(self, conf):
        super(BaseTrainer, self).__init__()
        self.conf = conf
        self.device = self.conf.logging.device
        self.save_files()

        self.build_datasets()
        self.build_models()
        self.build_losses()

        self.set_logger()

    # region init
    def save_files(self):
        savefiles = []
        for glob_path in self.conf.logging['save_files']:
            savefiles += glob.glob(glob_path)
        save_files(self.conf.logging['log_dir'], savefiles)

    def build_datasets(self):
        datasets, loaders, iterators = build_datasets_from_config(self.conf.datasets)
        self.datasets = datasets
        self.loaders = loaders
        self.iterators = iterators

    def build_models(self):
        models, optims = build_models_from_config(self.conf.models)

        for key, value in models.items():
            models[key] = value.to(self.device)
        self.models = models
        self.optims = optims

    def build_losses(self):
        losses_dict = {}
        losses_dict['L1'] = torch.nn.L1Loss()
        losses_dict['BCE'] = torch.nn.BCEWithLogitsLoss()
        self.losses = losses_dict

    def set_logger(self):
        logger = get_logger(self.conf.logging)
        self.logger = logger
        self.global_step = 0
        self.global_epoch = 0

    # endregion

    # region training
    def train_step(self, data, calc_log=False):
        raise NotImplementedError

    def eval_step(self, data, calc_log=False):
        raise NotImplementedError

    def train_epoch(self):
        pbar_step = trange(len(self.loaders['train']), position=1)
        pbar_step.set_description_str('STEP')

        losses_train = {}
        losses_eval = {}

        for step in pbar_step:
            calc_log = self.global_step % self.conf.logging.freq == 0
            train_data = next(self.iterators['train'])
            loss_train, log_train = self.train_step(train_data, calc_log=calc_log)
            for key, value in loss_train.items():
                if key not in losses_train.keys():
                    losses_train[key] = 0.
                losses_train[key] += value.data

            if calc_log:
                self.awesome_logging(loss_train, mode='train')
                self.awesome_logging(log_train, mode='train')

                eval_data = next(self.iterators['eval'])
                with torch.no_grad():
                    loss_eval, log_eval = self.eval_step(eval_data, calc_log=True)
                    for key, value in loss_eval.items():
                        if key not in losses_eval.keys():
                            losses_eval[key] = 0.
                        losses_eval[key] += value.data
                    self.awesome_logging(loss_eval, mode='eval')
                    self.awesome_logging(log_eval, mode='eval')

            self.global_step += 1

    def run(self):
        pbar_epoch = trange(self.conf.logging.nepochs, position=0)
        pbar_epoch.set_description_str('Epoch')

        for epoch in pbar_epoch:
            self.train_epoch()
            self.global_epoch += 1
            self.save()

    # endregion

    def save(self):
        models = {}
        for key in self.models.keys():
            models[key] = {}
            models[key]['state_dict'] = self.models[key].state_dict()
            if self.conf.logging.save_optimizer_state:
                models[key]['optimizer'] = self.optims[key].state_dict()

        dir_save = os.path.join(self.conf.logging.log_dir, 'checkpoint')
        os.makedirs(dir_save, exist_ok=True)
        path_save = os.path.join(dir_save, f'step_{self.global_step}.pth')

        torch.save(models, path_save, _use_new_zipfile_serialization=False)

    def awesome_logging(self, data, mode):
        raise NotImplementedError


def main():
    args = parse_args()
    conf = OmegaConf.load(args.config)

    conf.logging['log_dir'] = os.path.join(conf.logging['log_dir'], str(conf.logging['seed']))
    os.makedirs(conf.logging['log_dir'], exist_ok=True)

    if args.train:
        trainer = BaseTrainer(conf)
        trainer.run()

    if args.infer:
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/trainer.yaml',
                        help='config file path')

    parser.add_argument('--train', action='store_true',
                        help='')
    parser.add_argument('--infer', action='store_true',
                        help='')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
