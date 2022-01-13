import argparse
import glob
import os

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from utils.util import save_files, build_models_from_config, build_datasets_from_config
from utils.logging import get_logger
from trainer_base import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, conf):
        super(Trainer, self).__init__(conf)

    # region Training

    def forward(self, batch, calc_log=False):
        loss = {}
        logs = {}
        logs.update(batch)

        # input image: resting input image
        # pose1: pose image with left eye, right eye and mouth
        # pose2: pose image with neck tip x-rotation, neck tip y-rotation, and neck root z-rotation

        gt_rest_img = batch['img_base'].to(self.device)
        pose = batch['pose'].to(self.device)

        gt_morphed_img = batch['img_target'].to(self.device)

        gen_morphed_img = self.models['FaceMorpher'](gt_rest_img, pose)
        loss['l1_morph'] = F.l1_loss(gen_morphed_img, gt_morphed_img)

        logs['gen_morphed_img'] = gen_morphed_img
        loss['backward'] = loss['l1_morph']

        return loss, logs

    def train_step(self, batch, calc_log=False):
        # g_step
        for key in self.models.keys():
            if 'Discriminator' in self.models.keys():
                self.models['Discriminator'].eval()
            else:
                self.models[key].train()

        loss, logs = self.forward(batch)

        self.optims['FaceMorpher'].zero_grad()
        loss['backward'].backward()
        self.optims['FaceMorpher'].step()

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logs[key] = value.cpu().detach()
                batch[key] = value.to(self.device)
            else:
                logs[key] = value

        return loss, logs

    def eval_step(self, batch, calc_log=False):
        # g_step
        for key in self.models.keys():
            if 'Discriminator' in self.models.keys():
                self.models['Discriminator'].eval()
            else:
                self.models[key].train()

        loss, logs = self.forward(batch)

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logs[key] = value.cpu().detach()
                batch[key] = value.to(self.device)
            else:
                logs[key] = value

        return loss, logs

    # endregion


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
