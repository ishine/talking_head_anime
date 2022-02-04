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

        # input image: resting input image
        # pose1: pose image with left eye, right eye and mouth
        # pose2: pose image with neck tip x-rotation, neck tip y-rotation, and neck root z-rotation

        gt_rest_img = batch['img_base'].to(self.device)
        pose = batch['pose'].to(self.device)

        gt_morphed_img = batch['img_target'].to(self.device)
        normalized_diff = (((gt_morphed_img - gt_rest_img)))

        result = self.models['FaceMorpher'](gt_rest_img, pose)
        gen_morphed_img = result['e2']
        loss['l1_morph'] = F.l1_loss(gen_morphed_img, gt_morphed_img)

        loss['backward'] = loss['l1_morph']

        # target_mask = (normalized_diff != 0.0).float()
        target_mask = (torch.abs(normalized_diff) > 6 / 256.).float()
        loss['mask'] = F.mse_loss(result['e1'], target_mask)
        loss['backward'] = loss['backward'] + 1 * loss['mask']

        # loss['change'] = F.mse_loss(result['e0'], gt_morphed_img - gen_morphed_img)
        # loss['backward'] = loss['backward'] + 0.01 * loss['change']

        logs = {
            'img_base': batch['img_base'],
            'img_target': batch['img_target'],
            'img_gen': result['e2'],
            'e0': result['e0'],
            'e1': result['e1'][:, :3],
            'a1': result['a1'],
            'gt_mask': target_mask[:, :3],
            'gt_change': (target_mask * normalized_diff)[:, :3],
            'gt_diff': (normalized_diff)[:, :3]
        }

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

        return loss, logs

    def eval_step(self, batch, calc_log=False):
        # g_step
        for key in self.models.keys():
            if 'Discriminator' in self.models.keys():
                self.models['Discriminator'].eval()
            else:
                self.models[key].train()

        loss, logs = self.forward(batch)

        return loss, logs

    # endregion

    def awesome_logging(self, data, mode):
        tensorboard = self.logger

        images = []
        imagekeys = []

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:  # scalar
                    # tensorboard.add_scalar(f'{mode}/{key}', value, self.global_step)
                    tensorboard.add_scalar(f'{mode}/{key}', value, global_step=self.global_step)
                elif value.ndim == 4:  # image
                    small_batch = value[:2]

                    # batch to height
                    small_batch = torch.cat([small_batch[i] for i in range(small_batch.shape[0])], dim=-2)

                    if small_batch.shape[0] == 3:  # i.e. not 4
                        small_batch = torch.cat((small_batch, torch.ones_like(small_batch[0].unsqueeze(0))), dim=0)

                    images.append(small_batch.detach().cpu())
                    imagekeys.append(key)
                    # tensorboard.add_image(f'{mode}/{key}', small_batch, self.global_step, dataformats='CHW')

        if len(images) > 0:
            imgs = torch.cat(images, dim=-1)
            tensorboard.add_image(f'{mode}/{"/".join(imagekeys)}', imgs, self.global_step, dataformats='CHW')


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
    parser.add_argument('--config', type=str, default='configs/train_morpher.yaml',
                        help='config file path')

    parser.add_argument('--train', action='store_true',
                        help='')
    parser.add_argument('--infer', action='store_true',
                        help='')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
