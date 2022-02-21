import argparse
import os

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from trainer_base import BaseTrainer
from models.loss import THAVGGLoss


class RotatorTrainer(BaseTrainer):
    def __init__(self, conf):
        super(RotatorTrainer, self).__init__(conf)

    def build_losses(self):
        super(RotatorTrainer, self).build_losses()
        self.losses['VGG'] = THAVGGLoss().to(self.device)

    # region Training

    def forward(self, batch, calc_log=False):
        loss = {}
        logs = {}

        pose = batch['pose'].to(self.device)

        gt_shape_img = batch['img_shape'].to(self.device)
        gt_pose_img = batch['img_pose'].to(self.device)

        result = self.models['FaceRotator'](gt_shape_img, pose)
        gen_im1 = result['e2']
        gen_im2 = result['e4']

        loss['l1_im1'] = F.l1_loss(gen_im1, gt_pose_img)
        loss['l1_im2'] = F.l1_loss(gen_im2, gt_pose_img)
        loss['l1_rotator'] = loss['l1_im1'] + loss['l1_im2']

        loss['backward'] = loss['l1_rotator']

        # percep for im1
        loss['l1_percep_im1'] = torch.FloatTensor(0)
        losses_percep_rgb_im1 = self.losses['VGG'](gen_im1[:, :3], gt_pose_img[:, :3])
        losses_percep_gray_im1 = self.losses['VGG'](gen_im1[:, 3].unsqueeze(1), gt_pose_img[:, 3].unsqueeze(1))

        if self.global_epoch >= 1:
            loss['l1_percep_rgb_im1'] = losses_percep_rgb_im1['feat_0'] + losses_percep_rgb_im1['feat_1'] + \
                                        losses_percep_rgb_im1['feat_2']
            loss['l1_percep_gray_im1'] = losses_percep_gray_im1['feat_0'] + losses_percep_gray_im1['feat_1'] + \
                                         losses_percep_gray_im1['feat_2']
            loss['l1_percep_im1'] = loss['l1_percep_rgb_im1'] + loss['l1_percep_gray_im1']

        # percep for im2
        loss['l1_percep_im2'] = torch.FloatTensor(0)
        losses_percep_rgb_im2 = self.losses['VGG'](gen_im2[:, :3], gt_pose_img[:, :3])
        losses_percep_gray_im2 = self.losses['VGG'](gen_im2[:, 3].unsqueeze(1), gt_pose_img[:, 3].unsqueeze(1))

        if self.global_epoch >= 1:
            loss['l1_percep_rgb_im2'] = losses_percep_rgb_im2['feat_0'] + losses_percep_rgb_im2['feat_1'] + \
                                        losses_percep_rgb_im2['feat_2']
            loss['l1_percep_gray_im2'] = losses_percep_gray_im2['feat_0'] + losses_percep_gray_im2['feat_1'] + \
                                         losses_percep_gray_im2['feat_2']
            loss['l1_percep_im2'] = loss['l1_percep_rgb_im2'] + loss['l1_percep_gray_im2']
        loss['l1_percep'] = loss['l1_percep_im1'] + loss['l1_percep_im2']

        if self.global_epoch >= 1:
            loss['backward'] = loss['backward'] + loss['l1_percep']

        logs = {
            'img_shape': batch['img_shape'],
            'img_pose': batch['img_pose'],
            'e0': result['e0'],
            'e1': result['e1'][:, :3],
            'a1': result['a1'],
            'e2': result['e2'],
            'e3': result['e3'],
            'e4': result['e4'],
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

        self.optims['FaceRotator'].zero_grad()
        loss['backward'].backward()
        self.optims['FaceRotator'].step()

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

                    if small_batch.shape[0] != 4:
                        ones = torch.ones_like(small_batch[0].unsqueeze(0))
                        zeros = torch.zeros_like(ones)

                        if small_batch.shape[0] == 3:  # i.e. not 4
                            small_batch = torch.cat((small_batch, ones), dim=0)

                        if small_batch.shape[0] == 2:
                            small_batch = torch.cat((small_batch, zeros, ones), dim=0)

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
        trainer = RotatorTrainer(conf)
        trainer.run()

    if args.infer:
        inference()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_rotator.yaml',
                        help='config file path')

    parser.add_argument('--train', action='store_true',
                        help='')
    parser.add_argument('--infer', action='store_true',
                        help='')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
