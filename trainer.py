# -*- coding: utf-8 -*-
# ---------------------

import importlib
import json
import math
from datetime import datetime
from time import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

import utils
from conf import Conf
from dataset.jta_hmap_ds import JTAHMapDS
from test_metrics import joint_det_metrics
from models.vha import Autoencoder


class Trainer(object):

    def __init__(self, cnf):
        # type: (Conf) -> Trainer

        self.cnf = cnf

        # init model
        self.model = Autoencoder(hmap_d=cnf.hmap_d).to(cnf.device)

        # init optimizer
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=cnf.lr)

        # init train loader
        training_set = JTAHMapDS(mode='train', cnf=cnf)
        self.train_loader = DataLoader(
            dataset=training_set, batch_size=cnf.batch_size, num_workers=cnf.n_workers, shuffle=True
        )

        # init val loader
        val_set = JTAHMapDS(mode='val', cnf=cnf)
        self.val_loader = DataLoader(
            dataset=val_set, batch_size=1, num_workers=cnf.n_workers, shuffle=False
        )

        # init logging stuff
        self.log_path = cnf.exp_log_path
        tb_logdir = cnf.project_log_path.abspath()
        print(f'tensorboard --logdir={tb_logdir}\n')
        self.sw = SummaryWriter(self.log_path)
        self.log_freq = len(self.train_loader)
        self.train_losses = []
        self.val_losses = []
        self.val_f1s = []

        # starting values values
        self.epoch = 0
        self.best_val_f1 = None

        # possibly load checkpoint
        self.load_ck()


    def load_ck(self):
        """
        load training checkpoint
        """
        ck_path = self.log_path / 'training.ck'
        if ck_path.exists():
            ck = torch.load(ck_path, map_location=torch.device('cpu'))
            print('[loading checkpoint \'{}\']'.format(ck_path))
            self.epoch = ck['epoch']
            self.model.load_state_dict(ck['model'], strict=True)
            self.model.to(self.cnf.device)
            self.best_val_f1 = ck['best_val_f1']
            if ck.get('optimizer', None) is not None:
                self.optimizer.load_state_dict(ck['optimizer'])


    def save_ck(self):
        """
        save training checkpoint
        """
        ck = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1
        }
        torch.save(ck, self.log_path / 'training.ck')


    def train(self):
        """
        train model for one epoch on the Training-Set.
        """
        self.model.train()
        self.model.requires_grad(True)

        start_time = time()
        times = []
        t = time()
        for step, sample in enumerate(self.train_loader):

            self.optimizer.zero_grad()
            x = sample[0].to(self.cnf.device)

            y_pred = self.model.forward(x)

            loss = nn.MSELoss()(y_pred, x)
            loss.backward()
            self.train_losses.append(loss.item())

            self.optimizer.step(None)

            # print an incredible progress bar
            progress = (step + 1) / self.cnf.epoch_len
            progress_bar = ('█' * int(50 * progress)) + ('┈' * (50 - int(50 * progress)))
            times.append(time() - t)
            t = time()
            if self.cnf.log_each_step or (not self.cnf.log_each_step and progress == 1):
                print('\r[{}] Epoch {:0{e}d}.{:0{s}d}: │{}│ {:6.2f}% │ Loss: {:.6f} │ ↯: {:5.2f} step/s'.format(
                    datetime.now().strftime("%m-%d@%H:%M"), self.epoch, step + 1,
                    progress_bar, 100 * progress,
                    np.mean(self.train_losses), 1 / np.mean(times),
                    e=math.ceil(math.log10(self.cnf.epochs)),
                    s=math.ceil(math.log10(self.log_freq)),
                ), end='')

            if step >= self.cnf.epoch_len - 1:
                break

        # log average loss of this epoch
        mean_epoch_loss = np.mean(self.train_losses)
        self.sw.add_scalar(tag='train_loss', scalar_value=mean_epoch_loss, global_step=self.epoch)
        self.train_losses = []

        # log epoch duration
        print(f' │ T: {time() - start_time:.2f} s')


    def test(self):
        """
        test model on the Test-Set
        """

        self.model.eval()
        self.model.requires_grad(False)

        t = time()
        for step, sample in enumerate(self.val_loader):
            hmap_true, y_true, _ = sample
            hmap_true = hmap_true.to(self.cnf.device)
            y_true = json.loads(y_true[0])

            hmap_pred = self.model.forward(hmap_true)

            loss = nn.MSELoss()(hmap_pred, hmap_true)
            self.val_losses.append(loss.item())

            y_pred = utils.get_multi_local_maxima_3d(hmaps3d=hmap_pred.squeeze(), threshold=0.1, device=self.cnf.device)

            metrics = joint_det_metrics(points_pred=y_pred, points_true=y_true, th=1)
            f1 = metrics['f1']
            self.val_f1s.append(f1)

            if step < 3:
                hmap_pred = hmap_pred.squeeze()
                out_path = self.cnf.exp_log_path / f'{step}_pred.mp4'
                utils.save_3d_hmap(hmap=hmap_pred[0, ...], path=out_path)

                hmap_true = hmap_true.squeeze()
                out_path = self.cnf.exp_log_path / f'{step}_true.mp4'
                utils.save_3d_hmap(hmap=hmap_true[0, ...], path=out_path)

            if step >= self.cnf.test_len:
                break

        # log average loss on test set
        mean_val_loss = np.mean(self.val_losses)
        self.val_losses = []
        print(f'\t● AVG Loss on VAL-set: {mean_val_loss:.6f} │ T: {time() - t:.2f} s')
        self.sw.add_scalar(tag='val_loss', scalar_value=mean_val_loss, global_step=self.epoch)

        # log average f1 on test set
        mean_val_f1 = np.mean(self.val_f1s)
        self.val_f1s = []
        print(f'\t● AVG F1@1px on VAL-set: {mean_val_f1:.6f} │ T: {time() - t:.2f} s')
        self.sw.add_scalar(tag='val_F1', scalar_value=mean_val_f1, global_step=self.epoch)

        # save best model
        if self.best_val_f1 is None or mean_val_f1 < self.best_val_f1:
            self.best_val_f1 = mean_val_f1
            torch.save(self.model.state_dict(), self.log_path / 'best.pth')


    def run(self):
        """
        start model training procedure (train > test > checkpoint > repeat)
        """
        for e in range(self.epoch, self.cnf.epochs):
            self.train()
            if e % 10 == 0 and e != 0:
                self.test()
            self.epoch += 1
            self.save_ck()
