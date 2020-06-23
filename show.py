# -*- coding: utf-8 -*-
# ---------------------

import json

import cv2
import click
from torch.utils.data import DataLoader

import utils
from conf import Conf
from dataset.jta_hmap_ds import JTAHMapDS
from models.vha import Autoencoder
from test_metrics import joint_det_metrics


def results(cnf):
    # type: (Conf) -> None
    """
    Shows a visual representation of the obtained results
    using the test set images as input
    """

    # init Autoencoder
    autoencoder = Autoencoder(cnf.hmap_d)
    autoencoder.load_w(f'log/{cnf.exp_name}/best.pth')
    autoencoder.to(cnf.device)
    autoencoder.eval()
    autoencoder.requires_grad(False)

    # init test loader
    test_set = JTAHMapDS(mode='test', cnf=cnf)
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=0, shuffle=True)

    for step, sample in enumerate(test_loader):
        hmap_true, y_true, frame_path = sample
        frame_path = frame_path[0]
        hmap_true = hmap_true.to(cnf.device)
        y_true = json.loads(y_true[0])

        # hmap_true --> [autoencoder] --> hmap_pred
        hmap_pred = autoencoder.forward(hmap_true).squeeze()

        y_pred = utils.get_multi_local_maxima_3d(hmaps3d=hmap_pred, threshold=0.1, device=cnf.device)

        metrics = joint_det_metrics(points_pred=y_pred, points_true=y_true, th=1)
        f1 = metrics['f1']

        # show output
        print(f'\n\t▶▶ Showing results of \'{frame_path}\'')
        print(f'\t▶▶ F1@1px score:', f1)
        print(f'\t▶▶ Press some key to advance in the depth dimension')

        img = cv2.imread(frame_path)
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('img', img)

        utils.visualize_3d_hmap(hmap=hmap_pred[0, ...])


@click.command()
@click.argument('exp_name', type=str, default='default')
def main(exp_name):
    # type: (str) -> None

    cnf = Conf(exp_name=exp_name)

    print(f'▶ Results of experiment \'{exp_name}\'')
    results(cnf=cnf)


if __name__ == '__main__':
    main()
