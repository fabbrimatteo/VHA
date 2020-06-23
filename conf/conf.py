# -*- coding: utf-8 -*-
# ---------------------

import os

PYTHONPATH = '..:.'
if os.environ.get('PYTHONPATH', default=None) is None:
    os.environ['PYTHONPATH'] = PYTHONPATH
else:
    os.environ['PYTHONPATH'] += (':' + PYTHONPATH)

import yaml
import socket
import random
import torch
import numpy as np
from path import Path
from typing import Optional


def set_seed(seed=None):
    # type: (Optional[int]) -> int
    """
    set the random seed using the required value (`seed`)
    or a random value if `seed` is `None`
    :return: the newly set seed
    """
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


class Conf(object):
    HOSTNAME = socket.gethostname()

    def __init__(self, conf_file_path=None, seed=None, exp_name=None, log=True):
        # type: (str, int, str, bool) -> None
        """
        :param conf_file_path: optional path of the configuration file
        :param seed: desired seed for the RNG; if `None`, it will be chosen randomly
        :param exp_name: name of the experiment
        :param log: `True` if you want to log each step; `False` otherwise
        """
        self.exp_name = exp_name
        self.log_each_step = log

        # print project name and host name
        self.project_name = Path(__file__).parent.parent.basename()
        m_str = f'┃ {self.project_name}@{Conf.HOSTNAME} ┃'
        u_str = '┏' + '━' * (len(m_str) - 2) + '┓'
        b_str = '┗' + '━' * (len(m_str) - 2) + '┛'
        print(u_str + '\n' + m_str + '\n' + b_str)

        # define output paths
        self.project_log_path = Path('log')
        self.exp_log_path = self.project_log_path / exp_name

        # set random seed
        self.seed = set_seed(seed)  # type: int

        # if the configuration file is not specified
        # try to load a configuration file based on the experiment name
        tmp = Path(__file__).parent / (self.exp_name + '.yaml')
        if conf_file_path is None and tmp.exists():
            conf_file_path = tmp

        # read the YAML configuation file
        if conf_file_path is None:
            y = {}
        else:
            conf_file = open(conf_file_path, 'r')
            y = yaml.load(conf_file, Loader=yaml.FullLoader)

        # read configuration parameters from YAML file
        # or set their default value
        self.hmap_h = y.get('H', 128)  # type: int
        self.hmap_w = y.get('W', 128)  # type: int
        self.hmap_d = y.get('D', 100)  # type: int
        self.sigma = y.get('SIGMA', 4)  # type: int
        self.lr = y.get('LR', 0.0001)  # type: float
        self.epochs = y.get('EPOCHS', 10000)  # type: int
        self.n_workers = y.get('N_WORKERS', 8)  # type: int
        self.batch_size = y.get('BATCH_SIZE', 1)  # type: int
        self.test_len = y.get('TEST_LEN', 128)  # type: int
        self.epoch_len = y.get('EPOCH_LEN', 1024)  # type: int
        self.jta_path = y.get('JTA_PATH', None)  # type: str

        if y.get('DEVICE', None) is not None and y['DEVICE'] != 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = str(y.get('DEVICE').split(':')[1])
            self.device = 'cuda:0'
        elif y.get('DEVICE', None) is not None and y['DEVICE'] == 'cpu':
            self.device = 'cpu'
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.jta_path = Path(self.jta_path)
        assert self.jta_path.exists(), 'the specified directory for the JTA-Dataset does not exists'
