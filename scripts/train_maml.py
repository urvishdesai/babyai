# check all args passed
# create minibatch and shuffle
# 



#!/usr/bin/env python3

"""
Script to train agent through imitation learning using demonstrations.
"""

import os
import csv
import copy
import gym
import time
import datetime
import numpy as np
import sys
import logging
import torch
from babyai.arguments import ArgumentParser
import babyai.utils as utils
from babyai.imitation import ImitationLearning
from meta import MetaLearner







if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)


    argparser = ArgumentParser()
    argparser.add_argument("--demos", default=None,
                        help="demos filename (REQUIRED or demos-origin or multi-demos required)")
    argparser.add_argument("--demos-origin", required=False,
                        help="origin of the demonstrations: human | agent (REQUIRED or demos or multi-demos required)")
    argparser.add_argument("--episodes", type=int, default=0,
                        help="number of episodes of demonstrations to use"
                             "(default: 0, meaning all demos)")
    argparser.add_argument("--multi-env", nargs='*', default=None,
                      help="name of the environments used for validation/model loading")
    argparser.add_argument("--multi-demos", nargs='*', default=None,
                        help="demos filenames for envs to train on (REQUIRED when multi-env is specified)")
    argparser.add_argument("--multi-episodes", type=int, nargs='*', default=None,
                    help="number of episodes of demos to use from each file (REQUIRED when multi-env is specified)")




    args = argparser.parse_args()

    
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    device = torch.device('cuda')
    maml = MetaLearner(args).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    for meta_epoch in range(args.meta_epoch):

        for i in range(len(maml.train_demos)/args.batch_size):

            accs = maml.forward(maml.train_demos[i:i+args.batch_size])

