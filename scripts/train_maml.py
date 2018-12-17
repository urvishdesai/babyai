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
import babyai.utils as utils
from babyai.imitation import ImitationLearning
from babyai.meta import MetaLearner
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--demos", default=None,
                    help="demos filename (REQUIRED or demos-origin or multi-demos required)")
parser.add_argument("--demos-origin", required=False,
                    help="origin of the demonstrations: human | agent (REQUIRED or demos or multi-demos required)")
parser.add_argument("--episodes", type=int, default=0,
                    help="number of episodes of demonstrations to use"
                         "(default: 0, meaning all demos)")
parser.add_argument("--multi-env", nargs='*', default=None,
                  help="name of the environments used for validation/model loading")
parser.add_argument("--multi-demos", nargs='*', default=None,
                    help="demos filenames for envs to train on (REQUIRED when multi-env is specified)")
parser.add_argument("--multi-episodes", type=int, nargs='*', default=None,
                    help="number of episodes of demos to use from each file (REQUIRED when multi-env is specified)")

parser.add_argument("--task-num", type=int, nargs='*', default=100,
                    help="Number of grammars to train")


parser.add_argument("--meta-lr", type=float, nargs='*', default=.001,
                    help="")


parser.add_argument("--update-lr", type=float, nargs='*', default=.4,
                    help="")


parser.add_argument("--env", default=None, help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None, help="name of the model (default: ENV_ALGO_TIME)")
parser.add_argument("--pretrained-model", default=None, help='If you\'re using a pre-trained model and want the fine-tuned one to have a new name')
parser.add_argument("--seed", type=int, default=1, help="random seed; if 0, a random random seed will be used  (default: 1)")
parser.add_argument("--task-id-seed", action='store_true', help="use the task id within a Slurm job array as the seed")
parser.add_argument("--procs", type=int, default=64, help="number of processes (default: 64)")
parser.add_argument("--tb", action="store_true", default=False, help="log into Tensorboard")
parser.add_argument("--meta-epoch", type=int, default=10, help="maximum number of epochs")

# Training arguments
parser.add_argument("--frames", type=int, default=int(9e10), help="number of frames of training (default: 9e10)")
parser.add_argument("--epochs", type=int, default=1000000, help="maximum number of epochs")
parser.add_argument("--recurrence", type=int, default=20, help="number of timesteps gradient is backpropagated (default: 20)")
parser.add_argument("--batch-size", type=int, default=32,
help="batch size for PPO (default: 1280)")
parser.add_argument("--entropy-coef", type=float, default=0.01, help="entropy term coefficient (default: 0.01)")

# Model parameters
parser.add_argument("--image-dim", type=int, default=128,help="dimensionality of the image embedding")
parser.add_argument("--memory-dim", type=int, default=128, help="dimensionality of the memory LSTM")
parser.add_argument("--instr-dim", type=int, default=128, help="dimensionality of the memory LSTM")
parser.add_argument("--no-instr", action="store_true", default=False, help="don't use instructions in the model")
parser.add_argument("--instr-arch", default="gru", help="arch to encode instructions, possible values: gru, bigru, conv, bow (default: gru)")
parser.add_argument("--no-mem", action="store_true", default=False, help="don't use memory in the model")
parser.add_argument("--arch", default='expert_filmcnn', help="image embedding architecture")

# Validation parameters
parser.add_argument("--val-seed", type=int, default=0, help="seed for environment used for validation (default: 0)")
parser.add_argument("--val-interval", type=int, default=1, help="number of epochs between two validation checks (default: 1)")
parser.add_argument("--val-episodes", type=int, default=500, help="number of episodes used to evaluate the agent, and to evaluate v")







if __name__ == '__main__':



    args = parser.parse_args()

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    args.model = args.model or ImitationLearning.default_model_name(args)
    utils.configure_logging(args.model)
    logger = logging.getLogger(__name__)


    device = torch.device('cuda')
    maml = MetaLearner(args).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    for meta_epoch in range(args.meta_epoch):

        for i in range(int(len(maml.train_demos)/args.batch_size)):

            logs = maml.forward(maml.train_demos[args.batch_size*i:args.batch_size*i+args.batch_size])
            H = sum([log['entropy'] for log in logs])/float(len(logs))

            PL = sum([log['policy_loss'] for log in logs])/float(len(logs))
            A = sum([log['accuracy'] for log in logs])/float(len(logs))

            print (meta_epoch, i, H, PL, A)
            if i%10 ==0:

                logs = maml.validate(maml.val_demos)
                H = sum([log['entropy'] for log in logs])/float(len(logs))

                PL = sum([log['policy_loss'] for log in logs])/float(len(logs))
                A = sum([log['accuracy'] for log in logs])/float(len(logs))

                print ('val: ',meta_epoch , i, H, PL, A)
                with open('train5.txt','a') as f:
                    f.write(str(H)+' '+str(PL)+' '+str(A)+'\n')
