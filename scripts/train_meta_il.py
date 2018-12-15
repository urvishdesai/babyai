#!/usr/bin/env python3

"""
Script to train agent through meta learning for imitation learning using demonstrations.
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
import argparse
import babyai.utils as utils
from babyai.imitation import ImitationLearning



# Parse arguments
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

parser.add_argument("--task-num", type=int, nargs='*', default=1,
                    help="Number of grammars to train")


parser.add_argument("--meta-lr", type=float, nargs='*', default=.1,
                    help="")


parser.add_argument("--update-lr", type=float, nargs='*', default=.4,
                    help="")


parser.add_argument("--episodes", type=float, nargs='*', default=.4,
                    help="")



parser.add_argument("--env", default=None, help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None, help="name of the model (default: ENV_ALGO_TIME)")
parser.add_argument("--pretrained-model", default=None, help='If you\'re using a pre-trained model and want the fine-tuned one to have a new name')
parser.add_argument("--seed", type=int, default=1, help="random seed; if 0, a random random seed will be used  (default: 1)")
parser.add_argument("--task-id-seed", action='store_true', help="use the task id within a Slurm job array as the seed")
parser.add_argument("--procs", type=int, default=64, help="number of processes (default: 64)")
parser.add_argument("--tb", action="store_true", default=False, help="log into Tensorboard")

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

# # Validation parameters
# self.add_argument("--val-seed", type=int, default=0,
# help="seed for environment used for validation (default: 0)")
# self.add_argument("--val-interval", type=int, default=1,
# help="number of epochs between two validation checks (default: 1)")
# self.add_argument("--val-episodes", type=int, default=500,
# help="number of episodes used to evaluate the agent, and to evaluate v

def train(forward_model, backward_model, optimizer, meta_optimizer, train_data, meta_epochs):
  """ Train a meta-learner
  Inputs:
    forward_model, backward_model: Two identical PyTorch modules (can have shared Tensors)
    optimizer: a neural net to be used as optimizer (an instance of the MetaLearner class)
    meta_optimizer: an optimizer for the optimizer neural net, e.g. ADAM
    train_data: an iterator over an epoch of training data
    meta_epochs: meta-training steps
  To be added: intialization, early stopping, checkpointing, more control over everything
  """
  for meta_epoch in range(meta_epochs): # Meta-training loop (train the optimizer)
    optimizer.zero_grad()
    losses = []
    for inputs, labels in train_data:   # Meta-forward pass (train the model)
      forward_model.zero_grad()         # Forward pass
      inputs = Variable(inputs)
      labels = Variable(labels)
      output = forward_model(inputs)
      loss = loss_func(output, labels)  # Compute loss
      losses.append(loss)
      loss.backward()                   # Backward pass to add gradients to the forward_model
      optimizer(forward_model,          # Optimizer step (update the models)
                backward_model)
    meta_loss = sum(losses)             # Compute a simple meta-loss
    meta_loss.backward()                # Meta-backward pass
    meta_optimizer.step()               # Meta-optimizer step


def main(args):
    # Verify the arguments when we train on multiple environments
    # No need to check for the length of len(args.multi_env) in case, for some reason, we need to validate on other envs
    if args.multi_env is not None:
        assert len(args.multi_demos) == len(args.multi_episodes)

    args.model = args.model or ImitationLearning.default_model_name(args)
    utils.configure_logging(args.model)
    logger = logging.getLogger(__name__)

    il_learn = ImitationLearning(args)

    # Define logger and Tensorboard writer
    header = (["update", "frames", "FPS", "duration", "entropy", "policy_loss", "train_accuracy"]
              + ["validation_accuracy"])
    if args.multi_env is None:
        header.extend(["validation_return", "validation_success_rate"])
    else:
        header.extend(["validation_return_{}".format(env) for env in args.multi_env])
        header.extend(["validation_success_rate_{}".format(env) for env in args.multi_env])
    writer = None
    if args.tb:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(utils.get_log_dir(args.model))

    # Define csv writer
    csv_writer = None
    csv_path = os.path.join(utils.get_log_dir(args.model), 'log.csv')
    first_created = not os.path.exists(csv_path)
    # we don't buffer data going in the csv log, cause we assume
    # that one update will take much longer that one write to the log
    csv_writer = csv.writer(open(csv_path, 'a', 1))
    if first_created:
        csv_writer.writerow(header)

    # Get the status path
    status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')

    # Log command, availability of CUDA, and model
    logger.info(args)
    logger.info("CUDA available: {}".format(torch.cuda.is_available()))
    logger.info(il_learn.acmodel)

    il_learn.train(il_learn.train_demos, writer, csv_writer, status_path, header)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
