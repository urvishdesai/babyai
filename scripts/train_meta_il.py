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
from babyai.arguments import ArgumentParser
import babyai.utils as utils
from babyai.imitation import ImitationLearning


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
