#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

import gym
import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.core import Preprocessor
import preprocessors


def create_model(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understand your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    print("Creating model...")

    with tf.name_scope('deepq'): 
        model = Sequential()
        model.add(Convolution2D(16, 8, 8, subsample=(8,8),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same',input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 4, 4, subsample=(4,4),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(256, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
        model.add(Activation('relu'))
        model.add(Dense(num_actions,init=lambda shape, name: normal(shape, scale=0.01, name=name)))
   
    adam = Adam(lr=1e-6)
    model.compile(loss='mse',optimizer=adam)
    print("Model created...")
    return model


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari environment')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--iters', default=5000000, type=int, help='Number of interactions with environment')
    parser.add_argument('--max_episode_len', default=200, type=int, help='Maximum length of episode')
    parser.add_argument('--frame_count', default=4, type=int, help='Number of frames to feed to Q-network')
    parser.add_argument('--eps', default=0.05, type=float, help='Epsilon value for epsilon-greedy exploration')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate for training')
    parser.add_argument('--discount', default=0.99, type=float, help='Discounting factor')
    parser.add_argument('--replay_mem_size', default=1000000, type=int, help='Maximum size of replay memory')
    args = parser.parse_args()
    #args.input_shape = tuple(args.input_shape)

    args.output = get_output_folder(args.output, args.env)

    env = gym.make(args.env)
    env.reset()

    # Create model
    model = create_model(args.frame_count, env.observation_space.shape, env.action_space.n, args.env+"-test")
    
    # Create session
    sess = tf.Session()
    K.set_session(sess)

    replay_mem = ReplayMemory(args.replay_mem_size)

    for _ in range(args.iters):
        init_obs = env.reset()
        for t in range(args.max_episode_len):
            for _ in range(args.frame_count):
                obs, reward, is_terminal, debugging = env.step()
            if is_terminal: break
            PreProcessorSequence
            model.fit(obs)


    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

if __name__ == '__main__':
    main()
