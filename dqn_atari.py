#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Conv2D, Dense, Flatten, Input,
                          Permute, InputLayer, Lambda)
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K

import gym
import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.core import Preprocessor, ReplayMemory
from deeprl_hw2.preprocessors import PreprocessorSequence, AtariPreprocessor


def create_model(window, input_shape, num_actions,
                 model_name='q_network', model_type='deep'):  # noqa: D103
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
    model_name: str1
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    print("Creating model...")

    #img = tf.placeholder(tf.float32, shape=input_shape + (window,))
    model = Sequential()
    #model.add(InputLayer(input_tensor=custom_input_tensor, input_shape=input_shape + (window,)))
    
    if model_type=='deep':

        model.add(Conv2D(16, (8, 8), input_shape= (84, 84, 4)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (4, 4)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(num_actions))
        # model.add(Lambda(K.one_hot,
               # arguments={'nb_classes': num_actions},
               # output_shape=(num_actions)))

    elif model_type=='linear':

        model.add(Flatten(input_shape= (84, 84, 4)))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(num_actions))

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
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
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
    parser.add_argument('--mb_size', default=32, type=int, help='Minibatch size')
    parser.add_argument('--max_episode_len', default=200, type=int, help='Maximum length of episode')
    parser.add_argument('--frame_count', default=4, type=int, help='Number of frames to feed to Q-network')
    parser.add_argument('--eps', default=0.05, type=float, help='Epsilon value for epsilon-greedy exploration')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate for training')
    parser.add_argument('--discount', default=0.99, type=float, help='Discounting factor')
    parser.add_argument('--replay_mem_size', default=1000000, type=int, help='Maximum size of replay memory')
    parser.add_argument('--train_freq', default=3, type=int, help='Frequency of updating Q-network')
    parser.add_argument('--target_update_freq', default=10000, type=int, help='Frequency of updating target network')
    
    args = parser.parse_args()
    #args.input_shape = tuple(args.input_shape)

    # Get output folder
    args.output = get_output_folder(args.output, args.env)

    # Create environment
    env = gym.make(args.env)
    env.reset()

    # Create model
    preprocessed_input_shape = (84,84)
    # Model type should probably be given by a config argument,to support linear/deep networks
    model_type = 'linear'
    model = create_model(args.frame_count, preprocessed_input_shape, env.action_space.n, args.env+"-test", model_type)

    # Create session
    sess = tf.Session()
    K.set_session(sess)

    # Initialize replay memory
    replay_mem = ReplayMemory(args.replay_mem_size, args.frame_count)

    # Create agent
    preprocessor_seq = PreprocessorSequence([AtariPreprocessor(preprocessed_input_shape)])

    dqn = DQNAgent (model, preprocessor_seq, replay_mem, 
                   args.discount, args.target_update_freq, args.mb_size*3,
                   args.train_freq, args.mb_size, args.eps, args.output)

    dqn.compile()
    dqn.fit(env, args.iters, args.max_episode_len)

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

if __name__ == '__main__':
    main()
