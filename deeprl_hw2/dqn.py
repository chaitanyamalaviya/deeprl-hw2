"""Main DQN agent."""
from __future__ import print_function, division
from keras.optimizers import Adam
from keras.models import load_model
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
from deeprl_hw2.core import Preprocessor, ReplayMemory, Sample
from collections import namedtuple
from gym import wrappers
from . import utils, objectives, policy
import numpy as np
import os
import random

class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgent. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and function parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your

    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                 q_network,
                 preprocessor,
                 memory,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 epsilon,
                 output_folder,
                 eval_every,
                 model_type):

        self.q_network = q_network
        self.preprocessor = preprocessor
        self.memory = memory
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.output_folder = output_folder
        self.eval_every = eval_every
        self.model_type = model_type

        self.sampling = True
        self.is_training = True
        self.target_q_network = Sequential()
        self.policy = None
        self.evaluating = False

    def compile(self):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.

        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """

        if self.model_type == "dueling":
            self.target_q_network = \
                utils.get_hard_target_model_updates(self.target_q_network, self.q_network, True)
        else:
          self.target_q_network = \
                utils.get_hard_target_model_updates(self.target_q_network, self.q_network)

        # Uncomment to use Adam Optimizer
        adam = Adam(lr=1e-6)

        self.q_network.compile(loss=objectives.mean_huber_loss,
                               optimizer=adam,
                               metrics=[])

        if self.model_type == "linear_double" or self.model_type== "dueling":
            self.target_q_network.compile(loss=objectives.mean_huber_loss,
                                          optimizer=adam,
                                          metrics=[])

        # Uncomment to use MSE loss
        #self.q_network.compile(loss='mean_squared_error',
        #                       optimizer=adam,
        #                       metrics=[])

        print("Model compiled.")

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        # state is actually a batch of states
        actions = self.q_network.predict_on_batch(state)
        return actions

    def select_action(self, q_values=None, steps=None, decay=False):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """

        if self.is_training and self.sampling: # UniformRandomPolicy
          chosen_action = self.policy.select_action()
        elif self.is_training and decay: # LinearDecayGreedyEpsilonPolicy
          chosen_action = self.policy.select_action(q_values, steps, self.is_training)
        elif self.is_training:
          chosen_action = self.policy.select_action(q_values)
        else: # Greedy Policy
          chosen_action = policy.select_action()

        return chosen_action


    def update_network(self, states_batch, targets_batch):
        """Update your policy.

        Behavior may differ based on what stage you're
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """

        loss = self.q_network.train_on_batch(states_batch, targets_batch)
        return loss

    def update_target_network(self, states_batch, targets_batch):
        loss = self.target_q_network.train_on_batch(states_batch, targets_batch)
        return loss

    def in_eval(self, episode_id):
        return self.evaluating

    def fit_naive(self, env, num_iterations, max_episode_length):

        os.mkdir(self.output_folder)

        ## Logging stats and Recording video
        loss_file = open(self.output_folder + "/loss_file_" + self.model_type, "w")

        episode_lengths = []
        episode_counter = 0
        sum_tot_iters = 0
        self.sampling = False

        # Flag to indicate that eval should be done at the end of the episode
        should_eval = False
        eval_steps = self.eval_every
        next_eval = eval_steps

        state = env.reset()
        preprocessed_state = self.preprocessor.preprocess_state(state, mem=True)
        state = np.stack([preprocessed_state] * 4, axis=2)

        # Get a held out set of states whose Q-value would be observed during eval
        qvalue_held_out_states = [state]
        self.policy = policy.UniformRandomPolicy(env.action_space.n)
        self.sampling = True
        while True:
            action = self.select_action()
            next_state, reward, is_terminal, _ = env.step(action)
            next_state = self.preprocessor.preprocess_state(next_state, mem=True)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
            qvalue_held_out_states.append(next_state)
            if is_terminal:
                state = env.reset()
                preprocessed_state = self.preprocessor.preprocess_state(state, mem=True)
                state = np.stack([preprocessed_state] * 4, axis=2)
                break
            else:
                state = next_state

        # Randomly sample 20 of the states from the episode
        random.shuffle(qvalue_held_out_states)
        qvalue_held_out_states = np.array(qvalue_held_out_states[:20])
        self.sampling = False

        update_tick = 0
        total_steps = 0

        while sum_tot_iters < num_iterations:
          #self.policy = policy.GreedyEpsilonPolicy(self.epsilon)
          self.policy = policy.LinearDecayGreedyEpsilonPolicy(1, 0.1, 1000000)
          state = env.reset()
          preprocessed_state = self.preprocessor.preprocess_state(state, mem=True)
          state = np.stack([preprocessed_state] * 4, axis=2)
          state = np.expand_dims(state, axis=0)
          cum_reward = 0
          cum_loss = 0
          tot_updates = 0

          for t in range(max_episode_length):
              total_steps += 1

              ## Evaluate Model every ~10,000 updates, nearest episode end
              if (sum_tot_iters+tot_updates+1) % next_eval == 0:
                  should_eval = True

              choice = random.randint(0,1)
              ## Get next state
              if self.model_type == "linear_double":
                # if choice == 0:
                #     q_values = self.q_network.predict(state)
                # else:
                #     q_values = self.target_q_network.predict(state)
                q_values = self.q_network.predict(state) + self.target_q_network.predict(state)

              # Naive model
              else:
                q_values = self.q_network.predict(state)

              # chosen_action = self.select_action(q_values)
              chosen_action = \
                self.select_action(q_values, total_steps, True)

              next_state, reward, is_terminal, info = env.step(chosen_action)
              next_state = self.preprocessor.preprocess_state(next_state, mem=True)
              state = np.squeeze(state, axis=0)
              next_state = \
                np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
              reward = self.preprocessor.preprocess_reward(reward)

              # Only do the rest of the stuff once every self.train_freq steps
              # tot_updates is not updated, so all other update logic is intact
              update_tick = (update_tick + 1) % self.train_freq
              if not update_tick == 0 and not self.model_type == "linear_double":
                  state = np.expand_dims(next_state, axis=0)
                  cum_reward += reward
                  if is_terminal:
                      break
                  continue

              ## Calculate q-learning targets
              if self.model_type == "linear_double":
                  if choice == 0:
                      q_values = self.q_network.predict(np.expand_dims(next_state, axis=0))
                      max_action = np.argmax(q_values[0])
                      target_q_values = self.target_q_network.predict(np.expand_dims(next_state, axis=0))
                      target = \
                        reward + np.invert(is_terminal).astype(np.float32) * \
                        self.gamma * target_q_values[0][max_action]

                  else:
                      q_values = self.target_q_network.predict(np.expand_dims(next_state, axis=0))
                      max_action = np.argmax(q_values[0])
                      target_q_values = self.q_network.predict(np.expand_dims(next_state, axis=0))
                      target = \
                        reward + np.invert(is_terminal).astype(np.float32) * \
                        self.gamma * target_q_values[0][max_action]

              # Naive case
              else:
                  q_values = self.q_network.predict(np.expand_dims(next_state, axis=0))
                  max_action = np.argmax(q_values[0])
                  target = \
                    reward + np.invert(is_terminal).astype(np.float32) * \
                    self.gamma * q_values[0][max_action]

              target_one_hot = np.zeros(env.action_space.n)
              target_one_hot[chosen_action] = target

              ## Update parameters
              if self.model_type == "linear_double":
                  if choice == 0:
                    loss = self.update_network(np.expand_dims(state, axis=0), np.expand_dims(target_one_hot, axis=0))
                  else:
                    loss = self.update_target_network(np.expand_dims(state, axis=0), np.expand_dims(target_one_hot, axis=0))

              # Naive case
              else:
                  loss = self.update_network(np.expand_dims(state, axis=0), np.expand_dims(target_one_hot, axis=0))

              cum_loss += loss
              tot_reward = reward
              cum_reward += tot_reward

              state = next_state
              state = np.expand_dims(state, axis=0)
              tot_updates += 1
              print("\r[%i] Loss: %.8f" %
                      (sum_tot_iters+tot_updates, cum_loss/tot_updates),
                    "Reward:", cum_reward, end="")

              if is_terminal:
                break

          loss_file.write("%i %.8f\n" %
                         (sum_tot_iters+tot_updates, cum_loss/tot_updates))

          episode_counter += 1
          episode_lengths.append(tot_updates*4) # approximately this long
          print("\nAverage reward per frame this episode:",
                 cum_reward/(tot_updates*4*self.batch_size),
                "\nEpisode Length:", tot_updates*4,
                "\nNumber of Episodes:", episode_counter, "\n")

          sum_tot_iters += tot_updates

          # If we have to evaluate, do so and set the next evaluation iteration
          if should_eval:
              ## Save Model before evaluating
              self.q_network.save(self.output_folder+'/model_file.h5')
              env.reset()
              self.evaluate(env, num_iterations, max_episode_length,
                            self.output_folder+'/model_file.h5',
                            sum_tot_iters, qvalue_held_out_states)
              next_eval = sum_tot_iters + eval_steps
              should_eval = False
              loss_file.flush()

        loss_file.close()
        print("\nAverage Episode Length: ", sum(episode_lengths)/len(episode_lengths))



    def fit(self, env, num_iterations, max_episode_length):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """

        os.mkdir(self.output_folder)

        print("Filling up replay memory..")
        self.policy = policy.UniformRandomPolicy(env.action_space.n)
        state = env.reset()
        preprocessed_state = self.preprocessor.preprocess_state(state, mem=True)
        state = np.stack([preprocessed_state] * 4, axis=2)

        # Get a held out set of states whose Q-value would be observed during eval
        qvalue_held_out_states = [state]
        while True:
            action = self.select_action()
            next_state, reward, is_terminal, _ = env.step(action)
            next_state = self.preprocessor.preprocess_state(next_state, mem=True)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
            qvalue_held_out_states.append(next_state)
            if is_terminal:
                state = env.reset()
                preprocessed_state = self.preprocessor.preprocess_state(state, mem=True)
                state = np.stack([preprocessed_state] * 4, axis=2)
                break
            else:
                state = next_state

        # Randomly sample 20 of the states from the episode
        random.shuffle(qvalue_held_out_states)
        qvalue_held_out_states = np.array(qvalue_held_out_states[:20])

        # Initialize replay memory
        for i in range(self.num_burn_in):
            action = self.select_action()
            next_state, reward, is_terminal, _ = env.step(action)
            next_state = self.preprocessor.preprocess_state(next_state, mem=True)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
            reward = self.preprocessor.process_reward(reward)
            self.memory.append(Sample(state, action, reward, next_state, is_terminal))
            if is_terminal:
                state = env.reset()
                preprocessed_state = self.preprocessor.preprocess_state(state, mem=True)
                state = np.stack([preprocessed_state] * 4, axis=2)
            else:
                state = next_state

        loss_file = open(self.output_folder + "/loss_file_" + self.model_type, "w")

        episode_lengths = []
        episode_counter = 0
        sum_tot_iters = 0
        self.sampling = False

        # Flag to indicate that eval should be done at the end of the episode
        should_eval = False
        eval_steps = self.eval_every
        next_eval = eval_steps

        update_tick = 0
        total_steps = 0

        while sum_tot_iters < num_iterations:
          #self.policy = policy.GreedyEpsilonPolicy(self.epsilon)
          self.policy = policy.LinearDecayGreedyEpsilonPolicy(1, 0.1, 1000000)
          state = env.reset()
          preprocessed_state = self.preprocessor.preprocess_state(state, mem=True)
          state = np.stack([preprocessed_state] * 4, axis=2)
          state = np.expand_dims(state, axis=0)
          cum_reward = 0
          cum_loss = 0
          tot_updates = 0

          for t in range(max_episode_length):
              total_steps += 1

              ## Update target q-network at a frequency
              if (sum_tot_iters+tot_updates+1) % self.target_update_freq == 0 and self.model_type != 'dueling':
                self.target_q_network = \
                    utils.get_hard_target_model_updates(self.target_q_network,
                                                        self.q_network)


              ## Update target q-network at a frequency
              if (sum_tot_iters+tot_updates+1) % self.target_update_freq == 0 and self.model_type == 'dueling':
                self.target_q_network = \
                    utils.get_hard_target_model_updates(self.target_q_network,
                                                        self.q_network, True)

              ## Evaluate Model every ~10,000 updates, nearest episode end
              if (sum_tot_iters+tot_updates+1) % next_eval == 0:
                  should_eval = True

              ## Get next state
              q_values = self.q_network.predict(state)

              # chosen_action = self.select_action(q_values)
              chosen_action = \
                self.select_action(q_values, total_steps, True)

              next_state, reward, is_terminal, info = env.step(chosen_action)
              next_state = self.preprocessor.preprocess_state(next_state, mem=True)
              state = np.squeeze(state, axis=0)
              next_state = \
                np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
              reward = self.preprocessor.preprocess_reward(reward)
              ## Append current sample to replay memory
              self.memory.append(
                Sample(state, chosen_action, reward, next_state, is_terminal))

              # Only do the rest of the stuff once every self.train_freq steps
              # tot_updates is not updated, so all other update logic is intact
              update_tick = (update_tick + 1) % self.train_freq
              if not update_tick == 0:
                  state = np.expand_dims(next_state, axis=0)
                  cum_reward += reward
                  if is_terminal:
                      break
                  continue

              ## Sample minibatch from replay memory
              samples = self.memory.sample(self.batch_size)
              sample_props  = [s.get_props() for s in samples]

              states_batch, actions_batch, \
              rewards_batch, next_states_batch, is_terminal_batch = \
                map(np.array, zip(*sample_props))

              states_batch = \
                self.preprocessor.preprocess_state(states_batch, mem=False, batch=True)
              next_states_batch = \
                self.preprocessor.preprocess_state(next_states_batch,
                                                   mem=False,
                                                   batch=True)

              ## Calculate q-learning targets

              q_values = self.q_network.predict_on_batch(next_states_batch)
              target_q_values = self.target_q_network.predict_on_batch(next_states_batch)

              ## For double deep q-networks
              if self.model_type == 'deep_double' or self.model_type == 'dueling':
                max_actions = np.argmax(q_values, axis=1)

              # For deep q-networks
              else:
                max_actions = np.argmax(target_q_values, axis=1)

              targets_batch = \
                rewards_batch + np.invert(is_terminal_batch).astype(np.float32) * \
                self.gamma * target_q_values[np.arange(self.batch_size), max_actions]

              targets_batch_one_hot = np.zeros((self.batch_size, env.action_space.n))
              targets_batch_one_hot[np.arange(self.batch_size), actions_batch] = \
                targets_batch

              ## Update parameters
              loss = self.update_network(states_batch, targets_batch_one_hot)
              cum_loss += loss
              tot_reward = reward
              cum_reward += tot_reward

              state = next_state
              state = np.expand_dims(state, axis=0)
              tot_updates += 1
              print("\r[%i] Loss: %.8f" %
                      (sum_tot_iters+tot_updates, cum_loss/tot_updates),
                    "Reward:", cum_reward, end="")

              if is_terminal:
                break

          loss_file.write("%i %.8f\n" %
                          (sum_tot_iters+tot_updates, cum_loss/tot_updates))

          episode_counter += 1
          episode_lengths.append(tot_updates*4) # approximately this long
          print("\nAverage reward per frame this episode:",
                 cum_reward/(tot_updates*4*self.batch_size),
                "\nEpisode Length:", tot_updates*4,
                "\nNumber of Episodes:", episode_counter, "\n")

          sum_tot_iters += tot_updates

          # If we have to evaluate, do so and set the next evaluation iteration
          if should_eval:
              ## Save Model before evaluating
              self.q_network.save(self.output_folder+'/model_file.h5')
              env.reset()
              self.evaluate(env, num_iterations, max_episode_length,
                            self.output_folder+'/model_file.h5',
                            sum_tot_iters, qvalue_held_out_states)
              next_eval = sum_tot_iters + eval_steps
              should_eval = False
              loss_file.flush()

        loss_file.close()
        print("\nAverage Episode Length: ", sum(episode_lengths)/len(episode_lengths))

    def evaluate(self, env, num_iterations, max_episode_length, filename, update_no, qval_states):
        """Test your agent with a provided environment.

        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        # Run policy
        # Collect stats - reward, avg episode length
        # Render env

        ## Logging stats and Recording video
        env = wrappers.Monitor(env, self.output_folder+"/"+str(update_no),
                               video_callable=self.in_eval)

        rewards_file = open(self.output_folder + "/reward_file", "a")
        qvalues_file = open(self.output_folder + "/qvalues_file", "a")

        self.evaluating = True

        episode_count = 0
        total_episodes = 20
        self.sampling = False
        cum_reward = 0.0

        while episode_count < total_episodes:
          if episode_count < 2:
            self.evaluating = True
          else:
            self.evaluating = False

          self.policy = policy.GreedyPolicy()

          state = env.reset()
          preprocessed_state = self.preprocessor.preprocess_state(state, mem=True)
          state = np.stack([preprocessed_state] * 4, axis=2)
          state = np.expand_dims(state, axis=0)

          while True:

              #if t%20==0: env.render()

              ## Get next state
              q_values = self.q_network.predict(state)
              chosen_action = self.select_action(q_values)

              next_state, reward, is_terminal, info = env.step(chosen_action)
              next_state = self.preprocessor.preprocess_state(next_state, mem=True)
              state = np.squeeze(state, axis=0)
              next_state = \
                np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
              cum_reward += reward

              state = next_state
              state = np.expand_dims(state, axis=0)

              if is_terminal:
                break

          episode_count += 1

        print("Average total reward:", cum_reward/total_episodes)
        rewards_file.write("%i %f\n" % (update_no, cum_reward/total_episodes))
        rewards_file.close()

        # For each of the selected set of states, get the maximum Q-value, and average
        # these to get the max avg Q-value for this eval run
        avg_maxqval = np.mean(np.max(self.q_network.predict_on_batch(qval_states), axis=1))

        qvalues_file.write("%i %.4f\n" % (update_no, avg_maxqval))
        qvalues_file.close()
        self.evaluating = False

    def eval_on_file(self, env, filename):
        episode_count = 0
        total_episodes = 20
        self.sampling = False
        cum_reward = 0.0

        env = wrappers.Monitor(env, self.output_folder, video_callable=self.in_eval)
        self.q_network = load_model(filename, custom_objects={'mean_huber_loss':objectives.mean_huber_loss})

        while episode_count < total_episodes:
          self.policy = policy.GreedyPolicy()

          state = env.reset()
          preprocessed_state = self.preprocessor.preprocess_state(state, mem=True)
          state = np.stack([preprocessed_state] * 4, axis=2)
          state = np.expand_dims(state, axis=0)

          while True:

              ## Get next state
              q_values = self.q_network.predict(state)
              print("Q-values:")
              print(q_values[0], "\n")
              chosen_action = self.select_action(q_values)

              next_state, reward, is_terminal, info = env.step(chosen_action)
              next_state = self.preprocessor.preprocess_state(next_state, mem=True)
              state = np.squeeze(state, axis=0)
              next_state = \
                np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
              cum_reward += reward

              state = next_state
              state = np.expand_dims(state, axis=0)

              if is_terminal:
                break

          episode_count += 1

        print("Average total reward:", cum_reward/total_episodes)

        # For each of the selected set of states, get the maximum Q-value, and average
        # these to get the max avg Q-value for this eval run
        avg_maxqval = np.mean(np.max(self.q_network.predict_on_batch(qval_states), axis=1))
        print(avg_maxqval)
