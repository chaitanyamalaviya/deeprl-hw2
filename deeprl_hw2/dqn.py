"""Main DQN agent."""
from keras.optimizers import Adam
from keras.models import Model, Sequential
from deeprl_hw2.core import Preprocessor, ReplayMemory, Sample
import utils, objectives, policy

import numpy as np

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
                 epsilon):

        self.q_network = q_network
        self.preprocessor = preprocessor
        self.memory = memory
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size

        self.epsilon = epsilon
        self.sampling = True
        self.is_training = True
        self.target_q_network = Sequential()

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

        self.target_q_network = utils.get_hard_target_model_updates(self.target_q_network, self.q_network)
        adam = Adam(lr=1e-6)

        # self.q_network.compile(loss=objectives.mean_huber_loss,
        #                        optimizer=adam,
        #                        metrics=['accuracy'])
        self.q_network.compile(loss='mean_squared_error',
                               optimizer=adam,
                               metrics=['accuracy'])

        print("Model compiled.")

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        actions = self.q_network.predict_on_batch(state) # state is actually a batch of states
        return actions

    def select_action(self, env, state):
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

        if self.is_training and self.sampling:
          policy = self.policy.UniformRandomPolicy(env.action_space.n)
          chosen_action = policy.select_action()
        elif self.is_training:
          policy = self.policy.LinearDecayGreedyEpsilonPolicy()
          chosen_action = policy.select_action()
        else:
          policy = self.policy.GreedyEpsilonPolicy()
          chosen_action = policy.select_action()

        return chosen_action


    def update_policy(self):
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
        
        pass


    def fit(self, env, num_iterations, max_episode_length=None):
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

        if self.is_training and self.sampling:
          print("Filling up replay memory..")
          chosen_policy = policy.UniformRandomPolicy(env.action_space.n)
          state = env.reset()
          preprocessed_state = self.preprocessor.preprocess_state(state)
          state = np.stack([preprocessed_state] * 4, axis=2)

          for i in range(self.num_burn_in):
              action = chosen_policy.select_action()
              next_state, reward, is_terminal, _ = env.step(action)
              next_state = self.preprocessor.preprocess_state(next_state)
              next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
              self.memory.append(Sample(state, action, reward, next_state, is_terminal))
              if is_terminal:
                  state = env.reset()
                  preprocessed_state = self.preprocessor.preprocess_state(state)
                  state = np.stack([preprocessed_state] * 4, axis=2)
              else:
                  state = next_state


        # Logging and Recording
        env.monitor.start(output_folder, resume=True, video_callable=lambda count: count % 50 == 0)
        tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

        #model.fit(x_train, y_train, batch_size=16, epochs=10)
        #score = model.evaluate(x_test, y_test, batch_size=16)
        tot_iters = 0
        cum_reward = 0

        for i in range(num_iterations):
          chosen_policy = policy.GreedyEpsilonPolicy(self.epsilon)
          state = env.reset()
          preprocessed_state = self.preprocessor.preprocess_state(state)
          state = np.stack([preprocessed_state] * 4, axis=2)
          loss = 0

          for t in range(max_episode_len):

              env.render()
              if tot_iters % self.target_update_freq == 0:
                self.target_q_network = get_hard_target_model_updates(self.target_q_network, self.q_network)

              q_values = self.q_network.predict(state)
              chosen_action = chosen_policy.select_action(q_values)
              next_state, reward, is_terminal, _ = env.step(chosen_action)
              next_state = self.preprocessor.preprocess_state(next_state)
              next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
              self.memory.append(Sample(state, action, reward, next_state, is_terminal))
              samples = self.memory.sample(self.batch_size)
              states_batch, actions_batch, rewards_batch, next_states_batch, is_terminal_batch = map(np.array, zip(*samples))

              q_values_next = self.q_network.predict(next_states_batch)
              best_actions = np.argmax(q_values_next, axis=1)
              q_values_next_target = self.target_q_network.predict(next_states_batch)
              targets_batch = rewards_batch + np.invert(is_terminal_batch).astype(np.float32) * \
              self.gamma * q_values_next_target[np.arange(self.batch_size), best_actions]

              states_batch = np.array(states_batch)
              loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

              if is_terminal:
                break

              state = next_state
              cum_reward += reward
              tot_iters += 1

        env.monitor.close()


    def evaluate(self, env, num_episodes, max_episode_length=None):
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
