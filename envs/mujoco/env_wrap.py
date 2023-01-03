import gym
import torch
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from gym.spaces.discrete import Discrete
from gym.spaces.box import Box as Continuous
from copy import deepcopy


class MujocoEnvWrapper(gym.Env):
    def __init__(self, env, partial=False, observation_type=None):
        # Inscribe the environment and some of the parameters.
        self.env = env
        self._max_episode_steps = self.env.env._max_episode_steps
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.state_space = self.env.observation_space

        self.partial = partial

        # If it is a discrete env, force a single action.
        self.is_discrete = type(self.env.action_space) == Discrete
        assert self.is_discrete or type(self.env.action_space) == Continuous
        # if self.is_discrete:
        #     self.env.action_space.shape = (self.env.action_space.n, )

        # Reset the state, and the running total reward
        return_state, state = self.reset()
        if return_state.shape != self.observation_space.shape:
            self.observation_space = self.env.partial_observation_space
        # Number of actions
        action_shape = self.env.action_space.shape
        assert len(action_shape) <= 1  # scalar or vector actions
        self.num_actions = self.env.action_space.n if self.is_discrete else 0 \
                            if len(action_shape) == 0 else action_shape[0]

        # Running total reward (set to 0.0 at resets).
        self.total_true_reward = 0.0

    def reset(self):
        # Reset the state, and the running total reward
        # start_state = torch.tensor(self.env.reset())
        start_state = self.env.reset()
        self.state = start_state  # Keep track of state, why not?
        self.total_true_reward = 0.0
        self.counter = 0.0

        if self.partial:
            return_state = self.env.obscure_state(start_state)
            state = start_state
        else:
            return_state = start_state
            state = start_state

        return return_state, state

    def step(self, action):
        # Step the environment.
        try:
            state, reward, is_done, info = self.env.step(action)
        except Exception as err:
            print(err)
            print('Error in iterating environment.')
            raise RuntimeError

        # AW - lets keep track of the state in the env as well, will
        # make switching the obs_type later on much more straightforward.
        self.state = state
        self.total_true_reward += reward
        self.counter += 1

        # If we are done, inscribe some info,
        if is_done:
            info['done'] = (self.total_true_reward, self.counter)

        if self.partial:
            # Return type is flag used in WL code to return render for vanilla RL library.  manually overwrite the
            # none flag to indicate that this behaviour should be used.  Will break all other code...
            return_state = self.env.obscure_state(state)
            info['state'] = state
        else:
            return_state = state
            info['state'] = state

        return return_state, float(reward), is_done, info
