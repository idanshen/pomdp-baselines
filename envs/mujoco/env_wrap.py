from collections import deque

import gym
import torch
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.spaces import Box

from gym.spaces.discrete import Discrete
from gym.spaces.box import Box as Continuous
from copy import deepcopy


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class MujocoEnvWrapper(gym.Env):
    def __init__(self, env, partial=False, observation_type=None, normalize_obs=False):
        # Inscribe the environment and some of the parameters.
        self.env = env
        self._max_episode_steps = self.env._max_episode_steps
        self.action_space = self.env.action_space
        self.norm_obs = False

        if type(self.env.observation_space) is gym.spaces.dict.Dict:
            self.dict_obs = True
            self.observation_space = self.env.observation_space["observation"]
            self.state_space = self.env.observation_space["observation"]
        else:
            self.dict_obs = False
            self.observation_space = self.env.observation_space
            self.state_space = self.env.observation_space

        self.partial = partial

        # If it is a discrete env, force a single action.
        self.is_discrete = type(self.env.action_space) == Discrete
        assert self.is_discrete or type(self.env.action_space) == Continuous
        # if self.is_discrete:
        #     self.env.action_space.shape = (self.env.action_space.n, )

        # Reset the state, and the running total reward
        if self.env.spec.id == 'HalfCheetah-v3':
            self.state_queue = deque([np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])] * 10, 10)
        elif self.env.spec.id == 'Hopper-v3':
            self.state_queue = deque([np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])] * 10, 10)
        elif self.env.spec.id == 'Walker2d-v3':
            self.state_queue = deque([np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])] * 10, 10)
        return_state, state = self.reset()
        # if self.dict_obs:
        #     if return_state["observation"].shape != self.observation_space.shape:
        #         self.observation_space = self.env.partial_observation_space["observation"]
        # else:
        if return_state.shape != self.observation_space.shape:
            if self.env.spec.id == 'Hopper-v3':
                self.observation_space = spaces.Box(-np.inf, np.inf, shape=(60,), dtype="float64")
            elif self.env.spec.id == 'HalfCheetah-v3':
                # self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,), dtype="float64")  # velocity only
                self.observation_space = spaces.Box(-np.inf, np.inf, shape=(90,), dtype="float64")  # position only
            elif self.env.spec.id == 'Walker2d-v3':
                self.observation_space = spaces.Box(-np.inf, np.inf, shape=(90,), dtype="float64")
            else:
                self.observation_space = self.env.partial_observation_space
        # Number of actions
        action_shape = self.env.action_space.shape
        assert len(action_shape) <= 1  # scalar or vector actions
        self.num_actions = self.env.action_space.n if self.is_discrete else 0 \
                            if len(action_shape) == 0 else action_shape[0]

        # Running total reward (set to 0.0 at resets).
        self.total_true_reward = 0.0

        self.epsilon = 1e-6
        self.norm_obs = normalize_obs
        if self.norm_obs:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        if self.norm_obs:
            obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        return obs

    def reset(self):
        # Reset the state, and the running total reward
        # start_state = torch.tensor(self.env.reset())
        start_state = self.env.reset()
        if self.dict_obs:
            start_state = start_state["observation"]
        self.state = start_state  # Keep track of state, why not?
        self.total_true_reward = 0.0
        self.counter = 0.0

        if self.partial:
            return_state = self.obscure_state(start_state)
            state = start_state
        else:
            return_state = start_state
            state = start_state

        return_state = self.normalize_obs(return_state)
        return return_state, state

    def step(self, action):
        # Step the environment.
        try:
            state, reward, is_done, info = self.env.step(action)
            if self.dict_obs:
                state = state["observation"]
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
            return_state = self.obscure_state(np.copy(state))
            info['state'] = state
        else:
            return_state = state
            info['state'] = state

        if self.norm_obs:
            self.obs_rms.update(return_state)

        if 'reached_goal' not in info:
            info['reached_goal'] = False
        return_state = self.normalize_obs(return_state)

        return return_state, float(reward), is_done, info

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        return self.env.seed(seed)

    def obscure_state(self, obs):
        if self.env.spec.id == 'Hopper-v3':
            new_loc = obs[5:].copy()
            self.state_queue.pop()
            self.state_queue.appendleft(new_loc)
            return_state = np.hstack(self.state_queue)
        elif self.env.spec.id == 'HalfCheetah-v3':
            # return_state = obs[8:].copy()  # velocity only
            new_loc = obs[:9].copy()  # position only
            self.state_queue.pop()
            self.state_queue.appendleft(new_loc)
            return_state = np.hstack(self.state_queue)
        elif self.env.spec.id == 'Walker2d-v3':
            new_loc = obs[8:].copy()
            self.state_queue.pop()
            self.state_queue.appendleft(new_loc)
            return_state = np.hstack(self.state_queue)
        else:
            return_state = self.env.obscure_state(obs)
        return return_state
