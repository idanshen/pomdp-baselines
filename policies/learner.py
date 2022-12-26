# -*- coding: future_fstrings -*-
import os, sys
import time

import math
import numpy as np
import torch
from torch.nn import functional as F
import gym

from envs.gridworld.env_wrap import EnvWrapper
from .models import AGENT_CLASSES, AGENT_ARCHS
from torchkit.networks import ImageEncoder

# Markov policy
from buffers.simple_replay_buffer import SimpleReplayBuffer

# RNN policy on vector-based task
from buffers.seq_replay_buffer_vanilla import SeqReplayBuffer

# RNN policy on image/vector-based task
from buffers.seq_replay_buffer_efficient import RAMEfficient_SeqReplayBuffer

from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from utils import evaluation as utl_eval
from utils import logger
from .rl import EAACD

CFG_TO_RENDER_TYPE = {
    "partial": "partial_state",
    "full": "state",
    "image_full": "full_observe",
    "image_partial": "observe",
}

class Learner:
    def __init__(self, env_args, train_args, eval_args, policy_args, seed, **kwargs):
        self.seed = seed

        self.init_env(**env_args)

        self.init_agent(**policy_args)

        self.init_train(**train_args)

        self.init_eval(**eval_args)

    def init_env(
        self,
        env_type,
        env_name,
        max_rollouts_per_task=None,
        num_tasks=None,
        num_train_tasks=None,
        num_eval_tasks=None,
        eval_envs=None,
        worst_percentile=None,
        save_states=True,
        obseravibility="full",
        **kwargs
    ):

        # initialize environment
        assert env_type in [
            "meta",
            "pomdp",
            "credit",
            "rmdp",
            "generalize",
            "atari",
            "gridworld"
        ]
        self.env_type = env_type
        self.save_states = save_states

        if self.env_type == "meta":  # meta tasks: using varibad wrapper
            from envs.meta.make_env import make_env

            self.train_env = make_env(
                env_name,
                max_rollouts_per_task,
                seed=self.seed,
                n_tasks=num_tasks,
                **kwargs,
            )  # oracle in kwargs
            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)

            if self.train_env.n_tasks is not None:
                # NOTE: This is off-policy varibad's setting, i.e. limited training tasks
                # split to train/eval tasks
                assert num_train_tasks >= num_eval_tasks > 0
                shuffled_tasks = np.random.permutation(
                    self.train_env.unwrapped.get_all_task_idx()
                )
                self.train_tasks = shuffled_tasks[:num_train_tasks]
                self.eval_tasks = shuffled_tasks[-num_eval_tasks:]
            else:
                # NOTE: This is on-policy varibad's setting, i.e. unlimited training tasks
                assert num_tasks == num_train_tasks == None
                assert (
                    num_eval_tasks > 0
                )  # to specify how many tasks to be evaluated each time
                self.train_tasks = []
                self.eval_tasks = num_eval_tasks * [None]

            # calculate what the maximum length of the trajectories is
            self.max_rollouts_per_task = max_rollouts_per_task
            self.max_trajectory_len = self.train_env.horizon_bamdp  # H^+ = k * H

        elif self.env_type in ["pomdp","credit",]:  # pomdp/mdp task, using pomdp wrapper
            import envs.pomdp
            import envs.credit_assign

            assert num_eval_tasks > 0
            self.train_env = gym.make(env_name)
            self.train_env.seed(self.seed)
            self.train_env.action_space.np_random.seed(self.seed)  # crucial

            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)

            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps

        elif self.env_type == "atari":
            from envs.atari import create_env

            assert num_eval_tasks > 0
            self.train_env = create_env(env_name)
            self.train_env.seed(self.seed)
            self.train_env.action_space.np_random.seed(self.seed)  # crucial

            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)

            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps

        elif self.env_type == "rmdp":  # robust mdp task, using robust mdp wrapper
            sys.path.append("envs/rl-generalization")
            import sunblaze_envs

            assert (
                num_eval_tasks > 0 and worst_percentile > 0.0 and worst_percentile < 1.0
            )
            self.train_env = sunblaze_envs.make(env_name, **kwargs)  # oracle
            self.train_env.seed(self.seed)
            assert np.all(self.train_env.action_space.low == -1)
            assert np.all(self.train_env.action_space.high == 1)

            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)

            self.worst_percentile = worst_percentile

            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env.max_steps

        elif self.env_type == "generalize":
            sys.path.append("envs/rl-generalization")
            import sunblaze_envs

            self.train_env = sunblaze_envs.make(env_name, **kwargs)  # oracle in kwargs
            self.train_env.seed(self.seed)
            assert np.all(self.train_env.action_space.low == -1)
            assert np.all(self.train_env.action_space.high == 1)

            def check_env_class(env_name):
                if "Normal" in env_name:
                    return "R"
                if "Extreme" in env_name:
                    return "E"
                return "D"

            self.train_env_name = check_env_class(env_name)

            self.eval_envs = {}
            for env_name, num_eval_task in eval_envs.items():
                eval_env = sunblaze_envs.make(env_name, **kwargs)  # oracle in kwargs
                eval_env.seed(self.seed + 1)
                self.eval_envs[eval_env] = (
                    check_env_class(env_name),
                    num_eval_task,
                )  # several types of evaluation envs

            logger.log(self.train_env_name, self.train_env)
            logger.log(self.eval_envs)

            self.train_tasks = []
            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps

        elif self.env_type == "gridworld":
            import envs.gridworld

            assert num_eval_tasks > 0
            assert obseravibility in CFG_TO_RENDER_TYPE.keys()

            # all from https://github.com/plai-group/a2d/blob/368aaf5e0e425c2d48d25925c772a0d9a4f3823b/a2d_gridworld/tests/A2D/A2Dagger_arguments.py
            env_params = {"render_type": CFG_TO_RENDER_TYPE[obseravibility],
                          "return_type": CFG_TO_RENDER_TYPE[obseravibility],
                          "frame_stack": 1,
                          "env_name": env_name,
                          "env_tag": 'minigrid',
                          "resize": 42,
                          }

            self.train_env = EnvWrapper(gym.make(env_name), env_params)
            self.train_env.seed(self.seed)
            self.train_env.action_space.np_random.seed(self.seed)  # crucial

            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)

            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps

        else:
            raise ValueError

        # get action / observation dimensions
        if self.train_env.action_space.__class__.__name__ == "Box":
            # continuous action space
            self.act_dim = self.train_env.action_space.shape[0]
            self.act_continuous = True
        else:
            assert self.train_env.action_space.__class__.__name__ == "Discrete"
            self.act_dim = self.train_env.action_space.n
            self.act_continuous = False
        self.obs_dim = self.train_env.observation_space.shape  # include 1-dim done
        self.state_dim = self.train_env.state_space.shape

        logger.log("obs_dim", self.obs_dim, "act_dim", self.act_dim)

    def init_agent(
        self,
        seq_model,
        separate: bool = True,
        image_encoder=None,
        reward_clip=False,
        **kwargs
    ):
        # initialize agent
        agent_class, rnn_encoder_type = utl.parse_seq_model(seq_model, separate)
        self.agent_arch = agent_class.ARCH
        logger.log(agent_class, self.agent_arch)

        if image_encoder is not None:  # catch, keytodoor
            image_encoder_fn = lambda: ImageEncoder(
                image_shape=self.train_env.observation_space.shape, **image_encoder
            )
        else:
            image_encoder_fn = lambda: None

        self.agent = agent_class(
            encoder=rnn_encoder_type,
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            image_encoder_fn=image_encoder_fn,
            state_dim=self.state_dim[0],
            **kwargs,
        ).to(ptu.device)
        logger.log(self.agent)

        self.reward_clip = reward_clip  # for atari
        if kwargs["algo_name"] in ["eaacd", "DAgger"]:
            if kwargs['teacher_dir'] == 'oracle':
                self.teacher = "oracle"
            else:
                self.teacher = EAACD.load_teacher(kwargs['teacher_dir'], state_dim=self.state_dim[0], act_dim=self.act_dim)
        else:
            self.teacher = None

    def init_train(
        self,
        buffer_size,
        batch_size,
        num_iters,
        num_init_rollouts_pool,
        num_rollouts_per_iter,
        epsilon=0.1,
        num_updates_per_iter=None,
        sampled_seq_len=None,
        sample_weight_baseline=None,
        buffer_type=None,
        **kwargs
    ):
        if num_updates_per_iter is None:
            num_updates_per_iter = 1.0
        assert isinstance(num_updates_per_iter, int) or isinstance(
            num_updates_per_iter, float
        )
        # if int, it means absolute value; if float, it means the multiplier of collected env steps
        self.num_updates_per_iter = num_updates_per_iter

        if self.agent_arch == AGENT_ARCHS.Markov:
            self.policy_storage = SimpleReplayBuffer(
                max_replay_buffer_size=int(buffer_size),
                observation_dim=self.obs_dim,
                state_dim=self.state_dim,
                teacher_action_dim=self.act_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                max_trajectory_len=self.max_trajectory_len,
                add_timeout=False,  # no timeout storage
            )

        else:  # memory, memory-markov
            if sampled_seq_len == -1:
                sampled_seq_len = self.max_trajectory_len

            if buffer_type is None or buffer_type == SeqReplayBuffer.buffer_type:
                buffer_class = SeqReplayBuffer
            elif buffer_type == RAMEfficient_SeqReplayBuffer.buffer_type:
                buffer_class = RAMEfficient_SeqReplayBuffer
            logger.log(buffer_class)

            self.policy_storage = buffer_class(
                max_replay_buffer_size=int(buffer_size),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                teacher_action_dim=self.act_dim,
                sampled_seq_len=sampled_seq_len,
                sample_weight_baseline=sample_weight_baseline,
                observation_type=self.train_env.observation_space.dtype,
                state_dim=self.state_dim if self.save_states else None,
                state_type=self.train_env.observation_space.dtype,  # TODO: fix later
            )

        self.batch_size = batch_size
        self.num_iters = num_iters
        self.num_init_rollouts_pool = num_init_rollouts_pool
        self.num_rollouts_per_iter = num_rollouts_per_iter
        self.epsilon = epsilon

        total_rollouts = num_init_rollouts_pool + num_iters * num_rollouts_per_iter
        self.n_env_steps_total = self.max_trajectory_len * total_rollouts
        logger.log(
            "*** total rollouts",
            total_rollouts,
            "total env steps",
            self.n_env_steps_total,
        )

        self.data_collection_method = kwargs["data_collection_method"]

    def init_eval(
        self,
        log_interval,
        save_interval,
        log_tensorboard,
        eval_stochastic=False,
        num_episodes_per_task=1,
        **kwargs
    ):

        self.log_interval = log_interval
        self.save_interval = save_interval
        self.log_tensorboard = log_tensorboard
        self.eval_stochastic = eval_stochastic
        self.eval_num_episodes_per_task = num_episodes_per_task

    def _start_training(self):
        self._n_env_steps_total = 0
        self._n_env_steps_total_last = 0
        self._n_rl_update_steps_total = 0
        self._n_rollouts_total = 0
        self._successes_in_buffer = 0

        self._start_time = time.time()
        self._start_time_last = time.time()

    def train(self):
        """
        training loop
        """

        self._start_training()

        if self.num_init_rollouts_pool > 0:
            logger.log("Collecting initial pool of data..")
            # while (
            #     self._n_env_steps_total
            #     < self.num_init_rollouts_pool * self.max_trajectory_len
            # ):
            #     self.collect_rollouts(
            #         num_rollouts=1,
            #         # random_actions=True,
            #     )
            if self.agent.algo_name in ["eaacd", "Dagger"]:
                self.collect_rollouts(num_rollouts=self.num_init_rollouts_pool)
            else:
                self.collect_rollouts(num_rollouts=self.num_init_rollouts_pool, random_actions=True)
            logger.log(
                "Done! env steps",
                self._n_env_steps_total,
                "rollouts",
                self._n_rollouts_total,
            )

            if isinstance(self.num_updates_per_iter, float):
                # update: pomdp task updates more for the first iter_
                train_stats = self.update(
                    int(self._n_env_steps_total * self.num_updates_per_iter)
                )
                self.log_train_stats(train_stats)

        last_eval_num_iters = 0
        while self._n_env_steps_total < self.n_env_steps_total:
            # collect data from num_rollouts_per_iter train tasks:
            env_steps = self.collect_rollouts(num_rollouts=self.num_rollouts_per_iter, min_steps=0.25*self.num_rollouts_per_iter*self.max_trajectory_len)
            logger.log("env steps", self._n_env_steps_total)

            if self.env_type == "gridworld" and self.agent.algo_name == "eaacd":
                if self.agent.algo.coefficient_tuning == "EIPO":
                    self.agent.algo.current_policy = "main"
                    returns_eval_main, _, _, _ = self.evaluate(self.eval_tasks, deterministic=True)
                    self.agent.algo.obj_est_main = returns_eval_main.mean()
                    self.agent.algo.current_policy = "aux"
                    returns_eval_aux, _, _, _ = self.evaluate(self.eval_tasks, deterministic=True)
                    self.agent.algo.obj_est_aux = returns_eval_aux.mean()
                    self.agent.algo.current_policy = "main"

            train_stats = self.update(
                self.num_updates_per_iter
                if isinstance(self.num_updates_per_iter, int)
                else int(math.ceil(self.num_updates_per_iter * env_steps))
            )  # NOTE: ceil to make sure at least 1 step
            self.log_train_stats(train_stats)

            # evaluate and log
            current_num_iters = self._n_env_steps_total // (
                self.num_rollouts_per_iter * self.max_trajectory_len
            )
            if (
                current_num_iters != last_eval_num_iters
                and current_num_iters % self.log_interval == 0
            ):
                last_eval_num_iters = current_num_iters
                perf = self.log()
                if (
                    self.save_interval > 0
                    and self._n_env_steps_total > 0.75 * self.n_env_steps_total
                    and current_num_iters % self.save_interval == 0
                ):
                    # save models in later training stage
                    self.save_model(current_num_iters, perf)
        self.save_model(current_num_iters, perf)

    @torch.no_grad()
    def collect_rollouts(self, num_rollouts, min_steps=-1, random_actions=False):
        """collect num_rollouts of trajectories in task and save into policy buffer
        :param min_steps: minimum number of steps to collect
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        """

        before_env_steps = self._n_env_steps_total
        collected_rollouts = 0
        collected_steps = 0
        while (collected_rollouts < num_rollouts) or (collected_steps < min_steps):
            steps = 0

            if self.data_collection_method in ["start_student_than_teacher", "start_beta_student_aux_than_teacher"]:
                switch_step = np.random.randint(self.max_trajectory_len)

            if self.env_type == "meta" and self.train_env.n_tasks is not None:
                raise NotImplementedError
                # task = self.train_tasks[np.random.randint(len(self.train_tasks))]
                # obs = ptu.from_numpy(self.train_env.reset(task=task))  # reset task
            else:
                obs, state = self.train_env.reset()
                obs = ptu.from_numpy(obs)  # reset
                state = ptu.from_numpy(state)

            obs = obs.reshape(1, *obs.shape)
            state = state.reshape(1, *state.shape)
            done_rollout = False

            if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
                # temporary storage
                obs_list, act_list, rew_list, next_obs_list, term_list = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )

                if self.save_states:
                    state_list, next_state_list = [], []
                else:
                    state_list, next_state_list = None, None
                if self.teacher is not None:
                    teacher_log_prob_list = []
                    teacher_next_log_prob_list = []
                else:
                    teacher_log_prob_list = None
                    teacher_next_log_prob_list = None

            if self.agent_arch == AGENT_ARCHS.Memory:
                # get hidden state at timestep=0, None for markov
                # NOTE: assume initial reward = 0.0 (no need to clip)
                action, reward, internal_state = self.agent.get_initial_info(self.agent.algo.current_policy)

            while not done_rollout:
                # 1. Collect teacher action for the current state
                if self.teacher is not None:
                    if self.teacher == "oracle":
                        teacher_action = ptu.FloatTensor(
                            [self.train_env.env.query_expert()[0]]
                        )  # (1, A) for continuous action, (1) for discrete action
                        teacher_prob_action = F.one_hot(
                            teacher_action.long(), num_classes=self.act_dim
                        ).float()  # (1, A)
                        teacher_log_prob_action = torch.clip(torch.log(teacher_prob_action), -18.0, 18.0)
                    else:
                        teacher_prob_action, _, teacher_log_prob_action, _ = self.teacher["main"].act(state, return_log_prob=True)
                else:
                    teacher_log_prob_action = None

                # 2. determine action according to data collection policy
                if self.data_collection_method == "only_student":
                    collect_from_policy = True
                    self.agent.algo.current_policy = "main"
                elif self.data_collection_method == "only_teacher":
                    collect_from_policy = False
                elif self.data_collection_method == "based_on_coefficient_update":
                    collect_from_policy = True
                    if self.agent.algo.obj_est_main - self.agent.algo.obj_est_aux >= 0:
                        self.agent.algo.current_policy = "main"
                    else:
                        self.agent.algo.current_policy = "aux"
                elif self.data_collection_method == "beta_student_teacher":
                    prob = self.agent.algo.coefficient / (self.agent.algo.coefficient+1.0)  # Prob to get teacher
                    if np.random.random() < prob:
                        collect_from_policy = False
                    else:
                        collect_from_policy = True
                        self.agent.algo.current_policy = "main"
                elif self.data_collection_method == "beta_student_aux":
                    prob = self.agent.algo.coefficient / (self.agent.algo.coefficient + 1.0)  # Prob to get aux
                    if np.random.random() < prob:
                        collect_from_policy = True
                        self.agent.algo.current_policy = "aux"
                    else:
                        collect_from_policy = True
                        self.agent.algo.current_policy = "main"
                elif self.data_collection_method == "start_student_than_teacher":
                    if steps < switch_step:
                        collect_from_policy = True
                        self.agent.algo.current_policy = "main"
                    else:
                        collect_from_policy = False
                elif self.data_collection_method == "start_beta_student_aux_than_teacher":
                    if steps < switch_step:
                        prob = self.agent.algo.coefficient / (self.agent.algo.coefficient + 1.0)  # Prob to get aux
                        if np.random.random() < prob:
                            collect_from_policy = True
                            self.agent.algo.current_policy = "aux"
                        else:
                            collect_from_policy = True
                            self.agent.algo.current_policy = "main"
                    else:
                        collect_from_policy = False
                elif self.data_collection_method == "all":
                    i = collected_rollouts % 3
                    if i == 0:
                        collect_from_policy = False
                    elif i == 1:
                        collect_from_policy = True
                        self.agent.algo.current_policy = "aux"
                    else:
                        collect_from_policy = True
                        self.agent.algo.current_policy = "main"
                elif self.data_collection_method == "both":
                    i = collected_rollouts % 2
                    if i == 0:
                        collect_from_policy = True
                        self.agent.algo.current_policy = "aux"
                    else:
                        collect_from_policy = True
                        self.agent.algo.current_policy = "main"
                else:
                    raise NotImplementedError

                if random_actions or np.random.random() < self.epsilon:
                    # action = teacher_prob_action
                    action = ptu.FloatTensor(
                        [self.train_env.action_space.sample()]
                    )  # (1, A) for continuous action, (1) for discrete action
                    if not self.act_continuous:
                        action = F.one_hot(
                            action.long(), num_classes=self.act_dim
                        ).float()  # (1, A)
                elif collect_from_policy:
                    # policy takes hidden state as input for memory-based actor,
                    # while takes obs for markov actor
                    if self.agent_arch == AGENT_ARCHS.Memory:
                        (action, _, log_prob, _), internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            reward=reward,
                            obs=obs,
                            deterministic=False,
                            return_log_prob=True,
                        )
                    else:
                        action, _, _, _ = self.agent.act(obs, deterministic=False)
                else:  # collect from teacher
                    action = teacher_prob_action

                # 3. Take action, observe reward, next state and next obs
                next_obs, reward, done, info = utl.env_step(
                    self.train_env, action.squeeze(dim=0)
                )
                next_state = info["state"]

                # 4. Collect teacher action for the next state. If last state than append vector of zeroes.
                # TODO: can be done in more efficient way in the next iteration
                if self.teacher is not None:
                    if not done.item():
                        if self.teacher == "oracle":
                            teacher_next_action = ptu.FloatTensor(
                                [self.train_env.env.query_expert()[0]]
                            )  # (1, A) for continuous action, (1) for discrete action
                            teacher_prob_next_action = F.one_hot(
                                teacher_next_action.long(), num_classes=self.act_dim
                            ).float()  # (1, A)
                            teacher_log_prob_next_action = torch.clip(torch.log(teacher_prob_next_action), -18.0, 18.0)
                        else:
                            _, _, teacher_log_prob_next_action, _ = self.teacher["main"].act(next_state, return_log_prob=True)
                    else:
                        teacher_log_prob_next_action = torch.zeros(1, self.act_dim,
                                                                   device=teacher_log_prob_action.device).float()
                else:
                    teacher_log_prob_next_action = None

                # 5. Determine terminal flag per environment
                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True
                steps += 1

                if self.env_type == "meta" and "is_goal_state" in dir(self.train_env.unwrapped):
                    # NOTE: following varibad practice: for meta env, even if reaching the goal (term=True),
                    # the episode still continues.
                    term = self.train_env.unwrapped.is_goal_state()
                    self._successes_in_buffer += int(term)
                elif self.env_type == "credit":  # delayed rewards
                    term = done_rollout
                else:
                    # term ignore time-out scenarios, but record early stopping
                    term = (
                        False
                        if "TimeLimit.truncated" in info
                           or steps >= self.max_trajectory_len
                        else done_rollout
                    )

                # 6. Save current tuple to replay data or trajectory list
                if self.reward_clip and self.env_type == "atari":
                    reward = torch.tanh(reward)

                if self.agent_arch == AGENT_ARCHS.Markov:
                    self.policy_storage.add_sample(
                        observation=ptu.get_numpy(obs.squeeze(dim=0)),
                        action=ptu.get_numpy(
                            action.squeeze(dim=0)
                            if self.act_continuous
                            else torch.argmax(
                                action.squeeze(dim=0), dim=-1, keepdims=True
                            )  # (1,)
                        ),
                        reward=ptu.get_numpy(reward.squeeze(dim=0)),
                        terminal=np.array([term], dtype=float),
                        next_observation=ptu.get_numpy(next_obs.squeeze(dim=0)),
                        state=ptu.get_numpy(state.squeeze(dim=0)) if self.save_states else None,
                        next_state=ptu.get_numpy(next_state.squeeze(dim=0)) if self.save_states else None,
                        teacher_log_prob=ptu.get_numpy(teacher_log_prob_action.squeeze(dim=0)) if teacher_log_prob_action is not None else None,
                        teacher_next_log_prob=ptu.get_numpy(teacher_log_prob_next_action.squeeze(dim=0)) if teacher_log_prob_next_action is not None else None,
                    )
                else:  # append tensors to temporary storage
                    obs_list.append(obs)  # (1, dim)
                    act_list.append(action)  # (1, dim)
                    rew_list.append(reward)  # (1, dim)
                    term_list.append(term)  # bool
                    next_obs_list.append(next_obs)  # (1, dim)
                    if self.save_states:
                        state_list.append(state)
                        next_state_list.append(next_state)
                    if self.teacher is not None:
                        teacher_log_prob_list.append(teacher_log_prob_action)
                        teacher_next_log_prob_list.append(teacher_log_prob_next_action)

                # 7. set: obs <- next_obs
                obs = next_obs.clone()
                state = next_state.clone()

            if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
                if len(obs_list) > 1:
                    # add collected sequence to buffer
                    act_buffer = torch.cat(act_list, dim=0)  # (L, dim)
                    if not self.act_continuous:
                        act_buffer = torch.argmax(
                            act_buffer, dim=-1, keepdims=True
                        )  # (L, 1)

                    self.policy_storage.add_episode(
                        observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),  # (L, dim)
                        actions=ptu.get_numpy(act_buffer),  # (L, dim)
                        rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                        terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                        next_observations=ptu.get_numpy(torch.cat(next_obs_list, dim=0)),  # (L, dim)
                        states=ptu.get_numpy(torch.cat(state_list, dim=0)) if next_state_list is not None else None,  # (L, dim)
                        next_states=ptu.get_numpy(torch.cat(next_state_list, dim=0)) if next_state_list is not None else None,  # (L, dim)
                        teacher_log_probs=ptu.get_numpy(torch.cat(teacher_log_prob_list, dim=0)) if teacher_log_prob_list is not None else None,  # (L, dim)
                        teacher_next_log_probs=ptu.get_numpy(torch.cat(teacher_next_log_prob_list, dim=0)) if teacher_next_log_prob_list is not None else None,  # (L, dim)
                    )
                    print(
                        f"steps: {steps} term: {term} ret: {torch.cat(rew_list, dim=0).sum().item():.2f}"
                    )
                    self._n_env_steps_total += steps
                    collected_steps += steps
                    self._n_rollouts_total += 1
                    collected_rollouts += 1
            else:
                self._n_env_steps_total += steps
                collected_steps += steps
                self._n_rollouts_total += 1
                collected_rollouts += 1
        return self._n_env_steps_total - before_env_steps

    def sample_rl_batch(self, batch_size):
        """sample batch of episodes for vae training"""
        if self.agent_arch == AGENT_ARCHS.Markov:
            batch = self.policy_storage.random_batch(batch_size)
        else:  # rnn: all items are (sampled_seq_len, B, dim)
            batch = self.policy_storage.random_episodes(batch_size)
        batch = ptu.np_to_pytorch_batch(batch)
        batch["rew_mean"] = self.policy_storage.reward_mean
        batch["rew_std"] = self.policy_storage.reward_std
        return batch

    def update(self, num_updates):
        rl_losses_agg = {}
        for update in range(num_updates):
            # sample random RL batch: in transitions
            batch = self.sample_rl_batch(self.batch_size)

            # RL update
            rl_losses = self.agent.update(batch)

            for k, v in rl_losses.items():
                if update == 0:  # first iterate - create list
                    rl_losses_agg[k] = [v]
                else:  # append values
                    rl_losses_agg[k].append(v)
        # statistics
        for k in rl_losses_agg:
            rl_losses_agg[k] = np.mean(rl_losses_agg[k])
        self._n_rl_update_steps_total += num_updates

        return rl_losses_agg

    @torch.no_grad()
    def evaluate(self, tasks, deterministic=True):
        self.agent.eval()

        num_episodes = self.max_rollouts_per_task  # k
        # max_trajectory_len = k*H
        returns_per_episode = np.zeros((len(tasks), num_episodes))
        success_rate = np.zeros(len(tasks))
        total_steps = np.zeros(len(tasks))

        if self.env_type == "meta":
            num_steps_per_episode = self.eval_env.unwrapped._max_episode_steps  # H
            obs_size = self.eval_env.unwrapped.observation_space.shape[
                0
            ]  # original size
            observations = np.zeros((len(tasks), self.max_trajectory_len + 1, obs_size))
        elif self.env_type == "gridworld":
            num_steps_per_episode = self.eval_env.unwrapped._max_episode_steps  # H
            observations = None
        else:  # pomdp, rmdp, generalize
            num_steps_per_episode = self.eval_env._max_episode_steps
            observations = None

        for task_idx, task in enumerate(tasks):
            step = 0

            if self.env_type == "meta" and self.eval_env.n_tasks is not None:
                raise NotImplementedError
                # obs = ptu.from_numpy(self.eval_env.reset(task=task))  # reset task
                # observations[task_idx, step, :] = ptu.get_numpy(obs[:obs_size])
            else:
                obs, state = self.train_env.reset()
                obs = ptu.from_numpy(obs)  # reset
                state = ptu.from_numpy(state)

            obs = obs.reshape(1, *obs.shape)
            state = state.reshape(1, state.shape[-1])

            if self.agent_arch == AGENT_ARCHS.Memory:
                # assume initial reward = 0.0
                action, reward, internal_state = self.agent.get_initial_info(self.agent.algo.current_policy)

            for episode_idx in range(num_episodes):
                running_reward = 0.0
                # obs_list = [obs]
                # action_list = [action]
                # img_list = [self.eval_env.env.special_render()]
                for _ in range(num_steps_per_episode):
                    if self.agent_arch == AGENT_ARCHS.Memory:
                        (action, _, _, _), internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            reward=reward,
                            obs=obs,
                            deterministic=deterministic,
                        )
                    else:
                        action, _, _, _ = self.agent.act(
                            obs, deterministic=deterministic
                        )

                    # if self.teacher is not None:
                    #     if self.teacher == "oracle":
                    #         teacher_action = ptu.FloatTensor(
                    #             [self.eval_env.env.query_expert()[0]]
                    #         )  # (1, A) for continuous action, (1) for discrete action
                    #         teacher_prob_action = F.one_hot(
                    #             teacher_action.long(), num_classes=self.act_dim
                    #         ).float()  # (1, A)
                    #         teacher_log_prob_action = torch.clip(torch.log(teacher_prob_action), -18.0, 18.0)
                    #     else:
                    #         _, _, teacher_log_prob_action, _ = self.teacher.act(state, return_log_prob=True)
                    # else:
                    #     teacher_log_prob_action = None

                    # observe reward and next obs
                    next_obs, reward, done, info = utl.env_step(
                        self.eval_env, action.squeeze(dim=0)
                    )

                    # add raw reward
                    running_reward += reward.item()
                    # clip reward if necessary for policy inputs
                    if self.reward_clip and self.env_type == "atari":
                        reward = torch.tanh(reward)

                    step += 1
                    done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

                    if self.env_type == "meta":
                        observations[task_idx, step, :] = ptu.get_numpy(
                            next_obs[0, :obs_size]
                        )

                    # set: obs <- next_obs
                    obs = next_obs.clone()
                    state = info["state"].clone()
                    # obs_list.append(obs)
                    # action_list.append(action)
                    # img_list.append(self.eval_env.env.special_render())
                    if (
                        self.env_type == "meta"
                        and "is_goal_state" in dir(self.eval_env.unwrapped)
                        and self.eval_env.unwrapped.is_goal_state()
                    ):
                        success_rate[task_idx] = 1.0  # ever once reach
                    elif (
                        self.env_type == "generalize"
                        and self.eval_env.unwrapped.is_success()
                    ):
                        success_rate[task_idx] = 1.0  # ever once reach
                    elif (
                        self.env_type == "gridworld"
                        and info["reached_goal"]
                    ):
                        success_rate[task_idx] = 1.0  # ever once reach
                    elif "success" in info and info["success"] == True:  # keytodoor
                        success_rate[task_idx] = 1.0

                    if done_rollout:
                        # for all env types, same
                        break
                    if self.env_type == "meta" and info["done_mdp"] == True:
                        # for early stopping meta episode like Ant-Dir
                        break

                returns_per_episode[task_idx, episode_idx] = running_reward
            total_steps[task_idx] = step
        self.agent.train()

        return returns_per_episode, success_rate, observations, total_steps

    def log_train_stats(self, train_stats):
        logger.record_step(self._n_env_steps_total)
        ## log losses
        for k, v in train_stats.items():
            logger.record_tabular("rl_loss/" + k, v)
        ## log gradient norms
        results = self.agent.report_grad_norm()
        for k, v in results.items():
            logger.record_tabular("rl_loss/" + k, v)
        logger.dump_tabular()

    def log(self):
        # --- log training  ---
        ## set env steps for tensorboard: z is for lowest order
        logger.record_step(self._n_env_steps_total)
        logger.record_tabular("z/env_steps", self._n_env_steps_total)
        logger.record_tabular("z/rollouts", self._n_rollouts_total)
        logger.record_tabular("z/rl_steps", self._n_rl_update_steps_total)

        # --- evaluation ----
        if self.env_type == "meta":
            if self.train_env.n_tasks is not None:
                (
                    returns_train,
                    success_rate_train,
                    observations,
                    total_steps_train,
                ) = self.evaluate(self.train_tasks[: len(self.eval_tasks)])
            (
                returns_eval,
                success_rate_eval,
                observations_eval,
                total_steps_eval,
            ) = self.evaluate(self.eval_tasks)
            if self.eval_stochastic:
                (
                    returns_eval_sto,
                    success_rate_eval_sto,
                    observations_eval_sto,
                    total_steps_eval_sto,
                ) = self.evaluate(self.eval_tasks, deterministic=False)

            if self.train_env.n_tasks is not None and "plot_behavior" in dir(
                self.eval_env.unwrapped
            ):
                # plot goal-reaching trajs
                for i, task in enumerate(
                    self.train_tasks[: min(5, len(self.eval_tasks))]
                ):
                    self.eval_env.reset(task=task)  # must have task argument
                    logger.add_figure(
                        "trajectory/train_task_{}".format(i),
                        utl_eval.plot_rollouts(observations[i, :], self.eval_env),
                    )

                for i, task in enumerate(
                    self.eval_tasks[: min(5, len(self.eval_tasks))]
                ):
                    self.eval_env.reset(task=task)
                    logger.add_figure(
                        "trajectory/eval_task_{}".format(i),
                        utl_eval.plot_rollouts(observations_eval[i, :], self.eval_env),
                    )
                    if self.eval_stochastic:
                        logger.add_figure(
                            "trajectory/eval_task_{}_sto".format(i),
                            utl_eval.plot_rollouts(
                                observations_eval_sto[i, :], self.eval_env
                            ),
                        )

            if "is_goal_state" in dir(
                self.eval_env.unwrapped
            ):  # goal-reaching success rates
                # some metrics
                logger.record_tabular(
                    "metrics/successes_in_buffer",
                    self._successes_in_buffer / self._n_env_steps_total,
                )
                if self.train_env.n_tasks is not None:
                    logger.record_tabular(
                        "metrics/success_rate_train", np.mean(success_rate_train)
                    )
                logger.record_tabular(
                    "metrics/success_rate_eval", np.mean(success_rate_eval)
                )
                if self.eval_stochastic:
                    logger.record_tabular(
                        "metrics/success_rate_eval_sto", np.mean(success_rate_eval_sto)
                    )

            for episode_idx in range(self.max_rollouts_per_task):
                if self.train_env.n_tasks is not None:
                    logger.record_tabular(
                        "metrics/return_train_episode_{}".format(episode_idx + 1),
                        np.mean(returns_train[:, episode_idx]),
                    )
                logger.record_tabular(
                    "metrics/return_eval_episode_{}".format(episode_idx + 1),
                    np.mean(returns_eval[:, episode_idx]),
                )
                if self.eval_stochastic:
                    logger.record_tabular(
                        "metrics/return_eval_episode_{}_sto".format(episode_idx + 1),
                        np.mean(returns_eval_sto[:, episode_idx]),
                    )

            if self.train_env.n_tasks is not None:
                logger.record_tabular(
                    "metrics/total_steps_train", np.mean(total_steps_train)
                )
                logger.record_tabular(
                    "metrics/return_train_total",
                    np.mean(np.sum(returns_train, axis=-1)),
                )
            logger.record_tabular("metrics/total_steps_eval", np.mean(total_steps_eval))
            logger.record_tabular(
                "metrics/return_eval_total", np.mean(np.sum(returns_eval, axis=-1))
            )
            if self.eval_stochastic:
                logger.record_tabular(
                    "metrics/total_steps_eval_sto", np.mean(total_steps_eval_sto)
                )
                logger.record_tabular(
                    "metrics/return_eval_total_sto",
                    np.mean(np.sum(returns_eval_sto, axis=-1)),
                )

        elif self.env_type == "generalize":
            returns_eval, success_rate_eval, total_steps_eval = {}, {}, {}
            for env, (env_name, eval_num_episodes_per_task) in self.eval_envs.items():
                self.eval_env = env  # assign eval_env, not train_env
                for suffix, deterministic in zip(["", "_sto"], [True, False]):
                    if deterministic == False and self.eval_stochastic == False:
                        continue
                    return_eval, success_eval, _, total_step_eval = self.evaluate(
                        eval_num_episodes_per_task * [None],
                        deterministic=deterministic,
                    )
                    returns_eval[
                        self.train_env_name + env_name + suffix
                    ] = return_eval.squeeze(-1)
                    success_rate_eval[
                        self.train_env_name + env_name + suffix
                    ] = success_eval
                    total_steps_eval[
                        self.train_env_name + env_name + suffix
                    ] = total_step_eval

            for k, v in returns_eval.items():
                logger.record_tabular(f"metrics/return_eval_{k}", np.mean(v))
            for k, v in success_rate_eval.items():
                logger.record_tabular(f"metrics/succ_eval_{k}", np.mean(v))
            for k, v in total_steps_eval.items():
                logger.record_tabular(f"metrics/total_steps_eval_{k}", np.mean(v))

        elif self.env_type == "rmdp":
            returns_eval, _, _, total_steps_eval = self.evaluate(self.eval_tasks)
            returns_eval = returns_eval.squeeze(-1)
            # np.quantile is introduced in np v1.15, so we have to use np.percentile
            cutoff = np.percentile(returns_eval, 100 * self.worst_percentile)
            worst_indices = np.where(
                returns_eval <= cutoff
            )  # must be "<=" to avoid empty set
            returns_eval_worst, total_steps_eval_worst = (
                returns_eval[worst_indices],
                total_steps_eval[worst_indices],
            )

            logger.record_tabular("metrics/return_eval_avg", returns_eval.mean())
            logger.record_tabular(
                "metrics/return_eval_worst", returns_eval_worst.mean()
            )
            logger.record_tabular(
                "metrics/total_steps_eval_avg", total_steps_eval.mean()
            )
            logger.record_tabular(
                "metrics/total_steps_eval_worst", total_steps_eval_worst.mean()
            )

        elif self.env_type in ["pomdp", "credit", "atari"]:
            returns_eval, success_rate_eval, _, total_steps_eval = self.evaluate(
                self.eval_tasks
            )
            if self.eval_stochastic:
                (
                    returns_eval_sto,
                    success_rate_eval_sto,
                    _,
                    total_steps_eval_sto,
                ) = self.evaluate(self.eval_tasks, deterministic=False)

            logger.record_tabular("metrics/total_steps_eval", np.mean(total_steps_eval))
            logger.record_tabular(
                "metrics/return_eval_total", np.mean(np.sum(returns_eval, axis=-1))
            )
            logger.record_tabular(
                "metrics/success_rate_eval", np.mean(success_rate_eval)
            )

            if self.eval_stochastic:
                logger.record_tabular(
                    "metrics/total_steps_eval_sto", np.mean(total_steps_eval_sto)
                )
                logger.record_tabular(
                    "metrics/return_eval_total_sto",
                    np.mean(np.sum(returns_eval_sto, axis=-1)),
                )
                logger.record_tabular(
                    "metrics/success_rate_eval_sto", np.mean(success_rate_eval_sto)
                )

        elif self.env_type == "gridworld":
            returns_eval, success_rate_eval, _, total_steps_eval = self.evaluate(self.eval_tasks, deterministic=True)
            returns_eval = returns_eval.squeeze(-1)
            # np.quantile is introduced in np v1.15, so we have to use np.percentile
            # cutoff = np.percentile(returns_eval, 100 * self.worst_percentile)
            # worst_indices = np.where(
            #     returns_eval <= cutoff
            # )  # must be "<=" to avoid empty set
            # returns_eval_worst, total_steps_eval_worst = (
            #     returns_eval[worst_indices],
            #     total_steps_eval[worst_indices],
            # )

            logger.record_tabular("metrics/return_eval_avg", returns_eval.mean())
            # logger.record_tabular(
            #     "metrics/return_eval_worst", returns_eval_worst.mean()
            # )
            logger.record_tabular(
                "metrics/total_steps_eval_avg", total_steps_eval.mean()
            )
            logger.record_tabular(
                "metrics/success_rate_eval", np.mean(success_rate_eval)
            )
            # logger.record_tabular(
            #     "metrics/total_steps_eval_worst", total_steps_eval_worst.mean()
            # )

        else:
            raise ValueError

        logger.record_tabular("z/time_cost", int(time.time() - self._start_time))
        logger.record_tabular(
            "z/fps",
            (self._n_env_steps_total - self._n_env_steps_total_last)
            / (time.time() - self._start_time_last),
        )
        self._n_env_steps_total_last = self._n_env_steps_total
        self._start_time_last = time.time()

        logger.dump_tabular()

        if self.env_type == "generalize":
            return sum([v.mean() for v in success_rate_eval.values()]) / len(
                success_rate_eval
            )
        else:
            return np.mean(np.sum(returns_eval, axis=-1))

    def save_model(self, iter, perf):
        save_path = os.path.join(
            logger.get_dir(), "save", f"agent_{iter}_perf{perf:.3f}.pt"
        )
        torch.save(self.agent.state_dict(), save_path)

    def load_model(self, ckpt_path):
        self.agent.load_state_dict(torch.load(ckpt_path, map_location=ptu.device))
        print("load successfully from", ckpt_path)
