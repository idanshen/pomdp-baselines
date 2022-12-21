from __future__ import annotations

from envs.gridworld.minigrid.gym_minigrid.minigrid import *
from envs.gridworld.minigrid.gym_minigrid.register import register


class MemoryEnv(MiniGridEnv):

    """
    ## Description
    This environment is a memory test. The agent starts in a small room where it
    sees an object. It then has to go through a narrow hallway which ends in a
    split. At each end of the split there is an object, one of which is the same
    as the object in the starting room. The agent has to remember the initial
    object, and go to the matching object at split.
    ## Mission Space
    "go to the matching object at the end of the hallway"
    ## Action Space
    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |
    ## Observation Encoding
    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked
    ## Rewards
    A reward of '1' is given for success, and '0' for failure.
    ## Termination
    The episode ends if any one of the following conditions is met:
    1. The agent reaches the correct matching object.
    2. The agent reaches the wrong matching object.
    3. Timeout (see `max_steps`).
    ## Registered Configurations
    S: size of map SxS.
    - `MiniGrid-MemoryS17Random-v0`
    - `MiniGrid-MemoryS13Random-v0`
    - `MiniGrid-MemoryS13-v0`
    - `MiniGrid-MemoryS11-v0`
    """

    def __init__(
        self, size=8, random_length=False, max_steps: int | None = None, seed=None, **kwargs
    ):
        self.size = size
        self.random_length = random_length

        if max_steps is None:
            max_steps = size**2

        super().__init__(
            width=size,
            height=size,
            # Set this to True for maximum speed
            see_through_walls=False,
            max_steps=max_steps,
            seed=seed,
            agent_view_size=7,
            actions_type="forward_and_turns",
            **kwargs,
        )
        self.tile_size = 6
        _obs = self.reset()
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(_obs), ),
            dtype='uint8'
        )

        # Set up the renderer.
        self.render_type = 'observe'

    def _gen_grid(self, width, height, _seed=None):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        assert height % 2 == 1
        upper_room_wall = height // 2 - 2
        lower_room_wall = height // 2 + 2
        if self.random_length:
            hallway_end = self._rand_int(4, width - 2)
        else:
            hallway_end = width - 3

        # Start room
        for i in range(1, 5):
            self.grid.set(i, upper_room_wall, Wall())
            self.grid.set(i, lower_room_wall, Wall())
        self.grid.set(4, upper_room_wall + 1, Wall())
        self.grid.set(4, lower_room_wall - 1, Wall())

        # Horizontal hallway
        for i in range(5, hallway_end):
            self.grid.set(i, upper_room_wall + 1, Wall())
            self.grid.set(i, lower_room_wall - 1, Wall())

        # Vertical hallway
        for j in range(0, height):
            if j != height // 2:
                self.grid.set(hallway_end, j, Wall())
            self.grid.set(hallway_end + 2, j, Wall())

        # Fix the player's start position and orientation
        self.agent_pos = np.array((self._rand_int(1, hallway_end + 1), height // 2))
        self.agent_dir = 0

        # Place objects
        start_room_obj = self._rand_elem([Key, Ball])
        self.grid.set(1, height // 2 - 1, start_room_obj("green"))

        other_objs = self._rand_elem([[Ball, Key], [Key, Ball]])
        pos0 = (hallway_end + 1, height // 2 - 2)
        pos1 = (hallway_end + 1, height // 2 + 2)
        self.grid.set(*pos0, other_objs[0]("green"))
        self.grid.set(*pos1, other_objs[1]("green"))

        # Choose the target objects
        if start_room_obj == other_objs[0]:
            self.success_pos = (pos0[0], pos0[1] + 1)
            self.failure_pos = (pos1[0], pos1[1] - 1)
        else:
            self.success_pos = (pos1[0], pos1[1] - 1)
            self.failure_pos = (pos0[0], pos0[1] + 1)

        self.mission = "go to the matching object at the end of the hallway"

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if tuple(self.agent_pos) == self.success_pos:
            reward = self._reward()
            done = True
        if tuple(self.agent_pos) == self.failure_pos:
            reward = 0
            done = True

        return obs, reward, done, info

    def render(self, mode='human', _key=None):

        _key_to_render = _key if _key is not None else self.render_type

        # Add a partial aerial rendering.
        # Note that partial rendering is also designed such that the agent is always obs-up (as opposed to north up).
        if 'partial_view_rendering' == _key_to_render:
            # This is legacy code and should never be hit.
            raise NotImplementedError

        # Show it a compact rendering (state without the location of the holes)
        elif _key_to_render == 'partial_state':  #TODO: broken. do not use
            obs_dict = self.gen_obs_encode()
            view = obs_dict["image"][:, :, 0]
            pos_in_view = {0: (self.agent_view_size//2, 0),
                           1: (0, self.agent_view_size//2),
                           2: (self.agent_view_size//2, self.agent_view_size-1),
                           3: (self.agent_view_size-1, self.agent_view_size//2),
                           }
            agent_loc = pos_in_view[obs_dict['direction']]
            view[agent_loc] = 10
            return view.reshape(-1) / 10.0

        # Add a partial aerial rendering.
        # Note that partial rendering is also designed such that the agent is always obs-up (as opposed to north up).
        elif 'observe' == _key_to_render:
            try:
                obs_dict = self.gen_obs_encode()
                observe = self.get_obs_render(obs=obs_dict["image"], tile_size=self.tile_size)
                observe = observe.astype(np.double) / 255.0   # Need to make obsevrations in the range [0, 1.]
                if self.full_rendering_obs_noise > 0:
                    observe = np.clip(observe + np.random.normal(0, self.full_rendering_obs_noise, observe.shape), a_min=0.0, a_max=1.0)

            except Exception as err:
                observe = None

            return observe

        # Add a full aerial rendering.
        elif 'full_observe' == _key_to_render:
            try:
                full_observe = self.special_render(mode='DONTDISPLAY', highlight=False, tile_size=self.tile_size, render_lava=True)  # tilesize sets the fidelity of the rendering.
                full_observe = full_observe.astype(np.double) / 255.0   # Need to make obsevrations in the range [0, 1.]
                if self.full_rendering_obs_noise > 0:
                    full_observe = np.clip(full_observe + np.random.normal(0, self.full_rendering_obs_noise, full_observe.shape), a_min=0.0, a_max=1.0)
            except:
                full_observe = None

            return full_observe

        else:
            raise NotImplementedError  # Type of rendering not recognised.


class MemoryS11(MemoryEnv):
    def __init__(self):
        super().__init__(size=11, random_length=False)


register(
    id='MiniGrid-MemoryS11-v0',
    entry_point='envs.gridworld.minigrid.gym_minigrid.envs:MemoryS11'
)
