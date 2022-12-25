from __future__ import annotations

from typing import Tuple, Dict, Any, Optional

from envs.gridworld.minigrid.gym_minigrid.minigrid import *
from envs.gridworld.minigrid.gym_minigrid.register import register
import networkx as nx

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
    _ACTION_NAMES: Tuple[str, ...] = ("left", "right", "forward")
    _XY_DIFF_TO_AGENT_DIR = {
        tuple(vec): dir_ind for dir_ind, vec in enumerate(DIR_TO_VEC)
    }
    _NEIGHBOR_OFFSETS = tuple(
        [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1),]
    )

    def __init__(
        self, size=8, random_length=False, max_steps: int | None = None, seed=None, **kwargs
    ):
        self._graph: Optional[nx.DiGraph] = None
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
        # self.compact_obs = False

    def _gen_grid(self, width, height, _seed=None):
        self._graph = None
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
        self.grid.set(1, height // 2 - 1, start_room_obj(self._object_to_color(start_room_obj)))

        other_objs = self._rand_elem([[Ball, Key], [Key, Ball]])
        pos0 = (hallway_end + 1, height // 2 - 2)
        pos1 = (hallway_end + 1, height // 2 + 2)
        self.grid.set(*pos0, other_objs[0](self._object_to_color(other_objs[0])))
        self.grid.set(*pos1, other_objs[1](self._object_to_color(other_objs[1])))

        # Choose the target objects
        if start_room_obj == other_objs[0]:
            self.success_pos = (pos0[0], pos0[1] + 1)
            self.failure_pos = (pos1[0], pos1[1] - 1)
        else:
            self.success_pos = (pos1[0], pos1[1] - 1)
            self.failure_pos = (pos0[0], pos0[1] + 1)

        self.mission = "go to the matching object at the end of the hallway"

    def _object_to_color(self, object):
        if object is Key:
            return "blue"
        elif object is Ball:
            return "green"
        else:
            raise ValueError

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if tuple(self.agent_pos) == self.success_pos:
            reward = self._reward()
            done = True
            info['reached_goal'] = True
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
            # encoding = self.grid.encode()
            # encoding[self.agent_pos[0], self.agent_pos[1], :] = OBJECT_TO_IDX['agent']
            # return encoding / 10.0
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

    def _add_node_to_graph(
        self,
        graph: nx.DiGraph,
        s: Tuple[int, int, int],
        valid_node_types: Tuple[str, ...],
        attr_dict: Dict[Any, Any] = None,
        include_rotation_free_leaves: bool = False,
    ):
        if s in graph:
            return
        if attr_dict is None:
            print("adding a node with neighbor checks and no attributes")
        graph.add_node(s, **attr_dict)

        if include_rotation_free_leaves:
            rot_free_leaf = (*s[:-1], None)
            if rot_free_leaf not in graph:
                graph.add_node(rot_free_leaf)
            graph.add_edge(s, rot_free_leaf, action="NA")

        if attr_dict["type"] in valid_node_types:
            for o in self.possible_neighbor_offsets():
                t = (s[0] + o[0], s[1] + o[1], (s[2] + o[2]) % 4)
                if t in graph and graph.nodes[t]["type"] in valid_node_types:
                    self._add_from_to_edge(graph, s, t)
                    self._add_from_to_edge(graph, t, s)

    def generate_graph(self,) -> nx.DiGraph:
        """The generated graph is based on the fully observable grid (as the
        expert sees it all).
        env: environment to generate the graph over
        """

        image = self.grid.encode()
        width, height, _ = image.shape
        graph = nx.DiGraph()

        # In fully observable grid, there shouldn't be any "unseen"
        # Currently dealing with "empty", "wall", "goal", "lava"

        valid_object_ids = np.sort(
            [OBJECT_TO_IDX[o] for o in ["empty", "wall", "ball", "key"]]
        )

        assert np.all(np.union1d(image[:, :, 0], valid_object_ids) == valid_object_ids)

        # Grid to nodes
        for x in range(width):
            for y in range(height):
                for rotation in range(4):
                    type, color, state = image[x, y]
                    self._add_node_to_graph(
                        graph,
                        (x, y, rotation),
                        attr_dict={
                            "type": IDX_TO_OBJECT[type],
                            "color": color,
                            "state": state,
                        },
                        valid_node_types=("empty", "ball", "key"),
                    )
                    if x==self.success_pos[0] and y==self.success_pos[1]:
                        if not graph.has_node("unified_goal"):
                            graph.add_node("unified_goal")
                        graph.add_edge((x, y, rotation), "unified_goal")

        return graph

    @classmethod
    def _add_from_to_edge(
        cls, g: nx.DiGraph, s: Tuple[int, int, int], t: Tuple[int, int, int],
    ):
        """Adds nodes and corresponding edges to existing nodes.
        This approach avoids adding the same edge multiple times.
        Pre-requisite knowledge about MiniGrid:
        DIR_TO_VEC = [
            # Pointing right (positive X)
            np.array((1, 0)),
            # Down (positive Y)
            np.array((0, 1)),
            # Pointing left (negative X)
            np.array((-1, 0)),
            # Up (negative Y)
            np.array((0, -1)),
        ]
        or
        AGENT_DIR_TO_STR = {
            0: '>',
            1: 'V',
            2: '<',
            3: '^'
        }
        This also implies turning right (clockwise) means:
            agent_dir += 1
        """

        s_x, s_y, s_rot = s
        t_x, t_y, t_rot = t

        x_diff = t_x - s_x
        y_diff = t_y - s_y
        angle_diff = (t_rot - s_rot) % 4

        # If source and target differ by more than one action, continue
        if (x_diff != 0) + (y_diff != 0) + (angle_diff != 0) != 1 or angle_diff == 2:
            return

        action = None
        if angle_diff == 1:
            action = "right"
        elif angle_diff == 3:
            action = "left"
        elif cls._XY_DIFF_TO_AGENT_DIR[(x_diff, y_diff)] == s_rot:
            # if translation is the same direction as source
            # orientation, then it's a valid forward action
            action = "forward"
        else:
            # This is when the source and target aren't one action
            # apart, despite having dx=1 or dy=1
            pass

        if action is not None:
            g.add_edge(s, t, action=action)

    @property
    def graph_created(self):
        return self._graph is not None

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self.generate_graph()
        return self._graph

    @graph.setter
    def graph(self, graph: nx.DiGraph):
        self._graph = graph

    @classmethod
    def possible_neighbor_offsets(cls) -> Tuple[Tuple[int, int], ...]:
        # Tuples of format:
        # (X translation, Y translation, rotation by 90 degrees)
        # A constant is returned, this function can be changed if anything
        # more complex needs to be done.

        # offsets_superset = itertools.product(
        #     [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]
        # )
        #
        # valid_offsets = []
        # for off in offsets_superset:
        #     if (int(off[0] != 0) + int(off[1] != 0) + int(off[2] != 0)) == 1:
        #         valid_offsets.append(off       #
        # return tuple(valid_offsets)

        return cls._NEIGHBOR_OFFSETS

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._ACTION_NAMES

    def query_expert(self, **kwargs) -> Tuple[int, bool]:
        paths = []
        agent_x, agent_y = self.agent_pos
        agent_rot = self.agent_dir
        source_state_key = (agent_x, agent_y, agent_rot)
        assert source_state_key in self.graph

        paths.append(nx.shortest_path(self.graph, source_state_key, "unified_goal"))

        if len(paths) == 0:
            return -1, False

        shortest_path_ind = int(np.argmin([len(p) for p in paths]))

        # if self.closest_agent_has_been_to_goal is None:
        #     self.closest_agent_has_been_to_goal = len(paths[shortest_path_ind]) - 1
        # else:
        #     self.closest_agent_has_been_to_goal = min(
        #         len(paths[shortest_path_ind]) - 1, self.closest_agent_has_been_to_goal
        #     )

        if len(paths[shortest_path_ind]) == 2:
            # Since "unified_goal" is 1 step away from actual goals
            # if a path like [actual_goal, unified_goal] exists, then
            # you are already at a goal.
            print(
                "Shortest path computations suggest we are at"
                " the target but episode does not think so."
            )
            return -1, False

        next_key_on_shortest_path = paths[shortest_path_ind][1]
        return (
            self.class_action_names().index(
                self.graph.get_edge_data(source_state_key, next_key_on_shortest_path)[
                    "action"
                ]
            ),
            True,
        )


class MemoryS11(MemoryEnv):
    def __init__(self):
        super().__init__(size=11, random_length=False)


register(
    id='MiniGrid-MemoryS11-v0',
    entry_point='envs.gridworld.minigrid.gym_minigrid.envs:MemoryS11'
)
