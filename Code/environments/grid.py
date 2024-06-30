from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, Lava
from minigrid.minigrid_env import MiniGridEnv


class CustomEnv(MiniGridEnv):
    """
    ## Description

    This environment is an empty room, and the goal of the agent is to reach the
    green goal square, which provides a sparse reward. This environment is
    useful, with small rooms, to validate that your RL algorithm works
    correctly, and with large rooms to experiment with sparse rewards and
    exploration.

    ## Mission Space

    "get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |

    """

    def __init__(
        self,
        size=8,
        objects = {},
        map = None,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.walls = objects.get("walls", [])
        self.lavas = objects.get("lavas", [])
        self.map = map

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=MissionSpace(mission_func=self._gen_mission),
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        if self.map is not None:

            assert len(self.map) == height, f"Map height {len(self.map)} != grid height {height}"
            assert len(self.map[0]) == width, f"Map width {len(self.map[0])} != grid width {width}"

            self.walls = []
            self.lavas = []

            for i in range(len(self.map)):
                for j in range(len(self.map[i])):
                    if self.map[i][j] == "W" and self.grid.get(j, i) is None:
                        self.grid.set(j, i, Wall())
                        self.walls.append((j, i))
                    elif self.map[i][j] == "L" and self.grid.get(j, i) is None:
                        self.grid.set(j, i, Lava())
                        self.lavas.append((j, i))
                    elif self.map[i][j] == "G" and self.grid.get(j, i) is None:
                        self.put_obj(Goal(), j, i)
                    elif self.map[i][j] == "A" and self.grid.get(j, i) is None:
                        self.agent_start_pos = (j, i)

        else:

            # Generate the walls in the middle
            for x, y in self.walls:

                # Check if x,y is within the grid and is empty
                if x < width and y < height and self.grid.get(x, y) is None:
                    self.grid.set(x, y, Wall())
                else:
                    print("Wall position is out of bounds or is not empty")
            
            for x, y in self.lavas:
                
                # Check if x,y is within the grid and is empty
                if x < width and y < height and self.grid.get(x, y) is None:
                    self.grid.set(x, y, Lava())
                else:
                    print("Wall position is out of bounds or is not empty")

            # Place a goal square in the bottom-right corner
            self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
