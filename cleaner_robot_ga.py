import networkx as nx
import matplotlib.pyplot as plt

from discopygal.solvers import Robot, RobotDisc, RobotPolygon, RobotRod
from discopygal.solvers import Obstacle, ObstacleDisc, ObstaclePolygon, Scene
from discopygal.solvers import PathPoint, Path, PathCollection

from discopygal.solvers.samplers import Sampler, Sampler_Uniform
from discopygal.solvers.metrics import Metric, Metric_Euclidean
from discopygal.solvers.nearest_neighbors import NearestNeighbors, NearestNeighbors_sklearn
from discopygal.bindings import *
from discopygal.geometry_utils import collision_detection, conversions
from discopygal.solvers.Solver import Solver
from typing import Dict
import random
from dataclasses import dataclass
from utils import *


@dataclass
class RobotPath:
    def __init__(self, robot: Robot, path: list[Point_2]):
        self.robot: Robot = robot
        self.path: list[Point_2] = path

    def get_path_length(self) -> float:
        return get_point2_list_length(self.path)

    def get_path_cells(self, cell_size: float) -> set[tuple[int, int]]:
        cells = set()
        for point in self.path:
            cells.add(get_cell_indices(point, cell_size))
        return cells

    def print_summary(self, cell_size: float, writer, robot_idx: int):
        print(
            f"Robot: {robot_idx}"
            f"\n\tPath length: {self.get_path_length()}"
            f"\n\tPath cells: {len(self.get_path_cells(cell_size))}",
            file=writer)


class CleanerRobotGA(Solver):
    """
    The basic implementation of a Probabilistic Road Map (PRM) solver.
    Supports multi-robot motion planning, though might be inefficient for more than
    two-three robots.

    :param num_landmarks: number of landmarks to sample
    :type num_landmarks: :class:`int`
    :param k: number of nearest neighbors to connect
    :type k: :class:`int`

    """

    def __init__(self, num_landmarks, k, bounding_margin_width_factor=Solver.DEFAULT_BOUNDS_MARGIN_FACTOR,
                 population_size: int = 10, evolution_steps: int = 100,
                 length_weight: float = 1.0, num_cells_weight: float = 1.0,
                 cell_size: float = 1.0):
        super().__init__(bounding_margin_width_factor)
        self.num_landmarks = num_landmarks
        self.k = k
        self.population_size = population_size
        self.evolution_steps = evolution_steps
        self.length_weight = length_weight
        self.num_cells_weight = num_cells_weight
        self.cell_size = cell_size

        self.nearest_neighbors = NearestNeighbors_sklearn()

        self.metric = Metric_Euclidean
        self.sampler = Sampler_Uniform()

        self.roadmap = None
        self.roadmaps: Dict[Robot, nx.Graph] = {}
        self.collision_detection = {}
        self.start = None
        self.end = None
        self.population: list[list[RobotPath]] = []

    @staticmethod
    def get_arguments():
        """
        Return a list of arguments and their description, defaults and types.
        Can be used by a GUI to generate fields dynamically.
        Should be overridded by solvers.

        :return: arguments dict
        :rtype: :class:`dict`
        """
        return {
            'num_landmarks': ('Number of Landmarks:', 1000, int),
            'k': ('K for nearest neighbors:', 15, int),
            'bounding_margin_width_factor': (
                'Margin width factor (for bounding box):', Solver.DEFAULT_BOUNDS_MARGIN_FACTOR, FT),
        }

    @staticmethod
    def from_arguments(d):
        """
        Get a dictionary of arguments and return a solver.
        Should be overridded by solvers.

        :param d: arguments dict
        :type d: :class:`dict`
        """
        return CleanerRobotGA(d['num_landmarks'], d['k'], FT(d['bounding_margin_width_factor']))

    def get_graph(self):
        """
        Return a graph (if applicable).
        Can be overridded by solvers.

        :return: graph whose vertices are Point_2 or Point_d
        :rtype: :class:`networkx.Graph` or :class:`None`
        """
        return self.roadmap

    def collision_free(self, p1: Point_2, p2: Point_2, robot: Robot):
        """
        Get two points in the configuration space and decide if they can be connected
        """

        # Check validity of each edge seperately
        edge = Segment_2(p1, p2)
        if not self.collision_detection[robot].is_edge_valid(edge):
            return False
        return True

    def to_path(self):
        pass
        # path = Path([PathPoint(point) for point in path_start[:-1]] + [PathPoint(point) for point in path_end])

    def get_random_robots_paths(self) -> list[RobotPath]:
        # TODO: Consider starting with the shortest path for each robot without the middle point.
        robots_paths = []
        for robot in self.scene.robots:
            robot_roadmap = self.roadmaps[robot]
            random_point = random.choice(list(robot_roadmap.nodes()))
            while not nx.algorithms.has_path(robot_roadmap, robot.start, random_point) or not nx.algorithms.has_path(
                    robot_roadmap, random_point, robot.end):
                random_point = random.choice(list(robot_roadmap.nodes()))
            path_start = nx.algorithms.shortest_path(robot_roadmap, robot.start, random_point, weight='weight')
            path_end = nx.algorithms.shortest_path(robot_roadmap, random_point, robot.end, weight='weight')
            robots_paths.append(RobotPath(robot=robot, path=list(path_start)[:-1] + list(path_end)))
        return robots_paths

    def get_initial_population(self) -> list[list[RobotPath]]:
        population = []
        for i in range(self.population_size):
            population.append(self.get_random_robots_paths())
        return population

    def get_fitness(self, robots_paths: list[RobotPath]) -> tuple[float, float]:
        total_length = 0.0
        cells = set()
        for robot in robots_paths:
            total_length += robot.get_path_length()
            cells.update(robot.get_path_cells(self.cell_size))
        return len(cells), -total_length

    def new_sample_free(self, robot: Robot):
        """
        Sample a free random point
        """
        sample = self.sampler.sample()
        while not self.collision_detection[robot].is_point_valid(sample):
            sample = self.sampler.sample()
        return sample

    def create_robot_roadmap(self, robot: Robot):
        robot_roadmap = nx.Graph()

        # Add points to robot's roadmap
        robot_roadmap.add_node(robot.start)
        robot_roadmap.add_node(robot.end)
        for i in range(self.num_landmarks):
            p_rand = self.new_sample_free(robot)
            robot_roadmap.add_node(p_rand)

        self.nearest_neighbors.fit(list(robot_roadmap.nodes))

        # Connect all points to their k nearest neighbors
        for cnt, point in enumerate(robot_roadmap.nodes):
            neighbors = self.nearest_neighbors.k_nearest(point, self.k + 1)
            for neighbor in neighbors:
                if self.collision_free(neighbor, point, robot):
                    robot_roadmap.add_edge(point, neighbor, weight=self.metric.dist(point, neighbor).to_double())

            if cnt % 100 == 0 and self.verbose:
                print('connected', cnt, 'landmarks to their nearest neighbors', file=self.writer)
        assert nx.algorithms.has_path(robot_roadmap, robot.start, robot.end)
        return robot_roadmap

    def load_scene(self, scene: Scene):
        super().load_scene(scene)
        self.sampler.set_scene(scene, self._bounding_box)

        # Build collision detection and roadmap for each robot.
        for robot in scene.robots:
            self.collision_detection[robot] = collision_detection.ObjectCollisionDetection(scene.obstacles, robot)
            self.roadmaps[robot] = self.create_robot_roadmap(robot)

        self.population = self.get_initial_population()

        # TODO: Evolution loop:
        #   Compute fitness.
        #   Create next Generation:
        #      Reproduction
        #      Crossover + Mutation

        pass

    def solve(self):
        """
        Based on the start and end locations of each robot, solve the scene
        (i.e. return paths for all the robots)

        :return: path collection of motion planning
        :rtype: :class:`~discopygal.solvers.PathCollection`
        """
        path_collection = PathCollection()
        fittest_robot_paths = max(self.population, key=lambda robot_paths: self.get_fitness(robot_paths))
        for i, robot_path in enumerate(fittest_robot_paths):
            robot_path.print_summary(self.cell_size, self.writer, i)
            path_collection.add_robot_path(robot_path.robot, Path([PathPoint(point) for point in robot_path.path]))

        return path_collection
