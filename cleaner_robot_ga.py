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
                 population_size: int = 1):
        super().__init__(bounding_margin_width_factor)
        self.num_landmarks = num_landmarks
        self.k = k
        self.population_size = population_size

        self.nearest_neighbors = NearestNeighbors_sklearn()

        self.metric = Metric_Euclidean
        self.sampler = Sampler_Uniform()

        self.roadmap = None
        self.roadmaps: Dict[Robot, nx.Graph] = {}
        self.collision_detection = {}
        self.start = None
        self.end = None
        self.population = None

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

    def get_random_individual(self) -> Dict[Robot, list]:
        individual = {}
        for robot in self.scene.robots:
            robot_roadmap = self.roadmaps[robot]
            random_point = random.choice(list(robot_roadmap.nodes()))
            while not nx.algorithms.has_path(robot_roadmap, robot.start, random_point) or not nx.algorithms.has_path(
                    robot_roadmap, random_point, robot.end):
                random_point = random.choice(list(robot_roadmap.nodes()))
            print(random_point, file=self.writer)
            path_start = nx.algorithms.shortest_path(robot_roadmap, robot.start, random_point, weight='weight')
            path_end = nx.algorithms.shortest_path(robot_roadmap, random_point, robot.end, weight='weight')
            individual[robot] = list(path_start)[:-1] + list(path_end)
        return individual

    def get_initial_population(self) -> list[Dict[Robot, list]]:
        population = []
        for i in range(self.population_size):
            population.append(self.get_random_individual())
        return population

    def new_sample_free(self, robot: Robot):
        """
        Sample a free random point
        """
        sample = self.sampler.sample()
        while not self.collision_detection[robot].is_point_valid(sample):
            sample = self.sampler.sample()
        return sample

    def load_scene(self, scene: Scene):
        super().load_scene(scene)
        self.sampler.set_scene(scene, self._bounding_box)

        # Build collision detection for each robot.
        for robot in scene.robots:
            self.collision_detection[robot] = collision_detection.ObjectCollisionDetection(scene.obstacles, robot)

        # Build roadmap for each robot.
        for robot in scene.robots:
            self.roadmaps[robot] = nx.Graph()
            # Add points to robot's roadmap
            self.roadmaps[robot].add_node(robot.start)
            self.roadmaps[robot].add_node(robot.end)
            for i in range(self.num_landmarks):
                p_rand = self.new_sample_free(robot)
                self.roadmaps[robot].add_node(p_rand)
                if i % 100 == 0 and self.verbose:
                    print('added', i, 'landmarks in PRM', file=self.writer)
            # TODO if not nx.algorithms.has_path(self.roadmap, self.start, self.end)... (either add more points or return)

            self.nearest_neighbors.fit(list(self.roadmaps[robot].nodes))

            # Connect all points to their k nearest neighbors
            for cnt, point in enumerate(self.roadmaps[robot].nodes):
                neighbors = self.nearest_neighbors.k_nearest(point, self.k + 1)
                for neighbor in neighbors:
                    if self.collision_free(neighbor, point, robot):
                        self.roadmaps[robot].add_edge(point, neighbor, weight=self.metric.dist(point, neighbor).to_double())

                if cnt % 100 == 0 and self.verbose:
                    print('connected', cnt, 'landmarks to their nearest neighbors', file=self.writer)

        self.population = self.get_initial_population()[0]

        # Generate initial population - for each robot choose a random point and find shortest path...

        # Generate Initial population.

        # Evolution loop:
        #   Compute fitness.
        #   Create next Generation:
        #      Reproduction
        #      Crossover + Mutation

        # Select best individual

        pass

    def solve(self):
        """
        Based on the start and end locations of each robot, solve the scene
        (i.e. return paths for all the robots)

        :return: path collection of motion planning
        :rtype: :class:`~discopygal.solvers.PathCollection`
        """
        # TODO check what should be the types of the solution - try to create a solution manually, return it without
        #   adding it to the roadmap and check if we see it in the solver viewer. Then check what is the difference
        # if not nx.algorithms.has_path(self.roadmap, self.start, self.end):
        #     if self.verbose:
        #         print('no path found...', file=self.writer)
        #     return PathCollection()
        #
        # # Convert from a sequence of Point_d points to PathCollection
        # tensor_path = nx.algorithms.shortest_path(self.roadmap, self.start, self.end, weight='weight')
        # path_collection = PathCollection()
        # for i, robot in enumerate(self.scene.robots):
        #     points = []
        #     for point in tensor_path:
        #         points.append(PathPoint(Point_2(point[2 * i], point[2 * i + 1])))
        #     path = Path(points)
        #     path_collection.add_robot_path(robot, path)
        #
        # if self.verbose:
        #     print('successfully found a path...', file=self.writer)
        #

        path_collection = PathCollection()
        for robot, robot_path in self.population.items():
            path_collection.add_robot_path(robot, Path([PathPoint(point) for point in robot_path]))
        print(path_collection)
        #


        # path_collection = PathCollection()
        # tensor_path = [[-10, 0, 10, 0], [0, 2, 0, 2], [10, 0, -10, 0]]
        # for i, robot in enumerate(self.scene.robots):
        #     points = []
        #     for point in tensor_path:
        #         points.append(PathPoint(Point_2(point[2 * i], point[2 * i + 1])))
        #     path = Path(points)
        #     path_collection.add_robot_path(robot, path)

        return path_collection
