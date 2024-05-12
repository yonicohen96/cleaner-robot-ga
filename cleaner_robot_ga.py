import networkx as nx
import matplotlib.pyplot as plt
from scipy import special
from discopygal.solvers import Robot, RobotDisc, RobotPolygon, RobotRod
from discopygal.solvers import Obstacle, ObstacleDisc, ObstaclePolygon, Scene
from discopygal.solvers import PathPoint, Path, PathCollection

from discopygal.solvers.samplers import Sampler, Sampler_Uniform
from discopygal.solvers.metrics import Metric, Metric_Euclidean
from discopygal.solvers.nearest_neighbors import NearestNeighbors, NearestNeighbors_sklearn
from discopygal.bindings import *
from discopygal.geometry_utils import collision_detection, conversions
from discopygal.solvers.Solver import Solver
from typing import Dict, Any
import random
from dataclasses import dataclass
from utils import *


@dataclass
class RobotPath:
    def __init__(self, robot: Robot, path: list[Point_2], cell_size: float):
        self.robot: Robot = robot
        self.path: list[Point_2] = path
        self.path_length: float = get_point2_list_length(self.path)
        self.cells = set()
        for point in self.path:
            self.cells.add(get_cell_indices(point, cell_size))

    def print_summary(self, writer, robot_idx: int):
        print(
            f"Robot: {robot_idx}"
            f"\n\tPath length: {self.path_length}"
            f"\n\tPath cells: {len(self.cells)}",
            file=writer)


@dataclass
class FitnessValue:
    cells_num: int
    length: float

    def __lt__(self, other):
        return (self.cells_num, -self.length) < (other.cells_num, -other.length)


def get_fitness(robots_paths: list[RobotPath]) -> FitnessValue:
    total_length = 0.0
    cells = set()
    for robot_path in robots_paths:
        total_length += robot_path.path_length
        cells.update(robot_path.cells)
    return FitnessValue(len(cells), total_length)


def get_fitness_distribution(fitness_values: list[FitnessValue], cell_num_length_ratio: float):
    cell_num_logits = [fitness_value.cells_num for fitness_value in fitness_values]
    # The lower the length is, the highest probability it should get and therefore we use `-fitness_value.length`.
    length_logits = [-fitness_value.length for fitness_value in fitness_values]
    # TODO change softmax so simple normalization.
    cell_num_distribution = special.softmax(cell_num_logits)
    length_distribution = special.softmax(length_logits)
    return cell_num_length_ratio * cell_num_distribution + (1 - cell_num_length_ratio) * length_distribution


def get_highest_k_indices(values: list, k: int) -> list[int]:
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    return sorted_indices[:k]


# TODO: delete if we don't use it.
def merge(parent_0: list[RobotPath], parent_1: list[RobotPath], robot_path_idx: int, robot: Robot, cell_size: float,
          roadmaps: dict[Robot, nx.Graph]) -> RobotPath:
    parent_0_end_index = random.randint(0, len(parent_0[robot_path_idx].path) - 1)
    parent_1_start_index = random.randint(0, len(parent_1[robot_path_idx].path) - 1)
    parent_0_end_point = parent_0[robot_path_idx].path[parent_0_end_index]
    parent_1_start_point = parent_1[robot_path_idx].path[parent_1_start_index]

    if nx.algorithms.has_path(roadmaps[robot], parent_0_end_point,
                              parent_1_start_point):
        path_start = parent_0[robot_path_idx].path[:parent_0_end_index]
        path_middle = list(nx.algorithms.shortest_path(roadmaps[robot], parent_0_end_point,
                                                       parent_1_start_point, weight='weight'))
        path_end = parent_1[robot_path_idx].path[parent_1_start_index:]
        merged_robot_path = RobotPath(
            robot=robot,
            path=path_start + path_middle[:-1] + path_end,
            cell_size=cell_size)
        return merged_robot_path


def random_choices_no_repetitions(population: list[Any], weights: list[float] | np.ndarray, k: int) -> list[Any]:
    assert len(population) == len(weights)
    result = []
    population_indices = list(range(len(population)))
    for i in range(k):
        selected_item_idx = random.choices(population=population_indices, weights=weights, k=1)[0]
        result.append(population[selected_item_idx])
        weights[selected_item_idx] = 0
    return result


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
                 population_size: int = 10, evolution_steps: int = 20,
                 length_weight: float = 1.0, num_cells_weight: float = 1.0,
                 cell_size: float = 1.0, elite_proportion: float = 0.1,
                 cells_length_weights_ratio: float = 0.8,
                 mutation_rate: float = 0.3):
        assert population_size > 1
        super().__init__(bounding_margin_width_factor)
        self.num_landmarks = num_landmarks
        self.k = k
        self.population_size = population_size
        self.evolution_steps = evolution_steps
        self.length_weight = length_weight
        self.num_cells_weight = num_cells_weight
        self.cell_size = cell_size
        self.cells_length_weights_ratio = cells_length_weights_ratio
        self.mutation_rate = mutation_rate

        self.nearest_neighbors = NearestNeighbors_sklearn()

        self.metric = Metric_Euclidean
        self.sampler = Sampler_Uniform()

        self.roadmap = None
        self.roadmaps: Dict[Robot, nx.Graph] = {}
        self.collision_detection = {}
        self.start = None
        self.end = None
        self.population: list[list[RobotPath]] = []
        self.elite_size = int(elite_proportion * self.population_size)

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

    def crossover(self, fitness_distribution: np.ndarray, num_individuals) -> list[list[RobotPath]]:
        crossovers = []
        for child_idx in range(num_individuals):
            # Create the next child by merging two parents.
            child: list[RobotPath] = []
            parent_0, parent_1 = random_choices_no_repetitions(self.population, fitness_distribution.copy(), k=2)
            parents_for_crossover = [parent_0, parent_1]
            num_robots = len(parent_0)
            for robot_path_idx in range(num_robots):
                selected_parent = parents_for_crossover[random.randint(0, 1)]
                child.append(selected_parent[robot_path_idx])
            crossovers.append(child)
        return crossovers

    def mutate(self, crossovers: list[list[RobotPath]]) -> list[list[RobotPath]]:
        mutated_crossovers: list[list[RobotPath]] = []
        for robots_paths in crossovers:
            mutated_robots_paths: list[RobotPath] = []
            for robot_path in robots_paths:
                if random.random() <= self.mutation_rate:
                    mutated_robots_paths.append(self.add_point_to_robot_path(robot_path))
                else:
                    mutated_robots_paths.append(robot_path)
            mutated_crossovers.append(mutated_robots_paths)
        return mutated_crossovers



    def add_point_to_robot_path(self, robot_path: RobotPath) -> RobotPath:
        robot = robot_path.robot
        robot_roadmap = self.roadmaps[robot]
        orig_path = robot_path.path
        new_path = []
        while True:
            random_point = random.choice(list(robot_roadmap.nodes()))
            prev_node_idx = random.randint(0, len(orig_path) - 2)
            next_node_idx = random.randint(prev_node_idx + 1, len(orig_path) - 1)
            if not nx.algorithms.has_path(
                    robot_roadmap, orig_path[prev_node_idx], random_point) or not nx.algorithms.has_path(
                robot_roadmap, random_point, orig_path[next_node_idx]):
                continue
            path_to_random = list(nx.algorithms.shortest_path(robot_roadmap, orig_path[prev_node_idx], random_point, weight='weight'))
            path_from_random = list(nx.algorithms.shortest_path(robot_roadmap, random_point, orig_path[next_node_idx], weight='weight'))
            new_path = orig_path[:prev_node_idx] + path_to_random[:-1] + path_from_random[:-1] + orig_path[next_node_idx:]
            break
        return RobotPath(robot=robot, path=new_path, cell_size=self.cell_size)


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
            robots_paths.append(RobotPath(robot=robot, path=list(path_start)[:-1] + list(path_end),
                                          cell_size=self.cell_size))
        return robots_paths

    def get_initial_population(self) -> list[list[RobotPath]]:
        return [self.get_random_robots_paths() for _ in range(self.population_size)]


    def sample_free(self, robot: Robot):
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
            robot_roadmap.add_node(self.sample_free(robot))

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

        # Get random initial population.
        self.population = self.get_initial_population()

        # Evolution steps.
        for step in range(self.evolution_steps):
            print(f'evolution step {step}', file=self.writer)
            fitness_values = [get_fitness(robot_path) for robot_path in self.population]

            elite_population = [self.population[i] for i in get_highest_k_indices(fitness_values, self.elite_size)]

            fitness_distribution = get_fitness_distribution(fitness_values, self.cells_length_weights_ratio)
            crossover_population = self.crossover(fitness_distribution, self.population_size - len(elite_population))
            mutated_crossover_population = self.mutate(crossover_population)

            self.population = elite_population + mutated_crossover_population

    def solve(self):
        """
        Based on the start and end locations of each robot, solve the scene
        (i.e. return paths for all the robots)

        :return: path collection of motion planning
        :rtype: :class:`~discopygal.solvers.PathCollection`
        """
        path_collection = PathCollection()
        fittest_robot_paths = max(self.population, key=lambda robot_paths: get_fitness(robot_paths))
        for i, robot_path in enumerate(fittest_robot_paths):
            path_collection.add_robot_path(robot_path.robot, Path([PathPoint(point) for point in robot_path.path]))

        return path_collection
