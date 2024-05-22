"""
A solver class for the coverage path planning problem, that uses a genetic algorithm approach.
"""
import math
from dataclasses import dataclass
from typing import Dict

import networkx as nx
from discopygal.bindings import *
from discopygal.geometry_utils import collision_detection
from discopygal.solvers import PathPoint, Path
from discopygal.solvers import Robot
from discopygal.solvers import Scene
from discopygal.solvers.Solver import Solver
from discopygal.solvers.nearest_neighbors import NearestNeighbors_sklearn
from discopygal.solvers.samplers import Sampler_Uniform

from utils import *


@dataclass
class RobotPath:
    """
    A dataclass that represents a robot and a path, and the cells that contain the path's points.
    """

    def __init__(self, robot: Robot, path: list[Point_2], cell_size: float):
        self.robot: Robot = robot
        self.path: list[Point_2] = path
        self.cells = set()
        for point in self.path:
            self.cells.add(get_cell_indices(point, cell_size))


def get_fitness(robots_paths: list[RobotPath]) -> float:
    """
    Returns the fitness of a list of robots paths which is the number of cells visited by only a single robot.
    :param robots_paths: A list of RobotPath objects.
    :return: The fitness value for the given list of robots paths.
    """
    cells = dict()
    for robot_path in robots_paths:
        for cell in robot_path.cells:
            if cell in cells:
                cells[cell] = 0
            else:
                cells[cell] = 1
    return sum(cells.values())


class CoveragePathPlanner(Solver):
    """
    A solver class for the Coverage Path planning problem that uses a genetic algorithm approach.
    Supports multiple robots scenarios.

    :param num_landmarks: number of landmarks to sample
    :type num_landmarks: :class:`int`
    :param k: number of nearest neighbors to connect
    :type k: :class:`int`
    :param bounding_margin_width_factor: The margin width factor for the bounding box.
    :type bounding_margin_width_factor: :class:`int`
    :param population_size: The number of individuals (lists of robots paths) for the genetic algorithm.
    :type population_size: :class:`int`
    :param evolution_steps: The number of evolutions steps.
    :type evolution_steps: :class:`int`
    :param min_cell_size: The minimum cell size: the cell size decreases during evolution steps, and this value is the
     minimal cell size allowed.
    :type min_cell_size: :class:`float`
    :param cell_size_decrease_interval: This parameter determines the number of evolution steps with the same maximum
     fitness value after which the cell size decreases.
    :type cell_size_decrease_interval: :class:`int`
    :param final_steps_num: The minimum number of last steps for which the cell size is set to `min_cell_size`
    :type final_steps_num: :class:`int`
    :param random_point_initialization: Whether the initial population is randomly initialized. If True, each individual
     contains robots paths, each one is the shortest path from the start point to the end point of the robot, that
     passes through a random free point.
    :type random_point_initialization: :class:`int`
    :param elite_proportion: The portion of the population with the highest fitness values that is copied to the next
     generation.
    :type elite_proportion: :class:`float`
    :param crossover_merge: Whether to use a merging strategy in the crossover operator (1 to use this operator and 0 to
     not use it). This operator returns a new robot path by merging the two selected parents robot paths. The merged
     path consists of three parts: (1) The path of the first parent from the start point to a random point. (2) The
     shortest path from the latter random point to another random point in the second parent's path, and (3) the path
     of the second parent from the selected random point to the end point.
     If this operator is not used, then the path of the crossover is copied from one of the parents which is chosen
     randomly.
    :type crossover_merge: :class:`int`
    :param mutation_rate: The portion of the crossover individual on which the mutation operator is applied.
    :type mutation_rate: :class:`float`
    :param mutate_gauss: Whether to use a gaussian sampling strategy for mutation. If the value is 1, then the mutation
     is applied by choosing a random point for each robot path, and sampling a random point from a gaussian distribution
     (centered at the original point, with a standard deviation that is determined by another parameter) and replacing
     the original point with the new point (and connecting it to the previous and next points in the path by shortest
     paths).
    :type mutate_gauss: :class:`int`
    :param add_remove_mutation_ratio: The ratio between the number of times the mutation operator adds a new point to
     the path to the number of times the mutation operator selects two random points from the path and connect them by
     the shortest path, and by that aims to shorten the path.
    :type add_remove_mutation_ratio: :class:`float`
    :param mutation_std: The standard deviation for the gaussian mutation strategy.
    :type mutation_std: :class:`float`
    :param verbose: Whether to print the results.
    :type verbose: :class:`int`
    """

    def __init__(self,
                 num_landmarks=1000,
                 k=15,
                 bounding_margin_width_factor=Solver.DEFAULT_BOUNDS_MARGIN_FACTOR,
                 population_size: int = 10,
                 evolution_steps: int = 400,
                 min_cell_size: float = 2.0,
                 cell_size_decrease_interval: int = 5,
                 final_steps_num: int = 10,
                 random_point_initialization: int = 0,
                 elite_proportion: float = 0.2,
                 crossover_merge: int = 0,
                 mutation_rate: float = 0.5,
                 mutate_gauss: int = 1,
                 add_remove_mutation_ratio: float = 0.8,
                 mutation_std: float = 2,
                 verbose: int = 1,
                 print_prefix: str = ""):
        assert population_size > 1
        # Check that elite population is not the entire population.
        assert population_size > int(elite_proportion * population_size)
        super().__init__(bounding_margin_width_factor)

        # Roadmaps creation attributes
        self.nearest_neighbors: dict[Robot, NearestNeighbors_sklearn] = {}
        self.metric = Metric_Euclidean
        self.sampler = Sampler_Uniform()
        self.num_landmarks = num_landmarks
        self.k = k

        # Genetic algorithm attributes.
        self.population_size = population_size
        self.evolution_steps = evolution_steps
        self.min_cell_size = min_cell_size
        self.cell_size_decrease_interval = cell_size_decrease_interval
        self.final_steps_num = final_steps_num
        self.random_point_initialization = random_point_initialization
        self.elite_proportion = elite_proportion
        self.elite_size = int(elite_proportion * self.population_size)
        self.crossover_merge = crossover_merge
        self.mutation_rate = mutation_rate
        self.mutate_gauss = mutate_gauss
        self.add_remove_mutation_ratio = add_remove_mutation_ratio
        self.mutation_std = mutation_std

        # Datastructures initializations
        self.roadmap = None
        self.roadmaps: Dict[Robot, nx.Graph] = {}
        self.collision_detection = {}
        self.population: list[list[RobotPath]] = []
        self.cell_size = None
        self.free_cells: set[tuple[int, int]] = set()
        self.best_fitness_values: list[float] = []

        self.verbose = verbose
        self.print_prefix = print_prefix

    @staticmethod
    def get_arguments():
        """
        Return a list of arguments and their description, defaults and types.
        Can be used by a GUI to generate fields dynamically.
        Should be overridden by solvers.

        :return: arguments dict
        :rtype: :class:`dict`
        """
        return {
            'num_landmarks': ('Number of Landmarks:', 1000, int),
            'k': ('K for nearest neighbors:', 15, int),
            'bounding_margin_width_factor': (
                'Margin width factor (for bounding box):', Solver.DEFAULT_BOUNDS_MARGIN_FACTOR, FT),
            'population_size': ('population size:', 10, int),
            'evolution_steps': ('evolution steps:', 400, int),
            'min_cell_size': ('min cell size:', 2.0, float),
            'cell_size_decrease_interval': ('cell_size_decrease_interval', 5, int),
            'final_steps_num': ('final_steps_num', 10, int),
            'random_point_initialization': ('random_point_initialization', 0, int),
            'elite_proportion': ('elite proportion:', 0.2, float),
            'crossover_merge': ('crossover_merge', 0, int),
            'mutation_rate': ('mutation rate:', 0.5, float),
            'mutate_gauss': ('mutate_gauss', 1, int),
            'add_remove_mutation_ratio': ('add_remove_mutation_ratio', 0.8, float),
            'mutation_std': ('mutation_std', 2, float),
            'verbose': ('verbose:', 1, int),
        }

    @staticmethod
    def from_arguments(d):
        """
        Get a dictionary of arguments and return a solver.
        Should be overridden by solvers.

        :param d: arguments dict
        :type d: :class:`dict`
        """
        return CoveragePathPlanner(d['num_landmarks'],
                                   d['k'],
                                   FT(d['bounding_margin_width_factor']),
                                   d['population_size'],
                                   d['evolution_steps'],
                                   d['min_cell_size'],
                                   d['cell_size_decrease_interval'],
                                   d['final_steps_num'],
                                   d['random_point_initialization'],
                                   d['elite_proportion'],
                                   d['crossover_merge'],
                                   d['mutation_rate'],
                                   d['mutate_gauss'],
                                   d['add_remove_mutation_ratio'],
                                   d['mutation_std'],
                                   d['verbose'],
                                   )

    def collision_free(self, p1: Point_2, p2: Point_2, robot: Robot):
        """Gets two points in the configuration space and decide if they can be connected."""
        # Check validity of each edge separately
        edge = Segment_2(p1, p2)
        if not self.collision_detection[robot].is_edge_valid(edge):
            return False
        return True

    def merge(self, parent_0_robot_path: RobotPath, parent_1_robot_path: RobotPath, robot: Robot) -> RobotPath:
        """
        Given two robot paths, Returns a new robot path by merging the two robot paths. The merged path consists of
        three parts: (1) The path of the first parent from the start point to a random point. (2) The shortest path from
        the latter random point to another random point in the second parent's path, and (3) the path of the second
        parent from the selected random point to the end point.
        If fails to merge, returns the first parent.
        :param parent_0_robot_path: The first robot path to merge.
        :param parent_1_robot_path: The second robot path to merge.
        :param robot: The robot.
        :return: The merged path.
        """
        # Sample random points from parents' paths to connect.
        parent_0_end_index = random.randint(0, len(parent_0_robot_path.path) - 1)
        parent_1_start_index = random.randint(0, len(parent_1_robot_path.path) - 1)
        parent_0_end_point = parent_0_robot_path.path[parent_0_end_index]
        parent_1_start_point = parent_1_robot_path.path[parent_1_start_index]
        # Attempt to connect the ramdom point and return the merged robot path.
        roadmap = self.roadmaps[robot]
        if nx.algorithms.has_path(roadmap, parent_0_end_point,
                                  parent_1_start_point):
            path_start = parent_0_robot_path.path[:parent_0_end_index]
            path_middle = list(nx.algorithms.shortest_path(roadmap, parent_0_end_point,
                                                           parent_1_start_point, weight='weight'))
            path_end = parent_1_robot_path.path[parent_1_start_index:]
            merged_robot_path = RobotPath(
                robot=robot,
                path=path_start + path_middle[:-1] + path_end,
                cell_size=self.cell_size)
            return merged_robot_path
        # If fails to merge the paths, returns the first robot path without merging.
        return parent_0_robot_path

    def crossover_with_merge(self, fitness_distribution: np.ndarray, num_individuals: int) -> list[list[RobotPath]]:
        """
        Applies a merge crossover operator. For each new individual (a list of paths, one for each robot), two random
        parents from the previous population are selected. The merge operator is then applied to each pair of robot
        paths from the chosen parents, with each pair corresponding to the paths of the same robot.
        :param fitness_distribution: The distribution for sampling the parents.
        :param num_individuals: The number of individuals to create.
        :return: The crossover merged individuals
        """
        crossovers = []
        for child_idx in range(num_individuals):
            child: list[RobotPath] = []
            parent_0, parent_1 = random_choices_no_repetitions(self.population, fitness_distribution, k=2)
            num_robots = len(parent_0)
            for robot_path_idx in range(num_robots):
                robot = parent_0[robot_path_idx].robot
                merged_robot_path = self.merge(parent_0[robot_path_idx], parent_1[robot_path_idx], robot)
                child.append(merged_robot_path)
            crossovers.append(child)
        return crossovers

    def crossover_no_merge(self, fitness_distribution: np.ndarray, num_individuals: int) -> list[list[RobotPath]]:
        """
        Applies a crossover without merging paths. For each new individual (a list of robot paths), this operator
        randomly selects two parents. Then, for each robot, it copies the path from one of the two parents, chosen at
        random.
        :param fitness_distribution: The distribution for sampling the parents.
        :param num_individuals: The number of individuals to create.
        :return: The crossover individuals.
        """
        crossovers = []
        for child_idx in range(num_individuals):
            # Create the next child by merging two parents.
            child: list[RobotPath] = []
            parent_0, parent_1 = random_choices_no_repetitions(self.population, fitness_distribution, k=2)
            parents_for_crossover = [parent_0, parent_1]
            num_robots = len(parent_0)
            for robot_path_idx in range(num_robots):
                selected_parent = parents_for_crossover[random.randint(0, 1)]
                child.append(selected_parent[robot_path_idx])
            crossovers.append(child)
        return crossovers

    def crossover(self, fitness_distribution: np.ndarray, num_individuals: int) -> list[list[RobotPath]]:
        """
        Applies the crossover operator.
        :param fitness_distribution: The distribution for sampling the parents.
        :param num_individuals: The number of individuals to create.
        :return: The crossover individuals.
        """
        if self.crossover_merge:
            return self.crossover_with_merge(fitness_distribution, num_individuals)
        return self.crossover_no_merge(fitness_distribution, num_individuals)

    def add_point_to_robot_path(self, robot_path: RobotPath) -> RobotPath:
        """
        Samples a random point and connects it to two randomly selected points from robot's path with the shortest
        paths, and returns the new path.
        :param robot_path: the robot path to modify.
        :return: The modified robot path.
        """
        robot_path = self.ensure_middle_point_exists(robot_path)
        robot = robot_path.robot
        robot_roadmap = self.roadmaps[robot]
        orig_path = robot_path.path
        new_path = []
        while True:
            # Sample a random point to add to the path
            random_point = random.choice(list(robot_roadmap.nodes()))
            # Sample the points from the original path to connect to the new random point.
            prev_node_idx = random.randint(0, len(orig_path) - 2)
            next_node_idx = random.randint(prev_node_idx, len(orig_path) - 1)
            has_path_from_prev_to_random = nx.algorithms.has_path(robot_roadmap, orig_path[prev_node_idx], random_point)
            has_path_from_random_to_next = nx.algorithms.has_path(robot_roadmap, random_point, orig_path[next_node_idx])
            if not has_path_from_prev_to_random or not has_path_from_random_to_next:
                continue
            # Connect the points to create the new robot path.
            path_to_random = list(
                nx.algorithms.shortest_path(robot_roadmap, orig_path[prev_node_idx], random_point, weight='weight'))
            path_from_random = list(
                nx.algorithms.shortest_path(robot_roadmap, random_point, orig_path[next_node_idx], weight='weight'))
            new_path = orig_path[:prev_node_idx] + path_to_random[:-1] + path_from_random[:-1] + orig_path[
                                                                                                 next_node_idx:]
            break
        return RobotPath(robot=robot, path=new_path, cell_size=self.cell_size)

    def mutate_add_sample(self, individuals: list[list[RobotPath]]) -> list[list[RobotPath]]:
        """
        Applies a mutation operator that adds a random point to the robots path.
        :param individuals: The individuals to mutate.
        :return: The mutated individuals.
        """
        mutated_crossovers: list[list[RobotPath]] = []
        for robots_paths in individuals:
            mutated_robots_paths: list[RobotPath] = []
            for robot_path in robots_paths:
                if random.random() <= self.mutation_rate:
                    mutated_robots_paths.append(self.add_point_to_robot_path(robot_path))
                else:
                    mutated_robots_paths.append(robot_path)
            mutated_crossovers.append(mutated_robots_paths)
        return mutated_crossovers

    def add_gaussian_point(self, robot_path: RobotPath) -> RobotPath:
        """
        Given a robot path, sample a random point from the path (`orig_point`), then sample a point from a gaussian
        distribution centered at the orig_point, and replace the orig_point with the new sampled point.
        :param robot_path: The robot path to modify.
        :return: The modified robot path.
        """
        assert len(robot_path.path) >= 3
        robot = robot_path.robot
        robot_roadmap = self.roadmaps[robot]
        robot_nn = self.nearest_neighbors[robot]
        middle_points_indices = list(range(1, len(robot_path.path) - 1))
        orig_point_index = random.choice(middle_points_indices)
        orig_point = robot_path.path[orig_point_index]
        orig_point_coords = [orig_point.x().to_double(), orig_point.y().to_double()]
        random_point = np.random.multivariate_normal(mean=orig_point_coords,
                                                     cov=np.array([[self.mutation_std, 0], [0, self.mutation_std]]),
                                                     size=1)[0]
        random_point_point = Point_2(FT(random_point[0]), FT(random_point[1]))
        random_path_point = robot_nn.k_nearest(random_point_point, 1)[0]
        prev_point = robot_path.path[orig_point_index - 1]
        next_point = robot_path.path[orig_point_index + 1]

        if not nx.algorithms.has_path(robot_roadmap, prev_point, random_path_point) or not nx.algorithms.has_path(
                robot_roadmap, random_path_point, next_point):
            return robot_path
        path_to_random = nx.algorithms.shortest_path(robot_roadmap, prev_point, random_path_point, weight='weight')
        path_from_random = nx.algorithms.shortest_path(robot_roadmap, random_path_point, next_point, weight='weight')
        orig_path = robot_path.path
        start_to_random_path = orig_path[:orig_point_index - 1] + list(path_to_random)[:-1]
        random_to_end_path = list(path_from_random)[:-1] + orig_path[orig_point_index + 1:]
        return RobotPath(robot=robot, path=start_to_random_path + random_to_end_path, cell_size=self.cell_size)

    def remove_random_points(self, robot_path: RobotPath) -> RobotPath:
        """
        Removes random points from the path. This is done by selecting a sequence of points from the original path.
        and connecting the first and the last points in the sequence by the shortest path instead of the original path.
        :param robot_path: The robot path to modify.
        :return: The modified robot path.
        """
        assert len(robot_path.path) >= 3
        robot = robot_path.robot
        robot_roadmap = self.roadmaps[robot]
        points_indices = list(range(len(robot_path.path)))
        start_point_index = random.choice(points_indices[:-1])
        end_point_index = random.choice(points_indices[start_point_index + 1:])
        start_point = robot_path.path[start_point_index]
        end_point = robot_path.path[end_point_index]
        shorter_path = nx.algorithms.shortest_path(robot_roadmap, start_point, end_point, weight='weight')
        orig_path = robot_path.path
        return RobotPath(robot=robot,
                         path=orig_path[:start_point_index] + list(shorter_path)[:-1] + orig_path[end_point_index:],
                         cell_size=self.cell_size)

    def ensure_middle_point_exists(self, robot_path: RobotPath) -> RobotPath:
        """
        If the robot path consists of less then three points, returns a new robot path that contains three points - the
        start point and the end point twice.
        :param robot_path: The robot path.
        :return: The robot path that is modified to have at least three points.
        """
        if len(robot_path.path) >= 3:
            return robot_path
        else:
            return RobotPath(robot=robot_path.robot,
                             path=[robot_path.robot.start, robot_path.robot.end, robot_path.robot.end],
                             cell_size=self.cell_size)

    def mutate_gaussian_or_remove(self, individuals: list[list[RobotPath]]) -> list[list[RobotPath]]:
        """
        Applies a mutation operator that randomly applies for each individual to mutate one of two operations:
        (1) adds a new point from a gaussian distribution centered at one of the original points and (2) removes
        a randomly selected point from the original path.
        :param individuals: The individual to mutate
        :return:
        """
        mutated_crossovers: list[list[RobotPath]] = []
        for robots_paths in individuals:
            mutated_robots_paths: list[RobotPath] = []
            for robot_path in robots_paths:
                if random.random() > self.mutation_rate:
                    mutated_robots_paths.append(robot_path)
                    continue
                # For the mutation operation below, the path should contain at least three points.
                robot_path = self.ensure_middle_point_exists(robot_path)
                if random.random() < self.add_remove_mutation_ratio:
                    new_robot_path = self.add_gaussian_point(robot_path)
                else:
                    new_robot_path = self.remove_random_points(robot_path)
                mutated_robots_paths.append(new_robot_path)
            mutated_crossovers.append(mutated_robots_paths)
        return mutated_crossovers

    def get_random_robots_paths(self) -> list[RobotPath]:
        """
        For each robot, creates a random path by connecting (with the shortest paths) the start point to a random point
        to the end point of the robot.
        :return: The new random paths, one for each robot in the scene.
        """
        robots_paths = []
        for robot in self.scene.robots:
            # Create an initial path for the robot by connecting its start point, a random point and its end point.
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

    def get_shortest_paths(self) -> list[RobotPath]:
        """
        Returns a list of the shortest paths from start point to end point for each robot.
        :return: The shortest paths for each robot.
        """
        robots_paths = []
        for robot in self.scene.robots:
            robot_roadmap = self.roadmaps[robot]
            path = nx.algorithms.shortest_path(robot_roadmap, robot.start, robot.end, weight='weight')
            robots_paths.append(RobotPath(robot=robot, path=list(path), cell_size=self.cell_size))
        return robots_paths

    def get_initial_population(self) -> list[list[RobotPath]]:
        """
        Return the initial population for the genetic algorithm
        """
        if self.random_point_initialization:
            return [self.get_random_robots_paths() for _ in range(self.population_size)]
        return [self.get_shortest_paths() for _ in range(self.population_size)]

    def sample_free(self, robot: Robot):
        """
        Sample a free random point
        """
        sample = self.sampler.sample()
        while not self.collision_detection[robot].is_point_valid(sample):
            sample = self.sampler.sample()
        return sample

    def create_robot_roadmap(self, robot: Robot):
        """
        Creates the roadmap for the given robot.
        """
        robot_roadmap = nx.Graph()

        # Add points to robot's roadmap
        robot_roadmap.add_node(robot.start)
        robot_roadmap.add_node(robot.end)
        for i in range(self.num_landmarks):
            new_sample = self.sample_free(robot)
            self.free_cells.add(get_cell_indices(new_sample, self.min_cell_size))
            robot_roadmap.add_node(new_sample)

        self.nearest_neighbors[robot] = NearestNeighbors_sklearn()
        self.nearest_neighbors[robot].fit(list(robot_roadmap.nodes))

        # Connect all points to their k nearest neighbors
        for cnt, point in enumerate(robot_roadmap.nodes):
            neighbors = self.nearest_neighbors[robot].k_nearest(point, self.k + 1)
            for neighbor in neighbors:
                if self.collision_free(neighbor, point, robot):
                    robot_roadmap.add_edge(point, neighbor, weight=self.metric.dist(point, neighbor).to_double())

        assert nx.algorithms.has_path(robot_roadmap, robot.start, robot.end)
        return robot_roadmap

    def print(self, to_print: str, *args, **kwargs):
        """
        Prints if self.verbose is True
        :return:
        """
        if not self.verbose:
            return
        if self.print_prefix:
            to_print = " | ".join([self.print_prefix, to_print])
        print(to_print, file=self.writer, *args, **kwargs)

    def update_cell_size(self, new_cell_size: float) -> None:
        """
        Updates the attribute of the class according to a new cell size
        """
        self.cell_size = max(new_cell_size, self.min_cell_size)
        new_population = []
        for robots_paths in self.population:
            new_robots_paths = []
            for robot_path in robots_paths:
                new_robots_paths.append(RobotPath(robot_path.robot, robot_path.path, self.cell_size))
            new_population.append(new_robots_paths)
        self.population = new_population

    def get_bounding_box_size(self) -> float:
        """
        Returns the size of the bounding box.
        """
        bounding_box = self.calc_bounding_box()
        return max(bounding_box.max_x.to_double() - bounding_box.min_x.to_double(),
                   bounding_box.max_y.to_double() - bounding_box.min_y.to_double())

    def get_number_of_cells(self) -> int:
        """
        Returns the total number of cells.
        """
        bounding_box = self.calc_bounding_box()
        return math.ceil((bounding_box.max_x.to_double() - bounding_box.min_x.to_double()) / self.cell_size) * \
            math.ceil((bounding_box.max_y.to_double() - bounding_box.min_y.to_double()) / self.cell_size)

    def mutate(self, individuals: list[list[RobotPath]]) -> list[list[RobotPath]]:
        """
        Applies the mutation operator.
        :param individuals: The individuals to mutate.
        :return: The mutated individuals.
        """
        if self.mutate_gauss:
            return self.mutate_gaussian_or_remove(individuals)
        return self.mutate_add_sample(individuals)

    def load_scene(self, scene: Scene):
        """
        A function to load the scene. This function applies the genetic algorithm to get the final population.
        """
        super().load_scene(scene)
        self.best_fitness_values = []
        self.sampler.set_scene(scene, self._bounding_box)
        self.cell_size = max(self.min_cell_size, self.get_bounding_box_size() / 4)

        # Build collision detection and roadmap for each robot.
        self.print(f'Creating robot roadmaps...')
        for robot in scene.robots:
            self.collision_detection[robot] = collision_detection.ObjectCollisionDetection(scene.obstacles, robot)
            self.roadmaps[robot] = self.create_robot_roadmap(robot)

        # Get random initial population.
        self.print(f'Creating initial population...')
        self.population = self.get_initial_population()

        # Evolution steps.
        self.print(f'Evolution steps...')
        steps_without_progress = 0
        best_fitness_value = 0
        for step in range(self.evolution_steps):


            # In the last `self.final_steps_num` steps, change the cell size to min_cell_size: the final
            # fitness value is computed with respect to cell size of self.min_cell_size, so in the last iteration we
            # should perform evolution with the target of maximizing the final fitness function.
            if self.cell_size > self.min_cell_size and step >= self.evolution_steps - self.final_steps_num:
                self.update_cell_size(self.min_cell_size)

            # Compute fitness value.
            fitness_values = [get_fitness(robots_paths) for robots_paths in self.population]
            max_fitness_value = max(fitness_values)
            if step % 10 == 0:
                self.print(get_status_string("step", step, self.evolution_steps),
                           f'max fitness value: {max_fitness_value}')

            self.best_fitness_values.append(max(fitness_values))

            # If there is no improvement for `self.cell_size_decrease_interval` steps, decrease cell_size.
            if self.cell_size > self.min_cell_size and max(fitness_values) == best_fitness_value:
                steps_without_progress += 1
                if steps_without_progress > self.cell_size_decrease_interval:
                    steps_without_progress = 0
                    self.update_cell_size(max(self.cell_size / 2, self.min_cell_size))
                    fitness_values = [get_fitness(robots_paths) for robots_paths in self.population]

            best_fitness_value = max(fitness_values)
            fitness_distribution = get_distribution(np.array(fitness_values))

            # Get elite population.
            elite_population = [self.population[i] for i in
                                get_highest_k_indices(fitness_distribution, self.elite_size)]

            # Apply crossover and mutation operators.
            crossover_population = self.crossover(fitness_distribution, self.population_size - self.elite_size)
            mutated_crossover_population = self.mutate(crossover_population)

            self.population = elite_population + mutated_crossover_population
        self.best_fitness_values.append(max([get_fitness(robots_paths) for robots_paths in self.population]))

    def solve(self):
        """
        Based on the start and end locations of each robot, solve the scene
        (i.e. return paths for all the robots)

        :return: path collection of motion planning
        :rtype: :class:`~discopygal.solvers.PathCollection`
        """
        self.print(f'Fetching best individual...')
        path_collection = PathCollection()
        fittest_robot_paths = max(self.population, key=lambda robot_paths: get_fitness(robot_paths))
        self.print(f"chosen individual fitness: {get_fitness(fittest_robot_paths)}")
        self.print(f"total number of cells: {len(self.free_cells)}")
        for i, robot_path in enumerate(fittest_robot_paths):
            path_collection.add_robot_path(robot_path.robot, Path([PathPoint(point) for point in robot_path.path]))
        return path_collection
