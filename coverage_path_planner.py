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
    A dataclass that represents a robot and a path, and includes the cells that contain the path's points.
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
    """

    def __init__(self,
                 num_landmarks=1000,
                 k=15,
                 bounding_margin_width_factor=Solver.DEFAULT_BOUNDS_MARGIN_FACTOR,
                 population_size: int = 10,
                 evolution_steps: int = 20,
                 min_cell_size: float = 1.0,
                 cell_size_decrease_interval: int = 5,
                 random_point_initialization: int = 0,
                 elite_proportion: float = 0.1,
                 crossover_merge: int = 0,
                 mutation_rate: float = 0.3,
                 mutate_gauss: int = 1,
                 add_remove_mutation_ratio: float = 0.8,
                 mutation_std: float = 2,
                 verbose: int = 1):
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
        self.random_point_initialization = random_point_initialization
        self.elite_proportion = elite_proportion
        self.elite_size = int(elite_proportion * self.population_size)
        self.crossover_merge = crossover_merge
        self.mutation_rate = mutation_rate
        self.mutate_gauss = mutate_gauss
        self.add_remove_mutation_ratio = add_remove_mutation_ratio
        self.mutation_std = mutation_std

        # Datastructures initializations
        self.cell_size = None
        self.roadmap = None
        self.roadmaps: Dict[Robot, nx.Graph] = {}
        self.collision_detection = {}
        self.population: list[list[RobotPath]] = []

        self.verbose = verbose

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
            'evolution_steps': ('evolution steps:', 20, int),
            'min_cell_size': ('min cell size:', 1.0, float),
            'cell_size_decrease_interval': ('cell_size_decrease_interval', 5, int),
            'random_point_initialization': ('random_point_initialization', 0, int),
            'elite_proportion': ('elite proportion:', 0.1, float),
            'crossover_merge': ('crossover_merge', 0, int),
            'mutation_rate': ('mutation rate:', 0.3, float),
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
                                   d['random_point_initialization'],
                                   d['elite_proportion'],
                                   d['crossover_merge'],
                                   d['mutation_rate'],
                                   d['mutate_gauss'],
                                   d['add_remove_mutation_ratio'],
                                   d['mutation_std'],
                                   d['verbose'],
                                   )

    def get_graph(self):
        """
        Return a graph (if applicable).
        Can be overridden by solvers.

        :return: graph whose vertices are Point_2 or Point_d
        :rtype: :class:`networkx.Graph` or :class:`None`
        """
        return self.roadmap

    def collision_free(self, p1: Point_2, p2: Point_2, robot: Robot):
        """
        Get two points in the configuration space and decide if they can be connected
        """

        # Check validity of each edge separately
        edge = Segment_2(p1, p2)
        if not self.collision_detection[robot].is_edge_valid(edge):
            return False
        return True

    def merge(self, parent_0_robot_path: RobotPath, parent_1_robot_path: RobotPath, robot: Robot) -> RobotPath:
        parent_0_end_index = random.randint(0, len(parent_0_robot_path.path) - 1)
        parent_1_start_index = random.randint(0, len(parent_1_robot_path.path) - 1)
        parent_0_end_point = parent_0_robot_path.path[parent_0_end_index]
        parent_1_start_point = parent_1_robot_path.path[parent_1_start_index]
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

    def crossover_with_merge(self, fitness_distribution: np.ndarray, num_individuals: int) -> list[list[RobotPath]]:
        crossovers = []
        for child_idx in range(num_individuals):
            # Create the next child by merging two parents.
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
        if self.crossover_merge:
            return self.crossover_with_merge(fitness_distribution, num_individuals)
        return self.crossover_no_merge(fitness_distribution, num_individuals)

    def add_point_to_robot_path(self, robot_path: RobotPath) -> RobotPath:
        robot = robot_path.robot
        robot_roadmap = self.roadmaps[robot]
        orig_path = robot_path.path
        new_path = []
        while True:
            random_point = random.choice(list(robot_roadmap.nodes()))
            prev_node_idx = random.randint(0, len(orig_path) - 2)
            next_node_idx = random.randint(prev_node_idx, len(orig_path) - 1)
            if not nx.algorithms.has_path(
                    robot_roadmap, orig_path[prev_node_idx], random_point) or not nx.algorithms.has_path(
                robot_roadmap, random_point, orig_path[next_node_idx]):
                continue
            path_to_random = list(
                nx.algorithms.shortest_path(robot_roadmap, orig_path[prev_node_idx], random_point, weight='weight'))
            path_from_random = list(
                nx.algorithms.shortest_path(robot_roadmap, random_point, orig_path[next_node_idx], weight='weight'))
            new_path = orig_path[:prev_node_idx] + path_to_random[:-1] + path_from_random[:-1] + orig_path[
                                                                                                 next_node_idx:]
            break
        return RobotPath(robot=robot, path=new_path, cell_size=self.cell_size)

    def mutate_add_sample(self, crossovers: list[list[RobotPath]]) -> list[list[RobotPath]]:
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

    def add_gaussian_point(self, robot_path: RobotPath) -> RobotPath:
        assert len(robot_path.path) >= 3
        robot = robot_path.robot
        robot_roadmap = self.roadmaps[robot]
        robot_nn = self.nearest_neighbors[robot]
        middle_points_indices = list(range(1, len(robot_path.path) - 1))
        random_point_index = random.choice(middle_points_indices)
        random_point = robot_path.path[random_point_index]
        point_coords = [random_point.x().to_double(), random_point.y().to_double()]
        random_sample = np.random.multivariate_normal(mean=point_coords,
                                                      cov=np.array([[self.mutation_std, 0], [0, self.mutation_std]]),
                                                      size=1)[0]
        random_sample_point = Point_2(FT(random_sample[0]), FT(random_sample[1]))
        random_path_point = robot_nn.k_nearest(random_sample_point, 1)[0]
        prev_point = robot_path.path[random_point_index - 1]
        next_point = robot_path.path[random_point_index + 1]

        if not nx.algorithms.has_path(robot_roadmap, prev_point, random_path_point) or not nx.algorithms.has_path(
                robot_roadmap, random_path_point, next_point):
            return robot_path
        path_to_random = nx.algorithms.shortest_path(robot_roadmap, prev_point, random_path_point, weight='weight')
        path_from_random = nx.algorithms.shortest_path(robot_roadmap, random_path_point, next_point, weight='weight')
        orig_path = robot_path.path
        return RobotPath(robot=robot,
                         path=orig_path[:random_point_index - 1] + list(path_to_random)[:-1] + list(path_from_random)[
                                                                                               :-1] + orig_path[
                                                                                                      random_point_index + 1:],
                         cell_size=self.cell_size)

    def remove_random_points(self, robot_path: RobotPath) -> RobotPath:
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

    def add_middle_point(self, robot_path: RobotPath) -> RobotPath:
        if len(robot_path.path) >= 3:
            return robot_path
        else:
            return RobotPath(robot=robot_path.robot,
                             path=[robot_path.robot.start, robot_path.robot.end, robot_path.robot.end],
                             cell_size=self.cell_size)

    def mutate_gaussian_or_remove(self, crossovers: list[list[RobotPath]]) -> list[list[RobotPath]]:
        mutated_crossovers: list[list[RobotPath]] = []
        for robots_paths in crossovers:
            mutated_robots_paths: list[RobotPath] = []
            for robot_path in robots_paths:
                if random.random() > self.mutation_rate:
                    mutated_robots_paths.append(robot_path)
                    continue
                # For the mutation operation below, the path should contain at least three points.
                robot_path = self.add_middle_point(robot_path)
                if random.random() < self.add_remove_mutation_ratio:
                    new_robot_path = self.add_gaussian_point(robot_path)
                else:
                    new_robot_path = self.remove_random_points(robot_path)
                mutated_robots_paths.append(new_robot_path)
            mutated_crossovers.append(mutated_robots_paths)
        return mutated_crossovers

    def get_random_robots_paths(self) -> list[RobotPath]:
        robots_paths = []
        for robot in self.scene.robots:
            # Create a initial path for the robot by connecting its start point, a random point and its end point.
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
        robots_paths = []
        for robot in self.scene.robots:
            robot_roadmap = self.roadmaps[robot]
            path = nx.algorithms.shortest_path(robot_roadmap, robot.start, robot.end, weight='weight')
            robots_paths.append(RobotPath(robot=robot, path=list(path), cell_size=self.cell_size))
        return robots_paths

    def get_initial_population(self) -> list[list[RobotPath]]:
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
        robot_roadmap = nx.Graph()

        # Add points to robot's roadmap
        robot_roadmap.add_node(robot.start)
        robot_roadmap.add_node(robot.end)
        for i in range(self.num_landmarks):
            robot_roadmap.add_node(self.sample_free(robot))

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
        if not self.verbose:
            return
        print(to_print, file=self.writer, *args, **kwargs)

    def updated_cell_size(self, new_cell_size: float) -> None:
        self.cell_size = new_cell_size
        new_population = []
        for robots_paths in self.population:
            new_robots_paths = []
            for robot_path in robots_paths:
                new_robots_paths.append(RobotPath(robot_path.robot, robot_path.path, self.cell_size))
            new_population.append(new_robots_paths)
        self.population = new_population

    def get_bounding_box_size(self) -> float:
        bounding_box = self.calc_bounding_box()
        return max(bounding_box.max_x.to_double() - bounding_box.min_x.to_double(),
                   bounding_box.max_y.to_double() - bounding_box.min_y.to_double())

    def get_number_of_cells(self) -> int:
        bounding_box = self.calc_bounding_box()
        return math.ceil((bounding_box.max_x.to_double() - bounding_box.min_x.to_double()) / self.cell_size) * \
            math.ceil((bounding_box.max_y.to_double() - bounding_box.min_y.to_double()) / self.cell_size)

    def mutate(self, crossovers: list[list[RobotPath]]) -> list[list[RobotPath]]:
        if self.mutate_gauss:
            return self.mutate_gaussian_or_remove(crossovers)
        return self.mutate_add_sample(crossovers)

    def load_scene(self, scene: Scene):
        super().load_scene(scene)
        self.sampler.set_scene(scene, self._bounding_box)
        self.cell_size = self.get_bounding_box_size()

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
            self.print(f'\tevolution step {step + 1}/{self.evolution_steps}')

            # In the last `self.cell_size_decrease_interval` steps, change the cell size to min_cell_size: the final
            # fitness value is computed with respect to cell size of self.min_cell_size, so in the last iteration we
            # should perform evolution with the target of maximizing the final fitness function.
            if step >= self.evolution_steps - self.cell_size_decrease_interval:
                self.updated_cell_size(self.min_cell_size)

            # Compute fitness value.
            fitness_values = [get_fitness(robots_paths) for robots_paths in self.population]

            # If there is no improvement for `self.cell_size_decrease_interval` steps, decrease cell_size.
            if self.cell_size >= self.min_cell_size * 2 and max(fitness_values) == best_fitness_value:
                steps_without_progress += 1
                if steps_without_progress > self.cell_size_decrease_interval:
                    steps_without_progress = 0
                    self.updated_cell_size(self.cell_size / 2)
                    fitness_values = [get_fitness(robots_paths) for robots_paths in self.population]
            self.print("\tmax fitness value: ", max(fitness_values))

            best_fitness_value = max(fitness_values)
            fitness_distribution = get_distribution(np.array(fitness_values))

            # Get elite population.
            elite_population = [self.population[i] for i in
                                get_highest_k_indices(fitness_distribution, self.elite_size)]

            # Apply crossover and mutation operators.
            crossover_population = self.crossover(fitness_distribution, self.population_size - self.elite_size)
            mutated_crossover_population = self.mutate(crossover_population)

            self.population = elite_population + mutated_crossover_population

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
        self.print(f"number of cells: {self.get_number_of_cells()}")
        self.print(f"chosen individual fitness: {get_fitness(fittest_robot_paths)}")
        for i, robot_path in enumerate(fittest_robot_paths):
            path_collection.add_robot_path(robot_path.robot, Path([PathPoint(point) for point in robot_path.path]))
        return path_collection
