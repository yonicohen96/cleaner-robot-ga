import networkx as nx
from discopygal.solvers import Robot
from discopygal.solvers import Scene
from discopygal.solvers import PathPoint, Path

from discopygal.solvers.samplers import Sampler_Uniform
from discopygal.solvers.nearest_neighbors import NearestNeighbors_sklearn
from discopygal.bindings import *
from discopygal.geometry_utils import collision_detection
from discopygal.solvers.Solver import Solver
from typing import Dict, Any
import random
from dataclasses import dataclass
from utils import *
import math


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


def get_distribution(arr: np.ndarray, opposite_values=False) -> np.ndarray:
    """
    Given an array with non-negative values, derive a distribution which is proportional to the values of the array
    after shifting them so that the minimum value is 0.
    :param arr: The array with the non-negative values.
    :param opposite_values: Whether a smaller value in arr should get a higher probability.
    :return: The derived distribution.
    """
    scores = arr.max() - arr if opposite_values else arr - arr.min()
    if scores.max() == scores.min():
        return np.full(scores.size, 1 / scores.size)
    return scores / scores.sum()


def get_highest_k_indices(values: list | np.ndarray, k: int) -> list[int]:
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
    weights = weights.copy()
    result = []
    population_indices = list(range(len(population)))
    for i in range(k):
        selected_item_idx = random.choices(population=population_indices, weights=weights, k=1)[0]
        result.append(population[selected_item_idx])
        weights[selected_item_idx] = 0
        if sum(weights) == 0:
            weights = np.full(len(weights), 1 / len(weights))
    return result


def get_fitness(robots_paths: list[RobotPath]) -> float:
    cells = dict()
    for robot_path in robots_paths:
        for cell in robot_path.cells:
            if cell in cells:
                cells[cell] = 0
            else:
                cells[cell] = 1
    return sum(cells.values())


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

    def __init__(self,
                 num_landmarks=1000,
                 k=15,
                 bounding_margin_width_factor=Solver.DEFAULT_BOUNDS_MARGIN_FACTOR,
                 population_size: int = 10,
                 evolution_steps: int = 20,
                 min_cell_size: float = 1.0,
                 elite_proportion: float = 0.1,
                 mutation_rate: float = 0.3,
                 cell_size_decrease_interval: int = 5,
                 verbose: bool = True):
        assert population_size > 1
        # Check that elite population is not the entire population.
        assert population_size > int(elite_proportion * population_size)
        super().__init__(bounding_margin_width_factor)

        # Roadmaps creation attributes
        self.nearest_neighbors = NearestNeighbors_sklearn()
        self.metric = Metric_Euclidean
        self.sampler = Sampler_Uniform()
        self.num_landmarks = num_landmarks
        self.k = k

        # Genetic algorithm attributes.
        self.population_size = population_size
        self.evolution_steps = evolution_steps
        self.min_cell_size = min_cell_size
        self.elite_proportion = elite_proportion
        self.elite_size = int(elite_proportion * self.population_size)
        self.mutation_rate = mutation_rate
        self.cell_size_decrease_interval = cell_size_decrease_interval
        self.cell_size = None

        # Datastructures initializations
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
            'elite_proportion': ('elite proportion:', 0.1, float),
            'mutation_rate': ('mutation rate:', 0.3, float),
            'cell_size_decrease_interval': ('cell_size_decrease_interval', 5, int),
            'verbose': ('verbose:', True, bool),
        }

    @staticmethod
    def from_arguments(d):
        """
        Get a dictionary of arguments and return a solver.
        Should be overridden by solvers.

        :param d: arguments dict
        :type d: :class:`dict`
        """
        return CleanerRobotGA(d['num_landmarks'],
                              d['k'],
                              FT(d['bounding_margin_width_factor']),
                              d['population_size'],
                              d['evolution_steps'],
                              d['min_cell_size'],
                              d['elite_proportion'],
                              d['mutation_rate'],
                              d['cell_size_decrease_interval'],
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

    def crossover(self, fitness_distribution: np.ndarray, num_individuals: int) -> list[list[RobotPath]]:
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
