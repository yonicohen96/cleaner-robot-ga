import multiprocessing
import random
from typing import Any
from typing import Callable

import numpy as np
from discopygal.bindings import Point_2
from discopygal.solvers import PathCollection
from discopygal.solvers.metrics import Metric_Euclidean


def get_point2_list_length(points: list[Point_2]) -> float:
    length = 0.0
    metric = Metric_Euclidean()
    for i in range(len(points) - 1):
        length += metric.dist(points[i], points[i + 1]).to_double()
    return length


def get_path_collection_length(path_collection: PathCollection) -> float:
    length = 0.0
    for path in list(path_collection.paths.values()):
        length += get_point2_list_length([point.location for point in path.points])
    return length


def timeout_function(timeout: int, function: Callable, *args, **kwargs):
    # Start bar as a process
    p = multiprocessing.Process(target=function, args=args, kwargs=kwargs)
    p.start()

    # Wait for 10 seconds or until process finishes
    p.join(timeout)

    # If thread is still active
    if p.is_alive():
        print("running... let's kill it...")

        # Terminate - may not work if process is stuck for good
        p.terminate()
        # OR Kill - will work for sure, no chance for process to finish nicely however
        # p.kill()

        p.join()


def get_coord_index(coord: float, division_factor: float):
    return np.floor(coord / division_factor)


def get_cell_indices(point: Point_2, division_factor: float) -> tuple[int, int]:
    return (int(get_coord_index(point.x().to_double(), division_factor)),
            int(get_coord_index(point.y().to_double(), division_factor)))


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
