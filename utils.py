from discopygal.solvers import  PathCollection
from discopygal.solvers.metrics import Metric_Euclidean
import multiprocessing
import time
from typing import Callable
import numpy as np
from discopygal.bindings import Point_2


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



