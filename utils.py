from discopygal.solvers import  PathCollection
from discopygal.solvers.metrics import Metric_Euclidean
import multiprocessing
import time
from typing import Callable


def get_path_length(path_collection: PathCollection) -> float:
    length= 0.0
    for path in list(path_collection.paths.values()):
        points = path.points
        metric = Metric_Euclidean()
        length = 0.0
        for i in range(len(points) - 1):
            length += metric.dist(points[i].location, points[i + 1].location).to_double()
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