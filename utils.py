from discopygal.solvers import  PathCollection
from discopygal.solvers.metrics import Metric_Euclidean


def get_path_length(path_collection: PathCollection) -> float:
    assert len(path_collection.paths.values()) == 1
    points = list(path_collection.paths.values())[0].points
    metric = Metric_Euclidean()
    length = 0.0
    for i in range(len(points) - 1):
        length += metric.dist(points[i].location, points[i + 1].location).to_double()
    return length
