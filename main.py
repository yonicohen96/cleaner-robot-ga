import os.path

from discopygal.solvers.prm import PRM
import json
import time
from discopygal.solvers import Scene, Path, PathCollection
from typing import Optional
from discopygal.bindings import *
from discopygal.geometry_utils import conversions
from collections import defaultdict
from utils import *
from cleaner_robot_ga import *


def get_time_and_path_collection(solver_class, num_landmarks, k, scene):
    solver = solver_class(num_landmarks=num_landmarks, k=k)
    start_time = time.time()
    solver.load_scene(scene)
    path_collection = solver.solve()
    total_time = time.time() - start_time
    return total_time, path_collection


def path_collection_to_robot_paths(path_collection: PathCollection, cell_size: float) -> list[RobotPath]:
    robot_paths = []
    for robot, path in path_collection.paths.items():
        # TODO convert path to point_2 list.
        point_2_path = path
        robot_paths.append(RobotPath(robot, point_2_path, cell_size))
    return robot_paths

def run_exp(num_iterations=5, num_landmarks=1000, k=15):
    solver_name_to_class = {"CleanerRobotGA": CleanerRobotGA}
    scenes_filenames = ["basic.json"]
    result = []
    for scene_name in scenes_filenames:
        print(f"Scene: {scene_name}")
        with open(os.path.join("scenes", scene_name), 'r') as fp:
            scene = Scene.from_dict(json.load(fp))
            for solver_name, solver_class in solver_name_to_class.items():
                print(f"\tSolver: {solver_name}")
                times = []
                paths = []
                for iteration in range(num_iterations):
                    print(f"\t\tIteration {iteration + 1}\{num_iterations}")
                    total_time, path_collection = get_time_and_path_collection(solver_class, num_landmarks, k, scene)
                    # TODO compute fitness or return fitness

                    path_length = get_path_collection_length(path_collection)
                    times.append(total_time)
                    paths.append(path_length)

                # Calculate statistics for scene and algorithm.
                avg_time = sum(times) / len(times)
                avg_path_length = sum(paths) / len(paths)
                result.append([scene_name, solver_name, avg_time, avg_path_length])
    return result


if __name__ == '__main__':
    print(run_exp())
