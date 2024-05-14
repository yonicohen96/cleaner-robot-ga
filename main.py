import os.path

import json
from cleaner_robot_ga import *
import pandas as pd
import datetime
import itertools
import tqdm

SCENE_FILENAME_OPTION = 'scene_filename'
ITERATION_NUMBER_OPTION = 'iteration_number'
POPULATION_SIZE_OPTION = 'population_size'
EVOLUTION_STEPS_OPTION = 'evolution_steps'
CELL_SIZE_OPTION = 'cell_size'
ELITE_PROPORTION_OPTION = 'elite_proportion'
CELLS_LENGTH_WEIGHTS_RATIO_OPTION = 'cells_length_weights_ratio'
MUTATION_RATE_OPTION = 'mutation_rate'


def get_time_and_path_collection(solver, scene):
    start_time = time.time()
    solver.load_scene(scene)
    path_collection = solver.solve()
    total_time = time.time() - start_time
    return total_time, path_collection


def path_collection_to_robot_paths(path_collection: PathCollection, cell_size: float) -> list[RobotPath]:
    robot_paths = []
    for robot, path in path_collection.paths.items():
        point_2_path = [point.location for point in path.points]
        robot_paths.append(RobotPath(robot, point_2_path, cell_size))
    return robot_paths


def run_exp(hyperparams: dict) -> None:
    output_filename = datetime.datetime.now().strftime('%y%m%d-%H%M.csv')
    output_path = os.path.join("out", output_filename)
    assert not os.path.exists(output_path)
    headers = []
    parameters_values = []
    for key, value in hyperparams.items():
        headers.append(key)
        parameters_values.append(value)

    result = []
    parameter_combinations = list(itertools.product(*parameters_values))
    for combination in tqdm.tqdm(parameter_combinations, desc="params combination"):
        curr_params_dict = {name: combination[idx] for (idx, name) in enumerate(headers)}
        with open(os.path.join("scenes", curr_params_dict[SCENE_FILENAME_OPTION]), 'r') as fp:
            scene = Scene.from_dict(json.load(fp))
            solver = CleanerRobotGA(population_size=curr_params_dict[POPULATION_SIZE_OPTION],
                                    evolution_steps=curr_params_dict[EVOLUTION_STEPS_OPTION],
                                    cell_size=curr_params_dict[CELL_SIZE_OPTION],
                                    elite_proportion=curr_params_dict[ELITE_PROPORTION_OPTION],
                                    cells_length_weights_ratio=curr_params_dict[CELLS_LENGTH_WEIGHTS_RATIO_OPTION],
                                    mutation_rate=curr_params_dict[MUTATION_RATE_OPTION],
                                    verbose=False
                                    )
            times = []
            lengths = []
            cells_num = []
            for _ in range(curr_params_dict[ITERATION_NUMBER_OPTION]):
                total_time, path_collection = get_time_and_path_collection(solver, scene)
                robots_paths = path_collection_to_robot_paths(path_collection, solver.cell_size)
                fitness = get_fitness(robots_paths)
                times.append(total_time)
                lengths.append(fitness.length)
                cells_num.append(fitness.cells_num)

            # Calculate statistics for scene and algorithm.
            avg_time = sum(times) / len(times)
            avg_path_length = sum(lengths) / len(lengths)
            avg_cells_num = sum(cells_num) / len(cells_num)

            result.append(list(combination) + [avg_time, avg_path_length, avg_cells_num])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(result, columns=headers + ['avg_time', 'avg_path_length', 'avg_cells_num'])
    df.to_csv(output_path, index=False)


def first_params_initialization():
    hyperparams = {
        SCENE_FILENAME_OPTION: ["basic_scene.json"],
        ITERATION_NUMBER_OPTION: [2],
        POPULATION_SIZE_OPTION: [10, 20],
        EVOLUTION_STEPS_OPTION: [20, 40],
        CELL_SIZE_OPTION: [1.0],
        ELITE_PROPORTION_OPTION: [0.1, 0.5],
        CELLS_LENGTH_WEIGHTS_RATIO_OPTION: [0.2, 0.8],
        MUTATION_RATE_OPTION: [0.1, 0.5, 0.9],
    }
    run_exp(hyperparams)



if __name__ == '__main__':
    first_params_initialization()

