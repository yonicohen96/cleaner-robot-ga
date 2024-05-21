import datetime
import json
import os.path
import time

import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import itertools

from coverage_path_planner import *

SCENE_FILENAME_OPTION = 'scene_filename'
ITERATION_NUMBER_OPTION = 'iteration_number'
POPULATION_SIZE_OPTION = 'population_size'
EVOLUTION_STEPS_OPTION = 'evolution_steps'
MIN_CELL_SIZE_OPTION = 'min_cell_size'
CELL_SIZE_DECREASE_INTERVAL_OPTION = 'cell_size_decrease_interval'
FINAL_STEPS_NUM_OPTION = 'final_steps_num'
RANDOM_POINT_INITIALIZATION_OPTION = 'random_point_initialization'
ELITE_PROPORTION_OPTION = 'elite_proportion'
CROSSOVER_MERGE_OPTION = 'crossover_merge'
MUTATION_RATE_OPTION = 'mutation_rate'
MUTATE_GAUSS_OPTION = 'mutate_gauss'
ADD_REMOVE_MUTATION_RATIO_OPTION = 'add_remove_mutation_ratio'
MUTATION_STD_OPTION = 'mutation_st'
BASE_HYPERPARAMS = {
    SCENE_FILENAME_OPTION: ["scene3.json"],
    ITERATION_NUMBER_OPTION: [3],
    POPULATION_SIZE_OPTION: [10],
    EVOLUTION_STEPS_OPTION: [10],
    MIN_CELL_SIZE_OPTION: [2.0],
    CELL_SIZE_DECREASE_INTERVAL_OPTION: [5],
    FINAL_STEPS_NUM_OPTION: [10],
    RANDOM_POINT_INITIALIZATION_OPTION: [0],
    ELITE_PROPORTION_OPTION: [0.2],
    CROSSOVER_MERGE_OPTION: [0],
    MUTATION_RATE_OPTION: [0.5],
    MUTATE_GAUSS_OPTION: [1],
    ADD_REMOVE_MUTATION_RATIO_OPTION: [0.8],
    MUTATION_STD_OPTION: [2],
}
AVG_TIME_FIELD = "avg_time"
AVG_FITNESS_FIELD = "avg_fitness"
SCENE_DIR = "scenes"
OUT_DIR = "out"


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


def _get_solver_from_params(params_dict: dict[str, Any], verbose: bool = True) -> CoveragePathPlanner:
    return CoveragePathPlanner(population_size=params_dict[POPULATION_SIZE_OPTION],
                               evolution_steps=params_dict[EVOLUTION_STEPS_OPTION],
                               min_cell_size=params_dict[MIN_CELL_SIZE_OPTION],
                               cell_size_decrease_interval=params_dict[
                                   CELL_SIZE_DECREASE_INTERVAL_OPTION],
                               final_steps_num=params_dict[FINAL_STEPS_NUM_OPTION],
                               random_point_initialization=params_dict[
                                   RANDOM_POINT_INITIALIZATION_OPTION],
                               elite_proportion=params_dict[ELITE_PROPORTION_OPTION],
                               crossover_merge=params_dict[CROSSOVER_MERGE_OPTION],
                               mutation_rate=params_dict[MUTATION_RATE_OPTION],
                               mutate_gauss=params_dict[MUTATE_GAUSS_OPTION],
                               add_remove_mutation_ratio=params_dict[ADD_REMOVE_MUTATION_RATIO_OPTION],
                               mutation_std=params_dict[MUTATION_STD_OPTION],
                               verbose=verbose
                               )


def _get_parameters_combinations_and_names(parameters: dict[str, list]) -> tuple[list[str], list]:
    names = []
    parameters_values = []
    for key, value in parameters.items():
        names.append(key)
        parameters_values.append(value)
    parameter_combinations = list(itertools.product(*parameters_values))
    return names, parameter_combinations


def plot_all_runs(data, parameter_name, output_dir):
    fig, ax = plt.subplots()
    # Generate a list of colors
    colors = itertools.cycle(plt.cm.tab10.colors)
    added_legend = set()
    # Plot each key with a different color
    for key, list_of_lists in data.items():
        color = next(colors)
        for y_values in list_of_lists:
            x_values = range(len(y_values))
            if key not in added_legend:
                ax.plot(x_values, y_values, label=f'{key}', color=color)
                added_legend.add(key)
            else:
                ax.plot(x_values, y_values, color=color)
    # Add a legend
    ax.legend()
    ax.get_legend().set_title(parameter_name)
    # Add labels
    ax.set_xlabel('evolution step')
    ax.set_ylabel('fitness value')
    ax.set_title(f'Fitness values different {parameter_name} values')
    # Show the plot
    plt.savefig(os.path.join(output_dir, f"{parameter_name}_runs.pdf"))
    plt.show()


def plot_std(data, parameter_name, output_dir):
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Generate a list of colors
    colors = itertools.cycle(plt.cm.tab10.colors)
    # Plot each key with a different color
    for key, list_of_lists in data.items():
        color = next(colors)
        # Convert the list of lists to a numpy array for easy computation
        array = np.array(list_of_lists)
        # Calculate the mean and standard deviation along the first axis (time points)
        mean_values = np.mean(array, axis=0)
        std_values = np.std(array, axis=0)
        # Generate x values
        x_values = range(len(mean_values))
        # Plot the mean values
        ax.plot(x_values, mean_values, label=key, color=color)
        # Plot the standard deviation as a shaded area
        ax.fill_between(x_values, mean_values - std_values, mean_values + std_values, color=color, alpha=0.3)
    # Add a legend
    ax.legend()
    ax.get_legend().set_title(parameter_name)
    # Add labels
    ax.set_xlabel('evolution step')
    ax.set_ylabel('fitness value')
    ax.set_title(f'Fitness values for different {parameter_name} values')
    # Show the plot
    plt.savefig(os.path.join(output_dir, f"{parameter_name}_std.pdf"))
    plt.show()


def _get_out_dir() -> str:
    out_dir = os.path.join(OUT_DIR, datetime.datetime.now().strftime('%y%m%d-%H%M%S'))
    assert not os.path.exists(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def write_times(output_dir: str, value_to_change: str, parameter_values: list, avg_times: list[float]) -> None:
    df = pd.DataFrame(data=[parameter_values, avg_times]).T
    df.columns = [value_to_change, "avg_time"]
    df.to_csv(os.path.join(output_dir, "times.csv"), index=False)


def single_parameter_change(hyperparams: dict, value_to_change: str, value_options: list, verbose=False) -> dict:
    assert all([len(value) == 1 for value in hyperparams.values()])
    assert value_to_change in hyperparams
    output_dir = _get_out_dir()
    hyperparams[value_to_change] = value_options
    with open(os.path.join(output_dir, 'hyperparams.json'), 'w') as file:
        json.dump(hyperparams, file, indent=4)

    headers, parameter_combinations = _get_parameters_combinations_and_names(hyperparams)
    parameter_value_to_fitness_evolution: dict[Any, list[list[float]]] = {}
    parameter_values = []
    avg_times = []
    for combination in tqdm.tqdm(parameter_combinations, desc="params combination"):
        curr_params_dict = {name: combination[idx] for (idx, name) in enumerate(headers)}
        parameter_value = curr_params_dict[value_to_change]
        parameter_value_to_fitness_evolution[parameter_value] = []
        cur_combination_times = []
        with open(os.path.join(SCENE_DIR, curr_params_dict[SCENE_FILENAME_OPTION]), 'r') as fp:
            scene = Scene.from_dict(json.load(fp))
            for _ in range(curr_params_dict[ITERATION_NUMBER_OPTION]):
                solver = _get_solver_from_params(curr_params_dict, verbose)
                start_time = time.time()
                solver.load_scene(scene)
                cur_combination_times.append(time.time() - start_time)
                parameter_value_to_fitness_evolution[parameter_value].append(solver.best_fitness_values)
        parameter_values.append(parameter_value)
        avg_times.append(sum(cur_combination_times) / len(cur_combination_times))

    write_times(output_dir, value_to_change, parameter_values, avg_times)
    plot_all_runs(parameter_value_to_fitness_evolution, value_to_change, output_dir)
    plot_std(parameter_value_to_fitness_evolution, value_to_change, output_dir)
    return parameter_value_to_fitness_evolution


def combinations_final_results(hyperparams: dict, verbose=False) -> pd.DataFrame:
    output_dir = _get_out_dir()
    result = []
    headers, parameter_combinations = _get_parameters_combinations_and_names(hyperparams)
    for combination in tqdm.tqdm(parameter_combinations, desc="params combination"):
        curr_params_dict = {name: combination[idx] for (idx, name) in enumerate(headers)}
        with open(os.path.join(SCENE_DIR, curr_params_dict[SCENE_FILENAME_OPTION]), 'r') as fp:
            solver = _get_solver_from_params(curr_params_dict, verbose)
            scene = Scene.from_dict(json.load(fp))
            times = []
            fitness_values = []
            for _ in range(curr_params_dict[ITERATION_NUMBER_OPTION]):
                total_time, path_collection = get_time_and_path_collection(solver, scene)
                robots_paths = path_collection_to_robot_paths(path_collection, solver.cell_size)
                fitness = get_fitness(robots_paths)
                times.append(total_time)
                fitness_values.append(fitness)

            # Calculate statistics for scene and algorithm.
            avg_time = sum(times) / len(times)
            avg_fitness = sum(fitness_values) / len(fitness_values)
            result.append(list(combination) + [avg_time, avg_fitness])

    df = pd.DataFrame(data=result, columns=headers + [AVG_TIME_FIELD, AVG_FITNESS_FIELD])
    df.to_csv(os.path.join(output_dir, "out.csv"), index=False)
    return df



if __name__ == '__main__':
    # combinations_final_results(hyperparams=BASE_HYPERPARAMS))
    # combinations_final_results(BASE_HYPERPARAMS, True, verbose=True)
    # write_times("out", "population_size", [10, 20, 30], [1, 4, 6])
    single_parameter_change(BASE_HYPERPARAMS, POPULATION_SIZE_OPTION, [10, 20, 30], False)

    # TODO start with a fixed values and for each parameter check different values and plot graphs of differet values
    #  as a function of evolutio steps. for example different population size, and number iterations is 3,
    #  then create a graph that each of the population size values is a color so for each color we should have 3 curves.


