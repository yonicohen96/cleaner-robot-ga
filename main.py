"""
A script for running genetic algorithm-based experiments to solve the coverage path planning problem.

Example usage

for more information about the paramters for the script, please run: python main.py --help
"""
import argparse
import datetime
import itertools
import json
import os.path
import time

import matplotlib.pyplot as plt
import pandas as pd
import tqdm

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
MUTATION_STD_OPTION = 'mutation_std'
BASE_HYPERPARAMS = {
    SCENE_FILENAME_OPTION: ["scene3.json"],
    ITERATION_NUMBER_OPTION: [3],
    POPULATION_SIZE_OPTION: [10],
    EVOLUTION_STEPS_OPTION: [500],
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
SCENES = ["scene1.json", "scene2.json", "scene3.json"]


def _get_time_and_path_collection(solver, scene):
    start_time = time.time()
    solver.load_scene(scene)
    path_collection = solver.solve()
    total_time = time.time() - start_time
    return total_time, path_collection


def _path_collection_to_robot_paths(path_collection: PathCollection, cell_size: float) -> list[RobotPath]:
    robot_paths = []
    for robot, path in path_collection.paths.items():
        point_2_path = [point.location for point in path.points]
        robot_paths.append(RobotPath(robot, point_2_path, cell_size))
    return robot_paths


def _get_solver_from_params(params_dict: dict[str, Any], verbose: bool = True, prefix: str = "") -> CoveragePathPlanner:
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
                               verbose=verbose,
                               print_prefix=prefix
                               )


def _get_parameters_combinations_and_names(parameters: dict[str, list]) -> tuple[list[str], list]:
    names = []
    parameters_values = []
    for key, value in parameters.items():
        names.append(key)
        parameters_values.append(value)
    parameter_combinations = list(itertools.product(*parameters_values))
    return names, parameter_combinations


def _plot_all_runs(data, parameter_name, output_dir):
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


def _plot_std(data: dict[Any, list[list]], parameter_name: str, ax: plt.Axes):
    # Create a figure and axis
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


def _get_out_dir(dir_name: str | None = None) -> str:
    dir_name = dir_name or datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    out_dir = os.path.join(OUT_DIR, dir_name)
    assert not os.path.exists(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _write_times(output_dir: str, headers: list[str], values: list):
    df = pd.DataFrame(data=values).T
    df.columns = headers
    df.to_csv(os.path.join(output_dir, "times.csv"), index=False)


def single_scene_single_parameter_change(hyperparams: dict, value_to_change: str, value_options: list,
                                         verbose=False, out_dir: str | None = None) -> dict:
    assert all([len(value) == 1 for value in hyperparams.values()])
    assert value_to_change in hyperparams
    output_dir = _get_out_dir(out_dir)
    hyperparams[value_to_change] = value_options
    with open(os.path.join(output_dir, 'hyperparams.json'), 'w') as file:
        json.dump(hyperparams, file, indent=4)

    headers, parameter_combinations = _get_parameters_combinations_and_names(hyperparams)
    parameter_value_to_fitness_evolution: dict[Any, list[list[float]]] = {}
    parameter_values = []
    avg_times = []
    for params_combination_idx, combination in enumerate(parameter_combinations):
        param_prefix = get_status_string("param_value", params_combination_idx + 1, len(parameter_combinations))
        curr_params_dict = {name: combination[idx] for (idx, name) in enumerate(headers)}
        parameter_value = curr_params_dict[value_to_change]
        parameter_value_to_fitness_evolution[parameter_value] = []
        cur_combination_times = []
        with open(os.path.join(SCENE_DIR, curr_params_dict[SCENE_FILENAME_OPTION]), 'r') as fp:
            scene = Scene.from_dict(json.load(fp))
            total_iterations = curr_params_dict[ITERATION_NUMBER_OPTION]
            for iteration_number in range(total_iterations):
                iteration_prefix = get_status_string("iteration", iteration_number + 1, total_iterations)
                solver = _get_solver_from_params(curr_params_dict, verbose, " | ".join([param_prefix, iteration_prefix]))
                start_time = time.time()
                solver.load_scene(scene)
                cur_combination_times.append(time.time() - start_time)
                parameter_value_to_fitness_evolution[parameter_value].append(solver.best_fitness_values)
        parameter_values.append(parameter_value)
        avg_times.append(sum(cur_combination_times) / len(cur_combination_times))

    _write_times(output_dir, [value_to_change, "avg_time"], [parameter_values, avg_times])

    # _plot_all_runs(parameter_value_to_fitness_evolution, value_to_change, output_dir, ax)
    fig, ax = plt.subplots()
    _plot_std(parameter_value_to_fitness_evolution, value_to_change, ax)
    ax.set_title(f'Fitness values for different {value_to_change} values')
    # Show the plot
    plt.savefig(os.path.join(output_dir, f"{value_to_change}_std.pdf"))
    plt.show()
    return parameter_value_to_fitness_evolution


def all_scenes_single_parameter_change(hyperparams: dict, value_to_change: str, value_options: list,
                                       verbose: bool = False, out_dir: str | None = None) -> dict:
    assert all([len(value) == 1 for value in hyperparams.values()])
    assert value_to_change in hyperparams
    if SCENE_FILENAME_OPTION in hyperparams:
        del hyperparams[SCENE_FILENAME_OPTION]
    output_dir = _get_out_dir(out_dir)
    hyperparams[value_to_change] = value_options
    with open(os.path.join(output_dir, 'hyperparams.json'), 'w') as file:
        json.dump(hyperparams, file, indent=4)
    headers, parameter_combinations = _get_parameters_combinations_and_names(hyperparams)
    parameter_values = []
    avg_times = []
    scenes_names = []
    fig, ax = plt.subplots(1, len(SCENES), figsize=(20, 7))
    fig.suptitle(f'Fitness values for different {value_to_change} values')
    for scene_idx, scene_name in enumerate(SCENES):
        ax[scene_idx].set_title(scene_name.split(".")[0])
        scene_prefix = get_status_string("scene", scene_idx + 1, len(SCENES))
        with open(os.path.join(SCENE_DIR, scene_name), 'r') as fp:
            scene = Scene.from_dict(json.load(fp))
            parameter_value_to_fitness_evolution: dict[Any, list[list[float]]] = {}
            for params_combination_idx, combination in enumerate(parameter_combinations):
                param_prefix = get_status_string("param_value", params_combination_idx + 1, len(parameter_combinations))
                curr_params_dict = {name: combination[idx] for (idx, name) in enumerate(headers)}
                parameter_value = curr_params_dict[value_to_change]
                parameter_value_to_fitness_evolution[parameter_value] = []
                cur_combination_times = []
                total_iterations = curr_params_dict[ITERATION_NUMBER_OPTION]
                for iteration_number in range(total_iterations):
                    iteration_prefix = get_status_string("iteration", iteration_number + 1, total_iterations)
                    solver = _get_solver_from_params(curr_params_dict, verbose, " | ".join([
                        scene_prefix, param_prefix, iteration_prefix
                    ]))
                    start_time = time.time()
                    solver.load_scene(scene)
                    cur_combination_times.append(time.time() - start_time)
                    parameter_value_to_fitness_evolution[parameter_value].append(solver.best_fitness_values)
                parameter_values.append(parameter_value)
                scenes_names.append(scene_name)
                avg_times.append(sum(cur_combination_times) / len(cur_combination_times))

        # _plot_all_runs(parameter_value_to_fitness_evolution, value_to_change, output_dir)
        _plot_std(parameter_value_to_fitness_evolution, value_to_change, ax[scene_idx])
    # Show the plot
    plt.savefig(os.path.join(output_dir, f"{value_to_change}_std.pdf"))
    plt.show()
    _write_times(output_dir, ["scene_name", value_to_change, "avg_time"], [scenes_names, parameter_values, avg_times])
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
                total_time, path_collection = _get_time_and_path_collection(solver, scene)
                robots_paths = _path_collection_to_robot_paths(path_collection, solver.cell_size)
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


def flag(flag_name: str) -> str:
    return "--" + flag_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" A script for running experiments for a solver that uses a genetic"
                                                 " algorithm approach to solve the coverage path planning problem.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--out_dir", type=str, default="",
                        help="The dir name (under 'out' dir) to save the results. If not provided, the directory will"
                             " name is the date of the experiment.")

    parser.add_argument("--parameter_to_check", type=str, default="",
                        choices=['scene_filename', 'iteration_number', 'population_size', 'evolution_steps',
                                 'min_cell_size', 'cell_size_decrease_interval', 'final_steps_num',
                                 'random_point_initialization', 'elite_proportion', 'crossover_merge', 'mutation_rate',
                                 'mutate_gauss', 'add_remove_mutation_ratio', 'mutation_std'],
                        required=True,
                        help="The parameter name for which different values are being evaluated. The output are graphs"
                             " that show the fitness values as function of the evolution steps for different values of"
                             " this parameter. The parameter name is one of: 'scene_filename', 'iteration_number',"
                             " 'population_size', 'evolution_steps', 'min_cell_size', 'cell_size_decrease_interval',"
                             " 'final_steps_num', 'random_point_initialization', 'elite_proportion', 'crossover_merge',"
                             " 'mutation_rate', 'mutate_gauss', 'add_remove_mutation_ratio', 'mutation_std'")
    parser.add_argument("--parameter_values", nargs='+', type=str, required=True,
                        help="The values of parameter_to_check to ve evaluated.")

    parser.add_argument(flag(SCENE_FILENAME_OPTION), type=str,
                        choices=["scene1.json", "scene2.json", "scene3.json"],
                        help="The problem's scene, which determines the robots, start and end points, and obstacles."
                             " If not provided, all scenes are checked."
                             " Expected values: scene1.json, scene2.json, scene3.json")
    parser.add_argument(flag(ITERATION_NUMBER_OPTION), type=int, default=3,
                        help="Number of iteration per combination of the parameters for averaging results.")
    parser.add_argument(flag(POPULATION_SIZE_OPTION), type=int, default=10,
                        help="The number of individuals (lists of robots paths) for the genetic algorithm.")
    parser.add_argument(flag(EVOLUTION_STEPS_OPTION), type=int, default=500,
                        help="The number of evolutions steps.")
    parser.add_argument(flag(MIN_CELL_SIZE_OPTION), type=float, default=2.0,
                        help="The minimum cell size: the cell size decreases during evolution steps, and this value is"
                             "the minimal cell size allowed.")
    parser.add_argument(flag(CELL_SIZE_DECREASE_INTERVAL_OPTION), type=int, default=5,
                        help="This parameter determines the number of evolution steps with the same maximum fitness"
                             " value after which the cell size decreases.")
    parser.add_argument(flag(FINAL_STEPS_NUM_OPTION), type=int, default=10,
                        help="The minimum number of last steps for which the cell size is set to `min_cell_size.")
    parser.add_argument(flag(RANDOM_POINT_INITIALIZATION_OPTION), type=int, default=0,
                        help="Whether the initial population is randomly initialized. If True, each individual contains"
                             " robots paths, each one is the shortest path from the start point to the end point of the"
                             " robot, that passes through a random free point.")
    parser.add_argument(flag(ELITE_PROPORTION_OPTION), type=float, default=0.2,
                        help="The portion of the population with the highest fitness values that is copied to the next"
                             " generation.")
    parser.add_argument(flag(CROSSOVER_MERGE_OPTION), type=int, default=0,
                        help="Whether to use a merging strategy in the crossover operator (1 to use this operator and 0"
                             " to not use it). This operator returns a new robot path by merging the two selected"
                             " parents robot paths. The merged path consists of three parts: (1) The path of the first"
                             " parent from the start point to a random point. (2) The shortest path from the latter"
                             " random point to another random point in the second parent's path, and (3) the path of"
                             " the second parent from the selected random point to the end point. If this operator is"
                             " not used, then the path of the crossover is copied from one of the parents which is"
                             " chosen randomly.")
    parser.add_argument(flag(MUTATION_RATE_OPTION), type=float, default=0.5,
                        help="The portion of the crossover individual on which the mutation operator is applied.")
    parser.add_argument(flag(MUTATE_GAUSS_OPTION), type=int, default=1,
                        help="Whether to use a gaussian sampling strategy for mutation. If the value is 1, then the"
                             " mutation is applied by choosing a random point for each robot path, and sampling a"
                             " random point from a gaussian distribution (centered at the original point, with a"
                             " standard deviation that is determined by another parameter) and replacing the original"
                             " point with the new point (and connecting it to the previous and next points in the path"
                             " by shortest paths).")
    parser.add_argument(flag(ADD_REMOVE_MUTATION_RATIO_OPTION), type=float, default=0.8,
                        help="The ratio between the number of times the mutation operator adds a new point to the path"
                             " to the number of times the mutation operator selects two random points from the path and"
                             " connect them by the shortest path, and by that aims to shorten the path.")
    parser.add_argument(flag(MUTATION_STD_OPTION), type=int, default=2,
                        help="The standard deviation for the gaussian mutation strategy.")

    args = parser.parse_args()

    args_dict = vars(args)

    # Update hyperparameters dictionary with user choices.
    hyperparams = BASE_HYPERPARAMS.copy()
    for arg, value in args_dict.items():
        if value is not None and arg in hyperparams:
            # print(arg, type(value))
            hyperparams[arg] = [value]

    def get_arg_type(flag):
        for action in parser._actions:
            if action.dest == flag:
                return action.type
        return None

    # Add parameter value to check
    parameter_to_check = args_dict["parameter_to_check"]
    param_type = get_arg_type(parameter_to_check)
    input_parameter_values = [param_type(param_value) for param_value in args_dict["parameter_values"]]
    out_dir = args_dict["out_dir"]
    scene_filename = args_dict[SCENE_FILENAME_OPTION]

    # Run the experiment.
    if scene_filename:
        single_scene_single_parameter_change(hyperparams, parameter_to_check, input_parameter_values,
                                             True, out_dir)
    else:
        all_scenes_single_parameter_change(hyperparams, parameter_to_check, input_parameter_values,
                                           True, out_dir)

