import datetime
import itertools
import json
import os.path
import time

import pandas as pd
import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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
    EVOLUTION_STEPS_OPTION: [400],
    MIN_CELL_SIZE_OPTION: [1.0],
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


def run_exp(hyperparams: dict, save: bool = False, output_path: str = "", verbose=False) -> pd.DataFrame:
    if save:
        output_filename = datetime.datetime.now().strftime('%y%m%d-%H%M.csv')
        output_path = output_path or os.path.join(OUT_DIR, output_filename)
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
        with open(os.path.join(SCENE_DIR, curr_params_dict[SCENE_FILENAME_OPTION]), 'r') as fp:
            scene = Scene.from_dict(json.load(fp))
            solver = CoveragePathPlanner(population_size=curr_params_dict[POPULATION_SIZE_OPTION],
                                         evolution_steps=curr_params_dict[EVOLUTION_STEPS_OPTION],
                                         min_cell_size=curr_params_dict[MIN_CELL_SIZE_OPTION],
                                         cell_size_decrease_interval=curr_params_dict[
                                             CELL_SIZE_DECREASE_INTERVAL_OPTION],
                                         final_steps_num=curr_params_dict[FINAL_STEPS_NUM_OPTION],
                                         random_point_initialization=curr_params_dict[
                                             RANDOM_POINT_INITIALIZATION_OPTION],
                                         elite_proportion=curr_params_dict[ELITE_PROPORTION_OPTION],
                                         crossover_merge=curr_params_dict[CROSSOVER_MERGE_OPTION],
                                         mutation_rate=curr_params_dict[MUTATION_RATE_OPTION],
                                         mutate_gauss=curr_params_dict[MUTATE_GAUSS_OPTION],
                                         add_remove_mutation_ratio=curr_params_dict[ADD_REMOVE_MUTATION_RATIO_OPTION],
                                         mutation_std=curr_params_dict[MUTATION_STD_OPTION],
                                         verbose=verbose
                                         )

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
    if save:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
    return df

#
# def get_coef(filename):
#     # Step 1: Read the CSV file into a pandas DataFrame
#     data = pd.read_csv(os.path.join(OUT_DIR, filename))
#
#     # Step 2: Separate the features (A, B, C) and the target variable (D)
#     X = data[NUMERICAL_OPTIONS_LIST]
#     y = data[AVG_FITNESS_FIELD]
#
#     # Step 3: Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Step 4: Fit the linear regression model
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#
#     # Step 5: Make predictions using the testing set
#     y_pred = model.predict(X_test)
#
#     # Step 6: Evaluate the model
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#
#     print(f'Mean Squared Error: {mse}')
#     print(f'R^2 Score: {r2}')
#
#     # Optional: Print the coefficients of the linear regression model
#     print(f'Coefficients: {model.coef_}')
#     print(f'Intercept: {model.intercept_}')

#
# def get_correlation():
#     # Step 1: Read the CSV file into a pandas DataFrame
#     data = pd.read_csv(os.path.join(OUT_DIR, "240514-2048.csv"))
#
#     # Step 2: Separate the features (A, B, C) and the target variable (D)
#     X = data[NUMERICAL_OPTIONS_LIST + [AVG_TIME_FIELD, AVG_FITNESS_FIELD]]
#     print(X.corr().to_csv("out/correlation.csv"))


def single_debug_experiment(save: bool):
    run_exp(BASE_HYPERPARAMS, save=save, verbose=True)


if __name__ == '__main__':
    single_debug_experiment(False)
    # TODO start with a fixed values and for each parameter check different values and plot graphs of differet values
    #  as a function of evolutio steps. for example different population size, and number iterations is 3,
    #  then create a graph that each of the population size values is a color so for each color we should have 3 curves.
