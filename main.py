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
MIN_CELL_SIZE_OPTION = 'min_cell_size'
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


def run_exp(hyperparams: dict, save: bool = False, output_path: str = "", verbose=False) -> pd.DataFrame:
    if save:
        output_filename = datetime.datetime.now().strftime('%y%m%d-%H%M.csv')
        output_path = output_path or os.path.join("out", output_filename)
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
                                    min_cell_size=curr_params_dict[MIN_CELL_SIZE_OPTION],
                                    elite_proportion=curr_params_dict[ELITE_PROPORTION_OPTION],
                                    cells_length_weights_ratio=curr_params_dict[CELLS_LENGTH_WEIGHTS_RATIO_OPTION],
                                    mutation_rate=curr_params_dict[MUTATION_RATE_OPTION],
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

    df = pd.DataFrame(result, columns=headers + ['avg_time', 'avg_fitness_values'])
    if save:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


def first_params_initialization():
    hyperparams = {
        SCENE_FILENAME_OPTION: ["basic_scene.json"],
        ITERATION_NUMBER_OPTION: [2],
        POPULATION_SIZE_OPTION: [10, 20],
        EVOLUTION_STEPS_OPTION: [20, 40],
        MIN_CELL_SIZE_OPTION: [1.0],
        ELITE_PROPORTION_OPTION: [0.1, 0.5],
        CELLS_LENGTH_WEIGHTS_RATIO_OPTION: [0.2, 0.8],
        MUTATION_RATE_OPTION: [0.1, 0.5, 0.9],
    }
    run_exp(hyperparams, save=True, verbose=False)


def get_coef(filename):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    # Step 1: Read the CSV file into a pandas DataFrame
    data = pd.read_csv(os.path.join("out", filename))

    # Step 2: Separate the features (A, B, C) and the target variable (D)
    X = data[[
        'iteration_number',
        'population_size',
        'evolution_steps',
        'cell_size',
        'elite_proportion',
        'cells_length_weights_ratio',
        'mutation_rate']]
    y = data['avg_fitness_values']

    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 5: Make predictions using the testing set
    y_pred = model.predict(X_test)

    # Step 6: Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

    # Optional: Print the coefficients of the linear regression model
    print(f'Coefficients: {model.coef_}')
    print(f'Intercept: {model.intercept_}')


def get_correlation():
    import pandas as pd

    # Step 1: Read the CSV file into a pandas DataFrame
    data = pd.read_csv(os.path.join("out", "240514-2048.csv"))

    # Step 2: Separate the features (A, B, C) and the target variable (D)
    X = data[[
        'iteration_number',
        'population_size',
        'evolution_steps',
        'cell_size',
        'elite_proportion',
        'cells_length_weights_ratio',
        'mutation_rate',
        'avg_time',
        'fitness_values',
    ]]
    print(X.corr().to_csv("out/correlation.csv"))


def single_debug_experiment():
    hyperparams = {
        SCENE_FILENAME_OPTION: ["basic_scene.json"],
        ITERATION_NUMBER_OPTION: [1],
        POPULATION_SIZE_OPTION: [10],
        EVOLUTION_STEPS_OPTION: [50],
        MIN_CELL_SIZE_OPTION: [1.0],
        ELITE_PROPORTION_OPTION: [0.1],
        CELLS_LENGTH_WEIGHTS_RATIO_OPTION: [0.8],
        MUTATION_RATE_OPTION: [0.5],
    }
    run_exp(hyperparams, save=False, verbose=True)


if __name__ == '__main__':
    single_debug_experiment()
