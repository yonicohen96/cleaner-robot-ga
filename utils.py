import random
from typing import Any

import numpy as np
from discopygal.bindings import Point_2
from discopygal.solvers import PathCollection
from discopygal.solvers.metrics import Metric_Euclidean


def get_point2_list_length(points: list[Point_2]) -> float:
    """
    Returns the length of a path which is a list of Point_2 objects.
    """
    length = 0.0
    metric = Metric_Euclidean()
    for i in range(len(points) - 1):
        length += metric.dist(points[i], points[i + 1]).to_double()
    return length


def get_path_collection_length(path_collection: PathCollection) -> float:
    """
    Returns the total length of the paths in a PathCollection object.
    :param path_collection:
    :return:
    """
    length = 0.0
    for path in list(path_collection.paths.values()):
        length += get_point2_list_length([point.location for point in path.points])
    return length


def get_coord_index(coord: float, division_factor: float):
    """
    Returns an index for a coordinate.
    """
    return np.floor(coord / division_factor)


def get_cell_indices(point: Point_2, division_factor: float) -> tuple[int, int]:
    """
    Returns a cell indices for a given point.
    """
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
    """
    Returns a list of k indices that corresponds to the items in the input list with the highest values.
    """
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    return sorted_indices[:k]


def random_choices_no_repetitions(population: list[Any], weights: list[float] | np.ndarray, k: int) -> list[Any]:
    """
    Selects k random elements.
    """
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



# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split

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

