from dataclasses import dataclass
import os

SCENES_DIR = "scenes"


def _unpack_points(points: list[tuple[float, float]]):
    result = []
    for point in points:
        result.append(point[0])
        result.append(point[1])
    return result


def get_point_string(x, y):
    return f"""                [
                    {x},
                    {y}
                ]"""

def get_obstacle_string(points: list[float]):
    poly_points = [get_point_string(points[idx], points[idx + 1]) for idx in range(0, len(points), 2)]
    all_poly_points = ",\n".join(poly_points)
    ""
    return (f"""
    {{
        "__class__": "ObstaclePolygon",
        "poly": [
{all_poly_points}
        ],
        "data": {{
        }}
    }}""")


@dataclass
class BoxBounds:
    top: float
    bottom: float
    left: float
    right: float


def get_bounding_box_string(Box: BoxBounds) -> str:
    left, top, right, bottom = Box.left, Box.top, Box.right, Box.bottom
    left_obstacle = [(left, top), (left - 1, top + 1), (left - 1, bottom - 1), (left, bottom)]
    right_obstacle = [(right, top), (right + 1, top + 1), (right + 1, bottom - 1), (right, bottom)]
    top_obstacle = [(left, top), (right, top), (right + 1, top + 1), (left - 1, top + 1)]
    bottom_obstacle = [(left, bottom), (right, bottom), (right + 1, bottom - 1), (left - 1, bottom - 1)]
    obstacles_strings = []
    for obstacle in [left_obstacle, right_obstacle, top_obstacle, bottom_obstacle]:
        points = _unpack_points(obstacle)
        obstacles_strings.append(get_obstacle_string(points))
    return ",".join(obstacles_strings)


@dataclass
class RobotDiscInfo:
    start: tuple[float, float]
    end: tuple[float, float]
    radius: float


def get_robot_string(robot_info: RobotDiscInfo):
    return (f"""
    {{
      "__class__": "RobotDisc",
      "radius": {robot_info.radius},
      "start": [
        {robot_info.start[0]},
        {robot_info.start[1]}
      ],
      "end": [
        {robot_info.end[0]},
        {robot_info.end[1]}
      ],
      "data": {{}}
    }}""")

def write_to_file(filename, text):
    # Check if the directory exists, if not, create it
    if not os.path.exists(SCENES_DIR):
        os.makedirs(SCENES_DIR)

    # Write the text to the file
    with open(os.path.join(SCENES_DIR, filename), 'w') as file:
        file.write(text)

def get_scene_string(robots_info: list[RobotDiscInfo], obstacles: list[list[float]], box_bounds: BoxBounds):
    robots_string = ",\n".join([get_robot_string(robot_info) for robot_info in robots_info])
    obstacles_string = ",\n".join([get_obstacle_string(obstacle) for obstacle in obstacles])
    bounding_box_string = get_bounding_box_string(box_bounds)
    return (f"""{{
  "__class__": "Scene",
  "obstacles": [{obstacles_string},\n{bounding_box_string}],
  "robots": [{robots_string}],
  "metadata": {{}}
}}
    """)


def get_scene_1(filename: str | None) -> str:
    robot1 = RobotDiscInfo(start=(10, 0), end=(-10, 0), radius=1)
    robot2 = RobotDiscInfo(start=(-10, 0), end=(10, 0), radius=1)
    obstacle = [-1, -1, -1, 1, 1, 1, 1, -1]
    bounding_box = BoxBounds(17, -17, -22, 22)
    output_string = get_scene_string([robot1, robot2], [obstacle], bounding_box)
    if filename:
        write_to_file(filename, output_string)
    return output_string


def get_scene_2(filename: str | None) -> str:
    robot1 = RobotDiscInfo(start=(4, 10), end=(36, 10), radius=1)
    robot2 = RobotDiscInfo(start=(4, 20), end=(36, 20), radius=1)
    robot3 = RobotDiscInfo(start=(4, 30), end=(36, 30), radius=1)
    obstacle = [-1, -1, -1, 1, 1, 1, 1, -1]
    bounding_box = BoxBounds(40, 0, 0, 40)
    output_string = get_scene_string([robot1, robot2, robot3], [obstacle], bounding_box)
    if filename:
        write_to_file(filename, output_string)
    return output_string


def get_scene_3(filename: str | None) -> str:
    robot1 = RobotDiscInfo(start=(20, 12), end=(20, 12), radius=1)
    robot2 = RobotDiscInfo(start=(20, 12), end=(20, 12), radius=1)
    robot3 = RobotDiscInfo(start=(20, 12), end=(20, 12), radius=1)
    robot4 = RobotDiscInfo(start=(20, 12), end=(20, 12), radius=1)
    obstacles = [
        [5, 5, 15, 5, 15, 6, 5, 6],
        [0, 11, 15, 11, 15, 12, 0, 12],
        [5, 18, 10, 18, 10, 23, 5, 23],
        [15, 18, 16, 18, 16, 30, 15, 30],
        [25, 0, 26, 0, 26, 6, 25, 6],
        [25, 12, 26, 12, 26, 30, 25, 30],
        [40, 0, 50, 0, 50, 5, 40, 5],
        [35, 18, 40, 18, 40, 23, 35, 23]
    ]
    bounding_box = BoxBounds(30, 0, 0, 50)
    output_string = get_scene_string([robot1, robot2, robot3, robot4], obstacles, bounding_box)
    if filename:
        write_to_file(filename, output_string)
    return output_string

if __name__ == '__main__':
    get_scene_3("scene3.json")

