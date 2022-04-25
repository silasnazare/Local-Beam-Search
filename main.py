import cv2
import numpy as np
from enum import Enum
from math import sqrt
from random import random, shuffle

# TODO Wrong evaluation calculation

# CONSTANTS
BLACK = False
WHITE = True
IMAGE_PATH = "images/10.png"

# HYPER PARAMETERS
MAX_SOLUTIONS = 16  # Maximum number of solutions stored in memory
EVALUATION_FUNCTION = "both"  # matching_pixels | turning_angle (NOT RECOMMENDED) | both
MATCHING_PIXELS_WEIGHT = 0.75  # Takes effect if EVALUATION_FUNCTION == "both"
TURNING_ANGLE_WEIGHT = 0.25  # Takes effect if EVALUATION_FUNCTION == "both"
SA_MAX_ITERATIONS = 128  # Max iteration limit for simulated annealing

OPERATIONS = [
    lambda i, j: (i - 1, j),  # UP
    lambda i, j: (i, j - 1),  # LEFT
    lambda i, j: (i - 1, j - 1),  # UP_LEFT
    lambda i, j: (i - 1, j + 1),  # UP_RIGHT
    lambda i, j: (i + 1, j - 1),  # DOWN_LEFT
    lambda i, j: (i + 1, j + 1),  # DOWN_RIGHT
    lambda i, j: (i, j + 1),  # RIGHT
    lambda i, j: (i + 1, j),  # DOWN
]


class Operation(Enum):
    UP = 0
    LEFT = 1
    UP_LEFT = 2
    UP_RIGHT = 3
    DOWN_LEFT = 4
    DOWN_RIGHT = 5
    RIGHT = 6
    DOWN = 7


class Solution:
    def __init__(self, i, j, matching_pixel_fitness, turning_angle_fitness, history=[]):
        self.i = i
        self.j = j
        self.mp = matching_pixel_fitness if EVALUATION_FUNCTION != 'turning_angle' else 0
        self.ta = turning_angle_fitness if EVALUATION_FUNCTION != 'matching_pixels' else 0
        self.history = history

    def get_fitness(self):
        return self.mp * MATCHING_PIXELS_WEIGHT + self.ta * TURNING_ANGLE_WEIGHT


def read_image(path):
    img = cv2.imread(path, 2)
    return rgb_to_binary(img)


def rgb_to_binary(arr):
    return np.where(arr > 127, WHITE, BLACK).astype(np.bool_)


def binary_to_rgb(arr):
    return np.where(arr == WHITE, 255, 0).astype(np.uint8)


def get_bounds(arr):
    shape = np.shape(arr)
    i = shape[0] - 1
    j = shape[1] - 1
    return i, j


def find_black_pixel_count(target):
    shape = np.shape(target)
    pixel_count = shape[0] * shape[1]
    white_pixels = np.count_nonzero(target)
    return pixel_count - white_pixels


def can_move(i, j, last_row, last_col):
    return False if i < 0 or j < 0 or i > last_row or j > last_col else True


def visited_before(i, j, history):
    _i, _j = (i, j)
    op_count = len(OPERATIONS)

    for x in range(len(history) - 1, -1, -1):
        op_id = op_count - history[x] - 1
        _i, _j = OPERATIONS[op_id](_i, _j)
        if _i == i and _j == j:
            return True

    return False


def turning_angle(current_fitness, prev_op, next_op, op_count):
    if prev_op is None or op_count == 0:
        return 1.0
    i1, j1 = OPERATIONS[prev_op](1, 1)
    i2, j2 = OPERATIONS[next_op](1, 1)
    closeness = -sqrt(((i1 - i2) ** 2) + ((j1 - j2) ** 2))  # range(-2 , 0)
    return (current_fitness * op_count + (closeness + 2) / 2) / (op_count + 1)  # Scaled into (0, 1)


def matching_pixels(current_fitness, target, i, j, black_pixels):
    if target[i, j] == BLACK:
        return current_fitness + 1 / black_pixels
    elif current_fitness > 0:
        return current_fitness - 1 / black_pixels
    else:
        return 0


def evaluate(solution, target, i, j, black_pixels, next_op):
    prev_op = solution.history[-1] if len(solution.history) > 0 else None
    ta = solution.ta
    mp = solution.mp
    if EVALUATION_FUNCTION == "matching_pixels":
        return matching_pixels(mp, target, i, j, black_pixels), None
    elif EVALUATION_FUNCTION == "turning_angle":
        return None, turning_angle(ta, prev_op, next_op, len(solution.history))
    return matching_pixels(mp, target, i, j, black_pixels), turning_angle(ta, prev_op, next_op, len(solution.history))


def generate_solutions(solution, possible_solutions, target, last_row, last_col, black_pixels):
    op_ids = list(range(8))
    shuffle(op_ids)

    for op_id in op_ids:
        i, j = OPERATIONS[op_id](solution.i, solution.j)
        if can_move(i, j, last_row, last_col) and not visited_before(i, j, solution.history):
            mp, ta = evaluate(solution, target, i, j, black_pixels, op_id)
            history = solution.history.copy()
            history.append(op_id)

            new_solution = Solution(i, j, mp, ta, history)
            possible_solutions.append(new_solution)


def should_select(fitness, iteration_number):
    value = -iteration_number
    min_value = -SA_MAX_ITERATIONS
    max_value = 0
    normalized_value = (value - min_value) / (max_value - min_value)
    probability = fitness * 0.25 + normalized_value * 0.75
    return probability > random()


def select_new_solutions(possible_solutions, current_solutions, iteration_number):
    possible_solutions.sort(key=lambda x: x.get_fitness(), reverse=True)
    min_fitness_index = current_solutions.index(min(current_solutions, key=lambda x: x.get_fitness()))
    has_change = False

    for solution in possible_solutions:
        fitness = solution.get_fitness()

        if len(current_solutions) < MAX_SOLUTIONS:
            current_solutions.append(solution)
            min_fitness_index = current_solutions.index(min(current_solutions, key=lambda x: x.get_fitness()))
            has_change = True

        elif fitness > current_solutions[min_fitness_index].get_fitness():
            current_solutions[min_fitness_index] = solution
            min_fitness_index = current_solutions.index(min(current_solutions, key=lambda x: x.get_fitness()))
            has_change = True

        elif iteration_number < SA_MAX_ITERATIONS:
            if should_select(fitness, iteration_number):
                current_solutions[min_fitness_index] = solution
                min_fitness_index = current_solutions.index(min(current_solutions, key=lambda x: x.get_fitness()))
                has_change = True

    return has_change


def local_beam_search(target):
    last_row, last_col = get_bounds(target)
    black_pixels = find_black_pixel_count(target)
    initial_mp = matching_pixels(0, target, last_row, 0, black_pixels)
    solution = Solution(last_row, 0, initial_mp, 1.0)
    current_solutions = [solution]
    more_solutions = True
    iteration_number = 0

    while more_solutions:
        possible_solutions = []

        for solution in current_solutions:
            generate_solutions(solution, possible_solutions, target, last_row, last_col, black_pixels)

        more_solutions = select_new_solutions(possible_solutions, current_solutions, iteration_number)
        iteration_number += 1

    return max(current_solutions, key=lambda x: x.get_fitness())


def show_report(history, target):
    last_row, last_col = get_bounds(target)
    arr = np.ones((last_row + 1, last_col + 1)).astype(np.bool_)
    i = last_row
    j = 0
    arr[i, j] = BLACK

    for op_id in history:
        print(Operation(op_id).name)
        i, j = OPERATIONS[op_id](i, j)
        arr[i, j] = BLACK

    img = binary_to_rgb(arr)
    cv2.namedWindow("Target", cv2.WINDOW_NORMAL)
    cv2.imshow("Target", binary_to_rgb(target))
    cv2.namedWindow("Solution", cv2.WINDOW_NORMAL)
    cv2.imshow("Solution", img)
    print(best_solution.get_fitness())
    cv2.waitKey()


if __name__ == "__main__":
    target_arr = read_image(IMAGE_PATH)
    best_solution = local_beam_search(target_arr)
    show_report(best_solution.history, target_arr)
