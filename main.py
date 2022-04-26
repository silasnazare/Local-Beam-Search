import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from math import sqrt
from random import random, shuffle

# CONSTANTS
BLACK = False
WHITE = True
MATCHING_PIXELS_WEIGHT = 0.75  # Takes effect if evaluation_function == "matching_pixels_and_turning_angle"
TURNING_ANGLE_WEIGHT = 0.25  # Takes effect if evaluation_function == "matching_pixels_and_turning_angle"

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
    def __init__(self, i, j, matching_pixel_fitness, turning_angle_fitness, evaluation_function, history=[]):
        self.i = i
        self.j = j
        self.mp = matching_pixel_fitness if evaluation_function != 'turning_angle' else 0
        self.ta = turning_angle_fitness if evaluation_function != 'matching_pixels' else 0
        self.history = history

    def get_fitness(self):
        return self.mp * MATCHING_PIXELS_WEIGHT + self.ta * TURNING_ANGLE_WEIGHT


class Report:
    def __init__(self, max_solutions, evaluation_function, sa_max_iterations, target):
        self.max_solutions = max_solutions
        self.evaluation_function = evaluation_function
        self.sa_max_iterations = sa_max_iterations
        self.target = target
        self.generation_count = 0
        self.best_solutions_for_each_generation = []


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


def evaluate(solution, target, i, j, black_pixels, next_op, evaluation_function):
    prev_op = solution.history[-1] if len(solution.history) > 0 else None
    ta = solution.ta
    mp = solution.mp
    if evaluation_function == "matching_pixels":
        return matching_pixels(mp, target, i, j, black_pixels), None
    elif evaluation_function == "matching_pixels_and_turning_angle":
        return matching_pixels(mp, target, i, j, black_pixels), turning_angle(ta, prev_op, next_op, len(solution.history))
    return matching_pixels(mp, target, i, j, black_pixels), turning_angle(ta, prev_op, next_op, len(solution.history))


def should_select(fitness, iteration_number, sa_max_iterations):
    value = -iteration_number
    min_value = -sa_max_iterations
    max_value = 0
    normalized_value = (value - min_value) / (max_value - min_value)
    probability = fitness * 0.25 + normalized_value * 0.75
    return probability > random()


def generate_solutions(solution, possible_solutions, target, last_row, last_col, black_pixels, evaluation_function):
    op_ids = list(range(8))
    shuffle(op_ids)

    for op_id in op_ids:
        i, j = OPERATIONS[op_id](solution.i, solution.j)
        if can_move(i, j, last_row, last_col) and not visited_before(i, j, solution.history):
            mp, ta = evaluate(solution, target, i, j, black_pixels, op_id, evaluation_function)
            history = solution.history.copy()
            history.append(op_id)

            new_solution = Solution(i, j, mp, ta, evaluation_function, history)
            possible_solutions.append(new_solution)


def select_new_solutions(possible_solutions, current_solutions, iteration_number, max_solutions, sa_max_iterations):
    possible_solutions.sort(key=lambda x: x.get_fitness(), reverse=True)
    min_fitness_index = current_solutions.index(min(current_solutions, key=lambda x: x.get_fitness()))
    has_change = False

    for solution in possible_solutions:
        fitness = solution.get_fitness()

        if len(current_solutions) < max_solutions:
            current_solutions.append(solution)
            min_fitness_index = current_solutions.index(min(current_solutions, key=lambda x: x.get_fitness()))
            has_change = True

        elif fitness > current_solutions[min_fitness_index].get_fitness():
            current_solutions[min_fitness_index] = solution
            min_fitness_index = current_solutions.index(min(current_solutions, key=lambda x: x.get_fitness()))
            has_change = True

        elif iteration_number < sa_max_iterations:
            if should_select(fitness, iteration_number, sa_max_iterations):
                current_solutions[min_fitness_index] = solution
                min_fitness_index = current_solutions.index(min(current_solutions, key=lambda x: x.get_fitness()))
                has_change = True

    return has_change


def local_beam_search(target, max_solutions, evaluation_function, sa_max_iterations):
    report = Report(max_solutions, evaluation_function, sa_max_iterations, target)
    last_row, last_col = get_bounds(target)
    black_pixels = find_black_pixel_count(target)
    initial_mp = matching_pixels(0, target, last_row, 0, black_pixels)
    solution = Solution(last_row, 0, initial_mp, 1.0, evaluation_function)
    current_solutions = [solution]
    more_solutions = True
    iteration_number = 0

    while more_solutions:
        possible_solutions = []

        for solution in current_solutions:
            generate_solutions(solution, possible_solutions, target, last_row, last_col, black_pixels, evaluation_function)

        more_solutions = select_new_solutions(possible_solutions, current_solutions, iteration_number, max_solutions, sa_max_iterations)
        best_solution_of_generation = max(current_solutions, key=lambda x: x.get_fitness())
        report.best_solutions_for_each_generation.append(best_solution_of_generation)
        iteration_number += 1

    report.generation_count = iteration_number

    return max(current_solutions, key=lambda x: x.get_fitness()), report


def show_image(img, title, wait=True):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    if title != "Target":
        cv2.moveWindow(title, 480, 128)
    else:
        cv2.moveWindow(title, 64, 128)

    cv2.imshow(title, img)

    if wait:
        cv2.waitKey()


def show_solution(solution, title, bounds):
    history = solution.history
    last_row, last_col = bounds
    arr = np.ones((last_row + 1, last_col + 1)).astype(np.bool_)
    i = last_row
    j = 0
    arr[i, j] = BLACK

    for op_id in history:
        i, j = OPERATIONS[op_id](i, j)
        arr[i, j] = BLACK

    show_image(binary_to_rgb(arr), title)


def print_report(report, show_images=True):
    print("---------------- REPORT ---------------")
    print("Hyper Parameters:")
    print(f"* Evaluation Function = {report.evaluation_function}")
    print(f"* Max Solutions in Memory | Population Size = {report.max_solutions}")
    print(f"* Max Iterations for Simulated Annealing = {report.sa_max_iterations}")
    print("")
    print("Results:")
    print(f"* Generation Count: {report.generation_count}")
    print(f"* Fitness of Best Solution: {report.best_solutions_for_each_generation[-1]}")
    print("---------------------------------------")
    print("")
    print("")

    if show_images:
        show_image(binary_to_rgb(report.target), "Target", wait=False)
        i = 1
        for solution in report.best_solutions_for_each_generation:
            title = f"Best Solution (Each Step)"
            show_solution(solution, title, get_bounds(report.target))
            i += 1


def run_all(evaluation_functions_list, max_solutions_list, sa_max_iterations_list, images_count):
    best_solutions = []
    reports = []

    for evaluation_function in evaluation_functions_list:
        for max_solutions in max_solutions_list:
            for sa_max_iterations in sa_max_iterations_list:
                for image_number in range(images_count):
                    target_arr = read_image(f"images/{image_number}.png")
                    best_solution, report = local_beam_search(target_arr, max_solutions, evaluation_function, sa_max_iterations)
                    best_solutions.append(best_solution)
                    reports.append(report)

    return best_solutions, reports


def plot_effect_of_evaluation_function(reports, evaluation_functions_list):
    split_index = len(reports) // 2
    results_1 = reports[0: split_index]
    results_2 = reports[split_index:]

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(len(results_1)):
        item = results_1[i].best_solutions_for_each_generation[-1].get_fitness()
        x1.append(i)
        y1.append(item)

    for i in range(len(results_2)):
        item = results_2[i].best_solutions_for_each_generation[-1].get_fitness()
        x2.append(i)
        y2.append(item)

    plt.plot(x1, y1, label=evaluation_functions_list[0])
    plt.plot(x2, y2, label=evaluation_functions_list[1])
    plt.xlabel("Different Images")
    plt.ylabel("Fitness of Best Solution")
    plt.legend()
    plt.show()


def plot_effect_of_max_solutions(reports, images_count):
    groups = {}

    for report in reports:
        item = report.best_solutions_for_each_generation[-1].get_fitness()
        if report.max_solutions not in groups:
            groups[report.max_solutions] = [item]
        else:
            groups[report.max_solutions].append(item)

    images_range = list(range(images_count))

    for key, value in groups.items():
        plt.plot(images_range, value, label=f"Max Solutions = {key}")

    plt.xlabel("Different Images")
    plt.ylabel("Fitness of Best Solution")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def plot_effect_of_sa_max_iterations(reports, images_count):
    groups = {}

    for report in reports:
        item = report.best_solutions_for_each_generation[-1].get_fitness()
        if report.sa_max_iterations not in groups:
            groups[report.sa_max_iterations] = [item]
        else:
            groups[report.sa_max_iterations].append(item)

    images_range = list(range(images_count))

    for key, value in groups.items():
        plt.plot(images_range, value, label=f"SA Max Iterations = {key}")

    plt.xlabel("Different Images")
    plt.ylabel("Fitness of Best Solution")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    _evaluation_functions_list = ["matching_pixels_and_turning_angle"]
    _max_solutions_list = [128]
    _sa_max_iterations_list = [128]
    _images_count = 10

    _best_solutions, _reports = run_all(_evaluation_functions_list, _max_solutions_list, _sa_max_iterations_list, _images_count)

    for _report in _reports:
        print_report(_report, show_images=True)

    if len(_evaluation_functions_list) == 2:
        plot_effect_of_evaluation_function(_reports, _evaluation_functions_list)
    elif len(_max_solutions_list) > 1:
        plot_effect_of_max_solutions(_reports, _images_count)
    elif len(_sa_max_iterations_list) > 1:
        plot_effect_of_sa_max_iterations(_reports, _images_count)
