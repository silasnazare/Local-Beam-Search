import cv2
import numpy as np
from enum import Enum

# TODO: Minimize turning angles
# TODO: Alternative fitness function

BLACK = False
WHITE = True
LIMIT = 16  # Maximum number of solutions for each step


class Operation(Enum):
    UP = 0
    LEFT = 1
    UP_LEFT = 2
    UP_RIGHT = 3
    DOWN_LEFT = 4
    DOWN_RIGHT = 5
    RIGHT = 6
    DOWN = 7


operations = [
    lambda i, j: (i - 1, j),  # UP
    lambda i, j: (i, j - 1),  # LEFT
    lambda i, j: (i - 1, j - 1),  # UP_LEFT
    lambda i, j: (i - 1, j + 1),  # UP_RIGHT
    lambda i, j: (i + 1, j - 1),  # DOWN_LEFT
    lambda i, j: (i + 1, j + 1),  # DOWN_RIGHT
    lambda i, j: (i, j + 1),  # RIGHT
    lambda i, j: (i + 1, j),  # DOWN
]


class Solution:
    def __init__(self, i, j, fitness, history=[]):
        self.i = i
        self.j = j
        self.fitness = fitness
        self.history = history


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


def can_move(i, j, last_row, last_col):
    return False if i < 0 or j < 0 or i > last_row or j > last_col else True


def visited_before(i, j, history):
    _i, _j = (i, j)
    op_count = len(operations)

    for x in range(len(history) - 1, -1, -1):
        op_id = op_count - history[x] - 1
        _i, _j = operations[op_id](_i, _j)
        if _i == i and _j == j:
            return True

    return False


def find_fitness(current_fitness, target, i, j):
    if target[i, j] == BLACK:
        return current_fitness + 1
    elif current_fitness > 0:
        return current_fitness - 1
    else:
        return 0


def generate_solutions(solution, possible_solutions, target, last_row, last_col):
    for op_id in range(8):
        i, j = operations[op_id](solution.i, solution.j)
        if can_move(i, j, last_row, last_col) and not visited_before(i, j, solution.history):
            fitness = find_fitness(solution.fitness, target, i, j)
            history = solution.history.copy()
            history.append(op_id)

            new_solution = Solution(i, j, fitness, history)
            possible_solutions.append(new_solution)


def select_new_solutions(possible_solutions, current_solutions):
    possible_solutions.sort(key=lambda x: x.fitness, reverse=True)
    min_fitness_index = current_solutions.index(min(current_solutions, key=lambda x: x.fitness))
    has_change = False

    for solution in possible_solutions:
        if len(current_solutions) < LIMIT:
            current_solutions.append(solution)
            min_fitness_index = current_solutions.index(min(current_solutions, key=lambda x: x.fitness))
            has_change = True
        elif solution.fitness > current_solutions[min_fitness_index].fitness:  # TODO: Maybe greater or equal
            current_solutions[min_fitness_index] = solution
            min_fitness_index = current_solutions.index(min(current_solutions, key=lambda x: x.fitness))
            has_change = True

    return has_change


def local_beam_search(target):
    last_row, last_col = get_bounds(target)
    solution = Solution(last_row, 0, 0.0)
    current_solutions = [solution]
    more_solutions = True

    while more_solutions:
        possible_solutions = []

        for solution in current_solutions:
            generate_solutions(solution, possible_solutions, target, last_row, last_col)

        more_solutions = select_new_solutions(possible_solutions, current_solutions)

    return max(current_solutions, key=lambda x: x.fitness)


def show_report(history, target):
    last_row, last_col = get_bounds(target)
    arr = np.ones((last_row + 1, last_col + 1)).astype(np.bool_)
    i = last_row
    j = 0
    arr[i, j] = BLACK

    for op_id in history:
        print(Operation(op_id).name)
        i, j = operations[op_id](i, j)
        arr[i, j] = BLACK

    img = binary_to_rgb(arr)
    cv2.namedWindow("Target", cv2.WINDOW_NORMAL)
    cv2.imshow("Target", binary_to_rgb(target))
    cv2.namedWindow("Solution", cv2.WINDOW_NORMAL)
    cv2.imshow("Solution", img)
    print(best_solution.fitness)
    cv2.waitKey()


if __name__ == "__main__":
    target_arr = read_image("images/1.png")
    best_solution = local_beam_search(target_arr)
    show_report(best_solution.history, target_arr)
