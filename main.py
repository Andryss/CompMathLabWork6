import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

from solvers import *

warnings.filterwarnings("ignore")


def read_int_from_console(number_name: str) -> int:
    print(f"\nEnter {number_name}:")
    line = input().strip()
    try:
        return int(line)
    except Exception as e:
        raise Exception(f"Can't read int value: {e.__str__()}")


def read_float_from_console(number_name: str) -> float:
    print(f"\nEnter {number_name}:")
    line = input().strip()
    try:
        return float(line)
    except Exception as e:
        raise Exception(f"Can't read float value: {e.__str__()}")


def read_differential_equation() -> OrdinaryDifferentialEquation:
    print("\nChoose the differential equation:")
    differential_equations = get_all_differential_equations()
    for i, equation in enumerate(differential_equations):
        print(f"{i}\t{equation}")
    try:
        line = input("(enter the number) ").strip()
        index = int(line)
        return differential_equations[index]
    except Exception as e:
        raise Exception(f"can't read differential equation: {e.__str__()}")


def read_interval() -> [float, float]:
    line = input(f"\nEnter the interval boundaries:\n").strip()
    interval: [float, float]
    try:
        interval = [float(x) for x in line.split()]
        if len(interval) != 2 or interval[1] < interval[0]:
            raise Exception("not an interval")
        return interval
    except Exception as e:
        raise Exception(f"can't read interval: {e.__str__()}")


def read_start_point_value() -> float:
    return read_float_from_console("function value in the left most interval point (start value)")


def read_step_size() -> float:
    return read_float_from_console("step size")


def read_precision() -> float:
    return read_float_from_console("precision")


def print_result_entity(micro_result: SolveResultEntity):
    if isinstance(micro_result, SolveResultEntitySuccess):
        print(f"\n{micro_result.name}")
        table_result = micro_result.approximated_values_table
        print(f"{table_result.table().T}")
        print(f"with step: {table_result.x_values()[1] - table_result.x_values()[0]}")
    elif isinstance(micro_result, SolveResultEntityError):
        print(f"\n{micro_result.name}")
        print(f"{micro_result.error.__str__()}")


def read_bool(text: str) -> bool:
    line = input(f"{text} [Y/y for YES, NO otherwise]: ").strip().lower()
    if line == 'y':
        return True
    else:
        return False


def read_show_all() -> bool:
    return read_bool("\nDo you want to see all possible plots?")


def show_result_entity(micro_result: SolveResultEntity, info: DifferentialEquationInfo):
    if isinstance(micro_result, SolveResultEntitySuccess):
        fig, ax = plt.subplots()

        table = micro_result.approximated_values_table
        ax.scatter(table.x_values(), table.y_values(), c='blue', label=micro_result.name)

        x_vals = np.linspace(info.interval[0], info.interval[1], 20)
        y_vals = np.vectorize(lambda x: info.equation.answer_at(x, info.start_point))(x_vals)

        ax.scatter(x_vals, y_vals, c='red', label='real function')

        ax.set_title(micro_result.name)
        ax.legend()
        fig.show()


def print_result(result: SolveResult, info: DifferentialEquationInfo):
    print("\nHere is the computation result:")
    successful_entity_count = 0
    for micro_result in result.solving_results:
        if isinstance(micro_result, SolveResultEntitySuccess):
            successful_entity_count += 1
        print_result_entity(micro_result)
    if successful_entity_count > 0:
        if read_show_all():
            for micro_result in result.solving_results:
                show_result_entity(micro_result, info)


def run():
    try:
        equation_info = DifferentialEquationInfo()
        equation_info.equation = read_differential_equation()
        equation_info.interval = read_interval()
        equation_info.start_point = (equation_info.interval[0], read_start_point_value())
        equation_info.step_size = read_step_size()

        steps_count = (equation_info.interval[1] - equation_info.interval[0]) / equation_info.step_size
        if steps_count > 50_000:
            raise Exception(f"too many steps ({steps_count})")

        equation_info.precision = read_precision()
        result = solve_equation(equation_info)
        print_result(result, equation_info)
    except Exception as e:
        print(f'Some error occurred: {e}', file=sys.stderr)


if __name__ == '__main__':
    run()
