import math

from differential_equations import *


class DifferentialEquationInfo:
    equation: OrdinaryDifferentialEquation
    start_point: (float, float)     # (x_0, y_0)
    interval: (float, float)        # [x_0, x_n]
    step_size: float                # h
    precision: float                # eps


class SolveResultEntity:
    name: str


class SolveResultEntitySuccess(SolveResultEntity):
    approximated_values_table: TableFunction


class SolveResultEntityError(SolveResultEntity):
    error: Exception


class SolveResult:
    src_equation: OrdinaryDifferentialEquation
    solving_results: list[SolveResultEntity]


class DifferentialEquationSolver:
    name: str
    p: float

    def solve(self, equation: OrdinaryDifferentialEquation, start_point: (float, float),
              interval: (float, float), step_size: float) -> SolveResultEntity:
        raise Exception("method isn't overridden")


class SequentialDifferentialEquationSolver(DifferentialEquationSolver):

    def get_next_values(self, x_vals: list[float], y_vals: list[float],
                        step_size: float, equation: OrdinaryDifferentialEquation) -> (float, float):
        raise Exception("method isn't overridden")

    def solve(self, equation: OrdinaryDifferentialEquation, start_point: (float, float),
              interval: (float, float), step_size: float) -> SolveResultEntity:
        try:
            x_vals, y_vals = [start_point[0]], [start_point[1]]
            n = math.floor(round((interval[1] - interval[0]) / step_size, 3))

            pointer = start_point[0]
            for i in range(n):
                pointer += step_size
                x_new, y_new = self.get_next_values(x_vals, y_vals, step_size, equation)
                x_vals.append(x_new)
                y_vals.append(y_new)

            result = SolveResultEntitySuccess()
            result.name = self.name
            result.approximated_values_table = TableFunction(pd.DataFrame({
                'x': x_vals,
                'y': y_vals
            }))
            return result
        except Exception as e:
            result = SolveResultEntityError()
            result.name = "error " + self.name
            result.error = e
            return result


class NewtonSolver(SequentialDifferentialEquationSolver):
    name = "newton solver"
    p = 1

    def get_next_values(self, x_vals: list[float], y_vals: list[float],
                        step_size: float, equation: OrdinaryDifferentialEquation) -> (float, float):
        x_new = x_vals[-1] + step_size
        y_new = y_vals[-1] + step_size * equation.at(x_vals[-1], y_vals[-1])
        return x_new, y_new


class ModifiedNewtonSolver(SequentialDifferentialEquationSolver):
    name = "modified newton solver"
    p = 2

    def get_next_values(self, x_vals: list[float], y_vals: list[float],
                        step_size: float, equation: OrdinaryDifferentialEquation) -> (float, float):
        x_new = x_vals[-1] + step_size
        y_derivation = equation.at(x_vals[-1], y_vals[-1])
        y_new_tmp = y_vals[-1] + step_size * y_derivation
        y_new = y_vals[-1] + step_size / 2 * (y_derivation + equation.at(x_vals[-1], y_new_tmp))
        return x_new, y_new


class RungeKuttaSolver(SequentialDifferentialEquationSolver):
    name = "runge kutta solver"
    p = 4

    def get_next_values(self, x_vals: list[float], y_vals: list[float],
                        step_size: float, equation: OrdinaryDifferentialEquation) -> (float, float):
        x_new = x_vals[-1] + step_size
        k_1 = step_size * equation.at(x_vals[-1], y_vals[-1])
        k_2 = step_size * equation.at(x_vals[-1] + step_size / 2, y_vals[-1] + k_1 / 2)
        k_3 = step_size * equation.at(x_vals[-1] + step_size / 2, y_vals[-1] + k_2 / 2)
        k_4 = step_size * equation.at(x_vals[-1] + step_size, y_vals[-1] + k_3)
        y_new = y_vals[-1] + 1/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)
        return x_new, y_new


class RungeRuleSolver:

    @staticmethod
    def get_precision(first: SolveResultEntitySuccess, second: SolveResultEntitySuccess, p: float) -> float:
        first_values, second_values = first.approximated_values_table.y_values(), \
                                      second.approximated_values_table.y_values()
        precision = (first_values[1] - first_values[2]) / (2 ** p - 1)
        return precision

    @staticmethod
    def solve_with_precision(solver: SequentialDifferentialEquationSolver, info: DifferentialEquationInfo,
                             max_iterations: int = 100) -> SolveResultEntity:
        start_step_size = info.step_size
        prev_result = solver.solve(info.equation, info.start_point, info.interval, start_step_size)
        if not isinstance(prev_result, SolveResultEntitySuccess):
            return prev_result

        for i in range(max_iterations):
            start_step_size /= 2
            cur_result = solver.solve(info.equation, info.start_point, info.interval, start_step_size)
            if not isinstance(cur_result, SolveResultEntitySuccess):
                return cur_result
            cur_precision = RungeRuleSolver.get_precision(prev_result, cur_result, solver.p)
            if cur_precision < info.precision:
                return cur_result
            prev_result = cur_result

        result = SolveResultEntityError()
        result.name = "error " + solver.name
        result.error = Exception(f"can't reach precision {info.precision}: "
                                 f"max number of iterations ({max_iterations}) reached")
        return result


def get_all_differential_equation_solvers() -> list[DifferentialEquationSolver]:
    return [
        NewtonSolver(),
        ModifiedNewtonSolver(),
        RungeKuttaSolver()
    ]


def solve_equation(info: DifferentialEquationInfo) -> SolveResult:
    result = SolveResult()
    result.src_equation = info
    results = []
    for solver in get_all_differential_equation_solvers():
        if isinstance(solver, SequentialDifferentialEquationSolver):
            results.append(RungeRuleSolver.solve_with_precision(solver, info))
        else:
            raise Exception("faced with unusual type of solver")
    return result
