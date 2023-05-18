import math

from functions import *


class OrdinaryDifferentialEquation:
    func: TwoVariableFunction
    coefficient_extractor: TwoVariableFunction
    actual_solution: TwoVariableFunction

    def __init__(self, func, extr, sol):
        self.func = func
        self.coefficient_extractor = extr
        self.actual_solution = sol

    def at(self, x: float, y: float):
        return self.func.at(x, y)

    def answer_at(self, x: float, point: (float, float)):
        c = self.coefficient_extractor.at(point[0], point[1])
        return self.actual_solution.at(x, c)

    def __str__(self):
        return f"y' = {self.func.string()}"


def get_example_1_differential_equation() -> OrdinaryDifferentialEquation:
    return OrdinaryDifferentialEquation(        # interval [1,1.5] start_point (1,-1) step 0.1
        TwoVariableFunction(
            "y + (1 + x) * y^2",
            lambda x, y: y + (1 + x) * y**2
        ),
        TwoVariableFunction(
            "- e^x/y - x * e^x",
            lambda x, y: - math.exp(x) / y - math.exp(x) * x
        ),
        TwoVariableFunction(
            "- e^x/(c + x * e^x)",
            lambda x, c: - math.exp(x) / (c + math.exp(x) * x)
        )
    )


def get_all_differential_equations() -> list[OrdinaryDifferentialEquation]:
    return [
        get_example_1_differential_equation()
    ]
