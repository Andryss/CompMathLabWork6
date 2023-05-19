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
    return OrdinaryDifferentialEquation(
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


def get_example_2_differential_equation() -> OrdinaryDifferentialEquation:
    return OrdinaryDifferentialEquation(
        TwoVariableFunction(
            "-2y",
            lambda x, y: -2*y
        ),
        TwoVariableFunction(
            "y / e^(-2x)",
            lambda x, y: y / math.exp(-2*x)
        ),
        TwoVariableFunction(
            "c * e^(-2x)",
            lambda x, c: c * math.exp(-2*x)
        )
    )


def get_example_3_differential_equation() -> OrdinaryDifferentialEquation:
    return OrdinaryDifferentialEquation(
        TwoVariableFunction(
            "-y/x",
            lambda x, y: -y/x
        ),
        TwoVariableFunction(
            "x*y",
            lambda x, y: x * y
        ),
        TwoVariableFunction(
            "c/x",
            lambda x, c: c/x
        )
    )


def get_example_4_differential_equation() -> OrdinaryDifferentialEquation:
    return OrdinaryDifferentialEquation(
        TwoVariableFunction(
            "(x - y)^2 + 1",
            lambda x, y: (x - y)**2 + 1
        ),
        TwoVariableFunction(
            "1/(y - x) + x",
            lambda x, y: 1/(y - x) + x
        ),
        TwoVariableFunction(
            "1/(c - x) + x",
            lambda x, c: 1/(c - x) + x
        )
    )


def get_all_differential_equations() -> list[OrdinaryDifferentialEquation]:
    return [
        get_example_1_differential_equation(),
        get_example_2_differential_equation(),
        get_example_3_differential_equation(),
        get_example_4_differential_equation()
    ]
