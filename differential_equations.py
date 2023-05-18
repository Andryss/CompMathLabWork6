from functions import *


class OrdinaryDifferentialEquation:
    func: TwoVariableFunction
    actual_solution: Function

    def __init__(self, f, s):
        self.func = f
        self.actual_solution = s

    def at(self, x: float, y: float):
        return self.func.at(x, y)

    def answer_at(self, x: float):
        return self.actual_solution.at(x)

    def __str__(self):
        return f"y' = {self.func.string()}"


def get_example_1_differential_equation() -> OrdinaryDifferentialEquation:
    return OrdinaryDifferentialEquation(
        TwoVariableFunction(
            "y + (1 + x) * y^2",
            lambda x, y: y + (1 + x) * y**2
        ),
        Function(
            "-1/x",
            lambda x: -1/x
        )
    )


def get_all_differential_equations() -> list[OrdinaryDifferentialEquation]:
    return [
        get_example_1_differential_equation()
    ]
