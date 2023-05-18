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


def get_all_differential_equations() -> list[OrdinaryDifferentialEquation]:
    return [

    ]
