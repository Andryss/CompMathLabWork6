import pandas as pd

from differential_equations import *


class DifferentialEquationInfo:
    equation: OrdinaryDifferentialEquation
    start_point: (float, float)     # (x_0, y_0)
    interval: (float, float)        # [x_0, x_n]
    step_size: float                # h
    precision: float                # eps


class SolveResultEntity:
    name: str
    approximated_values_table: pd.DataFrame


class SolveResult:
    src_equation: OrdinaryDifferentialEquation
    solving_results: list[SolveResultEntity]


def solve_equation(info: DifferentialEquationInfo) -> SolveResult:
    pass
