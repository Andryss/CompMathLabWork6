from functions import *


class OrdinaryDifferentialEquation:
    string: str
    func: TwoVariableFunction

    def __init__(self, s, f):
        self.string = s
        self.func = f

    def at(self, x: float, y: float):
        return self.func.at(x, y)
