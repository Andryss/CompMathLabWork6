from typing import Callable


class Function:
    _string: str = ""
    _func: Callable[[float], float] = lambda x: 0

    def __init__(self, s, f):
        self._string = s
        self._func = f

    def at(self, x: float) -> float:
        return self._func(x)

    def __str__(self):
        return "function: (" + self._string + ")"


class TwoVariableFunction:
    _string: str = ""
    _func: Callable[[float, float], float] = lambda x, y: 0

    def __init__(self, s, f):
        self._string = s
        self._func = f

    def at(self, x: float, y: float) -> float:
        return self._func(x, y)

    def __str__(self):
        return "function: (" + self._string + ")"
