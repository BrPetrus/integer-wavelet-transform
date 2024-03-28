import enum
import numpy as np
from typing import List
from numpy.typing import NDArray


class LSType(enum.Enum):
    UPDATE = enum.auto()
    PREDICT = enum.auto()


class LSBoundaryCondition(enum.Enum):
    ZERO_PADDING = enum.auto()


class LSStep:
    def __init__(self, ls_type: LSType, coefficients: List[float], max_order: int,
                 boundary_condition: LSBoundaryCondition):
        self.ls_type = ls_type
        self.coefficients = coefficients
        self.max_order = max_order
        self.boundary_condition = boundary_condition

    def evaluate(self, approx: NDArray[int], diff: NDArray[int], inverse:bool = False) -> NDArray[int]:
        approx = approx.copy()
        diff = diff.copy()
        n = approx.shape[0]
        c = -1 if inverse else 1
        num_c = len(self.coefficients)

        change = 0
        for i in reversed(range(num_c)):
            order = self.max_order - (num_c-i-1)

            padding = abs(self.max_order) + num_c +2
            extract_from = approx if self.ls_type == LSType.PREDICT else diff
            extract_from = np.pad(extract_from, (padding, padding))

            extract_from = np.roll(extract_from, -1 * order)
            extract = extract_from[padding:padding+n]

            change += extract*self.coefficients[i]

        if self.ls_type == LSType.PREDICT:
            approx, diff = approx, diff + np.floor(change+0.5).astype(int)*c
        else:
            approx, diff = approx + np.floor(change+0.5).astype(int)*c, diff
        return approx, diff



Wavelet = List[LSStep]
