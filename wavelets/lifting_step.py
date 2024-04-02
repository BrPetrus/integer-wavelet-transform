import enum
from typing import List

import numpy as np
from numpy.typing import NDArray


class LSType(enum.Enum):
    UPDATE = enum.auto()
    PREDICT = enum.auto()


class LSBoundaryCondition(enum.Enum):
    ZERO_PADDING = enum.auto()


class LSStep:
    def __init__(self, ls_type: LSType, coefficients: List[float],
                 max_order: int):
        self.ls_type = ls_type
        self.coefficients = coefficients
        self.max_order = max_order

    def evaluate(self, approx: NDArray[int], diff: NDArray[int],
                 inverse: bool = False,
                 boundary_condition: LSBoundaryCondition = LSBoundaryCondition.ZERO_PADDING) -> \
            NDArray[int]:
        if boundary_condition != LSBoundaryCondition.ZERO_PADDING:
            raise NotImplementedError(
                "The boundary condition is not implemented")

        approx = approx.copy()
        diff = diff.copy()
        n = approx.shape[0]
        c = -1 if inverse else 1
        num_c = len(self.coefficients)
        # assert approx.dtype == diff.dtype
        # TODO: Investigate this assert

        change = np.zeros_like(approx, dtype=float)
        for i in reversed(range(num_c)):
            order = self.max_order - (num_c - i - 1)

            padding = abs(self.max_order) + num_c
            extract_from = approx if self.ls_type == LSType.PREDICT else diff
            extract_from = np.pad(extract_from, (padding, padding))

            extract_from = np.roll(extract_from, -1 * order)
            extract = extract_from[padding:n+padding]

            change += extract * self.coefficients[i]

        change = np.floor((change + 0.5))*c
        if self.ls_type == LSType.PREDICT:
            diff = diff + change.astype(int)
        else:
            approx = approx + change.astype(int)
        return approx, diff


Wavelet = List[LSStep]
