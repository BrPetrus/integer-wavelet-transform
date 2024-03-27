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
        n = approx.shape[0]
        padding = len(self.coefficients)
        c = -1 if inverse else 1

        # TODO: Duplication
        if self.ls_type == LSType.PREDICT:
            predict = np.zeros_like(approx, dtype=float)
            for i in reversed(range(len(self.coefficients))):
                order = self.max_order - (len(self.coefficients)-i-1)
                print("predict")
                print("-------")
                print(f"[D] order {order}")
                print(f"[D] coefficient {self.coefficients[i]}")

                if order >= 0:
                    #if order < len(self.coefficients):
                    predict += self.coefficients[i] * np.pad(approx[order:], (0, order))
                else:
                    # TODO: bug if the signal is shorter than order
                    if order < len(self.coefficients):
                        predict += self.coefficients[i] * np.pad(approx[:order], (-1*order, 0))

            diff += np.floor(predict + 0.5).astype(int) * c
        elif self.ls_type == LSType.UPDATE:
            update = np.zeros_like(approx, dtype=float)
            for i in reversed(range(len(self.coefficients))):
                order = self.max_order - (len(self.coefficients)-i-1)
                print("update")
                print("-------")
                print(f"[D] order {order}")
                print(f"[D] coefficient {self.coefficients[i]}")
                if order >= 0:
                    #if order < len(self.coefficients):
                    update += self.coefficients[i] * np.pad(diff[order:], (0, order))
                else:
                    if order < len(self.coefficients):
                        update += self.coefficients[i] * np.pad(diff[:order],(-1*order, 0))
            approx += np.floor(update + 0.5).astype(int) * c
        return approx, diff


Wavelet = List[LSStep]
