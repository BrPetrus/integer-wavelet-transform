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

    def evaluate(self, approx: NDArray[int], diff: NDArray[int]) -> NDArray[int]:
        n = approx.shape[0]
        padding = abs(self.max_order)

        # TODO: Duplication
        if self.ls_type == LSType.PREDICT:
            if self.boundary_condition == LSBoundaryCondition.ZERO_PADDING:
                padded_approx = np.pad(approx, (padding, padding), 'constant',
                                       constant_values=0)
            else:
                raise NotImplementedError(
                    f"Unknown boundary condition {self.ls_type}")
            predict = 0
            for i in reversed(range(len(self.coefficients))):
                order = self.max_order - (len(self.coefficients)-i-1)
                print("predict")
                print("-------")
                print(f"[D] order {order}")
                print(f"[D] coefficient {self.coefficients[i]}")
                print(f"[D] padded approx {padded_approx[padding+order:padding+order+n]}")
                predict += self.coefficients[i] * padded_approx[padding+order:padding+order+n]
            diff += np.floor(predict + 0.5).astype(int)
        elif self.ls_type == LSType.UPDATE:
            if self.boundary_condition == LSBoundaryCondition.ZERO_PADDING:
                padded_diff = np.pad(diff, (padding, padding), 'constant',
                                       constant_values=0)
            else:
                raise NotImplementedError(
                    f"Unknown boundary condition {self.ls_type}")

            update = 0
            for i in reversed(range(len(self.coefficients))):
                order = self.max_order - (len(self.coefficients)-i-1)
                print("update")
                print("-------")
                print(f"[D] order {order}")
                print(f"[D] coefficient {self.coefficients[i]}")
                print(f"[D] padded diff {padded_diff[padding+order:padding+order+n]}")
                update += self.coefficients[i] * padded_diff[padding+order:padding+order+n]
            approx += np.floor(update + 0.5).astype(int)
        return approx, diff


Wavelet = List[LSStep]
