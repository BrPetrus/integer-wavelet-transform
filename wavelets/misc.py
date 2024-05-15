from typing import List, Tuple, TypeVar
import numpy as np
from numpy.typing import NDArray

T = TypeVar("T", bound=np.generic, covariant=True)
Config = List[Tuple[int, int]]
Decomposition = Tuple[List[NDArray[T]], Config]


class TransformError(RuntimeError):
    """Exception class for errors during """
    pass
