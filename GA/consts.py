from typing import Union

import numpy as np

# TODO: where are we using this?
supported_int_types = Union[
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]
supported_float_types = Union[float, np.float16, np.float32, np.float64]