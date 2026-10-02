from typing import Any, Protocol

import numpy as np


class ClusterAlgorithm(Protocol):
    """
    Protocol for clustering algorithms supporting .fit() and .labels_.
    Matches sklearn clusterers and custom implementations.
    """

    labels_: np.ndarray

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> Any: ...
