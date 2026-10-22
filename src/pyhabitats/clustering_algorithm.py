from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ClusteringAlgorithm(Protocol):
    """
    Protocol for clustering algorithms supporting .fit() method and .labels_
    property.
    """

    labels_: np.ndarray

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> Any: ...
