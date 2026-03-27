from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TSPInstance:
    name: str
    dimension: int
    coordinates: np.ndarray   # shape (n, 2); None for non-coordinate instances
    distance_matrix: np.ndarray  # shape (n, n)


class InstanceLoader(ABC):
    """Interface for loading TSP instances from an arbitrary source."""

    @abstractmethod
    def load(self, source: str) -> TSPInstance:
        """Load a TSP instance from *source* (file path, URL, …)."""
        ...
