import numpy as np
import numba
import tsplib95

from data.loader import InstanceLoader, TSPInstance


@numba.njit(cache=True)
def _build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix via Numba JIT — O(n²) inner loop."""
    n = coords.shape[0]
    dist = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dist[i, j] = np.sqrt(dx * dx + dy * dy)
    return dist


class TSPLibReader(InstanceLoader):
    """Load TSP instances in TSPLIB95 format from a file path string.

    *source* is currently treated as a local file path.  Future subclasses or
    an updated implementation may resolve URLs or other schemes before passing
    the content to tsplib95.
    """

    def load(self, source: str) -> TSPInstance:
        problem = tsplib95.load(source)

        name = problem.name or source
        dimension = problem.dimension

        if not problem.node_coords:
            raise ValueError(
                f"Instance '{name}' has no NODE_COORD_SECTION. "
                "Only coordinate-based instances are supported by TSPLibReader."
            )

        # tsplib95 node indices start at 1; sort for a stable ordering
        nodes = sorted(problem.node_coords.keys())
        coords = np.array([problem.node_coords[n] for n in nodes], dtype=np.float64)

        distance_matrix = _build_distance_matrix(coords)

        return TSPInstance(
            name=name,
            dimension=dimension,
            coordinates=coords,
            distance_matrix=distance_matrix,
        )
