import numpy as np
import numba
import tsplib95

from data.loader import InstanceLoader, TSPInstance


@numba.njit(cache=True)
def _build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix via Numba JIT — O(n²) inner loop.

    Correct for EDGE_WEIGHT_TYPE: EUC_2D only.  Never use it for GEO instances;
    see _build_geo_distance_matrix for why.
    """
    n = coords.shape[0]
    dist = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dist[i, j] = np.sqrt(dx * dx + dy * dy)
    return dist


def _build_geo_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Great-circle distance matrix for EDGE_WEIGHT_TYPE: GEO.

    Implements the TSPLIB95 GEO formula exactly, including its quirks: the
    coordinates are DDD.MM (degrees and *minutes*, not decimal degrees), PI is
    the truncated 3.141592, the earth radius is 6378.388 km, and the result is
    truncated to an integer.

    This exists because the reader previously applied plain Euclidean distance
    to every instance regardless of its header.  On DDD.MM values that
    understates GEO distances by about two orders of magnitude -- the first node
    pair of ali535 came out as 20.67 instead of 2264 -- and tours measured over
    the resulting matrix were ~98.8 % "shorter" than the published optimum,
    which is impossible.  Every ali535 and gr666 result produced before this
    function existed is meaningless.
    """
    PI = 3.141592
    RRR = 6378.388

    degrees = np.trunc(coords)
    minutes = coords - degrees
    radians = PI * (degrees + 5.0 * minutes / 3.0) / 180.0

    lat = radians[:, 0]
    lon = radians[:, 1]

    q1 = np.cos(lon[:, None] - lon[None, :])
    q2 = np.cos(lat[:, None] - lat[None, :])
    q3 = np.cos(lat[:, None] + lat[None, :])

    inner = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
    # Guard against |inner| drifting just outside [-1, 1] through rounding.
    inner = np.clip(inner, -1.0, 1.0)

    dist = np.trunc(RRR * np.arccos(inner) + 1.0)
    np.fill_diagonal(dist, 0.0)
    return dist


# EDGE_WEIGHT_TYPE -> matrix builder.  Anything absent here is rejected at load
# time rather than silently measured with the wrong metric.
_DISTANCE_BUILDERS = {
    "EUC_2D": _build_distance_matrix,
    "GEO": _build_geo_distance_matrix,
}


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

        weight_type = (problem.edge_weight_type or "").strip().upper()
        try:
            build = _DISTANCE_BUILDERS[weight_type]
        except KeyError:
            raise ValueError(
                f"Instance '{name}' declares EDGE_WEIGHT_TYPE: "
                f"{weight_type or '<missing>'}, which this reader does not "
                f"implement.  Supported: {', '.join(sorted(_DISTANCE_BUILDERS))}.  "
                "Add a builder rather than falling back to Euclidean — using the "
                "wrong metric silently produces results that look plausible but "
                "are not comparable to published optima."
            ) from None

        distance_matrix = build(coords)

        return TSPInstance(
            name=name,
            dimension=dimension,
            coordinates=coords,
            distance_matrix=distance_matrix,
        )
