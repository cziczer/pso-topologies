from __future__ import annotations

import numpy as np

from topologies.base import TopologyMixin


class DynamicSimilarityTopologyMixin(TopologyMixin):
    """Dynamic topology: neighbourhoods recalculated periodically from path similarity.

    Similarity is measured via normalised Hamming distance on canonical paths
    (each path rolled to start with city 0 before comparison).

    Three modes:
    - ``'exploit'``: neighbours = most *similar* particles → shared refinement
    - ``'explore'``: neighbours = most *dissimilar* particles → diversity injection
    - ``'mix'``    : alternates exploit/explore each recalculation cycle;
                     ``explore_every`` controls the ratio (default: 2 exploit, 1 explore)

    Args:
        recalc_interval: Recalculate every N iterations (default 10).
        neighbor_pct:    Fraction of swarm per neighbourhood (default 0.05 = 5 %).
        similarity_mode: ``'exploit'``, ``'explore'``, or ``'mix'`` (default ``'exploit'``).
        explore_every:   In ``'mix'`` mode, explore fires every Nth recalculation (default 3).
    """

    def __init__(
        self,
        recalc_interval: int = 10,
        neighbor_pct: float = 0.05,
        similarity_mode: str = "exploit",
        explore_every: int = 3,
        **kwargs,
    ) -> None:
        self._recalc_interval = recalc_interval
        self._neighbor_pct = neighbor_pct
        self._similarity_mode = similarity_mode
        self._explore_every = explore_every
        self._neighbours: dict[int, list[int]] = {}
        self._iter_count: int = 0
        self._recalc_count: int = 0
        super().__init__(**kwargs)

    def build_topology(self) -> None:
        self._recalculate_neighbourhoods()

    def _active_mode(self) -> str:
        if self._similarity_mode != "mix":
            return self._similarity_mode
        # explore fires on recalc 3, 6, 9, ...; everything else is exploit
        return "explore" if self._recalc_count % self._explore_every == 0 else "exploit"

    def _canonical_paths(self) -> np.ndarray:
        """Return paths rolled to start with city 0, shape (num_particles, num_city)."""
        n = self.num_particles
        nc = self.num_city
        canonical = np.empty((n, nc), dtype=np.int64)
        for i, path in enumerate(self.particles):
            arr = np.asarray(path, dtype=np.int64)
            start = int(np.argwhere(arr == 0)[0, 0])
            canonical[i] = np.roll(arr, -start)
        return canonical

    def _recalculate_neighbourhoods(self) -> None:
        self._recalc_count += 1
        mode = self._active_mode()

        n = self.num_particles
        k = min(max(1, int(n * self._neighbor_pct)), n - 1)
        nc = self.num_city

        canonical = self._canonical_paths()

        # Pairwise normalised Hamming; loop avoids O(n² · nc) intermediate array
        dists = np.empty((n, n), dtype=np.float32)
        for i in range(n):
            dists[i] = (canonical != canonical[i]).sum(axis=1) / nc
        np.fill_diagonal(dists, np.inf)

        neighbours: dict[int, list[int]] = {}
        for i in range(n):
            row = dists[i]
            if mode == "exploit":
                idx = np.argpartition(row, k)[:k]
            else:  # explore
                idx = np.argpartition(row, n - 1 - k)[n - 1 - k:]
            neighbours[i] = idx.tolist()

        self._neighbours = neighbours

    def get_neighbourhood_best(self, particle_idx: int) -> list[int]:
        if particle_idx == 0:
            self._iter_count += 1
            if self._iter_count % self._recalc_interval == 0:
                self._recalculate_neighbourhoods()

        nbrs = self._neighbours.get(particle_idx)
        if not nbrs:
            return self.global_best

        best_idx = min(nbrs, key=lambda j: self.lengths[j])
        return self.particles[best_idx]


class DynamicOppositeTopologyMixin(DynamicSimilarityTopologyMixin):
    """Explore-only variant: neighbours are always the most dissimilar particles."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("similarity_mode", "explore")
        super().__init__(**kwargs)


class DynamicMixTopologyMixin(DynamicSimilarityTopologyMixin):
    """Alternating variant: cycles between exploit and explore recalculations."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("similarity_mode", "mix")
        super().__init__(**kwargs)
