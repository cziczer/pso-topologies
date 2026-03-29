from __future__ import annotations

from topologies.base import TopologyMixin


class GlobalTopologyMixin(TopologyMixin):
    """Global (fully-connected) topology — all particles share the single global best.

    This is the default behaviour of ``PSOBase`` and is used when no topology
    token is specified in the algorithm name (i.e. plain ``"PSO"``).
    """

    def build_topology(self) -> None:
        pass  # No structure to build

    def get_neighbourhood_best(self, particle_idx: int) -> list[int]:
        return self.global_best
