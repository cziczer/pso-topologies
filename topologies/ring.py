from __future__ import annotations

from topologies.base import TopologyMixin


class RingTopologyMixin(TopologyMixin):
    """Ring topology: particles arranged in a circular list.

    Each particle's neighbourhood is defined by traversing left and right
    along the ring up to ``neighbours_radius`` hops.
    """

    def build_topology(self) -> None:
        n = self.num_particles
        self._neighbours: dict[int, dict[str, int]] = {
            i: {"left": (i - 1) % n, "right": (i + 1) % n}
            for i in range(n)
        }

    def get_neighbourhood_best(self, particle_idx: int) -> list[int]:
        best_len = self.lengths[particle_idx]
        best_idx = particle_idx

        node = particle_idx
        for _ in range(self.neighbours_radius):
            node = self._neighbours[node]["left"]
            if self.lengths[node] < best_len:
                best_len = self.lengths[node]
                best_idx = node

        node = particle_idx
        for _ in range(self.neighbours_radius):
            node = self._neighbours[node]["right"]
            if self.lengths[node] < best_len:
                best_len = self.lengths[node]
                best_idx = node

        return self.particles[best_idx]
