from __future__ import annotations

import math

from topologies.base import TopologyMixin


class MeshTopologyMixin(TopologyMixin):
    """2-D mesh (grid) topology without wrap-around.

    Particles are arranged in a ``ceil(sqrt(N)) × ceil(sqrt(N))`` grid.
    The neighbourhood of particle *i* is all particles within
    ``neighbours_radius`` steps in both grid dimensions (Chebyshev distance).
    """

    def build_topology(self) -> None:
        n = self.num_particles
        grid_size = math.ceil(math.sqrt(n))
        self._neighbours: dict[int, list[int]] = {}

        for idx in range(n):
            row, col = divmod(idx, grid_size)
            neighbours: list[int] = []
            for dr in range(-self.neighbours_radius, self.neighbours_radius + 1):
                for dc in range(-self.neighbours_radius, self.neighbours_radius + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        nb_idx = nr * grid_size + nc
                        if nb_idx < n:
                            neighbours.append(nb_idx)
            self._neighbours[idx] = neighbours

    def get_neighbourhood_best(self, particle_idx: int) -> list[int]:
        best_len = self.lengths[particle_idx]
        best_idx = particle_idx

        for nb in self._neighbours[particle_idx]:
            if self.lengths[nb] < best_len:
                best_len = self.lengths[nb]
                best_idx = nb

        return self.particles[best_idx]
