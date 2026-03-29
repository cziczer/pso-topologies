from __future__ import annotations

from topologies.base import TopologyMixin


class TorusTopologyMixin(TopologyMixin):
    """Torus topology: a 2-D grid with wrap-around edges.

    Particles occupy the first ``num_particles`` cells of a
    ``grid_width × grid_width`` grid.  Each cell is connected to its four
    neighbours (left, right, up, down) with modular arithmetic.

    Neighbourhood search is a depth-limited DFS from the starting particle.

    Args:
        grid_width: Side length of the square grid (default 80).  Must satisfy
            ``grid_width ** 2 >= num_particles``.
    """

    def __init__(self, grid_width: int = 80, **kwargs) -> None:
        self._grid_width = grid_width
        super().__init__(**kwargs)

    def build_topology(self) -> None:
        w = self._grid_width
        if w * w < self.num_particles:
            raise ValueError(
                f"grid_width={w} is too small for num_particles={self.num_particles}. "
                f"Need grid_width >= {int(self.num_particles ** 0.5) + 1}."
            )

        self._neighbours: dict[int, list[int]] = {}
        for row in range(w):
            for col in range(w):
                idx = row * w + col
                left  = row * w + (col - 1) % w
                right = row * w + (col + 1) % w
                up    = ((row - 1) % w) * w + col
                down  = ((row + 1) % w) * w + col
                # Only keep neighbours that correspond to actual particles
                self._neighbours[idx] = [
                    nb for nb in (left, right, up, down) if nb < self.num_particles
                ]

    def get_neighbourhood_best(self, particle_idx: int) -> list[int]:
        best_len = self.lengths[particle_idx]
        best_idx = particle_idx

        def _dfs(node: int, depth: int, visited: set[int]) -> None:
            nonlocal best_len, best_idx
            if node in visited or depth > self.neighbours_radius:
                return
            visited.add(node)
            if self.lengths[node] < best_len:
                best_len = self.lengths[node]
                best_idx = node
            for nb in self._neighbours.get(node, []):
                _dfs(nb, depth + 1, visited)

        _dfs(particle_idx, depth=0, visited=set())
        return self.particles[best_idx]
