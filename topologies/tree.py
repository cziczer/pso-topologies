from __future__ import annotations

from topologies.base import TopologyMixin


class TreeTopologyMixin(TopologyMixin):
    """Binary-tree topology.

    Particles are indexed as nodes in an implicit binary tree:
      - left child of i  : 2*i + 1
      - right child of i : 2*i + 2
      - parent of i      : (i - 1) // 2

    Neighbourhood search is a BFS up to ``neighbours_radius`` hops from the
    starting particle, traversing parent, left, and right edges.
    """

    def build_topology(self) -> None:
        n = self.num_particles
        self._neighbours: dict[int, dict] = {}
        for i in range(n):
            left = 2 * i + 1
            right = 2 * i + 2
            self._neighbours[i] = {
                "parent": (i - 1) // 2 if i > 0 else None,
                "left": left if left < n else None,
                "right": right if right < n else None,
            }

    def get_neighbourhood_best(self, particle_idx: int) -> list[int]:
        best_len = self.lengths[particle_idx]
        best_idx = particle_idx

        queue = [(particle_idx, 0)]
        visited: set[int] = set()

        while queue:
            node, depth = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            if self.lengths[node] < best_len:
                best_len = self.lengths[node]
                best_idx = node

            if depth < self.neighbours_radius:
                for key in ("parent", "left", "right"):
                    nb = self._neighbours[node][key]
                    if nb is not None and nb not in visited:
                        queue.append((nb, depth + 1))

        return self.particles[best_idx]
