from __future__ import annotations

import networkx as nx

from topologies.base import TopologyMixin


class FreeScaleTopologyMixin(TopologyMixin):
    """Scale-free network topology using the Barabási-Albert (BA) model.

    A random BA graph is generated with ``num_particles`` nodes where each new
    node attaches to ``ba_attachment`` existing nodes.  Neighbourhood search is
    a depth-limited DFS from the starting particle.

    Args:
        ba_attachment: Number of edges each new node attaches to in the BA model
            (the *m* parameter).  Default 11.
    """

    def __init__(self, ba_attachment: int = 11, **kwargs) -> None:
        self._ba_attachment = ba_attachment
        super().__init__(**kwargs)

    def build_topology(self) -> None:
        graph = nx.barabasi_albert_graph(self.num_particles, self._ba_attachment)
        self._neighbours: dict[int, list[int]] = {
            node: list(graph.neighbors(node)) for node in graph.nodes
        }

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
