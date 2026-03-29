from __future__ import annotations

from abc import abstractmethod


class TopologyMixin:
    """Abstract base for topology mixins.

    Subclasses must implement ``build_topology()`` and
    ``get_neighbourhood_best()``.

    Cooperative ``__init__``: extracts ``neighbours_radius`` from kwargs,
    calls ``super().__init__(**kwargs)`` (which reaches ``PSOBase.__init__``),
    then calls ``build_topology()`` so that ``self.particles`` is already
    populated when the topology is constructed.

    Args:
        neighbours_radius: How far to traverse the topology graph when looking
            for the neighbourhood best (default 2).
    """

    def __init__(self, neighbours_radius: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.neighbours_radius = neighbours_radius
        self.build_topology()

    @abstractmethod
    def build_topology(self) -> None:
        """Build the neighbourhood data structure.  Called once after __init__."""
        ...

    @abstractmethod
    def get_neighbourhood_best(self, particle_idx: int) -> list[int]:
        """Return the best-known path visible to *particle_idx*."""
        ...
