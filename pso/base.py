from __future__ import annotations

import math
import numpy as np

from data.loader import TSPInstance


class PSOBase:
    """Core TSP-PSO implementation.

    State, greedy initialisation, path evaluation, and the main update loop all
    live here.  Topology and operator behaviour are injected by mixing in the
    appropriate subclasses:

        class MyPSO(DefaultOperatorsMixin, RingTopologyMixin, PSOBase):
            pass

    MRO: operator mixin → topology mixin → PSOBase.  Topology __init__ fires
    first (extracts its kwargs), calls super().__init__(**kwargs) which lands
    here; build_topology() is called by TopologyMixin after PSOBase.__init__
    so self.particles already exists.

    Args:
        instance: Loaded TSP problem instance.
        num_particles: Swarm size (default 700).
        max_iter: Maximum iterations (default 20 000).
        stagnation_window: Number of consecutive recorded results that must be
            equal before early stopping.  Set to ``None`` to disable early
            stopping and always run to *max_iter* (default 100).
        **kwargs: Forwarded to mixin ``__init__`` chains.
    """

    def __init__(
        self,
        instance: TSPInstance,
        num_particles: int = 700,
        max_iter: int = 20_000,
        stagnation_window: int | None = 100,
        **kwargs,
    ) -> None:
        # Forward any remaining kwargs up the MRO (operator / topology mixins may
        # have already consumed theirs before calling super().__init__)
        super().__init__(**kwargs)

        self.instance = instance
        self.num_city: int = instance.dimension
        self.dis_mat: np.ndarray = instance.distance_matrix
        self.num_particles: int = num_particles
        self.max_iter: int = max_iter
        self.stagnation_window: int | None = stagnation_window

        # Initialise swarm
        self.particles: list[list[int]] = self._greedy_init()
        self.lengths: list[float] = [self.path_length(p) for p in self.particles]

        init_best_len = min(self.lengths)
        init_best_idx = self.lengths.index(init_best_len)
        self.global_best: list[int] = self.particles[init_best_idx].copy()
        self.global_best_length: float = init_best_len

        # Mutable best trackers used inside the loop
        self.best_length: float = self.global_best_length
        self.best_path: list[int] = self.global_best.copy()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _greedy_init(self) -> list[list[int]]:
        """Nearest-neighbour greedy initialisation.

        For the first *num_city* particles each starts from a distinct city.
        Beyond that, a random already-computed path is copied.
        """
        result: list[list[int]] = []
        for i in range(self.num_particles):
            if i >= self.num_city:
                src = np.random.randint(0, self.num_city)
                result.append(result[src].copy())
                continue

            start = i
            rest = list(range(self.num_city))
            rest.remove(start)
            path = [start]
            current = start
            while rest:
                nearest = min(rest, key=lambda x: self.dis_mat[current][x])
                path.append(nearest)
                rest.remove(nearest)
                current = nearest
            result.append(path)
        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def path_length(self, path: list[int]) -> float:
        """Return the total tour length for *path* (including return edge)."""
        total = self.dis_mat[path[-1]][path[0]]
        for i in range(len(path) - 1):
            total += self.dis_mat[path[i]][path[i + 1]]
        return float(total)

    def _update_global_best(self) -> None:
        """Scan all particles and update global best if any improved."""
        min_len = min(self.lengths)
        min_idx = self.lengths.index(min_len)
        if min_len < self.global_best_length:
            self.global_best_length = min_len
            self.global_best = self.particles[min_idx].copy()

    # ------------------------------------------------------------------
    # Topology hooks (overridden by topology mixins)
    # ------------------------------------------------------------------

    def build_topology(self) -> None:
        """Called once after __init__.  Default: no-op (global topology)."""
        pass

    def get_neighbourhood_best(self, particle_idx: int) -> list[int]:
        """Return the guide path for particle *particle_idx*.

        Default (global topology): the single global best path.
        Topology mixins override this to return a local neighbourhood best.
        """
        return self.global_best

    # ------------------------------------------------------------------
    # Operator hooks (must be provided by an operator mixin)
    # ------------------------------------------------------------------

    def crossover(self, current: list[int], guide: list[int]) -> tuple[list[int], float]:
        raise NotImplementedError(
            f"{type(self).__name__} must include an operator mixin that implements crossover()"
        )

    def mutate(self, path: list[int]) -> tuple[list[int], float]:
        raise NotImplementedError(
            f"{type(self).__name__} must include an operator mixin that implements mutate()"
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> "RunResult":
        """Run the PSO and return a :class:`~experiments.result.RunResult`.

        The loop is identical across all topology/operator combinations — only
        ``get_neighbourhood_best`` and ``crossover``/``mutate`` differ.
        """
        # Import here to avoid circular imports at module level
        from experiments.result import RunResult

        # Stagnation tracking: record best every 20 iters
        recorded: list[float] = [self.best_length]
        # Full per-iteration history for convergence plots
        iter_history: list[float] = [self.best_length]

        iterations_run = 0

        for cnt in range(1, self.max_iter):
            iterations_run = cnt

            # Early stopping: stagnation check
            if (
                self.stagnation_window is not None
                and len(recorded) > 130
                and all(v == recorded[-1] for v in recorded[-self.stagnation_window:])
            ):
                break

            for i, particle in enumerate(self.particles):
                tmp_l = self.lengths[i]

                guide = self.get_neighbourhood_best(i)
                new_one, new_l = self.crossover(particle, guide)

                if new_l < self.best_length:
                    self.best_length = tmp_l
                    self.best_path = particle

                if new_l < tmp_l or np.random.rand() < 0.1:
                    particle = new_one
                    tmp_l = new_l

                particle, tmp_l = self.mutate(particle)

                if new_l < self.best_length:
                    self.best_length = tmp_l
                    self.best_path = particle

                if new_l < tmp_l or np.random.rand() < 0.1:
                    particle = new_one
                    tmp_l = new_l

                self.particles[i] = particle
                self.lengths[i] = tmp_l

            self._update_global_best()

            if self.global_best_length < self.best_length:
                self.best_length = self.global_best_length
                self.best_path = self.global_best.copy()

            if cnt % 20 == 0:
                recorded.append(self.best_length)

            iter_history.append(self.best_length)

        return RunResult(
            run_index=0,  # set by ExperimentRunner
            best_path_length=self.best_length,
            best_path=self.best_path.copy(),
            iteration_history=recorded,
            iterations_run=iterations_run,
            wall_time_seconds=0.0,  # set by ExperimentRunner
        )
