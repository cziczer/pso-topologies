from __future__ import annotations

import numpy as np


class StagnationExploreMixin:
    """Per-particle stagnation detection with trial-and-revert exploration.

    Each particle independently tracks how many iterations have passed since its
    personal best improved.  Once that count reaches *stagnation_trigger*, the
    mixin:

    1. Snapshots the particle's current position.
    2. Applies a double-bridge perturbation (4-opt; falls back to swap for tiny
       instances) to push the particle out of its local basin.
    3. Runs normal PSO crossover + mutate for *explore_window* iterations while
       tracking the best position found.
    4. At window end: keeps the best position found if it beats the snapshot,
       otherwise reverts the particle to the pre-perturbation snapshot.

    Early exit: if the particle finds a new all-time personal best during the
    exploration window, it commits immediately and exits the window early.

    Composed class example::

        class MyPSO(StagnationExploreMixin, DefaultOperatorsMixin, RingTopologyMixin, PSOBase): pass

    Algorithm name token: ``"Explore"`` (case-insensitive).
    Examples: ``"PSO-Explore"``, ``"PSO-Explore-Ring"``, ``"PSO-Cognitive-Explore-Ring"``.

    Args:
        stagnation_trigger: Iterations without personal-best improvement before
            exploration fires.  Defaults to ``stagnation_window // 2`` (from
            PSOBase), or ``50`` when early stopping is disabled.
        explore_window: Number of exploration iterations before committing or
            reverting.  Default: 10.
    """

    def __init__(self, *, stagnation_trigger: int | None = None, explore_window: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)

        sw = getattr(self, "stagnation_window", None)
        self._explore_trigger: int = stagnation_trigger if stagnation_trigger is not None else (sw // 2 if sw is not None else 50)
        self._explore_window: int = explore_window

        n = self.num_particles

        self._pb: list[list[int]] = [p.copy() for p in self.particles]
        self._pb_len: list[float] = self.lengths.copy()
        self._stag_count: list[int] = [0] * n

        self._expl_remaining: list[int] = [0] * n
        self._expl_snapshot: list[list[int] | None] = [None] * n
        self._expl_snapshot_len: list[float] = [0.0] * n
        self._expl_best: list[list[int] | None] = [None] * n
        self._expl_best_len: list[float] = [float("inf")] * n

    def _random_perturb(self, path: list[int]) -> tuple[list[int], float]:
        n = len(path)
        if n < 8:
            p = path.copy()
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
            return p, self.path_length(p)

        # Double-bridge (4-opt): guarantees the perturbation cannot be undone by 2-opt
        a, b, c = sorted(np.random.choice(range(1, n), 3, replace=False).tolist())
        new_path = path[:a] + path[b:c] + path[a:b] + path[c:]
        return new_path, self.path_length(new_path)

    def run(self) -> "RunResult":  # type: ignore[override]
        """Run loop with stagnation-triggered per-particle exploration.

        Update sequence per particle per iteration:
          1. Social crossover toward neighbourhood best.
          2. Swap mutation (both via operator mixin hooks).
          3. Personal-best update and stagnation counter increment.
          4. Exploration state machine: enter, progress, or commit/revert.
          5. Global best tracker update.
        """
        from experiments.result import RunResult

        recorded: list[float] = [self.best_length]
        iter_history: list[float] = [self.best_length]
        iterations_run = 0

        for cnt in range(1, self.max_iter):
            iterations_run = cnt

            if (
                self.stagnation_window is not None
                and len(recorded) > 130
                and all(v == recorded[-1] for v in recorded[-self.stagnation_window :])
            ):
                break

            for i in range(self.num_particles):
                particle = self.particles[i]
                tmp_l = self.lengths[i]

                # --- Social crossover + mutation (operator mixin hooks) ------
                guide = self.get_neighbourhood_best(i)
                new_one, new_l = self.crossover(particle, guide)
                if new_l < tmp_l or np.random.rand() < 0.1:
                    particle, tmp_l = new_one, new_l

                particle, tmp_l = self.mutate(particle)

                # --- Personal best and stagnation counter -------------------
                if tmp_l < self._pb_len[i]:
                    self._pb_len[i] = tmp_l
                    self._pb[i] = particle.copy()
                    self._stag_count[i] = 0
                    # Found a new personal best during exploration → commit early
                    if self._expl_remaining[i] > 0:
                        self._expl_remaining[i] = 0
                        self._expl_snapshot[i] = None
                else:
                    self._stag_count[i] += 1

                # --- Exploration state machine -------------------------------
                if self._expl_remaining[i] > 0:
                    # Inside exploration window: track best seen so far
                    if tmp_l < self._expl_best_len[i]:
                        self._expl_best_len[i] = tmp_l
                        self._expl_best[i] = particle.copy()

                    self._expl_remaining[i] -= 1

                    if self._expl_remaining[i] == 0:
                        # Window ended: stay if better, otherwise revert
                        if self._expl_best_len[i] < self._expl_snapshot_len[i]:
                            particle = self._expl_best[i]  # type: ignore[assignment]
                            tmp_l = self._expl_best_len[i]
                        else:
                            particle = self._expl_snapshot[i]  # type: ignore[assignment]
                            tmp_l = self._expl_snapshot_len[i]
                        self._expl_snapshot[i] = None

                elif self._stag_count[i] >= self._explore_trigger:
                    # Stagnation threshold reached: snapshot and perturb
                    self._expl_snapshot[i] = particle.copy()
                    self._expl_snapshot_len[i] = tmp_l
                    particle, tmp_l = self._random_perturb(particle)
                    self._expl_best[i] = particle.copy()
                    self._expl_best_len[i] = tmp_l
                    self._expl_remaining[i] = self._explore_window
                    self._stag_count[i] = 0

                # --- Global best tracker ------------------------------------
                if tmp_l < self.best_length:
                    self.best_length = tmp_l
                    self.best_path = particle.copy()

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
            run_index=0,
            best_path_length=self.best_length,
            best_path=self.best_path.copy(),
            iteration_history=recorded,
            iterations_run=iterations_run,
            wall_time_seconds=0.0,
        )
