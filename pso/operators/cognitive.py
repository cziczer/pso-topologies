from __future__ import annotations

import numpy as np

from pso.operators.default import DefaultOperatorsMixin


class CognitiveMixin(DefaultOperatorsMixin):
    """Extends the default operators with a personal-best (cognitive) component.

    Standard TSP-PSO only guides particles toward the neighbourhood best
    (social component).  This mixin adds the cognitive component: each particle
    also performs an OX crossover with its own personal best before the social
    crossover step, mirroring the classic PSO velocity equation:

        velocity = inertia + c1*cognitive + c2*social

    Personal bests are initialised from the greedy-initialised particles and
    updated whenever a particle's tour improves.

    Composed class example::

        class MyPSO(CognitiveMixin, RingTopologyMixin, PSOBase): pass

    Algorithm name tokens: ``"Cognitive"`` (case-insensitive).
    Examples: ``"PSO-Cognitive"``, ``"PSO-Cognitive-Ring"``.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # self.particles and self.lengths are set by PSOBase.__init__ before
        # this line executes (TopologyMixin calls super().__init__ first).
        self.personal_bests: list[list[int]] = [p.copy() for p in self.particles]
        self.personal_best_lengths: list[float] = self.lengths.copy()

    def run(self) -> "RunResult":  # type: ignore[override]
        """Run loop with cognitive + social crossover at each particle update.

        Update sequence per particle per iteration:
          1. Cognitive crossover: current × personal_best  → candidate
          2. Social crossover:    candidate × neighbourhood_best → candidate
          3. Swap mutation
          4. Update personal best if improved
          5. Update global best tracker
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

            for i, particle in enumerate(self.particles):
                tmp_l = self.lengths[i]

                # --- Cognitive step: pull toward personal best ---------------
                cog, cog_l = self.crossover(particle, self.personal_bests[i])
                if cog_l < tmp_l or np.random.rand() < 0.1:
                    particle, tmp_l = cog, cog_l

                # --- Social step: pull toward neighbourhood best -------------
                guide = self.get_neighbourhood_best(i)
                soc, soc_l = self.crossover(particle, guide)
                if soc_l < tmp_l or np.random.rand() < 0.1:
                    particle, tmp_l = soc, soc_l

                # --- Mutation ------------------------------------------------
                particle, tmp_l = self.mutate(particle)

                # --- Update personal best ------------------------------------
                if tmp_l < self.personal_best_lengths[i]:
                    self.personal_best_lengths[i] = tmp_l
                    self.personal_bests[i] = particle.copy()

                # --- Track swarm-wide best -----------------------------------
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
