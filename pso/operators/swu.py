from __future__ import annotations

import numpy as np

from pso.operators._numba_kernels import swu_full_iteration
from pso.operators.apv import APVOperatorsMixin


class SWUOperatorsMixin(APVOperatorsMixin):
    """Similarity-Weighted Position Update (SWU) velocity update operator.

    Extends APV by multiplying the pbest and gbest components of the velocity
    update by tour similarity coefficients:

        v_i(t+1) = clip(
            ω·v_i(t)
            + c_p·r1·sim(x_i, pbest_i)·P_p(i,t)
            + c_g·r2·sim(x_i, gbest)·P_g(i,t),
            0, 1
        )

    Higher similarity to a reference concentrates probability mass on the few
    remaining differences — exploration is focused, not scattered.

    Default similarity: Hamming (fraction of matching positions, computed in
    the JIT kernel).  The metric is baked into the Numba kernel and cannot be
    swapped at runtime without subclassing and providing a new kernel.

    Reference: Mastalerczyk et al., "New update operators for discrete problems
    in Particle Swarm Optimisation", JOCS 2026.

    Algorithm name token: ``"SWU"`` (case-insensitive).
    Examples: ``"PSO-SWU"``, ``"PSO-SWU-Ring"``.
    """

    def _run_iteration(
        self,
        particles_np: np.ndarray,
        pbests_np: np.ndarray,
        gbests_np: np.ndarray,
        velocities_np: np.ndarray,
        pbest_lengths_np: np.ndarray,
        dis_mat: np.ndarray,
        n: int,
        num_particles: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return swu_full_iteration(
            particles_np, pbests_np, gbests_np, velocities_np, pbest_lengths_np,
            self._pairs, self._omega, self._c_p, self._c_g, dis_mat, n, num_particles,
        )
