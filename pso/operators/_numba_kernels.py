"""Numba JIT kernels for APV and SWU operator updates.

First call compiles and caches to __pycache__; subsequent calls are fast.
All arrays must be C-contiguous int64/float64 numpy arrays.
"""
from __future__ import annotations

import numpy as np
import numba


# Probability of accepting a worsening position, mirroring the baseline loop in
# PSOBase.run() (``if new_l < tmp_l or np.random.rand() < 0.1``).  Kept identical
# so APV/SWU are compared against the default operator under the same selection
# pressure rather than a different one.
ACCEPT_WORSE_P = 0.1


@numba.njit(cache=True)
def build_trans_pairs(n: int) -> np.ndarray:
    """Return (num_trans, 2) array of (j, k) pairs for 1 <= j < k <= n-1."""
    num_trans = (n - 1) * (n - 2) // 2
    pairs = np.empty((num_trans, 2), dtype=np.int64)
    idx = 0
    for j in range(1, n):
        for k in range(j + 1, n):
            pairs[idx, 0] = j
            pairs[idx, 1] = k
            idx += 1
    return pairs


@numba.njit(cache=True)
def _trans_seq_indices(x: np.ndarray, target: np.ndarray, n: int) -> np.ndarray:
    """Flat transposition indices that transform x → target (position 0 fixed).

    Returns a variable-length view into a pre-allocated buffer.
    k > j is guaranteed because positions 0..j-1 are already fixed when j is
    processed, so target[j] can only be at some k >= j+1 in x.
    """
    work = x.copy()
    pos = np.empty(n, dtype=np.int64)
    for i in range(n):
        pos[work[i]] = i

    buf = np.empty(n, dtype=np.int64)
    count = 0
    for j in range(1, n):
        if work[j] != target[j]:
            city = target[j]
            k = pos[city]
            # flat index: (j-1)*(n-1) - (j-1)*j//2 + (k-j) - 1
            buf[count] = (j - 1) * (n - 1) - (j - 1) * j // 2 + (k - j) - 1
            count += 1
            pos[work[j]] = k
            pos[city] = j
            tmp = work[j]
            work[j] = work[k]
            work[k] = tmp
    return buf[:count]


@numba.njit(cache=True)
def _update_vel(
    vel: np.ndarray,
    P_p: np.ndarray,
    P_g: np.ndarray,
    omega: float,
    w_p: float,
    w_g: float,
) -> None:
    """Decay all velocity entries then add weighted occurrence components (in-place)."""
    for i in range(len(vel)):
        vel[i] *= omega
    for idx in P_p:
        v = vel[idx] + w_p
        vel[idx] = 1.0 if v > 1.0 else v
    for idx in P_g:
        v = vel[idx] + w_g
        vel[idx] = 1.0 if v > 1.0 else v


@numba.njit(cache=True)
def _clamp_vel(vel: np.ndarray, v_max: float) -> None:
    """Scale *vel* down so its L1 norm is at most *v_max* (in-place).

    Each entry of the velocity is the probability of applying one transposition,
    and they are applied independently, so sum(vel) IS the expected step length
    in transpositions.  Without this clamp the step grows with the NUMBER of
    reinforced entries rather than their size: |P_p| + |P_g| ~ 2(n-1) entries are
    reinforced each iteration and a continuously reinforced entry settles at
    c*r/(1-omega) ~ 0.5, giving sum(vel) ~ n.  An n-transposition step on an
    n-city tour resamples the tour instead of moving it, which is why APV and SWU
    returned the greedy initial tour unchanged.

    Clamping the L1 norm is the discrete counterpart of the usual v_max bound on
    velocity magnitude in continuous PSO; the per-entry clip to [0, 1] already in
    _update_vel bounds the components but not the norm.
    """
    total = 0.0
    for i in range(len(vel)):
        total += vel[i]
    if total > v_max and total > 0.0:
        scale = v_max / total
        for i in range(len(vel)):
            vel[i] *= scale


@numba.njit(cache=True)
def _apply_vel(particle: np.ndarray, vel: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Stochastically apply transpositions weighted by velocity probabilities."""
    new_p = particle.copy()
    for i in range(len(vel)):
        if vel[i] > 0.0 and np.random.random() < vel[i]:
            j = pairs[i, 0]
            k = pairs[i, 1]
            tmp = new_p[j]
            new_p[j] = new_p[k]
            new_p[k] = tmp
    return new_p


@numba.njit(cache=True)
def _explore(particle: np.ndarray, v_explore: float, n: int) -> None:
    """Apply ~*v_explore* uniformly random transpositions in place (turbulence).

    Needed because the velocity's support is exactly the set of transpositions
    leading to pbest / gbest.  A particle that coincides with both -- which the
    swarm's best particle does from iteration 1, since gbest IS its position --
    has an all-zero velocity, so it proposes nothing and never moves again.
    Every other particle can only move along transpositions towards an existing
    reference tour, so the swarm cannot leave the convex hull of its own
    initialisation and can never beat the best tour it started with.  Measured on
    berlin52 before this term existed: the best particle's ||v||_1 was 0.0000 for
    all 300 iterations, it never moved, and 0 of 12 000 proposed tours were
    shorter than the greedy start.

    This is memoryless on purpose: it perturbs the position, not the velocity,
    so inertia does not accumulate it.

    v_explore = 0.0 reproduces the operator exactly as published.
    """
    m = v_explore
    while m > 0.0:
        if m >= 1.0 or np.random.random() < m:
            j = 1 + np.random.randint(0, n - 1)
            k = 1 + np.random.randint(0, n - 1)
            if j != k:
                tmp = particle[j]
                particle[j] = particle[k]
                particle[k] = tmp
        m -= 1.0


@numba.njit(cache=True)
def _path_length_nb(path: np.ndarray, dis_mat: np.ndarray, n: int) -> float:
    total = dis_mat[path[n - 1], path[0]]
    for i in range(n - 1):
        total += dis_mat[path[i], path[i + 1]]
    return total


@numba.njit(cache=True)
def _hamming_sim(x: np.ndarray, y: np.ndarray, n: int) -> float:
    s = 0
    for i in range(n):
        if x[i] == y[i]:
            s += 1
    return s / n


# ---------------------------------------------------------------------------
# Per-particle update kernels (non-parallel — called from parallel outer loop)
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _apv_update_particle(
    particle: np.ndarray,
    pbest: np.ndarray,
    gbest: np.ndarray,
    vel: np.ndarray,
    pairs: np.ndarray,
    omega: float,
    c_p: float,
    c_g: float,
    v_max: float,
    v_explore: float,
    dis_mat: np.ndarray,
    n: int,
) -> tuple:
    r1 = np.random.random()
    r2 = np.random.random()
    P_p = _trans_seq_indices(particle, pbest, n)
    P_g = _trans_seq_indices(particle, gbest, n)
    _update_vel(vel, P_p, P_g, omega, c_p * r1, c_g * r2)
    _clamp_vel(vel, v_max)
    new_p = _apply_vel(particle, vel, pairs)
    _explore(new_p, v_explore, n)
    return new_p, _path_length_nb(new_p, dis_mat, n)


@numba.njit(cache=True)
def _swu_update_particle(
    particle: np.ndarray,
    pbest: np.ndarray,
    gbest: np.ndarray,
    vel: np.ndarray,
    pairs: np.ndarray,
    omega: float,
    c_p: float,
    c_g: float,
    v_max: float,
    v_explore: float,
    dis_mat: np.ndarray,
    n: int,
) -> tuple:
    r1 = np.random.random()
    r2 = np.random.random()
    sim_p = _hamming_sim(particle, pbest, n)
    sim_g = _hamming_sim(particle, gbest, n)
    P_p = _trans_seq_indices(particle, pbest, n)
    P_g = _trans_seq_indices(particle, gbest, n)
    _update_vel(vel, P_p, P_g, omega, c_p * r1 * sim_p, c_g * r2 * sim_g)
    _clamp_vel(vel, v_max)
    new_p = _apply_vel(particle, vel, pairs)
    _explore(new_p, v_explore, n)
    return new_p, _path_length_nb(new_p, dis_mat, n)


# ---------------------------------------------------------------------------
# Full-iteration parallel kernels — one call per outer loop iteration
# ---------------------------------------------------------------------------

@numba.njit(parallel=True, cache=True)
def apv_full_iteration(
    particles: np.ndarray,
    pbests: np.ndarray,
    gbests: np.ndarray,
    velocities: np.ndarray,
    pbest_lengths: np.ndarray,
    lengths: np.ndarray,
    pairs: np.ndarray,
    omega: float,
    c_p: float,
    c_g: float,
    v_max: float,
    v_explore: float,
    dis_mat: np.ndarray,
    n: int,
    num_particles: int,
) -> tuple:
    """Update all particles (parallel prange).

    Modifies *velocities*, *pbests*, and *pbest_lengths* in-place.
    Returns (new_particles, new_lengths).

    A candidate position replaces the current one only if it is shorter, or with
    probability ACCEPT_WORSE_P otherwise.  Without that test the particle
    accepted every candidate unconditionally: the velocity applies each of the
    (n-1)(n-2)/2 transpositions independently with probability v, so the expected
    step is sum(v) ~ n swaps per iteration, which resamples the tour instead of
    moving it towards a reference.  Measured on berlin52, the mean tour length
    went 8182 -> 18083 after a single iteration and settled at ~23500 against
    30040 for a uniformly random tour, so best_path_length stayed equal to
    init_path_length in 1758 of 1760 recorded APV runs.
    """
    new_particles = particles.copy()
    new_lengths = lengths.copy()
    for i in numba.prange(num_particles):
        new_p, new_l = _apv_update_particle(
            particles[i], pbests[i], gbests[i], velocities[i],
            pairs, omega, c_p, c_g, v_max, v_explore, dis_mat, n,
        )
        # pbest is a memory of the best position ever evaluated, so it records
        # the candidate whether or not the particle moves there.
        if new_l < pbest_lengths[i]:
            pbest_lengths[i] = new_l
            pbests[i] = new_p
        if new_l < lengths[i] or np.random.random() < ACCEPT_WORSE_P:
            new_particles[i] = new_p
            new_lengths[i] = new_l
    return new_particles, new_lengths


@numba.njit(parallel=True, cache=True)
def swu_full_iteration(
    particles: np.ndarray,
    pbests: np.ndarray,
    gbests: np.ndarray,
    velocities: np.ndarray,
    pbest_lengths: np.ndarray,
    lengths: np.ndarray,
    pairs: np.ndarray,
    omega: float,
    c_p: float,
    c_g: float,
    v_max: float,
    v_explore: float,
    dis_mat: np.ndarray,
    n: int,
    num_particles: int,
) -> tuple:
    """Update all particles with SWU similarity weighting (parallel prange).

    Modifies *velocities*, *pbests*, and *pbest_lengths* in-place.
    Returns (new_particles, new_lengths).

    Carries the same acceptance test as apv_full_iteration.  The Hamming
    weighting only slows the divergence it was missing -- measured on berlin52
    the expected step grew 0.8 -> 5.4 swaps per iteration over 60 iterations and
    the mean tour length still reached 18546 -- which is why 1231 of 1750
    recorded SWU runs showed no improvement over the greedy start.
    """
    new_particles = particles.copy()
    new_lengths = lengths.copy()
    for i in numba.prange(num_particles):
        new_p, new_l = _swu_update_particle(
            particles[i], pbests[i], gbests[i], velocities[i],
            pairs, omega, c_p, c_g, v_max, v_explore, dis_mat, n,
        )
        if new_l < pbest_lengths[i]:
            pbest_lengths[i] = new_l
            pbests[i] = new_p
        if new_l < lengths[i] or np.random.random() < ACCEPT_WORSE_P:
            new_particles[i] = new_p
            new_lengths[i] = new_l
    return new_particles, new_lengths
