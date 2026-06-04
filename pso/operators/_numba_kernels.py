"""Numba JIT kernels for APV and SWU operator updates.

First call compiles and caches to __pycache__; subsequent calls are fast.
All arrays must be C-contiguous int64/float64 numpy arrays.
"""
from __future__ import annotations

import numpy as np
import numba


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
    dis_mat: np.ndarray,
    n: int,
) -> tuple:
    r1 = np.random.random()
    r2 = np.random.random()
    P_p = _trans_seq_indices(particle, pbest, n)
    P_g = _trans_seq_indices(particle, gbest, n)
    _update_vel(vel, P_p, P_g, omega, c_p * r1, c_g * r2)
    new_p = _apply_vel(particle, vel, pairs)
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
    new_p = _apply_vel(particle, vel, pairs)
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
    pairs: np.ndarray,
    omega: float,
    c_p: float,
    c_g: float,
    dis_mat: np.ndarray,
    n: int,
    num_particles: int,
) -> tuple:
    """Update all particles (parallel prange).

    Modifies *velocities*, *pbests*, and *pbest_lengths* in-place.
    Returns (new_particles, new_lengths).
    """
    new_particles = particles.copy()
    new_lengths = np.empty(num_particles, dtype=np.float64)
    for i in numba.prange(num_particles):
        new_p, new_l = _apv_update_particle(
            particles[i], pbests[i], gbests[i], velocities[i],
            pairs, omega, c_p, c_g, dis_mat, n,
        )
        new_particles[i] = new_p
        new_lengths[i] = new_l
        if new_l < pbest_lengths[i]:
            pbest_lengths[i] = new_l
            pbests[i] = new_p
    return new_particles, new_lengths


@numba.njit(parallel=True, cache=True)
def swu_full_iteration(
    particles: np.ndarray,
    pbests: np.ndarray,
    gbests: np.ndarray,
    velocities: np.ndarray,
    pbest_lengths: np.ndarray,
    pairs: np.ndarray,
    omega: float,
    c_p: float,
    c_g: float,
    dis_mat: np.ndarray,
    n: int,
    num_particles: int,
) -> tuple:
    """Update all particles with SWU similarity weighting (parallel prange).

    Modifies *velocities*, *pbests*, and *pbest_lengths* in-place.
    Returns (new_particles, new_lengths).
    """
    new_particles = particles.copy()
    new_lengths = np.empty(num_particles, dtype=np.float64)
    for i in numba.prange(num_particles):
        new_p, new_l = _swu_update_particle(
            particles[i], pbests[i], gbests[i], velocities[i],
            pairs, omega, c_p, c_g, dis_mat, n,
        )
        new_particles[i] = new_p
        new_lengths[i] = new_l
        if new_l < pbest_lengths[i]:
            pbest_lengths[i] = new_l
            pbests[i] = new_p
    return new_particles, new_lengths
