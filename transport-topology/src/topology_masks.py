"""Deterministic sparse topology generators for transport-certificate checks."""

from __future__ import annotations

import numpy as np


def _ensure_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=bool)


def validate_support(mask: np.ndarray) -> np.ndarray:
    """Return a boolean support after checking every row and column is active."""
    support = _ensure_bool(mask)
    if support.ndim != 2:
        raise ValueError("support mask must be two-dimensional")
    if np.any(np.sum(support, axis=1) == 0):
        raise ValueError("support mask has an empty row")
    if np.any(np.sum(support, axis=0) == 0):
        raise ValueError("support mask has an empty column")
    return support


def dense_mask(n: int) -> np.ndarray:
    """Complete square support."""
    if n <= 0:
        raise ValueError("n must be positive")
    return np.ones((n, n), dtype=bool)


def band_mask(n: int, radius: int, *, wrap: bool = True) -> np.ndarray:
    """Local band support with optional circular wraparound."""
    if n <= 0:
        raise ValueError("n must be positive")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    mask = np.zeros((n, n), dtype=bool)
    for row in range(n):
        for delta in range(-radius, radius + 1):
            col = row + delta
            if wrap:
                mask[row, col % n] = True
            elif 0 <= col < n:
                mask[row, col] = True
    np.fill_diagonal(mask, True)
    return validate_support(mask)


def disconnected_blocks_mask(n: int, blocks: int = 2) -> np.ndarray:
    """Block-diagonal support with no communication between components."""
    if n <= 0:
        raise ValueError("n must be positive")
    if blocks <= 0 or blocks > n:
        raise ValueError("blocks must be in [1, n]")
    mask = np.zeros((n, n), dtype=bool)
    splits = np.array_split(np.arange(n), blocks)
    for block in splits:
        mask[np.ix_(block, block)] = True
    return validate_support(mask)


def random_regular_mask(n: int, degree: int, rng: np.random.Generator) -> np.ndarray:
    """Cyclic random-regular bipartite support from a union of permutations."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 1 <= degree <= n:
        raise ValueError("degree must be in [1, n]")
    offsets = rng.choice(n, size=degree, replace=False)
    if 0 not in offsets:
        offsets[0] = 0
    mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)
    for offset in offsets:
        mask[rows, (rows + int(offset)) % n] = True
    return validate_support(mask)


def global_token_mask(n: int, local_radius: int, num_global: int) -> np.ndarray:
    """Local support plus globally connected rows and columns."""
    if not 0 <= num_global <= n:
        raise ValueError("num_global must be in [0, n]")
    mask = band_mask(n, local_radius, wrap=True)
    if num_global:
        globals_ = np.arange(num_global)
        mask[globals_, :] = True
        mask[:, globals_] = True
    return validate_support(mask)


def bigbird_like_mask(
    n: int,
    local_radius: int,
    num_global: int,
    random_edges: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Longformer/BigBird-style local, global, and random support."""
    if random_edges < 0:
        raise ValueError("random_edges must be non-negative")
    mask = global_token_mask(n, local_radius, num_global)
    for row in range(n):
        if random_edges:
            cols = rng.choice(n, size=min(random_edges, n), replace=False)
            mask[row, cols] = True

            mask[cols, row] = True
    return validate_support(mask)


def butterfly_mask(n: int) -> np.ndarray:
    """XOR/butterfly-style square support for power-of-two lengths."""
    if n <= 0 or n & (n - 1):
        raise ValueError("n must be a positive power of two")
    mask = np.eye(n, dtype=bool)
    rows = np.arange(n)
    bit = 1
    while bit < n:
        mask[rows, rows ^ bit] = True
        bit <<= 1
    return validate_support(mask)


def bit_pair_mask(n: int, bit: int) -> np.ndarray:
    """Square two-token blocks that pair indices differing in one XOR bit.

    Every row attends to itself and its bit-flip partner.  The support is a
    disjoint union of dense 2-by-2 blocks, so it has exactly ``2 * n`` active
    edges and admits score-dependent doubly stochastic plans.
    """
    if n <= 0 or n & (n - 1):
        raise ValueError("n must be a positive power of two")
    if bit <= 0 or bit >= n or bit & (bit - 1):
        raise ValueError("bit must be a power of two smaller than n")
    rows = np.arange(n)
    mask = np.eye(n, dtype=bool)
    mask[rows, rows ^ bit] = True
    return validate_support(mask)


def bit_pair_schedule(n: int) -> list[np.ndarray]:
    """Return the logarithmic square bit-pair schedule for ``n`` tokens."""
    if n <= 0 or n & (n - 1):
        raise ValueError("n must be a positive power of two")
    return [bit_pair_mask(n, 1 << index) for index in range(int(np.log2(n)))]


def bit_partition_mask(n: int, bit: int) -> np.ndarray:
    """Rectangular one-row-per-pair support for an exact bit projection.

    Each token column has degree one and each latent row joins one XOR pair.
    With column marginal ``b`` and row marginal ``support @ b``, the feasible
    plan is unique.  The round trip ``C.T @ R`` is therefore the conditional
    expectation that averages within each pair, independently of scores.
    """
    square = bit_pair_mask(n, bit)
    partners = np.arange(n) ^ bit
    representatives = np.flatnonzero(np.arange(n) < partners)
    support = square[representatives]
    return validate_support(support)


def bit_partition_schedule(n: int) -> list[np.ndarray]:
    """Return exact rectangular coordinate projections for all XOR bits."""
    if n <= 0 or n & (n - 1):
        raise ValueError("n must be a positive power of two")
    return [bit_partition_mask(n, 1 << index) for index in range(int(np.log2(n)))]


def cyclic_pair_mask(n: int, offset: int = 1) -> np.ndarray:
    """Degree-two square support formed by identity and one cyclic shift."""
    if n <= 1:
        raise ValueError("n must be greater than one")
    if offset % n == 0:
        raise ValueError("offset must be nonzero modulo n")
    rows = np.arange(n)
    mask = np.eye(n, dtype=bool)
    mask[rows, (rows + offset) % n] = True
    return validate_support(mask)


def de_bruijn_like_mask(n: int) -> np.ndarray:
    """Directed de-Bruijn-like support, symmetrized for square Sinkhorn tests."""
    if n <= 0:
        raise ValueError("n must be positive")
    mask = np.eye(n, dtype=bool)
    rows = np.arange(n)
    for multiplier in (2, 2 * n - 1):
        mask[rows, (multiplier * rows) % n] = True
        mask[(multiplier * rows) % n, rows] = True
    mask[rows, (rows + 1) % n] = True
    mask[(rows + 1) % n, rows] = True
    return validate_support(mask)


def register_augmented_mask(
    n_tokens: int, local_radius: int, num_registers: int
) -> np.ndarray:
    """Token-local support plus register rows/columns that act as shared buses."""
    if n_tokens <= 0:
        raise ValueError("n_tokens must be positive")
    if num_registers < 0:
        raise ValueError("num_registers must be non-negative")
    n = n_tokens + num_registers
    mask = np.zeros((n, n), dtype=bool)
    token = np.arange(n_tokens)
    mask[np.ix_(token, token)] = band_mask(n_tokens, local_radius, wrap=True)
    if num_registers:
        registers = np.arange(n_tokens, n)
        mask[registers, :] = True
        mask[:, registers] = True
    return validate_support(mask)


def support_stats(mask: np.ndarray) -> dict[str, float]:
    """Basic support statistics for reports."""
    support = validate_support(mask)
    row_degrees = np.sum(support, axis=1)
    col_degrees = np.sum(support, axis=0)
    return {
        "rows": int(support.shape[0]),
        "cols": int(support.shape[1]),
        "edges": int(np.sum(support)),
        "density": float(np.mean(support)),
        "min_row_degree": int(np.min(row_degrees)),
        "max_row_degree": int(np.max(row_degrees)),
        "min_col_degree": int(np.min(col_degrees)),
        "max_col_degree": int(np.max(col_degrees)),
    }


def column_overlap_graph(mask: np.ndarray) -> np.ndarray:
    """Undirected graph where columns are adjacent when they share a support row."""
    support = validate_support(mask)
    graph = (support.T @ support) > 0
    np.fill_diagonal(graph, False)
    return graph


def _component_distances(graph: np.ndarray, start: int) -> np.ndarray:
    distances = np.full(graph.shape[0], -1, dtype=int)
    distances[start] = 0
    frontier = [start]
    while frontier:
        node = frontier.pop(0)
        for neighbor in np.flatnonzero(graph[node]):
            if distances[neighbor] < 0:
                distances[neighbor] = distances[node] + 1
                frontier.append(int(neighbor))
    return distances


def support_graph_descriptors(mask: np.ndarray) -> dict[str, float | int]:
    """Density, degree, diameter, and spectral descriptors for support topology."""
    support = validate_support(mask)
    stats = support_stats(support)
    graph = column_overlap_graph(support)
    n = graph.shape[0]
    seen = np.zeros(n, dtype=bool)
    component_count = 0
    diameter = 0
    for node in range(n):
        if seen[node]:
            continue
        component_count += 1
        distances = _component_distances(graph, node)
        component = distances >= 0
        seen[component] = True
        for source in np.flatnonzero(component):
            source_distances = _component_distances(graph, int(source))
            finite = source_distances[component]
            diameter = max(diameter, int(np.max(finite)))

    adjacency = graph.astype(float) + np.eye(n, dtype=float)
    degrees = np.sum(adjacency, axis=1)
    normalized = adjacency / np.sqrt(degrees[:, None] * degrees[None, :])
    eigvals = np.linalg.eigvalsh(normalized)
    eigvals = np.sort(eigvals)[::-1]
    spectral_gap = 0.0 if n < 2 else max(0.0, 1.0 - float(eigvals[1]))
    disconnected_diameter = int(n + 1)
    return {
        **stats,
        "avg_row_degree": float(np.mean(np.sum(support, axis=1))),
        "avg_col_degree": float(np.mean(np.sum(support, axis=0))),
        "column_overlap_components": int(component_count),
        "column_overlap_diameter": int(
            diameter if component_count == 1 else disconnected_diameter
        ),
        "column_overlap_spectral_gap": float(spectral_gap),
    }
