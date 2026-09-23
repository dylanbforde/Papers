from __future__ import annotations

import numpy as np


def logsumexp_masked(values: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    masked = np.where(mask, values, -np.inf)
    max_values = np.max(masked, axis=axis, keepdims=True)
    shifted = np.where(mask, np.exp(masked - max_values), 0.0)
    sums = np.sum(shifted, axis=axis, keepdims=True)
    return np.squeeze(np.log(sums) + max_values, axis=axis)


def sinkhorn_plan(
    score: np.ndarray,
    support: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    steps: int = 1500,
) -> np.ndarray:
    """Run log-domain Sinkhorn on a fixed feasible support."""
    score = np.asarray(score, dtype=float)
    support = np.asarray(support, dtype=bool)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    u = np.zeros_like(a)
    log_a = np.log(a)
    log_b = np.log(b)
    for _ in range(steps):
        v = log_b - logsumexp_masked(score + u[:, None], support, axis=0)
        u = log_a - logsumexp_masked(score + v[None, :], support, axis=1)
    return np.where(support, np.exp(score + u[:, None] + v[None, :]), 0.0)


def derivative_kernel(plan: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the column-to-column full-step kernel M = C^T R."""
    plan = np.asarray(plan, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    r_kernel = plan / a[:, None]
    c_kernel = plan / b[None, :]
    return c_kernel.T @ r_kernel


def biregular_two_step_kernel(support: np.ndarray) -> np.ndarray:
    """Return the pure-topology kernel for a biregular equal-score support."""
    support = np.asarray(support, dtype=bool)
    if support.ndim != 2:
        raise ValueError("support must be two-dimensional")
    row_degrees = np.sum(support, axis=1)
    col_degrees = np.sum(support, axis=0)
    if row_degrees.size == 0 or col_degrees.size == 0:
        raise ValueError("support must be nonempty")
    if np.any(row_degrees != row_degrees[0]) or np.any(col_degrees != col_degrees[0]):
        raise ValueError("support must be biregular")
    row_degree = int(row_degrees[0])
    col_degree = int(col_degrees[0])
    if row_degree <= 0 or col_degree <= 0:
        raise ValueError("support must have positive row and column degree")
    common_neighbors = support.astype(float).T @ support.astype(float)
    return common_neighbors / float(row_degree * col_degree)


def active_plan_distortion(plan: np.ndarray, support: np.ndarray) -> float:
    """Return max edgewise distortion from the uniform active-edge plan."""
    plan = np.asarray(plan, dtype=float)
    support = np.asarray(support, dtype=bool)
    if plan.shape != support.shape:
        raise ValueError("plan and support must have the same shape")
    edge_count = int(np.sum(support))
    if edge_count <= 0:
        raise ValueError("support must be nonempty")
    active_values = edge_count * plan[support]
    if np.any(active_values <= 0.0):
        raise ValueError("active plan entries must be positive")
    return float(max(np.max(active_values), np.max(1.0 / active_values)))


def minorization_dobrushin_bound(
    reference: np.ndarray,
    *,
    delta: float,
) -> float:
    """Bound tau(M) when a row-stochastic M satisfies M >= delta * reference."""
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must lie in [0, 1]")
    return float(min(1.0, 1.0 - delta * (1.0 - dobrushin_tau(reference))))


def zero_mass_spectral_rate(matrix: np.ndarray, *, tolerance: float = 1e-10) -> float:
    """Return the largest absolute eigenvalue on the zero-mean subspace."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=0.0):
        raise ValueError("matrix must be symmetric")
    eigvals = np.linalg.eigvalsh(matrix)
    if eigvals.size <= 1:
        return 0.0

    return float(np.max(np.abs(eigvals[:-1])))


def hub_floor_certificate(
    plan: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    hubs: list[int] | np.ndarray,
) -> dict[str, float | int | bool]:
    """Compute the realized-plan lower bound in the mass-floored hub theorem."""
    plan = np.asarray(plan, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    hubs = np.asarray(sorted(set(int(hub) for hub in hubs)), dtype=int)
    if hubs.size == 0:
        raise ValueError("at least one hub is required")
    if np.any(hubs < 0) or np.any(hubs >= plan.shape[1]):
        raise ValueError("hub index out of range")

    pair_floors = []
    for alpha in range(plan.shape[1]):
        for hub in hubs:
            witnesses = np.minimum(plan[:, alpha], plan[:, hub])
            pair_floors.append(float(np.max(witnesses)))
    p_min = float(np.min(pair_floors))
    delta = p_min**2 / float(np.max(a) * np.max(b))
    gamma_bound = min(1.0, float(hubs.size) * delta)
    matrix = derivative_kernel(plan, a, b)
    shared_gamma, _ = global_shared_component(matrix)
    return {
        "hub_count": int(hubs.size),
        "hub_floor_p_min": p_min,
        "hub_floor_delta": float(delta),
        "hub_floor_gamma_bound": float(gamma_bound),
        "hub_floor_actual_min": float(np.min(matrix[:, hubs])),
        "shared_gamma": float(shared_gamma),
        "tau": float(dobrushin_tau(matrix)),
        "certificate_ok": bool(
            np.min(matrix[:, hubs]) + 1e-12 >= delta
            and shared_gamma + 1e-12 >= gamma_bound
            and dobrushin_tau(matrix) <= 1.0 - gamma_bound + 1e-12
        ),
    }


def dobrushin_tau(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    max_dist = 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            max_dist = max(max_dist, float(np.sum(np.abs(matrix[i] - matrix[j]))))
    return 0.5 * max_dist


def global_shared_component(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    """Return largest row-common component mass gamma and distribution q."""
    matrix = np.asarray(matrix, dtype=float)
    lower = np.min(matrix, axis=0)
    gamma = float(np.sum(lower))
    if gamma <= 0.0:
        return 0.0, np.full(matrix.shape[1], 1.0 / matrix.shape[1], dtype=float)
    return gamma, lower / gamma


def tv_zero_mass(vector: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(vector)))


def random_zero_mass(rng: np.random.Generator, n: int) -> np.ndarray:
    vector = rng.normal(size=n)
    return vector - np.mean(vector)


def cotangent_decay_ratio(
    matrix: np.ndarray, eta: np.ndarray, *, power: int = 1
) -> float:
    propagated = np.asarray(eta, dtype=float)
    for _ in range(power):
        propagated = matrix.T @ propagated
    return tv_zero_mass(propagated) / max(tv_zero_mass(eta), 1e-30)


def zero_mass_tail_metrics(
    matrix: np.ndarray,
    rng: np.random.Generator,
    *,
    finite_depth: int,
    long_depth: int,
    reference_depth: int,
    trials: int,
) -> dict[str, float | int]:
    """Measure finite and long cotangent-tail errors against a long reference sum."""
    if not 0 < finite_depth < long_depth < reference_depth:
        raise ValueError("expected 0 < finite_depth < long_depth < reference_depth")
    if trials <= 0:
        raise ValueError("trials must be positive")

    matrix = np.asarray(matrix, dtype=float)
    finite_errors = []
    long_errors = []
    finite_decays = []
    long_decays = []
    finite_power = np.linalg.matrix_power(matrix, finite_depth)
    long_power = np.linalg.matrix_power(matrix, long_depth)
    for _ in range(trials):
        eta = random_zero_mass(rng, matrix.shape[0])
        term = eta.copy()
        finite_sum = np.zeros_like(eta)
        long_sum = np.zeros_like(eta)
        reference_sum = np.zeros_like(eta)
        for step in range(reference_depth):
            reference_sum = reference_sum + term
            if step < long_depth:
                long_sum = long_sum + term
            if step < finite_depth:
                finite_sum = finite_sum + term
            term = matrix.T @ term

        reference_norm = max(tv_zero_mass(reference_sum), 1e-30)
        finite_errors.append(tv_zero_mass(reference_sum - finite_sum) / reference_norm)
        long_errors.append(tv_zero_mass(reference_sum - long_sum) / reference_norm)
        finite_decays.append(
            tv_zero_mass(finite_power.T @ eta) / max(tv_zero_mass(eta), 1e-30)
        )
        long_decays.append(
            tv_zero_mass(long_power.T @ eta) / max(tv_zero_mass(eta), 1e-30)
        )

    return {
        "finite_depth": int(finite_depth),
        "long_depth": int(long_depth),
        "reference_depth": int(reference_depth),
        "trials": int(trials),
        "finite_tail_relative_error_median": float(np.median(finite_errors)),
        "long_tail_relative_error_median": float(np.median(long_errors)),
        "finite_depth_decay_median": float(np.median(finite_decays)),
        "long_depth_decay_median": float(np.median(long_decays)),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for one-dimensional numeric data, without a SciPy dependency."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return Spearman rank correlation with stable tie handling."""
    x_rank = rankdata(np.asarray(x, dtype=float))
    y_rank = rankdata(np.asarray(y, dtype=float))
    x_centered = x_rank - np.mean(x_rank)
    y_centered = y_rank - np.mean(y_rank)
    denom = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denom)


def uniform_marginals(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    mass = 1.0
    return np.full(rows, mass / rows, dtype=float), np.full(
        cols, mass / cols, dtype=float
    )


def score_on_support(
    rng: np.random.Generator,
    support: np.ndarray,
    *,
    scale: float = 0.15,
) -> np.ndarray:
    score = scale * rng.normal(size=support.shape)
    return np.where(support, score, 0.0)


def topology_report(
    name: str,
    support: np.ndarray,
    rng: np.random.Generator,
    *,
    score_scale: float = 0.15,
    steps: int = 1500,
    power: int = 3,
) -> dict[str, float | int | str]:
    """Run Sinkhorn, build M, and measure topology-certificate quantities."""
    support = np.asarray(support, dtype=bool)
    a, b = uniform_marginals(support.shape[0], support.shape[1])
    score = score_on_support(rng, support, scale=score_scale)
    plan = sinkhorn_plan(score, support, a, b, steps=steps)
    matrix = derivative_kernel(plan, a, b)
    tau = dobrushin_tau(matrix)
    matrix_power = np.linalg.matrix_power(matrix, power)
    tau_power = dobrushin_tau(matrix_power)
    eta = random_zero_mass(rng, matrix.shape[0])
    decay_1 = cotangent_decay_ratio(matrix, eta, power=1)
    decay_power = cotangent_decay_ratio(matrix, eta, power=power)
    gamma, _q = global_shared_component(matrix)
    return {
        "name": name,
        "rows": int(support.shape[0]),
        "cols": int(support.shape[1]),
        "edges": int(np.sum(support)),
        "density": float(np.mean(support)),
        "row_residual": float(np.max(np.abs(np.sum(plan, axis=1) - a))),
        "col_residual": float(np.max(np.abs(np.sum(plan, axis=0) - b))),
        "kernel_row_residual": float(np.max(np.abs(np.sum(matrix, axis=1) - 1.0))),
        "stationary_residual": float(np.max(np.abs(b @ matrix - b))),
        "tau": float(tau),
        "tau_power": float(tau_power),
        "tau_power_bound": float(tau**power),
        "decay_1": float(decay_1),
        "decay_power": float(decay_power),
        "shared_gamma": float(gamma),
        "shared_tau_bound": float(1.0 - gamma),
    }
