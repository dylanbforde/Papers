"""Reproducible numerical checks for Support Topology and Gradient Mixing in Sinkhorn Layers."""

from __future__ import annotations

import numpy as np

from src.topology_masks import (
    band_mask,
    bit_partition_schedule,
    butterfly_mask,
    dense_mask,
    disconnected_blocks_mask,
    random_regular_mask,
    register_augmented_mask,
    support_graph_descriptors,
)
from src.transport_topology import (
    derivative_kernel,
    dobrushin_tau,
    global_shared_component,
    logsumexp_masked,
    sinkhorn_plan,
    topology_report,
    uniform_marginals,
)


def check_finite_cycle_vjp(rng: np.random.Generator) -> None:
    print("=== 1. Finite-Cycle Two-Plan Jacobian & Adjoint Check (n=6, band r=1) ===")
    support = band_mask(6, 1, wrap=True)
    score = np.where(support, 0.4 * rng.normal(size=(6, 6)), 0.0)
    a_raw = np.array([0.8, 1.3, 0.9, 1.2, 1.1, 0.7])
    b_raw = np.array([1.1, 0.7, 1.4, 0.8, 1.0, 1.0])
    a = a_raw / np.sum(a_raw)
    b = b_raw / np.sum(b_raw)
    v0 = rng.normal(size=6) * 0.6

    def cycle(v_in: np.ndarray, s_in: np.ndarray, a_in: np.ndarray, b_in: np.ndarray):
        u_plus = np.log(a_in) - logsumexp_masked(s_in + v_in[None, :], support, axis=1)
        p_r = np.where(support, np.exp(s_in + u_plus[:, None] + v_in[None, :]), 0.0)
        v_plus = np.log(b_in) - logsumexp_masked(s_in + u_plus[:, None], support, axis=0)
        p_c = np.where(support, np.exp(s_in + u_plus[:, None] + v_plus[None, :]), 0.0)
        return v_plus, p_r, p_c

    v_plus, p_r, p_c = cycle(v0, score, a, b)
    r_mat = p_r / a[:, None]
    c_mat = p_c / b[None, :]
    m_two_plan = c_mat.T @ r_mat
    m_wrong_single = c_mat.T @ (p_c / a[:, None])

    eta_plus = rng.normal(size=6)
    eta_plus -= np.mean(eta_plus)
    eta_minus = m_two_plan.T @ eta_plus
    c_eta = c_mat @ eta_plus
    lambda_a = -c_eta
    lambda_b = eta_plus
    g_s = np.where(support, r_mat * c_eta[:, None] - c_mat * eta_plus[None, :], 0.0)

    eps = 1e-6
    fd_jac = np.zeros((6, 6))
    for j in range(6):
        dv = np.zeros(6)
        dv[j] = eps
        vp, _, _ = cycle(v0 + dv, score, a, b)
        vm, _, _ = cycle(v0 - dv, score, a, b)
        fd_jac[:, j] = (vp - vm) / (2.0 * eps)

    fd_v = np.zeros(6)
    for j in range(6):
        dv = np.zeros(6)
        dv[j] = eps
        vp, _, _ = cycle(v0 + dv, score, a, b)
        vm, _, _ = cycle(v0 - dv, score, a, b)
        fd_v[j] = np.dot(eta_plus, (vp - vm) / (2.0 * eps))

    fd_a = np.zeros(6)
    for i in range(6):
        da = np.zeros(6)
        da[i] = eps * a[i]
        vp, _, _ = cycle(v0, score, a + da, b)
        vm, _, _ = cycle(v0, score, a - da, b)
        fd_a[i] = np.dot(eta_plus, (vp - vm) / (2.0 * eps))

    fd_b = np.zeros(6)
    for j in range(6):
        db = np.zeros(6)
        db[j] = eps * b[j]
        vp, _, _ = cycle(v0, score, a, b + db)
        vm, _, _ = cycle(v0, score, a, b - db)
        fd_b[j] = np.dot(eta_plus, (vp - vm) / (2.0 * eps))

    fd_s = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if not support[i, j]:
                continue
            ds = np.zeros((6, 6))
            ds[i, j] = eps
            vp, _, _ = cycle(v0, score + ds, a, b)
            vm, _, _ = cycle(v0, score - ds, a, b)
            fd_s[i, j] = np.dot(eta_plus, (vp - vm) / (2.0 * eps))

    print(f"max|P_r - P_c|             = {np.max(np.abs(p_r - p_c)):.6e}")
    print(f"max|M_two_plan - FD|       = {np.max(np.abs(m_two_plan - fd_jac)):.6e}")
    print(f"max|M_wrong_single - FD|   = {np.max(np.abs(m_wrong_single - fd_jac)):.6e}")
    print(f"max|eta_minus - FD_v|      = {np.max(np.abs(eta_minus - fd_v)):.6e}")
    print(f"max|lambda_a - FD_a|       = {np.max(np.abs(lambda_a - fd_a)):.6e}")
    print(f"max|lambda_b - FD_b|       = {np.max(np.abs(lambda_b - fd_b)):.6e}")
    print(f"max|g_s - FD_s|            = {np.max(np.abs(g_s - fd_s)):.6e}")
    print()


def check_canonical_supports(rng: np.random.Generator) -> None:
    print("=== 2. Canonical Sparse Support Comparison (n=32, sigma=0.15) ===")
    supports = [
        ("Dense (32x32)", dense_mask(32)),
        ("Local band (r=2)", band_mask(32, 2, wrap=True)),
        ("Disconnected (2 blocks)", disconnected_blocks_mask(32, blocks=2)),
        ("Random 5-regular", random_regular_mask(32, 5, rng)),
        ("Local band (r=2) + 2 hubs", register_augmented_mask(30, 2, 2)),
        ("Butterfly (n=32)", butterfly_mask(32)),
    ]
    for name, mask in supports:
        rep = topology_report(name, mask, rng, score_scale=0.15, steps=2000, power=3)
        desc = support_graph_descriptors(mask)
        print(
            f"{name:26s} | E={rep['edges']:4d} ({rep['density']:.3f}) | "
            f"tau(M)={rep['tau']:.4f} | tau(M^3)={rep['tau_power']:.4f} | "
            f"decay_3={rep['decay_power']:.4f} | gamma={rep['shared_gamma']:.4f} | "
            f"diam={desc['column_overlap_diameter']:2d} | "
            f"max_res={max(rep['row_residual'], rep['col_residual']):.2e}"
        )

    # 5-step bit-partition hypercube schedule
    sched = bit_partition_schedule(32)
    b = np.full(32, 1.0 / 32.0)
    prod = np.eye(32)
    taus_single = []
    for mask in sched:
        a = mask.astype(float) @ b
        score = np.where(mask, 0.15 * rng.normal(size=mask.shape), 0.0)
        plan = sinkhorn_plan(score, mask, a, b, steps=500)
        m_t = derivative_kernel(plan, a, b)
        taus_single.append(dobrushin_tau(m_t))
        prod = m_t @ prod
    tau_prod = dobrushin_tau(prod)
    print(
        f"{'Bit-partition schedule':26s} | E={sum(int(np.sum(m)) for m in sched):4d} (5x32) | "
        f"per-layer tau={min(taus_single):.4f}..{max(taus_single):.4f} | "
        f"tau(prod)={tau_prod:.4e}"
    )
    print()


def check_face_lattice_stress() -> None:
    print("=== 3. Face-Lattice Dichotomy Stress Test (n=6, band r=1 + hub row 0) ===")
    support = band_mask(6, 1, wrap=True)
    support[0, :] = True  # Row 0 is a universal hub row

    # Adversarial boundary face {(0,0), (3,3)} U ({1,2}x{1,2}) U ({4,5}x{4,5})
    # inside support: four disconnected components failing two-hop column overlap
    # under uniform marginals.
    face_mask = np.zeros((6, 6), dtype=bool)
    face_mask[0, 0] = True
    face_mask[1:3, 1:3] = True
    face_mask[3, 3] = True
    face_mask[4:6, 4:6] = True
    assert np.all(face_mask <= support), "face_mask must be a subset of support"

    score_dir = np.where(support, np.where(face_mask, 1.0, -1.0), 0.0)

    a_light, b_light = uniform_marginals(6, 6)
    a_heavy = np.array([0.88, 0.024, 0.024, 0.024, 0.024, 0.024])
    b_heavy = np.full(6, 1.0 / 6.0)

    # Heavy hub bound from Corollary (Heavy Global Hub)
    delta_j = a_heavy[0] + b_heavy - 1.0
    gamma_h = np.min(delta_j / b_heavy) * np.sum(delta_j) / a_heavy[0]
    tau_bound = 1.0 - gamma_h
    print(f"Heavy hub analytical bound: delta_j={delta_j[0]:.6f}, gamma_h={gamma_h:.6f}, tau <= {tau_bound:.6f}")

    for L in [0.0, 2.0, 5.0, 10.0, 15.0]:
        s = L * score_dir
        p_light = sinkhorn_plan(s, support, a_light, b_light, steps=100000)
        m_light = derivative_kernel(p_light, a_light, b_light)
        res_light = max(np.max(np.abs(p_light.sum(1) - a_light)), np.max(np.abs(p_light.sum(0) - b_light)))

        p_heavy = sinkhorn_plan(s, support, a_heavy, b_heavy, steps=100000)
        m_heavy = derivative_kernel(p_heavy, a_heavy, b_heavy)
        res_heavy = max(np.max(np.abs(p_heavy.sum(1) - a_heavy)), np.max(np.abs(p_heavy.sum(0) - b_heavy)))

        print(
            f"L={L:4.1f} | Light hub tau={dobrushin_tau(m_light):.6f} (res={res_light:.2e}) | "
            f"Heavy hub tau={dobrushin_tau(m_heavy):.6f} <= {tau_bound:.6f} (res={res_heavy:.2e}, "
            f"min_hub_edge={np.min(p_heavy[0]):.6f} >= {delta_j[0]:.6f})"
        )
    print()


def check_closed_form_formulas() -> None:
    print("=== 4. Closed-Form Equal-Score Formulas Verification ===")
    # (a) Circular local band n=32, r=2
    n, r = 32, 2
    d = 2 * r + 1
    mask = band_mask(n, r, wrap=True)
    a, b = uniform_marginals(n, n)
    p_band = sinkhorn_plan(np.zeros((n, n)), mask, a, b, steps=2000)
    m_band = derivative_kernel(p_band, a, b)
    k = np.arange(1, n)
    lam_theory = (np.sin(np.pi * k * d / n) / (d * np.sin(np.pi * k / n))) ** 2
    lam_theory_all = np.sort(np.concatenate(([1.0], lam_theory)))
    lam_num_all = np.sort(np.linalg.eigvalsh(m_band))
    print(f"Circular band (n=32, r=2) max eigenvalue diff = {np.max(np.abs(lam_num_all - lam_theory_all)):.6e}")
    print(f"Theoretical lambda_band = {np.max(lam_theory):.6f}")

    # (b) Register/hub augmented n_tok=30, r=2 (d=5), num_reg=2
    n_tok, rad, n_reg = 30, 2, 2
    d_tok = 2 * rad + 1
    N = n_tok + n_reg
    reg_mask = register_augmented_mask(n_tok, rad, n_reg)
    a_r, b_r = uniform_marginals(N, N)
    p_reg = sinkhorn_plan(np.zeros((N, N)), reg_mask, a_r, b_r, steps=2000)
    m_reg = derivative_kernel(p_reg, a_r, b_r)

    t = 2.0 * d_tok / (np.sqrt((n_tok - n_reg) ** 2 + 4 * n_reg * d_tok) + (n_tok - n_reg))
    A = 1.0 / (N * (d_tok + n_reg * t))
    B = t * A
    D = t * t * A
    p_theory = np.zeros((N, N))
    p_theory[:n_tok, :n_tok] = np.where(reg_mask[:n_tok, :n_tok], A, 0.0)
    p_theory[:n_tok, n_tok:] = B
    p_theory[n_tok:, :n_tok] = B
    p_theory[n_tok:, n_tok:] = D

    m_reg_bound = t * (d_tok + n_reg * t * t) / ((d_tok + n_reg * t) ** 2)
    gamma_reg = n_reg * m_reg_bound
    shared_gamma, _ = global_shared_component(m_reg)
    print(f"Register-hub (30+2, d=5) max plan diff        = {np.max(np.abs(p_reg - p_theory)):.6e}")
    print(
        f"gamma_reg={gamma_reg:.6f}, shared_gamma={shared_gamma:.6f}, "
        f"tau(M)={dobrushin_tau(m_reg):.6f} <= 1-gamma_reg={1.0 - gamma_reg:.6f}"
    )


def main() -> None:
    rng = np.random.default_rng(0)
    check_finite_cycle_vjp(rng)
    check_canonical_supports(rng)
    check_face_lattice_stress()
    check_closed_form_formulas()


if __name__ == "__main__":
    main()
