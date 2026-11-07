r"""
Periodic far field for the rotation/solid-harmonics FMM.

``gen_pbc_op`` assembles the renormalised lattice operator ``PBC_op``: a dense
``((p+1)^2, (p+1)^2)`` matrix mapping the root-box multipole to the local expansion of the
ENTIRE image lattice beyond the near field, by hierarchical doubling — each virtual level
aggregates a ``mul_facs``-times larger supercell of images (M2M) and translates its
well-separated shell in (M2L).

Assembly runs through the PRODUCTION rotation transforms (:func:`jaxfmm.kernels.laplace.transf.M2M`
/ :func:`~jaxfmm.kernels.laplace.transf.M2L`): each per-level operator is built by pushing identity
coefficient columns through the same rotate → (folded) z-translate → rotate-back pipeline the
FMM itself executes, so the operator is natively in the production basis — a separate
dense assembly in another basis convention risks subtle sign errors on the far shells
that a near-field-dominated tolerance hides (the explicit image-sum arbiter lives in
tests/test_pbc_op.py).

Numerical hygiene:

- **Mirror-parity mask** (:func:`_mirror_parity_mask`): the image patterns are rectangular
  cuboids, closed under all three axis mirrors, so operator entries coupling components of
  opposite mirror parity cancel EXACTLY — they are forced to zero instead of accumulating
  rounding noise over the doubling levels.
- **Monopole column**: the lattice sum over a net cell charge is divergent (the aggregated
  image count ``mul^3i`` outgrows the ``1/L`` decay, entries grow ~``mul^2i``), so with
  ``pbc_no_monopole=True`` (default) the whole ``[:, 0]`` column is zeroed PER LEVEL — for
  neutral cells the column is irrelevant, and leaving it in would poison even the neutral
  case through the ~1e-16 monopole residue once ``pbc_lvls`` is large. Charged cells need an
  Ewald-style neutralising background, which this operator does not provide.
- **Self-adjoint projection** (:func:`_symmetrise`): at k=0 the exact operator is
  transpose-symmetric in the sqrt(2) real basis, so the doubling's skew noise is removed by
  a plain ``(P + P^T)/2``.
- **Bloch (k != 0)**: the phased monopole column is kept (a Bloch charge wave has a net cell
  charge), and the conditionally convergent entries -- degree n_i + n_j <= d - 1 for a
  d-dimensional lattice -- are taken from Ewald sums (:func:`_bloch_low_degree`): summed by
  doubling they depend on the summation shape and break translation invariance.
- Everything else converges geometrically: with 3D periodicity restricted to CUBIC cells
  (enforced in ``gen_hierarchy``), the shell sums of all ``l < 4`` anisotropies vanish by
  cubic symmetry — exactly the conditionally-convergent terms that make arbitrary 3D cells
  ill-defined. ``gen_pbc_op`` therefore iterates to ``pbc_lvls`` (a CAP, not a target)
  with an early stop once the increment stalls at ``rtol``.
"""
import warnings

import jax.numpy as jnp
import numpy as np

from jaxfmm.kernels.laplace import eval_regular_basis
from jaxfmm.kernels.laplace.rotation import precompute_combs
from jaxfmm.kernels.laplace.transf import M2M, M2L


def gen_img_connectivity(L0_boxlen, theta, periodic_axes):
    r"""
    Connectivity information for periodic images (cf. ``gen_connectivity``).

    The generated patterns are rectangular cuboids, which helps mitigate numerical
    inaccuracies for the resulting PBC operator (closure under the axis mirrors, see
    :func:`_mirror_parity_mask`).

    Parameters
    ----------
    L0_boxlen : (1, 3) array
        Root-box side lengths.
    theta : float
        Opening-angle MAC used to decide well-separatedness.
    periodic_axes : tuple of int
        Axes with periodic boundary conditions.

    Returns
    -------
    img_dists : (n_img, 3) array
        Displacements of the non-well-separated images (the near shell).
    img_dists_ws : (n_ws, 3) array
        Displacements of the well-separated child images of the parent supercell.
    mul_facs : (3,) int array
        Supercell multiplication factor per axis.
    """
    R = jnp.linalg.norm(L0_boxlen/2)                           # r = R in this case, all boxes are the same
    num_images = jnp.int32((1+theta)/(theta*L0_boxlen) * R)[0] # lower bound of non-ws images in axis directions
    non_per = jnp.array([False if i in periodic_axes else True for i in range(3)])
    num_images = num_images.at[non_per].set(0)
    mul_facs = 2*num_images + 1                                # total number of boxes in axis directions
    ids = [slice(-num_images[i],num_images[i]+1) if i in periodic_axes else slice(0,1) for i in range(3)]
    img_ids = jnp.reshape(jnp.mgrid[ids].T,(-1,3))  # spawn all the images

    L0_boxlen_up = L0_boxlen * mul_facs   # boxlength on the parent level
    R_up = jnp.linalg.norm(L0_boxlen_up/2)
    num_images_up = jnp.int32((1+theta)/(theta*L0_boxlen_up) * R_up)[0] # upper bound of ws images in axis directions
    ids_up = [slice(-num_images_up[i],num_images_up[i]+1) if i in periodic_axes else slice(0,1) for i in range(3)]
    img_ids_up = jnp.reshape(jnp.mgrid[ids_up].T,(-1,3))                                # spawn parent images
    img_ids_up = img_ids_up[jnp.any(img_ids_up!=0,axis=-1)]                             # we take out the non-ws area
    img_ids_up = ((mul_facs*img_ids_up)[:,None,:] + img_ids[None,:,:]).reshape((-1,3))  # spawn children for each remaining parent

    return img_ids*L0_boxlen, img_ids_up*L0_boxlen, mul_facs


def with_basis_offsets(img_dists, basis_offsets):
    r"""Extend an image displacement list to a lattice-with-basis: the union over
    {0} + basis_offsets of (img_dists + delta). Used for sublattice-symmetric setups
    (e.g. the FCC primitive-cell demag: one meshed sphere, images of ALL sublattices
    live in the operator). Basis offsets must be half-cell-commensurate translations
    so the combined pattern keeps the cubic mirror closure (FCC: a/2<110>)."""
    if basis_offsets is None or len(basis_offsets) == 0:
        return img_dists
    offs = jnp.concatenate([jnp.zeros((1, 3)), jnp.asarray(basis_offsets, img_dists.dtype)])
    return (img_dists[None, :, :] + offs[:, None, :]).reshape(-1, 3)


def _mirror_parity_mask(p):
    r"""
    (dim, dim) boolean mask of operator entries ALLOWED by mirror symmetry.

    Every real solid-harmonic component is an eigenvector of the three axis mirrors
    (parity ±1, obtained numerically at a generic point); an image pattern closed under
    a mirror makes the summed operator commute with it, so entries coupling
    opposite-parity components vanish exactly.  The cuboid image patterns are closed
    under ALL three axis mirrors (non-periodic axes hold offset 0), hence the full
    product mask.
    """
    dim = (p + 1) ** 2
    pt = jnp.array([1.2345, 2.3456, 3.4567])       # generic point: no component vanishes
    base = eval_regular_basis(pt, p)
    mask = jnp.ones((dim, dim), bool)
    for ax in range(3):
        flip = pt.at[ax].multiply(-1.0)
        par = jnp.sign(eval_regular_basis(flip, p) / base)   # ±1 per component
        mask &= (par[:, None] * par[None, :]) > 0
    return mask


def _symmetrise(P, no_monopole):
    r"""``P <- (P + P^T) / 2``: project onto the self-adjoint part.

    The exact k=0 lattice operator is transpose-symmetric in the sqrt(2) real basis
    (:mod:`jaxfmm.kernels.laplace.basis`; measured 1e-14 on any inversion-symmetric image
    set). The projection removes the doubling's skew noise (the l=1 entries were off by
    ~1e-3 at p=5 before any symmetrisation) while keeping the exact operator fixed. In the
    old weighted basis the same plain average halved/doubled every m=0 <-> m!=0 entry — the
    ~1e-6 uniform-field floor of the element path under 2-D PBC, fixed in 50be93d by a
    metric-weighted projection that this basis makes unnecessary.

    With ``no_monopole`` the zeroed monopole column has no partner for the (finite)
    monopole row, so only the ``[1:, 1:]`` block is projected: row 0 (the potential's
    constant) stays as built, column 0 stays zero.
    """
    if no_monopole:
        B = P[1:, 1:]
        return P.at[1:, 1:].set(0.5 * (B + B.T)).at[:, 0].set(0.0)
    return 0.5 * (P + P.T)


def _phase_row(dists, bloch_k):
    r"""e^{i k . d} per displacement row (complex128 host array)."""
    k = np.asarray(bloch_k, np.float64).reshape(3)
    return np.exp(1j * (np.asarray(dists, np.float64) @ k))


def _bloch_lattice_sums(Lv, per, k, D, alpha=None):
    r"""Ewald lattice sums :math:`\sum_{R \ne 0} e^{ik\cdot R}\,\partial^\alpha |x|^{-1}(R)`, :math:`|\alpha| \le D`,
    over the lattice :math:`\{\sum_{a \in per} n_a L_a e_a\}` embedded in 3D.

    Returns ``[S0, S1, S2][:D + 1]``: scalar, (3,) gradient, (3, 3) Hessian (complex).
    1/|x| = erfc(a|x|)/|x| (real space) + erf(a|x|)/|x| (Poisson summation over the lattice
    subspace at zero perpendicular offset, minus its R = 0 value). Derivatives across the
    lattice subspace are not implemented beyond what vanishes by symmetry, so ``D <= d - 1``
    (exactly the conditionally convergent degrees). ``k`` must miss the reciprocal lattice
    (k on it is the plain periodic case: the G = k term diverges). ``alpha`` is the Ewald
    splitting parameter (default 2.5 / cell size; the result does not depend on it).
    """
    from scipy.special import erfc, exp1
    d = len(per)
    assert 0 <= D <= d - 1
    Lp = np.array([Lv[a] for a in per], np.float64)
    Om = float(np.prod(Lp))
    al = 2.5 / Om ** (1.0 / d) if alpha is None else alpha
    def grid(n):
        g = np.stack(np.meshgrid(*[np.arange(-m, m + 1) for m in n], indexing="ij"), -1).reshape(-1, d)
        v = np.zeros((g.shape[0], 3))
        v[:, list(per)] = g
        return v
    kp = np.zeros(3)
    kp[list(per)] = np.asarray(k, np.float64).reshape(3)[list(per)]
    R = grid(np.ceil(6.5 / (al * Lp)).astype(int) + 1) * Lv
    R = R[np.any(R != 0, axis=1)]
    r = np.linalg.norm(R, axis=1)
    ph = np.exp(1j * (R @ kp))
    ec = erfc(al * r)
    ex = 2 * al / np.sqrt(np.pi) * np.exp(-(al * r) ** 2)
    s1 = -ec / r ** 2 - ex / r
    s2 = 2 * ec / r ** 3 + ex * (2 / r ** 2 + 2 * al ** 2)
    q = kp + grid(np.ceil(13 * al * Lp / (2 * np.pi)).astype(int) + 1) * (2 * np.pi / Lv) * \
        np.isin(np.arange(3), per)
    qn = np.linalg.norm(q, axis=1)
    if qn.min() < 1e-12 * float(np.max(2 * np.pi / Lp)):
        raise ValueError("bloch_k lies on the reciprocal lattice: that is the plain periodic case "
                         "(k = 0), whose lattice sum is divergent — pass bloch_k = None.")
    g = {3: 4 * np.pi * np.exp(-qn ** 2 / (4 * al ** 2)) / qn ** 2,
         2: 2 * np.pi * erfc(qn / (2 * al)) / qn,
         1: exp1(qn ** 2 / (4 * al ** 2))}[d] / Om
    out = [np.sum(ph * ec / r) + np.sum(g) - 2 * al / np.sqrt(np.pi)]
    if D >= 1:
        out.append(np.einsum("n,na->a", ph * s1 / r, R) + np.einsum("n,na->a", -1j * g, q))
    if D >= 2:
        RR = np.einsum("na,nb->nab", R, R)
        out.append(np.einsum("n,nab->ab", ph * s2 / r ** 2, RR)
                   + np.einsum("n,nab->ab", ph * s1, np.eye(3)[None] / r[:, None, None] - RR / r[:, None, None] ** 3)
                   - np.einsum("n,nab->ab", g, np.einsum("na,nb->nab", q, q))
                   + 4 * al ** 3 / (3 * np.sqrt(np.pi)) * np.eye(3))
    return out


def _bloch_low_degree(L, theta, periodic_axes, p, bloch_k, first_shell_lvl):
    r"""Exact (Ewald) values of the conditionally convergent entries of the phased lattice
    operator: entry (i, j) couples harmonics of degrees n_i, n_j through the degree-(n_i + n_j)
    irregular harmonic, whose d-dimensional lattice sum converges only conditionally for
    n_i + n_j <= d - 1. A doubling (expanding-square) summation returns a summation-shape
    dependent value there, which is NOT translation invariant: moving a charge by a lattice
    vector (with its Bloch phase) changed the potential by 4 % at kL = 0.16 and 45 % at kL = 1.6.

    Returns ``(mask, values)``, both ((p+1)^2, (p+1)^2). Each M2L entry of degree n is a fixed
    linear functional of the Cartesian derivatives of 1/|R| of order n; the functional is fitted
    on random displacements through the production M2L (basis-convention safe) and applied to
    the Ewald sums of :func:`_bloch_lattice_sums` minus the images the tree handles itself.
    """
    unqs, invs = precompute_combs(p)
    per = tuple(periodic_axes)
    D = len(per) - 1
    Lv = np.asarray(L, np.float64).reshape(3)
    S = _bloch_lattice_sums(Lv, per, bloch_k, D)
    non_ws, ws, _ = gen_img_connectivity(jnp.asarray(Lv)[None], theta, per)
    excl = np.asarray(non_ws, np.float64) if first_shell_lvl == 0 else \
        np.concatenate([np.asarray(non_ws, np.float64), np.asarray(ws, np.float64)])
    excl = excl[np.any(excl != 0, axis=1)]
    kp = np.zeros(3)
    kp[list(per)] = np.asarray(bloch_k, np.float64).reshape(3)[list(per)]
    iu = np.triu_indices(3)

    def basis(R, n):                       # Cartesian derivatives of 1/|R| of order n, (N, nb)
        r = np.linalg.norm(R, axis=1)
        if n == 0:
            return (1 / r)[:, None]
        if n == 1:
            return -R / r[:, None] ** 3
        H = 3 * np.einsum("na,nb->nab", R, R) / r[:, None, None] ** 5 - np.eye(3)[None] / r[:, None, None] ** 3
        return H[:, iu[0], iu[1]]

    ph = np.exp(1j * (excl @ kp))
    tail = [np.atleast_1d(S[0]), S[1] if D >= 1 else None, S[2][iu] if D >= 2 else None][:D + 1]
    tail = [t - ph @ basis(excl, n) for n, t in enumerate(tail)]

    dim = (p + 1) ** 2
    deg = np.floor(np.sqrt(np.arange(dim))).astype(int)
    rng = np.random.default_rng(0)
    Rs = rng.standard_normal((24, 3))
    Rs *= ((2.0 + 2.0 * rng.uniform(size=24)) / np.linalg.norm(Rs, axis=1))[:, None]
    cnct = jnp.stack([jnp.arange(24), jnp.arange(24)], 1)
    trg = jnp.zeros((24, 3))
    eye = jnp.eye(dim)
    mask = np.zeros((dim, dim), bool)
    vals = np.zeros((dim, dim), complex)
    for c in range((D + 1) ** 2):
        col = np.asarray(M2L(jnp.broadcast_to(eye[c], (24, dim)), jnp.asarray(Rs), trg, cnct,
                             unqs, invs))                         # (24, dim): op_R[:, c] per R
        for i in np.flatnonzero(deg + deg[c] <= D):
            n = deg[i] + deg[c]
            B = basis(Rs, n)
            A, *_ = np.linalg.lstsq(B, col[:, i], rcond=None)
            res = np.abs(B @ A - col[:, i]).max()
            if res > 1e-9 * max(np.abs(col[:, i]).max(), 1e-300):
                raise RuntimeError(f"_bloch_low_degree: entry ({i}, {c}) is not a degree-{n} "
                                   f"harmonic of the displacement (fit residual {res:.1e})")
            mask[i, c] = True
            vals[i, c] = A @ tail[n]
    return mask, vals


def _m2l_op(dists, p, unqs, invs, phases=None):
    r"""Dense lattice M2L operator ``sum_d M2L(local at 0 <- multipole at d)`` for the image
    displacements ``dists``, assembled by pushing identity coefficient columns through the
    production rotation M2L (all images enter as one pair list; the production scatter does
    the f64 summation).  Columns are pushed in a plain PYTHON LOOP of eager jit calls —
    vmap-of-jit has produced nondeterministic NaNs on loaded CPU nodes (see ARCHITECTURE
    §7.3) and the loop exercises exactly the production-shaped kernels."""
    dim = (p + 1) ** 2
    n = dists.shape[0]
    cnct = jnp.stack([jnp.zeros(n, dtype=int), jnp.arange(n, dtype=int)], 1)   # default int width (no x32 truncation warning)
    trg = jnp.zeros((1, 3), dists.dtype)
    eye = jnp.eye(dim)
    if phases is None:
        cols = [np.asarray(M2L(jnp.broadcast_to(eye[c], (n, dim)), dists, trg, cnct,
                               unqs, invs)[0]) for c in range(dim)]
        return jnp.asarray(np.stack(cols, axis=1))     # column c = op @ e_c
    # Bloch: phase the pushed identity columns per image row (linearity: the summed
    # operator is sum_d e^{ik.d} M2L_d) — real+imag parts pushed separately so the
    # production-shaped real kernels stay untouched
    ph = np.asarray(phases)
    cols = []
    for c in range(dim):
        base = np.broadcast_to(np.asarray(eye[c]), (n, dim))
        re = np.asarray(M2L(jnp.asarray(base * ph.real[:, None]), dists, trg, cnct, unqs, invs)[0])
        im = np.asarray(M2L(jnp.asarray(base * ph.imag[:, None]), dists, trg, cnct, unqs, invs)[0])
        cols.append(re + 1j * im)
    return jnp.asarray(np.stack(cols, axis=1))


def _m2m_op(dists, p, unqs, invs, phases=None):
    r"""Dense aggregation M2M operator ``sum_d M2M(parent at 0 <- child at d)`` (the zero
    displacement passes through as identity: cart_to_sph is origin-safe and the shift
    factors r^j vanish). Python-loop assembly — see _m2l_op."""
    dim = (p + 1) ** 2
    n = dists.shape[0]
    zero = jnp.zeros((1, 3), dists.dtype)
    eye = jnp.eye(dim)
    if phases is None:
        cols = [np.asarray(M2M(jnp.broadcast_to(eye[c], (n, dim)), dists, zero,
                               unqs, invs)[0]) for c in range(dim)]
        return jnp.asarray(np.stack(cols, axis=1))
    ph = np.asarray(phases)
    cols = []
    for c in range(dim):
        base = np.broadcast_to(np.asarray(eye[c]), (n, dim))
        re = np.asarray(M2M(jnp.asarray(base * ph.real[:, None]), dists, zero, unqs, invs)[0])
        im = np.asarray(M2M(jnp.asarray(base * ph.imag[:, None]), dists, zero, unqs, invs)[0])
        cols.append(re + 1j * im)
    return jnp.asarray(np.stack(cols, axis=1))


def _l2l_op(dists, p, unqs, invs):
    r"""Dense L2L translation matrix/matrices summed over ``dists`` (parent local at 0 ->
    child local at d), by identity-column pushes through the production transf.L2L —
    python loop, not vmap (see _m2l_op)."""
    from jaxfmm.kernels.laplace.transf import L2L
    dim = (p + 1) ** 2
    n = dists.shape[0]
    eye = jnp.eye(dim)
    cols = [np.asarray(L2L(eye[c][None], jnp.zeros((1, 3), dists.dtype), dists,
                           unqs, invs)).sum(0) for c in range(dim)]
    return jnp.asarray(np.stack(cols, axis=1))


def gen_pbc_uniform(p, domain_extent, periodic_axes=(0, 1, 2), pbc_lvls=30):
    r"""Periodic far-tail bundle for the UNIFORM spherical-harmonic engine (same basis as
    the flexible engine — its operator tables are pinned to transf.* by test): the
    harmonic lattice operator for the cell (theta=0.77 -> the wrapped 27-neighbourhood
    split) + the virtual-root translation matrices (the cubic tree has no level-0 box):
    ``M2M_root[k]`` aggregates level-1 octant k -> cell multipole, ``L2L_root[k]``
    distributes the cell local back. Returns a dict for uniform_fmm.make_operators(pbc=...).

    Parameters
    ----------
    p : int
        Expansion order.
    domain_extent : float
        Side length of the CUBIC unit cell (the uniform tree's root box).
    periodic_axes : tuple of int
        Axes with periodic boundary conditions (default all three).
    pbc_lvls : int
        Cap on the number of virtual lattice levels (see :func:`gen_pbc_op`).

    Returns
    -------
    dict
        ``PBC_op`` ((p+1)^2, (p+1)^2), ``M2M_root`` and ``L2L_root``
        (8, (p+1)^2, (p+1)^2) arrays for ``uniform_fmm.make_operators(pbc=...)``.
    """
    from jaxfmm.kernels.laplace.rotation import precompute_combs
    unqs, invs = precompute_combs(p)
    L = jnp.full((1, 3), float(domain_extent))
    PBC_op, _ = gen_pbc_op(L, 0.77, tuple(periodic_axes), p, pbc_lvls, True)
    # level-1 box centres relative to the cell centre: (+-extent/4)^3; the octant->axis
    # mapping comes from the ACTUAL morton decode (bit-order convention lives there)
    from jaxfmm.trees.cubic import _morton_to_grid
    gx, gy, gz = _morton_to_grid(jnp.arange(8, dtype=jnp.uint64))
    g = np.stack([np.asarray(gx), np.asarray(gy), np.asarray(gz)], 1).astype(np.float64)
    offs = (g - 0.5) * (float(domain_extent) / 2.0)
    M2M_root = jnp.stack([_m2m_op(jnp.asarray(offs[k][None]), p, unqs, invs) for k in range(8)])
    L2L_root = jnp.stack([_l2l_op(jnp.asarray(offs[k][None]), p, unqs, invs) for k in range(8)])
    return dict(PBC_op=jnp.asarray(PBC_op), M2M_root=M2M_root, L2L_root=L2L_root)


def gen_pbc_op_direct(L, theta, periodic_axes, p, pbc_no_monopole,
                      n_rings=256, basis_offsets=None, batch=8192, bloch_k=None):
    r"""
    EXACT lattice-tail operator by direct shell summation + Richardson extrapolation.

    Sums the order-p M2L operator over every lattice image outside the level-0
    non-well-separated set (the same exclusion as the doubling path), in nested
    ring counts ``n_rings/4, n_rings/2, n_rings``, and removes the 1/N and 1/N^2
    ring-cutoff tails by two-stage Richardson extrapolation. Unlike hierarchical
    doubling there is NO intermediate p-truncated aggregation: each M2L(d) is the
    exact order-p block for its displacement, so the result is the exact lattice
    constant matrix up to the O(1/N^3) extrapolation residual (~1e-8 at the
    default) — and transpose-symmetric to the same level by construction.

    k = 0 only (a phased direct sum is oscillatory and needs proper Ewald
    smoothing — future work). 1D/2D periodicity; the leading surviving entry
    decay is r^-3 (dipole-dipole with the monopole column zeroed), which sets
    the 1/N tail.

    Parameters mirror :func:`gen_pbc_op` where shared. Returns ``(PBC_op,
    img_cnct_list)`` with the level-0 connectivity entry only (callers use the
    deeper entries solely for debugging).
    """
    bloch = bloch_k is not None and np.any(np.asarray(bloch_k) != 0.0)
    if bloch:
        return _pbc_op_direct_bloch(L, theta, periodic_axes, p, n_rings,
                                    basis_offsets, batch, jnp.asarray(bloch_k, jnp.float64))
    unqs, invs = precompute_combs(p)
    mask = _mirror_parity_mask(p)
    _offs = [jnp.zeros(3)] if basis_offsets is None else \
        [jnp.zeros(3)] + [jnp.asarray(o, jnp.float64) for o in basis_offsets]
    img_non_ws, img_ws, _ = gen_img_connectivity(L, theta, periodic_axes)

    per = tuple(periodic_axes)
    Lv = np.asarray(L, np.float64).reshape(3)
    # excluded integer offsets: origin + the level-0 non-ws images
    excl = set()
    for d in np.asarray(img_non_ws, np.float64).reshape(-1, 3):
        excl.add(tuple(int(round(d[a] / Lv[a])) for a in per))
    excl.add(tuple(0 for _ in per))

    Ns = [max(n_rings // 4, 8), max(n_rings // 2, 16), max(n_rings, 32)]
    sums, ring_lo = [], 0
    acc = jnp.zeros(((p + 1) ** 2, (p + 1) ** 2))
    for N in Ns:
        coords = []
        rng = range(-N, N + 1)
        if len(per) == 1:
            it = ((i,) for i in rng)
        elif len(per) == 2:
            it = ((i, j) for i in rng for j in rng)
        else:
            it = ((i, j, k) for i in rng for j in rng for k in rng)
        for t in it:
            if max(abs(c) for c in t) < ring_lo or t in excl:
                continue
            d = np.zeros(3)
            for a, c in zip(per, t):
                d[a] = c * Lv[a]
            coords.append(d)
        coords = np.asarray(coords, np.float64)
        for lo in range(0, len(coords), batch):
            db = jnp.asarray(coords[lo : lo + batch])
            acc = acc + sum(_m2l_op(db + o[None, :], p, unqs, invs) for o in _offs)
        sums.append(acc)
        ring_lo = N + 1
    # Richardson: P(N) = P_inf - A/N - B/N^2 + O(N^-3)
    P1, P2, P3 = sums  # at N/4, N/2, N
    R2 = 2.0 * sums[2] - sums[1]           # kills 1/N   (N vs N/2)
    R1 = 2.0 * sums[1] - sums[0]           # kills 1/N   (N/2 vs N/4)
    P = (4.0 * R2 - R1) / 3.0              # kills 1/N^2
    P = jnp.where(mask, P, 0.0)
    if pbc_no_monopole:
        P = P.at[:, 0].set(0.0)
        P = P.at[0, 0].set(0.0)
    P = _symmetrise(P, pbc_no_monopole)
    return P, [[img_non_ws, img_ws]]


def _pbc_op_direct_bloch(L, theta, periodic_axes, p, n_rings, basis_offsets, batch, bloch_k):
    r"""
    Exact PHASED lattice tail by Gaussian-regulated direct summation.

    The bare phased sum sum_R' e^{ik.R} S(R) is oscillatory (and its monopole
    row only conditionally convergent), so plain ring truncation + power-law
    Richardson fails. Instead each term is damped by exp(-eta |R|^2) — the
    Ewald screening half; at k != 0 (k off the reciprocal lattice) the phase
    cancellation makes the compensating reciprocal-space half unnecessary —
    and the physical limit is recovered by Richardson extrapolation in eta
    (the regulated sum is analytic in eta: P(eta) = P + a*eta + b*eta^2 + ...).
    Three eta levels with ratio 4 (ring counts N, 2N, 4N) eliminate the eta
    and eta^2 terms, leaving O(eta^3) ~ 1e-9-class residuals at the default.

    No mirror mask and the monopole column is KEPT — matching the doubling
    path's Bloch semantics (the phased monopole is physical; zeroing it
    discards the k-linear long-range response).
    """
    unqs, invs = precompute_combs(p)
    _offs = [jnp.zeros(3)] if basis_offsets is None else \
        [jnp.zeros(3)] + [jnp.asarray(o, jnp.float64) for o in basis_offsets]
    img_non_ws, img_ws, _ = gen_img_connectivity(L, theta, periodic_axes)

    per = tuple(periodic_axes)
    Lv = np.asarray(L, np.float64).reshape(3)
    Lmin = float(min(Lv[a] for a in per))
    excl = set()
    for d in np.asarray(img_non_ws, np.float64).reshape(-1, 3):
        excl.add(tuple(int(round(d[a] / Lv[a])) for a in per))
    excl.add(tuple(0 for _ in per))

    def regulated(N, eta):
        acc = jnp.zeros(((p + 1) ** 2, (p + 1) ** 2), jnp.complex128)
        rng = range(-N, N + 1)
        if len(per) == 1:
            it = ((i,) for i in rng)
        elif len(per) == 2:
            it = ((i, j) for i in rng for j in rng)
        else:
            it = ((i, j, k) for i in rng for j in rng for k in rng)
        coords = []
        for t in it:
            if t in excl:
                continue
            d = np.zeros(3)
            for a, c in zip(per, t):
                d[a] = c * Lv[a]
            coords.append(d)
        coords = np.asarray(coords, np.float64)
        w_all = np.exp(-eta * np.einsum("id,id->i", coords, coords))
        keep = w_all > 1e-14
        coords, w_all = coords[keep], w_all[keep]
        for lo in range(0, len(coords), batch):
            db = jnp.asarray(coords[lo : lo + batch])
            wb = jnp.asarray(w_all[lo : lo + batch])
            for o in _offs:
                ph = _phase_row(db + o[None, :], bloch_k) * wb
                acc = acc + _m2l_op(db + o[None, :], p, unqs, invs, phases=ph)
        return acc

    # eta ladder: weight ~1e-8 at the ring cutoff of each level
    etas = [(4.3 / (N * Lmin)) ** 2 for N in (n_rings // 4, n_rings // 2, n_rings)]
    P1 = regulated(n_rings // 4, etas[0])
    P2 = regulated(n_rings // 2, etas[1])
    P3 = regulated(n_rings, etas[2])
    R12 = (4.0 * P2 - P1) / 3.0            # kills the eta term   (ratio 4)
    R23 = (4.0 * P3 - P2) / 3.0
    P = (16.0 * R23 - R12) / 15.0          # kills the eta^2 term
    return P, [[img_non_ws, img_ws]]


def gen_pbc_op(L, theta, periodic_axes, p, pbc_lvls, pbc_no_monopole, rtol=1e-13,
               first_shell_lvl=0, basis_offsets=None, bloch_k=None, method="doubling",
               n_rings=256):
    r"""
    Generate the full PBC operator matrix by hierarchical doubling (see module docstring).

    Parameters
    ----------
    L : (1, 3) array
        Unit-cell side lengths.
    theta : float
        Opening-angle MAC for the image shells.
    periodic_axes : tuple of int
        Axes with periodic boundary conditions.
    p : int
        Expansion order.
    pbc_lvls : int
        CAP on the number of virtual levels: the loop stops early once the level
        increment has stalled (``|inc| <= rtol * |res|``).
    pbc_no_monopole : bool
        Zero the (divergent) monopole column per level (see module docstring).
    rtol : float
        PER-ENTRY relative convergence tolerance: the loop stops once the largest
        increment, measured against that entry's own accumulated magnitude, falls
        below it. Default 1e-13. The achievable floor is ~2e-15 (about 10x the f64
        epsilon), so asking for 1e-15 is unreachable and merely runs to ``pbc_lvls``.
        Loosening this is the setup-time dial: fewer levels, proportionally more
        relative error in the returned operator.
    first_shell_lvl : int
        Exclude the M2L shells of the first levels (e.g. a hybrid engine that handles
        the level-0 shell exactly elsewhere) while keeping the aggregation.

    Returns
    -------
    PBC_op : ((p+1)^2, (p+1)^2) array
        Renormalised lattice operator (root multipole -> lattice-tail local).
    img_cnct_list : list
        One ``[img_non_ws, img_ws]`` entry per REALISED level.
    """
    if method == "direct":
        # exact lattice constants (direct + Richardson): the arbiter path. k=0
        # and first_shell_lvl=0 only. The doubling's ~1e-3 pathology (p=5) was pure
        # skew, removed by _symmetrise; compare the two PER
        # ENTRY (a Frobenius norm hides the O(1) l=1 block behind entries of 1e9+).
        # Use "direct" as reference, at low p, or as the base for the future
        # phased-Ewald k!=0 tail.
        if first_shell_lvl != 0:
            raise NotImplementedError("method='direct' does not support first_shell_lvl")
        return gen_pbc_op_direct(L, theta, periodic_axes, p, pbc_no_monopole,
                                 n_rings=n_rings, basis_offsets=basis_offsets,
                                 bloch_k=bloch_k)
    unqs, invs = precompute_combs(p)
    mask = _mirror_parity_mask(p)
    bloch = bloch_k is not None and np.any(np.asarray(bloch_k) != 0.0)
    if not bloch:
        bloch_k = None
    # lattice-with-basis: every M2L shell is summed over {0}+offsets (the offset is a
    # FIXED displacement — it does not scale with the virtual level), while the M2M
    # aggregation uses the plain cubic offsets (children sit at RELATIVE positions, the
    # common shift cancels). The parity mask stays valid for the SUMMED shells only
    # (the union has cubic symmetry for half-cell bases; a single offset does not), so
    # clean() must always act after the offset sum — which is how it is applied below.
    _offs = [jnp.zeros(3)] if basis_offsets is None else         [jnp.zeros(3)] + [jnp.asarray(o, jnp.float64) for o in basis_offsets]
    # Bloch: the entries of degree n_i + n_j <= d - 1 are conditionally convergent lattice sums;
    # doubling returns a summation-shape artefact for them, so they come from Ewald instead
    # (_bloch_low_degree) and are left out of the stopping test (they never settle)
    ewald = None
    if bloch_k is not None:
        if basis_offsets is None and first_shell_lvl <= 1:
            ewald = _bloch_low_degree(L, theta, periodic_axes, p, bloch_k, first_shell_lvl)
        else:
            warnings.warn("gen_pbc_op: Bloch lattice tail with basis_offsets or first_shell_lvl > 1 "
                          "keeps the doubling's conditionally convergent low-degree entries, which "
                          "depend on the summation shape (not translation invariant).",
                          RuntimeWarning, stacklevel=2)
    _settle = jnp.asarray(~ewald[0]) if ewald is not None else True
    _deg = np.floor(np.sqrt(np.arange((p + 1) ** 2))).astype(int)
    _kmag = 0.0 if bloch_k is None else \
        float(np.linalg.norm(np.asarray(bloch_k, np.float64).reshape(3)[list(periodic_axes)]))

    def _block_max(a):                     # max |entry| over each (degree_i, degree_j) block
        b = jnp.zeros((p + 1, p + 1)).at[_deg[:, None], _deg[None, :]].max(a)
        return b[_deg[:, None], _deg[None, :]]

    def _m2l_shell(dists):
        if bloch_k is None:
            return sum(_m2l_op(dists + o[None, :], p, unqs, invs) for o in _offs)
        return sum(_m2l_op(dists + o[None, :], p, unqs, invs,
                           phases=_phase_row(dists + o[None, :], bloch_k)) for o in _offs)

    def clean(op):
        # the mirror-parity mask encodes symmetries of the REAL lattice sum; a phased
        # (k != 0) shell has no mirror closure — never mask it. The monopole column is
        # kept under Bloch: at k != 0 the cell charge is legitimately nonzero (a Bloch
        # charge wave), and the phased monopole sum CONVERGES — the supercell aggregate
        # is a structure factor sum_c e^{ik.c} that decays instead of growing mul^3i.
        # Zeroing it discards the k-linear long-range response (measured: the film DE
        # dispersion collapses, h_yy ~ k^2 instead of ~k).
        if bloch_k is None:
            op = jnp.where(mask, op, 0.0)
            if pbc_no_monopole:                    # divergent monopole column, zero PER LEVEL
                op = op.at[:, 0].set(0.0)
        return op

    import os
    _dbg = os.environ.get("JAXFMM_PBC_DEBUG", "") != ""
    img_non_ws, img_ws, mul_facs = gen_img_connectivity(L, theta, periodic_axes)
    # first_shell_lvl > 0: EXCLUDE the shells of the first levels (e.g. the KIFMM hybrid
    # handles the level-0 shell exactly in the surface basis) but KEEP the aggregation.
    res = clean(_m2l_shell(img_ws)) if first_shell_lvl == 0 else \
          jnp.zeros(((p + 1) ** 2, (p + 1) ** 2))
    if _dbg:
        print(f"[pbc dbg] lvl0: res finite {bool(jnp.isfinite(res).all())} "
              f"norm {float(jnp.linalg.norm(res)):.3e}", flush=True)
    img_cnct_list = [[img_non_ws, img_ws]]
    M2M_cum = jnp.eye((p + 1) ** 2, dtype=jnp.complex128 if bloch_k is not None else None)
    prev_rel = float("inf")
    stopped = None
    for i in range(1, max(int(pbc_lvls), 0) + 1):
        # AGGREGATE FIRST, with the PREVIOUS level's non-ws offsets: the constituents of
        # the level-i supercell sit at c*L_{i-1}, not c*L_i. Aggregating with the NEW
        # scale's offsets places the children mul_facs times too far apart — a deep-level
        # bug that Madelung-type checks cannot see (the deep-level share of that sum is
        # tiny); only a per-level lattice-operator arbiter catches it.
        if bloch_k is None:
            M2M_cum = jnp.where(mask, _m2m_op(img_non_ws, p, unqs, invs), 0.0) @ M2M_cum
        else:
            M2M_cum = _m2m_op(img_non_ws, p, unqs, invs,
                              phases=_phase_row(img_non_ws, bloch_k)) @ M2M_cum
        L = L * mul_facs
        img_non_ws, img_ws, mul_facs = gen_img_connectivity(L, theta, periodic_axes)
        if i < first_shell_lvl:
            img_cnct_list.append([img_non_ws, img_ws])
            continue
        inc = clean(_m2l_shell(img_ws) @ M2M_cum)
        # ONE currency for every stopping decision: the largest PER-ENTRY increment,
        # relative to that entry's own accumulated magnitude. A Frobenius norm is
        # dominated by the largest entries, and this operator's entries span >1e21
        # already at p=6 (the solid-harmonic translation coefficients grow factorially
        # with n+n'), so a norm-based test reports "converged"/"noise" on behalf of the
        # big entries while the O(1) ones are still accumulating. Entries below the
        # accumulation's round-off floor cannot converge relatively and are held to that
        # floor instead.
        _a = jnp.abs(res)
        _den = jnp.maximum(_a, jnp.finfo(_a.dtype).eps * jnp.max(_a))
        if bloch_k is not None:
            # no mirror mask under Bloch: entries that vanish by lattice symmetry as k -> 0
            # (e.g. monopole <- degree-2 m=2 on a square cell) hover at round-off and never
            # settle relatively; hold each entry to round-off of its degree block instead
            _den = jnp.maximum(_den, jnp.finfo(_a.dtype).eps / rtol * _block_max(_a))
        inc_rel = float(jnp.max(jnp.where(_settle, jnp.abs(inc) / _den, 0.0)))
        # M2M_cum is the quantity that overflows if the loop runs too deep: check it
        # directly rather than inferring it from the increment's behaviour.
        finite = bool(jnp.isfinite(inc).all()) and bool(jnp.isfinite(M2M_cum).all())
        if _dbg:
            _r = np.asarray(jnp.where(_settle, jnp.abs(inc) / _den, 0.0))
            _e = np.unravel_index(int(np.argmax(_r)), _r.shape)
            print(f"[pbc dbg] lvl{i}: inc_rel {inc_rel:.3e} at entry {_e} finite {finite}", flush=True)
        # Stop WITHOUT accumulating on: non-finite, or increments no longer shrinking
        # (the noise floor — adding it would add noise). CPU BLAS reduction order is
        # run-to-run nondeterministic under load, so a marginal flip must not be allowed
        # to carry the loop deep enough for M2M_cum to overflow.
        # Bloch: below the coherence length ~1/|k| the phased supercell sums still grow
        # (the cancellation that makes them converge has not set in), so a rising increment
        # there is not the noise floor (k = 6e-4: increments ~2 up to level 6, converged at 20)
        coherent = bloch_k is not None and \
            float(np.min(np.asarray(L).reshape(3)[list(periodic_axes)])) * _kmag < 2.0
        if not finite or (inc_rel >= prev_rel and not coherent):
            stopped = "non-finite increment" if not finite else "noise floor"
            break
        res = res + inc
        img_cnct_list.append([img_non_ws, img_ws])
        if inc_rel <= rtol:
            stopped = "converged"
            break
        prev_rel = inc_rel
    else:
        stopped = "level cap"
    # A silent early stop is what made the old criterion dangerous: the caller could not
    # tell a converged tail from a truncated one. Say so. (Pre-fix history: the norm-based
    # test truncated the tail at 1.4e-3 for p >= 10 -- mesh- and parameter-independent, and
    # immune to every accuracy knob including the err_tol ladder's own R7-R9.)
    if stopped != "converged":
        warnings.warn(
            f"gen_pbc_op: lattice tail did not converge (stopped on {stopped} after "
            f"{len(img_cnct_list) - 1} levels; largest per-entry increment "
            f"{prev_rel:.2e} vs rtol {rtol:.1e}). The operator carries that much "
            f"relative error. Raise pbc_lvls, or loosen rtol if this accuracy suffices.",
            RuntimeWarning, stacklevel=2)
    if ewald is not None:
        res = jnp.where(jnp.asarray(ewald[0]), jnp.asarray(ewald[1]), res)
    if pbc_no_monopole and bloch_k is None:        # keep the historic flag semantics on [0,0]
        res = res.at[0, 0].set(0.0)                # (Bloch: the phased monopole is physical)
    if bloch_k is None:
        # The exact lattice operator is transpose-symmetric at k=0 (sqrt(2) real
        # basis). The skew part is pure doubling noise (6e-4 max relative at p=8,
        # 1e-8 at p=12, zero on l<=3) AND breaks the mode="diff" energy-gradient
        # property, which pumps low-alpha dynamics (eps ~ 1e-3 vs the eps <~ alpha
        # stability budget; see dev/PBC_SKEW_SYMMETRY_HANDOFF.md). Projecting onto
        # the self-adjoint part deletes it at zero runtime cost.
        # At k != 0 the pairing is not this simple (measured: neither P^T nor P^H
        # nor parity-twisted forms match) — blind symmetrisation is WRONG there;
        # the Bloch eigen route symmetrises at the operator level instead.
        res = _symmetrise(res, pbc_no_monopole)
    return res, img_cnct_list
