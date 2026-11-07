r"""Volume (continuous-source) FMM primitives for the Laplace kernel.

The source is a density :math:`f(y)` represented on each octree leaf by a
tensor-product polynomial sampled at quadrature nodes. The potential is

.. math:: u(x) = \int G(x, y)\, f(y)\, dy, \qquad G = \frac{1}{4\pi\,|x-y|}.

Key idea: the FAR field of a volume source reuses the point-charge KIFMM
unchanged. The upward check potential of a leaf is

.. math:: \mathrm{check}_i = \int_{\mathrm{box}} G(x_i, y)\, f(y)\, dy,

and since the check surface lies OUTSIDE the box the integrand is smooth, so a
Gauss–Legendre tensor rule turns the integral into a weighted sum over
quadrature nodes — i.e. the nodes act as point charges
:math:`q_k = f(y_k)\, w_k` and the existing point-charge P2M (with the per-leaf
h-scaling) produces the equivalent density. M2M / M2L / L2L / L2P are then
identical to the point-charge case.

What is NOT reused: the SELF-box near field
:math:`\int_{\mathrm{self}} G(x, y) f(y)\, dy` has an integrable :math:`1/r`
singularity and needs singular-aware quadrature (Duffy / graded rules below);
adjacent-box near fields are smooth but sharply peaked and use graded
composite Gauss rules.

Conventions match ``kifmm.operators``: surfaces are for a unit box (half-side
1); a leaf at half-side :math:`h` places quadrature nodes at
``node*h + centre`` with weight :math:`w\, h^3` (the volume Jacobian), and the
P2M decode carries an extra factor :math:`h`.
"""

import numpy as np

__all__ = ["tensor_gauss", "tensor_gauss_phys", "source_charges", "n_quad",
           "duffy_self_potential", "interp_matrix", "self_operator",
           "near_operator", "near_operator_general", "volume_operators",
           "volume_potential_uniform", "refinement_indicator", "refine_cells", "volume_potential_adaptive", "volume_potential_adaptive_jit", "volume_solver", "volume_solver_jit", "volume_evaluate_points", "build_volume_tree", "build_volume_tree_jit"]

_INV4PI = 1.0 / (4.0 * np.pi)


def _lagrange_1d(nodes, x):
    r"""1D Lagrange basis values: ``(len(x), len(nodes))`` with entry :math:`L_i(x_k)`."""
    nodes = np.asarray(nodes); x = np.asarray(x)
    n = nodes.shape[0]
    out = np.ones((x.shape[0], n))
    for i in range(n):
        for m in range(n):
            if m != i:
                out[:, i] *= (x - nodes[m]) / (nodes[i] - nodes[m])
    return out


def interp_matrix(n_per_dim, eval_local):
    r"""Tensor Lagrange interpolation matrix from the Gauss representation nodes.

    Interpolates from the ``n_per_dim**3`` Gauss nodes on :math:`[-1,1]^3` to
    arbitrary local points.

    Parameters
    ----------
    n_per_dim : int
        Gauss nodes per axis.
    eval_local : array_like
        Local evaluation points, shape ``(K, 3)``.

    Returns
    -------
    np.ndarray
        Matrix ``L``, shape ``(K, n_per_dim**3)``, with
        ``values_at_eval = L @ node_values``.
    """
    x1d, _ = np.polynomial.legendre.leggauss(n_per_dim)
    Lx = _lagrange_1d(x1d, eval_local[:, 0])     # (K, n)
    Ly = _lagrange_1d(x1d, eval_local[:, 1])
    Lz = _lagrange_1d(x1d, eval_local[:, 2])
    # tensor: L[k, (i*n+j)*n+l] = Lx[k,i]*Ly[k,j]*Lz[k,l]  (node order matches tensor_gauss)
    K = eval_local.shape[0]
    return (Lx[:, :, None, None] * Ly[:, None, :, None] * Lz[:, None, None, :]).reshape(K, -1)


def n_quad(n_per_dim: int) -> int:
    r"""Number of tensor Gauss nodes for ``n_per_dim`` points per axis.

    Parameters
    ----------
    n_per_dim : int
        Number of Gauss points per axis.

    Returns
    -------
    int
        Total node count of the tensor-product rule, ``n_per_dim**3``.
    """
    return n_per_dim ** 3


def tensor_gauss(n_per_dim: int):
    r"""Gauss–Legendre tensor-product quadrature on the unit box :math:`[-1, 1]^3`.

    Parameters
    ----------
    n_per_dim : int
        Number of Gauss points per axis (exact for degree
        :math:`\le 2 n - 1` per axis).

    Returns
    -------
    nodes : np.ndarray
        Node coordinates in :math:`[-1, 1]^3`, shape ``(n_per_dim**3, 3)``,
        float64.
    weights : np.ndarray
        Quadrature weights, shape ``(n_per_dim**3,)``; they sum to 8, the
        volume of :math:`[-1,1]^3`.
    """
    x, w = np.polynomial.legendre.leggauss(n_per_dim)   # [-1, 1]
    gx, gy, gz = np.meshgrid(x, x, x, indexing="ij")
    nodes = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    W = (w[:, None, None] * w[None, :, None] * w[None, None, :]).ravel()
    return nodes, W


def tensor_gauss_phys(n_per_dim: int, centre, h: float):
    r"""Physical Gauss nodes and weights for a box of half-side ``h`` at ``centre``.

    Parameters
    ----------
    n_per_dim : int
        Gauss nodes per axis.
    centre : array_like
        Box centre, shape ``(3,)``.
    h : float
        Box half-side.

    Returns
    -------
    pts : np.ndarray
        Physical node coordinates ``node*h + centre``, shape ``(M, 3)``.
    weights : np.ndarray
        Physical weights :math:`w\, h^3` (volume Jacobian), shape ``(M,)``.
    """
    nodes, W = tensor_gauss(n_per_dim)
    pts = nodes * h + np.asarray(centre)
    return pts, W * (h ** 3)


def source_charges(f_vals, weights):
    r"""Effective point charges for the volume S2M: :math:`q_k = f(y_k)\, w_k`.

    With these charges placed at the physical Gauss nodes, the existing
    point-charge P2M (which already applies the per-leaf half-side factor
    :math:`h`) yields the leaf's upward-equivalent density for the volume
    source.

    Parameters
    ----------
    f_vals : array_like
        Density sampled at the (physical) Gauss nodes, shape ``(..., M)``.
    weights : array_like
        Physical Gauss weights (from :func:`tensor_gauss_phys`), shape
        ``(M,)``.

    Returns
    -------
    np.ndarray
        Charges ``f_vals * weights``, shape ``(..., M)``.
    """
    return np.asarray(f_vals) * np.asarray(weights)


def duffy_self_potential(x0, fvals_at, ns=12, h=1.0, centre=(0.0, 0.0, 0.0)):
    r"""Self-box volume potential via the Duffy / 6-pyramid transformation.

    Computes :math:`\int_{\mathrm{box}} G(x_0, y) f(y)\, dy` for a target
    point :math:`x_0` INSIDE the box (integrable :math:`1/r` singularity).
    The box is split into (up to) six pyramids with apex at :math:`x_0`; in
    pyramid coordinates the :math:`1/r` singularity is cancelled by the
    :math:`s^2` Jacobian of the map,

    .. math:: G \, h^3\, s^2 = \frac{h^2\, s}{4\pi \sqrt{1 + p^2 + q^2}},

    so an ordinary Gauss–Legendre rule of order ``ns`` per axis converges
    spectrally for a smooth density.

    Parameters
    ----------
    x0 : array_like
        Target point inside the box (physical coords), shape ``(3,)``.
    fvals_at : callable
        Density ``f(Y)``, ``Y`` of shape ``(..., 3)`` physical coords →
        ``(...)``.
    ns : int, optional
        Gauss points per axis in each pyramid.
    h : float, optional
        Box half-side.
    centre : array_like, optional
        Box centre, shape ``(3,)``.

    Returns
    -------
    float
        The volume potential at ``x0``.
    """
    x0 = np.asarray(x0, float)
    c = np.asarray(centre, float)
    # work in local coords l = (x - centre)/h on the unit box [-1,1]^3
    l0 = (x0 - c) / h
    gs, ws = np.polynomial.legendre.leggauss(ns)
    total = 0.0
    for axis in range(3):
        o1, o2 = [a for a in range(3) if a != axis]
        for sign in (+1.0, -1.0):
            H = (1.0 - l0[axis]) if sign > 0 else (1.0 + l0[axis])   # > 0
            if H <= 0:
                continue
            plo, phi = (-1 - l0[o1]) / H, (1 - l0[o1]) / H
            qlo, qhi = (-1 - l0[o2]) / H, (1 - l0[o2]) / H
            s = 0.5 * H * (gs + 1.0);          ws_s = 0.5 * H * ws
            p = 0.5 * (phi - plo) * (gs + 1.0) + plo;  ws_p = 0.5 * (phi - plo) * ws
            q = 0.5 * (qhi - qlo) * (gs + 1.0) + qlo;  ws_q = 0.5 * (qhi - qlo) * ws
            S, P, Q = np.meshgrid(s, p, q, indexing="ij")
            W = ws_s[:, None, None] * ws_p[None, :, None] * ws_q[None, None, :]
            lu = np.empty(S.shape + (3,))
            lu[..., axis] = sign * S
            lu[..., o1] = S * P
            lu[..., o2] = S * Q
            y = c + h * (l0 + lu)                       # physical quadrature points
            # ∫ G f dy : local Jacobian h^3; G = 1/(4π h |lu|) = 1/(4π h S sqrt(1+P²+Q²))
            # ⇒ G·h^3·s²-Jacobian = h^2 · S/(4π sqrt(1+P²+Q²))
            integ = fvals_at(y) * (h * h) * S * _INV4PI / np.sqrt(1.0 + P * P + Q * Q)
            total += np.sum(integ * W)
    return total


def _duffy_points_weights(l0, ns):
    r"""Duffy quadrature points and weights for the self-box integral.

    Points are local coords on :math:`[-1,1]^3` with the target at local point
    ``l0``; :math:`\int_{\mathrm{box}} G f \approx \sum_k A_k f(\mathrm{pt}_k)`
    at unit half-side (multiply by :math:`h^2` for a box of half-side ``h``).
    """
    l0 = np.asarray(l0, float)
    gs, ws = np.polynomial.legendre.leggauss(ns)
    pts = []; A = []
    for axis in range(3):
        o1, o2 = [a for a in range(3) if a != axis]
        for sign in (+1.0, -1.0):
            H = (1.0 - l0[axis]) if sign > 0 else (1.0 + l0[axis])
            if H <= 0:
                continue
            plo, phi = (-1 - l0[o1]) / H, (1 - l0[o1]) / H
            qlo, qhi = (-1 - l0[o2]) / H, (1 - l0[o2]) / H
            s = 0.5 * H * (gs + 1.0);          ws_s = 0.5 * H * ws
            p = 0.5 * (phi - plo) * (gs + 1.0) + plo;  ws_p = 0.5 * (phi - plo) * ws
            q = 0.5 * (qhi - qlo) * (gs + 1.0) + qlo;  ws_q = 0.5 * (qhi - qlo) * ws
            S, P, Q = np.meshgrid(s, p, q, indexing="ij")
            W = ws_s[:, None, None] * ws_p[None, :, None] * ws_q[None, None, :]
            lu = np.empty(S.shape + (3,))
            lu[..., axis] = sign * S; lu[..., o1] = S * P; lu[..., o2] = S * Q
            pts.append((l0 + lu).reshape(-1, 3))
            A.append((S * _INV4PI / np.sqrt(1.0 + P * P + Q * Q) * W).reshape(-1))
    return np.concatenate(pts, 0), np.concatenate(A, 0)


def self_operator(n_per_dim, ns=14, graded=None, kappa=1.0, max_depth=10):
    r"""Precomputed SELF near-operator for the unit box.

    Target potential at the representation nodes from a source density given
    at the representation nodes: ``pot_at_nodes = S @ f_at_nodes``. For a
    physical box of half-side :math:`h`, scale by :math:`h^2`.

    Parameters
    ----------
    n_per_dim : int
        Gauss representation nodes per axis (``M = n_per_dim**3``).
    ns : int, optional
        Gauss points per axis in each Duffy pyramid.
    graded : bool, optional
        Use the graded-hybrid quadrature (module default from ``NEAR_QUAD``).
    kappa : float, optional
        Grading aggressiveness (see :func:`_graded_cells`).
    max_depth : int, optional
        Maximum grading depth.

    Returns
    -------
    np.ndarray
        Operator ``S``, shape ``(M, M)``.

    Notes
    -----
    Default is the GRADED-HYBRID quadrature: an octree graded toward the
    target node (a POINT target gives O(depth) cells, no face blowup), smooth
    cells via Gauss, and the tiny target-containing cells at ``max_depth`` via
    Duffy — whose stretched-pyramid error is negligible there (a
    :math:`2^{-D}` cell contributes :math:`\sim (2^{-D})^2`). The plain
    whole-box Duffy (``graded=False``) has extreme-aspect pyramids for nodes
    near faces (Gauss node at 0.98 gives pyramid height 0.02) and converges
    slowly and non-monotonically — the same silent-accuracy-floor class as the
    plain near operators.
    """
    graded = NEAR_QUAD["graded"] if graded is None else graded
    x1d, _ = np.polynomial.legendre.leggauss(n_per_dim)
    gx, gy, gz = np.meshgrid(x1d, x1d, x1d, indexing="ij")
    nodes = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)   # (M,3) rep nodes
    M = nodes.shape[0]
    S = np.empty((M, M))
    if not graded:
        for i in range(M):
            pts, A = _duffy_points_weights(nodes[i], ns)
            S[i, :] = A @ interp_matrix(n_per_dim, pts)         # (Nq,) @ (Nq,M)
        return S
    ng = n_per_dim + 1
    nodes1d = np.polynomial.legendre.leggauss(n_per_dim)[0]
    # per-node graded cells on host (cheap); smooth-cell quadrature on GPU with the cell
    # buffer PADDED to a shared cap (zero-half padding -> zero weight) so every node reuses
    # one executable; the few singular/touching tiny cells stay host-side Duffy.
    cells = []
    for i in range(M):
        cc, hh = _graded_cells(nodes[i][None, :], kappa, max_depth)
        near_mask = np.all(np.abs(nodes[i][None, :] - cc) <= hh[:, None] * (1 + 1e-12), axis=1)
        cells.append((cc[~near_mask], hh[~near_mask], cc[near_mask], hh[near_mask]))
    cap = max(len(c[0]) for c in cells)
    cap += -cap % 64                                            # quantize the shared shape
    for i in range(M):
        sm_c, sm_h, du_c, du_h = cells[i]
        ccp = np.zeros((cap, 3)); ccp[:len(sm_c)] = sm_c
        hhp = np.zeros(cap); hhp[:len(sm_h)] = sm_h             # zero half -> zero weight
        row = np.array(_graded_accum(_jnp.asarray(nodes[i][None, :]), _jnp.asarray(ccp),
                                     _jnp.asarray(hhp), _jnp.asarray(nodes1d),
                                     n_per_dim, ng)[0])          # copy: jax views are read-only
        for c, h in zip(du_c, du_h):                            # singular/near tiny cells
            l0_cell = np.clip((nodes[i] - c) / h, -1.0, 1.0)    # target in CELL coords
            pts, A = _duffy_points_weights(l0_cell, ns)
            row += (A * h * h) @ interp_matrix(n_per_dim, c[None, :] + pts * h)
        S[i, :] = row
    return S


def _graded_cells(tgt, kappa, max_depth):
    r"""Distance-adaptive octree over the unit source box for the near-operator quadrature.

    Splits a cell while ``diam > kappa * dist(cell, targets)``. Grading
    terminates naturally (Gauss target nodes never touch the box faces);
    depth ~5 suffices for machine precision.
    """
    cells = [(np.zeros(3), 1.0)]
    out_c, out_h = [], []
    for depth in range(max_depth + 1):
        nxt = []
        for c, h in cells:
            d = np.min(np.linalg.norm(tgt - c, axis=1))
            if depth < max_depth and 2 * h * np.sqrt(3) > kappa * max(d - h * np.sqrt(3), 1e-12):
                for sx in (-.5, .5):
                    for sy in (-.5, .5):
                        for sz in (-.5, .5):
                            nxt.append((c + np.array([sx, sy, sz]) * h, h / 2))
            else:
                out_c.append(c); out_h.append(h)
        cells = nxt
        if not cells:
            break
    return np.asarray(out_c), np.asarray(out_h)


import jax as _jax
import jax.numpy as _jnp
from functools import partial as _partial


@_partial(_jax.jit, static_argnames=["n_per_dim", "ng"])
def _graded_accum(tgt, cc, hh, nodes1d, n_per_dim, ng):
    r"""GPU accumulator for the graded quadratures.

    Sums over cells (centre ``cc``, half ``hh``; ZERO-half padding contributes
    nothing) of :math:`(G(\mathrm{tgt} - q)\, w) \cdot L(q)` with ``ng``-point
    tensor Gauss per cell and the tensor Lagrange basis of ``nodes1d``.
    Replaces a host per-node loop whose serial kernel evaluations do not scale.
    """
    qg, qw = np.polynomial.legendre.leggauss(ng)                 # host consts, traced once
    qg, qw = _jnp.asarray(qg), _jnp.asarray(qw)
    QX, QY, QZ = _jnp.meshgrid(qg, qg, qg, indexing="ij")
    qunit = _jnp.stack([QX.ravel(), QY.ravel(), QZ.ravel()], 1)          # (q, 3)
    wunit = (qw[:, None, None] * qw[None, :, None] * qw[None, None, :]).ravel()

    def lag1d(x):                                                # (..., n) Lagrange values
        d = x[..., None] - nodes1d[None, :]                      # (..., n)
        num = _jnp.prod(_jnp.where(_jnp.eye(n_per_dim, dtype=bool)[None, :, :],
                                   1.0, d[..., None, :]), axis=-1)       # (..., n)
        den = _jnp.prod(_jnp.where(_jnp.eye(n_per_dim, dtype=bool),
                                   1.0, nodes1d[:, None] - nodes1d[None, :]), axis=-1)
        return num / den

    def body(acc, cell):
        c, h = cell
        q = c[None, :] + qunit * h                               # (q, 3)
        w = wunit * h ** 3
        Lx, Ly, Lz = lag1d(q[:, 0]), lag1d(q[:, 1]), lag1d(q[:, 2])
        L = (Lx[:, :, None, None] * Ly[:, None, :, None]
             * Lz[:, None, None, :]).reshape(q.shape[0], -1)     # (q, n^3)
        r2 = _jnp.sum((tgt[:, None, :] - q[None, :, :]) ** 2, axis=-1)
        _ok = r2 > (16 * _jnp.finfo(r2.dtype).eps)**2 * _jnp.max(r2)  # ULP-relative floor
        G = _jnp.where(_ok, 1.0 / _jnp.sqrt(_jnp.where(_ok, r2, 1.0)), 0.0)
        return acc + ((G * w[None, :]) @ L) * _INV4PI, None

    acc0 = _jnp.zeros((tgt.shape[0], n_per_dim ** 3))
    acc, _ = _jax.lax.scan(body, acc0, (cc, hh))
    return acc


# graded near-op quality (module default): depth 3 gives ~1e-7-class operators;
# raise to 4-5 (~1e-9 / machine precision) for high-accuracy runs.
NEAR_QUAD = dict(graded=True, kappa=1.5, max_depth=3)


def near_operator_general(n_per_dim, tgt_local, ng=None, graded=None, kappa=None,
                          max_depth=None):
    r"""S2T near-operator for a unit source box and arbitrary local target points.

    Parameters
    ----------
    n_per_dim : int
        Gauss representation nodes per axis (``M = n_per_dim**3``).
    tgt_local : array_like
        Target points, shape ``(K, 3)``, expressed in the SOURCE box frame
        (unit box :math:`[-1,1]^3`).
    ng : int, optional
        Gauss points per axis per quadrature cell.
    graded, kappa, max_depth : optional
        Graded-quadrature controls; defaults come from the module-level
        ``NEAR_QUAD``.

    Returns
    -------
    np.ndarray
        Operator of shape ``(K, M)``: ``pot_at_targets = S @ f_at_source_nodes``.

    Notes
    -----
    By default uses GRADED (distance-adaptive composite) quadrature — the
    plain ``2*n`` tensor Gauss saturates around 1e-3-class error on
    face-adjacent boxes (the :math:`1/r` integrand peaks at the shared face),
    which silently floors the whole volume path. Graded depth 3 gives ~1e-7,
    4 ~1e-9, 5 machine precision; precompute-only cost (the apply-time
    operator is the same dense ``(K, M)`` matrix), memoized by the callers.
    ``graded=False`` restores the plain rule.

    For physical boxes, express the target points in the source frame
    ``x_local = (x_phys - centre_src) / h_src`` and scale the result by
    ``h_src**2``. This handles BOTH same-level and cross-level (coarse–fine)
    adjacency.
    """
    graded = NEAR_QUAD["graded"] if graded is None else graded
    tgt_local = np.asarray(tgt_local, float)
    if not graded:
        ng = ng or (2 * n_per_dim)
        qg, qw = np.polynomial.legendre.leggauss(ng)
        QX, QY, QZ = np.meshgrid(qg, qg, qg, indexing="ij")
        qpts = np.stack([QX.ravel(), QY.ravel(), QZ.ravel()], 1)
        qwt = (qw[:, None, None] * qw[None, :, None] * qw[None, None, :]).ravel()
        L = interp_matrix(n_per_dim, qpts)                      # (Nq, M)
        d = tgt_local[:, None, :] - qpts[None, :, :]
        r = np.sqrt(np.sum(d * d, axis=-1))                     # (K, Nq)
        Gw = np.where(r > 16 * np.finfo(r.dtype).eps * r.max(initial=0.0),
                      _INV4PI / r, 0.0) * qwt[None, :]                 # ULP-relative floor
        return Gw @ L                                           # (K, M)
    ng = ng or (n_per_dim + 1)
    kappa = NEAR_QUAD["kappa"] if kappa is None else kappa
    max_depth = NEAR_QUAD["max_depth"] if max_depth is None else max_depth
    cc, hh = _graded_cells(tgt_local, kappa, max_depth)
    nodes1d = np.polynomial.legendre.leggauss(n_per_dim)[0]
    return np.asarray(_graded_accum(_jnp.asarray(tgt_local), _jnp.asarray(cc),
                                    _jnp.asarray(hh), _jnp.asarray(nodes1d),
                                    n_per_dim, ng))


def near_operator(n_per_dim, disp, ng=None):
    r"""Same-level ADJACENT near-operator (special case of :func:`near_operator_general`).

    Unit source box :math:`[-1,1]^3`, target box at integer relative
    displacement ``disp`` (centre offset ``disp*2``). Scale by :math:`h^2`.

    Parameters
    ----------
    n_per_dim : int
        Gauss representation nodes per axis.
    disp : array_like
        Integer displacement, shape ``(3,)``, nonzero.
    ng : int, optional
        Gauss points per axis per quadrature cell.

    Returns
    -------
    np.ndarray
        Operator of shape ``(M, M)``.
    """
    disp = np.asarray(disp, float)
    if int(np.max(np.abs(disp))) == 0:
        raise ValueError("use self_operator for disp=0 (singular)")
    x1d, _ = np.polynomial.legendre.leggauss(n_per_dim)
    gx, gy, gz = np.meshgrid(x1d, x1d, x1d, indexing="ij")
    tgt = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1) + disp * 2.0   # same-level
    return near_operator_general(n_per_dim, tgt, ng=ng)


# ── Precomputed-operator bundle (memoized across solves) ─────────────────────
_VOL_OP_CACHE = {}


def volume_operators(p=6, ng=5, ns_self=14):
    r"""Build (and memoize) the scale-invariant operators a volume-FMM solve needs.

    Bundles the KIFMM surface operators (``build_operators(p)``), the singular
    SELF near-operator (``self_operator(ng, ns_self)``, at unit half-side),
    and the scale-invariant volume S2M / L2P kernel matrices.

    These depend only on ``(p, ng, ns_self)`` — NOT on the tree, density, or
    domain — so they are cached in memory and on disk and reused across
    repeated solves. Pass the returned dict to ``volume_potential_uniform`` /
    ``volume_potential_adaptive`` as ``vol_ops`` to skip the precompute on
    every call.

    Parameters
    ----------
    p : int, optional
        KIFMM surface order.
    ng : int, optional
        Gauss nodes per axis per leaf.
    ns_self : int, optional
        Duffy order for the singular self operator.

    Returns
    -------
    dict
        Keys ``"ops"`` (build_operators result), ``"S_self"`` (M×M unit self
        operator), ``"G_s2m"``, ``"G_l2p"``, ``"p"``, ``"ng"``, ``"ns_self"``.
    """
    key = (int(p), int(ng), int(ns_self))
    bundle = _VOL_OP_CACHE.get(key)
    if bundle is None:
        bundle = _load_vol_ops_disk(key)             # geometry-independent -> persist
        if bundle is not None:                       # across processes (high-p builds
            _VOL_OP_CACHE[key] = bundle              # take minutes; the load is fast)
            return bundle
    if bundle is None:
        from jaxfmm.kifmm.operators import build_operators
        from jaxfmm.kernels.laplace import laplace_G_np
        ops = build_operators(p)
        # Scale-invariant volume P2M / L2P operators (unit box): the leaf Gauss nodes
        # sit at fixed positions, so P2M and L2P are single precomputed GEMMs rather
        # than the per-box on-the-fly kernel eval in fmm.p2m / fmm.l2p.  With the
        # volume charge q = f·w·h³ already carrying h³:
        #   equiv[leaf] = pinv_up @ G_s2m @ q          (h-independent)
        #   far[leaf]   = (G_l2p @ local[leaf]) / h    (per-leaf 1/h)
        CH = np.asarray(ops["check_pts"]); nu = tensor_gauss(ng)[0]
        G_s2m = laplace_G_np(CH, nu)                  # (n_s, M)
        G_l2p = laplace_G_np(nu, CH)                  # (M, n_s)
        bundle = {"ops": ops, "S_self": self_operator(ng, ns=ns_self),
                  "G_s2m": G_s2m, "G_l2p": G_l2p,
                  "p": int(p), "ng": int(ng), "ns_self": int(ns_self)}
        _VOL_OP_CACHE[key] = bundle
        _save_vol_ops_disk(key, bundle)
    return bundle


def _vol_ops_path(key):
    import os
    d = os.environ.get("JAXFMM_OPS_CACHE",
                       os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    ".jaxfmm_ops_cache"))
    os.makedirs(d, exist_ok=True)
    p, ng, ns = key
    # the fingerprint must key EVERYTHING that changes the numerical content of the ops
    fp = f"g{int(NEAR_QUAD['graded'])}d{NEAR_QUAD['max_depth']}k{NEAR_QUAD['kappa']:g}v2"
    return os.path.join(d, f"volops_p{p}_ng{ng}_ns{ns}_{fp}.npz")


def _save_vol_ops_disk(key, bundle):
    try:
        flat = {"S_self": bundle["S_self"], "G_s2m": bundle["G_s2m"], "G_l2p": bundle["G_l2p"]}
        for k, v in bundle["ops"].items():
            if k == "M2L_K_table" and "M2L_K_class" in bundle["ops"]:
                continue                                     # store the compact class form only
            if isinstance(v, tuple):
                for i, vi in enumerate(v):
                    flat[f"ops.{k}.{i}"] = np.asarray(vi)
            else:
                flat[f"ops.{k}"] = np.asarray(v)
        np.savez_compressed(_vol_ops_path(key), **flat)
    except Exception as e:                                   # cache is best-effort
        print(f"volume_operators: disk-cache save failed ({e!r})")


def _load_vol_ops_disk(key):
    import os
    path = _vol_ops_path(key)
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path)
        ops, tups = {}, {}
        for k in z.files:
            if k.startswith("ops."):
                parts = k.split(".")
                if len(parts) == 3:
                    tups.setdefault(parts[1], {})[int(parts[2])] = z[k]
                else:
                    v = z[k]
                    ops[parts[1]] = int(v) if v.ndim == 0 else v
        for name, d in tups.items():
            ops[name] = tuple(d[i] for i in sorted(d))
        if "M2L_K_table" not in ops and "M2L_K_class" in ops:
            from jaxfmm.kifmm.operators import expand_m2l_table
            ops["M2L_K_table"] = expand_m2l_table(ops)       # rebuild from the class form
        return {"ops": ops, "S_self": z["S_self"], "G_s2m": z["G_s2m"], "G_l2p": z["G_l2p"],
                "p": key[0], "ng": key[1], "ns_self": key[2]}
    except Exception as e:
        print(f"volume_operators: disk-cache load failed ({e!r}); rebuilding")
        return None


def _assemble_near(near_self, fvals_arr, groups):
    r"""Batched near-field assembly.

    ``near_self`` ``(L, M)`` is the per-leaf self contribution (already
    scaled). Each ``group`` is ``(op, tgt_idx, src_idx)`` or
    ``(op, tgt_idx, src_idx, scale)``: ``op`` ``(M, M)`` is a near-operator
    shared by all listed pairs; each group adds ``op @ fvals[src]``
    (optionally times the per-pair ``scale``) into rows ``tgt`` via one
    batched GEMM + scatter-add. Returns the completed near field ``(L, M)``.
    """
    near = near_self
    for g in groups:
        op, tgt, src = g[0], np.asarray(g[1]), np.asarray(g[2])
        contrib = fvals_arr[src] @ op.T
        if len(g) == 4:
            contrib = contrib * np.asarray(g[3])[:, None]
        np.add.at(near, tgt, contrib)
    return near


# Cross-level near operators depend only on the RELATIVE geometry (ratio = h_tgt/
# h_src, rel = (c_tgt - c_src)/h_src); the absolute h_src is a per-pair h_src^2
# scale applied in _assemble_near.  Cache the (small, geometry-independent) set of
# distinct relative operators across solves.
_NEAR_OP_CACHE = {}


def _near_op_rel(ng, ratio, rel):
    key = (int(ng), round(float(ratio), 9), tuple(np.round(rel, 9)))
    op = _NEAR_OP_CACHE.get(key)
    if op is None:
        tgt_local = tensor_gauss(ng)[0] * float(ratio) + np.asarray(rel, float)
        op = near_operator_general(ng, tgt_local)
        _NEAR_OP_CACHE[key] = op
    return op


# The cross-level W (m2p) and X (p2l) interactions are ALSO scale-invariant in the
# relative config (ratio = h_tgt/h_src, rel = (c_tgt - c_src)/h_src): the equiv /
# check surfaces and Gauss nodes sit at fixed unit-box positions, so each distinct
# relative config has a fixed operator matrix (up to a per-pair 1/r scale).  This
# replaces the on-the-fly per-pair _green(r) eval in fmm.m2p / fmm.p2l with a small
# precomputed set + grouped GEMM — exactly the trick that makes the U-list fast.
_WX_OP_CACHE = {}


def _m2p_op_rel(ng, equiv_pts, ratio, rel):
    r"""W-list (m2p) relative operator: source equiv density -> target node potentials.

    ``op[i, j] = 1 / (4\pi |tg_rel_i - equiv_pts_j|)`` with
    ``tg_rel = gauss * ratio + rel``; shape ``(M, n_s)``. Per-pair scale
    ``1/h_src`` (physical ``r = h_src * r_rel``).
    """
    from jaxfmm.kernels.laplace import laplace_G_np
    key = ("m2p", int(ng), round(float(ratio), 9), tuple(np.round(rel, 9)))
    op = _WX_OP_CACHE.get(key)
    if op is None:
        tg_rel = tensor_gauss(ng)[0] * float(ratio) + np.asarray(rel, float)   # (M,3)
        op = laplace_G_np(tg_rel, np.asarray(equiv_pts, float))                # (M, n_s)
        _WX_OP_CACHE[key] = op
    return op


def _x_op_rel(ng, equiv_pts, ratio, rel):
    r"""X-list (p2l) relative operator, raw-K GRADED form.

    Source charges ``(M,)`` -> target downward CHECK potentials ``(n_s,)``:

    .. math:: K_{ij} = \frac{1}{w_j} \int L_j(y)\, G(\mathrm{check}_i - y)\, dy

    with the graded near quadrature. Evaluating the kernel AT the Gauss nodes
    instead would be an implicit plain-Gauss quadrature of an adjacent-box
    near integral (the same silent accuracy floor as the plain near
    operators); the check surface can also dip INTO the source box, hence the
    extra grading depth. Dividing by the Gauss weights keeps the apply
    contract (charges input, per-pair ``ratio`` scale) identical; the pinv
    decode moves to ONE application per target box (accumulate into
    ``check_pot`` before the shared decode — exact, the decode is linear per
    target row). ``check_rel = equiv_pts * ratio + rel`` (the downward check
    surface equals ``equiv_pts``).
    """
    key = ("x2", int(ng), round(float(ratio), 9), tuple(np.round(rel, 9)))
    op = _WX_OP_CACHE.get(key)
    if op is None:
        check_rel = np.asarray(equiv_pts, float) * float(ratio) + np.asarray(rel, float)  # (n_s,3)
        Kf = near_operator_general(ng, check_rel,                              # graded ∫ L_j G
                                   max_depth=NEAR_QUAD["max_depth"] + 3)       # interior dips
        op = Kf / tensor_gauss(ng)[1][None, :]                                 # -> charges form
        _WX_OP_CACHE[key] = op
    return op


def _x_apply_host(Xn, centres, hbox, q_box, ng, equiv_pts, n_s):
    r"""Host-grouped raw-K graded X apply for the adaptive reference drivers.

    Returns the per-box CHECK-potential contributions ``(n_boxes, n_s)`` —
    decode once (linear) and add to the local expansions. Groups the X pairs
    by exact relative config and reuses the memoized graded operators
    (:func:`_x_op_rel`).
    """
    ok = (Xn[:, 0] >= 0) & (Xn[:, 1] >= 0)
    check_X = np.zeros((len(centres), n_s))
    if not ok.any():
        return check_X
    tb, sb = Xn[ok, 0], Xn[ok, 1]
    ratio = hbox[tb] / hbox[sb]
    rel = (centres[tb] - centres[sb]) / hbox[sb][:, None]
    key = np.column_stack([np.round(ratio, 9), np.round(rel, 9)])
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    for k in range(len(uniq)):
        m = inv == k
        op = _x_op_rel(ng, equiv_pts, uniq[k, 0], uniq[k, 1:4])
        np.add.at(check_X, tb[m], (q_box[sb[m]] @ op.T) * ratio[m][:, None])
    return check_X


def _m2l_disp_groups(V, m_id, M2L_K_table):
    r"""Slice the V-list by box displacement :math:`d = \mathrm{grid}(t) - \mathrm{grid}(s)`.

    The M2L is a convolution over each level's regular box grid, so every pair
    at the same displacement (in :math:`[-3,3]^3`) shares the raw kernel
    matrix :math:`K_d`. Grouping by ``d`` lets the solve apply each ``K_d``
    ONCE via a batched GEMM (``equiv[src_d] @ K_d.T`` -> scatter to ``tgt_d``)
    instead of gathering an ``(n_s, n_s)`` matrix per pair (the memory-bound
    bottleneck of the generic per-pair m2l). Returns a list of
    ``(K_d, tgt_idx, src_idx)`` jnp groups; apply ``pinv_dn`` once per box
    afterwards. Host-side (fixed tree).
    """
    import jax.numpy as jnp
    from jaxfmm.trees import _morton_to_grid
    n_s = M2L_K_table.shape[-1]
    K_flat = np.asarray(M2L_K_table).reshape(343, n_s, n_s)
    gx, gy, gz = (np.asarray(g) for g in _morton_to_grid(m_id))
    Vn = np.asarray(V); ok = (Vn[:, 0] >= 0) & (Vn[:, 1] >= 0)
    tb = Vn[ok, 0].astype(np.int64); sb = Vn[ok, 1].astype(np.int64)
    disp = (gx[tb] - gx[sb] + 3) * 49 + (gy[tb] - gy[sb] + 3) * 7 + (gz[tb] - gz[sb] + 3)
    order = np.argsort(disp, kind="stable"); ds = disp[order]; tb = tb[order]; sb = sb[order]
    uniq, starts = np.unique(ds, return_index=True); ends = np.append(starts[1:], len(ds))
    return [(jnp.asarray(K_flat[d]), jnp.asarray(tb[a:b]), jnp.asarray(sb[a:b]))
            for d, a, b in zip(uniq, starts, ends)]


def _binned_cfg_apply(input_arr, tgt, src, valid, centres, h, ksort_j, n_cfg,
                      ops_j, scale_kind, n_pairs, CAP, max_boxes, d_out):
    r"""In-trace realization of the frozen host grouping for near/W/X applies.

    The "config int per pair, sort, single batched pass" form: compact the
    valid pairs to ``n_pairs``, key each by its scale-invariant config
    (``ratio = h_t/h_s``, ``rel = (c_t - c_s)/h_s``), bin by config into a
    fixed ``(n_cfg, CAP)`` layout (within-config rank via one argsort), then
    apply each config's dense operator ONCE via a single batched GEMM and
    scatter to targets. Avoids both the per-pair matrix gather (memory-bound)
    and the einsum-over-all-boxes (``max_boxes * n_cfg`` overcompute). All
    per-pair geometry is computed on the compact buffer. ``ops_j`` is
    ``(n_cfg, d_out, d_in)``; ``input_arr`` ``(max_boxes, d_in)``;
    ``scale_kind`` in ``{"inv_hs", "ratio", "hs2"}``. Returns
    ``(max_boxes, d_out)``.
    """
    import jax.numpy as jnp
    # compact valid pairs -> fixed n_pairs buffer
    dst = jnp.cumsum(valid.astype(jnp.int32)) - 1
    dst = jnp.where(valid, dst, n_pairs)
    ct = jnp.full(n_pairs, -1, jnp.int32).at[dst].set(tgt.astype(jnp.int32), mode="drop")
    cs = jnp.zeros(n_pairs, jnp.int32).at[dst].set(src.astype(jnp.int32), mode="drop")
    cok = ct >= 0
    cts = jnp.where(cok, ct, 0)
    # config + scale on the compact buffer
    ratio = h[cts] / h[cs]; rel = (centres[cts] - centres[cs]) / h[cs][:, None]
    key = _cfg_encode(*_cfg_quant(ratio, rel, jnp))
    pos = jnp.clip(jnp.searchsorted(ksort_j, key), 0, n_cfg - 1)
    cc = jnp.where(cok & (ksort_j[pos] == key), pos, n_cfg)          # unmatched/invalid -> dump
    csc = (1.0 / h[cs]) if scale_kind == "inv_hs" else (ratio if scale_kind == "ratio" else h[cs] ** 2)
    # within-config rank: sort by config, position minus the config's first index
    order = jnp.argsort(cc)
    seg0 = jnp.searchsorted(cc[order], cc[order], side="left")
    rank = jnp.zeros(n_pairs, jnp.int32).at[order].set(
        (jnp.arange(n_pairs) - seg0).astype(jnp.int32))
    inb = (cc < n_cfg) & (rank < CAP)
    bc = jnp.where(inb, cc, 0); br = jnp.where(inb, rank, CAP)        # invalid -> col CAP (dropped)
    bin_src = jnp.zeros((n_cfg, CAP), jnp.int32).at[bc, br].set(jnp.where(inb, cs, 0), mode="drop")
    bin_tgt = jnp.zeros((n_cfg, CAP), jnp.int32).at[bc, br].set(jnp.where(inb, cts, 0), mode="drop")
    bin_sc = jnp.zeros((n_cfg, CAP)).at[bc, br].set(jnp.where(inb, csc, 0.0), mode="drop")
    # single batched GEMM (batched over config) + scatter to targets
    contrib = jnp.einsum("cod,crd->cro", ops_j, input_arr[bin_src]) * bin_sc[..., None]
    return jnp.zeros((max_boxes, d_out)).at[bin_tgt.reshape(-1)].add(
        contrib.reshape(-1, d_out), mode="drop")


def volume_potential_uniform(f, level, p=6, ng=6, domain=(0.0, 1.0),
                             ns_self=16, batch_size=256, vol_ops=None, factored=None):
    r"""Volume FMM on a UNIFORM level-``level`` octree over a cube.

    Computes :math:`u(x) = \int G(x,y) f(y)\, dy` and returns it sampled at
    each leaf's Gauss nodes (a degree-``(ng-1)`` tensor polynomial per leaf —
    the per-leaf "potential polynomial"). Evaluate at arbitrary points by
    interpolating a leaf's node values (see :func:`interp_matrix`).

    Parameters
    ----------
    f : callable
        Density ``f(Y)``, ``Y`` of shape ``(..., 3)`` physical → ``(...)``.
    level : int
        Uniform octree level (``8**level`` leaves).
    p : int, optional
        KIFMM surface order.
    ng : int, optional
        Gauss nodes per axis per leaf (density polynomial degree ``ng-1``).
    domain : tuple of float, optional
        Cubic domain ``[lo, hi]^3``.
    ns_self : int, optional
        Duffy order for the singular self operator.
    batch_size : int, optional
        M2L batch size.
    vol_ops : dict, optional
        Precomputed operator bundle from :func:`volume_operators`.
    factored : bool, optional
        Use the backward-stable factored decodes when available.

    Returns
    -------
    dict
        Keys: ``"leaf_box"`` (``(L,)`` int32 FMM box index of each leaf),
        ``"centres"`` (``(L, 3)`` leaf centres), ``"h"`` (float leaf
        half-side), ``"nodes_unit"`` (``(M, 3)`` Gauss node offsets on
        :math:`[-1,1]^3`), ``"node_pot"`` (``(L, M)`` potential at each
        leaf's Gauss nodes).
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxfmm.trees import build_fmm_tree, _morton_to_grid, morton_to_coords
    from jaxfmm.trees import build_interaction_lists
    from jaxfmm.kernels.laplace import laplace_G_np
    from jaxfmm import kifmm as fmm

    lo, hi = domain
    if vol_ops is None:
        vol_ops = volume_operators(p, ng, ns_self)
    ops = vol_ops["ops"]
    n_s = ops["n_s"]; CH = np.asarray(ops["check_pts"])
    factored = ("F_up" in ops) if factored is None else (factored and "F_up" in ops)
    if factored:                       # backward-stable FACTORED decodes: the volume path
        _u, _d, _vt = (np.asarray(x) for x in ops["F_up"])   # runs the KIFMM at HIGH orders,
        dec_up = lambda X: ((X @ _u) * _d) @ _vt             # where materialised pinv products
        dec_dn = tuple(jnp.asarray(x) for x in ops["F_dn"])  # cap f64 accuracy at ~9 digits
    else:
        pinv_up = np.asarray(ops["pinv_up"])

    # uniform level-`level` tree: feed a uniform grid spanning the domain
    nside = 2 ** level
    gpts = np.linspace(lo + 1e-3, hi - 1e-3, 2 * nside)
    GX, GY, GZ = np.meshgrid(gpts, gpts, gpts, indexing="ij")
    grid = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 1)
    max_boxes = (8 ** (level + 1)) // 7 + nside ** 3   # generous
    out = build_fmm_tree(jnp.asarray(grid), max_boxes, max_depth=level, N_max=7)
    m_id, p_id, lvls, counts, is_leaf, sort_idx, rev, de, pmin, mde, mbe = out
    lvlsn, leafn = np.asarray(lvls), np.asarray(is_leaf); de = float(de); pmin = np.asarray(pmin)
    leaves = np.where(leafn & (lvlsn > 0))[0]
    assert np.all(lvlsn[leaves] == level), "tree not uniform at requested level"
    centres = np.asarray(morton_to_coords(m_id, lvls, de, pmin)); hbox = de * 0.5 ** (level + 1.0)

    nodes_unit, gw = tensor_gauss(ng); M = ng ** 3
    L = len(leaves)
    # per-leaf node densities and S2M.  G_s2m = laplace_G(check, node) is the same
    # matrix for every leaf up to scale (both surfaces translate with the centre
    # and scale with h): laplace_G(CH*h+c, node*h+c) = (1/h)·laplace_G(CH, nodes).
    G_s2m = laplace_G_np(CH, nodes_unit)                 # (n_s, M), unit box
    F = np.empty((L, M))
    for i, b in enumerate(leaves):
        F[i] = f(nodes_unit * hbox + centres[b])
    # equiv[b] = h·pinv_up @ G_s2m @ (f·w·h³) / h  = h³·pinv_up @ G_s2m @ (F·w)
    equiv = np.zeros((m_id.shape[0], n_s))
    check_leaf = (hbox ** 3) * ((F * gw) @ G_s2m.T)          # raw check potentials
    equiv[leaves] = dec_up(check_leaf) if factored else check_leaf @ pinv_up.T

    # far field
    _, V = build_interaction_lists(m_id, lvls, is_leaf, m_id.shape[0], level)
    if factored:
        equiv = fmm.m2m_factored(jnp.asarray(equiv), m_id, p_id, lvls,
                                 jnp.asarray(ops["M2M_K"]),
                                 tuple(jnp.asarray(x) for x in ops["F_par"]), level)
        local = fmm.m2l(equiv, V, m_id, jnp.asarray(ops["M2L_K_table"]), dec_dn,
                        n_pairs=fmm.v_pair_count(V, batch_size=batch_size), batch_size=batch_size)
        local = np.asarray(fmm.l2l_factored(local, m_id, p_id, lvls,
                                            jnp.asarray(ops["L2L_K"]), dec_dn, level))
    else:
        equiv = fmm.m2m(jnp.asarray(equiv), m_id, p_id, lvls, jnp.asarray(ops["M2M_mats"]), level)
        local = fmm.m2l(equiv, V, m_id, jnp.asarray(ops["M2L_K_table"]), jnp.asarray(ops["pinv_dn"]),
                        n_pairs=fmm.v_pair_count(V, batch_size=batch_size), batch_size=batch_size)
        local = np.asarray(fmm.l2l(local, m_id, p_id, lvls, jnp.asarray(ops["L2L_mats"]), level))

    # far at the leaf nodes: laplace_G(node*h+c, CH*h+c) = (1/h)·laplace_G(nodes, CH)
    G_l2p = laplace_G_np(nodes_unit, CH)                 # (M, n_s), unit box
    far = (local[leaves] @ G_l2p.T) / hbox               # (L, M)

    # near operators (unit box), scaled by h^2; same-level adjacency only (uniform)
    h2 = hbox ** 2
    S_self = vol_ops["S_self"] * h2
    dirs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
    near_ops = {d: near_operator(ng, d, ng=2 * ng) * h2 for d in dirs}
    gx, gy, gz = [np.asarray(a).astype(int) for a in _morton_to_grid(m_id)]

    # group neighbour pairs by relative displacement (rel = B_grid - A_grid)
    pos_of = {(int(gx[b]), int(gy[b]), int(gz[b])): i for i, b in enumerate(leaves)}
    grp = {d: ([], []) for d in dirs}                    # rel -> (tgt_pos, src_pos)
    for i, b in enumerate(leaves):
        bg = (int(gx[b]), int(gy[b]), int(gz[b]))
        for d in dirs:
            j = pos_of.get((bg[0] - d[0], bg[1] - d[1], bg[2] - d[2]))
            if j is not None:
                grp[d][0].append(i); grp[d][1].append(j)
    groups = [(near_ops[d], t, s) for d, (t, s) in grp.items() if t]

    node_pot = _assemble_near(F @ S_self.T, F, groups) + far
    return {"leaf_box": np.asarray(leaves), "centres": centres[leaves],
            "h": hbox, "nodes_unit": nodes_unit, "node_pot": node_pot}


def _legendre_coeff_matrix_1d(n_per_dim):
    r"""1D discrete Legendre-transform matrix ``C`` ``(n, n)``: ``coeffs = C @ node_values``.

    .. math:: c_k = \frac{2k+1}{2} \sum_j P_k(x_j)\, w_j\, f_j

    (exact projection for degree :math:`\le n-1`).
    """
    x, w = np.polynomial.legendre.leggauss(n_per_dim)
    P = np.polynomial.legendre.legvander(x, n_per_dim - 1)   # (n, n): P[j,k]=P_k(x_j)
    k = np.arange(n_per_dim)
    return ((2 * k + 1) / 2.0)[:, None] * (P * w[:, None]).T  # (n_k, n_j)


def refinement_indicator(f_vals, n_per_dim):
    r"""Relative interpolation-error estimate for a leaf density sample.

    The Legendre-coefficient-tail indicator: returns
    :math:`\lVert\text{top-degree Legendre coefficients}\rVert /
    \lVert\text{all coefficients}\rVert \in [0, 1]`. A value above a tolerance
    means the degree-``(n-1)`` tensor polynomial does not resolve ``f`` on the
    box, so the leaf should be refined (subdivided).

    Parameters
    ----------
    f_vals : array_like
        Density at the leaf's Gauss nodes (tensor order), shape
        ``(n_per_dim**3,)``.
    n_per_dim : int
        Gauss nodes per axis.

    Returns
    -------
    float
        The relative tail norm.
    """
    tail, total = _legendre_tail_total(f_vals, n_per_dim)
    return 0.0 if total == 0.0 else float(tail / total)


def _legendre_tail_total(f_vals, n_per_dim):
    r"""(top-degree coefficient norm, total coefficient norm) of the tensor Legendre expansion.

    The top-degree (tail) norm is an ABSOLUTE estimate of the degree-``(n-1)``
    interpolation error of the density on the box.
    """
    n = n_per_dim
    C = _legendre_coeff_matrix_1d(n)
    F = np.asarray(f_vals).reshape(n, n, n)
    coeffs = np.einsum("ai,bj,ck,ijk->abc", C, C, C, F)
    mask = np.zeros((n, n, n), bool)
    mask[-1, :, :] = True; mask[:, -1, :] = True; mask[:, :, -1] = True
    return float(np.linalg.norm(coeffs[mask])), float(np.linalg.norm(coeffs))


def refine_cells(f, n=6, tol=1e-6, max_depth=6, min_level=1, domain=(0.0, 1.0)):
    r"""Adaptive octree refinement driven by the density's interpolation error.

    Starting from a uniform ``min_level`` grid over ``[lo, hi]^3``,
    recursively subdivide any cell whose interpolation-error estimate exceeds
    the tolerance (down to ``max_depth``).

    Parameters
    ----------
    f : callable
        Density ``f(Y)``, ``Y`` of shape ``(..., 3)`` → ``(...)``.
    n : int, optional
        Gauss nodes per axis (interpolation degree ``n-1``).
    tol : float, optional
        RELATIVE density-error tolerance: a cell is refined when its absolute
        interpolation-error estimate exceeds ``tol * max|f|`` (so cells where
        ``f`` is negligible are NOT refined — the error that matters is
        absolute).
    max_depth : int, optional
        Maximum refinement level.
    min_level : int, optional
        Starting uniform level.
    domain : tuple of float, optional
        Cubic domain ``(lo, hi)``.

    Returns
    -------
    list of tuple
        The adaptive leaf cells ``(level, ix, iy, iz)`` (grid indices at each
        cell's own level). NOT 2:1-balanced (balancing happens when the tree
        is realised for the FMM).
    """
    lo, hi = domain; extent = float(hi - lo)
    nodes_unit, _ = tensor_gauss(n)
    # global density scale (so the threshold is absolute, not relative-per-cell)
    gs = np.linspace(lo, hi, 4 * 2 ** min_level + 1)
    GX, GY, GZ = np.meshgrid(gs, gs, gs, indexing="ij")
    f_scale = float(np.max(np.abs(f(np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 1)))))
    thresh = tol * max(f_scale, 1e-300)
    leaves = []
    stack = [(min_level, ix, iy, iz)
             for ix in range(2 ** min_level)
             for iy in range(2 ** min_level)
             for iz in range(2 ** min_level)]
    while stack:
        lvl, ix, iy, iz = stack.pop()
        hc = extent * 0.5 ** (lvl + 1)                      # cell half-side
        centre = lo + (np.array([ix, iy, iz], float) + 0.5) * (extent / 2 ** lvl)
        fb = f(nodes_unit * hc + centre)
        tail, _ = _legendre_tail_total(fb, n)               # absolute error estimate
        if lvl < max_depth and tail > thresh:
            for c in range(8):
                stack.append((lvl + 1, 2 * ix + (c & 1), 2 * iy + ((c >> 1) & 1),
                              2 * iz + ((c >> 2) & 1)))
        else:
            leaves.append((lvl, ix, iy, iz))
    return leaves


def volume_potential_adaptive(f, tol=1e-4, max_depth=5, min_level=2, p=6, ng=5,
                              domain=(0.0, 1.0), ns_self=14, batch_size=256,
                              vol_ops=None):
    r"""Adaptive volume FMM for :math:`u(x) = \int G(x,y) f(y)\, dy` (reference driver).

    (a) refine the octree where the density's interpolation error exceeds
    ``tol`` (:func:`refine_cells`), (b) form leaf expansions from the
    polynomial via S2M, and (c) evaluate the near field directly (Duffy self +
    cross-level operators). The far field reuses the adaptive point FMM with
    the volume quadrature charges; only the U-list near field is
    volume-specific.

    Parameters
    ----------
    f : callable
        Density ``f(Y)``, ``Y`` of shape ``(..., 3)`` → ``(...)``.
    tol, max_depth, min_level : optional
        Refinement controls (see :func:`refine_cells`).
    p, ng, ns_self : int, optional
        Surface order / Gauss nodes per axis / Duffy order.
    domain : tuple of float, optional
        Cubic domain ``(lo, hi)``.
    batch_size : int, optional
        M2L batch size.
    vol_ops : dict, optional
        Precomputed operator bundle from :func:`volume_operators`.

    Returns
    -------
    dict
        Keys: ``"leaf_box"`` (``(L,)`` int32), ``"centres"`` (``(L, 3)``),
        ``"h"`` (``(L,)`` leaf half-sides), ``"nodes_unit"`` (``(M, 3)``,
        ``M = ng**3``), ``"node_pot"`` (``(L, M)``).
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxfmm.trees import morton_to_coords
    from jaxfmm.trees import (build_interaction_lists, build_wx_lists,
                                     build_ulist_adaptive)
    from jaxfmm import kifmm as fmm

    if vol_ops is None:
        vol_ops = volume_operators(p, ng, ns_self)
    ops = vol_ops["ops"]
    nodes_unit, gw = tensor_gauss(ng); M = ng ** 3

    # (a) adaptive refinement → 2:1-balanced FMM tree over the EXACT domain
    T = build_volume_tree(f, tol=tol, max_depth=max_depth, min_level=min_level,
                          n=ng, domain=domain)
    m_id = jnp.asarray(T["m_id"]); p_id = jnp.asarray(T["p_id"])
    lvls = jnp.asarray(T["lvls"]); is_leaf = jnp.asarray(T["is_leaf"])
    de = T["domain_extent"]; pmin = T["pmin"]; md = T["max_depth"]
    lvlsn = np.asarray(T["lvls"]); leaves = np.asarray(T["leaves"])
    centres = np.asarray(morton_to_coords(m_id, lvls, de, pmin)); hbox = de * 0.5 ** (lvlsn + 1.0)

    # per-leaf Gauss nodes (tensor order) → global particles + volume charges (f·w·h³)
    pil = -np.ones((m_id.shape[0], M), dtype=np.int32)
    pos_list, chg_list, fvals = [], [], {}
    gi = 0
    for b in leaves:
        h = hbox[b]; cn = centres[b]; pb = nodes_unit * h + cn
        pil[b] = np.arange(gi, gi + M); gi += M
        fb = f(pb); fvals[int(b)] = fb
        pos_list.append(pb); chg_list.append(fb * (gw * h ** 3))
    positions = jnp.asarray(np.vstack(pos_list))
    charges = jnp.asarray(np.concatenate(chg_list)); pil = jnp.asarray(pil)

    # (b)+far field: adaptive point FMM with volume charges (upward S2M, M2L, X, L2L, L2P, W)
    equiv, cc, hh = fmm.upward_pass(positions, charges, pil, m_id, p_id, lvls, is_leaf,
                                    de, pmin, ops, batch_size=batch_size)
    _, V = build_interaction_lists(m_id, lvls, is_leaf, m_id.shape[0], md)
    W, X, _ = build_wx_lists(m_id, lvls, is_leaf, m_id.shape[0], md)
    dn = (tuple(jnp.asarray(x) for x in ops["F_dn"]) if "F_up" in ops
          else jnp.asarray(ops["pinv_dn"]))              # backward-stable decode when available
    local = fmm.m2l(equiv, V, m_id, jnp.asarray(ops["M2L_K_table"]), dn,
                    n_pairs=fmm.v_pair_count(V, batch_size), batch_size=batch_size)
    q_box = np.zeros((m_id.shape[0], M))                 # X-list: RAW-K graded ops (the
    for b in leaves:                                     # point-charge p2l would be an
        q_box[b] = fvals[int(b)] * (gw * hbox[b] ** 3)   # implicit plain-Gauss near integral)
    check_X = _x_apply_host(np.asarray(X), centres, hbox, q_box, ng,
                            np.asarray(ops["equiv_pts"]), ops["n_s"])
    local = local + fmm._decode(jnp.asarray(check_X), dn)
    far = fmm.downward_pass(local, positions, pil, m_id, p_id, lvls, is_leaf, cc, hh, ops,
                            batch_size=batch_size)
    far = np.asarray(far + fmm.m2p(equiv, positions, pil, cc, hh, W,
                                   jnp.asarray(ops["equiv_pts"]), batch_size=batch_size))

    # (c) near field: self Duffy + cross-level near operators over the adaptive U-list
    U, _ = build_ulist_adaptive(m_id, lvls, is_leaf, m_id.shape[0], md)
    Un = np.asarray(U)
    S_self = vol_ops["S_self"]
    by_target = {}
    for t, s in Un:
        if t >= 0:
            by_target.setdefault(int(t), []).append(int(s))

    leaf_idx = {int(b): i for i, b in enumerate(leaves)}
    hleaf = hbox[leaves]                                  # (L,)
    F = np.array([fvals[int(b)] for b in leaves])         # (L, M)

    # self contribution (scaled per leaf); then group cross-neighbour operators by
    # geometry so each distinct near-operator is built once and applied to all its
    # pairs as one batched GEMM (op identical when (hA, h/hA, (c-cA)/hA) match).
    near_self = (hleaf ** 2)[:, None] * (F @ S_self.T)
    grp = {}                                              # key -> [op, tgt_pos, src_pos]
    for b in leaves:
        i = leaf_idx[int(b)]; h = float(hbox[b]); c = centres[b]
        for A in by_target.get(int(b), []):
            hA = float(hbox[A]); cA = centres[A]
            rel = tuple(np.round((c - cA) / hA, 9)); ratio = round(h / hA, 9)
            key = (round(hA, 12), ratio, rel)
            g = grp.get(key)
            if g is None:
                tgt_local = nodes_unit * (h / hA) + (c - cA) / hA   # = (Bnodes-cA)/hA
                g = grp[key] = [near_operator_general(ng, tgt_local) * (hA ** 2), [], []]
            g[1].append(i); g[2].append(leaf_idx[int(A)])
    groups = [(op, t, s) for op, t, s in grp.values()]

    Far = np.asarray(far)[np.asarray(pil)[leaves]]        # (L, M)
    node_pot = _assemble_near(near_self, F, groups) + Far

    return {"leaf_box": leaves, "centres": centres[leaves], "h": hbox[leaves],
            "nodes_unit": nodes_unit, "node_pot": node_pot}


def volume_potential_adaptive_jit(f, tol=1e-4, max_depth=5, min_level=2, p=6, ng=5,
                                  domain=(0.0, 1.0), ns_self=14, batch_size=256,
                                  vol_ops=None, max_boxes=None, max_leaves=None,
                                  theta=None, m2l_n_pairs=None):
    r"""Fixed-shape adaptive volume FMM.

    Same result as :func:`volume_potential_adaptive` but built on the JIT tree
    (:func:`build_volume_tree_jit`) with fixed-size buffers (``max_boxes``
    boxes, ``max_leaves`` leaf slots), so the far-field KIFMM compiles ONCE
    and is reused across geometries (no per-density retrace).

    ``f`` must be JAX-traceable: ``f(Y)`` if ``theta is None`` else
    ``f(Y, theta)``. Vary ``theta`` (with fixed
    ``max_boxes``/``max_leaves``/``m2l_n_pairs``) to move the source /
    time-step with no recompilation at all.

    Parameters
    ----------
    f, tol, max_depth, min_level, p, ng, domain, ns_self, batch_size, vol_ops
        As in :func:`volume_potential_adaptive`.
    max_boxes : int, optional
        Fixed box-buffer size (static); default a generous bound.
    max_leaves : int, optional
        Fixed leaf-slot count (static); default ``max_boxes``.
    theta : array_like, optional
        Traced parameter array for a parametric density.
    m2l_n_pairs : int, optional
        Fixed M2L pair-buffer size; pass a constant covering the geometry
        family for guaranteed no-recompile (default: a granularity-rounded
        count from the current tree).

    Returns
    -------
    dict
        Same keys as :func:`volume_potential_adaptive`.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxfmm.trees import morton_to_coords
    from jaxfmm.trees import (build_interaction_lists, build_wx_lists,
                                     build_ulist_adaptive)
    from jaxfmm import kifmm as fmm

    if vol_ops is None:
        vol_ops = volume_operators(p, ng, ns_self)
    ops = vol_ops["ops"]
    nodes_unit, gw = tensor_gauss(ng); M = ng ** 3
    md = int(max_depth)
    if max_boxes is None:
        max_boxes = sum(8 ** k for k in range(1, min_level + 1)) + 8 ** max_depth + 64
    if max_leaves is None:
        max_leaves = max_boxes

    # (a) JIT tree over fixed max_boxes (rooted at level 1, fine-to-coarse)
    T = build_volume_tree_jit(f, tol=tol, max_depth=max_depth, min_level=min_level,
                              n=ng, domain=domain, max_boxes=max_boxes, theta=theta)
    m_id = jnp.asarray(T["m_id"]); p_id = jnp.asarray(T["p_id"])
    lvls = jnp.asarray(T["lvls"]); is_leaf = jnp.asarray(T["is_leaf"])
    de = T["domain_extent"]; pmin = T["pmin"]
    lvlsn = np.asarray(T["lvls"]); leaves = np.asarray(T["leaves"]); n_leaves = len(leaves)
    if n_leaves > max_leaves:
        raise ValueError(f"n_leaves={n_leaves} exceeds max_leaves={max_leaves}")
    centres = np.asarray(morton_to_coords(m_id, lvls, de, pmin))
    hbox = de * 0.5 ** (lvlsn + 1.0)

    # leaf slots → fixed-size particles (max_leaves*M); pil indexed by box.
    # Vectorised: build all leaf Gauss nodes at once and evaluate f in ONE call.
    fcall = (lambda Y: f(Y, theta)) if theta is not None else f
    nL = n_leaves
    lvj = jnp.asarray(leaves)
    cl = jnp.asarray(centres[leaves]); hl = jnp.asarray(hbox[leaves])     # (nL,3),(nL,)
    nu = jnp.asarray(nodes_unit); gwj = jnp.asarray(gw)
    pts = nu[None] * hl[:, None, None] + cl[:, None, :]                   # (nL,M,3)
    F = fcall(pts.reshape(-1, 3)).reshape(nL, M)                          # one GPU f-eval
    chg = F * (gwj[None] * (hl ** 3)[:, None])                           # (nL,M)
    slots = jnp.arange(nL * M, dtype=jnp.int32)
    positions = jnp.zeros((max_leaves * M, 3)).at[:nL * M].set(pts.reshape(-1, 3))
    charges = jnp.zeros(max_leaves * M).at[:nL * M].set(chg.reshape(-1))
    pil = jnp.full((max_boxes, M), -1, dtype=jnp.int32).at[lvj].set(slots.reshape(nL, M))
    F = np.asarray(F)                                                     # near field is host

    # (b)+far field — all shapes fixed; md (static) keeps m2m/l2l/lists shape-stable
    equiv, cc, hh = fmm.upward_pass(positions, charges, pil, m_id, p_id, lvls, is_leaf,
                                    de, pmin, ops, batch_size=batch_size, max_depth=md)
    _, V = build_interaction_lists(m_id, lvls, is_leaf, max_boxes, md)
    W, X, _ = build_wx_lists(m_id, lvls, is_leaf, max_boxes, md)
    if m2l_n_pairs is None:
        m2l_n_pairs = fmm.v_pair_count(V, batch_size)
    dn = (tuple(jnp.asarray(x) for x in ops["F_dn"]) if "F_up" in ops
          else jnp.asarray(ops["pinv_dn"]))              # backward-stable decode when available
    local = fmm.m2l(equiv, V, m_id, jnp.asarray(ops["M2L_K_table"]), dn,
                    n_pairs=m2l_n_pairs, batch_size=batch_size)
    q_box = np.zeros((max_boxes, M))                     # X-list: RAW-K graded ops (see
    q_box[leaves] = np.asarray(chg)                      # the reference driver)
    check_X = _x_apply_host(np.asarray(X), centres, hbox, q_box, ng,
                            np.asarray(ops["equiv_pts"]), ops["n_s"])
    local = local + fmm._decode(jnp.asarray(check_X), dn)
    far = fmm.downward_pass(local, positions, pil, m_id, p_id, lvls, is_leaf, cc, hh, ops,
                            batch_size=batch_size, max_depth=md)
    far = np.asarray(far + fmm.m2p(equiv, positions, pil, cc, hh, W,
                                   jnp.asarray(ops["equiv_pts"]), batch_size=batch_size))

    # (c) near field — Duffy self + cross-level ops over the U-list, fully
    # vectorised: group pairs by near-operator geometry via np.unique (no Python
    # pair loop), build each distinct operator once, batched-GEMM in _assemble_near.
    U, _ = build_ulist_adaptive(m_id, lvls, is_leaf, max_boxes, md)
    Un = np.asarray(U)
    S_self = vol_ops["S_self"]
    hleaf = hbox[leaves]
    near_self = (hleaf ** 2)[:, None] * (F @ S_self.T)

    pos_of_box = np.full(max_boxes, -1, dtype=np.int64)
    pos_of_box[leaves] = np.arange(len(leaves))
    valid = Un[:, 0] >= 0
    t_box = Un[valid, 0]; s_box = Un[valid, 1]
    groups = []
    if t_box.size:
        ct = centres[t_box]; cs = centres[s_box]
        ht = hbox[t_box]; hs = hbox[s_box]
        rel = np.round((ct - cs) / hs[:, None], 9)        # (P,3) target offset in src frame
        ratio = np.round(ht / hs, 9)                      # (P,)
        # group by RELATIVE geometry only (ratio, rel); h_src^2 is a per-pair scale
        key = np.column_stack([ratio, rel])               # (P,4)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        tpos = pos_of_box[t_box]; spos = pos_of_box[s_box]; hs2 = hs ** 2
        for k in range(uniq.shape[0]):
            m = inv == k
            op = _near_op_rel(ng, uniq[k, 0], uniq[k, 1:4])    # cached across solves
            groups.append((op, tpos[m], spos[m], hs2[m]))

    Far = far[np.asarray(pil)[leaves]]                    # (n_leaves, M)
    node_pot = _assemble_near(near_self, F, groups) + Far
    return {"leaf_box": leaves, "centres": centres[leaves], "h": hbox[leaves],
            "nodes_unit": nodes_unit, "node_pot": node_pot}


def volume_solver(f, theta0, tol=1e-4, max_depth=5, min_level=2, p=6, ng=5,
                  domain=(0.0, 1.0), ns_self=14, batch_size=256, vol_ops=None,
                  max_boxes=None, max_leaves=None, m2l_n_pairs=None):
    r"""Build a JIT + autodiff volume solve on a FIXED octree.

    The tree is refined once for the density ``f(., theta0)``; the returned
    ``solve(theta)`` evaluates :math:`u(x) = \int G(x,y) f(y, \theta)\, dy` at
    the leaf Gauss nodes of that fixed tree. ``solve`` is a pure-``jax``
    function: ``jax.jit``, ``jax.grad`` and ``jax.jacobian`` w.r.t. ``theta``
    all work. The tree topology is held fixed (it is discrete /
    non-differentiable, like a mesh in an adjoint solve); rebuild the solver
    when you want the discretisation to follow ``theta``.

    Parameters
    ----------
    f : callable
        JAX-traceable ``f(Y, theta)`` with ``Y`` of shape ``(..., 3)`` →
        ``(...)``.
    theta0 : array_like
        Reference parameters used to refine the tree.
    tol, max_depth, min_level, p, ng, domain, ns_self, batch_size, vol_ops,
    max_boxes, max_leaves, m2l_n_pairs
        As in :func:`volume_potential_adaptive_jit`.

    Returns
    -------
    solve : callable
        ``solve(theta) -> node_pot`` of shape ``(n_leaves, M)``,
        differentiable in ``theta``.
    info : dict
        ``leaf_box``, ``centres``, ``h``, ``nodes_unit`` (to
        interpret/evaluate ``node_pot``).
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxfmm.trees import morton_to_coords
    from jaxfmm.trees import (build_interaction_lists, build_wx_lists,
                                     build_ulist_adaptive)
    from jaxfmm import kifmm as fmm

    if vol_ops is None:
        vol_ops = volume_operators(p, ng, ns_self)
    ops = vol_ops["ops"]; S_self = vol_ops["S_self"]
    nodes_unit, gw = tensor_gauss(ng); M = ng ** 3
    md = int(max_depth)
    if max_boxes is None:
        max_boxes = sum(8 ** k for k in range(1, min_level + 1)) + 8 ** max_depth + 64

    # ── SETUP (host, once): fixed tree + all theta-independent structures ──
    T = build_volume_tree_jit(f, tol=tol, max_depth=max_depth, min_level=min_level,
                              n=ng, domain=domain, max_boxes=max_boxes, theta=theta0)
    m_id = jnp.asarray(T["m_id"]); p_id = jnp.asarray(T["p_id"])
    lvls = jnp.asarray(T["lvls"]); is_leaf = jnp.asarray(T["is_leaf"])
    de = float(T["domain_extent"]); pmin = T["pmin"]
    lvlsn = np.asarray(T["lvls"]); leaves = np.asarray(T["leaves"]); nL = len(leaves)
    if max_leaves is None:
        max_leaves = max_boxes
    if nL > max_leaves:
        raise ValueError(f"n_leaves={nL} exceeds max_leaves={max_leaves}")
    centres = np.asarray(morton_to_coords(m_id, lvls, de, pmin))
    hbox = de * 0.5 ** (lvlsn + 1.0)

    # fixed leaf Gauss nodes (theta-independent) + per-leaf charge scale (gw*h^3)
    cl = centres[leaves]; hl = hbox[leaves]
    node_pos = (nodes_unit[None] * hl[:, None, None] + cl[:, None, :]).reshape(-1, 3)
    positions = np.zeros((max_leaves * M, 3)); positions[:nL * M] = node_pos
    pil = -np.ones((max_boxes, M), np.int32); pil[leaves] = np.arange(nL * M).reshape(nL, M)
    positions = jnp.asarray(positions); pil = jnp.asarray(pil)
    node_pos_j = jnp.asarray(node_pos)
    cs_scale = jnp.asarray(gw[None] * (hl ** 3)[:, None])     # (nL,M)
    hl_j = jnp.asarray(hl); S_self_j = jnp.asarray(S_self)

    # interaction lists (constants for this fixed tree)
    _, V = build_interaction_lists(m_id, lvls, is_leaf, max_boxes, md)
    W, X, _ = build_wx_lists(m_id, lvls, is_leaf, max_boxes, md)
    U, _ = build_ulist_adaptive(m_id, lvls, is_leaf, max_boxes, md)
    if m2l_n_pairs is None:
        m2l_n_pairs = fmm.v_pair_count(V, batch_size)

    # near-field groups (constant index sets; cached relative operators as jnp)
    Un = np.asarray(U)
    pos_of = np.full(max_boxes, -1, np.int64); pos_of[leaves] = np.arange(nL)
    groups = []
    valid = Un[:, 0] >= 0
    if valid.any():
        tb = Un[valid, 0]; sb = Un[valid, 1]
        ct = centres[tb]; cs = centres[sb]; ht = hbox[tb]; hs = hbox[sb]
        rel = np.round((ct - cs) / hs[:, None], 9); ratio = np.round(ht / hs, 9)
        uniq, inv = np.unique(np.column_stack([ratio, rel]), axis=0, return_inverse=True)
        tp = pos_of[tb]; sp = pos_of[sb]; hs2 = hs ** 2
        for k in range(uniq.shape[0]):
            mk = inv == k
            op = _near_op_rel(ng, uniq[k, 0], uniq[k, 1:4])
            groups.append((jnp.asarray(op), jnp.asarray(tp[mk]),
                           jnp.asarray(sp[mk]), jnp.asarray(hs2[mk])))

    M2L_K = jnp.asarray(ops["M2L_K_table"]); pinv_dn = jnp.asarray(ops["pinv_dn"])
    eqp = jnp.asarray(ops["equiv_pts"])
    equiv_pts_np = np.asarray(ops["equiv_pts"]); pinv_dn_np = np.asarray(ops["pinv_dn"])
    n_s = equiv_pts_np.shape[0]
    # precomputed scale-invariant P2M / L2P operators + M2M / L2L mats (replace the
    # per-box on-the-fly kernel eval of fmm.upward_pass / fmm.downward_pass)
    G_s2m_j = jnp.asarray(vol_ops["G_s2m"]); G_l2p_j = jnp.asarray(vol_ops["G_l2p"])
    pinv_up_j = jnp.asarray(ops["pinv_up"])
    M2M_j = jnp.asarray(ops["M2M_mats"]); L2L_j = jnp.asarray(ops["L2L_mats"])
    factored = "F_up" in ops                       # backward-stable decodes (see the
    if factored:                                   # uniform driver): the volume path runs
        dec_up = tuple(jnp.asarray(x) for x in ops["F_up"])      # HIGH orders, where the
        dec_dn = tuple(jnp.asarray(x) for x in ops["F_dn"])      # materialised pinv products
        F_par_j = tuple(jnp.asarray(x) for x in ops["F_par"])    # cap f64 accuracy
        M2M_K_j = jnp.asarray(ops["M2M_K"]); L2L_K_j = jnp.asarray(ops["L2L_K"])
    else:
        dec_up, dec_dn = pinv_up_j, pinv_dn
    leaf_idx_j = jnp.asarray(leaves); inv_hl_j = jnp.asarray(1.0 / hl)

    # V-list (m2l) displacement groups: one batched GEMM per relative box
    # displacement (the regular-grid convolution), replacing fmm.m2l's per-pair
    # K-matrix gather.  See _m2l_disp_groups.
    m2l_groups = _m2l_disp_groups(V, m_id, ops["M2L_K_table"])

    # W-list (m2p) groups: source box equiv density (n_s) -> target leaf node pot (M),
    # via precomputed scale-invariant relative operators (see _m2p_op_rel), per-pair
    # scale 1/h_src.  Replaces the on-the-fly per-pair kernel eval in fmm.m2p.
    Wn = np.asarray(W); groups_W = []
    okW = (Wn[:, 0] >= 0) & (Wn[:, 1] >= 0)
    if okW.any():
        tbW = Wn[okW, 0]; sbW = Wn[okW, 1]
        relW = np.round((centres[tbW] - centres[sbW]) / hbox[sbW][:, None], 9)
        ratioW = np.round(hbox[tbW] / hbox[sbW], 9)
        uniq, inv = np.unique(np.column_stack([ratioW, relW]), axis=0, return_inverse=True)
        tpW = pos_of[tbW]; inv_hsW = 1.0 / hbox[sbW]
        for k in range(uniq.shape[0]):
            mk = inv == k
            op = _m2p_op_rel(ng, equiv_pts_np, uniq[k, 0], uniq[k, 1:4])     # (M, n_s)
            groups_W.append((jnp.asarray(op), jnp.asarray(tpW[mk]),
                             jnp.asarray(sbW[mk]), jnp.asarray(inv_hsW[mk])))

    # X-list (p2l) groups: source leaf charges (M) -> target box local density (n_s),
    # via precomputed relative operators (see _x_op_rel), per-pair scale ratio.
    Xn = np.asarray(X); groups_X = []
    okX = (Xn[:, 0] >= 0) & (Xn[:, 1] >= 0)
    if okX.any():
        tbX = Xn[okX, 0]; sbX = Xn[okX, 1]
        relX = np.round((centres[tbX] - centres[sbX]) / hbox[sbX][:, None], 9)
        ratioX = np.round(hbox[tbX] / hbox[sbX], 9)
        uniq, inv = np.unique(np.column_stack([ratioX, relX]), axis=0, return_inverse=True)
        spX = pos_of[sbX]
        for k in range(uniq.shape[0]):
            mk = inv == k
            op = _x_op_rel(ng, equiv_pts_np, uniq[k, 0], uniq[k, 1:4])  # (n_s, M) raw-K
            groups_X.append((jnp.asarray(op), jnp.asarray(tbX[mk]),
                             jnp.asarray(spX[mk]), jnp.asarray(ratioX[mk])))

    # ── SOLVE (pure jax: jit + autodiff w.r.t. theta) ──
    @jax.jit
    def solve(theta):
        F = f(node_pos_j, theta).reshape(nL, M)                  # (nL,M) — diff in theta
        chLM = F * cs_scale                                      # (nL,M) volume charges q=f·w·h³
        equiv = jnp.zeros((max_boxes, n_s)).at[leaf_idx_j].set(  # P2M (precomputed S2M) + M2M
            fmm._decode(chLM @ G_s2m_j.T, dec_up))
        equiv = (fmm.m2m_factored(equiv, m_id, p_id, lvls, M2M_K_j, F_par_j, md) if factored
                 else fmm.m2m(equiv, m_id, p_id, lvls, M2M_j, md))
        check_pot = jnp.zeros((max_boxes, n_s))                  # V-list (m2l), sliced by disp
        for Kd, tgt, src in m2l_groups:
            check_pot = check_pot.at[tgt].add(equiv[src] @ Kd.T)
        for op, tgt, src, scale in groups_X:                     # X-list (p2l): RAW-K into the
            check_pot = check_pot.at[tgt].add(                   # check potentials -> ONE decode
                (chLM[src] @ op.T) * scale[:, None])             # (with M2L) per target box
        local = fmm._decode(check_pot, dec_dn)
        local = (fmm.l2l_factored(local, m_id, p_id, lvls, L2L_K_j, dec_dn, md) if factored
                 else fmm.l2l(local, m_id, p_id, lvls, L2L_j, md))   # L2L
        Far = (local[leaf_idx_j] @ G_l2p_j.T) * inv_hl_j[:, None]  # L2P (precomputed)
        for op, tgt, src, scale in groups_W:                     # W-list (m2p)
            Far = Far.at[tgt].add((equiv[src] @ op.T) * scale[:, None])
        near = (hl_j ** 2)[:, None] * (F @ S_self_j.T)
        for op, tgt, src, scale in groups:                       # U-list (near/self)
            near = near.at[tgt].add((F[src] @ op.T) * scale[:, None])
        return near + Far

    info = {"leaf_box": leaves, "centres": cl, "h": hl, "nodes_unit": nodes_unit}
    return solve, info


# ── relative-config key encoding for the fully-jit near field ────────────────
_CFG_B = 256; _CFG_R = 128


def _cfg_encode(qratio, qrel):
    r"""Encode an integer relative-config key (``qratio`` scalar, ``qrel`` last-axis-3)
    into one int64. Works for numpy or jnp integer arrays."""
    k = qratio + _CFG_R
    for j in range(3):
        k = k * _CFG_B + (qrel[..., j] + _CFG_R)
    return k


def _cfg_quant(ratio, rel, xp):
    return (xp.round(ratio * 2).astype(xp.int64), xp.round(rel * 8).astype(xp.int64))


def volume_solver_jit(f, ref_theta, tol=1e-4, max_depth=5, min_level=2, p=6, ng=5,
                      domain=(0.0, 1.0), ns_self=14, batch_size=256, max_boxes=None,
                      m2l_n_pairs=None, validate_theta=None):
    r"""Fully self-contained, jit + autodiff volume solve.

    ONE ``jax.jit`` function ``solve(theta)`` that refines the octree,
    2:1-balances it, builds the interaction lists, and evaluates
    :math:`u = \int G\, f(\cdot, \theta)` — all inside the trace.

    Unlike :func:`volume_solver` (which freezes the tree at build time), the
    tree here follows ``theta`` on its own: a new ``theta`` re-refines and
    re-solves in one warm call with no recompilation. ``solve`` is
    differentiable w.r.t. ``theta`` — the discrete tree topology is
    ``stop_gradient``-ed, so ``jax.grad`` returns the (correct) fixed-tree
    gradient and reverse mode does not tape the refine/balance loops.

    Parameters
    ----------
    f : callable
        JAX-traceable ``f(Y, theta)`` with ``Y`` of shape ``(..., 3)`` →
        ``(...)``.
    ref_theta : array_like
        Reference parameters for the host-side relative-operator tables.
    tol, max_depth, min_level, p, ng, domain, ns_self, batch_size, max_boxes,
    m2l_n_pairs
        As in :func:`volume_potential_adaptive_jit`.
    validate_theta : array_like, optional
        Assert the relative-config tables are complete at a second theta (no
        new configs).

    Returns
    -------
    callable
        ``solve(theta) -> (node_pot, is_leaf, centres, h)``, all fixed shape
        ``(max_boxes, M)`` / ``(max_boxes,)``. ``node_pot`` is the per-box
        leaf-node potential (valid where ``is_leaf``) and is differentiable
        w.r.t. ``theta``.

    Notes
    -----
    The near field uses a bounded, scale-invariant set of relative near
    operators precomputed once on the host from the ``ref_theta`` tree;
    overflowing a frozen config bin drops pairs, so validate when the
    geometry family drifts.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxfmm.trees import morton_to_coords, _morton_to_grid
    from jaxfmm.trees import (build_interaction_lists, build_wx_lists,
                                     build_ulist_adaptive)
    from jaxfmm import kifmm as fmm

    vo = volume_operators(p, ng, ns_self); ops = vo["ops"]
    nodes_unit, gw = tensor_gauss(ng); M = ng ** 3
    md = int(max_depth); lo, hi = float(domain[0]), float(domain[1]); de = hi - lo
    if max_boxes is None:
        max_boxes = sum(8 ** k for k in range(1, min_level + 1)) + 8 ** max_depth + 64
    nu = jnp.asarray(nodes_unit); gwj = jnp.asarray(gw)
    S_self_j = jnp.asarray(vo["S_self"]); pmin_j = jnp.asarray(np.full(3, lo))
    M2L_K = jnp.asarray(ops["M2L_K_table"]); pinv_dn = jnp.asarray(ops["pinv_dn"])
    eqp = jnp.asarray(ops["equiv_pts"])
    equiv_pts_np = np.asarray(ops["equiv_pts"]); pinv_dn_np = np.asarray(ops["pinv_dn"])
    n_s = equiv_pts_np.shape[0]
    K_flat_j = jnp.asarray(np.asarray(ops["M2L_K_table"]).reshape(343, n_s, n_s))
    # precomputed P2M / L2P operators + M2M / L2L mats (replace per-box kernel evals)
    G_s2m_j = jnp.asarray(vo["G_s2m"]); G_l2p_j = jnp.asarray(vo["G_l2p"])
    pinv_up_j = jnp.asarray(ops["pinv_up"])
    M2M_j = jnp.asarray(ops["M2M_mats"]); L2L_j = jnp.asarray(ops["L2L_mats"])
    factored = "F_up" in ops                       # backward-stable decodes (see the uniform
    if factored:                                   # driver): the volume path runs HIGH P,
        dec_up = tuple(jnp.asarray(x) for x in ops["F_up"])      # where materialised pinv
        dec_dn = tuple(jnp.asarray(x) for x in ops["F_dn"])      # products cap f64 ~9 digits
        F_par_j = tuple(jnp.asarray(x) for x in ops["F_par"])
        M2M_K_j = jnp.asarray(ops["M2M_K"]); L2L_K_j = jnp.asarray(ops["L2L_K"])
    else:
        dec_up, dec_dn = pinv_up_j, pinv_dn
    # M2L displacement stencil: all d in [-3,3]^3 that are well-separated (|d|inf >= 2);
    # the near 3^3 are never in the V-list.  Static -> the sliced m2l loop unrolls.
    stencil_disps = tuple(
        (dx + 3) * 49 + (dy + 3) * 7 + (dz + 3)
        for dx in range(-3, 4) for dy in range(-3, 4) for dz in range(-3, 4)
        if max(abs(dx), abs(dy), abs(dz)) >= 2)
    # M2L stencil padded to a multiple of the chunk size with the unused centre
    # displacement (171, never a V-pair -> contributes 0); scanned in chunks so the
    # (max_boxes, chunk, n_s) gather stays sequential (no fusion into one ~17GB gather).
    _M2L_CHUNK = 32
    _n_chunk = (len(stencil_disps) + _M2L_CHUNK - 1) // _M2L_CHUNK
    stencil_pad = jnp.asarray(stencil_disps + (171,) * (_n_chunk * _M2L_CHUNK - len(stencil_disps))
                              ).reshape(_n_chunk, _M2L_CHUNK)
    K_chunks_j = K_flat_j[stencil_pad]                         # (_n_chunk, chunk, n_s, n_s)

    def _tree(theta):
        a = _vtree_refine(lambda Y: f(Y, theta), tol, max_depth, min_level, ng, lo, hi, max_boxes)
        b = _vtree_balance(*a, max_depth, min_level, max_boxes)
        return _vtree_emit(b[0], b[1], b[2], b[3], max_boxes, max_depth)
    _tree_j = jax.jit(_tree)

    def _configs(theta, which):
        r"""Relative (ratio, rel) configs of the U / W / X near-or-cross lists at theta."""
        m_id, _, lvls, is_leaf, _ = (np.asarray(x) for x in _tree_j(jnp.asarray(theta)))
        mj, lj, ij = jnp.asarray(m_id), jnp.asarray(lvls), jnp.asarray(is_leaf)
        centres = np.asarray(morton_to_coords(mj, lj, de, np.full(3, lo)))
        hbox = de * 0.5 ** (lvls + 1.0)
        if which == "U":
            L = np.asarray(build_ulist_adaptive(mj, lj, ij, max_boxes, md)[0])
        else:                                    # W (idx 0) or X (idx 1) of build_wx_lists
            wx = build_wx_lists(mj, lj, ij, max_boxes, md)
            L = np.asarray(wx[0 if which == "W" else 1])
        v = (L[:, 0] >= 0) & (L[:, 1] >= 0); tb = L[v, 0]; sb = L[v, 1]
        return hbox[tb] / hbox[sb], (centres[tb] - centres[sb]) / hbox[sb][:, None]

    def _counts(r, rl, ksort):
        r"""(number of valid pairs, per-config counts) for configs present in ksort."""
        key = np.asarray(_cfg_encode(*_cfg_quant(r, rl, np)))
        idx = np.searchsorted(ksort, key)
        ok = (idx < len(ksort)) & (ksort[np.clip(idx, 0, len(ksort) - 1)] == key)
        return len(key), np.bincount(idx[ok], minlength=len(ksort))

    def _cfg_table(which, op_fn, name):
        r"""HOST: scale-invariant relative-operator table + sorted key array + static
        bin sizes ``(n_pairs, CAP)`` for the in-trace config-bin batched apply.
        Returns ``(ops_j, ksort_j, n_cfg, n_pairs, CAP)`` or None if the list is
        empty at ref_theta. ``n_pairs``/``CAP`` are sized from ref (and
        validate_theta) with margin."""
        r1, rl1 = _configs(ref_theta, which)
        if len(r1) == 0:
            return None
        keyint = np.asarray(_cfg_encode(*_cfg_quant(r1, rl1, np)))
        n_raw = len(np.unique(np.column_stack([np.round(r1, 9), np.round(rl1, 9)]), axis=0))
        if n_raw != len(np.unique(keyint)):
            raise RuntimeError(f"{name}-config quantisation collision; increase _cfg resolution")
        uk, ui = np.unique(keyint, return_index=True)
        order = np.argsort(uk); ksort = uk[order]
        table = np.stack([op_fn(r1[ui[order[k]]], rl1[ui[order[k]]]) for k in range(len(order))])
        npair, cnt = _counts(r1, rl1, ksort)
        if validate_theta is not None:
            r2, rl2 = _configs(validate_theta, which)
            if len(r2):
                miss = np.setdiff1d(np.unique(np.asarray(_cfg_encode(*_cfg_quant(r2, rl2, np)))), ksort)
                if miss.size:
                    raise RuntimeError(f"{name}-config table incomplete: {miss.size} new configs at validate_theta")
                np2, cnt2 = _counts(r2, rl2, ksort)
                npair = max(npair, np2); cnt = np.maximum(cnt, cnt2)
        gran = 256
        n_pairs = int(np.ceil(npair * 1.15 / gran) * gran)      # +15% margin (theta drift)
        CAP = int(np.ceil(int(cnt.max()) * 1.30 / gran) * gran)  # +30% margin per config
        return jnp.asarray(table), jnp.asarray(ksort), table.shape[0], n_pairs, CAP

    near_tab = _cfg_table("U", lambda r, rl: _near_op_rel(ng, r, rl), "near")
    W_tab    = _cfg_table("W", lambda r, rl: _m2p_op_rel(ng, equiv_pts_np, r, rl), "W")
    X_tab    = _cfg_table("X", lambda r, rl: _x_op_rel(ng, equiv_pts_np, r, rl), "X")
    ops_table_j, ksort_j, n_cfg, near_np, near_CAP = near_tab   # U-list always non-empty
    if m2l_n_pairs is None:
        t0 = _tree_j(jnp.asarray(ref_theta))      # (m_id, p_id, lvls, is_leaf, n_active)
        _, V0 = build_interaction_lists(t0[0], t0[2], t0[3], max_boxes, md)
        m2l_n_pairs = fmm.v_pair_count(V0, batch_size)

    @jax.jit
    def solve(theta):
        ff = lambda Y: f(Y, theta)
        a = _vtree_refine(ff, tol, max_depth, min_level, ng, lo, hi, max_boxes)
        b = _vtree_balance(*a, max_depth, min_level, max_boxes)
        m_id, p_id, lvls, is_leaf, _ = _vtree_emit(b[0], b[1], b[2], b[3], max_boxes, max_depth)
        sg = jax.lax.stop_gradient            # freeze discrete topology (non-diff in theta)
        m_id, p_id, lvls, is_leaf = sg(m_id), sg(p_id), sg(lvls), sg(is_leaf)
        centres, h = fmm.box_geometry(m_id, lvls, de, pmin_j)
        centres, h = sg(centres), sg(h)
        leafmask = is_leaf.astype(jnp.float64)[:, None]
        pts = nu[None] * h[:, None, None] + centres[:, None, :]
        positions = pts.reshape(-1, 3)
        F = ff(positions).reshape(max_boxes, M)                       # diff in theta
        charges = (F * (gwj[None] * (h ** 3)[:, None]) * leafmask).reshape(-1)
        pil = jnp.where(is_leaf[:, None],
                        jnp.arange(max_boxes * M, dtype=jnp.int32).reshape(max_boxes, M),
                        jnp.int32(-1))
        # P2M (precomputed S2M) + M2M  (replaces fmm.upward_pass; non-leaf P2M = 0)
        equiv = fmm._decode(charges.reshape(max_boxes, M) @ G_s2m_j.T, dec_up)
        equiv = (fmm.m2m_factored(equiv, m_id, p_id, lvls, M2M_K_j, F_par_j, md) if factored
                 else fmm.m2m(equiv, m_id, p_id, lvls, M2M_j, md))
        _, V = build_interaction_lists(m_id, lvls, is_leaf, max_boxes, md)
        W, X, _ = build_wx_lists(m_id, lvls, is_leaf, max_boxes, md)
        # V-list (m2l) sliced by displacement on the regular box grid: build a
        # (max_boxes, 343) neighbour table (one source per displacement per target,
        # conflict-free) then apply each K_d once via a batched GEMM.
        gx, gy, gz = (g.astype(jnp.int32) for g in _morton_to_grid(m_id))
        vt, vs = V[:, 0], V[:, 1]; vv = (vt >= 0) & (vs >= 0)
        vt_s = jnp.where(vv, vt, 0); vs_s = jnp.where(vv, vs, 0)
        dispv = ((gx[vt_s] - gx[vs_s] + 3) * 49 + (gy[vt_s] - gy[vs_s] + 3) * 7
                 + (gz[vt_s] - gz[vs_s] + 3))           # invalid pairs -> 171 (centre, unread)
        nbr = jnp.full((max_boxes, 343), -1, jnp.int32).at[vt_s, dispv].set(
            jnp.where(vv, vs_s, -1).astype(jnp.int32))
        # chunked batched contraction over the displacement stencil, scanned so each
        # (max_boxes, chunk, n_s) gather is sequential (fewer launches than one-per-
        # displacement, bounded memory).  equiv[nbr] zeroed where no neighbour.
        def _m2l_chunk(cp, dk):
            ds, Kc = dk                                        # (chunk,), (chunk,n_s,n_s)
            sd = nbr[:, ds]; vd = sd >= 0                      # (max_boxes, chunk)
            e = equiv[jnp.where(vd, sd, 0)] * vd[..., None]    # (max_boxes, chunk, n_s)
            return cp + jnp.einsum("cts,bcs->bt", Kc, e), None
        check_pot, _ = jax.lax.scan(_m2l_chunk, jnp.zeros((max_boxes, n_s)),
                                    (stencil_pad, K_chunks_j))
        # X-list (p2l): config-bin + single batched GEMM, RAW-K into the CHECK potentials
        # so M2L + X share ONE decode per target box (exact: the decode is linear per row)
        if X_tab is not None:
            opsX, ksX, ncX, npX, capX = X_tab
            vX = (X[:, 0] >= 0) & (X[:, 1] >= 0)
            check_pot = check_pot + _binned_cfg_apply(charges.reshape(max_boxes, M), X[:, 0],
                                                      X[:, 1], vX, centres, h, ksX, ncX, opsX,
                                                      "ratio", npX, capX, max_boxes, n_s)
            local = fmm._decode(check_pot, dec_dn)
        else:
            local = fmm._decode(check_pot, dec_dn)
            local = fmm.p2l(local, positions, charges, pil, centres, h, X, eqp, dec_dn, batch_size=batch_size)
        local = (fmm.l2l_factored(local, m_id, p_id, lvls, L2L_K_j, dec_dn, md) if factored
                 else fmm.l2l(local, m_id, p_id, lvls, L2L_j, md))        # L2L
        Far = (local @ G_l2p_j.T) * (1.0 / h)[:, None]                    # L2P (precomputed)
        # W-list (m2p): config-bin + single batched GEMM
        if W_tab is not None:
            opsW, ksW, ncW, npW, capW = W_tab
            vW = (W[:, 0] >= 0) & (W[:, 1] >= 0)
            Far = Far + _binned_cfg_apply(equiv, W[:, 0], W[:, 1], vW, centres, h, ksW, ncW,
                                          opsW, "inv_hs", npW, capW, max_boxes, M)
        else:
            Far = Far + fmm.m2p(equiv, positions, pil, centres, h, W, eqp,
                                batch_size=batch_size).reshape(max_boxes, M)
        # U-list near field: config-bin + single batched GEMM (+ the self term)
        U = build_ulist_adaptive(m_id, lvls, is_leaf, max_boxes, md)[0]
        uvalid = (U[:, 0] >= 0) & (U[:, 1] >= 0)
        near = (h ** 2)[:, None] * (F @ S_self_j.T) * leafmask
        near = near + _binned_cfg_apply(F, U[:, 0], U[:, 1], uvalid, centres, h, ksort_j, n_cfg,
                                        ops_table_j, "hs2", near_np, near_CAP, max_boxes, M)
        return (near + Far) * leafmask, is_leaf, centres, h

    return solve


def volume_evaluate_points(result, points):
    r"""Evaluate the volume potential at arbitrary query points.

    Interpolates the per-leaf potential polynomial produced by
    :func:`volume_potential_uniform` / :func:`volume_potential_adaptive`.

    Parameters
    ----------
    result : dict
        Output of a volume driver (``centres``, ``h``, ``nodes_unit``,
        ``node_pot``).
    points : array_like
        Query coordinates, shape ``(K, 3)`` (must lie within the leaf cells).

    Returns
    -------
    np.ndarray
        Potential at each query point, shape ``(K,)``.
    """
    centres = np.asarray(result["centres"]); h = np.atleast_1d(np.asarray(result["h"]))
    if h.shape[0] == 1:
        h = np.full(centres.shape[0], float(h[0]))
    node_pot = np.asarray(result["node_pot"])
    M = node_pot.shape[1]
    ng = int(round(M ** (1.0 / 3.0)))
    pts = np.asarray(points, float)
    out = np.empty(pts.shape[0])
    for k in range(pts.shape[0]):
        # leaf whose box contains the point (boxes tile the domain)
        inside = np.all(np.abs(pts[k] - centres) <= h[:, None] + 1e-9, axis=1)
        b = int(np.argmax(inside))
        local = ((pts[k] - centres[b]) / h[b])[None, :]
        out[k] = (interp_matrix(ng, local) @ node_pot[b])[0]
    return out


# ── Dedicated volume octree builder ─────────────────────────────────────────
def _morton_encode(ix, iy, iz):
    r"""Interleave bits of ``(ix, iy, iz)`` into a Morton code (bit0=x, bit1=y, bit2=z)."""
    def part(v):
        v = int(v) & 0x1FFFFF
        v = (v | (v << 32)) & 0x1F00000000FFFF
        v = (v | (v << 16)) & 0x1F0000FF0000FF
        v = (v | (v << 8)) & 0x100F00F00F00F00F
        v = (v | (v << 4)) & 0x10C30C30C30C30C3
        v = (v | (v << 2)) & 0x1249249249249249
        return v
    return part(ix) | (part(iy) << 1) | (part(iz) << 2)


def _balance_leaves(leaves, max_depth):
    r"""2:1-balance an explicit leaf set (list of ``(level, ix, iy, iz)``).

    Returns a balanced set where no two face/edge/corner-adjacent leaves
    differ by more than one level.
    """
    leafset = set(leaves)                       # (level, ix, iy, iz)
    changed = True
    while changed:
        changed = False
        # quick lookup: morton-at-level present as a leaf
        present = {}                            # level -> set of morton
        for (l, ix, iy, iz) in leafset:
            present.setdefault(l, set()).add(_morton_encode(ix, iy, iz))
        for (l, ix, iy, iz) in list(leafset):
            if l < 2:
                continue
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == dy == dz == 0:
                            continue
                        nx, ny, nz = ix + dx, iy + dy, iz + dz
                        if not (0 <= nx < 2 ** l and 0 <= ny < 2 ** l and 0 <= nz < 2 ** l):
                            continue
                        nm = _morton_encode(nx, ny, nz)
                        # find the leaf covering this neighbour cell (self or coarser ancestor)
                        cover = None
                        for lc in range(l, 0, -1):
                            mm = nm >> (3 * (l - lc))
                            if lc in present and mm in present[lc]:
                                cover = (lc, mm); break
                        if cover is not None and cover[0] < l - 1:
                            # subdivide the too-coarse covering leaf
                            lc, mm = cover
                            cx, cy, cz = _morton_decode(mm)
                            leafset.discard((lc, cx, cy, cz))
                            for c in range(8):
                                leafset.add((lc + 1, 2 * cx + (c & 1), 2 * cy + ((c >> 1) & 1),
                                             2 * cz + ((c >> 2) & 1)))
                            changed = True
        # presence dict is stale after edits; loop restarts and rebuilds it
    return leafset


def _morton_decode(m):
    r"""Inverse of :func:`_morton_encode`: Morton code → ``(ix, iy, iz)``."""
    def comp(v):
        v = int(v) & 0x1249249249249249
        v = (v | (v >> 2)) & 0x10C30C30C30C30C3
        v = (v | (v >> 4)) & 0x100F00F00F00F00F
        v = (v | (v >> 8)) & 0x1F0000FF0000FF
        v = (v | (v >> 16)) & 0x1F00000000FFFF
        v = (v | (v >> 32)) & 0x1FFFFF
        return v
    return comp(m), comp(m >> 1), comp(m >> 2)


def build_volume_tree(f, tol=1e-4, max_depth=5, min_level=2, n=5, domain=(0.0, 1.0)):
    r"""Dedicated volume octree builder (host-side, eager).

    Refines by interpolation error, 2:1-balances, and emits FMM-format arrays
    over the EXACT ``domain`` (no point-tree proxy).

    Parameters
    ----------
    f, tol, max_depth, min_level, n, domain
        As in :func:`refine_cells`.

    Returns
    -------
    dict
        FMM-compatible tree: ``m_id`` (uint64), ``p_id`` (int32), ``lvls``
        (int32), ``is_leaf`` (bool) of shape ``(n_boxes,)``, plus
        ``domain_extent`` (float), ``pmin`` ``(3,)``, ``max_depth`` (int),
        and ``leaves`` (int32 indices of the leaf boxes).
    """
    lo, hi = domain
    raw = refine_cells(f, n=n, tol=tol, max_depth=max_depth, min_level=min_level, domain=domain)
    leafset = _balance_leaves([tuple(L) for L in raw], max_depth)

    # collect every box: leaves + all their ancestors down to level 1
    boxes = {}                                   # (level, morton) -> is_leaf
    for (l, ix, iy, iz) in leafset:
        m = _morton_encode(ix, iy, iz)
        boxes[(l, m)] = True
        for lc in range(l - 1, 0, -1):
            anc = (lc, m >> (3 * (l - lc)))
            if anc not in boxes:
                boxes[anc] = False               # internal (unless later found as leaf)

    keys = sorted(boxes.keys(), key=lambda lm: (-lm[0], lm[1]))   # fine-to-coarse
    n_boxes = len(keys)
    index = {lm: i for i, lm in enumerate(keys)}
    m_id = np.zeros(n_boxes, dtype=np.uint64)
    lvls = np.zeros(n_boxes, dtype=np.int32)
    is_leaf = np.zeros(n_boxes, dtype=bool)
    p_id = np.zeros(n_boxes, dtype=np.int32)
    for i, (l, m) in enumerate(keys):
        m_id[i] = m; lvls[i] = l; is_leaf[i] = boxes[(l, m)]
        p_id[i] = i if l == 1 else index[(l - 1, m >> 3)]

    return {"m_id": m_id, "p_id": p_id, "lvls": lvls, "is_leaf": is_leaf,
            "domain_extent": float(hi - lo), "pmin": np.full(3, float(lo)),
            "max_depth": int(max(l for (l, _) in keys)),
            "leaves": np.where(is_leaf)[0].astype(np.int32)}


# Imported at MODULE level (not inside the jitted builder) so trees.cubic's import-time
# constants (_NEIGH_OFFSETS etc.) are materialised eagerly, never under a jit trace.
import jax as _jax
from jaxfmm.trees import (_morton_to_grid as _m2g, get_26_neighbors as _get26,
                         _ADJ_CHILD_MASK as _ADJ)


# ── JIT top-down volume octree builder (fixed-shape, on-GPU) ─────────────────
# Mirrors jaxfmm.trees.build_fmm_tree's design (fixed max_boxes buffers, rooted at
# level 1, fine-to-coarse layout) but the leaf decision is the density's
# interpolation error rather than a point count, and refinement grows top-down so
# only cells that actually refine are instantiated.  Output shapes depend only on
# max_boxes (static), so the downstream KIFMM compiles ONCE and is reused across
# geometries (no per-density retrace of the expensive far field).
#
# `f` must be a JAX-traceable callable f(Y) with Y of shape (..., 3) -> (...).

import functools as _functools


def _vtree_refine(f, tol, max_depth, min_level, ng, lo, hi, max_boxes):
    r"""Phase A: top-down refinement into fixed ``max_boxes`` buffers (no balance)."""
    jax, jnp = _jax, _jnp
    MAXU = ~jnp.array(0, dtype=jnp.uint64)
    _morton_to_grid = _m2g
    extent = float(hi - lo); pmin = jnp.full(3, float(lo))
    nodes_unit = jnp.asarray(tensor_gauss(ng)[0])
    C = jnp.asarray(_legendre_coeff_matrix_1d(ng))

    gs = jnp.linspace(lo, hi, 4 * 2 ** min_level + 1)
    GX, GY, GZ = jnp.meshgrid(gs, gs, gs, indexing="ij")
    f_scale = jnp.max(jnp.abs(f(jnp.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 1))))
    thresh = tol * jnp.maximum(f_scale, 1e-300)

    def box_tail(mort, lvl_arr):
        gx, gy, gz = _morton_to_grid(mort)
        coords = jnp.stack([gx, gy, gz], -1).astype(jnp.float64)
        h = extent * 0.5 ** (lvl_arr[:, None] + 1)
        centre = coords * extent / (2.0 ** lvl_arr[:, None]) + h + pmin
        phys = nodes_unit[None] * h[:, None] + centre[:, None, :]
        F = f(phys).reshape(-1, ng, ng, ng)
        coeffs = jnp.einsum("ai,bj,ck,nijk->nabc", C, C, C, F)
        total = jnp.sum(coeffs ** 2, axis=(1, 2, 3))
        inner = jnp.sum(coeffs[:, :-1, :-1, :-1] ** 2, axis=(1, 2, 3))
        return jnp.sqrt(jnp.maximum(total - inner, 0.0))

    # seed the full coarse pyramid (levels 1..min_level), rooted at level 1
    offs = [0] * (min_level + 2)
    for k in range(2, min_level + 1):
        offs[k] = offs[k - 1] + 8 ** (k - 1)
    nseed = offs[min_level] + 8 ** min_level
    s_m = np.empty(nseed, np.uint64); s_l = np.empty(nseed, np.int32); s_p = np.empty(nseed, np.int32)
    for k in range(1, min_level + 1):
        sl = slice(offs[k], offs[k] + 8 ** k)
        mk = np.arange(8 ** k, dtype=np.uint64)
        s_m[sl] = mk; s_l[sl] = k
        s_p[sl] = (offs[k] + np.arange(8 ** k)) if k == 1 else (offs[k - 1] + (mk >> 3)).astype(np.int32)
    idx = jnp.arange(max_boxes, dtype=jnp.uint64)
    mort = jnp.where(idx < nseed, jnp.concatenate([jnp.asarray(s_m), jnp.full(max_boxes - nseed, MAXU)]), MAXU)
    lvl  = jnp.where(idx < nseed, jnp.concatenate([jnp.asarray(s_l), jnp.zeros(max_boxes - nseed, jnp.int32)]), jnp.int32(0))
    leaf = jnp.zeros(max_boxes, dtype=bool)
    pidx = jnp.concatenate([jnp.asarray(s_p), jnp.arange(nseed, max_boxes, dtype=jnp.int32)])
    nb   = jnp.int32(nseed)
    allidx = jnp.arange(max_boxes, dtype=jnp.int32)

    def step(k, val):
        mort, lvl, leaf, pidx, nb = val
        l = jnp.int32(min_level) + k
        mask = (lvl == l)
        resolved = box_tail(mort, lvl) <= thresh
        split = mask & (~resolved)
        leaf = leaf | (mask & ~split)
        rank = jnp.cumsum(split.astype(jnp.int32)) - 1
        nsel = jnp.sum(split.astype(jnp.int32))
        for c in range(8):
            tgt = jnp.where(split, nb + 8 * rank + jnp.int32(c), jnp.int32(max_boxes))
            mort = mort.at[tgt].set((mort << jnp.uint64(3)) | jnp.uint64(c), mode="drop")
            lvl  = lvl.at[tgt].set(l + jnp.int32(1), mode="drop")
            leaf = leaf.at[tgt].set(False, mode="drop")
            pidx = pidx.at[tgt].set(allidx, mode="drop")
        return mort, lvl, leaf, pidx, (nb + 8 * nsel).astype(jnp.int32)

    val = jax.lax.fori_loop(0, max_depth - min_level, step, (mort, lvl, leaf, pidx, nb))
    mort, lvl, leaf, pidx, nb = val
    leaf = leaf | (lvl == max_depth)
    return mort, lvl, leaf, pidx, nb


def _vtree_balance(mort, lvl, leaf, pidx, nb, max_depth, min_level, max_boxes):
    r"""Phase B: JIT 2:1-balance ripple (split any leaf two or more levels coarser
    than an adjacent box, until balanced)."""
    jax, jnp = _jax, _jnp
    get_26_neighbors, _ADJ_CHILD_MASK = _get26, _ADJ
    MAXU = ~jnp.array(0, dtype=jnp.uint64)
    allidx = jnp.arange(max_boxes, dtype=jnp.int32)
    adj = jnp.asarray(_ADJ_CHILD_MASK)

    def one_pass(val):
        mort, lvl, leaf, pidx, nb, _ = val
        present = lvl > 0
        key = jnp.where(present, (lvl.astype(jnp.uint64) << jnp.uint64(56)) | mort, MAXU)
        order = jnp.argsort(key); ksort = key[order]
        slots = (mort & jnp.uint64(7)).astype(jnp.int32)
        pp = jnp.where(lvl > min_level, pidx, jnp.int32(max_boxes))
        child_box = jnp.zeros((max_boxes, 8), jnp.int32).at[pp, slots].set(allidx, mode="drop")
        neighs, valid = get_26_neighbors(mort, lvl)
        qkey = (lvl[:, None].astype(jnp.uint64) << jnp.uint64(56)) | neighs
        pos = jnp.clip(jnp.searchsorted(ksort, qkey, side="left", method="scan_unrolled"), 0, max_boxes - 1)
        found = valid & (ksort[pos] == qkey)
        nbr_idx = order[pos]
        nbr_internal = found & (~leaf[nbr_idx]) & present[nbr_idx]
        ch_internal = ~leaf[child_box[nbr_idx]]
        adj_int = ch_internal & adj[None, :, :] & nbr_internal[:, :, None]
        must_split = jnp.any(adj_int, axis=(1, 2)) & leaf & (lvl < max_depth) & present
        rank = jnp.cumsum(must_split.astype(jnp.int32)) - 1
        nsel = jnp.sum(must_split.astype(jnp.int32))
        leaf = jnp.where(must_split, False, leaf)
        for c in range(8):
            tgt = jnp.where(must_split, nb + 8 * rank + jnp.int32(c), jnp.int32(max_boxes))
            mort = mort.at[tgt].set((mort << jnp.uint64(3)) | jnp.uint64(c), mode="drop")
            lvl  = lvl.at[tgt].set(lvl + jnp.int32(1), mode="drop")
            leaf = leaf.at[tgt].set(True, mode="drop")
            pidx = pidx.at[tgt].set(allidx, mode="drop")
        return mort, lvl, leaf, pidx, (nb + 8 * nsel).astype(jnp.int32), nsel > 0

    val = jax.lax.while_loop(lambda v: v[-1], one_pass,
                             (mort, lvl, leaf, pidx, nb, jnp.bool_(True)))
    return val[0], val[1], val[2], val[3], val[4]


def _vtree_emit(mort, lvl, leaf, pidx, max_boxes, max_depth):
    r"""Phase C: reorder fine-to-coarse (Morton within level), remap parents, pad."""
    jnp = _jnp
    MAXU = ~jnp.array(0, dtype=jnp.uint64)
    present = lvl > 0
    key = jnp.where(present, ((jnp.uint64(max_depth) - lvl.astype(jnp.uint64)) << jnp.uint64(56)) | mort, MAXU)
    order = jnp.argsort(key)
    inv = jnp.zeros(max_boxes, jnp.int32).at[order].set(jnp.arange(max_boxes, dtype=jnp.int32))
    m_id = mort[order]; lvls = lvl[order]; is_leaf = leaf[order]
    p_id = inv[pidx[order]]
    n_active = jnp.sum(present)
    lvls = jnp.where(jnp.arange(max_boxes) < n_active, lvls, jnp.int32(0))
    is_leaf = is_leaf & (lvls > 0)
    p_id = jnp.where(lvls == 1, jnp.arange(max_boxes, dtype=jnp.int32), p_id)
    return m_id, p_id, lvls, is_leaf, n_active


@_functools.lru_cache(maxsize=None)
def _vtree_jitted(f, has_theta, max_depth, min_level, ng, lo, hi, max_boxes):
    r"""Cache one compiled builder per (f, static params). ``f`` is a jax callable.

    If ``has_theta``, the compiled fn takes ``(tol, theta)`` and the density is
    ``f(Y, theta)`` with ``theta`` a TRACED array — so varying theta (e.g.
    moving the source) reuses the same executable with no recompile.
    """
    def core(tol, theta=None):
        ff = (lambda Y: f(Y, theta)) if has_theta else f
        a = _vtree_refine(ff, tol, max_depth, min_level, ng, lo, hi, max_boxes)
        b = _vtree_balance(*a, max_depth, min_level, max_boxes)
        return _vtree_emit(b[0], b[1], b[2], b[3], max_boxes, max_depth)
    return _jax.jit(core)


def build_volume_tree_jit(f, tol=1e-4, max_depth=5, min_level=2, n=5,
                          domain=(0.0, 1.0), max_boxes=None, theta=None):
    r"""JIT top-down volume octree builder (fixed-shape, on-GPU).

    Refines top-down by the density's interpolation error
    (Legendre-coefficient tail > ``tol * max|f|``), 2:1-balances via a JIT
    ripple, and emits FMM-format arrays of FIXED size ``max_boxes`` (padded
    with ``lvls == 0``), rooted at level 1 and laid out fine-to-coarse exactly
    like ``jaxfmm.trees.build_fmm_tree``. Because the output shape depends
    only on ``max_boxes``, the downstream KIFMM compiles once and is reused
    across geometries.

    Parameters
    ----------
    f : callable
        JAX-traceable density: ``f(Y)`` if ``theta is None``, else
        ``f(Y, theta)``; ``Y`` has shape ``(..., 3)`` → ``(...)``.
    tol : float
        Refine while the tail interpolation error exceeds ``tol * max|f|``.
    max_depth : int
        Maximum refinement level (static).
    min_level : int
        Starting uniform level (static).
    n : int
        Gauss nodes per axis for the error indicator (static).
    domain : tuple of float
        Cubic domain ``(lo, hi)`` (static).
    max_boxes : int, optional
        Fixed box-buffer size (static); default a generous bound.
    theta : array_like, optional
        If given, the density is ``f(Y, theta)`` and ``theta`` is passed as a
        TRACED argument — so varying theta (moving the source, time-stepping,
        parameter sweeps) reuses the SAME compiled builder with no recompile.
        With a plain ``f(Y)`` (``theta=None``), a different density object
        recompiles the builder (but never the downstream FMM).

    Returns
    -------
    dict
        Matches :func:`build_volume_tree` (``m_id``/``p_id``/``lvls``/
        ``is_leaf`` padded to ``max_boxes``, ``domain_extent``, ``pmin``,
        ``max_depth``, ``leaves``) plus ``"n_active"``.
    """
    import numpy as _np
    lo, hi = float(domain[0]), float(domain[1])
    if max_boxes is None:
        # generous: full coarse pyramid + the finest level it could reach
        max_boxes = sum(8 ** k for k in range(1, min_level + 1)) + 8 ** max_depth + 64
    fn = _vtree_jitted(f, theta is not None, int(max_depth), int(min_level),
                       int(n), lo, hi, int(max_boxes))
    if theta is None:
        m_id, p_id, lvls, is_leaf, n_active = fn(_jnp.asarray(float(tol)))
    else:
        m_id, p_id, lvls, is_leaf, n_active = fn(_jnp.asarray(float(tol)), _jnp.asarray(theta))
    m_id = _np.asarray(m_id); p_id = _np.asarray(p_id)
    lvls = _np.asarray(lvls); is_leaf = _np.asarray(is_leaf)
    return {"m_id": m_id, "p_id": p_id, "lvls": lvls, "is_leaf": is_leaf,
            "domain_extent": float(hi - lo), "pmin": _np.full(3, lo),
            "max_depth": int(max_depth), "n_active": int(n_active),
            "leaves": _np.where(is_leaf)[0].astype(_np.int32)}
