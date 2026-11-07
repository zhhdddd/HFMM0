r"""Reusable building blocks for differentiable shape (and magnetisation) optimisation.

A new problem only has to assemble a loss + constraints from these pieces:

  * regularisers       -- tet volume, surface area, umbrella-Laplacian (curvature) energy, mean-ratio
                          quality penalty, scale-invariant min-quality (validity).
  * mesh graphs        -- 1-ring neighbour tables from tet edges (volume) or boundary tris (surface).
  * Sobolev / H1 prec. -- Jacobi-smooth a per-vertex gradient over the volume graph so the whole mesh
                          deforms smoothly (spreads the surface-concentrated gradient; keeps quality).
  * constraints        -- fix volume (rescale about a centre) and recenter (pin centroid).
  * symmetry           -- dihedral-group operators + a Reynolds projector onto the symmetric subspace.
  * optimiser          -- :func:`lbfgs`: Sobolev-preconditioned L-BFGS (metric-correct initial inverse
                          Hessian = the smoother) with a trust-step backtracking line search and a
                          validity guard. Per-coordinate optimisers (e.g. Adam) discard the smoothed
                          gradient magnitudes and thereby the Sobolev metric; the preconditioned L-BFGS
                          adds curvature acceleration in the H1 metric while the trust step + validity
                          guard prevent mesh tangling.

All functions are pure JAX (differentiable, jit-able) except the eager graph/symmetry builders and the
``lbfgs`` driver loop (which calls a jitted ``value_and_grad``).
"""
import jax, jax.numpy as jnp, numpy as np

__all__ = ["tet_volume", "tet_signed_dets", "surface_area", "curvature_energy", "quality_penalty",
           "min_quality", "no_inversion", "vertex_neighbors", "surface_neighbors", "sobolev",
           "fix_volume", "recenter", "dihedral_ops", "build_symmetrizer", "lbfgs"]

_E6 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
_QREG = 6.0 * (6.0 * np.sqrt(2.0)) ** (2.0 / 3.0)                       # (sum L^2)/V^(2/3) of a regular tet


# ── regularisers / geometry ────────────────────────────────────────────────────────────────────────
def tet_signed_dets(verts, tets):
    r"""Signed edge-matrix determinant (six times the signed volume) of every tetrahedron.

    Parameters
    ----------
    verts : (N, 3) jnp.ndarray
        Vertex positions.
    tets : (T, 4) jnp.ndarray
        Tetrahedron vertex indices (int).

    Returns
    -------
    (T,) jnp.ndarray
        Signed determinant (= 6 x signed volume) per tet.
    """
    ev = verts[tets]; return jnp.linalg.det(ev[:, 1:] - ev[:, :1])      # 6 * signed volume per tet

def tet_volume(verts, tets):
    r"""Total unsigned volume of the tetrahedral mesh.

    Parameters
    ----------
    verts : (N, 3) jnp.ndarray
        Vertex positions.
    tets : (T, 4) jnp.ndarray
        Tetrahedron vertex indices (int).

    Returns
    -------
    jnp.ndarray
        Scalar total volume.
    """
    return jnp.abs(tet_signed_dets(verts, tets)).sum() / 6.0

def surface_area(verts, tris):
    r"""Total boundary area. Minimised at fixed volume by the sphere — a standard shape-opt regulariser.

    Parameters
    ----------
    verts : (N, 3) jnp.ndarray
        Vertex positions.
    tris : (F, 3) jnp.ndarray
        Boundary triangle vertex indices (int).

    Returns
    -------
    jnp.ndarray
        Scalar total area.
    """
    v = verts[tris]
    return 0.5 * jnp.linalg.norm(jnp.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum()

def curvature_energy(verts, nbr_idx, nbr_mask):
    r"""Umbrella-Laplacian (discrete mean-curvature) energy.

    .. math:: E = \sum_v \Big\| v - \tfrac{1}{|N(v)|}\sum_{w \in N(v)} w \Big\|^2

    over the vertices with neighbours in the table. Penalises bending (rounds sharp
    features / spikes).

    Parameters
    ----------
    verts : (N, 3) jnp.ndarray
        Vertex positions.
    nbr_idx : (N, D) jnp.ndarray
        Padded neighbour indices (int), e.g. from :func:`surface_neighbors`.
    nbr_mask : (N, D) jnp.ndarray
        1.0 where ``nbr_idx`` is a real neighbour, 0.0 for padding.

    Returns
    -------
    jnp.ndarray
        Scalar energy (vertices without neighbours contribute 0).
    """
    deg = jnp.maximum(nbr_mask.sum(1), 1.0)
    cen = (verts[nbr_idx] * nbr_mask[..., None]).sum(1) / deg[:, None]
    return (((verts - cen) * (nbr_mask.sum(1) > 0)[:, None]) ** 2).sum()

def quality_penalty(verts, tets, tau=3.0):
    r"""Mean-ratio-style differentiable quality penalty.

    Penalises only tets whose scale-invariant shape ratio :math:`(\sum L^2)/V^{2/3}`
    exceeds ``tau`` times that of a regular tetrahedron — lets the shape deform freely
    while blocking degeneration.

    Parameters
    ----------
    verts : (N, 3) jnp.ndarray
        Vertex positions.
    tets : (T, 4) jnp.ndarray
        Tetrahedron vertex indices (int).
    tau : float, optional
        Penalty threshold in units of the regular-tet shape ratio (default 3.0).

    Returns
    -------
    jnp.ndarray
        Scalar penalty (0 while every tet is within ``tau``).
    """
    ev = verts[tets]; e = ev[:, [p[0] for p in _E6], :] - ev[:, [p[1] for p in _E6], :]
    L2 = (e ** 2).sum((-1, -2)); V = jnp.abs(jnp.linalg.det(ev[:, 1:] - ev[:, :1])) / 6.0
    return (jax.nn.relu(L2 / (V ** (2.0 / 3.0) + 1e-12) / _QREG - tau) ** 2).sum()

def min_quality(verts, tets):
    r"""Scale-invariant worst-tet quality :math:`|\det| / \mathrm{rms\_edge}^3`.

    About 0.707 for a regular tet, tends to 0 for degenerate tets. Eager NumPy; use as
    a validity gate that survives mesh refinement (smaller tets keep the same quality).

    Parameters
    ----------
    verts : (N, 3) array_like
        Vertex positions.
    tets : (T, 4) array_like
        Tetrahedron vertex indices (int).

    Returns
    -------
    float
        Quality of the worst tet.
    """
    ev = np.asarray(verts)[np.asarray(tets)]
    det = np.abs(np.linalg.det(ev[:, 1:] - ev[:, :1]))
    e2 = sum(((ev[:, i] - ev[:, j]) ** 2).sum(-1) for i, j in _E6)
    return float((det / ((e2 / 6.0) ** 1.5 + 1e-30)).min())

def no_inversion(verts, tets, ref_sign):
    r"""True if no tet has flipped orientation relative to ``ref_sign`` (= sign(tet_signed_dets(verts0))).

    Parameters
    ----------
    verts : (N, 3) jnp.ndarray
        Vertex positions.
    tets : (T, 4) jnp.ndarray
        Tetrahedron vertex indices (int).
    ref_sign : (T,) array_like
        Reference orientation signs of the undeformed mesh.

    Returns
    -------
    bool
        True if every tet keeps its reference orientation (and none is degenerate).
    """
    d = np.asarray(tet_signed_dets(verts, tets))
    return bool(np.all(d * np.asarray(ref_sign) > 0))


# ── mesh 1-ring neighbour tables (eager) ────────────────────────────────────────────────────────────
def _pad_neighbors(nb, N):
    D = max((len(s) for s in nb), default=1); idx = np.zeros((N, D), int); msk = np.zeros((N, D))
    for v, s in enumerate(nb):
        for j, k in enumerate(s):
            idx[v, j] = k; msk[v, j] = 1.0
    return jnp.asarray(idx), jnp.asarray(msk)

def vertex_neighbors(tets, N):
    r"""Padded ``(idx, mask)`` 1-ring table over the volume mesh (tet edges). For the Sobolev preconditioner.

    Parameters
    ----------
    tets : (T, 4) array_like
        Tetrahedron vertex indices (int).
    N : int
        Number of vertices.

    Returns
    -------
    idx : (N, D) jnp.ndarray
        Padded neighbour indices (int); D = maximum 1-ring size.
    mask : (N, D) jnp.ndarray
        1.0 for real neighbours, 0.0 for padding.
    """
    nb = [set() for _ in range(N)]
    for t in np.asarray(tets):
        for a, b in _E6:
            nb[t[a]].add(int(t[b])); nb[t[b]].add(int(t[a]))
    return _pad_neighbors(nb, N)

def surface_neighbors(tris, N):
    r"""Padded ``(idx, mask)`` 1-ring table over the boundary (tris). For the curvature regulariser.

    Parameters
    ----------
    tris : (F, 3) array_like
        Boundary triangle vertex indices (int).
    N : int
        Number of vertices (interior vertices get empty rows).

    Returns
    -------
    idx : (N, D) jnp.ndarray
        Padded neighbour indices (int); D = maximum 1-ring size.
    mask : (N, D) jnp.ndarray
        1.0 for real neighbours, 0.0 for padding.
    """
    nb = [set() for _ in range(N)]
    for a, b, c in np.asarray(tris):
        nb[a] |= {int(b), int(c)}; nb[b] |= {int(a), int(c)}; nb[c] |= {int(a), int(b)}
    return _pad_neighbors(nb, N)


# ── Sobolev / H1 preconditioner ─────────────────────────────────────────────────────────────────────
@jax.jit
def sobolev(g, nbr_idx, nbr_mask, lam=0.5, passes=12):
    r"""Jacobi (umbrella) smoothing of a per-vertex field — an approximate Sobolev/H1 gradient.

    Damps the high-frequency (mesh-roughening) modes while keeping the smooth
    (large-scale deformation) modes, so a descent step deforms the whole mesh smoothly
    instead of just the surface where the gradient is large.

    Parameters
    ----------
    g : (N, 3) jnp.ndarray
        Per-vertex field (e.g. a gradient) to smooth.
    nbr_idx : (N, D) jnp.ndarray
        Padded neighbour indices (int), from :func:`vertex_neighbors`.
    nbr_mask : (N, D) jnp.ndarray
        1.0 for real neighbours, 0.0 for padding.
    lam : float, optional
        Jacobi blending weight per pass (default 0.5).
    passes : int, optional
        Number of smoothing passes (default 12).

    Returns
    -------
    (N, 3) jnp.ndarray
        The smoothed field.
    """
    deg = jnp.maximum(nbr_mask.sum(1), 1.0)[:, None]
    return jax.lax.fori_loop(0, passes, lambda _, g: (1 - lam) * g + lam * (g[nbr_idx] * nbr_mask[..., None]).sum(1) / deg, g)


# ── constraints ─────────────────────────────────────────────────────────────────────────────────────
def fix_volume(verts, tets, V0, center):
    r"""Rescale isotropically about ``center`` so the volume equals ``V0`` (shape-preserving scale fix).

    Parameters
    ----------
    verts : (N, 3) jnp.ndarray
        Vertex positions.
    tets : (T, 4) jnp.ndarray
        Tetrahedron vertex indices (int).
    V0 : float
        Target total volume.
    center : (3,) array_like
        Scaling centre.

    Returns
    -------
    (N, 3) jnp.ndarray
        The rescaled vertices.
    """
    return (verts - center) * (V0 / tet_volume(verts, tets)) ** (1.0 / 3.0) + center

def recenter(verts, center):
    r"""Translate so the vertex centroid is at ``center`` (the objectives here are translation-invariant).

    Parameters
    ----------
    verts : (N, 3) jnp.ndarray
        Vertex positions.
    center : (3,) array_like
        Target centroid.

    Returns
    -------
    (N, 3) jnp.ndarray
        The translated vertices.
    """
    return verts - verts.mean(0) + center


# ── symmetry (Reynolds projection onto the symmetric subspace) ──────────────────────────────────────
def dihedral_ops(n):
    r"""The 2n elements of the dihedral group :math:`D_n` about the z-axis (n rotations + n reflections), as 3x3 matrices.

    Parameters
    ----------
    n : int
        Order of the rotational symmetry.

    Returns
    -------
    list of jnp.ndarray
        The 2n orthogonal (3, 3) matrices.
    """
    R = []
    for k in range(n):
        a = 2 * np.pi * k / n; c, s = np.cos(a), np.sin(a)
        R.append(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]))        # rotation by 2pi k/n
        R.append(np.array([[c, s, 0], [s, -c, 0], [0, 0, 1.0]]))        # reflection
    return [jnp.asarray(r) for r in R]

def build_symmetrizer(verts0, ops, center=(0.0, 0.0, 0.0)):
    r"""Build a Reynolds projector onto the symmetric subspace.

    Returns ``symmetrize(field) -> field`` projecting a per-vertex (polar) vector field
    onto the subspace invariant under ``ops`` (a Reynolds average). ``ops`` are 3x3
    orthogonal matrices about ``center``; node images are matched geometrically on the
    reference mesh ``verts0`` (positions must be symmetric).

    Parameters
    ----------
    verts0 : (N, 3) array_like
        Reference vertex positions; must map onto themselves under every op
        (matched by rounding to 6 decimals).
    ops : sequence of array_like
        Orthogonal (3, 3) matrices, e.g. from :func:`dihedral_ops`.
    center : (3,) array_like, optional
        Symmetry centre the ops act about (default the origin).

    Returns
    -------
    callable
        ``symmetrize((N, 3) field) -> (N, 3) field``, the Reynolds projector.
    """
    V = np.asarray(verts0); c = np.asarray(center); key = {tuple(np.round(p, 6)): i for i, p in enumerate(V)}
    imgs = [jnp.asarray([key[tuple(np.round(p, 6))] for p in (V - c) @ np.asarray(R).T + c]) for R in ops]
    ops_j = [jnp.asarray(R) for R in ops]
    def symmetrize(f):
        return sum(f[img] @ R for R, img in zip(ops_j, imgs)) / len(ops_j)   # f[img]@R applies R^T per row
    return symmetrize


# ── optimiser: Sobolev-preconditioned L-BFGS + trust-step line search + validity guard ──────────────
def _dot(a, b):
    return (a * b).sum()

def lbfgs(value_and_grad, x0, precond=None, project=None, valid=None,
          maxiter=200, history=10, max_step=0.03, shrink=0.5, max_ls=25, ftol=1e-12, callback=None):
    r"""Minimise ``f(x)`` over a per-vertex array ``x`` with L-BFGS in a (Sobolev) metric.

    Parameters
    ----------
    value_and_grad : callable
        ``x -> (f, g)`` (e.g. ``jax.jit(jax.value_and_grad(loss))``).
    x0 : jnp.ndarray
        Initial per-vertex array.
    precond : callable, optional
        ``g -> H0 g`` applied as the L-BFGS initial inverse-Hessian (e.g.
        ``lambda g: sobolev(g, idx, msk)`` for the H1/Sobolev metric). Identity if
        ``None``.
    project : callable, optional
        ``x -> x`` enforcing constraints after each step (e.g. fix volume + recenter +
        symmetrise). Applied to ``x0`` and every trial.
    valid : callable, optional
        ``x -> bool`` rejecting invalid steps (e.g. :func:`no_inversion`); the line
        search backtracks until valid.
    maxiter : int, optional
        Maximum number of iterations.
    history : int, optional
        Number of L-BFGS curvature pairs kept.
    max_step : float, optional
        Trust radius — the search direction is scaled so the largest vertex move is
        ``max_step`` before backtracking (controlled geometric steps avoid tangling).
    shrink : float, optional
        Backtracking factor of the line search.
    max_ls : int, optional
        Maximum number of backtracking steps.
    ftol : float, optional
        Relative decrease below which the iteration stops.
    callback : callable, optional
        ``callback(it, x, f)`` invoked after every accepted step.

    Returns
    -------
    x : jnp.ndarray
        The minimiser found.
    hist : dict
        Per-iteration objective values in ``hist['f']``.

    Notes
    -----
    The first step is Sobolev-preconditioned steepest descent; subsequent steps add
    quasi-Newton curvature (in the H1 metric), accelerating past the plateaus where
    plain line-search gradient descent stalls.
    """
    pre = precond if precond is not None else (lambda g: g)
    prj = project if project is not None else (lambda x: x)
    val = valid if valid is not None else (lambda x: True)
    x = prj(x0); f, g = value_and_grad(x)
    S, Y, rho = [], [], []; hist = {"f": [float(f)]}
    for it in range(maxiter):
        gamma = (_dot(S[-1], Y[-1]) / (_dot(Y[-1], Y[-1]) + 1e-30)) if S else 1.0   # Barzilai-Borwein H0 scale
        q = g; alphas = []                                             # L-BFGS two-loop recursion (H0 = gamma * pre)
        for s, y, r in zip(reversed(S), reversed(Y), reversed(rho)):
            a = r * _dot(s, q); q = q - a * y; alphas.append(a)
        d = gamma * pre(q)
        for s, y, r, a in zip(S, Y, rho, reversed(alphas)):
            b = r * _dot(y, d); d = d + (a - b) * s
        d = -d                                                         # search direction = -H_k g (natural scale)
        if _dot(g, d) >= 0:                                            # safeguard: fall back to -precond(g)
            d = -pre(g)
        nrm = float(jnp.max(jnp.linalg.norm(d, axis=1)))               # trust radius: CAP the max vertex move at
        if nrm > max_step:                                             #   max_step (but keep smaller natural steps)
            d = d * (max_step / nrm)
        step = 1.0; xn = None                                          # try the (capped) L-BFGS step, then backtrack
        for _ in range(max_ls):
            trial = prj(x + step * d); fn, gn = value_and_grad(trial)
            if float(fn) < float(f) and val(trial):
                xn = trial; break
            step *= shrink
        if xn is None:
            break                                                      # no valid descent step -> converged/stuck
        s_vec = xn - x; y_vec = gn - g; sy = float(_dot(s_vec, y_vec))
        if sy > 1e-12:                                                 # curvature condition -> update history
            S.append(s_vec); Y.append(y_vec); rho.append(1.0 / sy)
            if len(S) > history:
                S.pop(0); Y.pop(0); rho.pop(0)
        prev = float(f); x, f, g = xn, fn, gn; hist["f"].append(float(f))
        if callback is not None:
            callback(it, x, float(f))
        if prev - float(f) < ftol * max(1.0, abs(prev)):               # negligible progress -> stop
            break
    return x, hist
