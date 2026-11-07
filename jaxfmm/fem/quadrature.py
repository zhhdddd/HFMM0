r"""Simplex quadrature for finite-element sources: collapsed Gauss-Jacobi rules on the
reference triangle/tetrahedron, Duffy rules for the :math:`1/r` corner singularity, and
helpers mapping barycentric rules to physical quadrature points and point-charges."""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import roots_jacobi

__all__ = ["simplex_quad", "element_quad_points", "element_quad_charges", "duffy_simplex"]

def _gj01(n, b):
    r"""
    Gauss-Jacobi rule on the unit interval with weight :math:`(1-\eta)^b`.

    The ``n``-point rule for :math:`\int_0^1 g(\eta)\,(1-\eta)^b\,d\eta`, exact for
    polynomials ``g`` of degree ``2n-1``.

    Parameters
    ----------
    n : int
        Number of quadrature points.
    b : int
        Jacobi exponent of the weight :math:`(1-\eta)^b`.

    Returns
    -------
    nodes : ndarray, shape (n,)
        Quadrature nodes in :math:`[0, 1]`.
    weights : ndarray, shape (n,)
        Quadrature weights (the :math:`(1-\eta)^b` factor is absorbed).
    """
    x, w = roots_jacobi(n, b, 0)                 # weight (1-x)^b (1+x)^0 on [-1,1]
    return (x + 1) / 2, w * (0.5 ** (b + 1))     # map to [0,1]; (1-eta)^b = ((1-x)/2)^b

def simplex_quad(dim, degree):
    r"""
    Collapsed (Duffy) Gauss-Jacobi quadrature on the reference ``dim``-simplex.

    Exact for polynomials up to total degree ``degree``. The collapsed coordinates map
    the cube :math:`[0,1]^{dim}` onto the simplex; the Jacobian factors
    :math:`(1-\eta_i)^{dim-i}` are absorbed exactly by Gauss-Jacobi weights, so only
    :math:`\lceil(\text{degree}+1)/2\rceil` points per direction are needed (positive
    weights throughout).

    Parameters
    ----------
    dim : int
        Simplex dimension: ``3`` for the tetrahedron, ``2`` for the triangle.
    degree : int
        Highest total polynomial degree integrated exactly.

    Returns
    -------
    bary : ndarray, shape (n_q, dim+1)
        Barycentric coordinates of the quadrature points.
    w : ndarray, shape (n_q,)
        Quadrature weights, normalised to sum to 1, so that

        .. math:: \int_{elem} f\,dV = \text{measure}\cdot\sum_q w_q\,f(x_q).

    Notes
    -----
    Returns float64 host (NumPy) constants; see :func:`element_quad_points` /
    :func:`element_quad_charges` for the mapping to physical elements.
    """
    from jaxfmm.fem.quad_tables import lookup
    tab = lookup(dim, degree)                    # tabulated symmetric rule, when it beats
    if tab is not None:                          # the collapsed count (positive weights,
        return tab[0].copy(), tab[1].copy()      # exactness-verified; see quad_tables)
    n = degree // 2 + 1                          # ceil((degree+1)/2): exact to 2n-1 >= degree
    if dim == 3:
        e1, W1 = _gj01(n, 2); e2, W2 = _gj01(n, 1); e3, W3 = _gj01(n, 0)
        A, B, C = [g.ravel() for g in np.meshgrid(e1, e2, e3, indexing="ij")]
        WW = (W1[:, None, None] * W2[None, :, None] * W3[None, None, :]).ravel()
        l1, l2, l3 = A, B * (1 - A), C * (1 - A) * (1 - B)
        bary = np.stack([1 - l1 - l2 - l3, l1, l2, l3], axis=1)
    elif dim == 2:
        e1, W1 = _gj01(n, 1); e2, W2 = _gj01(n, 0)
        A, B = [g.ravel() for g in np.meshgrid(e1, e2, indexing="ij")]
        WW = (W1[:, None] * W2[None, :]).ravel()
        l1, l2 = A, B * (1 - A)
        bary = np.stack([1 - l1 - l2, l1, l2], axis=1)
    else:
        raise ValueError("dim must be 2 (triangle) or 3 (tetrahedron)")
    return bary, WW / WW.sum()       # float64 host constants (weights sum to 1); see element_quad_*

def duffy_simplex(dim, degree):
    r"""
    Duffy quadrature for the :math:`1/r` corner singularity on the reference ``dim``-simplex.

    The singular vertex sits at **barycentric index 1**. The collapsed-coordinate Jacobian
    cancels the singularity exactly (one Gauss-Jacobi weight power :math:`(1-\eta_1)` per
    consumed power of :math:`1/r`), so a smooth integrand remains and convergence in
    ``degree`` is rapid.

    Parameters
    ----------
    dim : int
        Simplex dimension: ``3`` for the tetrahedron, ``2`` for the triangle.
    degree : int
        Rule degree (points per direction :math:`\lceil(\text{degree}+1)/2\rceil`).

    Returns
    -------
    bary : ndarray, shape (n_q, dim+1)
        Barycentric quadrature points.
    w : ndarray, shape (n_q,)
        Duffy weights.
    om1 : ndarray, shape (n_q,)
        The factor :math:`1-\eta_1` per point.

    Notes
    -----
    For an element whose vertices are ordered with the singular node ``V`` at index 1
    (``x_q = bary_q @ verts``):

    .. math:: \int_{elem} \frac{f(x)}{|x-V|}\,dW = \text{geomfac}\sum_q w_q\,\frac{f(x_q)}{\rho_q},

    with :math:`\rho_q=|x_q-V|/\text{om1}_q` and geomfac :math:`= 6\,\text{Vol}` (tet)
    or :math:`2\,\text{Area}` (tri).
    """
    n = degree // 2 + 1
    if dim == 3:                                     # tet: one (1-eta1) consumed by 1/r -> beta=(1,1,0)
        e1, W1 = _gj01(n, 1); e2, W2 = _gj01(n, 1); e3, W3 = _gj01(n, 0)
        A, B, C = [g.ravel() for g in np.meshgrid(e1, e2, e3, indexing="ij")]
        W = (W1[:, None, None] * W2[None, :, None] * W3[None, None, :]).ravel()
        l1, l2, l3 = A, B * (1 - A), C * (1 - A) * (1 - B)
        bary = np.stack([1 - l1 - l2 - l3, l1, l2, l3], axis=1)
    elif dim == 2:                                   # tri: the (1-eta1) Jacobian fully cancels 1/r -> Legendre
        e1, W1 = _gj01(n, 0); e2, W2 = _gj01(n, 0)
        A, B = [g.ravel() for g in np.meshgrid(e1, e2, indexing="ij")]
        W = (W1[:, None] * W2[None, :]).ravel()
        l1, l2 = A, B * (1 - A)
        bary = np.stack([1 - l1 - l2, l1, l2], axis=1)
    else:
        raise ValueError("dim must be 2 (triangle) or 3 (tetrahedron)")
    return bary, W, 1 - A

def element_quad_points(verts, elems, bary):
    r"""
    Physical quadrature points and measure (volume/area) per element.

    Parameters
    ----------
    verts : jnp.ndarray, shape (N_vert, 3)
        Vertex coordinates.
    elems : jnp.ndarray, shape (N_elem, k)
        Vertex indices per element (``k=4`` tets, ``k=3`` tris).
    bary : ndarray, shape (n_q, k)
        Barycentric quadrature points (from :func:`simplex_quad`).

    Returns
    -------
    qpos : jnp.ndarray, shape (N_elem, n_q, 3)
        Physical quadrature points.
    measure : jnp.ndarray, shape (N_elem,)
        Element volume (tet) or area (tri).
    """
    ev = verts[elems]                                        # (N_elem, k, 3)
    return _bary_points(jnp.asarray(bary, ev.dtype), ev), element_measure(ev)


@jax.jit
def _bary_points(bary, ev):
    r"""``sum_k bary[q, k] ev[e, k]`` -> ``(N_elem, n_q, 3)`` as one fused elementwise kernel.

    Not an einsum: as a dot_general it ran under TF32 on GPU unless pinned (points ~1e-3 off in
    f32) and its gemm temporary doubled the setup peak at production sizes (R9 at 1e7 tets:
    16 GB output + 26 GB temporary, OOM on an H100).

    Unrolled in a FIXED order over k: as a ``.sum(axis=2)`` reduce its rounding depended on
    the fusion context, so the points inside a whole-forward jit differed by ulps from the
    eager ones and moved them across balanced-tree splits (guarded fulljit driver vs a
    fresh setup: 2.6e-5 apart on an A100, test_fulljit_strayfield)."""
    out = bary[None, :, 0, None] * ev[:, None, 0, :]
    for k in range(1, ev.shape[1]):
        out = out + bary[None, :, k, None] * ev[:, None, k, :]
    return out


def element_measure(ev):
    r"""Volume (``ev`` of shape ``(N, 4, 3)``) or area (``(N, 3, 3)``) of each simplex."""
    if ev.shape[1] == 4:                                     # tetrahedron volume
        return jnp.abs(jnp.linalg.det(ev[:, 1:] - ev[:, :1])) / 6
    return jnp.linalg.norm(jnp.cross(ev[:, 1] - ev[:, 0], ev[:, 2] - ev[:, 0]), axis=-1) / 2

def element_quad_charges(verts, elems, density, bary, w):
    r"""
    Quadrature point-charges for a FEM charge density integrated against the kernel.

    The element charge :math:`\int_{elem}\rho\,(\cdot)\,dV` becomes a set of point
    charges :math:`q_q=\rho(x_q)\,w_q\,\text{measure}` at the physical quadrature points
    -- volume/surface and constant/linear densities are all handled the same way once
    reduced to point charges.

    Parameters
    ----------
    verts : jnp.ndarray, shape (N_vert, 3)
        Vertex coordinates.
    elems : jnp.ndarray, shape (N_elem, k)
        Vertex indices per element (``k=4`` tets, ``k=3`` tris).
    density : jnp.ndarray
        Per-element charge density. Shape (N_elem,) for a **P0** density (constant per
        element, e.g. volume :math:`\rho=-\nabla\!\cdot M`), or (N_elem, k) for a **P1**
        nodal density (e.g. surface :math:`\sigma=M\cdot n`).
    bary : ndarray, shape (n_q, k)
        Barycentric quadrature points from :func:`simplex_quad`.
    w : ndarray, shape (n_q,)
        Quadrature weights from :func:`simplex_quad`.

    Returns
    -------
    qpos : jnp.ndarray, shape (N_elem, n_q, 3)
        Physical quadrature points.
    qchg : jnp.ndarray, shape (N_elem, n_q)
        Point charges.
    """
    qpos, measure = element_quad_points(verts, elems, bary)
    if density.ndim == 1:                                    # P0: constant per element
        rho_q = density[:, None]                             # (N_elem, 1) broadcast over n_q
    else:                                                    # P1: interpolate nodal values
        rho_q = jnp.einsum("qk,ek->eq", bary, density)       # (N_elem, n_q)
    qchg = rho_q * w[None, :] * measure[:, None]
    return qpos, qchg
