r"""Real solid-harmonic basis functions for the Laplace kernel.

Regular (interior) and singular (exterior) solid harmonics evaluated with stable
recursion relations, plus the flat index mapping used for the coefficient
vectors: a coefficient of order :math:`m` and degree :math:`n` lives at flat
position

.. math:: \mathrm{idx}(m, n) = n^2 + (m + n), \qquad -n \le m \le n,

so all coefficients up to degree :math:`p` fit into a vector of length
:math:`(p+1)^2`.

Normalisation (the "sqrt(2) real basis"): with :math:`R^m_n`, :math:`S^m_n` the complex
solid harmonics, :math:`\hat R^0_n = R^0_n` and, for :math:`m > 0`,

.. math::

    \hat R^{\pm m}_n = \sqrt2\,\mathrm{Re}/\mathrm{Im}\,R^m_n, \qquad
    \hat S^{\pm m}_n = \sqrt2\,\mathrm{Re}/\mathrm{Im}\,S^m_n,

so the addition theorem is a PLAIN sum, :math:`1/|x-y| = \sum_{n,m} \hat R^m_n(x)\,
\hat S^m_n(y)` for :math:`|x| < |y|` — multipole/local coefficients pair with the bases
without contraction weights, and every operator is self-adjoint in the plain Euclidean
inner product (the k=0 lattice operator is transpose-symmetric). Until 2026-10 the code
used :math:`\hat R^{\pm m} = \mathrm{Re}/\mathrm{Im}\,\bar R^m` without the sqrt(2), which
needed the weights :math:`w_m = (2 - \delta_{m0})\,\mathrm{sign}(m)` at every pairing.
"""
import math
import jax
import jax.numpy as jnp
from functools import lru_cache, partial

import numpy as np

_SQRT2 = math.sqrt(2.0)

def mpl_idx(m, n):
    r"""Flat array position of the multipole coefficient :math:`C^m_n`.

    Parameters
    ----------
    m : int or array_like
        Order, :math:`-n \le m \le n`.
    n : int or array_like
        Degree, :math:`n \ge 0`.

    Returns
    -------
    int or ndarray
        Flat index :math:`n^2 + m + n`.
    """
    return n**2 + (m+n)

def inv_mpl_idx(idx):
    r"""Order and degree of a flat array position (inverse of :func:`mpl_idx`).

    Parameters
    ----------
    idx : array_like
        Flat coefficient index.

    Returns
    -------
    m : jnp.ndarray
        Order.
    n : jnp.ndarray
        Degree.
    """
    n = jnp.int32(jnp.floor(jnp.sqrt(idx)+1e-4))
    m = idx - n*(n+1)
    return m, n

class _Cols:
    r"""Basis coefficients keyed by ``(m, n)``, built one column at a time and stacked once.

    The recursions must not fill a ``(..., (p+1)**2)`` array with a chain of ``.at[..., k].set``:
    XLA (jax 0.10.2, sm_120) fuses that chain into one in-place dynamic-update-slice kernel which
    faults with CUDA_ERROR_ILLEGAL_ADDRESS for some ``lax.map`` batch shapes (P2M of a 9.6e6-tet
    f64 mesh, ARCHITECTURE §7). Each column is stored in the default float dtype, exactly as the
    array writes cast it: same arithmetic, results within 1 ulp of the array form (fusion changes
    which products XLA contracts to FMA).
    """
    def __init__(self, shape):
        self.dt, self.shape, self.c = jnp.result_type(float), shape, {}

    def __setitem__(self, mn, v):
        self.c[mpl_idx(*mn)] = jnp.broadcast_to(jnp.asarray(v, self.dt), self.shape)

    def __getitem__(self, mn):
        return self.c[mpl_idx(*mn)]

    def stack(self):
        return jnp.stack([self.c[k] for k in range(len(self.c))], axis=-1)

@partial(jax.jit, static_argnames=['p'])
def eval_regular_basis(rvec, p):
    r"""Evaluate the real regular (interior) solid-harmonic basis.

    Uses the recursion relations of Gumerov & Duraiswami [1]_: the diagonal
    ("extreme") values are built from :math:`x, y`, their neighbours by one
    :math:`z` step, and the interior via the degree-lowering three-term
    recursion

    .. math::

        R^m_n = \frac{(2n-1)\, z\, R^m_{n-1} - r^2 R^m_{n-2}}
                     {(n - |m|)(n + |m|)}.

    Parameters
    ----------
    rvec : jnp.ndarray
        Displacement vectors, shape ``(..., 3)``.
    p : int
        Maximum expansion degree (static).

    Returns
    -------
    jnp.ndarray
        Basis values, shape ``(..., (p+1)**2)``, flat-indexed by
        :func:`mpl_idx`.

    References
    ----------
    .. [1] Gumerov, N. A. & Duraiswami, R., "Fast multipole methods on graphics
       processors", J. Comput. Phys. 227, 8290 (2008).
    """
    x, y, z = rvec[...,0], rvec[...,1], rvec[...,2]
    r2 = (rvec**2).sum(axis=-1)
    c = _Cols(rvec.shape[:-1])
    c[0,0] = 1.0

    if(p>0):
        c[-1,1] = y/_SQRT2
        c[0,1] = z
        c[1,1] = x/_SQRT2
    for n in range(2,p+1):    # first/second recursion: extreme values and their neighbors
        c[-n,n] = (x*c[1-n,n-1] + y*c[n-1,n-1])/(2*n)
        c[1-n,n] = z*c[1-n,n-1]
        for m in range(2-n,n-1):   # third recursion: all values inbetween
            c[m,n] = ((2*n-1)*z*c[m,n-1] - r2*c[m,n-2])/((n-abs(m))*(n+abs(m)))
        c[n-1,n] = z*c[n-1,n-1]
        c[n,n] = (x*c[n-1,n-1] - y*c[1-n,n-1])/(2*n)
    return c.stack()

@lru_cache(maxsize=None)
def regular_grad_matrix(p):
    r"""Closed-form gradient of the regular basis: :math:`\nabla \hat R_n = D\,\hat R_{n-1}`.

    In the sqrt(2) real basis (terms with :math:`|m| > n-1` vanish),

    .. math::

        \partial_z \hat R^m_n = \hat R^m_{n-1}, \qquad
        \partial_x \hat R^{\pm m}_n = \tfrac12\big(\hat R^{\pm(m-1)}_{n-1}
            - \hat R^{\pm(m+1)}_{n-1}\big), \qquad
        \partial_y \hat R^{\pm m}_n = \mp\tfrac12\big(\hat R^{\mp(m-1)}_{n-1}
            + \hat R^{\mp(m+1)}_{n-1}\big)

    for :math:`m \ge 2`; the couplings to :math:`m = 0` carry :math:`1/\sqrt2` instead of
    :math:`1/2` (:math:`\partial_x \hat R^0_n = -\hat R^1_{n-1}/\sqrt2`,
    :math:`\partial_y \hat R^0_n = -\hat R^{-1}_{n-1}/\sqrt2`,
    :math:`\partial_x \hat R^1_n \ni \hat R^0_{n-1}/\sqrt2`,
    :math:`\partial_y \hat R^{-1}_n \ni \hat R^0_{n-1}/\sqrt2`), and the
    :math:`\hat R^{\pm(m-1)}` terms that would need :math:`\mathrm{Im}\,R^0` are absent. Consequence: the field of a degree-``p`` local expansion is a
    degree-``p-1`` local expansion with coefficients :math:`D^\mathsf{T} L` — the L2P field
    needs only :math:`\hat R` up to ``p-1`` per target.

    Parameters
    ----------
    p : int
        Degree of the differentiated basis.

    Returns
    -------
    np.ndarray
        ``D`` of shape ``(3, p**2, (p+1)**2)``: ``D[c, idx(k, n-1), idx(m, n)]`` is the
        weight of :math:`\hat R^k_{n-1}` in :math:`\partial_c \hat R^m_n`.
    """
    D = np.zeros((3, p * p, (p + 1) ** 2))
    h, r = 0.5, 1 / _SQRT2

    def put(c, m, n, k, w):
        if abs(k) <= n - 1:
            D[c, mpl_idx(k, n - 1), mpl_idx(m, n)] += w

    for n in range(1, p + 1):
        put(2, 0, n, 0, 1.0)
        put(0, 0, n, 1, -r)
        put(1, 0, n, -1, -r)
        for m in range(1, n + 1):
            lo = r if m == 1 else h                  # weight of the m-1 neighbour
            put(2, m, n, m, 1.0)
            put(2, -m, n, -m, 1.0)
            put(0, m, n, m - 1, lo)
            put(0, m, n, m + 1, -h)
            put(1, -m, n, m - 1, lo)
            put(1, -m, n, m + 1, h)
            if m > 1:
                put(0, -m, n, -(m - 1), h)
                put(1, m, n, -(m - 1), -h)
            put(0, -m, n, -(m + 1), -h)
            put(1, m, n, -(m + 1), -h)
    return D


@partial(jax.jit, static_argnames=['p'])
def eval_regular_basis_grad(rvec, p):
    r"""Evaluate the Cartesian gradient of the real regular solid-harmonic basis.

    Closed form (:func:`regular_grad_matrix`): :math:`\nabla \hat R_n` is a fixed sparse
    combination of :math:`\hat R_{n-1}`.

    Parameters
    ----------
    rvec : jnp.ndarray
        Displacement vectors, shape ``(..., 3)``.
    p : int
        Maximum expansion degree (static).

    Returns
    -------
    jnp.ndarray
        Gradients :math:`\nabla R^m_n`, shape ``(..., (p+1)**2, 3)``.
    """
    if p == 0:
        return jnp.zeros((*rvec.shape[:-1], 1, 3), rvec.dtype)
    D = jnp.asarray(regular_grad_matrix(p), rvec.dtype)
    return jnp.einsum("cjm,...j->...mc", D, eval_regular_basis(rvec, p - 1))


@partial(jax.jit, static_argnames=['p'])
def eval_singular_basis(rvec, p):
    r"""Evaluate the real singular (exterior) solid-harmonic basis.

    Companion of :func:`eval_regular_basis` for the exterior expansion, seeded
    with :math:`S^0_0 = 1/r` and built with the degree-raising recursion

    .. math::

        S^m_n = \frac{(2n-1)\, z\, S^m_{n-1} - (n-1-m)(n-1+m)\, S^m_{n-2}}{r^2}.

    Parameters
    ----------
    rvec : jnp.ndarray
        Displacement vectors, shape ``(..., 3)``; must be nonzero (the basis is
        singular at the origin).
    p : int
        Maximum expansion degree (static).

    Returns
    -------
    jnp.ndarray
        Basis values, shape ``(..., (p+1)**2)``, flat-indexed by
        :func:`mpl_idx`.
    """
    x, y, z = rvec[...,0], rvec[...,1], rvec[...,2]
    r2 = (rvec**2).sum(axis=-1)
    c = _Cols(rvec.shape[:-1])
    c[0,0] = 1/jnp.sqrt(r2)

    if(p>0):
        c[1,1] = _SQRT2*c[0,0]*x/r2
        c[0,1] = c[0,0]*z/r2
        c[-1,1] = _SQRT2*c[0,0]*y/r2
    for n in range(2,p+1): # first/second recursion: extreme values and their neighbors
        c[n,n] = (2*n-1)*(x*c[n-1,n-1] - y*c[1-n,n-1])/r2
        c[n-1,n] = (2*n-1)*z*c[n-1,n-1]/r2
        for m in range(2-n,n-1): # third recursion: all values inbetween
            c[m,n] = ((2*n-1)*z*c[m,n-1] - (n-1-m)*(n-1+m)*c[m,n-2])/r2
        c[1-n,n] = (2*n-1)*z*c[1-n,n-1]/r2
        c[-n,n] = (2*n-1)*(x*c[1-n,n-1] + y*c[n-1,n-1])/r2
    return c.stack()
