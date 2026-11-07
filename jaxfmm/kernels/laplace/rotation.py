r"""Wigner rotations of real solid-harmonic expansions.

The "point-and-shoot" translation operators (:mod:`jaxfmm.kernels.laplace.transf`)
rotate each expansion so the translation axis becomes :math:`z`, shift along
:math:`z`, and rotate back. A rotation splits into an azimuthal part (diagonal in
the order :math:`m`, :func:`rot_azimuth`) and a polar part (block-diagonal over
degrees :math:`j`, :func:`rot_polar`) whose blocks are Wigner-d matrices built
from Jacobi polynomials:

.. math::

    d^j_{mk}(\theta) = \epsilon_{mk}
        \sqrt{\frac{s!\,(s+\mu+\nu)!}{(s+\mu)!\,(s+\nu)!}}\,
        \sin^{\mu}\!\tfrac{\theta}{2}\, \cos^{\nu}\!\tfrac{\theta}{2}\,
        P^{(\mu,\nu)}_s(\cos\theta),

with :math:`\mu = |k-m|`, :math:`\nu = |k+m|` and :math:`s = j - (\mu+\nu)/2`.
Many :math:`(s, \mu, \nu)` combinations repeat across matrix entries, so only the
unique ones are evaluated (:func:`precompute_combs`).
"""
import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import factorial as fac
import math
from functools import lru_cache
import numpy as np
from jaxfmm.kernels.laplace.basis import mpl_idx, inv_mpl_idx

def precompute_combs(p):
    r"""Precompute the unique Jacobi-polynomial parameter combinations of all
    Wigner rotation matrices up to expansion order ``p``.

    Parameters
    ----------
    p : int
        Maximum expansion order.

    Returns
    -------
    unqs : list of jnp.ndarray
        Per degree ``j``, the unique ``(s, mu, nu)`` combinations, shape
        ``(3, n_unique_j)``.
    invs : list of jnp.ndarray
        Per degree ``j``, the index of each matrix entry into ``unqs[j]``,
        shape ``(j+1, 2j+1)`` (rows restricted to ``m >= 0``; the negative
        orders follow by symmetry).
    """
    unqs, invs, unq_num = [], [], [0]

    for j in range(p+1):
        rng = jnp.arange(-j,j+1)    # m and k
        mu = jnp.abs(rng[:,None]-rng[None,:]).flatten()
        nu = jnp.abs(rng[:,None]+rng[None,:]).flatten()
        s = j - (mu+nu)//2
        comb = jnp.stack((s,mu,nu),axis=0)
        unq, inv = jnp.unique(comb,axis=1,return_inverse=True)
        unqs.append(unq)
        invs.append(inv)
        unq_num.append(unq_num[j] + unq.shape[1])

    for j in range(p+1):
        invs[j] = invs[j].reshape((2*j+1,2*j+1))[j:,:]
    return unqs, invs

def cut(i, p):
    r"""Number of Jacobi parameter combinations with ``s == i`` at order ``p``.

    Parameters
    ----------
    i : int
        Jacobi polynomial degree.
    p : int
        Expansion order.

    Returns
    -------
    int
        ``2*(p-i)+1``.
    """
    return 2*(p-i)+1

def cut_cum(i, p):
    r"""Number of Jacobi parameter combinations with ``s <= i`` at order ``p``.

    Parameters
    ----------
    i : int
        Jacobi polynomial degree.
    p : int
        Expansion order.

    Returns
    -------
    int
        ``(p+1)**2 - (p-i)**2``.
    """
    return (p+1)**2 - (p-i)**2

@partial(jax.jit, static_argnames=['j'])
def fast_jacobi(j, mu, nu, x):
    r"""Evaluate Jacobi polynomials :math:`P^{(\mu,\nu)}_s(x)` by recursion.

    Evaluates all unique combinations for one Wigner degree in a single sweep of
    the standard three-term recurrence, ordered so that combinations with the
    same ``s`` are contiguous (see :func:`cut` / :func:`cut_cum`). Adapted from
    the recursive scheme in https://doi.org/10.1103/PhysRevC.106.054320.

    Parameters
    ----------
    j : int
        Wigner degree (static); the maximum Jacobi degree that occurs.
    mu, nu : jnp.ndarray
        Jacobi parameters per combination, sorted by ``s``.
    x : jnp.ndarray
        Evaluation points (:math:`\cos\theta`), any batch shape.

    Returns
    -------
    jnp.ndarray
        Polynomial values, shape ``(*x.shape, mu.shape[0])``.
    """
    res = jnp.ones((*x.shape,mu.shape[0])) # res[...,:cut_cum(0,j)] = 1
    mu = mu[None,...]
    nu = nu[None,...]
    x = x[...,None]

    if(j>0):
        mu = mu[...,cut(0,j):]
        nu = nu[...,cut(0,j):]
        res1 = res[...,cut_cum(0,j):]

        res = res.at[...,cut_cum(0,j):].set(0.5*(2*(mu+1)+(mu+nu+2)*(x-1)))

    for i in range(2,j+1):
        mu = mu[...,cut(i-1,j):]
        nu = nu[...,cut(i-1,j):]
        res2 = res1[...,cut(i-1,j):]
        res1 = res[...,cut_cum(i-1,j):]

        xn = i - 1
        two_nab = 2*xn + mu + nu
        a1 = 2*(xn+1) * (xn+mu+nu+1) * two_nab
        a2 = (two_nab+1) * (mu**2 - nu**2)
        a3 = two_nab * (two_nab+1) * (two_nab+2)
        a4 = 2*(xn+mu) * (xn+nu) * (two_nab+2)
        res = res.at[...,cut_cum(i-1,j):].set(((a2+a3*x)*res1 - a4*res2)/a1)
    return res

def factorialfac(s, mu, nu):
    r"""Wigner-d normalisation factor :math:`\sqrt{s!\,(s+\mu+\nu)! / ((s+\mu)!\,(s+\nu)!)}`.

    Parameters
    ----------
    s : jnp.ndarray
        Jacobi polynomial degrees (integer-valued).
    mu : jnp.ndarray
        Jacobi parameters :math:`\mu = |k-m|`, same shape as ``s``.
    nu : jnp.ndarray
        Jacobi parameters :math:`\nu = |k+m|`, same shape as ``s``.

    Returns
    -------
    jnp.ndarray
        The normalisation factors, same shape as the broadcast inputs.
    """
    return jnp.sqrt(fac(s)*fac(s+mu+nu) / (fac(s+mu)*fac(s+nu)))

@partial(jax.jit, static_argnames=['p'])
def get_polar_rot_coeff(p, unqs, theta):
    r"""Compute all unique Wigner-d coefficients for degree ``p``.

    Combines the Jacobi values with the normalisation and the
    :math:`\sin^{\mu}(\theta/2)\cos^{\nu}(\theta/2)` prefactors (see the module
    docstring for the full identity).

    Parameters
    ----------
    p : int
        Wigner degree (static).
    unqs : jnp.ndarray
        Unique ``(s, mu, nu)`` combinations for this degree (from
        :func:`precompute_combs`).
    theta : jnp.ndarray
        Polar rotation angles, any batch shape.

    Returns
    -------
    jnp.ndarray
        Unique coefficients, shape ``(*theta.shape, n_unique)``.
    """
    basemat = fast_jacobi(p, unqs[1], unqs[2], jnp.cos(theta))
    basemat *= factorialfac(unqs[0], unqs[1], unqs[2])[None,...] * \
               jnp.sin(theta/2)[...,None]**unqs[1][None,...] * \
               jnp.cos(theta/2)[...,None]**unqs[2][None,...]
    return basemat

@lru_cache(maxsize=None)
def _polar_fact(j, inv, loc):
    r"""Constant prefactor of the degree-``j`` Wigner block, built ONCE on the host.

    Shape ``(j+1, 2j+1)``: rows are the target orders ``m = 0..j``, columns the source
    orders ``k = -j..j`` (a ``k < 0`` column lands in block column ``|k|``). It combines the
    recursion's missing sign :math:`(-1)^{k-m}` for :math:`m < k`, the real-basis
    normalisation :math:`\sqrt{(j-m)!(j+m)!/((j-k)!(j+k)!)}^{\pm1}` (multipole/local), the
    inverse-rotation sign :math:`(\pm1)^{m+k}`, the ``k < 0`` sign flips, and the sqrt(2)
    basis rescaling: the ``m >= 0`` block transforms as ``c_m / c_k`` (``c_0 = 1``,
    ``c_{m>0} = sqrt(2)``), which touches only row ``m = 0`` and column ``k = 0`` — entries
    that feed nothing else. ``inv``/``loc`` must be static: built from traced flags, the
    prefactor was a per-call runtime computation in every M2M/L2L/M2L (measured: the sqrt(2)
    factor alone as a runtime multiply cost M2M/L2L +11-13% at p=3 on H100).
    """
    m = np.arange(j + 1)[:, None]
    k = np.arange(-j, j + 1)[None, :]
    f = np.vectorize(lambda n: float(math.factorial(n)))      # float: int64 overflows past p~10
    fact = ((-1.0) ** ((k - m) * (m < k))
            * np.sqrt(f(j - m) * f(j + m) / (f(j - k) * f(j + k))) ** (2 * loc - 1)
            * (2 * inv - 1.0) ** (k + m))
    fact[:, :j] *= (-1.0) ** np.arange(j, 2 * j)[None, :]
    scale = np.ones_like(fact)
    scale[0, :] = 1 / np.sqrt(2.0)
    scale[0, j] = 1.0
    scale[1:, j] = np.sqrt(2.0)
    return fact * scale


def polar_rot_base(unqs, theta, p):
    r"""Theta-dependent Wigner-d base coefficients for every degree.

    This is the expensive Jacobi part of :func:`rot_polar`. It depends only on
    ``theta`` — not on the ``inv``/``loc`` flags — so a forward + inverse
    rotation pair (as in M2M/M2L/L2L) can assemble it once and pass it to both
    :func:`rot_polar` calls via ``base=`` instead of recomputing it twice.

    Parameters
    ----------
    unqs : list of jnp.ndarray
        Unique parameter combinations (from :func:`precompute_combs`).
    theta : jnp.ndarray
        Polar rotation angles.
    p : int
        Maximum expansion order.

    Returns
    -------
    list of jnp.ndarray
        Per degree ``j``, the unique coefficients from
        :func:`get_polar_rot_coeff`.
    """
    return [get_polar_rot_coeff(j, unqs[j], theta) for j in range(p+1)]

@partial(jax.jit, static_argnames=['inv', 'loc'])
def rot_polar(unqs, invs, theta, coeffs, inv=False, loc=False, base=None):
    r"""Rotate multipole/local coefficients around the :math:`y`-axis.

    Parameters
    ----------
    unqs, invs : list of jnp.ndarray
        Precomputed Wigner tables (from :func:`precompute_combs`).
    theta : jnp.ndarray
        Rotation angles, one per expansion.
    coeffs : jnp.ndarray
        Coefficient vectors, shape ``(..., (p+1)**2)``.
    inv : bool, optional
        Apply the inverse rotation (:math:`\theta \to -\theta`).
    loc : bool, optional
        Coefficients are a local (regular) expansion instead of a multipole
        (singular) one; switches the normalisation of the real basis.
    base : list of jnp.ndarray, optional
        Precomputed theta-dependent Wigner-d coefficients from
        :func:`polar_rot_base`, reused across a forward + inverse pair. If
        ``None`` they are assembled here (bit-identical either way).

    Returns
    -------
    jnp.ndarray
        Rotated coefficients, same shape as ``coeffs``.

    Notes
    -----
    The rotation is block-diagonal over degrees: each block reads only its own
    slice of the ORIGINAL coefficients. At low orders (``p <= 4``) the
    transformed blocks are emitted independently and concatenated once — a
    chain of ``.at[].set`` updates serialises XLA into ~``2(p+1)`` small
    dependent fusions (latency-bound), while the concat form fuses across
    degrees. At high orders the mega-fusion inverts (wider blocks, register
    pressure), so ``p > 4`` keeps the sequential in-place form. Per-element
    sums are identical either way up to reassociation (~1e-16).
    """
    p = invs[-1].shape[0] - 1
    concat_form = p <= 4
    blocks = []
    for j in range(p+1):
        ### assemble rotation matrix
        coeff_j = get_polar_rot_coeff(j, unqs[j], theta) if base is None else base[j]
        real_part = coeff_j[...,invs[j][:,j:]]
        imag_part = coeff_j[...,invs[j][:,:j]]

        fact = jnp.asarray(_polar_fact(j, bool(inv), bool(loc)))          # host constant
        real_part *= fact[None, :, j:]
        imag_part *= fact[None, :, :j]

        ### imag part (m<0; empty for j=0)
        tmp_i = jnp.flip(real_part[...,1:,1:],axis=(-2,-1)) - jnp.flip(imag_part[...,1:,:],axis=(-2))    # flipped to match the real basis ordering
        ### real part (m>=0)
        tmp_r = real_part.at[...,1:].add(jnp.flip(imag_part,axis=-1))
        if concat_form:
            blocks.append(jnp.einsum("jmk, jk -> jm",tmp_i,coeffs[...,mpl_idx(-j,j):mpl_idx(0,j)]))
            blocks.append(jnp.einsum("jmk, jk -> jm",tmp_r,coeffs[...,mpl_idx(0,j):mpl_idx(j,j)+1]))
        else:
            coeffs = coeffs.at[...,mpl_idx(0,j):mpl_idx(j,j)+1].set(jnp.einsum("jmk, jk -> jm",tmp_r,coeffs[...,mpl_idx(0,j):mpl_idx(j,j)+1]))
            coeffs = coeffs.at[...,mpl_idx(-j,j):mpl_idx(0,j)].set(jnp.einsum("jmk, jk -> jm",tmp_i,coeffs[...,mpl_idx(-j,j):mpl_idx(0,j)]))
    return jnp.concatenate(blocks, axis=-1) if concat_form else coeffs

@partial(jax.jit, static_argnames=['p', 'inv', 'loc'])
def polar_rot_mats(unqs, invs, theta_vec, p, inv=False, loc=False):
    r"""Materialised Wigner application matrices for a (small) vector of thetas.

    Per degree ``j``, the ``(real, imag)`` block matrices that :func:`rot_polar`
    applies implicitly via its einsums — exactly rot_polar's assembly with the
    applications stripped out. Used by the theta-snapped grouped M2L
    (:func:`jaxfmm.kernels.laplace.transf.m2l_theta_setup`), where the matrices
    are shared per snapped-theta class instead of built per pair.

    Parameters
    ----------
    unqs, invs : list of jnp.ndarray
        Precomputed Wigner tables (from :func:`precompute_combs`).
    theta_vec : jnp.ndarray
        Polar angles, shape ``(K,)``.
    p : int
        Maximum expansion order (static).
    inv, loc : bool, optional
        As in :func:`rot_polar` (static).

    Returns
    -------
    reals : list of jnp.ndarray
        ``reals[j]`` has shape ``(K, j+1, j+1)``.
    imags : list of jnp.ndarray or None
        ``imags[j]`` has shape ``(K, j, j)``; ``imags[0]`` is ``None``.
    """
    reals, imags = [], []
    for j in range(p + 1):
        coeff_j = get_polar_rot_coeff(j, unqs[j], theta_vec)
        real_part = coeff_j[..., invs[j][:, j:]]
        imag_part = coeff_j[..., invs[j][:, :j]]

        fact = jnp.asarray(_polar_fact(j, bool(inv), bool(loc)))
        real_part = real_part * fact[None, :, j:]
        imag_part = imag_part * fact[None, :, :j]

        reals.append(real_part.at[..., 1:].add(jnp.flip(imag_part, axis=-1)))
        imags.append((jnp.flip(real_part[..., 1:, 1:], axis=(-2, -1))
                      - jnp.flip(imag_part[..., 1:, :], axis=(-2))) if j > 0 else None)
    return reals, imags


@partial(jax.jit, static_argnames=['inv', 'loc'])
def rot_azimuth(coeffs, phi, inv=False, loc=False):
    r"""Rotate multipole/local coefficients around the :math:`z`-axis.

    Diagonal in the order :math:`m`: in the real basis the pair
    :math:`(C^m_n, C^{-m}_n)` mixes by a :math:`m\phi` rotation. In the sqrt(2) basis
    multipole and local coefficients transform identically (``loc`` is accepted for
    symmetry with :func:`rot_polar` and has no effect).

    Parameters
    ----------
    coeffs : jnp.ndarray
        Coefficient vectors, shape ``(..., (p+1)**2)``.
    phi : jnp.ndarray
        Azimuthal rotation angles.
    inv : bool, optional
        Apply the inverse rotation.
    loc : bool, optional
        Local (regular) expansion instead of a multipole one.

    Returns
    -------
    jnp.ndarray
        Rotated coefficients, same shape as ``coeffs``.
    """
    ms, ns = inv_mpl_idx(jnp.arange(coeffs.shape[-1]))
    flipidx = mpl_idx(-ms,ns)
    sgns = 2*inv - 1
    mphi = ms[None,:]*phi[...,None]
    coeffs = coeffs * jnp.cos(mphi) - sgns*coeffs[...,flipidx] * jnp.sin(mphi)
    return coeffs
