r"""The 3-D Laplace Green's kernel and its :class:`~jaxfmm.kernels.spec.Kernel` bundle.

.. math:: G(x, y) = \frac{1}{4\pi\,|x - y|}

Provides the dense kernel-matrix evaluators (host and device) and the
``LAPLACE`` kernel bundle used as the default by the KIFMM/cubic stack.
"""

import numpy as np
import jax.numpy as jnp
import jax

__all__ = ["laplace_G_np", "laplace_G", "Kernel", "LAPLACE"]

_4PI = 4.0 * np.pi


def laplace_G_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Dense Laplace kernel matrix (NumPy, float64).

    .. math:: K_{ij} = \frac{1}{4\pi\,|x_i - y_j|}, \qquad K_{ij} = 0
              \text{ where } x_i = y_j.

    Parameters
    ----------
    x : np.ndarray
        Evaluation points, shape ``(m, 3)``.
    y : np.ndarray
        Source points, shape ``(n, 3)``.

    Returns
    -------
    np.ndarray
        Kernel matrix, shape ``(m, n)``, float64.
    """
    d = x[:, None, :] - y[None, :, :]      # (m, n, 3)
    r = np.sqrt(np.sum(d ** 2, axis=-1))   # (m, n)
    # Suppress the benign divide-by-zero at r=0 (where the result is 0).
    with np.errstate(divide='ignore', invalid='ignore'):
        # coincidence floor, ULP-relative (see near.py): an exact `== 0` test misses
        # points that arithmetic left one ULP apart (e.g. a PBC image shift).
        return np.where(r > 16 * np.finfo(r.dtype).eps * r.max(initial=0.0),
                        1.0 / (_4PI * r), 0.0)


@jax.jit
def laplace_G(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    r"""Dense Laplace kernel matrix (JAX device version of :func:`laplace_G_np`).

    Parameters
    ----------
    x : jnp.ndarray
        Evaluation points, shape ``(m, 3)``.
    y : jnp.ndarray
        Source points, shape ``(n, 3)``.

    Returns
    -------
    jnp.ndarray
        Kernel matrix, shape ``(m, n)``; zero on the diagonal coincidences.
    """
    d = x[:, None, :] - y[None, :, :]
    r = jnp.sqrt(jnp.sum(d ** 2, axis=-1))
    return jnp.where(r > 16 * jnp.finfo(r.dtype).eps * jnp.max(r),   # ULP-relative floor
                     1.0 / (4.0 * jnp.pi * r), 0.0)


from jaxfmm.kernels.spec import Kernel


def _lap_g(r):
    inv4pi = jnp.asarray(0.25 / jnp.pi, r.dtype)
    return jnp.where(r > 16 * jnp.finfo(r.dtype).eps * jnp.max(r),   # ULP-relative floor
                     inv4pi / r, jnp.asarray(0.0, r.dtype))


def _lap_g_grad(d):
    r2 = jnp.sum(d * d, axis=-1)
    inv4pi = jnp.asarray(0.25 / jnp.pi, d.dtype)
    fl = (16 * jnp.finfo(r2.dtype).eps)**2 * jnp.max(r2)             # ULP-relative floor
    ok = r2 > fl
    w = jnp.where(ok, inv4pi * jnp.where(ok, r2, 1.0) ** -1.5, 0.0)
    return d * w[..., None]


def _lap_k2(d2):
    r"""Squared-distance potential weight.

    PRECONDITION: ``d2`` must already carry the caller's coincidence floor, i.e.
    numerically-coincident pairs must have been set to exactly 0 (see
    :func:`jaxfmm.near.eval_direct`, which applies the ULP-relative floor once per
    tile). The exact test below is then sound. It is deliberately NOT repeated
    here: the floor needs a max-reduction over the tile, which would be a fusion
    barrier in this elementwise inner-loop kernel, and every call site reaches it
    through ``near.py``.
    """
    return jnp.where(d2 > 0, 1 / jnp.sqrt(jnp.where(d2 > 0, d2, 1.0)), 0.0)


def _lap_k2g(d2):
    return _lap_k2(d2) ** 3


LAPLACE = Kernel("laplace", laplace_G_np, _lap_g, _lap_g_grad, _lap_k2, _lap_k2g,
                 symmetric_Oh=True, has_pbc=True, homogeneity=-1.0)
