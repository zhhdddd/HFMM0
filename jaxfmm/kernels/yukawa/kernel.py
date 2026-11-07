r"""Screened-Coulomb (Yukawa) kernel :math:`G = e^{-\kappa r} / (4\pi r)`."""
import numpy as np
import jax.numpy as jnp

from jaxfmm.kernels.spec import Kernel

__all__ = ["yukawa"]


def yukawa(kappa):
    r"""Build the screened-Coulomb (Yukawa) :class:`~jaxfmm.kernels.spec.Kernel`.

    .. math:: G(x, y) = \frac{e^{-\kappa r}}{4\pi r}, \qquad r = |x - y|.

    Scalar and isotropic (octahedral-symmetric). The exponential screening makes
    the kernel short-ranged, so the KIFMM equivalent densities converge faster
    than for Laplace. Not homogeneous, and no periodic lattice operator exists.

    Parameters
    ----------
    kappa : float
        Inverse screening length :math:`\kappa \ge 0`.

    Returns
    -------
    Kernel
        Kernel bundle with ``symmetric_Oh=True``, ``has_pbc=False``,
        ``homogeneity=None``.
    """
    kappa = float(kappa)

    def G_np(x, y):
        d = x[:, None, :] - y[None, :, :]
        r = np.sqrt((d ** 2).sum(-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            K = np.exp(-kappa * r) / (4.0 * np.pi * r)
        K[r <= 16 * np.finfo(r.dtype).eps * r.max(initial=0.0)] = 0.0   # ULP-relative
        return K

    def g(r):
        k = jnp.asarray(kappa, r.dtype)
        inv4pi = jnp.asarray(0.25 / jnp.pi, r.dtype)
        ok = r > 16 * jnp.finfo(r.dtype).eps * jnp.max(r)             # ULP-relative floor
        return jnp.where(ok, inv4pi * jnp.exp(-k * r) / jnp.where(ok, r, 1.0),
                         jnp.asarray(0.0, r.dtype))

    def g_grad(d):
        # -grad_x G = (1 + κr) exp(-κr) / (4π r³) · d
        r2 = jnp.sum(d * d, axis=-1)
        ok = r2 > (16 * jnp.finfo(r2.dtype).eps)**2 * jnp.max(r2)   # ULP-relative floor
        safe = jnp.where(ok, r2, 1.0)
        r = jnp.sqrt(safe)
        k = jnp.asarray(kappa, d.dtype)
        inv4pi = jnp.asarray(0.25 / jnp.pi, d.dtype)
        w = jnp.where(ok, inv4pi * (1 + k * r) * jnp.exp(-k * r) / (safe * r), 0.0)
        return d * w[..., None]

    def k2(d2):
        # PRECONDITION: d2 pre-floored by the caller (near.eval_direct); the exact
        # test is sound there. Not repeated: the floor's max-reduction would be a
        # fusion barrier in this elementwise inner-loop kernel.
        safe = jnp.where(d2 > 0, d2, 1.0)
        r = jnp.sqrt(safe)
        k = jnp.asarray(kappa, d2.dtype)
        return jnp.where(d2 > 0, jnp.exp(-k * r) / r, 0.0)

    def k2g(d2):
        # PRECONDITION: see k2 -- d2 pre-floored by the caller.
        safe = jnp.where(d2 > 0, d2, 1.0)
        r = jnp.sqrt(safe)
        k = jnp.asarray(kappa, d2.dtype)
        return jnp.where(d2 > 0, (1 + k * r) * jnp.exp(-k * r) / (safe * r), 0.0)

    return Kernel(f"yukawa(kappa={kappa:g})", G_np, g, g_grad, k2, k2g,
                  symmetric_Oh=True, has_pbc=False, homogeneity=None)
