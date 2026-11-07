import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jaxfmm import kifmm
from jaxfmm.kernels import Kernel
from jaxfmm.kifmm.operators import build_operators_leveled


def helmholtz_component(k, imaginary=False):

    k = float(k)
    if not np.isfinite(k):
        raise ValueError("k must be a finite real number")
    trig_np = np.sin if imaginary else np.cos
    trig = jnp.sin if imaginary else jnp.cos

    def G_np(x, y):
        r = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
        ok = r > 16 * np.finfo(r.dtype).eps * r.max(initial=0.0)
        return np.where(ok, trig_np(k * r) / np.where(ok, r, 1.0), 0.0)

    def g(r):
        ok = r > 16 * jnp.finfo(r.dtype).eps * jnp.max(r)
        return jnp.where(ok, trig(k * r) / jnp.where(ok, r, 1.0), 0.0)

    def field_weight(d2):
        safe = jnp.where(d2 > 0, d2, 1.0)
        r = jnp.sqrt(safe)

        numerator = (jnp.sin(k * r) - k * r * jnp.cos(k * r) if imaginary
                     else jnp.cos(k * r) + k * r * jnp.sin(k * r))
        return jnp.where(d2 > 0, numerator / (safe * r), 0.0)

    def g_grad(d):
        d2 = jnp.sum(d * d, axis=-1)
        ok = d2 > (16 * jnp.finfo(d2.dtype).eps)**2 * jnp.max(d2)
        return d * field_weight(jnp.where(ok, d2, 0.0))[..., None]

    def k2(d2):
        r = jnp.sqrt(jnp.where(d2 > 0, d2, 1.0))
        return 4 * jnp.pi * jnp.where(d2 > 0, trig(k * r) / r, 0.0)

    def k2g(d2):
        return 4 * jnp.pi * field_weight(d2)


    return Kernel(f"helmholtz_{'imag' if imaginary else 'real'}({k:g})",
                  G_np, g, g_grad, k2, k2g,
                  symmetric_Oh=False, has_pbc=False, homogeneity=None)


def setup(positions, k, p=4, N_max=16, max_depth=None):

    x = np.asarray(positions, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("positions must be a nonempty finite (N,3) array")
    if p < 2 or int(p) != p or N_max < 1 or int(N_max) != N_max:
        raise ValueError("p must be an integer >=2; N_max must be a positive integer")

    if max_depth is not None and (max_depth < 1 or int(max_depth) != max_depth):
        raise ValueError("max_depth must be a positive integer or None")
    real = kifmm.setup(
        jnp.asarray(x), p=int(p), N_max=int(N_max), max_depth=max_depth,
        kernel=helmholtz_component(k), p2p_mixed=False,
        bin_gate=np.inf, batch_size=16, mem_limit=256e6)


    if float(k) == 0:
        return real, None


    imag_ops = build_operators_leveled(int(p), real["max_depth"],
                                     float(real["domain_extent"]),
                                     helmholtz_component(k, True), eps_rel=1e-12)
    imag_ops = {key: jax.device_put(value) if isinstance(value, np.ndarray) else value
                for key, value in imag_ops.items()}
    return real, dict(real, ops=imag_ops)


def evaluate(state, charges):

    q = jnp.asarray(charges)
    if q.shape != (state[0]["positions"].shape[0],) or jnp.iscomplexobj(q):
        raise ValueError("charges must be a real array of shape (N,)")
    q = q.astype(jnp.float64)
    real = kifmm.evaluate_setup(state[0], q)
    imag = kifmm.evaluate_setup(state[1], q) if state[1] is not None else jnp.zeros_like(real)
    return real + 1j * imag
