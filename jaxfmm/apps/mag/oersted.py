r"""Oersted (magnetic) fields of current distributions via the FMM.

:math:`\mathbf{B} = \nabla\times\mathbf{A}`, with each vector-potential component
:math:`A_c` the scalar Laplace potential of the current-element moments (the "volume
charge density" view of currents):

.. math::

    \mathbf{m}_e = \mathbf{J}_e V_e \quad\text{(volume current elements, e.g. conductor
    cell centres)}, \qquad
    \mathbf{m}_e = I\,\mathrm{d}\mathbf{l}_e \quad\text{(thin-wire filaments,
    :func:`wire_elements`)},

so B costs three potential-gradient FMM solves (fewer for planar currents, see
``components``) plus an analytic curl assembly. Units: :math:`\mu_0 = 1`, i.e.

.. math:: \mathbf{B} = \frac{1}{4\pi}\sum_e \mathbf{m}_e \times
          \frac{\mathbf{d}_e}{|\mathbf{d}_e|^3}

— multiply by your :math:`\mu_0` for SI.

The evaluation targets are frozen into :func:`setup` (they ride along as zero-charge
particles on the source tree), so a fixed geometry yields one compiled evaluator
reusable for any moments — solver loops, animations and inverse design reuse the same
closure. B is linear in the moments and :func:`compile_evaluator`'s closure is
differentiable with respect to them, which is what makes current-density inverse design
(examples/oersted_inverse.py) a convex problem.

Validation referee: :func:`biot_savart_direct` (chunked direct sum, no shared code with
the FMM path).
"""
import jax
import jax.numpy as jnp
import numpy as np

from jaxfmm import kifmm

__all__ = ["setup", "compile_evaluator", "evaluate", "wire_elements",
           "biot_savart_direct"]


def setup(sources, targets, p=6, N_max=128, **kifmm_kw):
    r"""Freeze the geometry: one KIFMM setup over sources plus ride-along targets.

    Parameters
    ----------
    sources : array_like
        Current-element positions of shape (n_src, 3) (wire filaments / conductor
        cells).
    targets : array_like
        Points of shape (n_trg, 3) where B is evaluated; must be disjoint from the
        sources.
    p, N_max, **kifmm_kw
        Forwarded to :func:`jaxfmm.kifmm.setup`.

    Returns
    -------
    dict
        Wraps the kifmm setup plus the source/target split.
    """
    sources = np.asarray(sources, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    s = kifmm.setup(jnp.asarray(np.concatenate([sources, targets])),
                    p=p, N_max=N_max, **kifmm_kw)
    return dict(kifmm=s, n_src=sources.shape[0], n_trg=targets.shape[0])


def _curl(F, components):
    r"""Assemble B = curl A from the per-component F_c = -grad A_c (missing components = 0)."""
    Z = jnp.zeros_like(F[components[0]][:, 0])
    g = {c: F[c] for c in components}                        # gc[:, k] = -dA_c/dx_k
    gx = lambda c, k: -g[c][:, k] if c in g else Z           # dA_c/dx_k
    return jnp.stack([gx(2, 1) - gx(1, 2),
                      gx(0, 2) - gx(2, 0),
                      gx(1, 0) - gx(0, 1)], -1)


def compile_evaluator(S, components=(0, 1, 2)):
    r"""Build the frozen-geometry evaluator ``B(moments)``.

    Maps moments of shape (n_src, len(components)) to B of shape (n_trg, 3).

    Parameters
    ----------
    S : dict
        Setup from :func:`setup`.
    components : tuple of int, optional
        Which moment components exist (static): planar currents in the xy-plane pass
        ``(0, 1)`` and pay only two FMM solves. ``moments`` columns are ordered as
        ``components``.

    Returns
    -------
    callable
        Jit-compiled closure ``moments -> B``, differentiable with respect to
        ``moments`` (B is linear in them).
    """
    components = tuple(components)
    fmm = kifmm.compile_evaluator(S["kifmm"], field=True)    # F(q) = -grad(potential)
    n_src, n_trg = S["n_src"], S["n_trg"]
    zpad = jnp.zeros(n_trg)

    @jax.jit
    def B_of(moments):
        F = {c: fmm(jnp.concatenate([moments[:, i], zpad]))[n_src:]
             for i, c in enumerate(components)}
        return _curl(F, components)                          # kifmm carries the 1/(4pi)

    return B_of


def evaluate(S, moments, components=(0, 1, 2)):
    r"""One-shot ``B`` at the targets.

    Convenience wrapper; for repeated evaluation on a fixed geometry keep the
    :func:`compile_evaluator` closure instead.

    Parameters
    ----------
    S : dict
        Setup from :func:`setup`.
    moments : array_like
        Current-element moments of shape (n_src, len(components)).
    components : tuple of int, optional
        Which moment components exist (static); see :func:`compile_evaluator`.
        Default ``(0, 1, 2)``.

    Returns
    -------
    jnp.ndarray
        Magnetic field B at the targets, shape (n_trg, 3).
    """
    return compile_evaluator(S, components)(jnp.asarray(moments))


def wire_elements(polylines, dl=0.002, radius=0.0, n_fil=1, seed=0):
    r"""Discretise polylines into thin-wire current elements.

    Resamples each polyline at approximately ``dl`` spacing; for ``radius > 0``
    replicates onto ``n_fil`` parallel filaments spread over the wire cross-section
    (regularises the near field of finite-thickness wires).

    Each polyline carries unit current :math:`I = 1` in the direction of its vertex
    order — scale the returned moments (or per-wire slices of them) by your currents.

    Parameters
    ----------
    polylines : sequence of array_like
        Each of shape (n_i, 3): the wire paths.
    dl : float, optional
        Target element length.
    radius : float, optional
        Wire radius; 0 keeps a single centre filament.
    n_fil : int, optional
        Number of parallel filaments for ``radius > 0``.
    seed : int, optional
        Seed for the random angular offset of the filament fan.

    Returns
    -------
    positions : np.ndarray
        Element positions of shape (N, 3).
    moments : np.ndarray
        Tangential moments :math:`I\,\mathrm{d}\mathbf{l}` (with :math:`I = 1`) of
        shape (N, 3).
    wire_id : np.ndarray
        int32 index of the originating polyline, shape (N,) (enables per-wire
        switching).
    """
    rng = np.random.default_rng(seed)
    n_fil = max(int(n_fil), 1) if radius > 0.0 else 1
    ang = np.linspace(0, 2 * np.pi, n_fil, endpoint=False) + rng.uniform(0, np.pi)
    pos, mom, wid = [], [], []
    for li, p in enumerate(polylines):
        p = np.asarray(p, dtype=np.float64)
        seg = np.diff(p, axis=0)
        for s0, sv in zip(p[:-1], seg):
            ln = np.linalg.norm(sv)
            if ln < 1e-12:
                continue
            n = max(int(np.ceil(ln / dl)), 1)
            t = (np.arange(n) + 0.5) / n
            centr = s0[None] + t[:, None] * sv[None]
            th = sv / ln
            ref = np.eye(3)[np.argmin(np.abs(th))]           # never parallel to th
            u = np.cross(th, ref); u /= np.linalg.norm(u)
            v = np.cross(th, u)
            for a in ang:
                pos.append(centr + radius * (np.cos(a) * u + np.sin(a) * v))
                mom.append(np.tile(sv / n / n_fil, (n, 1)))
                wid.append(np.full(n, li))
    return (np.concatenate(pos), np.concatenate(mom),
            np.concatenate(wid).astype(np.int32))


def biot_savart_direct(sources, moments, targets, chunk=1024):
    r"""Independent direct-sum referee for the FMM path.

    Evaluates :math:`\mathbf{B} = \frac{1}{4\pi}\sum_e \mathbf{m}_e \times
    \mathbf{d}_e/|\mathbf{d}_e|^3` in target chunks on device. ``moments`` may have 2
    columns (planar, z-moment 0) or 3.

    Parameters
    ----------
    sources : array_like
        Current-element positions, shape (n_src, 3).
    moments : array_like
        Element moments, shape (n_src, 2) (planar, z-moment padded to 0) or
        (n_src, 3).
    targets : array_like
        Evaluation points, shape (n_trg, 3).
    chunk : int, optional
        Number of targets evaluated per device batch (default 1024).

    Returns
    -------
    np.ndarray
        Magnetic field B at the targets, shape (n_trg, 3).
    """
    m = np.asarray(moments, dtype=np.float64)
    if m.shape[1] == 2:
        m = np.concatenate([m, np.zeros((m.shape[0], 1))], 1)
    src, m, trg = jnp.asarray(sources), jnp.asarray(m), jnp.asarray(targets)

    @jax.jit
    def one(tc):
        d = tc[:, None, :] - src[None, :, :]
        w = jnp.sum(d * d, axis=-1) ** -1.5
        return jnp.einsum("ts,tsd->td", w, jnp.cross(m[None, :, :], d)) / (4 * jnp.pi)

    return np.concatenate([np.asarray(one(trg[i:i + chunk]))
                           for i in range(0, trg.shape[0], chunk)])
