r"""The pluggable-kernel interface.

A :class:`Kernel` bundles every kernel-specific callable the KIFMM/cubic stack
needs; the engines themselves stay kernel-agnostic.
"""
import jax.numpy as jnp  # noqa: F401  (conventions reference device fns)
from typing import Callable, NamedTuple


class Kernel(NamedTuple):
    r"""A scalar, translation-invariant kernel for the KIFMM/cubic stack.

    The callables must match the Laplace conventions exactly so that the default
    kernel stays bit-identical to the dedicated Laplace code paths.

    Parameters
    ----------
    name : str
        Human-readable kernel name (used in cache keys and diagnostics).
    G_np : Callable
        ``G_np(x, y)`` — host (NumPy, float64) kernel matrix including all
        constants; used for operator assembly.
    g : Callable
        ``g(r)`` — device kernel value at distance ``r``, including constants,
        with ``g(0) = 0`` in a gradient-safe form (the KIFMM phase convention).
    g_grad : Callable
        ``g_grad(d)`` — device :math:`-\nabla_x G` for displacement
        ``d = eval - src`` of shape ``(..., 3)``; ``0`` at ``d = 0``.
    k2 : Callable
        ``k2(d2)`` — device :math:`4\pi`-free potential weight from the SQUARED
        distance, ``0`` at ``0`` (the near-field convention:
        :func:`jaxfmm.near.eval_direct` divides by :math:`4\pi` once at the end;
        for Laplace this is :math:`1/r`).
    k2g : Callable
        ``k2g(d2)`` — device :math:`4\pi`-free field factor :math:`w` with
        ``field = w * d`` (Laplace: :math:`1/r^3`).
    symmetric_Oh : bool
        Invariant under the cube rotation group (enables the octahedral M2L
        class compression); anisotropic kernels must pass ``False``.
    has_pbc : bool
        A periodic lattice operator exists (Laplace only today).
    homogeneity : float or None
        Degree :math:`a` with :math:`G(s x) = s^a G(x)`, or ``None`` if not
        homogeneous.

    Notes
    -----
    The KIFMM operator precompute builds ONE unit-scale operator set and
    rescales it by box size — valid only for ``homogeneity = -1``
    (Laplace-like). Other kernels need per-level operator tables (the uniform
    engine's ``precompute_operators`` pattern), which is not implemented;
    ``kifmm.setup`` gates on this field.
    """
    name: str
    G_np: Callable
    g: Callable
    g_grad: Callable
    k2: Callable
    k2g: Callable
    symmetric_Oh: bool = False
    has_pbc: bool = False
    homogeneity: float = None
