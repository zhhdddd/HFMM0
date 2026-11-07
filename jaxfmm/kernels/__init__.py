r"""Pluggable kernels for the KIFMM/cubic stack.

A :class:`~jaxfmm.kernels.spec.Kernel` bundles every kernel-specific callable the
engines need, so the tree/traversal code stays kernel-agnostic.

Examples
--------
>>> from jaxfmm.kernels import LAPLACE, yukawa
>>> s = kifmm.setup(pts, p=4, kernel=yukawa(2.0))
"""
from jaxfmm.kernels.spec import Kernel
from jaxfmm.kernels.laplace.kernel import LAPLACE, laplace_G_np, laplace_G
from jaxfmm.kernels.yukawa import yukawa

__all__ = ["Kernel", "LAPLACE", "yukawa", "laplace_G_np", "laplace_G"]
