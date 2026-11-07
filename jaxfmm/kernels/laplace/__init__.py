r"""Laplace-specific kernel routines, isolated from the kernel-general core.

Holds the Laplace Green's kernel :math:`G = 1/(4\pi r)` (``kernel``) and the
analytic spherical-harmonic "point-and-shoot" FMM operators — solid-harmonic
``basis``, Wigner-d ``rotation``, and the ``transf`` M2M/M2L/L2L translations.
Everything here only makes sense for the Laplace kernel; the KIFMM core stays
kernel-independent and just imports the kernel from here.

Submodules carry their own ``__all__``, so this package re-exports every defined
name (public + the private helpers), skipping imported modules, for one flat
surface: ``from jaxfmm.kernels.laplace import M2L, ...``.
"""
import types as _types
from jaxfmm.kernels.laplace import kernel as _kernel, basis as _basis, rotation as _rotation, transf as _transf

for _m in (_kernel, _basis, _rotation, _transf):
    for _k, _v in vars(_m).items():
        if not _k.startswith("__") and not isinstance(_v, _types.ModuleType):
            globals()[_k] = _v

__all__ = [_k for _k, _v in list(globals().items())
           if not _k.startswith("_") and not isinstance(_v, _types.ModuleType)]
del _types, _kernel, _basis, _rotation, _transf, _m, _k, _v
