r"""The uniform spherical-harmonic engine (precomputed per-level operators)."""
from jaxfmm.uniform.engine import *
from jaxfmm.uniform.engine import __all__
from jaxfmm.uniform import engine as _impl


def __getattr__(name):
    """Delegate non-__all__ attribute access (private helpers, constants) to the
    implementation module, so the package behaves like the pre-split flat module."""
    return getattr(_impl, name)
