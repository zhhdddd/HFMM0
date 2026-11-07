r"""The flexible (median-split, rotation-based) FMM engine."""
from jaxfmm.flex.fmm import *
from jaxfmm.flex.fmm import __all__
from jaxfmm.flex import fmm as _impl


def __getattr__(name):
    """Delegate non-__all__ attribute access (private helpers, constants) to the
    implementation module, so the package behaves like the pre-split flat module."""
    return getattr(_impl, name)
