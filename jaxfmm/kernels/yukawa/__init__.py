r"""Screened-Coulomb (Yukawa) kernel."""
from jaxfmm.kernels.yukawa.kernel import *
from jaxfmm.kernels.yukawa.kernel import __all__
from jaxfmm.kernels.yukawa import kernel as _impl


def __getattr__(name):
    """Delegate non-__all__ attribute access (private helpers, constants) to the
    implementation module, so the package behaves like the pre-split flat module."""
    return getattr(_impl, name)
