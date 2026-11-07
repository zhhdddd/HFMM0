r"""The volume FMM (continuous polynomial source densities on octree leaves)."""
from jaxfmm.volume.engine import *
from jaxfmm.volume.engine import __all__
from jaxfmm.volume import engine as _impl


def __getattr__(name):
    """Delegate non-__all__ attribute access (private helpers, constants) to the
    implementation module, so the package behaves like the pre-split flat module."""
    return getattr(_impl, name)
