r"""jaxFMM — fast multipole methods in JAX.

Front door: :func:`jaxfmm.plan` picks, tunes and certifies an evaluator for a
point set and error tolerance. The engines live in subpackages (``jaxfmm.flex``,
``jaxfmm.kifmm``, ``jaxfmm.uniform``, ``jaxfmm.volume``), finite-element sources
in ``jaxfmm.fem``, kernels in ``jaxfmm.kernels`` and concrete applications
(micromagnetics) in ``jaxfmm.apps``.
"""
from jaxfmm.flex import *
from jaxfmm.trees import *
from jaxfmm.util import *

_LAZY = {
    "plan": ("jaxfmm.tune", "plan"),          # one-knob certified evaluator (the front door)
    "flex": ("jaxfmm", "flex"),
    "kifmm": ("jaxfmm", "kifmm"),
    "uniform": ("jaxfmm", "uniform"),
    "volume": ("jaxfmm", "volume"),
    "fem": ("jaxfmm", "fem"),
    "kernels": ("jaxfmm", "kernels"),
    "applications": ("jaxfmm", "applications"),
    "mag": ("jaxfmm.apps", "mag"),
    "mesh_util": ("jaxfmm.fem", "mesh_util"),
}


def __getattr__(name):
    """Lazy top-level access to the engine/application subpackages, without importing
    their (heavy, possibly cyclic) machinery at package import."""
    if name in _LAZY:
        import importlib
        mod, attr = _LAZY[name]
        if mod == "jaxfmm":
            return importlib.import_module(f"jaxfmm.{attr}")
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(f"module 'jaxfmm' has no attribute {name!r}")
