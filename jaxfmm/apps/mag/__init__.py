r"""Magnetics applications built on top of the core FMM library.

The boundary (enforced by tests/test_boundary.py): core jaxfmm maps discrete sources to
potentials/fields and knows nothing about physics; everything that speaks magnetization,
Ms or current density lives here. Core never imports jaxfmm.apps.mag, so the core
remains a standalone general FMM library.

Front doors (re-exported): :func:`plan_strayfield` (one-knob frozen-geometry evaluator)
and :func:`fulljit_strayfield_driver` (end-to-end jitted, moving geometry).

Modules
-------
strayfield
    Micromagnetic stray-field FEM application (magnetization -> H at mesh nodes).
oersted
    Oersted fields of current distributions (vector potential + curl).
"""
from jaxfmm.apps.mag import strayfield, oersted
from jaxfmm.apps.mag.strayfield import plan_strayfield, fulljit_strayfield_driver

__all__ = ["strayfield", "oersted", "plan_strayfield", "fulljit_strayfield_driver"]
