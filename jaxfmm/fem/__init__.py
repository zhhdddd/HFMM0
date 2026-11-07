r"""Finite-element sources for jaxFMM: P0 volume + P1 surface charge densities
integrated per element (quadrature + Duffy self terms), evaluated through the
element tree, the point quad-tree or the uniform engine."""
from jaxfmm.fem import quadrature, mesh_util
from jaxfmm.fem.elements import *
from jaxfmm.fem.elements import _as_elements, _coerce_charges
from jaxfmm.fem.farfield import *
from jaxfmm.fem.farfield import _elem_frozen_fwd
from jaxfmm.fem.quadtree import *
from jaxfmm.fem.quadtree import (_equad_frozen_fwd, _fulljit_fwd, _fulljit_split,
    _fulljit_join, _fulljit_vg)
from jaxfmm.fem.uniform import *
from jaxfmm.fem.uniform import _udiff_fwd, _udiff_split, _udiff_join

__all__ = ["element_farfield_setup", "element_farfield_eval", "element_nearfield_eval",
           "element_selfcorrection", "element_eval",
           "element_quadtree_setup", "element_quadtree_eval", "element_quadtree_compile", "element_uniform_compile",
           "element_compile",
           "element_quadtree_fulljit_setup", "element_quadtree_fulljit_eval", "element_quadtree_fulljit_check",
           "element_quadtree_fulljit_compile", "fulljit_counts_ok", "uniform_diff_counts_ok",
           "element_uniform_setup", "element_uniform_eval",
           "element_uniform_diff_setup", "element_uniform_diff_eval", "element_uniform_diff_compile",
           "fulljit_oversized", "fulljit_caps", "fulljit_refit"]
