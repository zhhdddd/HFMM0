r"""Shared tree assemblies, usable across every FMM approach.

Cubic 2:1-balanced octree:

- ``cubic.py``   — ``build_fmm_tree``, ``assign_particles``, Morton helpers,
  ``get_26_neighbors``, ``box_geometry`` (centres + half-sides shared by the
  cubic drivers).
- ``lists.py``   — the cubic interaction lists: ``build_interaction_lists`` /
  ``build_ulist_adaptive`` / ``build_wx_lists``.

Perfectly-balanced median-split cuboid tree:

- ``flexible.py``     — the builders: ``balanced_tree`` /
  ``balanced_tree_elements`` / ``element_aabb`` / ``get_max_l`` /
  ``reduce_max_lvl``.
- ``connectivity.py`` — the interaction lists / setup: ``gen_connectivity``,
  ``AdaptiveConnectivity``, ``gen_connectivity_jit`` / ``connectivity_static``,
  ``gen_hierarchy``, ``handle_padding``.

The submodules carry their own ``__all__`` (which omit several names consumers
import), so this package re-exports *every* defined name (public + the private
``_...`` helpers), skipping imported modules, to present one flat import
surface: ``from jaxfmm.trees import build_fmm_tree, gen_hierarchy, ...``.
"""
import types as _types
from jaxfmm.trees import cubic as _cubic, lists as _lists, flexible as _flexible, connectivity as _connectivity

for _m in (_cubic, _lists, _flexible, _connectivity):
    for _k, _v in vars(_m).items():
        if not _k.startswith("__") and not isinstance(_v, _types.ModuleType):
            globals()[_k] = _v

__all__ = [_k for _k, _v in list(globals().items())
           if not _k.startswith("_") and not isinstance(_v, _types.ModuleType)]
del _types, _cubic, _lists, _flexible, _connectivity, _m, _k, _v
