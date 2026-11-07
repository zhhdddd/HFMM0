r"""KIFMM operators (Laplace 3D) for the shared cubic driver.

Orchestration lives in :mod:`jaxfmm.core`; this package supplies the
kernel-general equivalent-density representation: the per-phase functions,
bundled by :func:`make_operators` into a :class:`jaxfmm.core.CubicOperators`
set (the back-compat :func:`evaluate` wraps tree + factory +
:func:`jaxfmm.core.evaluate`).  :mod:`jaxfmm.volume` calls the phase functions
directly, so their signatures are stable.

Implements the translation phases on a 2:1-balanced octree using the
precomputed operators from :func:`jaxfmm.kifmm.operators.build_operators`:

Upward pass
    P2M : particles → leaf upward-equivalent density.
    M2M : child equivalent → parent equivalent (bottom-up over levels).
Interaction
    M2L : source equivalent → target downward-equivalent density (V-list).
    P2P : near-field direct particle sums (U-list).
Downward pass
    L2L : parent downward equivalent → child downward equivalent (top-down).
    L2P : leaf downward equivalent → particle potentials.

Notes
-----
Scaling convention for the homogeneous Laplace kernel
:math:`G(r) = 1/(4\pi r)` (homogeneity degree :math:`-1`): surfaces in
:mod:`jaxfmm.kifmm.operators` are defined for a *unit* box (half-side 1).  A
box at level :math:`l` has half-side
:math:`h_l = \mathrm{domain\_extent} \cdot 2^{-(l+1)}`; its physical surface
points are ``surf_pts * h_l + centre``.

* P2M: ``density = h_l * pinv_up @ (K(check_phys, src) @ q)`` — needs the
  factor :math:`h_l`.
* M2M, M2L, L2L: the matrices are scale-invariant and are used as-is at every
  level.
* L2P, P2P: direct kernel evaluation at physical points — no factor.

The :math:`h_l` factor and the :math:`1/h_l` from the homogeneous kernel
cancel through M2L/M2M/L2L, so the equivalent densities are *physical* charge
densities throughout.
"""

from functools import partial

import jax
import jax.numpy as jnp

from jaxfmm.trees import _morton_to_grid, box_geometry   # box_geometry lives in trees
from jaxfmm import core   # shared cubic driver


from jaxfmm.kifmm.phases import *
from jaxfmm.kifmm.phases import (_decode, _green, _green_grad, _F_at, _decode_leveled)
from jaxfmm.kifmm.assemble import *
from jaxfmm.kifmm.assemble import make_operators, evaluate, _make_operators_leveled
from jaxfmm.kifmm.setup import *
from jaxfmm.kifmm.setup import _frozen_fwd
from jaxfmm.kifmm.fulljit import *
from jaxfmm.kifmm.fulljit import (fulljit_setup, compile_fulljit, fulljit_counts_ok,
    fulljit_oversized, fulljit_caps, fulljit_refit, _kifmm_fulljit_fwd, _kifmm_fulljit_body, _compact_capped)

__all__ = ["box_geometry", "p2m", "p2m_padded", "m2m", "m2m_factored", "upward_pass",
           "m2l", "m2l_groups_padded", "m2l_batched", "m2l_groups_tiered",
           "m2l_batched_tiered", "v_pair_count", "m2p", "p2l",
           "l2l", "l2l_factored", "l2p", "l2p_padded", "downward_pass",
           "make_operators", "evaluate", "setup", "evaluate_setup", "compile_evaluator",
           "fulljit_setup", "compile_fulljit", "fulljit_counts_ok", "fulljit_oversized", "fulljit_caps", "fulljit_refit", "p2m_lean", "m2p_lean", "p2l_lean", "l2p_lean"]
