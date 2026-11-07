r"""Shared driver for the cubic-tree FMM engines.

Every FMM on the 2:1-balanced cubic octree walks the same phase sequence

    S2M -> M2M -> M2L -> [P2L] -> L2L -> L2P  (+ shared near field, + [M2P])

and differs only in what an "expansion" is and how each phase applies it (spherical
harmonics with precomputed per-level operators in :mod:`jaxfmm.uniform`; kernel-general
equivalent densities in :mod:`jaxfmm.kifmm`).  This module owns the shared orchestration:
:func:`evaluate` sequences the phases of a :class:`CubicOperators` set (built once per tree
by a factory such as :func:`jaxfmm.kifmm.make_operators` or
:func:`jaxfmm.uniform.make_operators`) and runs the representation-agnostic near field
(:mod:`jaxfmm.near`) itself.  A new kernel therefore needs a new operator set, not a new
driver.

Conventions
-----------
* Factories capture only SETUP artifacts (operator tables, V/W/X lists, displacement groups,
  level ranges, batch sizes).  Runtime data — positions, charges and everything derived from
  them — flows through :func:`evaluate` as arguments, so a fixed tree can be re-evaluated
  with new charges (the micromagnetic solver loop) without rebuilding the operator set.
* :func:`evaluate` is an un-jitted Python orchestrator calling individually-jitted phases;
  the closures add no tracing or dispatch cost.
* The NamedTuples are pytree-compatible, leaving the door open to jitting the whole driver.
"""

from typing import Any, Callable, NamedTuple, Optional

import numpy as np
import jax.numpy as jnp

from jaxfmm.trees import box_geometry
from jaxfmm.near import (_build_padded, eval_direct, eval_direct_binned,
                         eval_direct_tiered_lean)

__all__ = ["CubicTree", "SourceData", "CubicOperators", "setup_geometry", "evaluate"]


def setup_geometry(positions, N_max=64, max_depth=None, bin_gate=1.2, bin_K=16,
                   periodic_axes=(), domain=None, v_as_blocks=False, want_tiers=False):
    r"""Engine-agnostic cubic-tree setup.

    Builds the 2:1-balanced tree, the eager block-chunked U/V/W/X interaction lists and
    the gated near-field pair strategy — everything an operator factory does NOT own.
    Both :func:`jaxfmm.kifmm.setup` and :func:`jaxfmm.uniform.setup` start here and add
    their representation-specific pieces (particle layout, operator tables, PBC operator)
    on top.

    Parameters
    ----------
    positions : (N, 3) array
        Source/eval point cloud.
    N_max : int
        Leaf-splitting threshold (particles per box).
    max_depth : int, optional
        Cap on the tree depth; ``None`` lets the builder choose.
    bin_gate : float
        Threshold on :func:`jaxfmm.near.predicted_p2p_speedup` above which the
        occupancy-binned near field is chosen over the flat pair list.
    bin_K : int
        Number of occupancy tiers for the binned near field.
    periodic_axes : tuple of int
        Axes with periodic boundary conditions.
    domain : (pmin, extent), optional
        Root-box specification; required to pin the true unit cell under PBC
        (defaults to the bounding cube with a warning).
    v_as_blocks : bool
        Keep V as the list of host blocks (streamed by the KIFMM's
        ``m2l_groups_tiered`` so the full pair list never exists on device); the
        default concatenates to one device array (the uniform engine's M2L reads it
        whole).
    want_tiers : bool
        Additionally build the tier-local near pairs for the lean (sorted-layout) P2P.

    Returns
    -------
    g : dict
        Tree arrays, interaction lists and the chosen near-field structures; engines
        pop what they need (large intermediates are not duplicated —
        ``sort_idx``/``counts`` stay so callers can build their particle layout).

    Notes
    -----
    The tuned choices:

    * eager block-chunked list builders — bounded device peak (the monolithic jitted
      builders spike device memory at large N) and COMPACT pair arrays from the start;
    * near field: FLAT pair list by default, occupancy-BINNED only when
      :func:`jaxfmm.near.predicted_p2p_speedup` exceeds ``bin_gate`` (a clear win on
      clustered clouds, a large LOSS on near-uniform ones — the gate decides);
    * W/X cross-level lists only when the tree is actually adaptive.

    Under ``periodic_axes`` the lists come back wrapped: V is 3-col ``(tgt, src, img)``
    and U/W/X sources are image-ENCODED (``box + max_boxes*img``).
    """
    from jaxfmm.trees import build_fmm_tree_auto, build_v_eager, build_ulist_eager, build_wx_eager
    from jaxfmm.near import near_pairs_binned, predicted_p2p_speedup

    positions = jnp.asarray(positions)
    periodic = len(periodic_axes) > 0
    if periodic and domain is None:
        # default the cell to the bounding CUBE — usually right for meshes that fill the
        # cell, but a cloud not touching all faces gets a smaller cell than intended, so warn.
        pmin_d = jnp.min(positions, axis=0)
        ext_d = float(jnp.max(jnp.max(positions, axis=0) - pmin_d))
        domain = (pmin_d, ext_d)
        import warnings
        warnings.warn("setup_geometry: periodic_axes without domain= — defaulting the unit cell "
                      f"to the bounding cube (pmin={np.asarray(pmin_d)}, extent={ext_d:.4g}). "
                      "Pass domain=(pmin, extent) to pin the true cell.")
    out, mb, md, maxleaf = build_fmm_tree_auto(positions, N_max=N_max, max_depth=max_depth,
                                               refine_factor=2, domain=domain,
                                               periodic_axes=tuple(periodic_axes))
    (m_id, p_id, lvls, counts, is_leaf, sort_idx, _, domain_extent, pmin, _, _) = out
    leaf = np.asarray(is_leaf) & (np.asarray(lvls) > 0)
    del out                                        # sort_idx kept: the lean layout IS it
    adaptive = np.unique(np.asarray(lvls)[leaf]).size > 1
    if periodic:
        V = jnp.asarray(build_v_eager(m_id, lvls, is_leaf, mb, md,
                                      periodic_axes=tuple(periodic_axes)))
        U = build_ulist_eager(m_id, lvls, is_leaf, mb, md,
                              periodic_axes=tuple(periodic_axes))
    else:
        V = build_v_eager(m_id, lvls, is_leaf, mb, md, concat=not v_as_blocks)
        if not v_as_blocks:
            V = jnp.asarray(V)
        U = build_ulist_eager(m_id, lvls, is_leaf, mb, md)

    W = X = None
    if adaptive:
        # compact from the start: a padded-row lax.map form of m2p/p2l allocates
        # prohibitively on adaptive trees (see build_wx_eager)
        Wn, Xn = build_wx_eager(m_id, lvls, is_leaf, mb, md,
                                periodic_axes=tuple(periodic_axes))
        W = jnp.asarray(Wn) if Wn.shape[0] else None   # periodic: image-ENCODED src
        X = jnp.asarray(Xn) if Xn.shape[0] else None

    leaf_idx = np.nonzero(leaf)[0].astype(np.int32)               # host near_pairs: U-list +
    dc = np.concatenate([U, np.stack([leaf_idx, leaf_idx], 1)])   # leaf self-pairs (compact
    del U                                                         # int32 already, no device trip)

    # the predictor is conservative relative to the measured win and binning overhead is
    # small, so gate LOW and evaluate the predictor at the K actually deployed
    bin_ratio = predicted_p2p_speedup(dc, counts, maxleaf, K=bin_K)
    p2p_tiers = None
    if bin_ratio > bin_gate:                              # binning pays on most clouds
        dir_cnct, offs, wds = near_pairs_binned(dc, counts, maxleaf, K=bin_K)
        p2p_bins = (offs, wds)
        if want_tiers:
            from jaxfmm.near import near_pairs_tiered_lean   # tier-local pair form for the
            p2p_tiers = near_pairs_tiered_lean(dc, counts, maxleaf, K=bin_K,
                                               n_img=(27 if periodic else 1))   # lean eval
    else:                                                 # near-uniform: flat wins big
        dir_cnct, p2p_bins = jnp.asarray(dc), None

    return dict(m_id=m_id, p_id=p_id, lvls=lvls, counts=counts, is_leaf=is_leaf,
                sort_idx=sort_idx, domain_extent=domain_extent, pmin=pmin,
                max_boxes=mb, max_depth=int(md), maxleaf=maxleaf, adaptive=bool(adaptive),
                periodic=periodic, V=V, W=W, X=X, dir_cnct=dir_cnct, p2p_bins=p2p_bins,
                p2p_tiers=p2p_tiers, bin_ratio=float(bin_ratio))


class CubicTree(NamedTuple):
    r"""Cubic-octree topology (fixed per setup; from :func:`jaxfmm.trees.build_fmm_tree`)."""
    m_id: Any            # (max_boxes,) uint64 Morton ids
    p_id: Any            # (max_boxes,) int32 parent box index
    lvls: Any            # (max_boxes,) int32 box levels
    is_leaf: Any         # (max_boxes,) bool
    pil: Any             # (max_boxes, N_max) int32 particles-in-leaf, -1 padded
    domain_extent: Any   # float root-box side length
    pmin: Any            # (3,) root-box min corner
    max_depth: int       # static tree depth (Python int)


class SourceData(NamedTuple):
    r"""Runtime data derived once per :func:`evaluate` call and handed to every phase.

    Carries BOTH the raw layout (``positions``/``charges`` indexed through ``tree.pil``)
    and the padded per-box layout (``ppts``/``pchg``/``rev`` from
    :func:`jaxfmm.near._build_padded`) — operator sets pick whichever their phase
    functions consume.  In LEAN mode (memory diet) the padded fields are ``None`` and the
    sorted-layout fields are set instead: ``pos_s``/``q_s`` are the points/charges in
    LEAF-MAJOR SORTED order (each box a contiguous slice ``[starts[b], starts[b]+cnts[b])``),
    so phase kernels use arithmetic slice indices instead of padded tensors."""
    positions: Any       # (N, 3)
    charges: Any         # (N,)
    ppts: Any            # (max_boxes, N_max, 3) padded per-box points
    pchg: Any            # (max_boxes, N_max)    padded per-box charges (0 on padding)
    rev: Any             # (N,) flat index of each particle in the padded layout
    centres: Any         # (max_boxes, 3) box centres
    h: Any               # (max_boxes,)   box half-sides
    pos_s: Any = None    # (N, 3) leaf-major sorted points        (LEAN mode)
    q_s: Any = None      # (N,)   leaf-major sorted charges       (LEAN mode)
    starts: Any = None   # (max_boxes,) int32 slice starts        (LEAN mode)
    cnts: Any = None     # (max_boxes,) int32 slice counts        (LEAN mode)


class CubicOperators(NamedTuple):
    r"""The pluggable operator interface: one closure per FMM phase.

    ``up``/``dn`` are the upward/downward expansions, shape ``(max_boxes, n_dof)`` with
    ``n_dof`` chosen by the representation.  ``p2l``/``m2p`` are the adaptive cross-level
    phases (X/W lists) and may be ``None`` on uniform trees."""
    s2m: Callable        # (src, tree) -> up
    m2m: Callable        # (up, tree)  -> up
    m2l: Callable        # (up, tree)  -> dn
    l2l: Callable        # (dn, tree)  -> dn
    l2p: Callable        # (dn, src, tree) -> pot (N,)
    p2l: Optional[Callable] = None   # (dn, src, tree) -> dn
    m2p: Optional[Callable] = None   # (up, src, tree) -> pot (N,)
    pbc: Optional[Callable] = None   # (dn, up, tree) -> dn  (periodic far tail, root level)


def evaluate(positions, charges, tree, ops, dir_cnct, p2p_bins=None,
             p2p_mixed=False, mem_limit=jnp.inf, remat=False, field=False,
             materialize_padded=True, sorted_meta=None, p2p_tiers=None,
             img_cnct=jnp.array([[]]), near_kernel=None):
    r"""Full FMM potential at every particle on a cubic tree, for ANY operator set.

    Parameters
    ----------
    positions : (N, 3) array
        Runtime source/eval points.
    charges : (N,) array
        Runtime source charges.
    tree : CubicTree
        Topology from ``build_fmm_tree``/``assign_particles``.
    ops : CubicOperators
        Phase closures from a factory (``kifmm``/``uniform_fmm.make_operators``).
    dir_cnct : (n_near, 2) array
        Near pair list (:func:`jaxfmm.near.near_pairs`), or the flat binned list when
        ``p2p_bins`` is given.
    p2p_bins : tuple, optional
        ``(bin_offsets, bin_widths)`` static tuples from
        :func:`jaxfmm.near.near_pairs_binned` (occupancy-binned near field).
    p2p_mixed : bool
        float32 near-field kernel (float64 accumulation).
    mem_limit : float
        Near-field batch sizing (see :func:`jaxfmm.near.auto_mem_limit`).
    remat : bool
        Rematerialise the near field in the backward pass (shape opt).
    field : bool
        Evaluate :math:`-\nabla\phi` (N, 3) instead of the potential.  The operator set
        must have been built with the matching flag (its l2p/m2p closures produce
        (N, 3)); the shared near field switches via its own ``field`` argument.
        Currently supported by the KIFMM operator set (point-charge equivalent
        densities -> analytic gradient kernel); the spherical-harmonic set is
        potential-only.
    materialize_padded : bool
        Build the padded per-box layout (``ppts``/``pchg``/``rev``) once for the phases
        (required by the spherical-harmonic operator set).  ``False`` = LEAN mode
        (memory diet): requires ``sorted_meta``; S2M/L2P/P2P run on the LEAF-MAJOR
        SORTED layout with arithmetic slice indices — no ``(max_boxes, maxleaf, ...)``
        tensor exists.  The operator set must support it (the KIFMM set does; see
        p2m_lean/l2p_lean and the near field's lean mode).
    sorted_meta : tuple, optional
        ``(perm, rank, starts, cnts)`` from the setup: ``perm`` sorts particles
        leaf-major, ``rank`` inverts it, ``starts``/``cnts`` delimit each box's slice.
    p2p_tiers : tuple, optional
        Tier-local pair structures from :func:`jaxfmm.near.near_pairs_tiered_lean`
        (lean near field).
    img_cnct : array, optional
        Periodic image offsets for the near field.
    near_kernel : optional
        Kernel spec; non-Laplace kernels pass their ``k2``/``k2g`` weight functions to
        the near field.

    Returns
    -------
    pot : (N,) or (N, 3) array
        Far field (M2L [+P2L] -> L2L -> L2P) + near field (P2P) [+ M2P];
        (N, 3) when ``field=True``.
    """
    centres, h = box_geometry(tree.m_id, tree.lvls, tree.domain_extent, tree.pmin)
    centres = centres.astype(positions.dtype)                # follow the runtime dtype: one f64
    h = h.astype(positions.dtype)                            # leak would upcast every phase
    lean = not materialize_padded and sorted_meta is not None
    if lean:
        perm, rank, starts, cnts = sorted_meta[:4]
        pos_sorted = len(sorted_meta) > 4 and sorted_meta[4]
        wpad = tree.pil.shape[1]                             # pad by one tile width so every
        if pos_sorted:                                       # slice start+iota(w) is in
            pos_s = positions                                # bounds WITHOUT clip (a clipped
        else:                                                # index chain blocks fusion)
            pos_s = jnp.pad(positions[perm], ((0, wpad), (0, 0)))
        q_s = jnp.pad(charges[perm], (0, wpad))              # charges arrive original-order
        src = SourceData(positions, charges, None, None, None, centres, h,
                         pos_s=pos_s, q_s=q_s, starts=starts, cnts=cnts)
    else:
        ppts, pchg, rev = _build_padded(positions, charges, tree.pil)
        src = SourceData(positions, charges, ppts, pchg, rev, centres, h)

    nk = ({} if not hasattr(near_kernel, "name") or near_kernel.name == "laplace"
          else dict(k2=near_kernel.k2, k2g=near_kernel.k2g))
    up = ops.s2m(src, tree)
    up = ops.m2m(up, tree)
    dn = ops.m2l(up, tree)
    if ops.pbc is not None:                                  # periodic far tail (lattice op
        dn = ops.pbc(dn, up, tree)                           # at the virtual root)
    if ops.p2l is not None:
        dn = ops.p2l(dn, src, tree)
    dn = ops.l2l(dn, tree)
    far = ops.l2p(dn, src, tree)                             # lean: SORTED order

    cen = centres if p2p_mixed else None                     # box-local f32 kernel coords (near.py)
    near = None                                              # near in ORIGINAL order unless set
    N_out = sorted_meta[1].shape[0] if lean else positions.shape[0]   # rank length = true N
    if jnp.asarray(dir_cnct).size == 0:                      # no near pairs
        near = jnp.zeros((N_out, 3) if field else N_out, dtype=positions.dtype)
    elif lean and p2p_tiers is not None:                     # LEAN: tier-local padded P2P,
        far = far + eval_direct_tiered_lean(pos_s, q_s, starts, cnts, cen,   # sorted order
                                            p2p_tiers, img_cnct=img_cnct,
                                            field=field, mem_limit=mem_limit,
                                            mixed=p2p_mixed, remat=remat, **nk)
    elif p2p_bins is not None:                               # occupancy-binned near field
        if lean:                                             # no tier structs: transient padded
            ppts, pchg, rev = _build_padded(positions, charges, tree.pil)
        near = eval_direct_binned(ppts, pchg, ppts, rev, dir_cnct, p2p_bins[0], p2p_bins[1],
                                  img_cnct=img_cnct,
                                  field=field, mem_limit=mem_limit, mixed=p2p_mixed, remat=remat,
                                  centres=cen, **nk)
    else:                                                    # flat near field
        if lean:                                             # transient padded build (flat is
            ppts, pchg, rev = _build_padded(positions, charges, tree.pil)   # the small-N choice)
        near = eval_direct(ppts, pchg, ppts, rev, dir_cnct, img_cnct=img_cnct, field=field,
                           mem_limit=mem_limit, mixed=p2p_mixed, remat=remat, centres=cen, **nk)

    if lean and ops.m2p is not None:
        far = far + ops.m2p(up, src, tree)                   # lean m2p is SORTED-order
    pot = far[rank] if lean else far                         # sorted -> original order
    if near is not None:
        pot = pot + near
    if not lean and ops.m2p is not None:
        pot = pot + ops.m2p(up, src, tree)                   # padded m2p is original-order
    return pot
