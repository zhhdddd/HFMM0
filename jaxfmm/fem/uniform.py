r"""Element sources on the uniform cubic engine, incl. the differentiable
(recompile-free) moving-geometry variant."""
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jaxfmm.trees import (build_fmm_tree_auto, assign_particles,
                          build_interaction_lists, build_ulist_adaptive, build_wx_lists)
import jaxfmm.uniform as _ufmm
from jaxfmm.fem.quadrature import simplex_quad, element_quad_points

from jaxfmm.fem.elements import *
from jaxfmm.fem.elements import (_as_elements, _self_correction_geom, _coerce_charges)

__all__ = ["element_uniform_setup", "element_uniform_eval", "element_uniform_compile",
           "element_uniform_diff_setup", "element_uniform_diff_eval",
           "element_uniform_diff_compile", "uniform_diff_counts_ok"]



# ---------------------------------------------------------------------------------------------------
# Strayfield potential on the UNIFORM-grid precomputed-operator FMM (jaxfmm.uniform), the fast
# at-scale path: same quad cloud + Duffy as the quad-tree, but a regular 2:1-balanced octree with
# precomputed M2M/M2L/L2L (no per-pair rotations). The mesh nodes ride along as ZERO-charge particles
# so their potential is read off the self-evaluation. Uses a tight pil buffer and a compacted near
# list. Runs in x64, x64 + mixed-precision P2P (p2p_mixed), or full single precision
# (x32, max_depth<=9).
# ---------------------------------------------------------------------------------------------------
def element_uniform_setup(verts, tets, tris, p=3, N_max=64, duffy_deg=3, autotune=False, bin_p2p=True, K=16, bin_ratio=1.2, max_depth=None):
    r"""
    Geometry-fixed setup for the strayfield potential at the mesh nodes via the uniform-grid FMM.

    Builds the quad cloud + nodes (zero-charge eval particles), the 2:1-balanced uniform
    tree + interaction lists (tight pil buffer, compacted near list), the precomputed
    per-level operators, and the Duffy self-correction. Consumed by
    :func:`element_uniform_eval`.

    Parameters
    ----------
    verts : jnp.ndarray, shape (N_vert, 3)
        Mesh vertex coordinates (also the evaluation nodes).
    tets : jnp.ndarray or None, shape (N_tet, 4)
        Tetrahedra carrying the P0 volume charge; ``None`` disables the family.
    tris : jnp.ndarray or None, shape (N_tri, 3)
        Boundary triangles carrying the P1 surface charge; ``None`` disables the family.
    p : int, optional
        Multipole order and volume quadrature degree (surface uses ``p+1``).
    N_max : int, optional
        Refinement target for points per leaf box.
    duffy_deg : int, optional
        Degree of the Duffy rule for the self-correction.
    autotune : bool, optional
        Pick the near-field ``mem_limit`` by a setup-time P2P sweep
        (:func:`~jaxfmm.uniform.autotune_mem_limit`) instead of the heuristic
        :func:`jaxfmm.uniform.auto_mem_limit`.
    bin_p2p : bool, optional
        Build the occupancy-binned near field (see Notes).
    K : int, optional
        Number of occupancy-bin width tiers.
    bin_ratio : float, optional
        No-regret gate: binning is applied only when the predicted P2P tile-work
        speedup (:func:`~jaxfmm.uniform.predicted_p2p_speedup`) exceeds this ratio.
    max_depth : int or None, optional
        Pin the tree depth explicitly (default: automatic, see Notes).

    Returns
    -------
    dict
        Setup dict consumed by :func:`element_uniform_eval` /
        :func:`element_uniform_compile`.

    Notes
    -----
    A hardware-adaptive ``mem_limit`` for the near field is stored in the returned dict
    (so the same setup runs without OOM across cards / leaf densities);
    ``element_uniform_eval`` uses it unless an explicit ``mem_limit`` is passed.

    ``bin_p2p`` builds the occupancy-binned near field (``K`` width tiers, see
    :func:`jaxfmm.uniform.near_pairs_binned`) so the P2P avoids the
    ``maxleaf x maxleaf`` padding waste -- numerically equal to the flat near field,
    faster when leaf occupancy is uneven. The bins are frozen into the dict alongside the
    rest of the (fixed) topology. NO-REGRET GUARD: below ``bin_ratio`` the tiers recover
    too little to beat their overhead (e.g. a nearly-uniform dense cloud), so the flat
    near field is used. Strayfield meshes are surface-clustered (most pairs in small
    tiers), so they bin. (A pure setup-time decision: it only sets ``p2p_bins`` or not,
    with no effect on jit-compilability -- the eval handles both paths.)

    The tree is built via :func:`jaxfmm.trees.build_fmm_tree_auto`: a COMPACT box buffer
    (doubling on overflow, not the dense ``8^{d+1}/7`` bound that host-OOMs at large
    depth) and, when ``max_depth`` is None, a depth that REFINES past the volume-filling
    heuristic while the densest leaf holds > 4*N_max points — the heuristic under-refines
    quasi-2D (thin-film) meshes, which pay heavily in overfull-leaf P2P.
    """
    tets, tris = _as_elements(tets, tris)
    bvt, wvt = simplex_quad(3, p); bst, wst = simplex_quad(2, p + 1)
    qpt, vt = element_quad_points(verts, tets, bvt)
    qps, ar = element_quad_points(verts, tris, bst)
    qpos = jnp.concatenate([qpt.reshape(-1, 3), qps.reshape(-1, 3)])
    n_src = qpos.shape[0]
    positions = jnp.concatenate([qpos, verts])                       # sources (charged) + nodes (0-charge eval)
    N = positions.shape[0]
    # residual depth overflow (densest leaf still > N_max at the refined depth, e.g. the surface-quad
    # layer) is fine: pil is sized to the actual max leaf count below, so the FMM stays correct
    # (slightly larger near field). refine_factor=4 tolerates the quad-layer concentration; genuine
    # under-refinement (thin films: maxleaf ~10x N_max) still triggers the deeper tree.
    out, mb, md, maxleaf = build_fmm_tree_auto(positions, N_max=N_max, max_depth=max_depth,
                                               refine_factor=4)
    (m_id, p_id, lvls, counts, is_leaf, sort_idx, rev, dext, pmin, ovd, ovb) = out
    _, pil = assign_particles(sort_idx, m_id, lvls, counts, is_leaf, md, N_max=maxleaf)
    _, V = build_interaction_lists(m_id, lvls, is_leaf, mb, md)
    Uad, _ = build_ulist_adaptive(m_id, lvls, is_leaf, mb, md)
    W, X, _ = build_wx_lists(m_id, lvls, is_leaf, mb, md)
    idt = np.asarray(jnp.array(0)).dtype                             # default int dtype for eval_direct's pad
    dc = np.asarray(_ufmm.near_pairs(Uad, is_leaf, lvls))
    dc = dc[(dc[:, 0] >= 0) & (dc[:, 1] >= 0)].astype(idt)           # compacted near list
    do_bin = bin_p2p and dc.shape[0] > 0 and _ufmm.predicted_p2p_speedup(dc, counts, int(pil.shape[1]), K) > bin_ratio
    if do_bin:                                                       # occupancy-binned near field (frozen with topology)
        dir_cnct, b_offs, b_widths = _ufmm.near_pairs_binned(dc, counts, int(pil.shape[1]), K=K)
        p2p_bins = (b_offs, b_widths)
    else:
        dir_cnct, p2p_bins = jnp.asarray(dc), None
    nw = int((np.asarray(W[:, 0]) >= 0).sum()); nx = int((np.asarray(X[:, 0]) >= 0).sum())
    rnd = lambda n: (((n + 8191) // 8192) * 8192) if n else None
    ops = _ufmm.precompute_operators(p, float(dext), md)
    lr = _ufmm.m2l_level_ranges(lvls, md)                            # static per-level box slices (faster M2L)
    meml = (_ufmm.autotune_mem_limit(positions, pil, dir_cnct) if autotune     # hardware-adaptive near-field batch
            else _ufmm.auto_mem_limit(int(pil.shape[1]), positions.itemsize))
    wm_tet, wm_tri = wvt[None, :] * vt[:, None], wst[None, :] * ar[:, None]
    ct = _self_correction_geom(verts, tets, duffy_deg, qpt, wm_tet, jnp.asarray(bvt))
    cs = _self_correction_geom(verts, tris, duffy_deg, qps, wm_tri, jnp.asarray(bst))
    return dict(positions=positions, n_src=n_src, n_nodes=verts.shape[0], m_id=m_id, p_id=p_id, lvls=lvls,
                is_leaf=is_leaf, pil=pil, V=V, dir_cnct=dir_cnct, p2p_bins=p2p_bins, W=(W if nw else None), X=(X if nx else None),
                w_np=rnd(nw), x_np=rnd(nx), dext=dext, pmin=pmin, ops=ops, md=md, lr=lr, mem_limit=meml,
                wm_tet=wm_tet, wm_tri=wm_tri, bary_tri=jnp.asarray(bst), tets=tets, tris=tris, corr_t=ct, corr_s=cs)

def element_uniform_eval(s, vol_chrg, surf_nodal, p2p_mixed=False, mem_limit=None):
    r"""
    Strayfield potential at the mesh nodes for current charges via the uniform-grid FMM.

    Volume P0 + surface P1 charges -> quad point-charges; the nodes (zero charge) receive
    the potential, which is read off and Duffy-self-corrected. Differentiable in the
    charges. See :func:`element_uniform_setup`.

    Parameters
    ----------
    s : dict
        Setup dict from :func:`element_uniform_setup`.
    vol_chrg : jnp.ndarray or None, shape (N_tet,)
        P0 volume charge density per tet.
    surf_nodal : jnp.ndarray or None, shape (N_tri, 3)
        P1 nodal surface charge per boundary triangle.
    p2p_mixed : bool, optional
        Run the near field in float32 (f64 accumulation).
    mem_limit : float or None, optional
        Defaults to the setup's hardware-adaptive value; pass a float to override.

    Returns
    -------
    jnp.ndarray, shape (N_vert,)
        Potential at the mesh nodes.
    """
    if mem_limit is None:
        mem_limit = s.get("mem_limit", jnp.inf)
    vol_chrg, surf_nodal = _coerce_charges(s, vol_chrg, surf_nodal)
    qct = vol_chrg[:, None] * s["wm_tet"]
    qcs = jnp.einsum("qk,ek->eq", s["bary_tri"], surf_nodal) * s["wm_tri"]
    qchg = jnp.concatenate([qct.reshape(-1), qcs.reshape(-1)])
    charges = jnp.concatenate([qchg, jnp.zeros(s["n_nodes"], qchg.dtype)])
    u_all = _ufmm.uniform_eval(s["positions"], charges, s["m_id"], s["p_id"], s["lvls"], s["is_leaf"],
                               s["pil"], s["V"], s["dir_cnct"], s["dext"], s["pmin"], s["ops"], s["md"],
                               W_list=s["W"], X_list=s["X"], w_n_pairs=s["w_np"], x_n_pairs=s["x_np"],
                               mem_limit=mem_limit, p2p_mixed=p2p_mixed, level_ranges=s.get("lr"),
                               p2p_bins=s.get("p2p_bins"))
    out = u_all[s["n_src"]:]                                          # node potentials
    corr = jnp.zeros(s["n_nodes"], out.dtype)
    corr = corr.at[s["tets"]].add(vol_chrg[:, None] * s["corr_t"].sum(axis=2))
    corr = corr.at[s["tris"]].add(jnp.einsum("elv,ev->el", s["corr_s"], surf_nodal))
    return out + corr / (4 * jnp.pi)


# ---------------------------------------------------------------------------------------------------
# Shape-differentiable UNIFORM-grid FMM (d u / d verts via autodiff), mirroring the quad-tree diff_*.
# The uniform path is structurally simpler to differentiate:
#   * the TOPOLOGY (Morton sort, pil, V/U/W/X lists) is integer/combinatorial -> frozen at setup,
#     held fixed (valid for small shape steps; re-setup when the sort/lists would change);
#   * the box centres are a regular LATTICE from (pmin, domain_extent), which we FREEZE -- this is
#     required, not just convenient: the precomputed M2L/M2M/L2L operators are exact only for the
#     fixed lattice spacing domain_extent/2^l, so the bounding box must stay fixed. Frozen centres
#     therefore need no NaN-safe AABB recompute (cf. _recompute_boxcenters) and contribute no grad.
#   * the operators are precomputed constants -> the far-field backward is a plain matmul-transpose
#     (no differentiated rotation pipeline, the quad-tree's expensive part). Shape dependence enters
#     only through S2M (positions vs centres), L2P, P2P and the charges/quad-weights -- all cheap.
# A differentiable forward recomputes the quad cloud, charges, Duffy factors from `verts` and runs
# uniform_eval (pmin/dext/ops/topology are stored constants, so grad flows only through positions).
# ---------------------------------------------------------------------------------------------------
def element_uniform_diff_setup(verts, tets, tris, p=3, N_max=64, duffy_deg=3, bin_p2p=True, K=16, bin_ratio=1.2, max_depth=None):
    r"""
    Build the FIXED topology + operators for the shape-differentiable uniform-grid FMM.

    Built at the reference geometry ``verts``; consumed by
    :func:`element_uniform_diff_eval`. Like :func:`element_uniform_setup` but stores the
    raw quadrature bases (so the quad points, weights and charges can be recomputed from
    a perturbed ``verts`` inside the differentiable forward). Rebuild when the geometry
    moves enough to change the Morton sort / interaction lists (valid for small steps).

    Parameters
    ----------
    verts, tets, tris, p, N_max, duffy_deg, bin_p2p, K, bin_ratio, max_depth
        As in :func:`element_uniform_setup`.

    Returns
    -------
    dict
        Setup dict consumed by :func:`element_uniform_diff_eval` /
        :func:`element_uniform_diff_compile`.

    Notes
    -----
    ``bin_p2p`` freezes the occupancy-binned near field (``K`` tiers) into the topology,
    so the differentiable forward/backward P2P avoids the ``maxleaf x maxleaf`` padding
    waste (numerically equal, faster on uneven occupancy). The bins are part of the
    frozen topology, so they add no recompiles within a setup. Binning is applied only
    when the predicted P2P tile-work speedup exceeds ``bin_ratio`` (no-regret guard; see
    :func:`element_uniform_setup`).
    """
    tets, tris = _as_elements(tets, tris)
    bvt, wvt = simplex_quad(3, p); bst, wst = simplex_quad(2, p + 1)
    qpt, _ = element_quad_points(verts, tets, bvt)
    qps, _ = element_quad_points(verts, tris, bst)
    qpos = jnp.concatenate([qpt.reshape(-1, 3), qps.reshape(-1, 3)])
    n_src = qpos.shape[0]
    positions = jnp.concatenate([qpos, verts])
    N = positions.shape[0]
    # same compact-buffer/auto-depth build as element_uniform_setup (identical tree for
    # identical geometry — the diff and non-diff paths must agree to roundoff)
    out, mb, md, maxleaf = build_fmm_tree_auto(positions, N_max=N_max, max_depth=max_depth,
                                               refine_factor=4)
    (m_id, p_id, lvls, counts, is_leaf, sort_idx, rev, dext, pmin, ovd, ovb) = out
    _, pil = assign_particles(sort_idx, m_id, lvls, counts, is_leaf, md, N_max=maxleaf)
    _, V = build_interaction_lists(m_id, lvls, is_leaf, mb, md)
    Uad, _ = build_ulist_adaptive(m_id, lvls, is_leaf, mb, md)
    W, X, _ = build_wx_lists(m_id, lvls, is_leaf, mb, md)
    idt = np.asarray(jnp.array(0)).dtype
    dc = np.asarray(_ufmm.near_pairs(Uad, is_leaf, lvls))
    dc = dc[(dc[:, 0] >= 0) & (dc[:, 1] >= 0)].astype(idt)
    do_bin = bin_p2p and dc.shape[0] > 0 and _ufmm.predicted_p2p_speedup(dc, counts, int(pil.shape[1]), K) > bin_ratio
    if do_bin:                                                       # occupancy-binned near field (frozen with topology)
        dir_cnct, b_offs, b_widths = _ufmm.near_pairs_binned(dc, counts, int(pil.shape[1]), K=K)
        p2p_bins = (b_offs, b_widths)
    else:
        dir_cnct, p2p_bins = jnp.asarray(dc), None
    nw = int((np.asarray(W[:, 0]) >= 0).sum()); nx = int((np.asarray(X[:, 0]) >= 0).sum())
    nv = int(((np.asarray(V[:, 0]) >= 0) & (np.asarray(V[:, 1]) >= 0)).sum())
    rnd = lambda n: (((n + 8191) // 8192) * 8192) if n else None
    ops = _ufmm.precompute_operators(p, float(dext), md)
    lr = _ufmm.m2l_level_ranges(lvls, md)
    meml = _ufmm.auto_mem_limit(int(pil.shape[1]), positions.itemsize)          # hardware-adaptive near-field batch
    return dict(p=p, n_src=n_src, n_nodes=verts.shape[0], m_id=m_id, p_id=p_id, lvls=lvls,
                is_leaf=is_leaf, pil=pil, V=V, dir_cnct=dir_cnct, p2p_bins=p2p_bins, W=(W if nw else None), X=(X if nx else None),
                w_np=rnd(nw), x_np=rnd(nx), v_np=rnd(nv), dext=dext, pmin=pmin, ops=ops, md=md, lr=lr, mem_limit=meml,
                tets=tets, tris=tris, bvt=jnp.asarray(bvt), wvt=jnp.asarray(wvt),
                bst=jnp.asarray(bst), wst=jnp.asarray(wst), duffy_deg=duffy_deg)

def _uniform_stale_counts(positions, s):
    r"""
    IN-JIT staleness test for the frozen uniform-grid topology.

    The interaction lists are box-level and the lattice is frozen, so the topology is
    EXACT iff (a) every particle is inside the frozen root box and (b) every particle's
    Morton cell still matches its assigned leaf. Returns traced counts of violations of
    each -- both zero <=> the frozen ``s`` reproduces a fresh rebuild.
    """
    from jaxfmm.trees.cubic import _morton_3d
    md, gs = s["md"], 2 ** s["md"]
    scaled = (positions - s["pmin"]) / s["dext"] * gs
    eps = gs * 1e-9                                                    # build clips pmax-corner points
    n_outside = jnp.any((scaled < -eps) | (scaled > gs + eps), axis=1).sum()   # INTO the boundary cell
    grid = jnp.clip(scaled.astype(jnp.uint64), 0, gs - 1)
    code_md = _morton_3d(grid[:, 0], grid[:, 1], grid[:, 2])
    pil, lvls = s["pil"], s["lvls"]
    valid = (pil >= 0) & s["is_leaf"][:, None] & (lvls > 0)[:, None]
    shift = (jnp.uint64(3) * (md - lvls).astype(jnp.uint64))[:, None]
    box_code = code_md[jnp.clip(pil, 0, positions.shape[0] - 1)] >> shift
    n_misplaced = (valid & (box_code != s["m_id"][:, None])).sum()
    return dict(n_outside_root=n_outside, n_misplaced=n_misplaced)


def uniform_diff_counts_ok(s, counts):
    r"""
    Host-side verdict on the uniform-diff staleness counts.

    Checks the counts from ``element_uniform_diff_eval(..., return_counts=True)``: the
    frozen topology is exact iff both are zero.

    Parameters
    ----------
    s : dict
        Setup dict from :func:`element_uniform_diff_setup` (unused; kept for driver-API
        parity with :func:`jaxfmm.fem.quadtree.fulljit_counts_ok`).
    counts : dict
        Traced staleness counts ``n_outside_root`` / ``n_misplaced`` from
        ``element_uniform_diff_eval(..., return_counts=True)``.

    Returns
    -------
    ok : bool
        Whether the frozen topology is still exact.
    info : dict
        Plain-int staleness counts.
    """
    info = {k: np.asarray(v).tolist() for k, v in counts.items()}
    return info["n_outside_root"] == 0 and info["n_misplaced"] == 0, info


def element_uniform_diff_eval(s, verts, charge_fn, mem_limit=None, p2p_mixed=False, remat=False, offset_m2l=True,
                              return_counts=False):
    r"""
    Differentiable strayfield potential via the uniform-grid FMM for a (perturbed) geometry.

    Uses the fixed topology + frozen lattice/operators in ``s``. Differentiable w.r.t.
    ``verts`` (shape) and, through ``charge_fn``, the magnetisation. Potential only
    (cf. ``uniform_eval``).

    Parameters
    ----------
    s : dict
        Setup dict from :func:`element_uniform_diff_setup`.
    verts : jnp.ndarray, shape (N_vert, 3)
        Current (perturbed) vertex positions.
    charge_fn : callable
        ``charge_fn(verts) -> (rho, sig)``, the P0 volume / P1 surface densities (called
        inside the forward so the charges' geometry dependence is differentiated).
    mem_limit : float or None, optional
        Defaults to the setup's hardware-adaptive value.
    p2p_mixed : bool, optional
        Run the near field in float32 (f64 accumulation).
    remat : bool, optional
        For ``jax.grad`` at scale, pass ``True`` so the near-field P2P backward batches +
        recomputes rather than storing every pair's pairwise tensor as a residual
        (otherwise the P2P backward dominates memory and OOMs).
    offset_m2l : bool, optional
        Use the offset-batched M2L (``False`` selects the per-pair, AD-light form).
    return_counts : bool, optional
        Additionally return the in-jit staleness counts (see Notes).

    Returns
    -------
    out : jnp.ndarray, shape (N_vert,)
        Potential at the mesh nodes.
    counts : dict, optional
        Staleness counts (only with ``return_counts=True``).

    Notes
    -----
    The staleness counts (:func:`_uniform_stale_counts`; check with
    :func:`uniform_diff_counts_ok`) detect exactly how the frozen uniform topology
    degrades -- unlike the capped quad-tree buffers, it degrades by particles LEAVING
    their assigned boxes.
    """
    if mem_limit is None:
        mem_limit = s.get("mem_limit", jnp.inf)
    tets, tris = s["tets"], s["tris"]
    rho, sig = charge_fn(verts)
    qpt, vt = element_quad_points(verts, tets, s["bvt"])
    qps, ar = element_quad_points(verts, tris, s["bst"])
    qpos = jnp.concatenate([qpt.reshape(-1, 3), qps.reshape(-1, 3)])
    qchg = jnp.concatenate([(rho[:, None] * s["wvt"][None, :] * vt[:, None]).reshape(-1),
                            (jnp.einsum("qk,ek->eq", s["bst"], sig) * s["wst"][None, :] * ar[:, None]).reshape(-1)])
    positions = jnp.concatenate([qpos, verts])                       # same ordering as setup -> pil stays valid
    charges = jnp.concatenate([qchg, jnp.zeros(s["n_nodes"], qchg.dtype)])
    u_all = _ufmm.uniform_eval(positions, charges, s["m_id"], s["p_id"], s["lvls"], s["is_leaf"],
                               s["pil"], s["V"], s["dir_cnct"], s["dext"], s["pmin"], s["ops"], s["md"],
                               W_list=s["W"], X_list=s["X"], w_n_pairs=s["w_np"], x_n_pairs=s["x_np"],
                               m2l_n_pairs=s.get("v_np"), mem_limit=mem_limit, p2p_mixed=p2p_mixed,
                               remat=remat, offset_m2l=offset_m2l, level_ranges=s.get("lr"),
                               p2p_bins=s.get("p2p_bins"))
    out = u_all[s["n_src"]:]                                          # node potentials
    ct = _self_correction_geom(verts, tets, s["duffy_deg"], qpt, s["wvt"][None, :] * vt[:, None], s["bvt"])
    cs = _self_correction_geom(verts, tris, s["duffy_deg"], qps, s["wst"][None, :] * ar[:, None], s["bst"])
    corr = jnp.zeros(verts.shape[0]).at[tets].add(rho[:, None] * ct.sum(axis=2)).at[tris].add(jnp.einsum("elv,ev->el", cs, sig))
    out = out + corr / (4 * jnp.pi)
    if return_counts:
        sg = jax.lax.stop_gradient
        return out, {k: sg(v) for k, v in _uniform_stale_counts(positions, s).items()}
    return out

_UDIFF_ARR_KEYS = ("m_id", "p_id", "lvls", "is_leaf", "pil", "V", "dir_cnct", "dext", "pmin",
                   "tets", "tris", "bvt", "wvt", "bst", "wst")
_UDIFF_META_KEYS = ("p", "n_src", "n_nodes", "md", "duffy_deg", "w_np", "x_np", "v_np",
                    "mem_limit", "lr", "p2p_bins")


def _udiff_split(s):
    r""":func:`_fulljit_split` for the uniform-diff setup (see there for why). ``ops`` (the
    per-level operator tables) and the optional W/X lists ride in the array pytree; their
    presence/absence is part of the jit cache key via the pytree structure."""
    meta = tuple(s[k] for k in _UDIFF_META_KEYS) + (s["ops"]["p"],)
    arrs = {k: s[k] for k in _UDIFF_ARR_KEYS}
    arrs["ops"] = {k: v for k, v in s["ops"].items() if k != "p"}   # p: static int, not a leaf
    if s["W"] is not None:
        arrs["W"] = s["W"]
    if s["X"] is not None:
        arrs["X"] = s["X"]
    return meta, arrs


def _udiff_join(meta, arrs):
    r"""Rebuild the setup-dict view :func:`element_uniform_diff_eval` consumes."""
    s = dict(zip(_UDIFF_META_KEYS, meta[:-1]))
    s.update({k: arrs[k] for k in _UDIFF_ARR_KEYS})
    s["ops"] = dict(arrs["ops"], p=meta[-1])
    s["W"] = arrs.get("W")
    s["X"] = arrs.get("X")
    return s


@partial(jax.jit, static_argnames=["meta", "charge_fn", "mem_limit", "p2p_mixed", "remat",
                                   "return_counts"])
def _udiff_fwd(arrs, verts, meta, charge_fn, mem_limit=None, p2p_mixed=False, remat=False,
               return_counts=False):
    r"""MODULE-LEVEL jitted uniform-diff forward (shared trace cache across rebuilds)."""
    return element_uniform_diff_eval(_udiff_join(meta, arrs), verts, charge_fn,
                                     mem_limit=mem_limit, p2p_mixed=p2p_mixed, remat=remat,
                                     return_counts=return_counts)


def element_uniform_diff_compile(s, charge_fn, mem_limit=None, p2p_mixed=False,
                                 remat=False, return_counts=False):
    r"""
    Compile :func:`element_uniform_diff_eval` into a single jitted ``fwd(verts) -> u``.

    ``charge_fn`` and the static options are closed over. Compile once, then every
    shape-opt iteration and its ``jax.grad`` reuse the same executable. Valid while the
    topology in ``s`` holds; rebuild when the geometry moves enough to change the Morton
    sort (prefer the guarded :func:`jaxfmm.driver.element_uniform_diff_driver`, which
    detects that exactly).

    Like :func:`element_quadtree_fulljit_compile`, the setup is passed to a MODULE-LEVEL
    jitted forward as (static meta, array pytree): a rebuild whose tree lands on the same
    buffer shapes re-uses the previous executable with no retrace/compile (``charge_fn``
    must be the same function object across rebuilds).

    Parameters
    ----------
    s : dict
        Setup dict from :func:`element_uniform_diff_setup`.
    charge_fn : callable
        ``charge_fn(verts) -> (rho, sig)``, the P0 volume / P1 surface densities. Must
        be the SAME function object across rebuilds for executable reuse.
    mem_limit : float or None, optional
        Near-field byte budget; defaults to the setup's hardware-adaptive value.
    p2p_mixed : bool, optional
        Run the near field in float32 (f64 accumulation).
    remat : bool, optional
        Rematerialise the near-field P2P in the backward pass (needed for ``jax.grad``
        at scale).
    return_counts : bool, optional
        Have the compiled forward additionally return the in-jit staleness counts.

    Returns
    -------
    callable
        ``fwd(verts) -> (N_vert,) jnp.ndarray`` (with ``return_counts=True``:
        ``fwd(verts) -> (u, counts)``), backed by the module-level jitted forward.
    """
    meta, arrs = _udiff_split(s)

    def fwd(verts):
        return _udiff_fwd(arrs, verts, meta, charge_fn, mem_limit, p2p_mixed, remat,
                          return_counts)
    return fwd



@partial(jax.jit, static_argnames=["skel", "p2p_mixed", "mem_limit"])
def _euni_frozen_fwd(arrs, vol_chrg, surf_nodal, skel, p2p_mixed, mem_limit):
    r"""MODULE-LEVEL jitted frozen uniform-grid evaluator (setup-as-arguments)."""
    from jaxfmm.util import join_static
    return element_uniform_eval(join_static(skel, arrs), vol_chrg, surf_nodal,
                                p2p_mixed=p2p_mixed, mem_limit=mem_limit)


def element_uniform_compile(s, p2p_mixed=False, mem_limit=None):
    r"""
    FROZEN-geometry evaluator for the cubic (precomputed-operator) element path.

    A jitted ``fmm(vol_chrg, surf_nodal) -> u`` over the :func:`element_uniform_setup`
    dict (lists there are already compact/binned). Setup-as-arguments: same-geometry
    recompiles are free; NOTE cross-geometry reuse is limited here -- the tight ``pil``
    width, compact pair counts and ``p2p_bins`` segment offsets are exact per geometry by
    design (deliberate perf choices), so different geometries usually key different
    executables.

    Parameters
    ----------
    s : dict
        Setup dict from :func:`element_uniform_setup`.
    p2p_mixed : bool, optional
        Run the near field in float32 (f64 accumulation).
    mem_limit : float or None, optional
        Near-field byte budget; defaults to the setup's hardware-adaptive value.

    Returns
    -------
    callable
        ``fmm(vol_chrg, surf_nodal) -> (N_vert,) jnp.ndarray``, the jitted
        frozen-geometry evaluator.
    """
    from jaxfmm.util import split_static
    skel, arrs = split_static(s)

    def fmm(vol_chrg, surf_nodal):
        return _euni_frozen_fwd(arrs, vol_chrg, surf_nodal, skel, p2p_mixed, mem_limit)
    return fmm
