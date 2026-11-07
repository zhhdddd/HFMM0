r"""Element sources on the point quad-tree (boxes split elements; quadrature points
as point charges), incl. the fully-jit moving-geometry variant."""
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jaxfmm.trees import balanced_tree, reduce_max_lvl, gen_hierarchy, get_max_l
from jaxfmm.trees import connectivity_static, gen_connectivity_jit, _connectivity_core, _fit_cap, _caps_oversized
from jaxfmm.fem.quadrature import simplex_quad, element_quad_points
from jaxfmm.kernels.laplace import precompute_combs
from jaxfmm.flex import eval_potential
from jaxfmm.kernels.laplace import m2l_theta_setup, m2l_theta_setup_frozen, m2l_theta_bin

from jaxfmm.fem.elements import *
from jaxfmm.fem.elements import (_as_elements, _self_correction_geom, _coerce_charges, _compact_pairs, _quantize_snap_grp)

__all__ = ["element_quadtree_setup", "element_quadtree_eval", "element_quadtree_compile",
           "element_quadtree_fulljit_setup", "element_quadtree_fulljit_eval",
           "element_quadtree_fulljit_compile", "element_quadtree_fulljit_check",
           "fulljit_counts_ok", "fulljit_oversized", "fulljit_caps", "fulljit_refit"]



# ---------------------------------------------------------------------------------------------------
# Quad-point tree (recommended): the sources are the element quadrature POINTS, sorted into an
# ordinary point FMM (gen_hierarchy / eval_potential). Boxes are small point-AABBs -> no element-
# extent bloat -> faster and aspect-ratio robust vs the element tree, same accuracy. The singular
# self/touching term is fixed by a geometry-fixed Duffy correction per (element, node) -- here it
# subtracts the *degree-p* quad sum the point FMM uses (vs the element tree's separate near rule);
# the only price is a negligible residual (FMM truncation on a touching element's far quad pts).
# Differentiable in the charges; same M-gradient story as the element tree.
# ---------------------------------------------------------------------------------------------------
def element_quadtree_setup(verts, tets, tris, eval_pts, p = None, theta = None, N_max = None,
                           duffy_deg = None, jit_cnct = True, m2l_snap_K = None, err_tol = None, **kwargs):
    r"""
    Geometry-fixed setup for the quad-point-tree FEM far + near field.

    Builds the element quadrature point cloud (volume degree ``p``, surface degree
    ``p+1``) and an ordinary point hierarchy (:func:`gen_hierarchy`) with the mesh nodes
    as evaluation points, plus the geometry-fixed Duffy self-correction factors. Built
    once per geometry; charge-independent.

    Parameters
    ----------
    verts : jnp.ndarray, shape (N_vert, 3)
        Mesh vertex coordinates.
    tets : jnp.ndarray or None, shape (N_tet, 4)
        Tetrahedra carrying the P0 volume charge; ``None`` disables the family.
    tris : jnp.ndarray or None, shape (N_tri, 3)
        Boundary triangles carrying the P1 surface charge; ``None`` disables the family.
    eval_pts : jnp.ndarray, shape (N_eval, 3)
        Evaluation points (usually ``verts``; the Duffy self-correction is only built
        when they are).
    p : int, optional
        Multipole order and volume quadrature degree (surface uses ``p+1``).
    theta : float, optional
        Multipole acceptance criterion.
    N_max : int, optional
        Maximum points per leaf box.
    duffy_deg : int, optional
        Degree of the Duffy rule for the self-correction.
    jit_cnct : bool, optional
        Use the jit, fixed-buffer connectivity.
    m2l_snap_K : int or None, optional
        Enable the theta-snapped grouped M2L with ``K`` theta bins (open boundaries
        only; :func:`jaxfmm.kernels.laplace.m2l_theta_setup`). Snap error is
        :math:`\sim 0.1/K`, so pick ``K`` to match the truncation of the chosen ``p``.
    err_tol : float, optional
        Error indication: picks the cheapest :data:`jaxfmm.fem.tolerances.INTERIOR_LADDER` rung
        at or below it (explicitly passed ``p``/``theta``/``N_max``/``duffy_deg`` win). Without
        it: no knobs -> rung I1 (``DEFAULT_TOL``: ``p=3, theta=0.7, N_max=256, duffy_deg=6``);
        any explicit knob bypasses the ladder and unset knobs take ``p=3, theta=0.8,
        N_max=128, duffy_deg=6`` (the per-element path's rule).
    **kwargs
        Passed through to :func:`gen_hierarchy`.

    Returns
    -------
    dict
        Setup dict consumed by :func:`element_quadtree_eval` /
        :func:`element_quadtree_compile`.
    """
    from jaxfmm.fem.tolerances import interior_params
    _r = interior_params(err_tol, jnp.asarray(verts).dtype == jnp.float64, "element_quadtree_setup",
                         p=p, theta=theta, N_max=N_max, duffy_deg=duffy_deg)
    p, theta, N_max, duffy_deg = _r["p"], _r["theta"], _r["N_max"], _r["duffy_deg"]
    tets, tris = _as_elements(tets, tris)
    bvt, wvt = simplex_quad(3, p)
    bst, wst = simplex_quad(2, p + 1)
    qpt, vt = element_quad_points(verts, tets, bvt)
    qps, ar = element_quad_points(verts, tris, bst)
    qpos = jnp.concatenate([qpt.reshape(-1, 3), qps.reshape(-1, 3)])
    hier = gen_hierarchy(qpos, eval_pts=eval_pts, p=p, theta=theta, N_max=N_max, jit_cnct=jit_cnct, **kwargs)
    if m2l_snap_K:
        if int(np.asarray(hier["img_cnct"]).shape[1]) != 0:
            raise ValueError("m2l_snap_K: the theta-snapped M2L does not support PBC images")
        hier = {**hier, "m2l_grp": m2l_theta_setup(hier["mpl_cnct"], hier["boxcenters"],
                hier["eval_boxcenters"], hier["unqs"], hier["invs"], K=m2l_snap_K)}
    wm_tet, wm_tri = wvt[None, :] * vt[:, None], wst[None, :] * ar[:, None]
    self_correct = eval_pts is verts
    ct = _self_correction_geom(verts, tets, duffy_deg, qpt, wm_tet, jnp.asarray(bvt)) if self_correct else None
    cs = _self_correction_geom(verts, tris, duffy_deg, qps, wm_tri, jnp.asarray(bst)) if self_correct else None
    ctf = _self_correction_geom(verts, tets, duffy_deg, qpt, wm_tet, jnp.asarray(bvt), field=True) if self_correct else None
    return dict(hier=hier, wm_tet=wm_tet, wm_tri=wm_tri, bary_tri=jnp.asarray(bst),
                tets=tets, tris=tris, corr_t=ct, corr_s=cs, corr_t_field=ctf,
                self_correct=self_correct, eval_pts=eval_pts)

def element_quadtree_eval(s, vol_chrg, surf_nodal, field = False, mem_limit = jnp.inf, remat = False,
                          m2l_mixed = False):
    r"""
    Evaluate the quad-point-tree FEM potential (or field) for current charges.

    Recomputes the quadrature point-charges (volume P0, surface P1), runs the point FMM,
    and adds the Duffy self-correction when ``eval_pts`` are the mesh nodes. See
    :func:`element_quadtree_setup`.

    Parameters
    ----------
    s : dict
        Setup dict from :func:`element_quadtree_setup`.
    vol_chrg : jnp.ndarray or None, shape (N_tet,)
        P0 volume charge density per tet.
    surf_nodal : jnp.ndarray or None, shape (N_tri, 3)
        P1 nodal surface charge per boundary triangle.
    field : bool, optional
        Return :math:`H=-\nabla u` (vector) instead of the potential ``u`` (see Notes).
    mem_limit : float, optional
        Byte budget for the batched transforms.
    remat : bool, optional
        Rematerialise the near field in the backward pass (see Notes).
    m2l_mixed : bool, optional
        Use the mixed-precision M2L.

    Returns
    -------
    jnp.ndarray, shape (N_eval,) or (N_eval, 3)
        Potential (or field) at the eval points (original order).

    Notes
    -----
    For ``field=False`` (potential) both volume and surface self-terms are corrected. For
    ``field=True`` (vector field) only the **volume** (tet) self-term is corrected -- the
    surface field self-term at a node is log-divergent (in-plane corner term), so the
    surface contribution to H must come from the lumped weak gradient
    (:func:`jaxfmm.apps.mag.strayfield.get_lumped_strayfield`); the direct
    ``field=True`` route is for the volume charges / off-surface eval points.

    For the charge gradient (``jax.grad`` w.r.t. ``vol_chrg``/``surf_nodal`` or, upstream,
    the magnetisation), set a finite ``mem_limit`` (e.g. the L2 cache size) and
    ``remat=True`` so the near-field backward pass batches and recomputes rather than
    storing every pair's residual; this is what keeps the gradient in memory at scale.
    """
    vol_chrg, surf_nodal = _coerce_charges(s, vol_chrg, surf_nodal)
    qct = vol_chrg[:, None] * s["wm_tet"]
    qcs = jnp.einsum("qk,ek->eq", s["bary_tri"], surf_nodal) * s["wm_tri"]
    qchg = jnp.concatenate([qct.reshape(-1), qcs.reshape(-1)])
    out = eval_potential(qchg, **{**s["hier"], "mem_limit": mem_limit}, field=field, remat=remat,
                         m2l_mixed=m2l_mixed)
    if (not field) and s["self_correct"]:
        corr = jnp.zeros(s["eval_pts"].shape[0])
        corr = corr.at[s["tets"]].add(vol_chrg[:, None] * s["corr_t"].sum(axis=2))
        corr = corr.at[s["tris"]].add(jnp.einsum("elv,ev->el", s["corr_s"], surf_nodal))
        out = out + corr / (4 * jnp.pi)
    elif field and s["self_correct"]:                                # volume (tet) field self-correction only
        corr = jnp.zeros((s["eval_pts"].shape[0], 3))
        corr = corr.at[s["tets"]].add(vol_chrg[:, None, None] * s["corr_t_field"].sum(axis=2))  # P0: sum over v
        out = out + corr / (4 * jnp.pi)
    return out



# ---------------------------------------------------------------------------------------------------
# Shape-differentiable quad-tree (d u / d verts via autodiff, the volume_fmm pattern):
#   * the TOPOLOGY (sort indices, interaction lists, offsets, lvl_info) is built ONCE, eagerly, and
#     held FIXED (combinatorial -> effectively stop_gradient; valid for small shape steps);
#   * a differentiable forward recomputes the CONTINUOUS geometry (quad points, box centres, charges,
#     Duffy factors) from `verts` and runs eval_potential + the Duffy correction.
# eval_potential is already differentiable in its array args; the only fixes needed for the *position*
# gradient were NaN-safe box-centre recompute (here) and a NaN-safe inverse distance in eval_direct.
# ---------------------------------------------------------------------------------------------------
def _recompute_boxcenters(pts, leaf_idcs, max_l, s):
    r"""
    Differentiable AABB box centres at every level for the FIXED leaf sort ``leaf_idcs``.

    NaN-safe: gathers with clamped indices and masks padding with a finite sentinel (no
    nanmin/nanmax, whose NaN-padding gradient is NaN). Matches :func:`balanced_tree`'s
    box centres on the forward.
    """
    from jaxfmm.trees.flexible import splits_of
    splits = splits_of(s, max_l)
    N = pts.shape[0]
    valid = leaf_idcs < N
    ps = pts[jnp.clip(leaf_idcs, 0, N - 1)]                      # real points only (no NaNs)
    v = valid; bcs = []; BIG = jnp.asarray(1e30, pts.dtype)
    for l in range(max_l, -1, -1):
        mn = jnp.where(v[..., None], ps, BIG).min(axis=1)        # padding -> +BIG (never argmin)
        mx = jnp.where(v[..., None], ps, -BIG).max(axis=1)
        bcs.append((mn + mx) / 2)
        if l > 0:
            nc = 2 ** splits[l - 1]                              # children of level l-1
            ps = ps.reshape((ps.shape[0] // nc, -1, 3)); v = v.reshape((v.shape[0] // nc, -1))
    return jnp.concatenate(bcs[::-1])

def _build_idcs_list(src_pts, src_idcs, src_rev, eval_pts, eval_idcs, eval_rev, s, src_max_l, eval_max_l, lvl_info, s_eval=None):
    r"""
    Point->box sort indices regrouped to the actual M2L (mpl) and direct (dir) levels.

    Mirrors :func:`jaxfmm.trees.gen_hierarchy`'s ``idcs_list`` for the ASYMMETRIC tree
    (src != eval). ``lvl_info`` is static, so the level targets and the equality branches
    resolve at trace time -- works eagerly and inside jit. Not restricted to
    'mpl level == leaf': when the M2L bottoms out above the leaf, :func:`reduce_max_lvl`
    regroups the leaf sort into the coarser mpl/dir boxes (when mpl == leaf it is the
    leaf sort, just padding-trimmed).
    """
    if len(lvl_info) == 1:                                   # direct-only: eval_potential ignores the regrouping
        return [(src_idcs, src_rev), (eval_idcs, eval_rev), (src_idcs, src_rev), (eval_idcs, eval_rev)]
    mpl_e, mpl_s = lvl_info[-2]
    dir_e, dir_s = lvl_info[-1]
    se = s if s_eval is None else s_eval                     # per-tree split schedules
    src_mpl = reduce_max_lvl(src_pts, src_idcs, s, src_max_l, mpl_s)
    eval_mpl = reduce_max_lvl(eval_pts, eval_idcs, se, eval_max_l, mpl_e)
    src_dir = src_mpl if mpl_s == dir_s else reduce_max_lvl(src_pts, src_idcs, s, src_max_l, dir_s)
    eval_dir = eval_mpl if mpl_e == dir_e else reduce_max_lvl(eval_pts, eval_idcs, se, eval_max_l, dir_e)
    return [src_mpl, eval_mpl, src_dir, eval_dir]

# ---------------------------------------------------------------------------------------------------
# Fully-jit shape-differentiable quad-tree: recompute the WHOLE (capped) topology from verts INSIDE
# one jit, so there is no per-rebuild recompile (a frozen-topology variant would have to
# rebuild+recompile whenever the geometry drifts). Only the geometry-INDEPENDENT tree *structure*
# (level counts, traversal steps, offsets, fixed buffer caps, lvl_info) is frozen at setup; the
# sort, box centres and interaction PAIRS are recomputed each call (combinatorial parts
# stop_gradient'd, continuous geometry differentiated). The frozen-structure connectivity is
# bit-identical to a fresh gen_connectivity across moderate shape changes. Caps carry a `headroom` fraction
# spare so moderate deformations don't overflow (overflow silently truncates -> re-setup if the
# geometry drifts a lot). Handles any (static) level structure via _build_idcs_list (reduce_max_lvl),
# so it is not restricted to the mpl-level-==-leaf (dense-cloud) case.
# ---------------------------------------------------------------------------------------------------
def element_quadtree_fulljit_setup(verts, tets, tris, p=None, theta=None, N_max=None, s=3, duffy_deg=None, headroom=0.15,
                                   m2l_snap_K=None, targets=None, err_tol=None, cap_floor=None):
    r"""
    Freeze the geometry-independent static metadata for the fully-jitted differentiable forward.

    Consumed by :func:`element_quadtree_fulljit_compile`. Probes the realised connectivity
    buffer counts at the reference ``verts``, pads them by the ``headroom`` fraction and rounds
    them up onto the geometric cap ladder (:func:`jaxfmm.trees.connectivity._fit_cap`).

    Parameters
    ----------
    verts : jnp.ndarray, shape (N_vert, 3)
        Reference vertex positions.
    tets : jnp.ndarray or None, shape (N_tet, 4)
        Tetrahedra carrying the P0 volume charge; ``None`` disables the family.
    tris : jnp.ndarray or None, shape (N_tri, 3)
        Boundary triangles carrying the P1 surface charge; ``None`` disables the family.
    p : int, optional
        Multipole order and volume quadrature degree (surface uses ``p+1``).
    theta : float, optional
        Multipole acceptance criterion.
    N_max : int, optional
        Maximum points per leaf box.
    s : int or "auto", optional
        Split schedule; ``"auto"`` picks per-tree fractional-depth schedules.
    duffy_deg : int, optional
        Degree of the Duffy rule for the self-correction.
    headroom : float, optional
        Spare capacity on every frozen buffer cap, as a fraction of the realised count
        (before the ladder rounding, which adds up to one more 19 % bin).
    m2l_snap_K : int or None, optional
        Enable the theta-snapped grouped M2L INSIDE the fulljit forward: the class->tier
        capacities are probed at the reference geometry and FROZEN
        (:func:`jaxfmm.kernels.laplace.m2l_theta_setup_frozen`), the per-pair class
        binning runs in-jit.
    targets : jnp.ndarray or None, optional
        Optional FIXED evaluation points (e.g. a sensor plane) replacing the mesh nodes
        as the eval side. Their tree is built ONCE here and frozen -- the compiled
        forward then recomputes only the source side per call. No Duffy self-correction
        is applied (targets must lie outside the elements).
    err_tol : float, optional
        Error indication: picks the cheapest :data:`jaxfmm.fem.tolerances.INTERIOR_LADDER` rung
        at or below it (explicitly passed ``p``/``theta``/``N_max``/``duffy_deg`` win). Without
        it: no knobs -> rung I1 (``DEFAULT_TOL``: ``p=3, theta=0.7, N_max=256, duffy_deg=6``);
        any explicit knob bypasses the ladder and unset knobs take ``p=3, theta=0.8,
        N_max=128, duffy_deg=6`` (the per-element path's rule). The
        rung is resolved ONCE here: the compiled forward and the guarded driver's rebuilds
        reuse the frozen values.
    cap_floor : dict or None, optional
        Lower bounds ``{"W", "n_mpl", "n_dir"}`` for the connectivity caps (e.g.
        :func:`fulljit_caps` of an earlier setup): the guarded driver sizes from extra geometries
        and never shrinks a cap on a rejection rebuild through it.

    Returns
    -------
    dict
        Setup dict consumed by :func:`element_quadtree_fulljit_eval` /
        :func:`element_quadtree_fulljit_compile`.

    Notes
    -----
    Overflowing ANY frozen cap silently drops pairs — for moving-geometry loops use
    :func:`jaxfmm.driver.element_quadtree_fulljit_driver`, which certifies every call
    against the realised in-jit counts and rebuilds on rejection (manual alternative:
    :func:`element_quadtree_fulljit_check` after motion; it covers the class caps too).
    """
    from jaxfmm.fem.tolerances import interior_params
    _r = interior_params(err_tol, jnp.asarray(verts).dtype == jnp.float64, "element_quadtree_fulljit_setup",
                         p=p, theta=theta, N_max=N_max, duffy_deg=duffy_deg)
    p, theta, N_max, duffy_deg = _r["p"], _r["theta"], _r["N_max"], _r["duffy_deg"]
    tets, tris = _as_elements(tets, tris)
    bvt, wvt = simplex_quad(3, p); bst, wst = simplex_quad(2, p + 1)
    qpt, _ = element_quad_points(verts, tets, bvt); qps, _ = element_quad_points(verts, tris, bst)
    qpos = jnp.concatenate([qpt.reshape(-1, 3), qps.reshape(-1, 3)])
    eval_pts = verts if targets is None else jnp.asarray(targets)
    if s == "auto":                                # per-tree fractional-depth schedules
        from jaxfmm.trees import split_schedule
        s_eval = split_schedule(eval_pts.shape[0], N_max, 2)
        s = split_schedule(qpos.shape[0], N_max, 2)
    else:
        s_eval = s
    src_max_l = get_max_l(qpos.shape[0], N_max, s); eval_max_l = get_max_l(eval_pts.shape[0], N_max, s_eval)
    _, _, sbc, sbl = balanced_tree(qpos, src_max_l, s)
    ei, eirev, ebc, ebl = balanced_tree(eval_pts, eval_max_l, s_eval)
    # probe counts — RETRY with grown buffers until the probe itself is untruncated:
    # a truncated probe (info["overflow"]) reports capped counts, and caps fitted from
    # those undersize the frozen buffers -> the guarded driver livelocks (rebuilds
    # re-probe with the same defaults). Bites when few/far targets concentrate the
    # near field onto one coarse eval box (nws ~ all source leaves).
    _pw = _pm = _pd = None
    for _ in range(8):
        res_ref, info = gen_connectivity_jit(sbc, sbl, ebc, ebl, theta=theta, s=s,
                                             W=_pw, n_mpl=_pm, n_dir=_pd, return_info=True)
        if not info.get("overflow"):
            break
        _pw, _pm, _pd = 2 * info["W"], 2 * info["n_mpl"], 2 * info["n_dir"]
    else:
        raise RuntimeError(f"connectivity probe still truncated after retries: {info}")
    fl = cap_floor or {}
    W = max(_fit_cap(info["max_nws"], 16, headroom), fl.get("W", 0))
    n_mpl = max(_fit_cap(info["n_mpl_used"], 4096, headroom), fl.get("n_mpl", 0))
    n_dir = max(_fit_cap(info["n_dir_used"], 1024, headroom), fl.get("n_dir", 0))
    conn = connectivity_static(sbc, sbl, ebc, ebl, theta, s, W, n_mpl, n_dir)
    lvl_info = conn["lvl_info"]
    unqs, invs = precompute_combs(p)
    m2l_grp_frozen = None
    if m2l_snap_K:
        m2l_grp_frozen = m2l_theta_setup_frozen(res_ref[4], res_ref[0], res_ref[1], unqs, invs,
                                                K=m2l_snap_K, headroom=headroom)
    out = dict(p=p, s=s, s_eval=s_eval, theta=theta, src_max_l=src_max_l, eval_max_l=eval_max_l, conn=conn,
               lvl_info=lvl_info, src_ofs=conn["src_ofs"], trg_ofs=conn["eval_ofs"], unqs=unqs, invs=invs,
               img_out=jnp.array([[]]), PBC_op=jnp.zeros(((p + 1) ** 2, (p + 1) ** 2)),
               tets=tets, tris=tris, bvt=jnp.asarray(bvt), wvt=jnp.asarray(wvt),
               bst=jnp.asarray(bst), wst=jnp.asarray(wst), duffy_deg=duffy_deg,
               m2l_grp_frozen=m2l_grp_frozen)
    if targets is not None:                        # frozen eval side: tree built once, here
        out.update(targets=eval_pts, t_ei=ei, t_eirev=eirev,
                   t_ebc=jnp.concatenate(ebc), t_bc_e=tuple(ebc), t_bl_e=tuple(ebl))
    return out

def element_quadtree_fulljit_eval(s, verts, charge_fn, field=False, mem_limit=jnp.inf, remat=False,
                                  return_counts=False, m2l_mixed=False, charge_args=()):
    r"""
    Fully-jit differentiable quad-tree potential/field at the nodes for ``verts``.

    Recomputes the quad cloud, charges, both balanced trees, box centres, the capped
    interaction lists and runs ``eval_potential`` + the Duffy correction -- all inside
    (use :func:`element_quadtree_fulljit_compile` to jit it). The sort indices, box
    lengths and interaction pairs are ``stop_gradient``'d (integer / combinatorial); the
    box centres, quad points and charges are differentiated. Differentiable w.r.t.
    ``verts`` (shape) and, through ``charge_fn``, the magnetisation.

    Parameters
    ----------
    s : dict
        Setup dict from :func:`element_quadtree_fulljit_setup`.
    verts : jnp.ndarray, shape (N_vert, 3)
        Current vertex positions.
    charge_fn : callable
        ``charge_fn(verts, *charge_args) -> (rho, sig)``, the P0 volume / P1 surface
        densities.
    field : bool, optional
        Return the vector field instead of the potential.
    mem_limit : float, optional
        Byte budget for the batched transforms.
    remat : bool, optional
        Rematerialise the near field in the backward pass.
    return_counts : bool, optional
        Additionally return the realised buffer occupancies (see Notes).
    m2l_mixed : bool, optional
        Use the mixed-precision M2L.
    charge_args : tuple, optional
        Extra TRACED arguments forwarded to ``charge_fn`` (e.g. the nodal
        magnetisation) — lets state vary per call without rebuilding the closure
        (a per-call closure would retrace).

    Returns
    -------
    out : jnp.ndarray, shape (N_eval,) or (N_eval, 3)
        Potential (or field) at the nodes / targets.
    counts : dict, optional
        Realised buffer occupancies (only with ``return_counts=True``).

    Notes
    -----
    The realised occupancies are traced scalars and free -- the capped fills compute them
    anyway: ``n_mpl_used``/``n_dir_used``/``max_nws`` and, with theta-snap,
    ``theta_tier_max``/``theta_orphans``. Compare against the frozen caps with
    :func:`fulljit_counts_ok` -- this is what :class:`jaxfmm.driver.GuardedShapeEvaluator`
    uses to make silent truncation impossible.
    """
    sg = jax.lax.stop_gradient
    tets, tris, sp, conn = s["tets"], s["tris"], s["s"], s["conn"]
    src_ofs, trg_ofs = s["src_ofs"], s["trg_ofs"]
    rho, sig = charge_fn(verts, *charge_args)
    qpt, vt = element_quad_points(verts, tets, s["bvt"])
    qps, ar = element_quad_points(verts, tris, s["bst"])
    qpos = jnp.concatenate([qpt.reshape(-1, 3), qps.reshape(-1, 3)])
    qchg = jnp.concatenate([(rho[:, None] * s["wvt"][None, :] * vt[:, None]).reshape(-1),
                            (jnp.einsum("qk,ek->eq", s["bst"], sig) * s["wst"][None, :] * ar[:, None]).reshape(-1)])
    spe = s.get("s_eval", sp)
    si, sirev, _, sbl = balanced_tree(qpos, s["src_max_l"], sp)       # sort + box lengths (integer/combinatorial)
    si, sirev = sg(si), sg(sirev)
    sbc = _recompute_boxcenters(qpos, si, s["src_max_l"], sp)         # differentiable AABB box centres (concatenated)
    if s.get("targets") is None:
        eval_pts = verts
        ei, eirev, _, ebl = balanced_tree(verts, s["eval_max_l"], spe)
        ei, eirev = sg(ei), sg(eirev)
        ebc = _recompute_boxcenters(verts, ei, s["eval_max_l"], spe)
        bc_e = tuple(sg(ebc[trg_ofs[l]:trg_ofs[l + 1]]) for l in range(len(trg_ofs) - 1))
        bl_e = tuple(sg(b) for b in ebl)
    else:                                                             # frozen eval side (fixed targets)
        eval_pts, ei, eirev = s["targets"], s["t_ei"], s["t_eirev"]
        ebc, bc_e, bl_e = s["t_ebc"], s["t_bc_e"], s["t_bl_e"]
    bc_s = tuple(sg(sbc[src_ofs[l]:src_ofs[l + 1]]) for l in range(len(src_ofs) - 1))   # per-level for connectivity
    bl_s = tuple(sg(b) for b in sbl)
    mpl_cnct, dir_cnct, n_mpl_used, n_dir_used, max_nws = _connectivity_core(
        bc_s, bl_s, bc_e, bl_e, s["theta"], sp,
        conn["n_mpl"], conn["n_dir"], src_ofs, trg_ofs, conn["steps"],
        (conn["nbox_s"], conn["nbox_e"], conn["width_in"], conn["keep"]), conn["img_cnct"], conn["nbox_tot"])
    mpl_cnct, dir_cnct = sg(mpl_cnct), sg(dir_cnct)
    counts = dict(n_mpl_used=sg(n_mpl_used), n_dir_used=sg(n_dir_used), max_nws=sg(max_nws))
    idcs_list = _build_idcs_list(qpos, si, sirev, eval_pts, ei, eirev,  # regroup to the (static) mpl/dir levels
                                 sp, s["src_max_l"], s["eval_max_l"], s["lvl_info"], s_eval=spe)
    m2l_grp = None
    if s.get("m2l_grp_frozen") is not None:                          # theta-snap inside the fulljit
        g = s["m2l_grp_frozen"]                                      # forward: in-jit class binning
        tier_cncts, tc = m2l_theta_bin(mpl_cnct, sg(sbc), sg(ebc), g, return_counts=True)
        counts.update({k: sg(v) for k, v in tc.items()})             # (combinatorial -> sg'd)
        m2l_grp = dict(cls=g["cls"], cnct=tuple(sg(c) for c in tier_cncts),
                       RF=g["RF"], RB=g["RB"], T=g["T"])
    out = eval_potential(qchg, qpos, eval_pts, idcs_list, sbc, ebc, src_ofs, trg_ofs, mpl_cnct, dir_cnct,
                         s["unqs"], s["invs"], s["lvl_info"], s["img_out"], s["PBC_op"],
                         field=field, mem_limit=mem_limit, remat=remat, m2l_grp=m2l_grp,
                         m2l_mixed=m2l_mixed)
    if s.get("targets") is not None:                                 # fixed external targets:
        return (out, counts) if return_counts else out               # no Duffy self-correction
    if not field:                                                    # Duffy self-correction (differentiable)
        ct = _self_correction_geom(verts, tets, s["duffy_deg"], qpt, s["wvt"][None, :] * vt[:, None], s["bvt"])
        cs = _self_correction_geom(verts, tris, s["duffy_deg"], qps, s["wst"][None, :] * ar[:, None], s["bst"])
        corr = jnp.zeros(verts.shape[0]).at[tets].add(rho[:, None] * ct.sum(axis=2)).at[tris].add(jnp.einsum("elv,ev->el", cs, sig))
        out = out + corr / (4 * jnp.pi)
    else:                                                            # field: volume (tet) self-correction only
        ctf = _self_correction_geom(verts, tets, s["duffy_deg"], qpt, s["wvt"][None, :] * vt[:, None], s["bvt"], field=True)
        out = out + (jnp.zeros((verts.shape[0], 3)).at[tets].add(rho[:, None, None] * ctf.sum(axis=2))) / (4 * jnp.pi)
    return (out, counts) if return_counts else out

def fulljit_counts_ok(s, counts):
    r"""
    Host-side verdict on the realised in-jit occupancies vs the frozen caps in ``s``.

    The zero-cost per-call adequacy test the guarded driver runs on the ``counts`` from
    ``element_quadtree_fulljit_eval(..., return_counts=True)`` (contrast
    :func:`element_quadtree_fulljit_check`, an independent eager re-probe).

    Parameters
    ----------
    s : dict
        Setup dict from :func:`element_quadtree_fulljit_setup` (holds the frozen caps).
    counts : dict
        Realised occupancies from ``element_quadtree_fulljit_eval(...,
        return_counts=True)`` (``n_mpl_used``/``n_dir_used``/``max_nws`` and, with
        theta-snap, ``theta_tier_max``/``theta_orphans``).

    Returns
    -------
    ok : bool
        Whether every realised count fits its frozen cap.
    info : dict
        Plain-int realised counts.
    """
    conn = s["conn"]
    info = {k: (np.asarray(v).tolist()) for k, v in counts.items()}
    ok = (info["max_nws"] <= conn["W"] and info["n_mpl_used"] <= conn["n_mpl"]
          and info["n_dir_used"] <= conn["n_dir"])
    # direct-only drift guard: a direct-only setup (one level) cannot run M2L pairs. The converse is
    # legitimate -- close targets near-connect to every source leaf, so a multi-level setup realises
    # zero M2L (empty lists, correct all-direct field) -- and rejecting it livelocked the driver
    ok = ok and not (info["n_mpl_used"] > 0 and len(s["lvl_info"]) == 1)
    if s.get("m2l_grp_frozen") is not None:
        g = s["m2l_grp_frozen"]
        ok = (ok and all(t <= c for t, c in zip(info["theta_tier_max"], g["caps"]))
              and info["theta_orphans"] == 0)
    return bool(ok), info


def fulljit_caps(s):
    r"""The connectivity caps ``{"W", "n_mpl", "n_dir"}`` frozen in a fulljit setup ``s`` (the
    ``cap_floor`` currency of :func:`element_quadtree_fulljit_setup`)."""
    return {k: int(s["conn"][k]) for k in ("W", "n_mpl", "n_dir")}


def fulljit_refit(s, info, headroom=0.15, cap_floor=None):
    r"""The connectivity caps a refit at the realised counts ``info`` would freeze (same
    currency as :func:`fulljit_caps`; never below ``cap_floor``) -- prices the guarded
    driver's lazy downsizes."""
    fl = cap_floor or {}
    return {"W": max(_fit_cap(info["max_nws"], 16, headroom), fl.get("W", 0)),
            "n_mpl": max(_fit_cap(info["n_mpl_used"], 4096, headroom), fl.get("n_mpl", 0)),
            "n_dir": max(_fit_cap(info["n_dir_used"], 1024, headroom), fl.get("n_dir", 0))}


def fulljit_oversized(s, info, headroom=0.15, cap_floor=None):
    r"""
    Lazy-downsize predicate for the guarded driver.

    Would a refit at the realised counts ``info`` (from a CERTIFIED call) SHRINK some frozen
    buffer by at least ``DOWNSIZE_BINS`` ladder bins (>= 29 %), while growing none? Smaller
    shrinks barely pay and would oscillate; a dimension near its cap vetoes -- growth is the
    overflow path's job.

    Parameters
    ----------
    s : dict
        Setup dict from :func:`element_quadtree_fulljit_setup` (holds the frozen caps).
    info : dict
        Plain-int realised counts from a CERTIFIED call (:func:`fulljit_counts_ok`).
    headroom : float, optional
        Spare-capacity fraction the refit would keep (match the setup's value).
    cap_floor : dict or None, optional
        Caps the refit would keep regardless (the driver's ``size_from`` floor).

    Returns
    -------
    bool
        Whether a refit would shrink at least one buffer by two bins without growing any.
    """
    conn, fl = s["conn"], cap_floor or {}
    dims = [(info["max_nws"], conn["W"], 16, fl.get("W", 0)),
            (info["n_mpl_used"], conn["n_mpl"], 4096, fl.get("n_mpl", 0)),
            (info["n_dir_used"], conn["n_dir"], 1024, fl.get("n_dir", 0))]
    if s.get("m2l_grp_frozen") is not None:
        g = s["m2l_grp_frozen"]
        dims += [(t, cap, 256) for t, cap in zip(info["theta_tier_max"], g["caps"])]
    return _caps_oversized(dims, headroom)


def element_quadtree_fulljit_check(s, verts):
    r"""
    Buffer-adequacy probe for the fulljit forward at the CURRENT ``verts``.

    The compiled forward runs with FROZEN caps and **silently truncates** the interaction
    lists on overflow (wrong physics, no error), so moving-geometry loops should call this
    every few steps (eager; reuses the jitted connectivity) and re-run
    :func:`element_quadtree_fulljit_setup` + ``_compile`` when not ok.

    Parameters
    ----------
    s : dict
        Setup dict from :func:`element_quadtree_fulljit_setup`.
    verts : jnp.ndarray, shape (N_vert, 3)
        Current vertex positions.

    Returns
    -------
    ok : bool
        Whether the frozen caps are still adequate.
    info : dict
        Realised counts.

    Notes
    -----
    Checks (a) buffer overflow (``W``/``n_mpl``/``n_dir`` realised counts vs the frozen
    caps) and (b) level-structure drift (``lvl_info`` changed, e.g. the mpl level bottomed
    out differently).
    """
    qpt, _ = element_quad_points(verts, s["tets"], s["bvt"])
    qps, _ = element_quad_points(verts, s["tris"], s["bst"])
    qpos = jnp.concatenate([qpt.reshape(-1, 3), qps.reshape(-1, 3)])
    _, _, sbc, sbl = balanced_tree(qpos, s["src_max_l"], s["s"])
    if s.get("targets") is None:
        _, _, ebc, ebl = balanced_tree(verts, s["eval_max_l"], s.get("s_eval", s["s"]))
    else:
        ebc, ebl = list(s["t_bc_e"]), list(s["t_bl_e"])
    conn = s["conn"]
    res, info = gen_connectivity_jit(sbc, sbl, ebc, ebl, theta=s["theta"], s=s["s"],
                                     W=conn["W"], n_mpl=conn["n_mpl"], n_dir=conn["n_dir"],
                                     return_info=True)
    ok = (not info["overflow"]) and (res[6] == s["lvl_info"])
    if s.get("m2l_grp_frozen") is not None:                          # frozen theta-class caps too
        g = s["m2l_grp_frozen"]; K = g["K"]
        cn = np.asarray(res[4])
        okm = ((cn[:, 0] >= 0) & (cn[:, 1] >= 0)
               & (cn[:, 0] < res[1].shape[0]) & (cn[:, 1] < res[0].shape[0]))
        d = np.asarray(res[1])[cn[okm, 0]] - np.asarray(res[0])[cn[okm, 1]]
        th = np.arctan2(np.hypot(d[:, 0], d[:, 1]), np.abs(d[:, 2]))
        counts = np.bincount(np.clip((th / (0.5 * np.pi / K)).astype(int), 0, K - 1), minlength=K)
        for cls_idx, cap in zip(g["cls"], g["caps"]):
            worst = int(counts[np.asarray(cls_idx)].max())
            info[f"theta_cap_{cap}"] = worst
            ok = ok and worst <= cap
        orphans = int(counts[np.asarray(g["tier_of_cls"]) < 0].sum())   # classes with no tier row:
        info["theta_orphans"] = orphans                                 # binner drops them entirely
        ok = ok and orphans == 0
    return ok, info


def _fulljit_split(s):
    r"""
    Split a fulljit setup dict into hashable static ``meta`` and an array pytree ``arrs``.

    ``meta`` holds shapes, caps, traversal steps, level structure -- everything that
    legitimately keys a compilation; ``arrs`` holds topology, quadrature bases, rotation
    tables -- runtime data. Rejoined by :func:`_fulljit_join`. This is what makes the
    fulljit forward a MODULE-LEVEL jitted function of ``(arrs, verts)``: a rebuilt setup
    whose caps land on the same bin lattice re-uses the SAME executable -- zero retrace,
    zero recompile. (Closure-captured setups would bake the arrays into the HLO as
    constants, so every rebuild would pay an uncached retrace and usually a cache-missed
    compile.)
    """
    conn = s["conn"]
    conn_meta = (conn["symmetric"], conn["W"], conn["n_chi"], conn["src_max"], conn["eval_max"],
                 conn["n_img"], conn["nbox_s"], conn["nbox_e"], conn["steps"], conn["width_in"],
                 conn["keep"], conn["src_ofs"], conn["eval_ofs"], conn["nbox_tot"],
                 conn["n_mpl"], conn["n_dir"], conn["lvl_info"], tuple(conn["periodic_axes"]))
    g = s.get("m2l_grp_frozen")
    snap_meta = None if g is None else (g["caps"], g["K"], len(g["cls"]))
    meta = (s["p"], s["s"], s.get("s_eval", s["s"]), float(s["theta"]), s["src_max_l"],
            s["eval_max_l"], s["duffy_deg"], conn_meta, snap_meta)
    arrs = dict(tets=s["tets"], tris=s["tris"], bvt=s["bvt"], wvt=s["wvt"],
                bst=s["bst"], wst=s["wst"], unqs=s["unqs"], invs=s["invs"],
                img_out=s["img_out"], PBC_op=s["PBC_op"], img_cnct=conn["img_cnct"])
    if s.get("targets") is not None:               # frozen-targets arrays (presence keys the
        arrs["tgt"] = dict(targets=s["targets"], t_ei=s["t_ei"], t_eirev=s["t_eirev"],   # pytree)
                           t_ebc=s["t_ebc"], t_bc_e=s["t_bc_e"], t_bl_e=s["t_bl_e"])
    if g is not None:
        arrs["snap"] = dict(cls=tuple(g["cls"]), RF=g["RF"], RB=g["RB"], T=g["T"],
                            tier_of_cls=g["tier_of_cls"], row_of_cls=g["row_of_cls"])
    return meta, arrs


def _fulljit_join(meta, arrs):
    r"""Rebuild the setup-dict view :func:`element_quadtree_fulljit_eval` consumes."""
    (p, sp, sp_e, theta, src_max_l, eval_max_l, duffy_deg, cm, snap_meta) = meta
    conn = dict(symmetric=cm[0], W=cm[1], n_chi=cm[2], src_max=cm[3], eval_max=cm[4],
                n_img=cm[5], nbox_s=cm[6], nbox_e=cm[7], steps=cm[8], width_in=cm[9],
                keep=cm[10], src_ofs=cm[11], eval_ofs=cm[12], nbox_tot=cm[13],
                n_mpl=cm[14], n_dir=cm[15], lvl_info=cm[16], periodic_axes=cm[17],
                img_cnct=arrs["img_cnct"])
    grp = None
    if snap_meta is not None:
        caps, K, _ = snap_meta
        sn = arrs["snap"]
        grp = dict(cls=sn["cls"], RF=sn["RF"], RB=sn["RB"], T=sn["T"], caps=caps, K=K,
                   tier_of_cls=sn["tier_of_cls"], row_of_cls=sn["row_of_cls"])
    tgt = arrs.get("tgt", {})
    return dict(p=p, s=sp, s_eval=sp_e, theta=theta, src_max_l=src_max_l, eval_max_l=eval_max_l,
                duffy_deg=duffy_deg, conn=conn, lvl_info=conn["lvl_info"], **tgt,
                src_ofs=conn["src_ofs"], trg_ofs=conn["eval_ofs"],
                tets=arrs["tets"], tris=arrs["tris"], bvt=arrs["bvt"], wvt=arrs["wvt"],
                bst=arrs["bst"], wst=arrs["wst"], unqs=arrs["unqs"], invs=arrs["invs"],
                img_out=arrs["img_out"], PBC_op=arrs["PBC_op"], m2l_grp_frozen=grp)


@partial(jax.jit, static_argnames=["meta", "charge_fn", "field", "mem_limit", "remat",
                                   "return_counts", "m2l_mixed"])
def _fulljit_fwd(arrs, verts, meta, charge_fn, field=False, mem_limit=jnp.inf, remat=False,
                 return_counts=False, m2l_mixed=False, charge_args=()):
    r"""MODULE-LEVEL jitted fulljit forward: the one trace cache all rebuilds share.
    ``charge_args`` is a TRACED pytree (extra ``charge_fn`` inputs, e.g. magnetisation)."""
    return element_quadtree_fulljit_eval(_fulljit_join(meta, arrs), verts, charge_fn,
                                         field, mem_limit, remat, return_counts, m2l_mixed,
                                         charge_args)


@partial(jax.jit, static_argnames=["meta", "charge_fn", "loss_fn", "geom_fn", "field",
                                   "mem_limit", "remat", "m2l_mixed", "charge_args_fn"])
def _fulljit_vg(arrs, x, extra, meta, charge_fn, loss_fn, geom_fn, field=False,
                mem_limit=jnp.inf, remat=False, m2l_mixed=False, charge_args_fn=None):
    r"""MODULE-LEVEL jitted value_and_grad through the fulljit forward (counts as aux) --
    the guarded driver's gradient path, retrace-free across rebuilds for the same
    (loss_fn, geom_fn, charge_args_fn, meta, shapes). ``charge_args_fn(x)`` extracts the
    traced ``charge_fn`` extras from the design variable, so the gradient flows through
    BOTH the geometry and e.g. the magnetisation."""
    def f(x_):
        v = geom_fn(x_) if geom_fn is not None else x_
        cargs = charge_args_fn(x_) if charge_args_fn is not None else ()
        out, counts = element_quadtree_fulljit_eval(_fulljit_join(meta, arrs), v, charge_fn,
                                                    field, mem_limit, remat,
                                                    return_counts=True, m2l_mixed=m2l_mixed,
                                                    charge_args=tuple(cargs))
        return loss_fn(out, x_, *extra), counts
    return jax.value_and_grad(f, has_aux=True)(x)


def element_quadtree_fulljit_compile(s, charge_fn, field=False, mem_limit=jnp.inf, remat=False,
                                     return_counts=False, m2l_mixed=False):
    r"""
    Compile :func:`element_quadtree_fulljit_eval` into a single jitted
    ``fwd(verts, *charge_args) -> u/H`` (``charge_args`` are traced extras for
    ``charge_fn``, e.g. the magnetisation).

    The forward recomputes the capped topology from ``verts`` each call -- so a shape-opt
    loop runs with NO per-rebuild recompile (compile once, then every iteration and its
    ``jax.grad`` reuse the same executable; a fixed-topology forward would have to
    rebuild+recompile as the geometry drifts).

    The setup is passed to a MODULE-LEVEL jitted forward as (static meta, array pytree)
    (:func:`_fulljit_split`), NOT closed over: re-running
    :func:`element_quadtree_fulljit_setup` after geometry drift and compiling again
    re-uses the previous executable whenever the refit caps land on the same granularity
    bins -- the rebuild then costs only the eager setup probe, no retrace, no compile.
    ``charge_fn`` must be the SAME function object across rebuilds for that reuse (define
    it once, not inline per rebuild).

    For moving-geometry loops prefer :func:`jaxfmm.driver.element_quadtree_fulljit_driver`
    -- the guarded wrapper around this forward that certifies every call against the
    realised buffer counts and rebuilds transparently; this raw compile silently truncates
    on cap overflow (validate with :func:`element_quadtree_fulljit_check`).

    Parameters
    ----------
    s : dict
        Setup dict from :func:`element_quadtree_fulljit_setup`.
    charge_fn : callable
        ``charge_fn(verts) -> (rho, sig)``, the P0 volume / P1 surface densities. Must
        be the SAME function object across rebuilds for executable reuse.
    field : bool, optional
        Return the vector field instead of the potential.
    mem_limit : float, optional
        Byte budget for the batched transforms (default unlimited).
    remat : bool, optional
        Rematerialise the near field in the backward pass.
    return_counts : bool, optional
        Have the compiled forward additionally return the realised buffer occupancies.
    m2l_mixed : bool, optional
        Use the mixed-precision M2L.

    Returns
    -------
    callable
        ``fwd(verts) -> (N_eval,) or (N_eval, 3) jnp.ndarray`` (with
        ``return_counts=True``: ``fwd(verts) -> (out, counts)``), backed by the
        module-level jitted forward.
    """
    meta, arrs = _fulljit_split(s)

    def fwd(verts, *charge_args):
        return _fulljit_fwd(arrs, verts, meta, charge_fn, field, mem_limit, remat,
                            return_counts, m2l_mixed, tuple(charge_args))
    return fwd



@partial(jax.jit, static_argnames=["skel", "field", "mem_limit", "remat", "m2l_mixed"])
def _equad_frozen_fwd(arrs, vol_chrg, surf_nodal, skel, field, mem_limit, remat,
                      m2l_mixed=False):
    r"""MODULE-LEVEL jitted frozen quad-tree evaluator (setup as (skeleton, arrays):
    geometries with repeating array shapes share this executable)."""
    from jaxfmm.util import join_static
    return element_quadtree_eval(join_static(skel, arrs), vol_chrg, surf_nodal,
                                 field=field, mem_limit=mem_limit, remat=remat,
                                 m2l_mixed=m2l_mixed)


def element_quadtree_compile(s, field=False, mem_limit=jnp.inf, remat=False, m2l_mixed=False):
    r"""
    FROZEN-geometry evaluator for the quad-tree element path.

    Compacts the (sentinel-padded) interaction lists once and returns a jitted
    ``fmm(vol_chrg, surf_nodal) -> u/H`` -- the solver-loop form when only the
    charges/magnetisation change (the topology is never rebuilt; contrast the fulljit
    path, which rebuilds per call for SHAPE differentiation).

    Setup-as-arguments + GRANULAR list shapes: the compacted lists are padded up to
    granularity multiples and the setup rides through a module-level jitted forward as
    (static skeleton, array pytree) -- two geometries whose tree shapes match (same
    ``pad_mesh`` bucket -> same balanced-tree box counts) re-use ONE executable, so the
    per-geometry first-call compile collapses to the eager setup cost in remesh /
    multi-geometry loops.

    Parameters
    ----------
    s : dict
        Setup dict from :func:`element_quadtree_setup`.
    field : bool, optional
        Return the vector field instead of the potential.
    mem_limit : float, optional
        Byte budget for the batched transforms (default unlimited).
    remat : bool, optional
        Rematerialise the near field in the backward pass.
    m2l_mixed : bool, optional
        Use the mixed-precision M2L.

    Returns
    -------
    callable
        ``fmm(vol_chrg, surf_nodal) -> (N_eval,) or (N_eval, 3) jnp.ndarray``, the
        jitted frozen-geometry evaluator.
    """
    from jaxfmm.util import split_static
    hier = s["hier"]
    nb_e = int(hier["eval_boxcenters"].shape[0]); nb_s = int(hier["boxcenters"].shape[0])
    hier = {**hier, "mpl_cnct": _compact_pairs(hier["mpl_cnct"], nb_e, nb_s, gran=4096),
            "dir_cnct": _compact_pairs(hier["dir_cnct"], nb_e, nb_s, gran=1024)}
    if hier.get("m2l_grp") is not None:
        hier = {**hier, "m2l_grp": _quantize_snap_grp(hier["m2l_grp"])}
    skel, arrs = split_static({**s, "hier": hier})

    def fmm(vol_chrg, surf_nodal):
        return _equad_frozen_fwd(arrs, vol_chrg, surf_nodal, skel, field, mem_limit, remat,
                                 m2l_mixed)
    return fmm
