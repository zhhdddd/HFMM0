r"""Flexible-tree FMM engine (rotation-based solid-harmonics on the adaptive quad-tree).

The default point-charge Laplace engine: per-level rectangular boxes with a theta-MAC,
rotation-accelerated M2M/M2L/L2L, and the shared near field from :mod:`jaxfmm.near`.
Entry points: :func:`setup` / :func:`evaluate_setup` / :func:`compile_evaluator`
(verb symmetry with the other engines) and the jitted :func:`eval_potential`.
"""
import jax
import jax.numpy as jnp
from jaxfmm.kernels.laplace import M2M, M2L, L2L, M2L_grouped, M2L_mixed
from jaxfmm.kernels.laplace import eval_regular_basis, regular_grad_matrix
from jaxfmm.trees import handle_padding
import numpy as np
from jaxfmm.near import eval_direct, eval_direct_binned, ws_limit, m2l_budget   # near field lives in jaxfmm.near; re-exported here for back-compat
from functools import partial

__all__ = ["setup", "evaluate_setup", "compile_evaluator",
           "eval_potential", "eval_potential_direct", "eval_direct", "eval_direct_binned",
           "compile_point_evaluator", "get_initial_mpls", "get_initial_mpls_elements", "eval_local", "m2m", "l2l"]

@partial(jax.jit, static_argnames=['p', 'mem_limit', 'remat'])
def get_initial_mpls(padded_pts, padded_chrgs, boxcenters, p, mem_limit = jnp.inf, remat = False):
    r"""
    Initial multipole expansions for each box on the highest level (S2M).

    Parameters
    ----------
    padded_pts : (nbox, w, 3) array
        Padded per-box source points.
    padded_chrgs : (nbox, w) array
        Padded per-box charges (0 on padding).
    boxcenters : (nbox, 3) array
        Expansion centres.
    p : int
        Expansion order.
    mem_limit : float
        Working-set budget in bytes for the batched ``lax.map``.

    Returns
    -------
    (nbox, (p+1)^2) array
        Multipole coefficients per box.
    """
    mem = padded_chrgs.size * (p+1)**2 * padded_chrgs.dtype.itemsize
    batch_size = int((mem_limit/mem) * boxcenters.shape[0]) + 1 if mem > mem_limit else boxcenters.shape[0]

    def impl_body(args):
        padded_pts_loc, padded_chrgs_loc, boxcenters_loc = args
        dist = padded_pts_loc - boxcenters_loc
        return (eval_regular_basis(dist,p) * padded_chrgs_loc).sum(axis=0)
    # remat: reverse-mode through the batched lax.map otherwise STACKS the
    # per-batch solid-harmonic basis (and its recurrence intermediates) as
    # residuals -- at ~1e7 sources that is a parade of GB-scale buffers and the
    # dominant adjoint cost (measured: the differentiable shape-opt OOM at
    # 2.24M tets). Checkpointing the body recomputes the basis per batch in
    # the backward pass instead; only effective together with a finite
    # mem_limit (a single monolithic batch has nothing to recompute in parts).
    body = jax.checkpoint(impl_body) if remat else impl_body
    return jax.lax.map(body, [padded_pts, padded_chrgs[...,None], boxcenters[:,None,:]], batch_size=batch_size)

@partial(jax.jit, static_argnames=['p', 'mem_limit'])
def get_initial_mpls_elements(boxcenters, idcs, qpos, qchg, p, idx_offset = 0, mem_limit = jnp.inf, remat = False):
    r"""
    Initial leaf-box multipole expansions from element quadrature charges.

    S2M for a finite-element source tree.  Each element contributes
    :math:`\int_{elem}\rho\,R_n^m\,dV`, evaluated by quadrature -> the quadrature points
    act as point charges, so this reduces to :func:`get_initial_mpls` with per-leaf
    gathered quadrature points.

    Parameters
    ----------
    boxcenters : (n_leaf, 3) array
        Leaf-box centres (from the element tree).
    idcs : (n_leaf, cap) array
        Element indices per leaf (tree padding uses out-of-range entries).  When one
        tree holds several element types, ``idcs`` carries *global* indices; pass
        ``idx_offset`` = the start of this type's block so local indices are
        ``idcs - idx_offset`` -- entries outside ``[0, N_elem)`` (other types, padding)
        contribute zero, so the per-type multipoles simply sum (volume + surface are
        additive).
    qpos : (N_elem, n_q, 3) array
        Quadrature points.
    qchg : (N_elem, n_q) array
        Quadrature charges (see :func:`jaxfmm.fem.quadrature.element_quad_charges`).
    p : int
        Expansion order.
    idx_offset : int
        Start of this element type's global index block.
    mem_limit : float
        Working-set budget in bytes.

    Returns
    -------
    (n_leaf, (p+1)^2) array
        Multipole coefficients per leaf box.
    """
    nbox, cap = idcs.shape
    nq, M = qpos.shape[1], qpos.shape[0]
    if M == 0:                                               # disabled element family: XLA gathers
        pos = jnp.zeros((nbox, 1, 3), qpos.dtype)            # cannot slice a zero-length dim ->
        chg = jnp.zeros((nbox, 1), qchg.dtype)               # one zero charge per box (exact zeros)
        return get_initial_mpls(pos, chg, boxcenters, p, mem_limit, remat)
    loc = idcs - idx_offset
    loc = jnp.where((loc >= 0) & (loc < M), loc, M)          # out-of-range -> sentinel (mode=fill); avoids negative-wrap
    pos = qpos.at[loc].get(mode="fill", fill_value=0.0).reshape((nbox, cap * nq, 3))
    chg = qchg.at[loc].get(mode="fill", fill_value=0.0).reshape((nbox, cap * nq))
    return get_initial_mpls(pos, chg, boxcenters, p, mem_limit, remat)

# Level scans (M2M/L2L) pad every level to the finest level's box count: levels x n_fin work
# for one compiled body. Above this finest-level size the per-level loop wins outright
# (interior point tree, 2.4e6 finest boxes, p=6, H100: M2M 574-617 ms -> 88-97 ms, 2026-10-09);
# the element tree at 1e7 (1.3e5 leaves) keeps the scan (M2M <= 10 ms, cheaper compile).
LEVEL_SCAN_MAX_BOXES = 2**17


@partial(jax.jit, static_argnames=['mem_limit', 'src_ofs', 'max_src_lvl', 'level_scan'])
def m2m(coeff, boxcenters, src_ofs, max_src_lvl, unqs, invs, mem_limit = jnp.inf, level_scan = True):
    r"""
    M2M: aggregate leaf multipoles up to the root (physics-upward; walks the box
    array FORWARD since coarser levels sit earlier in it).

    With ``level_scan=True`` (default) the per-level M2M sweep runs as a single
    ``lax.scan`` over levels with every level padded to the finest-level box count.
    This compiles the (expensive) M2M body once instead of once per level — much less
    compile time — at the cost of a little extra M2M compute on the padded coarse
    levels (the cheap pass).  Padded boxes carry zero multipoles and only feed the
    discarded trailing parents, so the real output is bit-exact.  ``level_scan=False``
    falls back to the plain Python level-loop (one M2M trace per level); mixed
    per-level child counts force that fallback automatically.

    Parameters
    ----------
    coeff : (n_finest, (p+1)^2) array
        Finest-source-level multipoles (from :func:`get_initial_mpls`);
        ``n_finest = src_ofs[max_src_lvl+1] - src_ofs[max_src_lvl]``.
    boxcenters : (n_src_boxes, 3) array
        Source box centres, all levels concatenated (level ``l`` occupies
        ``src_ofs[l]:src_ofs[l+1]``).
    src_ofs : tuple of int
        Per-level offsets into the concatenated centre array (static).
    max_src_lvl : int
        Finest source level (static).
    unqs : list of jnp.ndarray
        Wigner rotation tables from :func:`jaxfmm.kernels.laplace.precompute_combs`.
    invs : list of jnp.ndarray
        Index tables into ``unqs`` (same origin).
    mem_limit : float
        Working-set budget in bytes for the batched M2M kernel.
    level_scan : bool
        Run the sweep as a single ``lax.scan`` (default, see above); ``False`` uses
        the plain Python level-loop.

    Returns
    -------
    (n_src_boxes, (p+1)^2) array
        Multipole coefficients for every source box (root through finest level).
    """
    if level_scan and max_src_lvl > 1:              # the scan pads all levels to the finest
        ratios = {(src_ofs[i+1] - src_ofs[i]) // (src_ofs[i] - src_ofs[i-1])
                  for i in range(1, max_src_lvl + 1)}
        if len(ratios) > 1:                         # per-level split schedule: mixed child
            level_scan = False                      # counts break the uniform scan layout
        elif src_ofs[max_src_lvl+1] - src_ofs[max_src_lvl] > LEVEL_SCAN_MAX_BOXES:
            level_scan = False                      # padding every level to the finest costs levels x n_fin
    if not level_scan:
        mpls = jnp.zeros((src_ofs[max_src_lvl+1],coeff.shape[1]), coeff.dtype)
        mpls = mpls.at[src_ofs[max_src_lvl]:src_ofs[max_src_lvl+1]].set(coeff)
        for i in range(max_src_lvl,0,-1):
            mpls = mpls.at[src_ofs[i-1]:src_ofs[i]].set(M2M(mpls[src_ofs[i]:src_ofs[i+1]],boxcenters[src_ofs[i]:src_ofs[i+1]],boxcenters[src_ofs[i-1]:src_ofs[i]], unqs, invs, mem_limit))
        return mpls
    if max_src_lvl == 0:
        return coeff
    F = coeff.shape[1]
    n_fin_c = src_ofs[max_src_lvl+1] - src_ofs[max_src_lvl]      # finest box count (8^max)
    n_chi = n_fin_c // (src_ofs[max_src_lvl] - src_ofs[max_src_lvl-1])
    n_fin_p = n_fin_c // n_chi
    # stacked, finest-padded child/parent centres for each descending step (host slicing on static ofs)
    child_cs, par_cs = [], []
    for k in range(max_src_lvl):
        i = max_src_lvl - k
        cc, pc = boxcenters[src_ofs[i]:src_ofs[i+1]], boxcenters[src_ofs[i-1]:src_ofs[i]]
        child_cs.append(jnp.pad(cc, ((0, n_fin_c - cc.shape[0]), (0, 0))))
        par_cs.append(jnp.pad(pc, ((0, n_fin_p - pc.shape[0]), (0, 0))))
    child_cs, par_cs = jnp.stack(child_cs), jnp.stack(par_cs)
    cur0 = jnp.zeros((n_fin_c, F), coeff.dtype).at[:coeff.shape[0]].set(coeff)
    def body(cur, xs):
        cc, pc = xs
        par = M2M(cur, cc, pc, unqs, invs, mem_limit)           # (n_fin_p, F)
        return jnp.zeros((n_fin_c, F), par.dtype).at[:n_fin_p].set(par), par
    _, ys = jax.lax.scan(body, cur0, (child_cs, par_cs))         # ys[k] = parents of level (max-k-1)
    parts = [None] * (max_src_lvl + 1)
    parts[max_src_lvl] = coeff
    for k in range(max_src_lvl):
        ip = max_src_lvl - k - 1
        parts[ip] = ys[k, :src_ofs[ip+1] - src_ofs[ip]]
    return jnp.concatenate(parts, axis=0)

@partial(jax.jit, static_argnames=['lvl_info', 'mem_limit', 'trg_ofs', 'level_scan'])
def l2l(locs, eval_boxcenters, trg_ofs, unqs, invs, lvl_info, mem_limit = jnp.inf, level_scan = True):
    r"""
    L2L: propagate local expansions from the root down to the leaves (physics-downward;
    walks the box array BACKWARD).

    With ``level_scan=True`` (default) the per-level L2L sweep runs as a single
    ``lax.scan`` over levels padded to the finest-level box count, compiling the L2L
    body once instead of once per level (see :func:`m2m`).  Bit-exact; padded
    parents/children only produce discarded trailing rows.  ``level_scan=False`` uses
    the plain Python level-loop.

    Parameters
    ----------
    locs : (n_eval_boxes, (p+1)^2) array
        Per-box local expansions (the M2L output), all eval levels concatenated
        (level ``j`` occupies ``trg_ofs[j]:trg_ofs[j+1]``).
    eval_boxcenters : (n_eval_boxes, 3) array
        Eval box centres, all levels concatenated.
    trg_ofs : tuple of int
        Per-level offsets into the concatenated eval centre array (static).
    unqs : list of jnp.ndarray
        Wigner rotation tables from :func:`jaxfmm.kernels.laplace.precompute_combs`.
    invs : list of jnp.ndarray
        Index tables into ``unqs`` (same origin).
    lvl_info : tuple of tuple
        The traversed ``(l_eval, l_src)`` sequence from the connectivity build;
        ``lvl_info[-2][0]`` is the highest eval level needed (static).
    mem_limit : float
        Working-set budget in bytes for the batched L2L kernel.
    level_scan : bool
        Run the sweep as a single ``lax.scan`` (default); ``False`` uses the plain
        Python level-loop.

    Returns
    -------
    (n_eval_boxes, (p+1)^2) array
        Accumulated local expansions after the downward sweep (levels above the
        highest eval level pass through unchanged).
    """
    l_eval = lvl_info[-2][0]    # highest eval level needed
    if level_scan and l_eval > 1:                   # mixed child counts break the scan layout
        ratios = {(trg_ofs[j+2] - trg_ofs[j+1]) // (trg_ofs[j+1] - trg_ofs[j])
                  for j in range(l_eval)}
        if len(ratios) > 1 or trg_ofs[l_eval+1] - trg_ofs[l_eval] > LEVEL_SCAN_MAX_BOXES:
            level_scan = False
    if not level_scan:
        for j in range(l_eval):
            locs = locs.at[trg_ofs[j+1]:trg_ofs[j+2]].add(L2L(locs[trg_ofs[j]:trg_ofs[j+1]], eval_boxcenters[trg_ofs[j]:trg_ofs[j+1]], eval_boxcenters[trg_ofs[j+1]:trg_ofs[j+2]], unqs, invs, mem_limit))
        return locs
    if l_eval == 0:
        return locs
    F = locs.shape[1]
    n_fin = trg_ofs[l_eval+1] - trg_ofs[l_eval]                  # finest box count (8^l_eval)
    n_chi = n_fin // (trg_ofs[l_eval] - trg_ofs[l_eval-1])
    n_fin_pa = n_fin // n_chi
    # stacked, finest-padded parent/child centres + the M2L local at each child level
    par_cs, child_cs, m2l = [], [], []
    for j in range(l_eval):
        pc, cc = eval_boxcenters[trg_ofs[j]:trg_ofs[j+1]], eval_boxcenters[trg_ofs[j+1]:trg_ofs[j+2]]
        ml = locs[trg_ofs[j+1]:trg_ofs[j+2]]
        par_cs.append(jnp.pad(pc, ((0, n_fin_pa - pc.shape[0]), (0, 0))))
        child_cs.append(jnp.pad(cc, ((0, n_fin - cc.shape[0]), (0, 0))))
        m2l.append(jnp.pad(ml, ((0, n_fin - ml.shape[0]), (0, 0))))
    par_cs, child_cs, m2l = jnp.stack(par_cs), jnp.stack(child_cs), jnp.stack(m2l)
    car0 = jnp.zeros((n_fin, F), locs.dtype).at[:trg_ofs[1] - trg_ofs[0]].set(locs[trg_ofs[0]:trg_ofs[1]])
    def body(carry, xs):
        pc, cc, ml = xs
        full = L2L(carry[:n_fin_pa], pc, cc, unqs, invs, mem_limit) + ml   # (n_fin, F)
        return full, full
    _, ys = jax.lax.scan(body, car0, (par_cs, child_cs, m2l))    # ys[j] = full local at level j+1
    parts = [locs[trg_ofs[0]:trg_ofs[1]]] + [ys[j, :trg_ofs[j+2] - trg_ofs[j+1]] for j in range(l_eval)]
    parts += [locs[trg_ofs[lv]:trg_ofs[lv+1]] for lv in range(l_eval+1, len(trg_ofs)-1)]  # levels above l_eval: unchanged
    return jnp.concatenate(parts, axis=0)

@partial(jax.jit, static_argnames=['p', 'field', 'mem_limit'])
def eval_local(locs, padded_eval_pts, rev_idcs, boxcenters, p, field = False, mem_limit = jnp.inf):
    r"""
    Evaluate local expansions at the padded eval points on the highest level (L2P).

    Parameters
    ----------
    locs : (nbox, (p+1)^2) array
        Local expansions of the highest-level (finest) eval boxes.
    padded_eval_pts : (nbox, w, 3) array
        Padded per-box eval points.
    rev_idcs : (N,) int array
        Flat index of each eval point in the padded layout (restores the original
        particle order).
    boxcenters : (nbox, 3) array
        Expansion centres.
    p : int
        Expansion order.
    field : bool
        Evaluate the field (negative gradient) instead of the potential.
    mem_limit : float
        Working-set budget in bytes for the batched ``lax.map``.

    Returns
    -------
    (N,) or (N, 3) array
        Potential (or field, ``field=True``) in original particle order.
    """
    mem = rev_idcs.size * (p+1)**2 * locs.dtype.itemsize
    batch_size = int((mem_limit/mem) * locs.shape[0]) + 1 if mem > mem_limit else locs.shape[0]
    if field:   # grad of a degree-p local = degree-(p-1) local with coefficients D^T L, per box
        D = jnp.asarray(regular_grad_matrix(p), locs.dtype)
        locs = jnp.einsum("cjm,bm->bcj", D, locs)               # (nbox, 3, p^2)
    def evloc_body(args):
        pts_loc, box_loc, loc_loc = args
        if(field):
            return jnp.einsum("wj,cj->wc", eval_regular_basis(pts_loc - box_loc, p-1), loc_loc[0])
        return (loc_loc * eval_regular_basis(pts_loc - box_loc,p)).sum(axis=-1)
    padded_res = jax.lax.map(evloc_body, [padded_eval_pts, boxcenters[:,None], locs[:,None]], batch_size=batch_size)
    if(field):
        return -padded_res.reshape((-1,3))[rev_idcs] / (4*jnp.pi)
    else:
        return padded_res.flatten()[rev_idcs] / (4*jnp.pi)

@partial(jax.jit, static_argnames=['field', 'lvl_info', 'mem_limit', 'src_ofs', 'trg_ofs', 'level_scan', 'remat', 'm2l_mixed', 'p2p_mixed'])
def eval_potential(chrgs, pts, eval_pts, idcs, boxcenters, eval_boxcenters, src_ofs, trg_ofs, mpl_cnct, dir_cnct, unqs, invs, lvl_info, img_cnct, PBC_op, field = False, mem_limit = jnp.inf, level_scan = True, remat = False, m2l_grp = None, m2l_mixed = None, p2p_mixed = None, img_phase = None, fmm_geom_scale = 1.0, **kwargs):
    r"""
    Evaluate the potential/field via FMM.

    Requires an array of charge values and the hierarchy information generated by
    ``gen_hierarchy`` (splat the dict).

    Parameters
    ----------
    chrgs : (N,) array
        Source charges.
    pts, eval_pts, idcs, boxcenters, eval_boxcenters, src_ofs, trg_ofs, mpl_cnct, \
    dir_cnct, unqs, invs, lvl_info, img_cnct, PBC_op
        Hierarchy members from ``gen_hierarchy``.
    field : bool
        Evaluate the field (negative gradient) instead of the potential.
    mem_limit : float
        Working-set budget in bytes for the batched phases.
    level_scan : bool
        Run the M2M/L2L level sweeps (:func:`m2m`/:func:`l2l`) as fixed-size
        ``lax.scan``s, compiling those bodies once instead of once per level (much
        faster compile, bit-exact).  ``False`` uses the plain Python level-loops.
    remat : bool
        Rematerialise the near field in the backward pass.
    m2l_grp : optional
        From :func:`jaxfmm.kernels.laplace.m2l_theta_setup` (open boundaries only):
        selects the theta-snapped GROUPED M2L — shared per-class rotation GEMMs instead
        of per-pair Wigner builds, at a controlled snap error ~0.1/K (kept below
        truncation by the recommended K).
    m2l_mixed, p2p_mixed : bool, optional
        Mixed-precision M2L/P2P; ``None`` selects the automatic defaults (see Notes).

    Returns
    -------
    (N,) or (N, 3) array
        Potential (or field) at the eval points.

    Notes
    -----
    Automatic mixed-precision defaults are **dtype-aware**: "mixed" means an f32 kernel
    with f64 accumulation, which is a downcast win only when the inputs are f64. For f64
    charges the P2P mixed kernel (BOX-LOCAL coordinates) is on for ``p <= 6`` (its ~3e-7
    noise floor is p-independent and would bind beyond) and the mixed M2L
    is enabled for ``p <= 4`` (f32 phase noise below truncation; at higher p keep f64 or
    opt in explicitly). For f32 charges both default OFF — the kernel is already f32 and
    the f64 accumulation is pure overhead (measured +31% on a 16.5M-tet strayfield eval
    at unchanged error).
    """
    if(len(lvl_info) == 1): # only direct interactions - compute potential directly
        if img_cnct.shape[1] != 0:   # periodic + direct-only would silently drop the images
            raise NotImplementedError(
                "eval_potential: direct-only hierarchy (no multipole levels) with PBC — the "
                "open-BC direct sum would silently ignore the periodic images. Use a deeper "
                "tree (smaller N_max) so the far field exists.")
        return eval_potential_direct(pts,chrgs,eval_pts,field)
    p = invs[-1].shape[0] - 1
    if img_phase is not None:                     # Bloch: complex charges, plain (complex) M2L;
        if m2l_grp is not None:                   # the P2P handles complex charges under mixed
            raise ValueError("Bloch phases (img_phase) are incompatible with the grouped M2L")
        m2l_mixed = False
    f64_run = jnp.dtype(chrgs.dtype) in (jnp.float64, jnp.complex128)
    if p2p_mixed is None:
        # "mixed" = f32 kernel (BOX-LOCAL coords) + f64 accumulation. That is a
        # DOWNCAST win only in f64 runs; in f32 runs the kernel is already f32, so
        # mixed just adds f64-accumulate overhead (measured +31% on a 16.5M-tet
        # strayfield eval, error unchanged at 6e-7) — auto only for f64 inputs, and only
        # up to p=6: its ~3e-7 noise floor is p-independent and reaches the truncation
        # error at p=8/theta=0.4 (measured 2e4 random pts: mixed 3.5e-7 vs 2.0e-7 exact;
        # at p=12/theta=0.3 3.1e-7 vs 2.8e-11). Pass p2p_mixed=False for reference grade.
        p2p_mixed = f64_run and p <= 6
    if m2l_mixed is None:
        # mixed M2L only where its f32 phase noise sits below truncation (p<=4) AND
        # the run is f64 (same downcast-vs-upcast asymmetry as p2p_mixed; at f32 it
        # is speed-neutral at best)
        m2l_mixed = f64_run and p <= 4
    pad_arr = handle_padding(pts, chrgs, eval_pts, idcs, kwargs.get("pad_geom"))
    # mem_limit (0.3.1 semantics) is the PEAK-TRANSIENT budget: route it whole to the
    # materialising P2P (whose mem model prices the (batch, w, w, 3) tensor), and route
    # the derived gathered-row working set (budget / 6*maxleaf = the historical
    # L2-anchored operating point) to the transient-priced MINOR stages -- bigger is
    # NOT better there (at high p the small M2L batches measurably win).
    ws = ws_limit(mem_limit, pad_arr[3].shape[1]) if np.isfinite(mem_limit) else mem_limit
    coeff = get_initial_mpls(pad_arr[0], pad_arr[1], boxcenters[src_ofs[lvl_info[-2][1]]:src_ofs[lvl_info[-2][1]+1]], p, ws, remat)
    coeff = m2m(coeff, boxcenters, src_ofs, lvl_info[-2][1], unqs, invs, ws, level_scan)
    loc_init = PBC_op@coeff[0]    # PBC far field
    if m2l_grp is not None:       # theta-snapped grouped M2L (no PBC images)
        coeff = M2L_grouped(coeff, boxcenters, eval_boxcenters, m2l_grp, ws)
    elif m2l_mixed:               # f32 rotations + folded translation, f64 accumulate
        # M2L batch budget (jaxfmm.near.m2l_budget): the L2-anchored working set is right
        # for the P2P rows but fragments the M2L into many scan steps that each scatter
        # into the full locals carry; batch-count capped at p<=4, `ws` at high p.
        coeff = M2L_mixed(coeff, boxcenters, eval_boxcenters, mpl_cnct, unqs, invs, img_cnct,
                          m2l_budget(ws, mem_limit, p, mpl_cnct.shape[0], coeff.dtype.itemsize))
    else:
        coeff = M2L(coeff, boxcenters, eval_boxcenters, mpl_cnct, unqs, invs, img_cnct,
                    m2l_budget(ws, mem_limit, p, mpl_cnct.shape[0], coeff.dtype.itemsize), remat, img_phase=img_phase)
    coeff = coeff.at[0].add(loc_init.astype(coeff.dtype))
    coeff = l2l(coeff, eval_boxcenters, trg_ofs, unqs, invs, lvl_info, ws, level_scan)
    del_, dsl = lvl_info[-1][0], lvl_info[-1][1]             # direct (near) levels: eval, src
    res = eval_local(coeff[trg_ofs[-2]:trg_ofs[-1]], pad_arr[2], idcs[1][1], eval_boxcenters[trg_ofs[-2]:trg_ofs[-1]], p, field, ws) + \
          eval_direct(pad_arr[3], pad_arr[4], pad_arr[5], idcs[3][1], dir_cnct, img_cnct, field, mem_limit, remat,
                      mixed=p2p_mixed, centres=boxcenters[src_ofs[dsl]:src_ofs[dsl+1]],
                      eval_centres=eval_boxcenters[trg_ofs[del_]:trg_ofs[del_+1]], img_phase=img_phase)
    # auto-rescaled hierarchy: restore the caller's units (no-op at scale 1;
    # `field` is static, the scale may be traced -- keep this branch-free)
    return res / (fmm_geom_scale ** 2 if field else fmm_geom_scale)

@partial(jax.jit, static_argnames=["skel", "field", "m2l_mixed", "p2p_mixed"])
def _frozen_point_fwd(arrs, chrgs, skel, field, m2l_mixed=None, p2p_mixed=None):
    r"""MODULE-LEVEL jitted frozen-geometry point evaluator (setup-as-arguments)."""
    from jaxfmm.util import join_static
    return eval_potential(chrgs, **join_static(skel, arrs), field=field, m2l_mixed=m2l_mixed, p2p_mixed=p2p_mixed)


def compile_point_evaluator(hierarchy, field=False, mem_limit=None, m2l_mixed=None, p2p_mixed=None):
    r"""FROZEN-geometry evaluator on the flexible tree.

    One jitted ``fmm(charges)`` over a :func:`jaxfmm.trees.gen_hierarchy` dict — the
    solver-loop form when only the charges change (the counterpart of
    :func:`jaxfmm.kifmm.compile_evaluator`).

    Parameters
    ----------
    hierarchy : dict
        From :func:`jaxfmm.trees.gen_hierarchy` (or :func:`setup`).
    field : bool
        Evaluate the field instead of the potential.
    mem_limit : float, optional
        Override the hierarchy's batch budget.
    m2l_mixed : bool, optional
        Mixed-precision M2L (f32 rotations, f64 accumulate — see
        :func:`jaxfmm.kernels.laplace.transf.M2L_mixed`).  ``None`` = AUTO: on for
        p<=4, where the end-to-end error is unchanged (f32 phase noise below
        truncation); off for p>=5, where it bumps the error mildly — pass True to
        opt in.
    p2p_mixed : bool, optional
        Mixed-precision near field (``None`` = the :func:`eval_potential` default).

    Returns
    -------
    callable
        ``fmm(charges) -> pot/field``.

    Notes
    -----
    Setup-as-arguments matters here: a naive ``jax.jit(lambda q: eval_potential(q,
    **h))`` BAKES the hierarchy arrays as constants — enormous compiles at large N and
    an executable that can exceed the compilation cache's serialization limit.  As
    arguments the executable is shared across same-shape hierarchies and caches
    persistently."""
    from jaxfmm.util import split_static
    h = dict(hierarchy)
    if mem_limit is not None:
        from jaxfmm.near import hint_legacy_mem_limit
        hint_legacy_mem_limit(mem_limit)
        h["mem_limit"] = float(mem_limit)
    skel, arrs = split_static(h)
    arrs = tuple(jnp.asarray(a) for a in arrs)   # DEVICE-PUT ONCE: hierarchy dicts can
    # carry host (numpy) members, and passing those as jit arguments re-uploads AND
    # re-pins them EVERY call (host registration dominates the eval with the GPU idle)
    return lambda chrgs: _frozen_point_fwd(arrs, chrgs, skel, field, m2l_mixed, p2p_mixed)


@partial(jax.jit, static_argnames=['field'])
def eval_potential_direct(pts, chrgs, eval_pts = None, field = False):
    r"""
    Evaluate the potential directly via pairwise sums.

    Parameters
    ----------
    pts : (N, 3) array
        Point positions.
    chrgs : (N,) array
        Point charges.
    eval_pts : (M, 3) array, optional
        Points to evaluate the potential at.  Defaults to ``pts``.
    field : bool
        Evaluate the field (negative gradient) instead of the potential.

    Returns
    -------
    (M,) or (M, 3) array
        Electrostatic potential (or field) of the points and corresponding charges
        (self-interactions excluded).
    """
    if(eval_pts is None):
        eval_pts = pts
    res = jnp.zeros(eval_pts.shape[:field+1], chrgs.dtype)
    def eval_direct_body(i, val):
        distsvec = pts[:,:] - eval_pts[i,None,:]
        inv_dists = jnp.linalg.norm(distsvec,axis=-1)
        # take out self-interaction / numerically coincident points (ULP-relative floor:
        # an exact `== 0` test misses points arithmetic left one ULP apart)
        inv_dists = 1/jnp.where(inv_dists > 16 * jnp.finfo(inv_dists.dtype).eps
                                * jnp.max(inv_dists), inv_dists, jnp.inf)
        if(field):
            val = val.at[i].set(-((chrgs * inv_dists**3)[:,None] * distsvec).sum(axis=0))
        else:
            val = val.at[i].set((chrgs * inv_dists).sum())
        return val
    return jax.lax.fori_loop(0,eval_pts.shape[0],eval_direct_body,res)/(4*jnp.pi)

# ── one-call verb symmetry with jaxfmm.kifmm / jaxfmm.uniform ──────────────────────────
def setup(positions, eval_pts=None, **kwargs):
    r"""One-call FLEXIBLE-engine setup (verb symmetry with ``kifmm.setup``/``uniform_fmm.setup``).

    A thin wrapper around :func:`jaxfmm.trees.gen_hierarchy`; all its knobs pass
    through (``N_max``, ``theta``, ``p``, ``s``, ``periodic_axes``,
    ``domain=(pmin, extent)`` — the same unit-cell convention as the other
    engines — ...).

    Parameters
    ----------
    positions : (N, 3) array
        Positions of the N point charges.
    eval_pts : (M, 3) array, optional
        Evaluation points; defaults to ``positions``.
    **kwargs
        Forwarded to :func:`jaxfmm.trees.gen_hierarchy` (``N_max``, ``theta``, ``p``,
        ``s``, ``periodic_axes``, ...).

    Returns
    -------
    dict
        The hierarchy dict for :func:`evaluate_setup` / :func:`compile_evaluator`.
    """
    from jaxfmm.trees import gen_hierarchy
    return gen_hierarchy(positions, eval_pts, **kwargs)


def evaluate_setup(h, chrgs, field=False, **kwargs):
    r"""Potential (or field) for the fixed hierarchy ``h`` (:func:`setup`) and new ``chrgs`` —
    the solver-loop entry, ``eval_potential(chrgs, **h)`` (cf. ``kifmm.evaluate_setup``).

    Parameters
    ----------
    h : dict
        Hierarchy dict from :func:`setup` / :func:`jaxfmm.trees.gen_hierarchy`.
    chrgs : (N,) array
        Source charges.
    field : bool
        Evaluate the field (negative gradient) instead of the potential.
    **kwargs
        Forwarded to :func:`eval_potential` (``mem_limit``, ``level_scan``, ``remat``,
        ``m2l_mixed``, ``p2p_mixed``, ...).

    Returns
    -------
    (M,) or (M, 3) array
        Potential (or field, ``field=True``) at the eval points.
    """
    return eval_potential(chrgs, **h, field=field, **kwargs)


#: frozen jitted ``fmm(chrgs)`` over a hierarchy dict — the historical name is
#: :func:`compile_point_evaluator`; this alias completes the cross-engine verb symmetry.
compile_evaluator = compile_point_evaluator
