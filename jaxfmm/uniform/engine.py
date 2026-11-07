r"""
jaxfmm.uniform — spherical-harmonic FMM on a 2:1-balanced UNIFORM grid with PRECOMPUTED
translation operators (no per-pair rotations).

Companion to the nonuniform rotation quad-tree (:mod:`jaxfmm.trees.flexible` /
:mod:`jaxfmm.kernels.laplace` / :mod:`jaxfmm.fem`). The orchestration lives in
:mod:`jaxfmm.core`; this module supplies the precomputed-spherical-harmonic
REPRESENTATION: :func:`make_operators` bundles the per-phase functions into a
:class:`jaxfmm.core.CubicOperators` set (the back-compat :func:`uniform_eval` wraps
tree + factory + :func:`jaxfmm.core.evaluate`).

Idea: on a regular cubic octree the M2M/M2L/L2L relative offsets are a finite set, so
the full multipole/local translation operators are PRECOMPUTED once per (level, offset)
by composing the existing rotation+translation pipeline on the identity basis. Runtime
translations are then plain matmuls. S2M / L2P / P2P reuse the existing solid-harmonic
operators (get_initial_mpls / eval_local / eval_direct) per box on the uniform tree's
particles_in_leaf + U-list. The tree + interaction lists come from the 2:1-balanced
uniform builder (:mod:`jaxfmm.trees.cubic`).

Eval points different from sources (strayfield: sources = quad points, eval = mesh nodes)
are handled the KIFMM way: include the eval points as ZERO-charge particles and read off
their potential.
"""
import jax, jax.numpy as jnp, numpy as np
from functools import partial
from jax.scipy.special import factorial as fac
from jaxfmm.kernels.laplace import rot_azimuth, rot_polar, polar_rot_base, precompute_combs
from jaxfmm.util import cart_to_sph
from jaxfmm.kernels.laplace import mpl_idx, inv_mpl_idx, eval_singular_basis
from jaxfmm.flex import get_initial_mpls, eval_local
from jaxfmm.near import (near_pairs, near_pairs_binned,
                         predicted_p2p_speedup, auto_mem_limit, autotune_mem_limit)   # near field lives in jaxfmm.near; re-exported for jaxfmm.fem's _ufmm.<name>
from jaxfmm.near import eval_direct, eval_direct_binned, _build_padded  # noqa: F401  (re-exports: fem + tests use them via jaxfmm.uniform)
from jaxfmm.trees import _morton_to_grid, box_geometry   # box_geometry shared in trees
from jaxfmm import core   # shared cubic driver

__all__ = ["setup", "evaluate_setup", "compile_evaluator",
           "box_geometry", "precompute_operators", "make_operators", "uniform_eval", "m2l_level_ranges",
           "auto_mem_limit", "autotune_mem_limit", "near_pairs", "near_pairs_binned", "predicted_p2p_speedup"]

# Morton octant ordering: child k centre = parent centre + child_offsets[k] * h_child
_CHILD_OFF = jnp.array([(-1, -1, -1), (1, -1, -1), (-1, 1, -1), (1, 1, -1),
                        (-1, -1, 1), (1, -1, 1), (-1, 1, 1), (1, 1, 1)], float)   # default float width (no x32 truncation warning)



# ── per-pair transforms (copied from transf.py M2M/M2L/L2L), used ONLY to precompute operators ──────
def _m2l_apply(rvec, coeffs, unqs, invs, p):
    r, theta, phi = cart_to_sph(rvec); base = polar_rot_base(unqs, theta, p)
    coeffs = rot_azimuth(coeffs, phi); coeffs = rot_polar(unqs, invs, theta, coeffs, base=base)
    for j in range(p + 1):
        ns = jnp.arange(j, p + 1); pws = ns[:, None] + ns[None, :]
        f = (-1) ** (ns[:, None] + j) * fac(pws)[None, ...] / (r[..., None, None] ** ((pws + 1)[None, ...]))
        idxs = mpl_idx(j, ns); coeffs = coeffs.at[..., idxs].set((f * coeffs[..., None, idxs]).sum(-1))
        if j > 0:
            idxs = mpl_idx(-j, ns); coeffs = coeffs.at[..., idxs].set((f * coeffs[..., None, idxs]).sum(-1))
    coeffs = rot_polar(unqs, invs, theta, coeffs, True, True, base=base)
    return rot_azimuth(coeffs, phi, True, True)

def _m2m_apply(rvec, coeffs, unqs, invs, p):
    r, theta, phi = cart_to_sph(rvec); base = polar_rot_base(unqs, theta, p)
    coeffs = rot_azimuth(coeffs, phi); coeffs = rot_polar(unqs, invs, theta, coeffs, base=base)
    new = coeffs
    for j in range(1, p + 1):
        f = (r ** j) / fac(j); ms, ns = inv_mpl_idx(jnp.arange((p - j + 1) ** 2))
        idxs = ms + ns + (ns + j) ** 2 + j
        new = new.at[..., idxs].add(f[..., None] * coeffs[..., :mpl_idx(p - j, p - j) + 1])
    new = rot_polar(unqs, invs, theta, new, True, base=base)
    return rot_azimuth(new, phi, True)

def _l2l_apply(rvec, coeffs, unqs, invs, p):
    r, theta, phi = cart_to_sph(rvec); base = polar_rot_base(unqs, theta, p)
    coeffs = rot_azimuth(coeffs, phi, False, True); coeffs = rot_polar(unqs, invs, theta, coeffs, False, True, base=base)
    new = coeffs
    for j in range(1, p + 1):
        f = (r ** j) / fac(j); ms, ns = inv_mpl_idx(jnp.arange((p - j + 1) ** 2))
        idxs = ms + ns + (ns + j) ** 2 + j
        new = new.at[..., :mpl_idx(p - j, p - j) + 1].add(f[..., None] * coeffs[..., idxs])
    new = rot_polar(unqs, invs, theta, new, True, True, base=base)
    return rot_azimuth(new, phi, True, True)


@partial(jax.jit, static_argnames=["applyfn", "p"])
def _ops_from(applyfn, rvecs, unqs, invs, p):
    r"""(n, n_c, n_c) operators with local = op @ mpl, from per-offset physical displacements rvecs.
    Jitted (applyfn/p static): the eager rotation pipeline is dispatch-dominated per call,
    so this leaves one cacheable compile per operator type, reused across every level
    (and across processes via the persistent cache)."""
    n_c = (p + 1) ** 2; n = rvecs.shape[0]
    rv = jnp.repeat(jnp.asarray(rvecs, jnp.float64), n_c, axis=0)
    basis = jnp.tile(jnp.eye(n_c), (n, 1))
    out = applyfn(rv, basis, unqs, invs, p)                            # out[o*n_c+k, i] = op[o,i,k]
    return out.reshape(n, n_c, n_c).transpose(0, 2, 1)


def _disp_table():
    r"""343 displacement vectors in disp_id = (dx+3)*49+(dy+3)*7+(dz+3) order; mask of far (V-list) ones."""
    d = np.array([(dx, dy, dz) for dx in range(-3, 4) for dy in range(-3, 4) for dz in range(-3, 4)])
    far = (np.abs(d).max(1) >= 2)                                      # 2 <= |o|_inf <= 3
    return d, far


def precompute_operators(p, domain_extent, max_depth):
    r"""
    Precompute per-level translation operators.

    Geometry-independent given ``p``, ``domain_extent`` and ``max_depth``.

    Parameters
    ----------
    p : int
        Multipole order.
    domain_extent : float
        Root box side length.
    max_depth : int
        Tree depth.

    Returns
    -------
    dict
        ``M2M``/``L2L``: (max_depth+1, 8, n_c, n_c); ``M2L``: (max_depth+1, 343, n_c, n_c)
        (near offsets zeroed); plus the rotation tables ``unqs``/``invs`` and ``p``.
    """
    unqs, invs = precompute_combs(p)
    disp, far = _disp_table()
    M2M, L2L, M2L = [], [], []
    for l in range(max_depth + 1):
        h_child = domain_extent * 0.5 ** (l + 2)                       # child half-side (parent at level l)
        spacing = domain_extent * 0.5 ** l                             # box spacing at level l (= 2*h_l)
        rv_child = _CHILD_OFF * h_child                                # (8,3) child centre - parent centre
        M2M.append(_ops_from(_m2m_apply, rv_child, unqs, invs, p))
        L2L.append(_ops_from(_l2l_apply, rv_child, unqs, invs, p))
        rv_off = jnp.where(jnp.asarray(far)[:, None], jnp.asarray(disp, jnp.float64) * spacing, 1.0)  # avoid r=0
        ops = _ops_from(_m2l_apply, rv_off, unqs, invs, p)
        ops = jnp.where(jnp.asarray(far)[:, None, None], ops, 0.0)     # near offsets unused -> zero
        M2L.append(ops)
    return dict(M2M=jnp.stack(M2M), L2L=jnp.stack(L2L), M2L=jnp.stack(M2L), unqs=unqs, invs=invs, p=p)


# ── padded per-box arrays from particles_in_leaf ────────────────────────────────────────────────────
@partial(jax.jit, static_argnames=["max_depth"])
def _m2m(mpls, m_id, p_id, lvls, M2M_ops, max_depth):
    r"""Bottom-up M2M with the precomputed per-(level, octant) operators (8-octant loop)."""
    octant = (m_id % jnp.uint64(8)).astype(jnp.int32)

    def merge(i, mpls):
        child_lvl = max_depth - i; parent_lvl = child_lvl - 1
        is_child = lvls == child_lvl
        for k in range(8):
            contrib = mpls @ M2M_ops[parent_lvl, k].T
            contrib = jnp.where((is_child & (octant == k))[:, None], contrib, 0.0)
            mpls = mpls.at[p_id].add(contrib)
        return mpls
    return jax.lax.fori_loop(0, max_depth - 1, merge, mpls)


@partial(jax.jit, static_argnames=["max_depth"])
def _m2l_offset(mpls, V_list, m_id, lvls, M2L_ops, max_depth):
    r"""Offset-batched M2L (avoids the per-pair operator gather of :func:`_m2l`). Builds, ONCE,
    ``src_idx[disp_id, target] = source`` from the V-list (parity-correct: only valid V-pairs populate
    it). Then for each level l, scans the 343 displacements applying the SINGLE precomputed operator
    M2L_ops[l, d] (one matmul per (level, displacement) -- no per-pair operator gather), masked to
    level-l targets. Bit-identical to :func:`_m2l`; memory traffic is the mpls gather (no big operator
    gather)."""
    max_boxes, n_c = mpls.shape
    gx, gy, gz = (g.astype(jnp.int32) for g in _morton_to_grid(m_id))
    tgt, src = V_list[:, 0], V_list[:, 1]
    valid = (tgt >= 0) & (src >= 0); tgt_s = jnp.where(valid, tgt, 0); src_s = jnp.where(valid, src, 0)
    if V_list.shape[1] == 3:                        # periodic 3-col V: unwrap the displacement
        from jaxfmm.trees.cubic import _IMG_COMBOS  # (the image box's multipole == the real
        io = jnp.asarray(_IMG_COMBOS, jnp.int32)[jnp.where(valid, V_list[:, 2], 0)]   # box's,
        grid = (2 ** lvls[src_s]).astype(jnp.int32)                                   # so only
        dx = gx[tgt_s] - (gx[src_s] + io[:, 0] * grid)                                # disp_id
        dy = gy[tgt_s] - (gy[src_s] + io[:, 1] * grid)                                # changes)
        dz = gz[tgt_s] - (gz[src_s] + io[:, 2] * grid)
    else:
        dx = gx[tgt_s] - gx[src_s]; dy = gy[tgt_s] - gy[src_s]; dz = gz[tgt_s] - gz[src_s]
    disp_id = (dx + 3) * 49 + (dy + 3) * 7 + (dz + 3)
    flat = jnp.where(valid, disp_id * max_boxes + tgt_s, 343 * max_boxes)          # (disp,tgt) -> src, built ONCE
    src_idx = jnp.full((343 * max_boxes,), max_boxes, jnp.int32).at[flat].set(
        src_s.astype(jnp.int32), mode="drop").reshape(343, max_boxes)
    mpls_aug = jnp.concatenate([mpls, jnp.zeros((1, n_c), mpls.dtype)])            # sentinel row -> 0
    gathered = mpls_aug[src_idx]                                                  # (343, max_boxes, n_c) gathered ONCE
    local = jnp.zeros((max_boxes, n_c), mpls.dtype)
    # open BC has provably no level-1 V pairs (all 8 boxes adjacent); under PBC the level-1
    # boxes DO have wrapped V partners (images of each other) -> start the loop at 1.
    l_start = 1 if V_list.shape[1] == 3 else 2
    for l in range(l_start, max_depth + 1):                                       # one einsum per level (no scan)
        contrib = jnp.einsum("dik,dbk->bi", M2L_ops[l], gathered)                 # sum over all displacements
        local = local + jnp.where((lvls == l)[:, None], contrib, 0.0)
    return local


def m2l_level_ranges(lvls, max_depth):
    r"""Static per-level ``(start, size)`` of the box array for levels 2..max_depth. ``build_fmm_tree``
    lays boxes out CONTIGUOUSLY by level, so the M2L einsum can run on each level's slice instead of the
    full box array + mask (cf. :func:`_m2l_offset`, which does ~depth x redundant work since most boxes
    sit at the finest level). Returns a hashable tuple for :func:`_m2l_sliced`, or ``None`` if the layout
    is not level-contiguous (then fall back to :func:`_m2l_offset`).

    Parameters
    ----------
    lvls : (max_boxes,) int array
        Per-box levels (host numpy or device array; inspected on the host).
    max_depth : int
        Tree depth.

    Returns
    -------
    tuple of (int, int), or None
        ``(start, size)`` of each level's contiguous box slice for levels 2..max_depth
        (``(0, 0)`` for empty levels) — hashable, so usable as a static jit argument;
        ``None`` if the box array is not level-contiguous.
    """
    lv = np.asarray(lvls); out = []
    for L in range(2, max_depth + 1):
        idx = np.where(lv == L)[0]
        if idx.size == 0:
            out.append((0, 0)); continue
        if int(idx.max() - idx.min() + 1) != int(idx.size):
            return None
        out.append((int(idx.min()), int(idx.size)))
    return tuple(out)


@partial(jax.jit, static_argnames=["max_depth", "level_ranges"])
def _m2l_sliced(mpls, V_list, m_id, lvls, M2L_ops, max_depth, level_ranges):
    # NOTE: level_ranges covers levels 2..max_depth — periodic (3-col) V-lists have level-1
    # pairs and must use _m2l_offset (uniform_eval routes them there).
    r"""Per-level-SLICED M2L: same offset-batched src gather as :func:`_m2l_offset`, but the einsum runs
    only on each level's contiguous box slice (``level_ranges`` from :func:`m2l_level_ranges`), removing
    the ~depth x redundant full-box einsum. Bit-identical to :func:`_m2l_offset`, just faster."""
    max_boxes, n_c = mpls.shape
    gx, gy, gz = (g.astype(jnp.int32) for g in _morton_to_grid(m_id))
    tgt, src = V_list[:, 0], V_list[:, 1]
    valid = (tgt >= 0) & (src >= 0); tgt_s = jnp.where(valid, tgt, 0); src_s = jnp.where(valid, src, 0)
    if V_list.shape[1] == 3:                        # periodic 3-col V: unwrap the displacement
        from jaxfmm.trees.cubic import _IMG_COMBOS  # (the image box's multipole == the real
        io = jnp.asarray(_IMG_COMBOS, jnp.int32)[jnp.where(valid, V_list[:, 2], 0)]   # box's,
        grid = (2 ** lvls[src_s]).astype(jnp.int32)                                   # so only
        dx = gx[tgt_s] - (gx[src_s] + io[:, 0] * grid)                                # disp_id
        dy = gy[tgt_s] - (gy[src_s] + io[:, 1] * grid)                                # changes)
        dz = gz[tgt_s] - (gz[src_s] + io[:, 2] * grid)
    else:
        dx = gx[tgt_s] - gx[src_s]; dy = gy[tgt_s] - gy[src_s]; dz = gz[tgt_s] - gz[src_s]
    disp_id = (dx + 3) * 49 + (dy + 3) * 7 + (dz + 3)
    flat = jnp.where(valid, disp_id * max_boxes + tgt_s, 343 * max_boxes)
    src_idx = jnp.full((343 * max_boxes,), max_boxes, jnp.int32).at[flat].set(
        src_s.astype(jnp.int32), mode="drop").reshape(343, max_boxes)
    mpls_aug = jnp.concatenate([mpls, jnp.zeros((1, n_c), mpls.dtype)])
    gathered = mpls_aug[src_idx]                                                  # (343, max_boxes, n_c)
    local = jnp.zeros((max_boxes, n_c), mpls.dtype)
    for i, l in enumerate(range(2, max_depth + 1)):
        st, sz = level_ranges[i]
        if sz == 0:
            continue
        contrib = jnp.einsum("dik,dbk->bi", M2L_ops[l], gathered[:, st:st + sz, :])  # only this level's boxes
        local = local.at[st:st + sz].set(contrib)
    return local


@partial(jax.jit, static_argnames=["max_depth"])
def _l2l(local, m_id, p_id, lvls, L2L_ops, max_depth):
    r"""Top-down L2L with the precomputed per-(level, octant) operators (8-octant loop)."""
    octant = (m_id % jnp.uint64(8)).astype(jnp.int32)

    def push(i, local):
        child_lvl = i + 1
        parent_local = local[p_id]
        for k in range(8):
            contrib = parent_local @ L2L_ops[i, k].T          # L2L_ops[l] is for parent level l (=i)
            mask = (lvls == child_lvl) & (octant == k)
            local = jnp.where(mask[:, None], local + contrib, local)
        return local
    return jax.lax.fori_loop(1, max_depth, push, local)


@partial(jax.jit, static_argnames=["max_depth", "level_ranges"])
def _m2m_sliced(mpls, m_id, p_id, M2M_ops, max_depth, level_ranges):
    r"""Batched + per-level-sliced M2M: replaces the 8-octant loop (8 full-box matmuls + 8 masked
    scatters per level) with ONE per-box-operator gather + batched matvec over each level's contiguous
    slice, then one scatter. Bit-identical to :func:`_m2m`, much faster in both forward and reverse
    mode (the small per-octant ops were the reverse-mode cost). ``level_ranges`` from
    :func:`m2l_level_ranges`."""
    octant = (m_id % jnp.uint64(8)).astype(jnp.int32)
    for child_lvl in range(max_depth, 1, -1):                            # bottom-up: finest children first
        st, sz = level_ranges[child_lvl - 2]
        if sz == 0:
            continue
        op_l = M2M_ops[child_lvl - 1][octant[st:st + sz]]                # (sz, n_c, n_c) per-box operator
        contrib = jnp.einsum("bij,bj->bi", op_l, mpls[st:st + sz])
        mpls = mpls.at[p_id[st:st + sz]].add(contrib)                    # accumulate into parents
    return mpls


@partial(jax.jit, static_argnames=["max_depth", "level_ranges"])
def _l2l_sliced(local, m_id, p_id, L2L_ops, max_depth, level_ranges):
    r"""Batched + per-level-sliced L2L (top-down companion of :func:`_m2m_sliced`). Bit-identical to
    :func:`_l2l`; collapses the 8-octant loop to one per-box-operator matvec per level slice."""
    octant = (m_id % jnp.uint64(8)).astype(jnp.int32)
    for child_lvl in range(2, max_depth + 1):                            # top-down: coarsest children first
        st, sz = level_ranges[child_lvl - 2]
        if sz == 0:
            continue
        op_l = L2L_ops[child_lvl - 1][octant[st:st + sz]]                # L2L_ops[parent_lvl] = child_lvl-1
        contrib = jnp.einsum("bij,bj->bi", op_l, local[p_id[st:st + sz]])
        local = local.at[st:st + sz].add(contrib)
    return local


@partial(jax.jit, static_argnames=["n_pairs", "batch_size"])
def _m2l(mpls, V_list, m_id, lvls, M2L_ops, n_pairs=None, batch_size=2048):
    r"""Per-pair M2L over the V-list (one operator matvec per pair, ``jax.lax.map``-batched).
    The AD-light form: its backward stores per-batch residuals only."""
    max_boxes, n_c = mpls.shape
    gx, gy, gz = (g.astype(jnp.int32) for g in _morton_to_grid(m_id))
    tgt, src = V_list[:, 0], V_list[:, 1]
    if n_pairs is not None:
        ok = (tgt >= 0) & (src >= 0)
        dst = jnp.where(ok, jnp.cumsum(ok.astype(jnp.int32)) - 1, n_pairs)
        tgt = jnp.full((n_pairs,), -1, jnp.int32).at[dst].set(tgt, mode="drop")
        src = jnp.full((n_pairs,), -1, jnp.int32).at[dst].set(src, mode="drop")
    valid = (tgt >= 0) & (src >= 0)
    tgt_s = jnp.where(valid, tgt, 0); src_s = jnp.where(valid, src, 0)
    dx = gx[tgt_s] - gx[src_s]; dy = gy[tgt_s] - gy[src_s]; dz = gz[tgt_s] - gz[src_s]
    disp_id = jnp.where(valid, (dx + 3) * 49 + (dy + 3) * 7 + (dz + 3), 0)
    lvl_t = lvls[tgt_s]

    def per_pair(args):
        lt, did, s_idx, ok = args
        return jnp.where(ok, M2L_ops[lt, did] @ mpls[s_idx], 0.0)
    contribs = jax.lax.map(per_pair, (lvl_t, disp_id, src_s, valid), batch_size=batch_size)
    scatter = jnp.where(valid, tgt_s, max_boxes)
    return jnp.zeros((max_boxes, n_c), mpls.dtype).at[scatter].add(contribs, mode="drop")


@partial(jax.jit, static_argnames=["p", "n_pairs", "batch_size"])
def _m2p(mpls, centres, ppts, rev, W_list, p, n_pairs=None, batch_size=2048):
    r"""W-list M2P: evaluate each source box's multipole directly at the target leaf's particles."""
    max_boxes, N_max = ppts.shape[0], ppts.shape[1]
    tgt, src = W_list[:, 0], W_list[:, 1]
    if n_pairs is not None:
        ok = (tgt >= 0) & (src >= 0); dst = jnp.where(ok, jnp.cumsum(ok.astype(jnp.int32)) - 1, n_pairs)
        tgt = jnp.full((n_pairs,), -1, jnp.int32).at[dst].set(tgt, mode="drop")
        src = jnp.full((n_pairs,), -1, jnp.int32).at[dst].set(src, mode="drop")
    valid = (tgt >= 0) & (src >= 0); tgt_s = jnp.where(valid, tgt, 0); src_s = jnp.where(valid, src, 0)

    def per_pair(args):
        t, s, okv = args
        S = eval_singular_basis(ppts[t] - centres[s], p)          # (N_max, n_c)
        return jnp.where(okv, (S * mpls[s]).sum(-1), 0.0)         # (N_max,)
    contribs = jax.lax.map(per_pair, (tgt_s, src_s, valid), batch_size=batch_size)
    pot = jnp.zeros((max_boxes, N_max), mpls.dtype).at[jnp.where(valid, tgt_s, max_boxes)].add(contribs, mode="drop")
    return (pot / (4 * jnp.pi)).flatten()[rev]


@partial(jax.jit, static_argnames=["p", "n_pairs", "batch_size"])
def _p2l(ppts, pchg, centres, X_list, p, n_pairs=None, batch_size=2048):
    r"""X-list P2L: deposit each source leaf's particles onto the target box's local expansion."""
    max_boxes, n_c = centres.shape[0], (p + 1) ** 2
    tgt, src = X_list[:, 0], X_list[:, 1]
    if n_pairs is not None:
        ok = (tgt >= 0) & (src >= 0); dst = jnp.where(ok, jnp.cumsum(ok.astype(jnp.int32)) - 1, n_pairs)
        tgt = jnp.full((n_pairs,), -1, jnp.int32).at[dst].set(tgt, mode="drop")
        src = jnp.full((n_pairs,), -1, jnp.int32).at[dst].set(src, mode="drop")
    valid = (tgt >= 0) & (src >= 0); tgt_s = jnp.where(valid, tgt, 0); src_s = jnp.where(valid, src, 0)

    def per_pair(args):
        t, s, okv = args
        S = eval_singular_basis(ppts[s] - centres[t], p)          # (N_max, n_c)
        return jnp.where(okv, (S * pchg[s][:, None]).sum(0), 0.0)     # (n_c,)
    contribs = jax.lax.map(per_pair, (tgt_s, src_s, valid), batch_size=batch_size)
    return jnp.zeros((max_boxes, n_c), ppts.dtype).at[jnp.where(valid, tgt_s, max_boxes)].add(contribs, mode="drop")


def make_operators(ops, V_list, W_list=None, X_list=None, m2l_n_pairs=None, w_n_pairs=None,
                   x_n_pairs=None, batch_size=2048, mem_limit=jnp.inf, offset_m2l=True,
                   level_ranges=None, pbc=None):
    r"""
    Precomputed-spherical-harmonic operator set for the shared cubic driver.

    Consumed by :func:`jaxfmm.core.evaluate`. Closures over the SETUP artifacts only (the
    per-level operator tables from :func:`precompute_operators`, the V/W/X lists,
    ``level_ranges`` from :func:`m2l_level_ranges`); each phase keeps ``uniform_eval``'s
    selection ladder: ``level_ranges`` picks the batched+sliced M2M/M2L/L2L,
    ``offset_m2l=False`` drops M2L to the per-pair form (AD-light backward).
    ``W_list``/``X_list`` enable the adaptive cross-level phases.

    Parameters
    ----------
    ops : dict
        Per-level operator tables from :func:`precompute_operators`
        (``M2M``/``L2L``/``M2L``, rotation tables ``unqs``/``invs``, ``p``).
    V_list : (n_v, 2) or (n_v, 3) int array
        V-list pairs ``(tgt, src)`` (3-col ``(tgt, src, img)`` under PBC); rows with
        negative entries are padding.
    W_list : (n_w, 2) int array, optional
        Cross-level W-list ``(tgt leaf, src box)`` for the M2P phase (adaptive trees).
    X_list : (n_x, 2) int array, optional
        Cross-level X-list ``(tgt box, src leaf)`` for the P2L phase (adaptive trees).
    m2l_n_pairs, w_n_pairs, x_n_pairs : int, optional
        Static compacted pair counts for the per-pair M2L / W / X phases (the padded
        lists are compacted to this length before the batched ``lax.map``).
    batch_size : int
        ``lax.map`` batch size for the per-pair phases (default 2048).
    mem_limit : float
        Working-set budget in bytes for the batched S2M/L2P phases (default ``inf``).
    offset_m2l : bool
        Offset-batched M2L (default True); ``False`` selects the per-pair form
        (AD-light backward).
    level_ranges : tuple of (int, int), optional
        Per-level ``(start, size)`` slices from :func:`m2l_level_ranges`; selects the
        batched+sliced M2M/M2L/L2L.
    pbc : dict, optional
        Periodic far-field bundle from :func:`jaxfmm.pbc.gen_pbc_uniform`
        (``PBC_op``, ``M2M_root``, ``L2L_root``); enables the root-level PBC phase.

    Returns
    -------
    jaxfmm.core.CubicOperators
        The bundled per-phase functions.
    """
    p = ops["p"]

    def s2m_fn(src, tree):
        from jaxfmm.near import ws_limit
        import numpy as _np                                                           # transient-priced minor
        ws = ws_limit(mem_limit, src.ppts.shape[1]) if _np.isfinite(mem_limit) else mem_limit   # stages take the row
        return get_initial_mpls(src.ppts, src.pchg, src.centres, p, ws)              # S2M   working set (0.3.1)

    def m2m_fn(mpls, tree):
        if level_ranges is not None:                                                 # M2M (batched+sliced)
            return _m2m_sliced(mpls, tree.m_id, tree.p_id, ops["M2M"], tree.max_depth, level_ranges)
        return _m2m(mpls, tree.m_id, tree.p_id, tree.lvls, ops["M2M"], tree.max_depth)  # M2M (bottom-up, 8-octant)

    def m2l_fn(mpls, tree):
        if offset_m2l and level_ranges is not None:                                  # M2L (sliced: fastest)
            return _m2l_sliced(mpls, V_list, tree.m_id, tree.lvls, ops["M2L"], tree.max_depth, level_ranges)
        if offset_m2l:                                                               # M2L (V, offset-batched)
            return _m2l_offset(mpls, V_list, tree.m_id, tree.lvls, ops["M2L"], tree.max_depth)
        return _m2l(mpls, V_list, tree.m_id, tree.lvls, ops["M2L"], n_pairs=m2l_n_pairs, batch_size=batch_size)  # M2L (per-pair: AD-light bwd)

    def l2l_fn(local, tree):
        if level_ranges is not None:                                                 # L2L (batched+sliced)
            return _l2l_sliced(local, tree.m_id, tree.p_id, ops["L2L"], tree.max_depth, level_ranges)
        return _l2l(local, tree.m_id, tree.p_id, tree.lvls, ops["L2L"], tree.max_depth)  # L2L (top-down, 8-octant)

    def l2p_fn(local, src, tree):
        from jaxfmm.near import ws_limit
        import numpy as _np
        ws = ws_limit(mem_limit, src.ppts.shape[1]) if _np.isfinite(mem_limit) else mem_limit
        return eval_local(local, src.ppts, src.rev, src.centres, p, False, ws)        # L2P (row working set, 0.3.1)

    p2l_fn = None
    if X_list is not None:
        def p2l_fn(local, src, tree):                                                # P2L (X, cross-level coarse->local)
            return local + _p2l(src.ppts, src.pchg, src.centres, X_list, p,
                                n_pairs=x_n_pairs, batch_size=batch_size)

    m2p_fn = None
    if W_list is not None:
        def m2p_fn(mpls, src, tree):                                                 # M2P (W, cross-level fine->particles)
            return _m2p(mpls, src.centres, src.ppts, src.rev, W_list, p,
                        n_pairs=w_n_pairs, batch_size=batch_size)

    pbc_fn = None
    if pbc is not None:
        # periodic far tail at the VIRTUAL ROOT (harmonic basis is native here): aggregate
        # the 8 level-1 boxes into the cell multipole (pbc["M2M_root"], octant = level-1
        # m_id), apply the flexible engine's lattice operator, distribute the cell local
        # back with pbc["L2L_root"]. Bundle from jaxfmm.pbc.gen_pbc_uniform.
        PBC_m = pbc["PBC_op"]; M2Mr = pbc["M2M_root"]; L2Lr = pbc["L2L_root"]

        def pbc_fn(dn, up, tree):
            is_l1 = (tree.lvls == 1)
            cell_mpl = jnp.zeros((PBC_m.shape[1],), up.dtype)
            idxs, press = [], []
            for k in range(8):
                hit = is_l1 & (tree.m_id == jnp.uint64(k))
                idx = jnp.argmax(hit); pres = jnp.any(hit).astype(up.dtype)
                idxs.append(idx); press.append(pres)
                cell_mpl = cell_mpl + pres * (M2Mr[k] @ up[idx])
            loc0 = PBC_m @ cell_mpl
            for k in range(8):
                dn = dn.at[idxs[k]].add(press[k] * (L2Lr[k] @ loc0))
            return dn

    return core.CubicOperators(s2m=s2m_fn, m2m=m2m_fn, m2l=m2l_fn, l2l=l2l_fn,
                               l2p=l2p_fn, p2l=p2l_fn, m2p=m2p_fn, pbc=pbc_fn)


def uniform_eval(positions, charges, m_id, p_id, lvls, is_leaf, pil, V_list, dir_cnct,
                 domain_extent, pmin, ops, max_depth, W_list=None, X_list=None,
                 m2l_n_pairs=None, w_n_pairs=None, x_n_pairs=None, batch_size=2048, mem_limit=jnp.inf,
                 p2p_mixed=False, remat=False, offset_m2l=True, level_ranges=None, p2p_bins=None,
                 pbc=None, img_cnct=jnp.array([[]])):
    r"""
    Full uniform-grid spherical-harmonic FMM potential at every particle.

    Back-compat wrapper around the shared cubic driver: builds a
    :class:`jaxfmm.core.CubicTree` + :func:`make_operators` and calls
    :func:`jaxfmm.core.evaluate`. ``dir_cnct`` is the precomputed near-field pair list
    from :func:`near_pairs` (so this stays jittable).

    Parameters
    ----------
    positions : (N, 3) array
        Source/eval points.
    charges : (N,) array
        Source charges.
    m_id : (max_boxes,) uint64 array
        Morton ids of the boxes.
    p_id : (max_boxes,) int32 array
        Parent box index of each box.
    lvls : (max_boxes,) int32 array
        Box levels.
    is_leaf : (max_boxes,) bool array
        Leaf flags.
    pil : (max_boxes, N_max) int32 array
        Particles-in-leaf indices, -1 padded (from :func:`jaxfmm.trees.assign_particles`).
    V_list : (n_v, 2) or (n_v, 3) int array
        V-list pairs ``(tgt, src)`` (3-col ``(tgt, src, img)`` under PBC); rows with
        negative entries are padding.
    dir_cnct : (n_near, 2) int32 array
        Near pair list from :func:`near_pairs` (or the flat ``dir_cnct_flat`` from
        :func:`near_pairs_binned` when ``p2p_bins`` is given).
    domain_extent : float
        Root box side length.
    pmin : (3,) array
        Root-box min corner.
    ops : dict
        Per-level operator tables from :func:`precompute_operators`.
    max_depth : int
        Tree depth (static).
    W_list : (n_w, 2) int array, optional
        Cross-level W-list for the M2P phase; required (with ``X_list``) on adaptive trees.
    X_list : (n_x, 2) int array, optional
        Cross-level X-list for the P2L phase.
    m2l_n_pairs, w_n_pairs, x_n_pairs : int, optional
        Static compacted pair counts for the per-pair M2L / W / X phases.
    batch_size : int
        ``lax.map`` batch size for the per-pair phases (default 2048).
    mem_limit : float
        Working-set budget in bytes for the batched S2M/L2P/P2P phases (default ``inf``).
    p2p_mixed : bool
        float32 near-field kernel with float64 accumulation (default False).
    remat : bool
        Rematerialise the near field in the backward pass (see Notes).
    offset_m2l : bool
        Offset-batched M2L (default True); ``False`` = per-pair form (AD-light backward).
    level_ranges : tuple of (int, int), optional
        Per-level slices from :func:`m2l_level_ranges` (see Notes).
    p2p_bins : tuple, optional
        ``(bin_offsets, bin_widths)`` static tuples from :func:`near_pairs_binned`
        (see Notes).
    pbc : dict, optional
        Periodic far-field bundle from :func:`jaxfmm.pbc.gen_pbc_uniform`.
    img_cnct : (27, 3) array, optional
        Periodic image offsets for the near field (from
        :func:`jaxfmm.trees.cubic.cubic_img_table`); the empty default means open
        boundaries.

    Returns
    -------
    (N,) array
        Potential at every particle (far field + near field), in input order.

    Notes
    -----
    ``remat=True`` rematerialises the P2P near field in the backward (batches +
    recomputes each batch instead of storing every pair's pairwise tensor as a residual)
    -- essential for ``jax.grad`` at scale.

    ``level_ranges`` (from :func:`m2l_level_ranges`) selects the faster per-level-sliced
    M2L.

    ``p2p_bins`` (the ``(bin_offsets, bin_widths)`` STATIC tuples from
    :func:`near_pairs_binned`, with ``dir_cnct`` then the matching ``dir_cnct_flat``)
    selects the occupancy-binned near field (:func:`jaxfmm.near.eval_direct_binned`) --
    bit-exact, faster P2P when leaf occupancy is uneven. Requires the padded-point width
    (``pil``/``N_max``) to be >= the largest tier width.
    """
    # adaptive-tree guard: non-empty leaves on MULTIPLE levels require the W/X cross-level
    # phases — without them interactions are silently DROPPED. Checkable only when the tree
    # arrays are concrete (the usual jit-closure pattern); traced args skip the check.
    if W_list is None and X_list is None:
        try:
            # np.asarray on the UNTOUCHED closure constants works during tracing; slicing
            # first (pil[:, 0]) would lift them to tracers and defeat the check.
            lv, il = np.asarray(lvls), np.asarray(is_leaf)
            p0 = np.asarray(pil)[:, 0]
        except Exception:                       # tracer args -> cannot inspect here
            lv = None
        if lv is not None and len(set(lv[il & (p0 >= 0)].tolist())) > 1:
            raise ValueError(
                "uniform_eval: ADAPTIVE tree (non-empty leaves on multiple levels) without "
                "W_list/X_list — cross-level interactions would be silently dropped. Build "
                "them via jaxfmm.trees.build_wx_lists and pass W_list/X_list (see "
                "tests/test_core.py), or use kifmm.setup which handles this automatically.")
    if level_ranges is not None and V_list.shape[1] == 3:
        level_ranges = None                        # periodic V has level-1 pairs: offset path
    tree = core.CubicTree(m_id, p_id, lvls, is_leaf, pil, domain_extent, pmin, max_depth)
    op_set = make_operators(ops, V_list, W_list=W_list, X_list=X_list, m2l_n_pairs=m2l_n_pairs,
                            w_n_pairs=w_n_pairs, x_n_pairs=x_n_pairs, batch_size=batch_size,
                            mem_limit=mem_limit, offset_m2l=offset_m2l, level_ranges=level_ranges,
                            pbc=pbc)
    return core.evaluate(positions, charges, tree, op_set, dir_cnct, p2p_bins=p2p_bins,
                         p2p_mixed=p2p_mixed, mem_limit=mem_limit, remat=remat,
                         img_cnct=img_cnct)


# ── one-call setup / evaluate (mirrors jaxfmm.kifmm.setup) ──────────────────────────────────────
def setup(positions, p=3, N_max=128, max_depth=None, mem_limit=None, p2p_mixed=True,
          bin_gate=1.2, bin_K=16, batch_size=2048, periodic_axes=(), domain=None):
    r"""
    One-call uniform (precomputed-spherical-harmonic) FMM setup.

    The shared cubic-tree assembly (:func:`jaxfmm.core.setup_geometry` — tree, eager
    U/V/W/X lists, gated near field) plus this engine's pieces (padded particle layout,
    per-level operator tables, ``m2l_level_ranges`` for the sliced M2L, the PBC bundle
    under ``periodic_axes``). Evaluate repeatedly with :func:`evaluate_setup` /
    :func:`compile_evaluator` (fixed tree, changing charges). Potential-only (``field``:
    use :func:`jaxfmm.kifmm.setup`).

    Parameters
    ----------
    positions : (N, 3) array
        Source/eval point cloud.
    p : int
        Multipole order (default 3).
    N_max : int
        Leaf-splitting threshold (particles per box, default 128).
    max_depth : int, optional
        Cap on the tree depth; ``None`` lets the builder choose.
    mem_limit : float, optional
        Working-set budget in bytes for the batched phases; ``None`` picks
        :func:`jaxfmm.near.auto_mem_limit`.
    p2p_mixed : bool
        float32 near-field kernel with float64 accumulation (default True).
    bin_gate : float
        Threshold on :func:`jaxfmm.near.predicted_p2p_speedup` above which the
        occupancy-binned near field is chosen over the flat pair list.
    bin_K : int
        Number of occupancy tiers for the binned near field.
    batch_size : int
        ``lax.map`` batch size for the per-pair W/X (and per-pair M2L) phases
        (default 2048).
    periodic_axes : tuple of int
        Axes with periodic boundary conditions.
    domain : (pmin, extent), optional
        Root-box specification; required to pin the true unit cell under PBC
        (defaults to the bounding cube with a warning).

    Returns
    -------
    dict
        Setup dict ``s``; extra keys: ``maxleaf``, ``bin_ratio``, ``adaptive``.

    Notes
    -----
    PBC: the periodic far field PROJECTS OUT the net cell charge (the divergent monopole
    direction) — a non-neutral cell silently computes the neutralised system (jellium
    convention). Pass neutral charges, or account for the background.
    """
    from jaxfmm.util import validate_points
    validate_points(positions, "positions")
    positions = jnp.asarray(positions)
    g = core.setup_geometry(positions, N_max=N_max, max_depth=max_depth, bin_gate=bin_gate,
                            bin_K=bin_K, periodic_axes=tuple(periodic_axes), domain=domain)
    md, maxleaf = g["max_depth"], g["maxleaf"]
    from jaxfmm.trees import assign_particles
    _, pil = assign_particles(g.pop("sort_idx"), g["m_id"], g["lvls"], g["counts"],
                              g["is_leaf"], md, N_max=maxleaf)
    ops = precompute_operators(p, float(g["domain_extent"]), md)
    if g["periodic"]:
        from jaxfmm.pbc import gen_pbc_uniform
        from jaxfmm.trees.cubic import cubic_img_table
        pbc = gen_pbc_uniform(p, float(g["domain_extent"]), tuple(periodic_axes))
        img_cnct = cubic_img_table(g["domain_extent"],
                                   tuple(periodic_axes)).astype(positions.dtype)
        level_ranges = None                # periodic V is 3-col with level-1 pairs: offset M2L
    else:
        pbc, img_cnct = None, jnp.array([[]])
        level_ranges = m2l_level_ranges(g["lvls"], md)
    if mem_limit is not None:
        from jaxfmm.near import hint_legacy_mem_limit
        hint_legacy_mem_limit(mem_limit)                 # pre-0.3.1 working-set values are ~6*N_max too small
    if mem_limit is None:
        mem_limit = auto_mem_limit(maxleaf, dtype_bytes=positions.dtype.itemsize,
                                   binned=g["p2p_bins"] is not None)
        if jax.devices()[0].platform != "cpu":   # CPU: fused kernels never materialise the model
            print(f"[jaxfmm] mem_limit auto: {mem_limit/2**30:.1f} GiB peak-transient budget "
                  f"(speed-optimal; pass mem_limit=<bytes> to cap)")
    return dict(positions=positions, pil=pil, dir_cnct=g["dir_cnct"], p2p_bins=g["p2p_bins"],
                V=g["V"], W=g["W"], X=g["X"], m_id=g["m_id"], p_id=g["p_id"], lvls=g["lvls"],
                is_leaf=g["is_leaf"], domain_extent=g["domain_extent"], pmin=g["pmin"],
                ops=ops, level_ranges=level_ranges, batch_size=batch_size,
                mem_limit=mem_limit, p2p_mixed=p2p_mixed, max_depth=md,
                maxleaf=maxleaf, bin_ratio=g["bin_ratio"], adaptive=g["adaptive"],
                pbc=pbc, img_cnct=img_cnct)


def evaluate_setup(s, charges):
    r"""Potential at every source point for the fixed tree in ``s`` (:func:`setup`) and new
    ``charges``.  Jit-friendly (statics in ``s`` are hashable):
    ``fmm = jax.jit(lambda q: uniform_fmm.evaluate_setup(s, q))``.

    Parameters
    ----------
    s : dict
        Setup dict from :func:`setup`.
    charges : (N,) array
        Source charges (matching ``s["positions"]``).

    Returns
    -------
    (N,) array
        Potential at every source point, in input order.
    """
    return uniform_eval(s["positions"], charges, s["m_id"], s["p_id"], s["lvls"],
                        s["is_leaf"], s["pil"], s["V"], s["dir_cnct"],
                        s["domain_extent"], s["pmin"], s["ops"], s["max_depth"],
                        W_list=s["W"], X_list=s["X"], batch_size=s["batch_size"],
                        mem_limit=s["mem_limit"], p2p_mixed=s["p2p_mixed"],
                        level_ranges=s["level_ranges"], p2p_bins=s["p2p_bins"],
                        pbc=s["pbc"], img_cnct=s["img_cnct"])


@partial(jax.jit, static_argnames=["skel"])
def _frozen_fwd(arrs, charges, skel):
    r"""MODULE-LEVEL jitted frozen-geometry evaluator (setup-as-arguments, cf. kifmm)."""
    from jaxfmm.util import join_static
    return evaluate_setup(join_static(skel, arrs), charges)


def compile_evaluator(s):
    r"""FROZEN-geometry evaluator: one jitted ``fmm(charges) -> potential`` over the
    :func:`setup` dict.  Setup-as-arguments (not a baked closure): identical-shape setups
    re-use the same executable with zero retrace (see :func:`jaxfmm.kifmm.compile_evaluator`).

    Parameters
    ----------
    s : dict
        Setup dict from :func:`setup`.

    Returns
    -------
    callable
        ``fmm(charges) -> (N,) potential`` (jitted, geometry frozen).
    """
    from jaxfmm.util import split_static
    skel, arrs = split_static(s)
    return lambda charges: _frozen_fwd(arrs, charges, skel)
