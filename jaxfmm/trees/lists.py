r"""Cubic-octree interaction lists.

V-list (well-separated M2L pairs), U-list (near neighbours + the adaptive
cross-level variant), and the W/X cross-level lists — all derived from the
2:1-balanced cubic octree in ``trees/cubic.py``. Jitted fixed-buffer builders
plus eager block-chunked variants that bound the device memory peak.
"""
import jax
import jax.numpy as jnp
from functools import partial
from jaxfmm.trees.cubic import (_morton_to_grid, get_26_neighbors,
                               get_26_neighbors_periodic, _IMG_COMBOS)

def _not_adjacent(cx, cy, cz, tx, ty, tz):
    r"""``max(|cx-tx|, |cy-ty|, |cz-tz|) > 1`` on int32 grid coordinates, written WITHOUT ``abs``.

    The ``abs`` form is miscompiled by XLA on H100 (sm_90, jax/jaxlib 0.10.2) when fused with
    the uint64 Morton decode, cast and max: the jitted V-list dropped 19 % of its pairs
    (640584 -> 520242 on a 1e5-point KIFMM tree; the eager trace and CPU/A100 are correct),
    which made the KIFMM and uniform far fields 25-45 % wrong on Musica. Only a barrier AFTER
    ``abs`` or this abs-free form fix it (int64 also does, but degrades to int32 with x64 off).
    """
    return ((cx - tx > 1) | (tx - cx > 1) | (cy - ty > 1) | (ty - cy > 1)
            | (cz - tz > 1) | (tz - cz > 1))


@partial(jax.jit, static_argnames=["max_boxes", "max_depth"])
def build_interaction_lists(m_id, lvls, is_leaf, max_boxes, max_depth):
    r"""Build the U-list (P2P leaf-leaf) and V-list (M2L same-level separated) pairs.

    The U-list pairs every active leaf with its active same-level leaf
    neighbours. The V-list pairs every active box with the same-level
    children of its parent's neighbours that are not adjacent to it (at most
    189 partners per box).

    Parameters
    ----------
    m_id : jnp.ndarray
        ``(max_boxes,)`` uint64 — box Morton codes.
    lvls : jnp.ndarray
        ``(max_boxes,)`` int32 — box levels (0 = inactive).
    is_leaf : jnp.ndarray
        ``(max_boxes,)`` bool.
    max_boxes, max_depth : int
        Static buffer size / tree depth.

    Returns
    -------
    U_list : jnp.ndarray
        ``(max_boxes*27, 2)`` int32 ``(tgt, src)`` pairs, -1 padded.
    V_list : jnp.ndarray
        ``(max_boxes*189, 2)`` int32 ``(tgt, src)`` pairs, -1 padded.
    """
    MAX_UINT64 = ~jnp.array(0, dtype=jnp.uint64)
    active = lvls > 0

    # Fixed output sizes (generous upper bounds for any valid 3-D 2:1-balanced octree).
    max_U = max_boxes * 27    # ≤ 26 same-level leaf neighbours per leaf
    max_V = max_boxes * 189   # ≤ 189 V-list partners per box (pvfmm bound)

    # ── Unique sortable box-ID: (level << (3*max_depth)) | morton ─────────────
    # This encoding is globally unique across all levels because the Morton code
    # at level l fits in exactly 3*l ≤ 3*max_depth bits.
    level_shift = jnp.uint64(3 * max_depth)
    box_ids = (lvls.astype(jnp.uint64) << level_shift) | m_id
    active_box_ids = jnp.where(active, box_ids, MAX_UINT64)

    # Sort so we can use searchsorted
    sorted_order = jnp.argsort(active_box_ids)          # (max_boxes,)
    sorted_box_ids = active_box_ids[sorted_order]        # (max_boxes,), ascending

    # ── U-list ────────────────────────────────────────────────────────────────
    # For each active leaf t, find same-level neighbours that are active leaves.

    neighs_26, valid_26 = get_26_neighbors(m_id, lvls)  # (max_boxes, 26)

    # Box-IDs of same-level neighbours
    neigh_box_ids = (lvls[:, None].astype(jnp.uint64) << level_shift) | neighs_26
    # (max_boxes, 26) uint64

    # Lookup
    u_ins = jnp.searchsorted(sorted_box_ids, neigh_box_ids.reshape(-1), side="left", method="scan_unrolled")
    u_clipped = jnp.clip(u_ins, 0, max_boxes - 1)
    u_orig = sorted_order[u_clipped].reshape(max_boxes, 26).astype(jnp.int32)

    found_ids = active_box_ids[u_orig]                           # (max_boxes, 26)
    neigh_found = valid_26 & (found_ids == neigh_box_ids)        # correct box found
    neigh_leaf = is_leaf[u_orig] & active[u_orig]                # neighbour is active leaf
    t_leaf = (is_leaf & active)[:, None]                         # target is active leaf

    u_valid = t_leaf & neigh_found & neigh_leaf                  # (max_boxes, 26)

    # Compact to fixed array
    u_tgt = jnp.broadcast_to(
        jnp.arange(max_boxes, dtype=jnp.int32)[:, None], (max_boxes, 26)
    ).reshape(-1)
    u_src = u_orig.reshape(-1)
    u_flat_valid = u_valid.reshape(-1)

    u_pos = jnp.cumsum(u_flat_valid.astype(jnp.int32)) - 1
    u_scatter = jnp.where(u_flat_valid, u_pos, jnp.int32(max_U))
    U_list = jnp.full((max_U, 2), jnp.int32(-1))
    U_list = U_list.at[u_scatter, 0].set(u_tgt, mode="drop")
    U_list = U_list.at[u_scatter, 1].set(u_src, mode="drop")

    # ── V-list ────────────────────────────────────────────────────────────────
    # For each active box t, find same-level non-adjacent boxes whose parents
    # are adjacent to t's parent.

    # Parent Morton and level
    parent_mortons = m_id // jnp.uint64(8)                        # (max_boxes,)
    parent_lvls = jnp.maximum(lvls - 1, jnp.int32(0))             # (max_boxes,)

    # 26 neighbours of each parent
    p_neighs, p_valid = get_26_neighbors(parent_mortons, parent_lvls)
    # (max_boxes, 26) uint64

    # 8 children of each parent neighbour (at t's level)
    child_off = jnp.arange(8, dtype=jnp.uint64)
    v_cand_mortons = (
        p_neighs[:, :, None] * jnp.uint64(8) + child_off[None, None, :]
    )  # (max_boxes, 26, 8) uint64

    # Box-IDs for candidates (at same level as t)
    v_cand_ids = (
        lvls[:, None, None].astype(jnp.uint64) << level_shift
    ) | v_cand_mortons  # (max_boxes, 26, 8) uint64

    # Lookup candidates
    v_ins = jnp.searchsorted(sorted_box_ids, v_cand_ids.reshape(-1), side="left", method="scan_unrolled")
    v_clipped = jnp.clip(v_ins, 0, max_boxes - 1)
    v_orig = sorted_order[v_clipped].reshape(max_boxes, 26, 8).astype(jnp.int32)

    v_found_ids = active_box_ids[v_orig]                          # (max_boxes, 26, 8)
    v_found = p_valid[:, :, None] & (v_found_ids == v_cand_ids)  # candidate is active

    # Adjacency check: L-inf grid distance at level lvls[t]
    # If max(|dx|, |dy|, |dz|) ≤ 1 the pair is adjacent (U-list territory).
    tx, ty, tz = _morton_to_grid(m_id)                            # (max_boxes,) uint64
    cx, cy, cz = _morton_to_grid(v_cand_mortons.reshape(-1))      # (max_boxes*26*8,)
    cx = cx.reshape(max_boxes, 26, 8)
    cy = cy.reshape(max_boxes, 26, 8)
    cz = cz.reshape(max_boxes, 26, 8)

    not_adjacent = _not_adjacent(*(c.astype(jnp.int32) for c in (cx, cy, cz)),
                                 *(t[:, None, None].astype(jnp.int32) for t in (tx, ty, tz)))
    not_self = v_cand_mortons != m_id[:, None, None]              # exclude t itself

    t_active_col = active[:, None, None]
    v_valid = t_active_col & v_found & not_adjacent & not_self    # (max_boxes, 26, 8)

    # Compact to fixed array
    v_tgt = jnp.broadcast_to(
        jnp.arange(max_boxes, dtype=jnp.int32)[:, None, None], (max_boxes, 26, 8)
    ).reshape(-1)
    v_src = v_orig.reshape(-1)
    v_flat_valid = v_valid.reshape(-1)

    v_pos = jnp.cumsum(v_flat_valid.astype(jnp.int32)) - 1
    v_scatter = jnp.where(v_flat_valid, v_pos, jnp.int32(max_V))
    V_list = jnp.full((max_V, 2), jnp.int32(-1))
    V_list = V_list.at[v_scatter, 0].set(v_tgt, mode="drop")
    V_list = V_list.at[v_scatter, 1].set(v_src, mode="drop")

    return U_list, V_list


@partial(jax.jit, static_argnames=["max_boxes", "max_depth", "periodic_axes"])
def build_interaction_lists_periodic(m_id, lvls, is_leaf, max_boxes, max_depth,
                                     periodic_axes=()):
    r"""Periodic (toroidal) variant of :func:`build_interaction_lists` for uniform trees.

    Neighbours wrap along ``periodic_axes`` (root box = the unit cell) and
    every wrapped pair carries an image index into
    :func:`jaxfmm.trees.cubic.cubic_img_table`.

    Parameters
    ----------
    m_id, lvls, is_leaf, max_boxes, max_depth
        As in :func:`build_interaction_lists`.
    periodic_axes : tuple of int, optional
        Static — wrapping axes.

    Returns
    -------
    U_list : jnp.ndarray
        ``(max_boxes*27, 2)`` int32 — ``(tgt, src + max_boxes*img_idx)``: the
        ENCODED form consumed directly by ``near.eval_direct``'s
        ``partner // nbox`` decode.
    V_list : jnp.ndarray
        ``(max_boxes*189, 3)`` int32 — ``(tgt, src, img_idx)``: M2L consumers
        must compute the displacement from UNWRAPPED grid coords
        ``src + img * 2**lvl``.

    Notes
    -----
    A box CAN be its own V-partner through an image (small periodic trees).
    """
    from jaxfmm.trees.cubic import get_26_neighbors_periodic, _IMG_COMBOS
    MAX_UINT64 = ~jnp.array(0, dtype=jnp.uint64)
    active = lvls > 0
    max_U = max_boxes * 27
    max_V = max_boxes * 189
    level_shift = jnp.uint64(3 * max_depth)
    box_ids = (lvls.astype(jnp.uint64) << level_shift) | m_id
    active_box_ids = jnp.where(active, box_ids, MAX_UINT64)
    sorted_order = jnp.argsort(active_box_ids)
    sorted_box_ids = active_box_ids[sorted_order]

    # ── U-list: wrapped same-level leaf neighbours ────────────────────────────
    neighs, valid, img_u = get_26_neighbors_periodic(m_id, lvls, periodic_axes)
    nbi = (lvls[:, None].astype(jnp.uint64) << level_shift) | neighs
    ins = jnp.searchsorted(sorted_box_ids, nbi.reshape(-1), side="left", method="scan_unrolled")
    u_orig = sorted_order[jnp.clip(ins, 0, max_boxes - 1)].reshape(max_boxes, 26).astype(jnp.int32)
    u_ok = ((is_leaf & active)[:, None] & valid & (active_box_ids[u_orig] == nbi)
            & is_leaf[u_orig] & active[u_orig])
    u_tgt = jnp.broadcast_to(jnp.arange(max_boxes, dtype=jnp.int32)[:, None],
                             (max_boxes, 26)).reshape(-1)
    u_src = (u_orig + jnp.int32(max_boxes) * img_u).reshape(-1)     # image-ENCODED
    u_pos = jnp.cumsum(u_ok.reshape(-1).astype(jnp.int32)) - 1
    u_sc = jnp.where(u_ok.reshape(-1), u_pos, jnp.int32(max_U))
    U_list = jnp.full((max_U, 2), jnp.int32(-1))
    U_list = U_list.at[u_sc, 0].set(u_tgt, mode="drop")
    U_list = U_list.at[u_sc, 1].set(u_src, mode="drop")

    # ── V-list: children of the parent's WRAPPED neighbours, non-adjacent ─────
    parent_m = m_id // jnp.uint64(8)
    parent_lvl = jnp.maximum(lvls - 1, jnp.int32(0))
    pn, pv, img_p = get_26_neighbors_periodic(parent_m, parent_lvl, periodic_axes)
    child_off = jnp.arange(8, dtype=jnp.uint64)
    vc = pn[:, :, None] * jnp.uint64(8) + child_off[None, None, :]   # (mb, 26, 8)
    vci = (lvls[:, None, None].astype(jnp.uint64) << level_shift) | vc
    ins = jnp.searchsorted(sorted_box_ids, vci.reshape(-1), side="left", method="scan_unrolled")
    v_orig = sorted_order[jnp.clip(ins, 0, max_boxes - 1)].reshape(max_boxes, 26, 8).astype(jnp.int32)
    v_found = pv[:, :, None] & (active_box_ids[v_orig] == vci)
    # adjacency + self test on UNWRAPPED child coords (child inherits the parent's image)
    tx, ty, tz = _morton_to_grid(m_id)
    cx, cy, cz = _morton_to_grid(vc.reshape(-1))
    grid = (2 ** lvls)[:, None, None].astype(jnp.int32)
    combos = jnp.asarray(_IMG_COMBOS, jnp.int32)                     # (27, 3), row-idx aligned
    ix = combos[img_p][:, :, 0][:, :, None]                          # (mb, 26, 1)
    iy = combos[img_p][:, :, 1][:, :, None]
    iz = combos[img_p][:, :, 2][:, :, None]
    ux = cx.reshape(max_boxes, 26, 8).astype(jnp.int32) + ix * grid
    uy = cy.reshape(max_boxes, 26, 8).astype(jnp.int32) + iy * grid
    uz = cz.reshape(max_boxes, 26, 8).astype(jnp.int32) + iz * grid
    not_adj = _not_adjacent(ux, uy, uz, *(t[:, None, None].astype(jnp.int32) for t in (tx, ty, tz)))
    # a wrapped self-image IS a legitimate V partner; only the true (img 0) self is excluded
    not_self = (vc != m_id[:, None, None]) | (img_p[:, :, None] != 0)
    v_ok = active[:, None, None] & v_found & not_adj & not_self
    v_tgt = jnp.broadcast_to(jnp.arange(max_boxes, dtype=jnp.int32)[:, None, None],
                             (max_boxes, 26, 8)).reshape(-1)
    v_src = v_orig.reshape(-1)
    v_img = jnp.broadcast_to(img_p[:, :, None], (max_boxes, 26, 8)).reshape(-1)
    v_pos = jnp.cumsum(v_ok.reshape(-1).astype(jnp.int32)) - 1
    v_sc = jnp.where(v_ok.reshape(-1), v_pos, jnp.int32(max_V))
    V_list = jnp.full((max_V, 3), jnp.int32(-1))
    V_list = V_list.at[v_sc, 0].set(v_tgt, mode="drop")
    V_list = V_list.at[v_sc, 1].set(v_src, mode="drop")
    V_list = V_list.at[v_sc, 2].set(v_img, mode="drop")
    return U_list, V_list


# ── adaptive U-list + W/X cross-level lists ───────────────────────────────────
_MAX_U_PER_LEAF = 64


@partial(jax.jit, static_argnames=["max_boxes", "max_depth", "periodic_axes"])
def build_ulist_adaptive(m_id, lvls, is_leaf, max_boxes, max_depth, periodic_axes=()):
    r"""Complete near-field (U-list) for a 2:1-balanced ADAPTIVE octree.

    Pairs every leaf target with every geometrically adjacent leaf source,
    across levels: same level, one level coarser, and one level finer. These
    three categories partition the adjacent leaves with no overlap — the proof
    relies on 2:1 balance: a coarse leaf touching ``B`` also touches ``B``'s
    parent; a finer adjacent leaf is a child of one of ``B``'s same-level
    neighbours.

    Parameters
    ----------
    m_id, lvls, is_leaf : jnp.ndarray
        Tree arrays, shape ``(max_boxes,)``.
    max_boxes, max_depth : int
        Static.
    periodic_axes : tuple of int, optional
        Static — wrapping axes (sources become image-encoded).

    Returns
    -------
    U_list : jnp.ndarray
        ``(max_boxes * 64, 2)`` int32 — ``(target_leaf, source_leaf)`` pairs,
        -1 padded; excludes self pairs.
    overflow : jnp.ndarray
        Scalar bool — True if the output buffer was too small (pairs dropped).
    """
    MAX_UINT64 = ~jnp.array(0, dtype=jnp.uint64)
    active = lvls > 0
    leaf = is_leaf & active
    level_shift = jnp.uint64(3 * max_depth)

    box_ids = (lvls.astype(jnp.uint64) << level_shift) | m_id
    active_box_ids = jnp.where(active, box_ids, MAX_UINT64)
    sorted_order = jnp.argsort(active_box_ids)
    sorted_box_ids = active_box_ids[sorted_order]

    def lookup(cand_ids):
        r"""Map candidate box_ids to (original box index, found mask)."""
        flat = cand_ids.reshape(-1)
        ins = jnp.searchsorted(sorted_box_ids, flat, side="left", method="scan_unrolled")
        ins = jnp.clip(ins, 0, max_boxes - 1)
        orig = sorted_order[ins]
        found = active_box_ids[orig] == flat
        return orig.reshape(cand_ids.shape), found.reshape(cand_ids.shape)

    Lb = lvls
    bx, by, bz = _morton_to_grid(m_id)            # (max_boxes,) uint64, B grid coords
    bx = bx.astype(jnp.int64); by = by.astype(jnp.int64); bz = bz.astype(jnp.int64)
    tgt_leaf = leaf                               # target must be a leaf

    # ── (i) same-level neighbours (always adjacent by construction) ────────────
    per = len(periodic_axes) > 0
    IMG = jnp.asarray(_IMG_COMBOS, jnp.int64)
    neighs_S, valid_S, img_S = get_26_neighbors_periodic(m_id, lvls, periodic_axes)
    ids_S = (Lb[:, None].astype(jnp.uint64) << level_shift) | neighs_S
    orig_S, found_S = lookup(ids_S)
    valid_pairs_S = tgt_leaf[:, None] & valid_S & found_S & leaf[orig_S]

    # ── (ii) coarser neighbours: parent's 26 neighbours at level Lb-1 ──────────
    parent_m = m_id // jnp.uint64(8)
    parent_lvl = jnp.maximum(Lb - 1, jnp.int32(0))
    pn, pn_valid, img_C = get_26_neighbors_periodic(parent_m, parent_lvl, periodic_axes)
    ids_C = ((Lb - 1)[:, None].astype(jnp.uint64) << level_shift) | pn
    orig_C, found_C = lookup(ids_C)
    cx, cy, cz = _morton_to_grid(pn.reshape(-1))
    cx = cx.astype(jnp.int64).reshape(max_boxes, 26)
    cy = cy.astype(jnp.int64).reshape(max_boxes, 26)
    cz = cz.astype(jnp.int64).reshape(max_boxes, 26)
    if per:                                          # UNWRAP for the adjacency test
        gsz_p = (2 ** jnp.maximum(Lb - 1, 0)).astype(jnp.int64)[:, None]
        ioc = IMG[img_C]
        cx = cx + ioc[:, :, 0] * gsz_p
        cy = cy + ioc[:, :, 1] * gsz_p
        cz = cz + ioc[:, :, 2] * gsz_p
    # adjacency at common finest level Lb: B has factor 1 (lo=hi=b); coarse has
    # factor 2 (lo=2c, hi=2c+1).  adj_axis = (b <= 2c+1+1) & (2c <= b+1).
    def _adj_coarse(b, c):
        return (b <= 2 * c + 2) & (2 * c <= b + 1)
    adj_C = _adj_coarse(bx[:, None], cx) & _adj_coarse(by[:, None], cy) & _adj_coarse(bz[:, None], cz)
    valid_pairs_C = (tgt_leaf[:, None] & (Lb[:, None] > 1) & pn_valid
                     & found_C & leaf[orig_C] & adj_C)

    # ── (iii) finer neighbours: children of same-level neighbours at Lb+1 ──────
    child_off = jnp.arange(8, dtype=jnp.uint64)
    child_m = neighs_S[:, :, None] * jnp.uint64(8) + child_off[None, None, :]   # (mb,26,8)
    ids_F = ((Lb + 1)[:, None, None].astype(jnp.uint64) << level_shift) | child_m
    orig_F, found_F = lookup(ids_F)
    fx, fy, fz = _morton_to_grid(child_m.reshape(-1))
    fx = fx.astype(jnp.int64).reshape(max_boxes, 26, 8)
    fy = fy.astype(jnp.int64).reshape(max_boxes, 26, 8)
    fz = fz.astype(jnp.int64).reshape(max_boxes, 26, 8)
    if per:                                          # children INHERIT the neighbour's image
        gsz_f = (2 ** (Lb + 1)).astype(jnp.int64)[:, None, None]
        iof = IMG[img_S][:, :, None, :]
        fx = fx + iof[:, :, :, 0] * gsz_f
        fy = fy + iof[:, :, :, 1] * gsz_f
        fz = fz + iof[:, :, :, 2] * gsz_f
    # adjacency at common finest level Lb+1: B has factor 2 (lo=2b, hi=2b+1);
    # child has factor 1 (lo=hi=f).  adj_axis = (2b <= f+1) & (f <= 2b+1+1).
    def _adj_fine(b, f):
        return (2 * b <= f + 1) & (f <= 2 * b + 2)
    adj_F = (_adj_fine(bx[:, None, None], fx) & _adj_fine(by[:, None, None], fy)
             & _adj_fine(bz[:, None, None], fz))
    valid_pairs_F = (tgt_leaf[:, None, None] & valid_S[:, :, None]
                     & found_F & leaf[orig_F] & adj_F)

    # ── assemble (target, source) pairs and compact ───────────────────────────
    tgt_idx = jnp.arange(max_boxes, dtype=jnp.int32)
    t_S = jnp.broadcast_to(tgt_idx[:, None], (max_boxes, 26)).reshape(-1)
    t_C = jnp.broadcast_to(tgt_idx[:, None], (max_boxes, 26)).reshape(-1)
    t_F = jnp.broadcast_to(tgt_idx[:, None, None], (max_boxes, 26, 8)).reshape(-1)
    if per:                                          # image-ENCODED src (eval_direct decodes)
        s_S = (orig_S + max_boxes * img_S).reshape(-1).astype(jnp.int32)
        s_C = (orig_C + max_boxes * img_C).reshape(-1).astype(jnp.int32)
        s_F = (orig_F + max_boxes * jnp.broadcast_to(img_S[:, :, None],
                                                     orig_F.shape)).reshape(-1).astype(jnp.int32)
    else:
        s_S = orig_S.reshape(-1).astype(jnp.int32)
        s_C = orig_C.reshape(-1).astype(jnp.int32)
        s_F = orig_F.reshape(-1).astype(jnp.int32)

    tgt_all = jnp.concatenate([t_S, t_C, t_F])
    src_all = jnp.concatenate([s_S, s_C, s_F])
    valid_all = jnp.concatenate([
        valid_pairs_S.reshape(-1), valid_pairs_C.reshape(-1), valid_pairs_F.reshape(-1)
    ])

    max_U = max_boxes * _MAX_U_PER_LEAF
    pos = jnp.cumsum(valid_all.astype(jnp.int32)) - 1
    n_valid = pos[-1] + 1
    scatter = jnp.where(valid_all, pos, jnp.int32(max_U))
    U_list = jnp.full((max_U, 2), jnp.int32(-1))
    U_list = U_list.at[scatter, 0].set(tgt_all, mode="drop")
    U_list = U_list.at[scatter, 1].set(src_all, mode="drop")

    overflow = n_valid > max_U
    return U_list, overflow


# Per-target capacities for the cross-level far-field lists.
_MAX_W_PER_LEAF = 152   # finer non-adjacent children of the 26 colleagues
_MAX_X_PER_LEAF = 27    # coarser non-adjacent parent-colleagues


@partial(jax.jit, static_argnames=["max_boxes", "max_depth", "periodic_axes"])
def build_wx_lists(m_id, lvls, is_leaf, max_boxes, max_depth, periodic_axes=()):
    r"""Cross-level far-field lists for a 2:1-balanced ADAPTIVE octree.

    W-list (M2P): for each leaf target ``B``, the FINER boxes ``A`` (level
    ``L+1``) that are children of ``B``'s same-level colleagues and are NOT
    adjacent to ``B``. ``A``'s multipole (upward equivalent density) is
    evaluated directly at ``B``'s particles. ``A`` may be leaf or internal
    (its multipole covers its subtree; 2:1 balance guarantees ``A`` is
    separated from ``B``).

    X-list (P2L): for each target box ``B`` (leaf OR internal), the COARSER
    leaves ``A`` (level ``L-1``) that are adjacent to ``B``'s parent but NOT
    adjacent to ``B``. ``A``'s particles are evaluated onto ``B``'s downward
    check surface (contributes to ``B``'s local expansion, like M2L). When
    ``B`` is internal the deposit reaches ``B``'s leaf descendants via L2L —
    this is what makes the far field complete for level gaps greater than one.

    Parameters
    ----------
    m_id, lvls, is_leaf : jnp.ndarray
        Tree arrays, shape ``(max_boxes,)``.
    max_boxes, max_depth : int
        Static.
    periodic_axes : tuple of int, optional
        Static — wrapping axes (sources become image-encoded).

    Returns
    -------
    W_list : jnp.ndarray
        ``(max_boxes * 152, 2)`` int32 — ``(target_leaf, source_box)``.
    X_list : jnp.ndarray
        ``(max_boxes * 27, 2)`` int32 — ``(target_box, source_leaf)``.
    overflow : jnp.ndarray
        Scalar bool.
    """
    MAX_UINT64 = ~jnp.array(0, dtype=jnp.uint64)
    active = lvls > 0
    leaf = is_leaf & active
    level_shift = jnp.uint64(3 * max_depth)

    box_ids = (lvls.astype(jnp.uint64) << level_shift) | m_id
    active_box_ids = jnp.where(active, box_ids, MAX_UINT64)
    sorted_order = jnp.argsort(active_box_ids)
    sorted_box_ids = active_box_ids[sorted_order]

    def lookup(cand_ids):
        flat = cand_ids.reshape(-1)
        ins = jnp.searchsorted(sorted_box_ids, flat, side="left", method="scan_unrolled")
        ins = jnp.clip(ins, 0, max_boxes - 1)
        orig = sorted_order[ins]
        found = active_box_ids[orig] == flat
        return orig.reshape(cand_ids.shape), found.reshape(cand_ids.shape)

    Lb = lvls
    bx, by, bz = _morton_to_grid(m_id)
    bx = bx.astype(jnp.int64); by = by.astype(jnp.int64); bz = bz.astype(jnp.int64)
    tgt_leaf = leaf
    tgt_idx = jnp.arange(max_boxes, dtype=jnp.int32)

    # ── X-list: coarser parent-colleagues, non-adjacent ───────────────────────
    parent_m = m_id // jnp.uint64(8)
    parent_lvl = jnp.maximum(Lb - 1, jnp.int32(0))
    per = len(periodic_axes) > 0
    IMG = jnp.asarray(_IMG_COMBOS, jnp.int64)
    pn, pn_valid, img_C = get_26_neighbors_periodic(parent_m, parent_lvl, periodic_axes)
    ids_C = ((Lb - 1)[:, None].astype(jnp.uint64) << level_shift) | pn
    orig_C, found_C = lookup(ids_C)
    cx, cy, cz = _morton_to_grid(pn.reshape(-1))
    cx = cx.astype(jnp.int64).reshape(max_boxes, 26)
    cy = cy.astype(jnp.int64).reshape(max_boxes, 26)
    cz = cz.astype(jnp.int64).reshape(max_boxes, 26)
    if per:                                          # UNWRAP for the adjacency test
        gsz_p = (2 ** jnp.maximum(Lb - 1, 0)).astype(jnp.int64)[:, None]
        ioc = IMG[img_C]
        cx = cx + ioc[:, :, 0] * gsz_p
        cy = cy + ioc[:, :, 1] * gsz_p
        cz = cz + ioc[:, :, 2] * gsz_p

    def _adj_coarse(b, c):
        return (b <= 2 * c + 2) & (2 * c <= b + 1)
    adj_C = (_adj_coarse(bx[:, None], cx) & _adj_coarse(by[:, None], cy)
             & _adj_coarse(bz[:, None], cz))
    # X-list target may be ANY active box (leaf or internal): a deposit into an
    # internal box's local expansion propagates down to its leaves via L2L.
    x_valid = (active[:, None] & (Lb[:, None] > 1) & pn_valid
               & found_C & leaf[orig_C] & ~adj_C)

    t_X = jnp.broadcast_to(tgt_idx[:, None], (max_boxes, 26)).reshape(-1)
    s_X = ((orig_C + max_boxes * img_C) if per else orig_C).reshape(-1).astype(jnp.int32)
    x_flat = x_valid.reshape(-1)
    max_X = max_boxes * _MAX_X_PER_LEAF
    x_pos = jnp.cumsum(x_flat.astype(jnp.int32)) - 1
    n_X = x_pos[-1] + 1
    x_scatter = jnp.where(x_flat, x_pos, jnp.int32(max_X))
    X_list = jnp.full((max_X, 2), jnp.int32(-1))
    X_list = X_list.at[x_scatter, 0].set(t_X, mode="drop")
    X_list = X_list.at[x_scatter, 1].set(s_X, mode="drop")

    # ── W-list: finer children of colleagues, non-adjacent ────────────────────
    neighs_S, valid_S, img_S = get_26_neighbors_periodic(m_id, lvls, periodic_axes)
    child_off = jnp.arange(8, dtype=jnp.uint64)
    child_m = neighs_S[:, :, None] * jnp.uint64(8) + child_off[None, None, :]
    ids_F = ((Lb + 1)[:, None, None].astype(jnp.uint64) << level_shift) | child_m
    orig_F, found_F = lookup(ids_F)
    fx, fy, fz = _morton_to_grid(child_m.reshape(-1))
    fx = fx.astype(jnp.int64).reshape(max_boxes, 26, 8)
    fy = fy.astype(jnp.int64).reshape(max_boxes, 26, 8)
    fz = fz.astype(jnp.int64).reshape(max_boxes, 26, 8)
    if per:                                          # children INHERIT the colleague's image
        gsz_f = (2 ** (Lb + 1)).astype(jnp.int64)[:, None, None]
        iof = IMG[img_S][:, :, None, :]
        fx = fx + iof[:, :, :, 0] * gsz_f
        fy = fy + iof[:, :, :, 1] * gsz_f
        fz = fz + iof[:, :, :, 2] * gsz_f

    def _adj_fine(b, f):
        return (2 * b <= f + 1) & (f <= 2 * b + 2)
    adj_F = (_adj_fine(bx[:, None, None], fx) & _adj_fine(by[:, None, None], fy)
             & _adj_fine(bz[:, None, None], fz))
    # W source need not be a leaf (its multipole represents its whole subtree).
    w_valid = (tgt_leaf[:, None, None] & valid_S[:, :, None] & found_F & ~adj_F)

    t_W = jnp.broadcast_to(tgt_idx[:, None, None], (max_boxes, 26, 8)).reshape(-1)
    s_W = ((orig_F + max_boxes * jnp.broadcast_to(img_S[:, :, None], orig_F.shape))
           if per else orig_F).reshape(-1).astype(jnp.int32)
    w_flat = w_valid.reshape(-1)
    max_W = max_boxes * _MAX_W_PER_LEAF
    w_pos = jnp.cumsum(w_flat.astype(jnp.int32)) - 1
    n_W = w_pos[-1] + 1
    w_scatter = jnp.where(w_flat, w_pos, jnp.int32(max_W))
    W_list = jnp.full((max_W, 2), jnp.int32(-1))
    W_list = W_list.at[w_scatter, 0].set(t_W, mode="drop")
    W_list = W_list.at[w_scatter, 1].set(s_W, mode="drop")

    overflow = (n_W > max_W) | (n_X > max_X)
    return W_list, X_list, overflow


# ── eager BLOCK-CHUNKED builders (setup-time memory diet) ─────────────────────
#
# The jitted builders above materialise several (max_boxes, 26, 8) uint64 candidate
# tensors at once — the transient dominates the whole setup memory peak at large N.
# These eager variants run the identical logic over BLOCKS of target boxes (bounded
# peak ~ block*208 per tensor) and return COMPACT host pair arrays (no max_boxes*189
# dense padding), which is what kifmm.setup wants anyway. The jitted originals stay:
# the fulljit moving-geometry path needs the whole build inside one trace, and its
# geometries are small.

_BLOCK = 262144


def _adj_coarse_ax(b, c):
    r"""Per-axis adjacency, target at level L (factor 1) vs coarser box at L-1 (factor 2)."""
    return (b <= 2 * c + 2) & (2 * c <= b + 1)


def _adj_fine_ax(b, f):
    r"""Per-axis adjacency, target at level L (factor 2 at L+1) vs finer box at L+1."""
    return (2 * b <= f + 1) & (f <= 2 * b + 2)


def _sorted_boxids(m_id, lvls, max_depth):
    r"""Global sorted box-id tables shared by every block lookup (max_boxes-sized, 1-D)."""
    MAX_UINT = ~jnp.array(0, dtype=m_id.dtype)
    level_shift = jnp.asarray(3 * max_depth, m_id.dtype)
    active = lvls > 0
    box_ids = (lvls.astype(m_id.dtype) << level_shift) | m_id
    active_box_ids = jnp.where(active, box_ids, MAX_UINT)
    sorted_order = jnp.argsort(active_box_ids)
    return active_box_ids, sorted_order, active_box_ids[sorted_order]


def _blk_lookup(cand_ids, sorted_box_ids, sorted_order, active_box_ids, max_boxes):
    flat = cand_ids.reshape(-1)
    ins = jnp.searchsorted(sorted_box_ids, flat, side="left", method="scan_unrolled")
    ins = jnp.clip(ins, 0, max_boxes - 1)
    orig = sorted_order[ins]
    found = active_box_ids[orig] == flat
    return orig.reshape(cand_ids.shape).astype(jnp.int32), found.reshape(cand_ids.shape)


def _blk_compact3(tgt, src, img, valid, cap):
    r"""Like :func:`_blk_compact` but ``(rows, 3)``: ``(tgt, src, img_idx)`` for periodic V-lists."""
    t, s, m, v = tgt.reshape(-1), src.reshape(-1), img.reshape(-1), valid.reshape(-1)
    pos = jnp.cumsum(v.astype(jnp.int32)) - 1
    scatter = jnp.where(v, pos, cap)
    out = jnp.full((cap, 3), jnp.int32(-1))
    out = out.at[scatter, 0].set(t, mode="drop")
    out = out.at[scatter, 1].set(s, mode="drop")
    out = out.at[scatter, 2].set(m, mode="drop")
    return out, pos[-1] + 1


def _blk_compact(tgt, src, valid, cap):
    r"""Block-local compaction: ``(rows, 2)`` int32 padded to ``cap`` with -1, plus count."""
    t, s, v = tgt.reshape(-1), src.reshape(-1), valid.reshape(-1)
    pos = jnp.cumsum(v.astype(jnp.int32)) - 1
    scatter = jnp.where(v, pos, cap)
    out = jnp.full((cap, 2), jnp.int32(-1))
    out = out.at[scatter, 0].set(t, mode="drop")
    out = out.at[scatter, 1].set(s, mode="drop")
    return out, pos[-1] + 1


@partial(jax.jit, static_argnames=["max_boxes", "max_depth", "block", "periodic_axes"])
def _v_block(m_b, lvl_b, tgt0, tabs, m_id, lvls, is_leaf, max_boxes, max_depth, block,
             periodic_axes=()):
    r"""V-list pairs for target boxes ``[tgt0, tgt0+block)`` — logic identical to
    :func:`build_interaction_lists`'s V section (periodic: 3-col output, wrapped
    parents, unwrapped adjacency, wrapped self-image legal)."""
    active_box_ids, sorted_order, sorted_box_ids = tabs
    level_shift = jnp.asarray(3 * max_depth, m_id.dtype)
    active_b = lvl_b > 0
    one = jnp.asarray(8, m_id.dtype)
    per = len(periodic_axes) > 0

    parent_m = m_b // one
    parent_lvl = jnp.maximum(lvl_b - 1, jnp.int32(0))
    p_neighs, p_valid, img_p = get_26_neighbors_periodic(parent_m, parent_lvl, periodic_axes)
    child_off = jnp.arange(8, dtype=m_id.dtype)
    v_cand_m = p_neighs[:, :, None] * one + child_off[None, None, :]      # (B, 26, 8)
    v_cand_ids = (lvl_b[:, None, None].astype(m_id.dtype) << level_shift) | v_cand_m
    v_orig, v_found = _blk_lookup(v_cand_ids, sorted_box_ids, sorted_order,
                                  active_box_ids, max_boxes)
    v_found = p_valid[:, :, None] & v_found

    tx, ty, tz = _morton_to_grid(m_b)
    cx, cy, cz = _morton_to_grid(v_cand_m.reshape(-1))
    shp = (block, 26, 8)
    cx = cx.reshape(shp).astype(jnp.int32); cy = cy.reshape(shp).astype(jnp.int32)
    cz = cz.reshape(shp).astype(jnp.int32)
    if per:                                            # unwrap (children inherit the image)
        grid = (2 ** lvl_b)[:, None, None].astype(jnp.int32)
        combos = jnp.asarray(_IMG_COMBOS, jnp.int32)
        io = combos[img_p][:, :, None, :]
        cx = cx + io[:, :, :, 0] * grid
        cy = cy + io[:, :, :, 1] * grid
        cz = cz + io[:, :, :, 2] * grid
    not_adjacent = _not_adjacent(cx, cy, cz, *(t[:, None, None].astype(jnp.int32) for t in (tx, ty, tz)))
    not_self = (v_cand_m != m_b[:, None, None]) | (img_p[:, :, None] != 0)
    v_valid = active_b[:, None, None] & v_found & not_adjacent & not_self

    t_idx = (tgt0 + jnp.arange(block, dtype=jnp.int32))[:, None, None]
    if per:
        return _blk_compact3(jnp.broadcast_to(t_idx, shp), v_orig,
                             jnp.broadcast_to(img_p[:, :, None], shp), v_valid, block * 189)
    return _blk_compact(jnp.broadcast_to(t_idx, shp), v_orig, v_valid, block * 189)


@partial(jax.jit, static_argnames=["max_boxes", "max_depth", "block", "periodic_axes"])
def _ua_block(m_b, lvl_b, leaf_b, tgt0, tabs, m_id, lvls, is_leaf,
              max_boxes, max_depth, block, periodic_axes=()):
    r"""Adaptive U-list pairs for target boxes ``[tgt0, tgt0+block)`` — logic identical
    to :func:`build_ulist_adaptive` (same/coarser/finer adjacency partition)."""
    active_box_ids, sorted_order, sorted_box_ids = tabs
    level_shift = jnp.asarray(3 * max_depth, m_id.dtype)
    leaf_g = is_leaf & (lvls > 0)
    one = jnp.asarray(8, m_id.dtype)

    bx, by, bz = (g.astype(jnp.int32) for g in _morton_to_grid(m_b))

    per = len(periodic_axes) > 0
    IMGc = jnp.asarray(_IMG_COMBOS, jnp.int32)
    # (i) same-level neighbours
    neighs_S, valid_S, img_S = get_26_neighbors_periodic(m_b, lvl_b, periodic_axes)
    ids_S = (lvl_b[:, None].astype(m_id.dtype) << level_shift) | neighs_S
    orig_S, found_S = _blk_lookup(ids_S, sorted_box_ids, sorted_order,
                                  active_box_ids, max_boxes)
    valid_S_pairs = leaf_b[:, None] & valid_S & found_S & leaf_g[orig_S]

    # (ii) coarser neighbours (parent colleagues), adjacency-checked
    parent_m = m_b // one
    parent_lvl = jnp.maximum(lvl_b - 1, jnp.int32(0))
    pn, pn_valid, img_C = get_26_neighbors_periodic(parent_m, parent_lvl, periodic_axes)
    ids_C = ((lvl_b - 1)[:, None].astype(m_id.dtype) << level_shift) | pn
    orig_C, found_C = _blk_lookup(ids_C, sorted_box_ids, sorted_order,
                                  active_box_ids, max_boxes)
    cx, cy, cz = (g.astype(jnp.int32).reshape(block, 26)
                  for g in _morton_to_grid(pn.reshape(-1)))
    if per:                                            # unwrap coarse coords
        gsz_p = (2 ** jnp.maximum(lvl_b - 1, 0)).astype(jnp.int32)[:, None]
        ioc = IMGc[img_C]
        cx = cx + ioc[:, :, 0] * gsz_p
        cy = cy + ioc[:, :, 1] * gsz_p
        cz = cz + ioc[:, :, 2] * gsz_p
    adj_C = (_adj_coarse_ax(bx[:, None], cx) & _adj_coarse_ax(by[:, None], cy)
             & _adj_coarse_ax(bz[:, None], cz))
    valid_C_pairs = (leaf_b[:, None] & (lvl_b[:, None] > 1) & pn_valid
                     & found_C & leaf_g[orig_C] & adj_C)

    # (iii) finer neighbours (children of colleagues), adjacency-checked
    child_off = jnp.arange(8, dtype=m_id.dtype)
    child_m = neighs_S[:, :, None] * one + child_off[None, None, :]
    ids_F = ((lvl_b + 1)[:, None, None].astype(m_id.dtype) << level_shift) | child_m
    orig_F, found_F = _blk_lookup(ids_F, sorted_box_ids, sorted_order,
                                  active_box_ids, max_boxes)
    fx, fy, fz = (g.astype(jnp.int32).reshape(block, 26, 8)
                  for g in _morton_to_grid(child_m.reshape(-1)))
    if per:                                            # children inherit the image
        gsz_f = (2 ** (lvl_b + 1)).astype(jnp.int32)[:, None, None]
        iof = IMGc[img_S][:, :, None, :]
        fx = fx + iof[:, :, :, 0] * gsz_f
        fy = fy + iof[:, :, :, 1] * gsz_f
        fz = fz + iof[:, :, :, 2] * gsz_f
    adj_F = (_adj_fine_ax(bx[:, None, None], fx) & _adj_fine_ax(by[:, None, None], fy)
             & _adj_fine_ax(bz[:, None, None], fz))
    valid_F_pairs = (leaf_b[:, None, None] & valid_S[:, :, None]
                     & found_F & leaf_g[orig_F] & adj_F)

    t_idx = tgt0 + jnp.arange(block, dtype=jnp.int32)
    tgt = jnp.concatenate([jnp.broadcast_to(t_idx[:, None], (block, 26)).reshape(-1),
                           jnp.broadcast_to(t_idx[:, None], (block, 26)).reshape(-1),
                           jnp.broadcast_to(t_idx[:, None, None], (block, 26, 8)).reshape(-1)])
    if per:                                            # image-ENCODED src
        src = jnp.concatenate([
            (orig_S + max_boxes * img_S).reshape(-1),
            (orig_C + max_boxes * img_C).reshape(-1),
            (orig_F + max_boxes * jnp.broadcast_to(img_S[:, :, None],
                                                   orig_F.shape)).reshape(-1)])
    else:
        src = jnp.concatenate([orig_S.reshape(-1), orig_C.reshape(-1), orig_F.reshape(-1)])
    valid = jnp.concatenate([valid_S_pairs.reshape(-1), valid_C_pairs.reshape(-1),
                             valid_F_pairs.reshape(-1)])
    return _blk_compact(tgt, src, valid, block * 64)


@partial(jax.jit, static_argnames=["max_boxes", "max_depth", "block", "periodic_axes"])
def _wx_block(m_b, lvl_b, leaf_b, tgt0, tabs, m_id, lvls, is_leaf,
              max_boxes, max_depth, block, periodic_axes=()):
    r"""W/X cross-level pairs for target boxes ``[tgt0, tgt0+block)`` — logic identical
    to :func:`build_wx_lists`."""
    active_box_ids, sorted_order, sorted_box_ids = tabs
    level_shift = jnp.asarray(3 * max_depth, m_id.dtype)
    active_b = lvl_b > 0
    leaf_g = is_leaf & (lvls > 0)
    one = jnp.asarray(8, m_id.dtype)

    bx, by, bz = (g.astype(jnp.int32) for g in _morton_to_grid(m_b))
    t_idx = tgt0 + jnp.arange(block, dtype=jnp.int32)

    per = len(periodic_axes) > 0
    IMGc = jnp.asarray(_IMG_COMBOS, jnp.int32)
    # X-list: coarser parent-colleagues, NON-adjacent, source must be leaf
    parent_m = m_b // one
    parent_lvl = jnp.maximum(lvl_b - 1, jnp.int32(0))
    pn, pn_valid, img_C = get_26_neighbors_periodic(parent_m, parent_lvl, periodic_axes)
    ids_C = ((lvl_b - 1)[:, None].astype(m_id.dtype) << level_shift) | pn
    orig_C, found_C = _blk_lookup(ids_C, sorted_box_ids, sorted_order,
                                  active_box_ids, max_boxes)
    cx, cy, cz = (g.astype(jnp.int32).reshape(block, 26)
                  for g in _morton_to_grid(pn.reshape(-1)))
    if per:                                            # unwrap coarse coords
        gsz_p = (2 ** jnp.maximum(lvl_b - 1, 0)).astype(jnp.int32)[:, None]
        ioc = IMGc[img_C]
        cx = cx + ioc[:, :, 0] * gsz_p
        cy = cy + ioc[:, :, 1] * gsz_p
        cz = cz + ioc[:, :, 2] * gsz_p
    adj_C = (_adj_coarse_ax(bx[:, None], cx) & _adj_coarse_ax(by[:, None], cy)
             & _adj_coarse_ax(bz[:, None], cz))
    x_valid = (active_b[:, None] & (lvl_b[:, None] > 1) & pn_valid
               & found_C & leaf_g[orig_C] & ~adj_C)
    X_pairs, n_X = _blk_compact(jnp.broadcast_to(t_idx[:, None], (block, 26)),
                                (orig_C + max_boxes * img_C) if per else orig_C,
                                x_valid, block * 27)

    # W-list: finer children of colleagues, NON-adjacent (source leaf or internal)
    neighs_S, valid_S, img_S = get_26_neighbors_periodic(m_b, lvl_b, periodic_axes)
    child_off = jnp.arange(8, dtype=m_id.dtype)
    child_m = neighs_S[:, :, None] * one + child_off[None, None, :]
    ids_F = ((lvl_b + 1)[:, None, None].astype(m_id.dtype) << level_shift) | child_m
    orig_F, found_F = _blk_lookup(ids_F, sorted_box_ids, sorted_order,
                                  active_box_ids, max_boxes)
    fx, fy, fz = (g.astype(jnp.int32).reshape(block, 26, 8)
                  for g in _morton_to_grid(child_m.reshape(-1)))
    if per:                                            # children inherit the image
        gsz_f = (2 ** (lvl_b + 1)).astype(jnp.int32)[:, None, None]
        iof = IMGc[img_S][:, :, None, :]
        fx = fx + iof[:, :, :, 0] * gsz_f
        fy = fy + iof[:, :, :, 1] * gsz_f
        fz = fz + iof[:, :, :, 2] * gsz_f
    adj_F = (_adj_fine_ax(bx[:, None, None], fx) & _adj_fine_ax(by[:, None, None], fy)
             & _adj_fine_ax(bz[:, None, None], fz))
    w_valid = leaf_b[:, None, None] & valid_S[:, :, None] & found_F & ~adj_F
    W_pairs, n_W = _blk_compact(jnp.broadcast_to(t_idx[:, None, None], (block, 26, 8)),
                                (orig_F + max_boxes * jnp.broadcast_to(
                                    img_S[:, :, None], orig_F.shape)) if per else orig_F,
                                w_valid, block * 152)
    return W_pairs, n_W, X_pairs, n_X


def _eager_blocks(m_id, lvls, is_leaf, max_boxes, max_depth, block):
    r"""Pad the box arrays to a block multiple (inactive sentinel) and yield block slices."""
    n_blk = -(-max_boxes // block)
    pad = n_blk * block - max_boxes
    m_pad = jnp.pad(m_id, (0, pad))
    lvl_pad = jnp.pad(lvls, (0, pad))                  # lvl 0 = inactive: never a valid target
    leaf_pad = jnp.pad(is_leaf & (lvls > 0), (0, pad))
    for b in range(n_blk):
        lo = b * block
        yield (m_pad[lo:lo + block], lvl_pad[lo:lo + block], leaf_pad[lo:lo + block],
               jnp.int32(lo))


def build_v_eager(m_id, lvls, is_leaf, max_boxes, max_depth, block=_BLOCK, concat=True,
                  periodic_axes=()):
    r"""Compact V-list as a host int32 array, built block-by-block (bounded device peak).

    Parameters
    ----------
    m_id, lvls, is_leaf, max_boxes, max_depth
        As in :func:`build_interaction_lists`.
    block : int, optional
        Target boxes per block.
    concat : bool, optional
        ``False`` returns the per-block list instead — consumers that stream
        the pairs (m2l grouping) then never hold the full list on device.
    periodic_axes : tuple of int, optional
        Wrapping axes (3-column output).

    Returns
    -------
    np.ndarray or list of np.ndarray
        Compact pair array ``(n_pairs, 2)`` (open) or ``(n_pairs, 3)``
        (periodic), or the per-block list when ``concat=False``.
    """
    import numpy as _np
    tabs = _sorted_boxids(m_id, lvls, max_depth)
    parts = []
    for m_b, lvl_b, leaf_b, tgt0 in _eager_blocks(m_id, lvls, is_leaf, max_boxes,
                                                  max_depth, block):
        pairs, n = _v_block(m_b, lvl_b, tgt0, tabs, m_id, lvls, is_leaf,
                            max_boxes=max_boxes, max_depth=max_depth, block=block,
                            periodic_axes=periodic_axes)
        n = int(n)
        if n:
            parts.append(_np.asarray(pairs[:n]))
    if not concat:
        return parts
    ncol = 3 if len(periodic_axes) > 0 else 2
    return (_np.concatenate(parts) if parts else _np.empty((0, ncol), _np.int32))


def build_ulist_eager(m_id, lvls, is_leaf, max_boxes, max_depth, block=_BLOCK,
                      periodic_axes=()):
    r"""Compact adaptive U-list as a host int32 array (block-chunked :func:`build_ulist_adaptive`).

    Parameters
    ----------
    m_id, lvls, is_leaf, max_boxes, max_depth
        As in :func:`build_interaction_lists`.
    block : int, optional
        Target boxes per block.
    periodic_axes : tuple of int, optional
        Wrapping axes (sources become image-encoded:
        ``src + max_boxes * image``).

    Returns
    -------
    U_list : (n_pairs, 2) np.ndarray of int32
        Compact ``(target_leaf, source_leaf)`` pairs; excludes self pairs.
    """
    import numpy as _np
    tabs = _sorted_boxids(m_id, lvls, max_depth)
    parts = []
    for m_b, lvl_b, leaf_b, tgt0 in _eager_blocks(m_id, lvls, is_leaf, max_boxes,
                                                  max_depth, block):
        pairs, n = _ua_block(m_b, lvl_b, leaf_b, tgt0, tabs, m_id, lvls, is_leaf,
                             max_boxes=max_boxes, max_depth=max_depth, block=block,
                             periodic_axes=periodic_axes)
        n = int(n)
        if n > block * 64:                             # same bound as build_ulist_adaptive
            raise ValueError("build_ulist_eager: U-list block overflow")
        if n:
            parts.append(_np.asarray(pairs[:n]))
    return (_np.concatenate(parts) if parts else _np.empty((0, 2), _np.int32))


def build_wx_eager(m_id, lvls, is_leaf, max_boxes, max_depth, block=_BLOCK,
                   periodic_axes=()):
    r"""Compact W/X lists as host int32 arrays (block-chunked :func:`build_wx_lists`).

    Parameters
    ----------
    m_id, lvls, is_leaf, max_boxes, max_depth
        As in :func:`build_interaction_lists`.
    block : int, optional
        Target boxes per block.
    periodic_axes : tuple of int, optional
        Wrapping axes (sources become image-encoded:
        ``src + max_boxes * image``).

    Returns
    -------
    W_list : (n_W, 2) np.ndarray of int32
        Compact ``(target_leaf, source_box)`` pairs.
    X_list : (n_X, 2) np.ndarray of int32
        Compact ``(target_box, source_leaf)`` pairs.
    """
    import numpy as _np
    tabs = _sorted_boxids(m_id, lvls, max_depth)
    Wp, Xp = [], []
    for m_b, lvl_b, leaf_b, tgt0 in _eager_blocks(m_id, lvls, is_leaf, max_boxes,
                                                  max_depth, block):
        W_pairs, n_W, X_pairs, n_X = _wx_block(m_b, lvl_b, leaf_b, tgt0, tabs, m_id, lvls,
                                               is_leaf, max_boxes=max_boxes,
                                               max_depth=max_depth, block=block,
                                               periodic_axes=periodic_axes)
        n_W, n_X = int(n_W), int(n_X)
        if n_W > block * 152 or n_X > block * 27:      # same bounds as build_wx_lists
            raise ValueError("build_wx_eager: W/X block overflow")
        if n_W:
            Wp.append(_np.asarray(W_pairs[:n_W]))
        if n_X:
            Xp.append(_np.asarray(X_pairs[:n_X]))
    W = _np.concatenate(Wp) if Wp else _np.empty((0, 2), _np.int32)
    X = _np.concatenate(Xp) if Xp else _np.empty((0, 2), _np.int32)
    return W, X
