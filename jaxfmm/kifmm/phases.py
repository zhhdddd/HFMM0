r"""KIFMM translation phases (P2M/M2M/M2L/L2L/L2P, W/X lists, lean and per-level
variants). Shapes follow the padded box layout of :mod:`jaxfmm.core`."""


from functools import partial

import jax
import jax.numpy as jnp

from jaxfmm.trees import _morton_to_grid, box_geometry   # box_geometry lives in trees

__all__ = ["box_geometry", "p2m", "p2m_padded", "p2m_lean", "m2m", "m2m_factored",
           "upward_pass", "m2l", "m2l_groups_padded", "m2l_batched", "m2l_groups_tiered",
           "m2l_batched_tiered", "v_pair_count", "m2p", "m2p_lean", "p2l", "p2l_lean",
           "l2l", "l2l_factored", "l2p", "l2p_padded", "l2p_lean", "downward_pass",
           "p2m_padded_leveled", "m2m_factored_leveled", "m2l_leveled",
           "l2l_factored_leveled", "p2l_leveled"]



def _decode(X, dec):
    r"""Row-batched check→equivalent decode :math:`X (A^{+})^T`.

    Accepts EITHER the materialised pseudoinverse (n_s, n_s) or the
    backward-stable FACTOR triple ``(U, d, Vt)`` from
    :func:`jaxfmm.kifmm.operators._pinv_factors`.  The factored form applies
    orthogonal matvec → diagonal → orthogonal matvec, never touching the
    cond-sized products of the materialised matrix — it is what makes f32
    KIFMM viable at P=3-4 and lifts the f64 ceiling at high P
    (pvfmm cicp15 §2.1)."""
    if isinstance(dec, (tuple, list)):
        U, d, Vt = dec
        return ((X @ U) * d[None, :]) @ Vt
    return X @ dec.T


def _green(r):
    r"""Laplace Green's function :math:`1/(4\pi r)` in r's own dtype (0 at r=0).

    Computing the :math:`1/(4\pi)` constant in ``r.dtype`` keeps a float32
    distance from being upcast to float64, so the single-precision path stays
    in float32 throughout.
    """
    inv4pi = jnp.asarray(0.25 / jnp.pi, r.dtype)
    return jnp.where(r > 16 * jnp.finfo(r.dtype).eps * jnp.max(r),   # ULP-relative floor
                     inv4pi / r, jnp.asarray(0.0, r.dtype))


def _green_grad(d):
    r"""Gradient kernel :math:`-\nabla\phi` of the Laplace Green's function.

    Returns :math:`d/(4\pi |d|^3)` for a displacement ``d = eval - src``
    (…, 3), 0 at d=0 — the same convention as the near field's ``field=True``
    (:func:`jaxfmm.near.eval_direct`), so far + near add coherently."""
    r2 = jnp.sum(d * d, axis=-1)
    inv4pi = jnp.asarray(0.25 / jnp.pi, d.dtype)
    fl = (16 * jnp.finfo(r2.dtype).eps)**2 * jnp.max(r2)             # ULP-relative floor
    ok = r2 > fl
    w = jnp.where(ok, inv4pi * jnp.where(ok, r2, 1.0) ** -1.5, 0.0)
    return d * w[..., None]                                  # (..., 3)


def _F_at(F_lv, l):
    r"""Per-level factored decode triple: dynamic level index into the stacked factors."""
    U, d, Vt = F_lv
    return (U[l], d[l], Vt[l])


def _decode_leveled(X, F_lv, lvls, max_depth):
    r"""Row-wise factored decode with a PER-LEVEL operator (leveled kernels): static
    python loop over levels, masked accumulate — at most ``depth`` cheap GEMMs, no
    per-box operator materialisation."""
    out = jnp.zeros_like(X)
    for l in range(max_depth + 1):
        out = out + jnp.where((lvls == l)[:, None], _decode(X, _F_at(F_lv, l)), 0.0)
    return out


@partial(jax.jit, static_argnames=["batch_size", "max_depth", "g"])
def p2m_padded_leveled(ppts, pchg, centres, h, is_leaf, lvls, check_pts, F_up_lv,
                       max_depth, batch_size=64, g=_green):
    r""":func:`p2m_padded` for PER-LEVEL operators (non-homogeneous kernels): physical
    check potentials as usual, then the level's own decode — NO h rescale factor
    (that factor IS the degree -1 homogeneity shortcut).

    Parameters
    ----------
    ppts : (max_boxes, N_max, 3) ndarray
        Padded per-box source points (:func:`jaxfmm.near._build_padded`).
    pchg : (max_boxes, N_max) ndarray
        Padded per-box charges (0 in padding slots).
    centres : (max_boxes, 3) ndarray
        Box centres.
    h : (max_boxes,) ndarray
        Box half-sides.
    is_leaf : (max_boxes,) ndarray of bool
        Leaf mask.
    lvls : (max_boxes,) ndarray of int32
        Box levels.
    check_pts : (n_s, 3) ndarray
        Upward check surface (unit box).
    F_up_lv : (U, d, Vt) tuple
        Level-stacked upward factored decodes, shapes
        ``((max_depth+1, n_s, n_s), (max_depth+1, n_s), (max_depth+1, n_s, n_s))``
        (``F_up_lv`` from :func:`jaxfmm.kifmm.operators.build_operators_leveled`).
    max_depth : int
        Static tree depth.
    batch_size : int, optional
        Boxes per chunk (static), default 64.
    g : callable, optional
        Kernel function.

    Returns
    -------
    equiv : (max_boxes, n_s) ndarray
        Upward-equivalent density per box (0 for non-leaves).
    """
    def per_box(args):
        centre, hb, pts, q, leaf = args
        check_phys = check_pts * hb + centre
        d = check_phys[:, None, :] - pts[None, :, :]
        K = g(jnp.sqrt(jnp.sum(d * d, axis=-1)))
        return jnp.where(leaf, K @ q, 0.0)
    check = jax.lax.map(per_box, (centres, h, ppts, pchg, is_leaf), batch_size=batch_size)
    return _decode_leveled(check, F_up_lv, lvls, max_depth)


@partial(jax.jit, static_argnames=["max_depth"])
def m2m_factored_leveled(equiv, m_id, p_id, lvls, M2M_K_lv, F_up_lv, max_depth):
    r""":func:`m2m_factored` with PER-LEVEL kernels/decodes: iteration ``i`` merges child
    level ``max_depth - i`` using the PARENT level's tables (``M2M_K_lv`` indexed by
    parent).

    Parameters
    ----------
    equiv : (max_boxes, n_s) ndarray
        Leaf P2M densities (filled in for leaves).
    m_id : (max_boxes,) ndarray of uint64
        Morton ids.
    p_id : (max_boxes,) ndarray of int32
        Parent box index.
    lvls : (max_boxes,) ndarray of int32
        Box levels.
    M2M_K_lv : (max_depth, 8, n_s, n_s) ndarray
        Per-level raw child→parent-check kernels, indexed by PARENT level
        (from :func:`jaxfmm.kifmm.operators.build_operators_leveled`).
    F_up_lv : (U, d, Vt) tuple
        Level-stacked upward factored decodes, shapes
        ``((max_depth+1, n_s, n_s), (max_depth+1, n_s), (max_depth+1, n_s, n_s))``
        (indexed by parent level).
    max_depth : int
        Static tree depth.

    Returns
    -------
    equiv : (max_boxes, n_s) ndarray
        Equivalent density for every (leaf and internal) box.
    """
    n_s = equiv.shape[1]
    octant = (m_id % jnp.uint64(8)).astype(jnp.int32)
    rows = jnp.arange(equiv.shape[0])

    def merge_level(i, eq):
        child_lvl = max_depth - i
        parent = child_lvl - 1
        K_stack = M2M_K_lv[parent].reshape(8 * n_s, n_s)
        big = (eq @ K_stack.T).reshape(-1, 8, n_s)
        contrib = jnp.where((lvls == child_lvl)[:, None], big[rows, octant], 0.0)
        check = jnp.zeros_like(eq).at[p_id].add(contrib)
        solved = _decode(check, _F_at(F_up_lv, parent))
        return eq + jnp.where((lvls == parent)[:, None], solved, 0.0)

    return jax.lax.fori_loop(0, max_depth - 1, merge_level, equiv)


@partial(jax.jit, static_argnames=["batch_size", "n_pairs", "max_depth"])
def m2l_leveled(equiv, V_list, m_id, lvls, M2L_K_lv, F_dn_lv, max_depth,
                n_pairs=None, batch_size=64):
    r""":func:`m2l` with PER-LEVEL kernel tables: the pair's matrix is selected by
    (target level, displacement); the downward decode uses the level's pinv.

    Parameters
    ----------
    equiv : (max_boxes, n_s) ndarray
        Upward equivalent densities (from :func:`upward_pass`).
    V_list : (max_V, 2) ndarray of int32
        (target, source) pairs, -1 padded (no periodic 3-column form here).
    m_id : (max_boxes,) ndarray of uint64
        Morton ids.
    lvls : (max_boxes,) ndarray of int32
        Box levels (select the target level's kernel table and decode).
    M2L_K_lv : (max_depth+1, 343, n_s, n_s) ndarray
        Per-level raw M2L kernel tables (343 displacements per level, from
        :func:`jaxfmm.kifmm.operators.build_operators_leveled`).
    F_dn_lv : (U, d, Vt) tuple
        Level-stacked downward factored decodes, shapes
        ``((max_depth+1, n_s, n_s), (max_depth+1, n_s), (max_depth+1, n_s, n_s))``.
    max_depth : int
        Static tree depth.
    n_pairs : int or None, optional
        Compacted pair-buffer size (static, see :func:`v_pair_count`);
        ``None`` evaluates the full padded ``V_list``.
    batch_size : int, optional
        Pairs per chunk (static), default 64.

    Returns
    -------
    local : (max_boxes, n_s) ndarray
        Downward equivalent density per box.
    """
    n_s = equiv.shape[1]
    max_boxes = equiv.shape[0]
    K_flat = M2L_K_lv.reshape((max_depth + 1) * 343, n_s, n_s)
    gx, gy, gz = (c.astype(jnp.int32) for c in _morton_to_grid(m_id))
    tgt, src = V_list[:, 0], V_list[:, 1]
    if n_pairs is not None:
        v_ok = (tgt >= 0) & (src >= 0)
        dst = jnp.cumsum(v_ok.astype(jnp.int32)) - 1
        dst = jnp.where(v_ok, dst, n_pairs)
        tgt = jnp.full((n_pairs,), -1, jnp.int32).at[dst].set(tgt, mode="drop")
        src = jnp.full((n_pairs,), -1, jnp.int32).at[dst].set(src, mode="drop")
    valid = (tgt >= 0) & (src >= 0)
    tgt_s = jnp.where(valid, tgt, 0)
    src_s = jnp.where(valid, src, 0)
    dx = gx[tgt_s] - gx[src_s]; dy = gy[tgt_s] - gy[src_s]; dz = gz[tgt_s] - gz[src_s]
    disp_id = (dx + 3) * 49 + (dy + 3) * 7 + (dz + 3)
    kidx = jnp.where(valid, lvls[tgt_s] * 343 + disp_id, 0)

    def per_pair(args):
        k_id, s_idx, ok = args
        return jnp.where(ok, K_flat[k_id] @ equiv[s_idx], 0.0)

    contribs = jax.lax.map(per_pair, (kidx, src_s, valid), batch_size=batch_size)
    scatter_tgt = jnp.where(valid, tgt_s, max_boxes)
    check_pot = jnp.zeros((max_boxes, n_s), dtype=equiv.dtype)
    check_pot = check_pot.at[scatter_tgt].add(contribs, mode="drop")
    return _decode_leveled(check_pot, F_dn_lv, lvls, max_depth)


@partial(jax.jit, static_argnames=["max_depth"])
def l2l_factored_leveled(local, m_id, p_id, lvls, L2L_K_lv, F_dn_lv, max_depth):
    r""":func:`l2l_factored` with PER-LEVEL kernels/decodes (child-level tables).

    Parameters
    ----------
    local : (max_boxes, n_s) ndarray
        Downward equivalent densities seeded by M2L.
    m_id : (max_boxes,) ndarray of uint64
        Morton ids.
    p_id : (max_boxes,) ndarray of int32
        Parent box index.
    lvls : (max_boxes,) ndarray of int32
        Box levels.
    L2L_K_lv : (max_depth, 8, n_s, n_s) ndarray
        Per-level raw parent→child-check kernels, index ``cl - 1`` for child
        level ``cl`` (from :func:`jaxfmm.kifmm.operators.build_operators_leveled`).
    F_dn_lv : (U, d, Vt) tuple
        Level-stacked downward factored decodes, shapes
        ``((max_depth+1, n_s, n_s), (max_depth+1, n_s), (max_depth+1, n_s, n_s))``
        (indexed by child level).
    max_depth : int
        Static tree depth.

    Returns
    -------
    local : (max_boxes, n_s) ndarray
        Downward densities with parents pushed to children.
    """
    n_s = local.shape[1]
    octant = (m_id % jnp.uint64(8)).astype(jnp.int32)
    rows = jnp.arange(local.shape[0])

    def push_level(i, loc):
        child_lvl = i + 1
        K_stack = L2L_K_lv[child_lvl - 1].reshape(8 * n_s, n_s)
        big = (loc[p_id] @ K_stack.T).reshape(-1, 8, n_s)
        add = _decode(big[rows, octant], _F_at(F_dn_lv, child_lvl))
        return loc + jnp.where((lvls == child_lvl)[:, None], add, 0.0)

    return jax.lax.fori_loop(1, max_depth, push_level, local)


@partial(jax.jit, static_argnames=["batch_size", "g", "g_grad"])
def p2m(positions, charges, particles_in_leaf, centres, h, is_leaf,
        equiv_pts, check_pts, pinv_up, batch_size=64, g=_green, g_grad=_green_grad):
    r"""Particle-to-multipole for all leaf boxes (vectorised, chunked over boxes).

    Evaluates each leaf's sources on the physical upward check surface and
    decodes to the upward-equivalent density,

    .. math:: \sigma_b = h_b \, K(\mathrm{check\_up}, \mathrm{equiv\_up})^{+}
              \, K(\mathrm{check\_phys}, x)\, q,

    with the half-side factor :math:`h_b` from the degree :math:`-1` kernel
    homogeneity (see the :mod:`jaxfmm.kifmm` scaling convention).

    Parameters
    ----------
    positions : (N, 3) ndarray
        Particle coordinates.
    charges : (N,) ndarray
        Particle charges.
    particles_in_leaf : (max_boxes, N_max) ndarray of int32
        Per-leaf particle indices, -1 padded (from ``assign_particles``).
    centres : (max_boxes, 3) ndarray
        Box centres.
    h : (max_boxes,) ndarray
        Box half-sides.
    is_leaf : (max_boxes,) ndarray of bool
        Leaf mask.
    equiv_pts : (n_s, 3) ndarray
        Upward equivalent surface (unit box).
    check_pts : (n_s, 3) ndarray
        Upward check surface (unit box).
    pinv_up : (n_s, n_s) ndarray or (U, d, Vt) tuple
        Upward decode (materialised or factored, see :func:`_decode`).
    batch_size : int, optional
        Boxes per chunk (static; keep small).
    g, g_grad : callable, optional
        Kernel and gradient-kernel functions.

    Returns
    -------
    equiv : (max_boxes, n_s) ndarray
        Upward-equivalent density per box (0 for non-leaves).
    """
    n_s = equiv_pts.shape[0]
    max_boxes = centres.shape[0]

    pil = particles_in_leaf                               # (max_boxes, N_max)
    valid = pil >= 0                                      # (max_boxes, N_max)
    pil_safe = jnp.where(valid, pil, 0)

    def per_box(args):
        centre, hb, idx, vmask, leaf = args
        src = positions[idx]                              # (N_max, 3)
        q = jnp.where(vmask, charges[idx], 0.0)           # (N_max,)
        check_phys = check_pts * hb + centre              # (n_s, 3)
        d = check_phys[:, None, :] - src[None, :, :]      # (n_s, N_max, 3)
        r = jnp.sqrt(jnp.sum(d * d, axis=-1))             # (n_s, N_max)
        K = g(r)
        check_pot = K @ q                                 # (n_s,)
        return jnp.where(leaf, check_pot, 0.0)

    check = jax.lax.map(
        per_box, (centres, h, pil_safe, valid, is_leaf), batch_size=batch_size
    )                                                     # (max_boxes, n_s)
    return h[:, None] * _decode(check, pinv_up)           # dens = hb * (pinv_up @ check)


@partial(jax.jit, static_argnames=["batch_size", "g", "g_grad"])
def p2m_padded(ppts, pchg, centres, h, is_leaf, equiv_pts, check_pts, pinv_up,
               batch_size=64, g=_green, g_grad=_green_grad):
    r""":func:`p2m` consuming the shared padded per-box layout (``ppts``/``pchg`` from
    :func:`jaxfmm.near._build_padded`, as carried by :class:`jaxfmm.core.SourceData`):
    no per-box re-gather of positions/charges inside the map (padding slots carry
    charge 0, so they drop out).  Same result as :func:`p2m`.

    Parameters
    ----------
    ppts : (max_boxes, N_max, 3) ndarray
        Padded per-box source points (:func:`jaxfmm.near._build_padded`).
    pchg : (max_boxes, N_max) ndarray
        Padded per-box charges (0 in padding slots).
    centres : (max_boxes, 3) ndarray
        Box centres.
    h : (max_boxes,) ndarray
        Box half-sides.
    is_leaf : (max_boxes,) ndarray of bool
        Leaf mask.
    equiv_pts : (n_s, 3) ndarray
        Upward equivalent surface (unit box; kept for signature parity with
        :func:`p2m`, not used in the body).
    check_pts : (n_s, 3) ndarray
        Upward check surface (unit box).
    pinv_up : (n_s, n_s) ndarray or (U, d, Vt) tuple
        Upward decode (materialised or factored, see :func:`_decode`).
    batch_size : int, optional
        Boxes per chunk (static; keep small).
    g, g_grad : callable, optional
        Kernel and gradient-kernel functions.

    Returns
    -------
    equiv : (max_boxes, n_s) ndarray
        Upward-equivalent density per box (0 for non-leaves).
    """
    def per_box(args):
        centre, hb, pts, q, leaf = args
        check_phys = check_pts * hb + centre              # (n_s, 3)
        d = check_phys[:, None, :] - pts[None, :, :]      # (n_s, N_max, 3)
        K = g(jnp.sqrt(jnp.sum(d * d, axis=-1)))
        return jnp.where(leaf, K @ q, 0.0)                # (n_s,) check potential

    check = jax.lax.map(per_box, (centres, h, ppts, pchg, is_leaf),
                        batch_size=batch_size)            # (max_boxes, n_s)
    return h[:, None] * _decode(check, pinv_up)           # dens = hb * (pinv_up @ check)


def p2m_lean(pos_s, q_s, starts, cnts, w, centres, h, is_leaf, equiv_pts, check_pts,
             pinv_up, batch_size=64, g=_green, g_grad=_green_grad):
    r""":func:`p2m_padded` without the padded layout (memory diet): the points live in
    LEAF-MAJOR SORTED order, so each mapped box reads the contiguous slice
    ``pos_s[start[b] + iota(w)]`` — arithmetic single-level gathers that fuse (a composed
    ``positions[pil[b]]`` does not; see the near field's lean mode).  Out-of-count slots
    carry charge 0 and drop out — same result as :func:`p2m_padded`.

    Parameters
    ----------
    pos_s : (N, 3) ndarray
        Particle coordinates in LEAF-MAJOR SORTED order.
    q_s : (N,) ndarray
        Particle charges, sorted like ``pos_s``.
    starts : (max_boxes,) ndarray of int32
        Per-box slice start into the sorted arrays.
    cnts : (max_boxes,) ndarray of int32
        Per-box particle count.
    w : int
        Static slice width (max particles per leaf).
    centres : (max_boxes, 3) ndarray
        Box centres.
    h : (max_boxes,) ndarray
        Box half-sides.
    is_leaf : (max_boxes,) ndarray of bool
        Leaf mask.
    equiv_pts : (n_s, 3) ndarray
        Upward equivalent surface (unit box; kept for signature parity with
        :func:`p2m`, not used in the body).
    check_pts : (n_s, 3) ndarray
        Upward check surface (unit box).
    pinv_up : (n_s, n_s) ndarray or (U, d, Vt) tuple
        Upward decode (materialised or factored, see :func:`_decode`).
    batch_size : int, optional
        Boxes per chunk (static; keep small).
    g, g_grad : callable, optional
        Kernel and gradient-kernel functions.

    Returns
    -------
    equiv : (max_boxes, n_s) ndarray
        Upward-equivalent density per box (0 for non-leaves).
    """
    N = pos_s.shape[0]
    iw = jnp.arange(w, dtype=jnp.int32)

    def per_box(args):
        centre, hb, st, ct, leaf = args
        sl = jnp.clip(st + iw, 0, N - 1)                  # (w,) contiguous slice indices
        pts = pos_s[sl]                                   # (w, 3)
        q = jnp.where(iw < ct, q_s[sl], 0.0)
        check_phys = check_pts * hb + centre              # (n_s, 3)
        d = check_phys[:, None, :] - pts[None, :, :]      # (n_s, w, 3)
        K = g(jnp.sqrt(jnp.sum(d * d, axis=-1)))
        return jnp.where(leaf, K @ q, 0.0)                # (n_s,) check potential

    check = jax.lax.map(per_box, (centres, h, starts, cnts, is_leaf),
                        batch_size=batch_size)            # (max_boxes, n_s)
    return h[:, None] * _decode(check, pinv_up)           # dens = hb * (pinv_up @ check)


@partial(jax.jit, static_argnames=["max_depth"])
def m2m(equiv, m_id, p_id, lvls, M2M_mats, max_depth):
    r"""Multipole-to-multipole accumulation, bottom-up over levels.

    Adds each box's contribution to its parent's equivalent density,
    :math:`\sigma_\mathrm{parent} \mathrel{+}= \mathrm{M2M}_k\,
    \sigma_\mathrm{child}`, processing levels from finest to coarsest so
    children are complete before their parents.

    Parameters
    ----------
    equiv : (max_boxes, n_s) ndarray
        Leaf P2M densities (filled in for leaves).
    m_id : (max_boxes,) ndarray of uint64
        Morton ids.
    p_id : (max_boxes,) ndarray of int32
        Parent box index.
    lvls : (max_boxes,) ndarray of int32
        Box levels.
    M2M_mats : (8, n_s, n_s) ndarray
        Per-octant child→parent operators.
    max_depth : int
        Static tree depth.

    Returns
    -------
    equiv : (max_boxes, n_s) ndarray
        Equivalent density for every (leaf and internal) box.
    """
    n_s = equiv.shape[1]
    octant = (m_id % jnp.uint64(8)).astype(jnp.int32)     # (max_boxes,)
    rows = jnp.arange(equiv.shape[0])
    # Stack the 8 octant operators so one GEMM produces all of them at once; each box
    # then gathers its own octant's result.  Avoids the per-octant full-size GEMM +
    # scatter (an 8x redundancy): one stacked GEMM + one gather + one scatter per level.
    M2M_stack = M2M_mats.reshape(8 * n_s, n_s)            # (8·n_s, n_s)

    def merge_level(i, equiv):
        # Iteration i merges child level (max_depth - i) into parent (max_depth - i - 1).
        child_lvl = max_depth - i
        is_child = (lvls == child_lvl)                    # (max_boxes,)
        big = (equiv @ M2M_stack.T).reshape(-1, 8, n_s)   # (max_boxes, 8, n_s)
        contrib = big[rows, octant]                       # per-box octant operator
        contrib = jnp.where(is_child[:, None], contrib, 0.0)
        return equiv.at[p_id].add(contrib)

    # i = 0 .. max_depth-2  (parents reach down to level 1, the coarsest)
    equiv = jax.lax.fori_loop(0, max_depth - 1, merge_level, equiv)
    return equiv


@partial(jax.jit, static_argnames=["max_depth"])
def m2m_factored(equiv, m_id, p_id, lvls, M2M_K, F_par, max_depth):
    r"""Bottom-up M2M in the backward-stable FACTORED form: raw child→parent-check
    kernel matvecs (``M2M_K`` from :func:`jaxfmm.kifmm.operators.build_operators`), then a
    per-level factored decode (``F_par``) — never the materialised ``pinv @ K``
    product.  Same result as :func:`m2m` in f64 at P<=6; required for f32 and for
    deep f64 accuracy at high P (see :func:`_decode`).

    Parameters
    ----------
    equiv : (max_boxes, n_s) ndarray
        Leaf P2M densities (filled in for leaves).
    m_id : (max_boxes,) ndarray of uint64
        Morton ids.
    p_id : (max_boxes,) ndarray of int32
        Parent box index.
    lvls : (max_boxes,) ndarray of int32
        Box levels.
    M2M_K : (8, n_s, n_s) ndarray
        Per-octant raw child→parent-check kernels (``M2M_K`` from
        :func:`jaxfmm.kifmm.operators.build_operators`).
    F_par : (U, d, Vt) tuple or (n_s, n_s) ndarray
        Parent upward decode (factored or materialised, see :func:`_decode`).
    max_depth : int
        Static tree depth.

    Returns
    -------
    equiv : (max_boxes, n_s) ndarray
        Equivalent density for every (leaf and internal) box.
    """
    n_s = equiv.shape[1]
    octant = (m_id % jnp.uint64(8)).astype(jnp.int32)
    rows = jnp.arange(equiv.shape[0])
    K_stack = M2M_K.reshape(8 * n_s, n_s)                 # stacked raw kernels

    def merge_level(i, eq):
        child_lvl = max_depth - i
        big = (eq @ K_stack.T).reshape(-1, 8, n_s)        # child equiv -> parent check
        contrib = jnp.where((lvls == child_lvl)[:, None], big[rows, octant], 0.0)
        check = jnp.zeros_like(eq).at[p_id].add(contrib)  # parent check potentials
        solved = _decode(check, F_par)
        return eq + jnp.where((lvls == child_lvl - 1)[:, None], solved, 0.0)

    return jax.lax.fori_loop(0, max_depth - 1, merge_level, equiv)


def upward_pass(positions, charges, particles_in_leaf, m_id, p_id, lvls,
                is_leaf, domain_extent, pmin, ops, batch_size=64, max_depth=None):
    r"""Full upward pass: P2M on leaves followed by bottom-up M2M.

    Uses the factored decodes when the ops dict carries them (``F_up``/
    ``F_par``), the materialised legacy operators otherwise.

    Parameters
    ----------
    positions, charges, particles_in_leaf : ndarray
        Sources and per-leaf indices (see :func:`p2m`).
    m_id, p_id, lvls, is_leaf : ndarray
        Tree arrays.
    domain_extent : float
        Root box edge length.
    pmin : (3,) ndarray
        Root box minimum corner.
    ops : dict
        Operator tables from :func:`jaxfmm.kifmm.operators.build_operators`.
    batch_size : int, optional
        Boxes per chunk.
    max_depth : int or None, optional
        Defaults to ``int(lvls.max())``; pass it explicitly (the static tree
        depth) to keep ``m2m``'s compiled shape fixed across geometries —
        needed for the no-recompile fixed-shape volume path.

    Returns
    -------
    equiv : (max_boxes, n_s) ndarray
        Upward-equivalent density for every box.
    centres : (max_boxes, 3) ndarray
        Box centres.
    h : (max_boxes,) ndarray
        Box half-sides.
    """
    centres, h = box_geometry(m_id, lvls, domain_extent, pmin)
    factored = "F_up" in ops                       # backward-stable decodes when available
    dec_up = (tuple(jnp.asarray(x) for x in ops["F_up"]) if factored
              else jnp.asarray(ops["pinv_up"]))
    equiv = p2m(positions, charges, particles_in_leaf, centres, h, is_leaf,
                jnp.asarray(ops["equiv_pts"]), jnp.asarray(ops["check_pts"]),
                dec_up, batch_size=batch_size)
    if max_depth is None:
        max_depth = int(lvls.max())
    if factored:
        equiv = m2m_factored(equiv, m_id, p_id, lvls, jnp.asarray(ops["M2M_K"]),
                             tuple(jnp.asarray(x) for x in ops["F_par"]), max_depth)
    else:
        equiv = m2m(equiv, m_id, p_id, lvls, jnp.asarray(ops["M2M_mats"]), max_depth)
    return equiv, centres, h


# ──────────────────────────────────────────────────────────────────────────
# Interaction phase
# ──────────────────────────────────────────────────────────────────────────
def v_pair_count(V_list, batch_size=64, granularity=None):
    r"""Static-friendly compacted V-list size (valid pairs rounded up to granularity).

    Parameters
    ----------
    V_list : (max_V, 2) or (max_V, 3) ndarray of int32
        (target, source[, img_idx]) pairs, -1 padded (read on the host).
    batch_size : int, optional
        The evaluator's chunk size; sets the default granularity floor.
    granularity : int or None, optional
        Round the valid-pair count up to this multiple; defaults to
        ``max(batch_size, 4096)``.

    Returns
    -------
    n_pairs : int
        Compacted pair-buffer size for :func:`m2l`'s ``n_pairs``.
    """
    import numpy as _np
    n_v = int((_np.asarray(V_list)[:, 0] >= 0).sum())
    gran = granularity if granularity is not None else max(batch_size, 4096)
    return _round_up(n_v, gran)


@partial(jax.jit, static_argnames=["batch_size", "n_pairs"])
def m2l(equiv, V_list, m_id, M2L_K_table, pinv_dn, n_pairs=None, batch_size=64,
        lvls=None):
    r"""Multipole-to-local over the V-list — FACTORED, backward-stable form.

    For each V-list pair :math:`(t, s)` at the same level, the displacement
    :math:`d = \mathrm{grid}(t) - \mathrm{grid}(s) \in [-3, 3]^3` selects the
    RAW kernel matrix ``M2L_K_table[d]``; the source's upward equivalent
    density is mapped onto the target's downward CHECK potential, and
    ``pinv_dn`` is applied ONCE per target box to obtain the local
    (downward-equivalent) density:

    .. math::
       \mathrm{check}[t] = \sum_{s \in V(t)} K_{d(t,s)}\, \mathrm{equiv}[s],
       \qquad
       \mathrm{local}[t] = \mathrm{pinv\_dn}\; \mathrm{check}[t].

    This is numerically more robust than folding the pinv into every operator,
    and is the same structure used by FFT-accelerated M2L.

    Parameters
    ----------
    equiv : (max_boxes, n_s) ndarray
        Upward equivalent densities (from :func:`upward_pass`).
    V_list : (max_V, 2) or (max_V, 3) ndarray of int32
        (target, source) pairs, -1 padded; a third ``img_idx`` column marks
        the periodic wrapped form (requires ``lvls``).
    m_id : (max_boxes,) ndarray of uint64
        Morton ids.
    M2L_K_table : (7, 7, 7, n_s, n_s) ndarray
        Raw kernel matrices.
    pinv_dn : (n_s, n_s) ndarray or (U, d, Vt) tuple
        Downward decode (applied per box).
    n_pairs : int or None, optional
        Compacted pair-buffer size (static).  The ``V_list`` buffer
        (``max_boxes * 189``) is mostly padding; like p2p, passing ``n_pairs``
        compacts the valid pairs into a tight buffer to avoid evaluating the
        padding rows (see :func:`v_pair_count`).
    batch_size : int, optional
        Pairs per chunk (static).
    lvls : (max_boxes,) ndarray of int32, optional
        Box levels; needed only for the periodic 3-column ``V_list``.

    Returns
    -------
    local : (max_boxes, n_s) ndarray
        Downward equivalent density per box.
    """
    n_s = equiv.shape[1]
    max_boxes = equiv.shape[0]
    K_flat = M2L_K_table.reshape(343, n_s, n_s)

    gx, gy, gz = _morton_to_grid(m_id)                 # (max_boxes,) uint64 each
    gx = gx.astype(jnp.int32); gy = gy.astype(jnp.int32); gz = gz.astype(jnp.int32)

    tgt = V_list[:, 0]
    src = V_list[:, 1]
    # periodic 3-column V-list (tgt, src, img_idx): the displacement must be computed on
    # UNWRAPPED source coords (src_grid + image * 2^level); requires ``lvls``.
    periodic = V_list.shape[1] == 3
    img = V_list[:, 2] if periodic else None
    if n_pairs is not None:
        # compact valid pairs into a tight n_pairs buffer
        v_ok = (tgt >= 0) & (src >= 0)
        dst = jnp.cumsum(v_ok.astype(jnp.int32)) - 1
        dst = jnp.where(v_ok, dst, n_pairs)
        tgt = jnp.full((n_pairs,), -1, jnp.int32).at[dst].set(tgt, mode="drop")
        src = jnp.full((n_pairs,), -1, jnp.int32).at[dst].set(src, mode="drop")
        if periodic:
            img = jnp.full((n_pairs,), 0, jnp.int32).at[dst].set(img, mode="drop")

    valid = (tgt >= 0) & (src >= 0)
    tgt_s = jnp.where(valid, tgt, 0)
    src_s = jnp.where(valid, src, 0)

    if periodic:
        from jaxfmm.trees.cubic import _IMG_COMBOS
        combos = jnp.asarray(_IMG_COMBOS, jnp.int32)
        grid = (2 ** lvls[src_s]).astype(jnp.int32)          # same level as the target
        io = combos[jnp.where(valid, img, 0)]                # (n, 3) wrap direction
        dx = gx[tgt_s] - (gx[src_s] + io[:, 0] * grid)
        dy = gy[tgt_s] - (gy[src_s] + io[:, 1] * grid)
        dz = gz[tgt_s] - (gz[src_s] + io[:, 2] * grid)
    else:
        dx = gx[tgt_s] - gx[src_s]
        dy = gy[tgt_s] - gy[src_s]
        dz = gz[tgt_s] - gz[src_s]
    disp_id = (dx + 3) * 49 + (dy + 3) * 7 + (dz + 3)  # in [0, 343)
    disp_id = jnp.where(valid, disp_id, 0)

    def per_pair(args):
        d_id, s_idx, ok = args
        contrib = K_flat[d_id] @ equiv[s_idx]          # (n_s,) → target check potential
        return jnp.where(ok, contrib, 0.0)

    contribs = jax.lax.map(
        per_pair, (disp_id, src_s, valid), batch_size=batch_size
    )                                                  # (n_pairs or max_V, n_s)

    scatter_tgt = jnp.where(valid, tgt_s, max_boxes)   # dump invalid rows
    check_pot = jnp.zeros((max_boxes, n_s), dtype=equiv.dtype)
    check_pot = check_pot.at[scatter_tgt].add(contribs, mode="drop")
    # apply the downward pseudoinverse ONCE per box: local = pinv_dn @ check_pot
    return _decode(check_pot, pinv_dn)


# ──────────────────────────────────────────────────────────────────────────
# Displacement-batched M2L (fast path).  The generic `m2l` above gathers a
# (n_s, n_s) kernel matrix per V-pair — memory-bound, far slower than needed.
# The M2L is a convolution over each level's regular box grid: every pair at
# the SAME displacement d shares the raw kernel K_d, so we group pairs by d
# (≤316 distinct), pad each group to a (n_disp, CAP) layout ONCE per tree
# (`m2l_groups_padded`), and apply each K_d exactly once via a single batched
# GEMM (`m2l_batched`, jittable).  Numerically identical to `m2l`.
# ──────────────────────────────────────────────────────────────────────────
def m2l_groups_padded(V_list, m_id, M2L_K_table):
    r"""Host-side (fixed-tree) precompute for :func:`m2l_batched`: group the valid
    V-pairs by displacement :math:`d = \mathrm{grid}(t) - \mathrm{grid}(s) \in
    [-3, 3]^3` and pad to a dense ``(n_disp, CAP)`` layout.  Returns
    ``(K_stack, src_pad, tgt_pad, valid)``.

    Parameters
    ----------
    V_list : (max_V, 2) ndarray of int32
        (target, source) pairs, -1 padded (read on the host).
    m_id : (max_boxes,) ndarray of uint64
        Morton ids.
    M2L_K_table : (7, 7, 7, n_s, n_s) ndarray
        Raw kernel matrices (read on the host).

    Returns
    -------
    K_stack : (n_disp, n_s, n_s) ndarray
        The raw kernel matrix of each occupied displacement group.
    src_pad : (n_disp, CAP) ndarray of int32
        Source box index per group slot (0 in padding).
    tgt_pad : (n_disp, CAP) ndarray of int32
        Target box index per group slot (-1 in padding).
    valid : (n_disp, CAP) ndarray of bool
        Occupied-slot mask.
    """
    import numpy as np
    n_s = M2L_K_table.shape[-1]
    K_flat = np.asarray(M2L_K_table).reshape(343, n_s, n_s)
    gx, gy, gz = (np.asarray(g).astype(np.int64) for g in _morton_to_grid(m_id))
    Vn = np.asarray(V_list); ok = (Vn[:, 0] >= 0) & (Vn[:, 1] >= 0)
    tb = Vn[ok, 0].astype(np.int64); sb = Vn[ok, 1].astype(np.int64)
    disp = (gx[tb] - gx[sb] + 3) * 49 + (gy[tb] - gy[sb] + 3) * 7 + (gz[tb] - gz[sb] + 3)
    uniq, counts = np.unique(disp, return_counts=True)
    n_disp, CAP = len(uniq), int(counts.max())
    K_stack = np.zeros((n_disp, n_s, n_s), K_flat.dtype); src_pad = np.zeros((n_disp, CAP), np.int32)
    tgt_pad = np.full((n_disp, CAP), -1, np.int32); valid = np.zeros((n_disp, CAP), bool)
    order = np.argsort(disp, kind="stable")           # group rows by displacement
    ds, tbs, sbs = disp[order], tb[order], sb[order]
    starts = np.searchsorted(ds, uniq); ends = np.append(starts[1:], len(ds))
    for i, (d, a, b) in enumerate(zip(uniq, starts, ends)):
        m = b - a
        K_stack[i] = K_flat[d]
        src_pad[i, :m] = sbs[a:b]; tgt_pad[i, :m] = tbs[a:b]; valid[i, :m] = True
    return jnp.asarray(K_stack), jnp.asarray(src_pad), jnp.asarray(tgt_pad), jnp.asarray(valid)


@jax.jit
def m2l_batched(equiv, K_stack, src_pad, tgt_pad, valid, pinv_dn):
    r"""Displacement-batched M2L: one GEMM per displacement instead of per pair.
    ``K_stack`` etc. from :func:`m2l_groups_padded`.  Returns local (= ``pinv_dn @
    check_pot``), identical to :func:`m2l`.

    Parameters
    ----------
    equiv : (max_boxes, n_s) ndarray
        Upward equivalent densities (from :func:`upward_pass`).
    K_stack : (n_disp, n_s, n_s) ndarray
        Per-displacement raw kernel matrices (:func:`m2l_groups_padded`).
    src_pad : (n_disp, CAP) ndarray of int32
        Source box index per group slot.
    tgt_pad : (n_disp, CAP) ndarray of int32
        Target box index per group slot.
    valid : (n_disp, CAP) ndarray of bool
        Occupied-slot mask.
    pinv_dn : (n_s, n_s) ndarray or (U, d, Vt) tuple
        Downward decode (applied per box).

    Returns
    -------
    local : (max_boxes, n_s) ndarray
        Downward equivalent density per box.
    """
    max_boxes, n_s = equiv.shape
    gathered = equiv[jnp.where(valid, src_pad, 0)]                  # (n_disp, CAP, n_s)
    contrib = jnp.where(valid[..., None],
                        jnp.einsum("dij,drj->dri", K_stack, gathered), 0.0)
    scatter_tgt = jnp.where(valid, tgt_pad, max_boxes).reshape(-1)  # dump invalid rows
    check_pot = jnp.zeros((max_boxes, n_s), equiv.dtype).at[scatter_tgt].add(
        contrib.reshape(-1, n_s), mode="drop")
    return _decode(check_pot, pinv_dn)


def m2l_groups_tiered(V_list, m_id, M2L_K_table, K_tiers=4, lvls=None):
    r"""Tiered variant of :func:`m2l_groups_padded`.

    The displacement groups are bucketed into up to ``K_tiers`` CAP tiers so a
    small group is not padded to the GLOBAL max group size.  The tier
    boundaries are chosen by exact dynamic programming over the count-sorted
    groups (minimising total padded rows).

    Parameters
    ----------
    V_list : ndarray or list of ndarray
        One pair array, or a LIST of compact chunks
        (``build_v_eager(..., concat=False)``) — with chunks, the full pair
        list never exists on device and the transient peak is one chunk.
    m_id : (max_boxes,) ndarray of uint64
        Morton ids.
    M2L_K_table : (7, 7, 7, n_s, n_s) ndarray
        Raw kernel table (read on the host).
    K_tiers : int, optional
        Maximum number of CAP tiers.
    lvls : ndarray or None, optional
        Box levels; required for the periodic 3-column ``V_list``.

    Returns
    -------
    tuple of tuple
        Per-tier ``(K_stack, src_pad, tgt_pad, valid)`` 4-tuples for
        :func:`m2l_batched_tiered`.

    Notes
    -----
    Pair-scale work runs ON DEVICE and STREAMS over pair chunks: the
    displacement key takes only 343 values, so grouping is a two-pass counting
    sort (pass A: per-chunk bincounts → global counts + tier DP; pass B:
    per-chunk within-group ranks + running host group cursors → scatter
    straight into the final tier arrays).  Only pair-length transients are
    chunk-sized; only the 343-length DP stays on the host.  (A host-side sort
    over every pair, or a device form that keeps pair-length intermediates
    alive, does not scale — hence this streaming structure.)"""
    import numpy as np
    n_s = M2L_K_table.shape[-1]
    K_flat = np.asarray(M2L_K_table).reshape(343, n_s, n_s)

    chunks = V_list if isinstance(V_list, (list, tuple)) else [V_list]
    # ONE static chunk shape (pad with -1 rows -> overflow bucket, dropped): compact
    # chunks have data-dependent lengths and would retrace the jitted passes per chunk
    L = max(int(c.shape[0]) for c in chunks)
    chunks = [c if c.shape[0] == L else                # keep device arrays device-side
              np.pad(np.asarray(c), ((0, L - c.shape[0]), (0, 0)), constant_values=-1)
              for c in chunks]
    gxyz = tuple(g.astype(jnp.int32) for g in _morton_to_grid(m_id))

    @jax.jit
    def _chunk_disp(pairs):
        gx, gy, gz = gxyz
        ok = (pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)
        tb = jnp.where(ok, pairs[:, 0], 0).astype(jnp.int32)
        sb = jnp.where(ok, pairs[:, 1], 0).astype(jnp.int32)
        if pairs.shape[1] == 3:                        # periodic 3-col V: unwrap the disp
            from jaxfmm.trees.cubic import _IMG_COMBOS
            io = jnp.asarray(_IMG_COMBOS, jnp.int32)[jnp.where(ok, pairs[:, 2], 0)]
            grid = (2 ** lvls[sb]).astype(jnp.int32)
            dx = gx[tb] - (gx[sb] + io[:, 0] * grid)
            dy = gy[tb] - (gy[sb] + io[:, 1] * grid)
            dz = gz[tb] - (gz[sb] + io[:, 2] * grid)
        else:
            dx = gx[tb] - gx[sb]; dy = gy[tb] - gy[sb]; dz = gz[tb] - gz[sb]
        disp = ((dx + 3) * 49 + (dy + 3) * 7 + (dz + 3)).astype(jnp.int32)
        return jnp.where(ok, disp, 343), tb, sb        # padding -> overflow bucket 343

    # ── pass A: global displacement-group counts (343 ints per chunk to host) ──
    counts_all = np.zeros(344, np.int64)
    for c in chunks:
        d, _, _ = _chunk_disp(jnp.asarray(c))
        counts_all += np.asarray(jnp.bincount(d, length=344))
    counts_all = counts_all[:343]

    uniq = np.nonzero(counts_all)[0]                   # occupied displacement groups
    counts = counts_all[uniq]
    n_g = len(uniq)
    if n_g == 0:
        return ()
    if counts.max() >= 2 ** 31:                        # int32 COLUMN positions are per-group
        raise ValueError("m2l_groups_tiered: a displacement group exceeds 2^31 pairs; "
                         "raise N_max (fewer boxes)")   # (total pairs may exceed 2^31)
    grp_dense = np.full(344, -1, np.int32)             # disp value -> dense group index
    grp_dense[uniq] = np.arange(n_g, dtype=np.int32)

    csort = np.argsort(counts, kind="stable")          # groups by ascending count
    c = counts[csort]
    K_tiers = min(K_tiers, n_g)
    # DP: best[k][i] = min padded rows covering the first i sorted groups with k tiers
    # (a tier is a contiguous run of the sorted groups; its cap = the run's last count).
    INF = float("inf")
    best = np.full((K_tiers + 1, n_g + 1), INF); best[0, 0] = 0.0
    cut = np.zeros((K_tiers + 1, n_g + 1), np.int64)
    for k in range(1, K_tiers + 1):
        for i in range(k, n_g + 1):                    # need >= k groups for k non-empty tiers
            js = np.arange(k - 1, i)
            cost = best[k - 1, js] + (i - js) * c[i - 1]
            j = int(js[np.argmin(cost)])
            best[k, i] = cost[j - (k - 1)]; cut[k, i] = j
    bounds = [n_g]
    for k in range(K_tiers, 0, -1):
        bounds.append(int(cut[k, bounds[-1]]))
    bounds = bounds[::-1]                              # K_tiers+1 ascending cut points

    # tier index + row-within-tier per group (host, 343-length), then one device sort
    tier_of_grp = np.zeros(n_g, np.int32)
    row_of_grp = np.zeros(n_g, np.int32)
    tier_shapes = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi <= lo:                                   # empty tier (DP degenerate case)
            continue
        sel = csort[lo:hi]
        tier_of_grp[sel] = len(tier_shapes)            # dense tier index (empties skipped)
        row_of_grp[sel] = np.arange(hi - lo, dtype=np.int32)
        tier_shapes.append((hi - lo, int(c[hi - 1]), uniq[sel]))

    # ── pass B: stream chunks, scatter into the final tier arrays ──────────────
    # Within-chunk rank per displacement group + a running host cursor per group give
    # each pair its global column; the only pair-length transients are chunk-sized.
    n_tiers = len(tier_shapes)
    grp_dense_j = jnp.asarray(grp_dense)
    tier_of_grp_j = jnp.asarray(tier_of_grp)
    row_of_grp_j = jnp.asarray(row_of_grp)

    @partial(jax.jit, donate_argnums=(0,))
    def _scatter_chunk(tier_arrs, pairs, cursor):
        d, tb, sb = _chunk_disp(pairs)
        cnt = jnp.bincount(d, length=344)
        starts = jnp.cumsum(cnt) - cnt                 # exclusive prefix per disp value
        order = jnp.argsort(d).astype(jnp.int32)       # chunk-local counting order
        ds = d[order]
        pos = jnp.arange(ds.shape[0], dtype=jnp.int32) - starts[ds].astype(jnp.int32)
        grp = grp_dense_j[ds]
        col = cursor[jnp.clip(grp, 0, None)].astype(jnp.int32) + pos
        tier = tier_of_grp_j[grp]                      # padding wraps; masked per tier
        row = row_of_grp_j[grp]
        okp = grp >= 0
        tbs, sbs = tb[order], sb[order]
        out = []
        for t, (src_pad, tgt_pad, valid) in enumerate(tier_arrs):
            m = okp & (tier == t)
            r = jnp.where(m, row, src_pad.shape[0])    # invalid -> dropped (mode="drop")
            c = jnp.where(m, col, src_pad.shape[1])
            out.append((src_pad.at[r, c].set(sbs, mode="drop"),
                        tgt_pad.at[r, c].set(tbs, mode="drop"),
                        valid.at[r, c].set(True, mode="drop")))
        return out, cnt[:343]

    tier_arrs = [(jnp.zeros((n_sel, cap), jnp.int32),
                  jnp.full((n_sel, cap), -1, jnp.int32),
                  jnp.zeros((n_sel, cap), bool)) for n_sel, cap, _ in tier_shapes]
    cursor = np.zeros(n_g, np.int64)                   # pairs already placed per dense group
    for c in chunks:
        tier_arrs, cnt = _scatter_chunk(tier_arrs, jnp.asarray(c), jnp.asarray(cursor))
        cursor += np.asarray(cnt)[uniq]                # disp-value counts -> dense order
    tiers = []
    for (n_sel, cap, uniq_sel), (src_pad, tgt_pad, valid) in zip(tier_shapes, tier_arrs):
        K_stack = K_flat[uniq_sel].astype(K_flat.dtype)
        tiers.append((jnp.asarray(K_stack), src_pad, tgt_pad, valid))
    return tuple(tiers)


@partial(jax.jit, static_argnames=["mem_limit"])
def m2l_batched_tiered(equiv, tiers, pinv_dn, mem_limit=jnp.inf):
    r"""Tiered displacement-batched M2L (tiers from :func:`m2l_groups_tiered`): one einsum
    per CAP tier, each at its tight width.  Identical result to :func:`m2l_batched`.
    ``mem_limit`` (bytes, static) chunks each tier along the cap axis — the gathered
    (n_tier, cap, n_s) working set grows linearly with the pair count (OOM guard).

    Parameters
    ----------
    equiv : (max_boxes, n_s) ndarray
        Upward equivalent densities (from :func:`upward_pass`).
    tiers : tuple of tuple
        Per-tier ``(K_stack, src_pad, tgt_pad, valid)`` 4-tuples from
        :func:`m2l_groups_tiered` (shapes as in :func:`m2l_batched`, one CAP
        per tier).
    pinv_dn : (n_s, n_s) ndarray or (U, d, Vt) tuple
        Downward decode (applied per box).
    mem_limit : float, optional
        Working-set budget in bytes (static), default ``jnp.inf`` (no
        chunking).

    Returns
    -------
    local : (max_boxes, n_s) ndarray
        Downward equivalent density per box.
    """
    max_boxes, n_s = equiv.shape
    check_pot = jnp.zeros((max_boxes, n_s), equiv.dtype)
    for K_stack, src_pad, tgt_pad, valid in tiers:
        n_t, cap = src_pad.shape
        mem = n_t * cap * n_s * equiv.dtype.itemsize * 3            # ~3 live intermediates
        n_chunks = int(mem / mem_limit) + 1 if mem > mem_limit else 1

        def tier_body(cp, x):
            sp, tp, va = x
            gathered = equiv[jnp.where(va, sp, 0)]                  # (n_tier, cw, n_s)
            contrib = jnp.where(va[..., None],
                                jnp.einsum("dij,drj->dri", K_stack, gathered), 0.0)
            scatter_tgt = jnp.where(va, tp, max_boxes).reshape(-1)
            return cp.at[scatter_tgt].add(contrib.reshape(-1, n_s), mode="drop"), None

        if n_chunks == 1:
            check_pot, _ = tier_body(check_pot, (src_pad, tgt_pad, valid))
        else:
            # lax.scan, NOT an unrolled python loop: the chunk gathers only depend on
            # `equiv`, so XLA's scheduler is free to HOIST every unrolled chunk ahead of
            # the sequential scatter chain — all chunks alive at once == the full
            # (n_tier, cap, n_s) tier tensor and the chunking is operationally void.
            # scan serialises the body: peak == one chunk.
            cw = -(-cap // n_chunks)
            padc = n_chunks * cw - cap
            sp3, tp3, va3 = (jnp.pad(a, ((0, 0), (0, padc)))        # pad cols: valid=False
                             .reshape(n_t, n_chunks, cw).transpose(1, 0, 2)
                             for a in (src_pad, tgt_pad, valid))
            check_pot, _ = jax.lax.scan(tier_body, check_pot, (sp3, tp3, va3))
    return _decode(check_pot, pinv_dn)




# ──────────────────────────────────────────────────────────────────────────
# Cross-level interactions (adaptive far field)
# ──────────────────────────────────────────────────────────────────────────
@partial(jax.jit, static_argnames=["batch_size", "field", "g", "g_grad"])
def m2p(equiv, positions, particles_in_leaf, centres, h, W_list, equiv_pts,
        batch_size=64, field=False, img_cnct=jnp.array([[]]), g=_green, g_grad=_green_grad):
    r"""W-list multipole-to-particle: a finer source box's multipole evaluated
    directly at a coarser leaf target's particles.

    Parameters
    ----------
    equiv : (max_boxes, n_s) ndarray
        Upward equivalent densities.
    positions : (N, 3) ndarray
        Particle coordinates.
    particles_in_leaf : (max_boxes, N_max) ndarray of int32
        Per-leaf particle indices, -1 padded.
    centres : (max_boxes, 3) ndarray
        Box centres.
    h : (max_boxes,) ndarray
        Box half-sides.
    W_list : (max_W, 2) ndarray of int32
        (target_leaf, source_box) pairs, -1 padded.
    equiv_pts : (n_s, 3) ndarray
        Upward equivalent surface (``R_EQUIV``, unit box).
    batch_size : int, optional
        Pairs per chunk (static).
    field : bool, optional
        Return the analytic :math:`-\nabla\phi` contribution instead.
    img_cnct : ndarray, optional
        Periodic image offsets (source boxes may be image-encoded).
    g, g_grad : callable, optional
        Kernel and gradient-kernel functions.

    Returns
    -------
    pot : (N,) or (N, 3) ndarray
        W-list potential contribution at every particle
        ((N, 3) analytic :math:`-\nabla\phi` when ``field=True``).
    """
    N = positions.shape[0]
    pil = particles_in_leaf
    valid_p = pil >= 0
    pil_safe = jnp.where(valid_p, pil, 0)

    tb = W_list[:, 0]
    sb = W_list[:, 1]
    ok = (tb >= 0) & (sb >= 0)
    tb_s = jnp.where(ok, tb, 0)
    sb_s = jnp.where(ok, sb, 0)

    nbox = centres.shape[0]
    pbc = img_cnct.shape[1] != 0

    def per_pair(args):
        t_box, s_box, valid = args
        t_idx = pil_safe[t_box]                       # (N_max,)
        t_pos = positions[t_idx]                      # (N_max, 3)
        if pbc:                                       # image-encoded source box
            off = img_cnct[s_box // nbox]
            s_box = s_box % nbox
        else:
            off = 0.0
        src_equiv_phys = equiv_pts * h[s_box] + centres[s_box] + off   # (n_s, 3)
        d = t_pos[:, None, :] - src_equiv_phys[None, :, :]
        if field:
            contrib = jnp.einsum("psd,s->pd", g_grad(d), equiv[s_box])
            contrib = jnp.where((valid & valid_p[t_box])[:, None], contrib, 0.0)
        else:
            K = g(jnp.sqrt(jnp.sum(d * d, axis=-1)))    # (N_max, n_s)
            contrib = jnp.where(valid & valid_p[t_box], K @ equiv[s_box], 0.0)
        return t_idx, contrib

    t_idx_all, contrib_all = jax.lax.map(
        per_pair, (tb_s, sb_s, ok), batch_size=batch_size
    )
    pot = jnp.zeros((N, 3) if field else N, dtype=positions.dtype)
    pot = pot.at[t_idx_all.reshape(-1)].add(contrib_all.reshape(-1, 3) if field
                                            else contrib_all.reshape(-1), mode="drop")
    return pot


@partial(jax.jit, static_argnames=["batch_size", "w", "field", "g", "g_grad"])
def m2p_lean(equiv, pos_s, starts, cnts, w, centres, h, W_list, equiv_pts,
             batch_size=64, field=False, img_cnct=jnp.array([[]]), g=_green, g_grad=_green_grad):
    r""":func:`m2p` on the LEAF-MAJOR SORTED layout (lean mode on adaptive trees): the
    target leaf's particles are the contiguous slice ``pos_s[starts[t] + iota(w)]`` and
    the contributions scatter to SORTED particle ids (the caller adds them to the far
    field BEFORE un-sorting by rank).  Runs as a SCAN with an in-body scatter — a map
    would STACK the full (n_W, w[, 3]) contribution tensor before scattering, which is
    unaffordable at scale (the padded :func:`m2p` shares this flaw but only serves
    small problems).

    Parameters
    ----------
    equiv : (max_boxes, n_s) ndarray
        Upward equivalent densities.
    pos_s : (N, 3) ndarray
        Particle coordinates in LEAF-MAJOR SORTED order.
    starts : (max_boxes,) ndarray of int32
        Per-box slice start into the sorted arrays.
    cnts : (max_boxes,) ndarray of int32
        Per-box particle count.
    w : int
        Static slice width (max particles per leaf).
    centres : (max_boxes, 3) ndarray
        Box centres.
    h : (max_boxes,) ndarray
        Box half-sides.
    W_list : (max_W, 2) ndarray of int32
        (target_leaf, source_box) pairs, -1 padded.
    equiv_pts : (n_s, 3) ndarray
        Upward equivalent surface (``R_EQUIV``, unit box).
    batch_size : int, optional
        Pairs per chunk (static), default 64.
    field : bool, optional
        Return the analytic :math:`-\nabla\phi` contribution instead.
    img_cnct : ndarray, optional
        Periodic image offsets (source boxes may be image-encoded).
    g, g_grad : callable, optional
        Kernel and gradient-kernel functions.

    Returns
    -------
    pot : (N,) or (N, 3) ndarray
        W-list contribution at every SORTED particle ((N, 3) analytic
        :math:`-\nabla\phi` when ``field=True``).
    """
    N = pos_s.shape[0]
    n_w = W_list.shape[0]
    iw = jnp.arange(w, dtype=jnp.int32)
    bs = max(int(batch_size), 1)
    pad = (-n_w) % bs
    WL = jnp.pad(W_list, ((0, pad), (0, 0)), constant_values=-1)
    WL = WL.reshape(-1, bs, 2)

    def body(pot, pairs):
        tb, sb = pairs[:, 0], pairs[:, 1]
        valid = (tb >= 0) & (sb >= 0)
        tb_s = jnp.where(valid, tb, 0)
        sb_s = jnp.where(valid, sb, 0)
        sl = jnp.clip(starts[tb_s][:, None] + iw[None, :], 0, N - 1)   # (bs, w)
        t_pos = pos_s[sl]                                              # (bs, w, 3)
        if img_cnct.shape[1] != 0:                    # image-encoded source box
            off = img_cnct[sb_s // centres.shape[0]][:, None, :]
            sb_s = sb_s % centres.shape[0]
        else:
            off = 0.0
        seq = equiv_pts[None] * h[sb_s][:, None, None] + centres[sb_s][:, None, :] + off
        d = t_pos[:, :, None, :] - seq[:, None, :, :]                  # (bs, w, n_s, 3)
        v_ok = valid[:, None] & (iw[None, :] < cnts[tb_s][:, None])
        if field:
            contrib = jnp.einsum("bpsd,bs->bpd", g_grad(d), equiv[sb_s])
        else:
            K = g(jnp.sqrt(jnp.sum(d * d, axis=-1)))              # (bs, w, n_s)
            contrib = jnp.einsum("bps,bs->bp", K, equiv[sb_s])
        idx = jnp.where(v_ok, sl, N)                                   # invalid -> dropped
        return pot.at[idx].add(jnp.where(v_ok[..., None] if field else v_ok,
                                         contrib, 0.0).astype(pot.dtype),
                               mode="drop"), None

    pot = jnp.zeros((N, 3) if field else N, dtype=pos_s.dtype)
    pot, _ = jax.lax.scan(body, pot, WL)
    return pot


@partial(jax.jit, static_argnames=["batch_size", "w", "g", "g_grad"])
def p2l_lean(local, pos_s, q_s, starts, cnts, w, centres, h, X_list,
             check_dn_pts, pinv_dn, batch_size=64, img_cnct=jnp.array([[]]), g=_green, g_grad=_green_grad):
    r""":func:`p2l` on the LEAF-MAJOR SORTED layout: the source leaf's particles are the
    contiguous slice of the sorted arrays (out-of-count slots carry charge 0).

    Parameters
    ----------
    local : (max_boxes, n_s) ndarray
        Downward equivalent densities (accumulated into).
    pos_s : (N, 3) ndarray
        Particle coordinates in LEAF-MAJOR SORTED order.
    q_s : (N,) ndarray
        Particle charges, sorted like ``pos_s``.
    starts : (max_boxes,) ndarray of int32
        Per-box slice start into the sorted arrays.
    cnts : (max_boxes,) ndarray of int32
        Per-box particle count.
    w : int
        Static slice width (max particles per leaf).
    centres : (max_boxes, 3) ndarray
        Box centres.
    h : (max_boxes,) ndarray
        Box half-sides.
    X_list : (max_X, 2) ndarray of int32
        (target_leaf, source_leaf) pairs, -1 padded.
    check_dn_pts : (n_s, 3) ndarray
        Downward check surface (``R_EQUIV``, unit box).
    pinv_dn : (n_s, n_s) ndarray or (U, d, Vt) tuple
        Downward decode.
    batch_size : int, optional
        Pairs per chunk (static), default 64.
    img_cnct : ndarray, optional
        Periodic image offsets (source boxes may be image-encoded).
    g, g_grad : callable, optional
        Kernel and gradient-kernel functions.

    Returns
    -------
    local : (max_boxes, n_s) ndarray
        With the X-list contributions added.
    """
    N = pos_s.shape[0]
    iw = jnp.arange(w, dtype=jnp.int32)
    tb = X_list[:, 0]
    sb = X_list[:, 1]
    ok = (tb >= 0) & (sb >= 0)
    tb_s = jnp.where(ok, tb, 0)
    sb_s = jnp.where(ok, sb, 0)

    def per_pair(args):
        t_box, s_box, valid = args
        if img_cnct.shape[1] != 0:                    # image-encoded source box
            off = img_cnct[s_box // centres.shape[0]]
            s_box = s_box % centres.shape[0]
        else:
            off = 0.0
        sl = jnp.clip(starts[s_box] + iw, 0, N - 1)
        s_pos = pos_s[sl] + off                       # (w, 3)
        q = jnp.where((iw < cnts[s_box]) & valid, q_s[sl], 0.0)
        check_phys = check_dn_pts * h[t_box] + centres[t_box]    # (n_s, 3)
        d = check_phys[:, None, :] - s_pos[None, :, :]
        K = g(jnp.sqrt(jnp.sum(d * d, axis=-1)))            # (n_s, w)
        return jnp.where(valid, t_box, local.shape[0]), K @ q

    t_all, chk_all = jax.lax.map(per_pair, (tb_s, sb_s, ok), batch_size=batch_size)
    check = jnp.zeros_like(local).at[t_all].add(chk_all * h[jnp.clip(t_all, 0,
                                                h.shape[0] - 1), None], mode="drop")
    return local + _decode(check, pinv_dn)


@partial(jax.jit, static_argnames=["batch_size", "g", "g_grad"])
def p2l(local, positions, charges, particles_in_leaf, centres, h, X_list,
        check_dn_pts, pinv_dn, batch_size=64, img_cnct=jnp.array([[]]), g=_green, g_grad=_green_grad):
    r"""X-list particle-to-local: a coarser source leaf's particles evaluated onto a
    finer leaf target's downward check surface (adds to the target's local
    expansion; the dual of P2M, so it carries the per-box half-side factor
    :math:`h`).

    Parameters
    ----------
    local : (max_boxes, n_s) ndarray
        Downward equivalent densities (accumulated into).
    positions : (N, 3) ndarray
        Particle coordinates.
    charges : (N,) ndarray
        Particle charges.
    particles_in_leaf : (max_boxes, N_max) ndarray of int32
        Per-leaf particle indices, -1 padded.
    centres : (max_boxes, 3) ndarray
        Box centres.
    h : (max_boxes,) ndarray
        Box half-sides.
    X_list : (max_X, 2) ndarray of int32
        (target_leaf, source_leaf) pairs, -1 padded.
    check_dn_pts : (n_s, 3) ndarray
        Downward check surface (``R_EQUIV``, unit box).
    pinv_dn : (n_s, n_s) ndarray or (U, d, Vt) tuple
        Downward decode.
    batch_size : int, optional
        Pairs per chunk (static).
    img_cnct : ndarray, optional
        Periodic image offsets (source boxes may be image-encoded).
    g, g_grad : callable, optional
        Kernel and gradient-kernel functions.

    Returns
    -------
    local : (max_boxes, n_s) ndarray
        With the X-list contributions added.
    """
    pil = particles_in_leaf
    valid_p = pil >= 0
    pil_safe = jnp.where(valid_p, pil, 0)

    tb = X_list[:, 0]
    sb = X_list[:, 1]
    ok = (tb >= 0) & (sb >= 0)
    tb_s = jnp.where(ok, tb, 0)
    sb_s = jnp.where(ok, sb, 0)

    nbox = centres.shape[0]
    pbc = img_cnct.shape[1] != 0

    def per_pair(args):
        t_box, s_box, valid = args
        if pbc:                                       # image-encoded source box
            off = img_cnct[s_box // nbox]
            s_box = s_box % nbox
        else:
            off = 0.0
        s_idx = pil_safe[s_box]                       # (N_max,)
        s_pos = positions[s_idx] + off                # (N_max, 3)
        s_q = jnp.where(valid_p[s_box], charges[s_idx], 0.0)
        check_phys = check_dn_pts * h[t_box] + centres[t_box]    # (n_s, 3)
        d = check_phys[:, None, :] - s_pos[None, :, :]
        r = jnp.sqrt(jnp.sum(d * d, axis=-1))
        K = g(r)   # (n_s, N_max)
        check_pot = K @ s_q                           # (n_s,)
        contrib = h[t_box] * _decode(check_pot[None, :], pinv_dn)[0]   # (n_s,)
        contrib = jnp.where(valid, contrib, 0.0)
        return t_box, contrib

    t_box_all, contrib_all = jax.lax.map(
        per_pair, (tb_s, sb_s, ok), batch_size=batch_size
    )
    return local.at[t_box_all].add(contrib_all, mode="drop")


# ──────────────────────────────────────────────────────────────────────────
# Downward pass
# ──────────────────────────────────────────────────────────────────────────
@partial(jax.jit, static_argnames=["max_depth"])
def l2l(local, m_id, p_id, lvls, L2L_mats, max_depth):
    r"""Local-to-local propagation, top-down over levels.

    Parameters
    ----------
    local : (max_boxes, n_s) ndarray
        Downward equivalent densities seeded by M2L.
    m_id : (max_boxes,) ndarray of uint64
        Morton ids.
    p_id : (max_boxes,) ndarray of int32
        Parent box index.
    lvls : (max_boxes,) ndarray of int32
        Box levels.
    L2L_mats : (8, n_s, n_s) ndarray
        Per-octant parent→child operators.
    max_depth : int
        Static tree depth.

    Returns
    -------
    local : (max_boxes, n_s) ndarray
        Downward densities with parents pushed to children.
    """
    n_s = local.shape[1]
    octant = (m_id % jnp.uint64(8)).astype(jnp.int32)
    rows = jnp.arange(local.shape[0])
    # one stacked GEMM + per-box octant gather + single add per level (vs 8x masked GEMM)
    L2L_stack = L2L_mats.reshape(8 * n_s, n_s)            # (8·n_s, n_s)

    def push_level(i, local):
        # Iteration i pushes parent level i into child level i+1.
        child_lvl = i + 1
        big = (local[p_id] @ L2L_stack.T).reshape(-1, 8, n_s)   # (max_boxes, 8, n_s)
        contrib = big[rows, octant]                            # per-box octant operator
        return local + jnp.where((lvls == child_lvl)[:, None], contrib, 0.0)

    # i = 1 .. max_depth-1  (push level 1→2, 2→3, …)
    local = jax.lax.fori_loop(1, max_depth, push_level, local)
    return local


@partial(jax.jit, static_argnames=["max_depth"])
def l2l_factored(local, m_id, p_id, lvls, L2L_K, F_dn, max_depth):
    r"""Top-down L2L in the backward-stable FACTORED form: raw parent→child-check
    kernel matvecs (``L2L_K``), then the factored decode per child level (the decode
    is linear, so it is applied to the per-child contribution).  Same result as
    :func:`l2l` in f64 at P<=6; see :func:`m2m_factored`.

    Parameters
    ----------
    local : (max_boxes, n_s) ndarray
        Downward equivalent densities seeded by M2L.
    m_id : (max_boxes,) ndarray of uint64
        Morton ids.
    p_id : (max_boxes,) ndarray of int32
        Parent box index.
    lvls : (max_boxes,) ndarray of int32
        Box levels.
    L2L_K : (8, n_s, n_s) ndarray
        Per-octant raw parent→child-check kernels (``L2L_K`` from
        :func:`jaxfmm.kifmm.operators.build_operators`).
    F_dn : (U, d, Vt) tuple or (n_s, n_s) ndarray
        Downward decode (factored or materialised, see :func:`_decode`).
    max_depth : int
        Static tree depth.

    Returns
    -------
    local : (max_boxes, n_s) ndarray
        Downward densities with parents pushed to children.
    """
    n_s = local.shape[1]
    octant = (m_id % jnp.uint64(8)).astype(jnp.int32)
    rows = jnp.arange(local.shape[0])
    K_stack = L2L_K.reshape(8 * n_s, n_s)                 # stacked raw kernels

    def push_level(i, loc):
        child_lvl = i + 1
        big = (loc[p_id] @ K_stack.T).reshape(-1, 8, n_s)   # parent local -> child check
        add = _decode(big[rows, octant], F_dn)
        return loc + jnp.where((lvls == child_lvl)[:, None], add, 0.0)

    return jax.lax.fori_loop(1, max_depth, push_level, local)


@partial(jax.jit, static_argnames=["batch_size", "field", "g", "g_grad"])
def l2p(local, positions, particles_in_leaf, centres, h, is_leaf,
        check_pts, batch_size=64, field=False, g=_green, g_grad=_green_grad):
    r"""Local-to-particle: evaluate each leaf's downward expansion at its particles.

    The downward equivalent surface is the LARGE surface (``R_CHECK``) scaled
    by the box half-side; eval points lie inside it so the single-layer
    representation is accurate (maximum principle).

    Parameters
    ----------
    local : (max_boxes, n_s) ndarray
        Downward equivalent densities.
    positions : (N, 3) ndarray
        Particle coordinates.
    particles_in_leaf : (max_boxes, N_max) ndarray of int32
        Per-leaf particle indices, -1 padded.
    centres : (max_boxes, 3) ndarray
        Box centres.
    h : (max_boxes,) ndarray
        Box half-sides.
    is_leaf : (max_boxes,) ndarray of bool
        Leaf mask.
    check_pts : (n_s, 3) ndarray
        ``R_CHECK`` surface (the downward equivalent surface, unit box).
    batch_size : int, optional
        Boxes per chunk (static).
    field : bool, optional
        Return the analytic :math:`-\nabla\phi` instead.
    g, g_grad : callable, optional
        Kernel and gradient-kernel functions.

    Returns
    -------
    pot : (N,) or (N, 3) ndarray
        Far-field potential at every particle, or :math:`-\nabla\phi` when
        ``field=True`` (the equivalent densities are point charges, so the
        gradient kernel is analytic).
    """
    N = positions.shape[0]
    pil = particles_in_leaf
    valid_p = pil >= 0
    pil_safe = jnp.where(valid_p, pil, 0)

    def per_box(args):
        centre, hb, idx, vmask, leaf, dn = args
        eval_pts = positions[idx]                      # (N_max, 3)
        equiv_dn_phys = check_pts * hb + centre        # (n_s, 3)
        d = eval_pts[:, None, :] - equiv_dn_phys[None, :, :]   # (N_max, n_s, 3)
        if field:
            contrib = jnp.einsum("psd,s->pd", g_grad(d), dn)   # (N_max, 3)
            contrib = jnp.where((vmask & leaf)[:, None], contrib, 0.0)
        else:
            K = g(jnp.sqrt(jnp.sum(d * d, axis=-1)))
            contrib = jnp.where(vmask & leaf, K @ dn, 0.0)          # (N_max,)
        return idx, contrib

    idx_all, contrib_all = jax.lax.map(
        per_box, (centres, h, pil_safe, valid_p, is_leaf, local),
        batch_size=batch_size,
    )                                                  # (max_boxes, N_max[, 3]) each

    pot = jnp.zeros((N, 3) if field else N, dtype=positions.dtype)
    pot = pot.at[idx_all.reshape(-1)].add(contrib_all.reshape(-1, 3) if field
                                          else contrib_all.reshape(-1), mode="drop")
    return pot


@partial(jax.jit, static_argnames=["batch_size", "field", "g", "g_grad"])
def l2p_padded(local, ppts, rev_idcs, centres, h, is_leaf, check_pts, batch_size=64,
               field=False, g=_green, g_grad=_green_grad):
    r""":func:`l2p` consuming the shared padded per-box layout: pre-gathered points and a
    ``rev_idcs`` gather-out (each particle lives in exactly one leaf slot) instead of the
    ``(max_boxes*N_max)``-element scatter-add — the scatter, not the kernel, dominates
    :func:`l2p`'s cost.  Same result as :func:`l2p` (``field=True`` → (N, 3) analytic
    :math:`-\nabla\phi`).

    Parameters
    ----------
    local : (max_boxes, n_s) ndarray
        Downward equivalent densities.
    ppts : (max_boxes, N_max, 3) ndarray
        Padded per-box particle points (:func:`jaxfmm.near._build_padded`).
    rev_idcs : (N,) ndarray of int32
        Flat padded-slot index of each particle (gather-out map).
    centres : (max_boxes, 3) ndarray
        Box centres.
    h : (max_boxes,) ndarray
        Box half-sides.
    is_leaf : (max_boxes,) ndarray of bool
        Leaf mask.
    check_pts : (n_s, 3) ndarray
        ``R_CHECK`` surface (the downward equivalent surface, unit box).
    batch_size : int, optional
        Boxes per chunk (static), default 64.
    field : bool, optional
        Return the analytic :math:`-\nabla\phi` instead.
    g, g_grad : callable, optional
        Kernel and gradient-kernel functions.

    Returns
    -------
    pot : (N,) or (N, 3) ndarray
        Far-field potential at every particle ((N, 3) analytic
        :math:`-\nabla\phi` when ``field=True``).
    """
    def per_box(args):
        centre, hb, pts, leaf, dn = args
        equiv_dn_phys = check_pts * hb + centre        # (n_s, 3)
        d = pts[:, None, :] - equiv_dn_phys[None, :, :]
        if field:
            return jnp.where(leaf, jnp.einsum("psd,s->pd", g_grad(d), dn), 0.0)
        K = g(jnp.sqrt(jnp.sum(d * d, axis=-1)))
        return jnp.where(leaf, K @ dn, 0.0)            # (N_max,) / (N_max, 3) if field

    contrib = jax.lax.map(per_box, (centres, h, ppts, is_leaf, local),
                          batch_size=batch_size)       # (max_boxes, N_max[, 3])
    return contrib.reshape(-1, 3)[rev_idcs] if field else contrib.reshape(-1)[rev_idcs]


def l2p_lean(local, pos_s, starts, cnts, w, centres, h, is_leaf, check_pts, batch_size=64,
             field=False, g=_green, g_grad=_green_grad):
    r""":func:`l2p_padded` without the padded layout (memory diet): a scan over box chunks
    reads each box's contiguous LEAF-MAJOR SORTED slice (arithmetic indices, see
    :func:`p2m_lean`) and scatter-adds straight into the sorted-particle output.  The
    stacked ``(max_boxes, maxleaf[, 3])`` map output that :func:`l2p_padded` materialises
    (then rev-gathers) never exists.  The scatter carries ``unique_indices`` (each
    particle lives in exactly one leaf slot; invalid slots get UNIQUE out-of-bounds
    sentinels — a repeated sentinel would break the promise).  It is also faster than
    :func:`l2p_padded` at scale; if that ever regresses, :func:`l2p_padded` is the
    fallback.

    Parameters
    ----------
    local : (max_boxes, n_s) ndarray
        Downward equivalent densities.
    pos_s : (N, 3) ndarray
        Particle coordinates in LEAF-MAJOR SORTED order.
    starts : (max_boxes,) ndarray of int32
        Per-box slice start into the sorted arrays.
    cnts : (max_boxes,) ndarray of int32
        Per-box particle count.
    w : int
        Static slice width (max particles per leaf).
    centres : (max_boxes, 3) ndarray
        Box centres.
    h : (max_boxes,) ndarray
        Box half-sides.
    is_leaf : (max_boxes,) ndarray of bool
        Leaf mask.
    check_pts : (n_s, 3) ndarray
        ``R_CHECK`` surface (the downward equivalent surface, unit box).
    batch_size : int, optional
        Boxes per chunk (static), default 64.
    field : bool, optional
        Return the analytic :math:`-\nabla\phi` instead.
    g, g_grad : callable, optional
        Kernel and gradient-kernel functions.

    Returns
    -------
    pot : (N,) or (N, 3) ndarray
        Far-field potential at every SORTED particle ((N, 3) analytic
        :math:`-\nabla\phi` when ``field=True``).
    """
    N = pos_s.shape[0]
    nb = starts.shape[0]
    out = jnp.zeros((N, 3) if field else (N,), pos_s.dtype)
    bs = max(int(batch_size), 1)
    pad = (-nb) % bs
    args = (jnp.pad(centres, ((0, pad), (0, 0))), jnp.pad(h, (0, pad)),
            jnp.pad(starts, (0, pad)), jnp.pad(cnts, (0, pad)),
            jnp.pad(is_leaf, (0, pad)), jnp.pad(local, ((0, pad), (0, 0))))
    args = tuple(a.reshape((-1, bs) + a.shape[1:]) for a in args)
    iw = jnp.arange(w, dtype=jnp.int32)

    def body(buf, chunk):
        centre, hb, st, ct, leaf, dn = chunk               # (bs, ...)
        sl = jnp.clip(st[:, None] + iw[None, :], 0, N - 1) # (bs, w) contiguous slices
        pts = pos_s[sl]                                    # (bs, w, 3) fused gather
        equiv_dn_phys = check_pts[None] * hb[:, None, None] + centre[:, None, :]
        d = pts[:, :, None, :] - equiv_dn_phys[:, None, :, :]      # (bs, w, n_s, 3)
        if field:
            contrib = jnp.einsum("bpsd,bs->bpd", g_grad(d), dn)
        else:
            K = g(jnp.sqrt(jnp.sum(d * d, axis=-1)))
            contrib = jnp.einsum("bps,bs->bp", K, dn)
        ok = (iw[None, :] < ct[:, None]) & leaf[:, None]
        sentinel = N + jnp.arange(bs * w, dtype=jnp.int32).reshape(bs, w)
        idx = jnp.where(ok, st[:, None] + iw[None, :], sentinel)
        return buf.at[idx].add(contrib.astype(buf.dtype), mode="drop",
                               unique_indices=True), None

    out, _ = jax.lax.scan(body, out, args)
    return out


def downward_pass(local, positions, particles_in_leaf, m_id, p_id, lvls,
                  is_leaf, centres, h, ops, batch_size=64, max_depth=None, field=False):
    r"""L2L (top-down) followed by L2P on leaves.

    Uses the factored decodes when the ops dict carries ``F_dn``, the
    materialised legacy operators otherwise.

    Parameters
    ----------
    local : (max_boxes, n_s) ndarray
        Downward equivalent densities seeded by M2L (and P2L).
    positions, particles_in_leaf : ndarray
        Particles and per-leaf indices (see :func:`l2p`).
    m_id, p_id, lvls, is_leaf : ndarray
        Tree arrays.
    centres, h : ndarray
        Box geometry (from :func:`upward_pass`).
    ops : dict
        Operator tables.
    batch_size : int, optional
        Boxes per chunk.
    max_depth : int or None, optional
        Static tree depth; pass it to keep :func:`l2l`'s compiled shape fixed
        across geometries (defaults to ``int(lvls.max())``).
    field : bool, optional
        Return the analytic :math:`-\nabla\phi` instead.

    Returns
    -------
    (N,) or (N, 3) ndarray
        Far-field potential ((N, 3) :math:`-\nabla\phi` when ``field=True``).
    """
    if max_depth is None:
        max_depth = int(lvls.max())
    if "F_dn" in ops:                              # backward-stable decodes when available
        local = l2l_factored(local, m_id, p_id, lvls, jnp.asarray(ops["L2L_K"]),
                             tuple(jnp.asarray(x) for x in ops["F_dn"]), max_depth)
    else:
        local = l2l(local, m_id, p_id, lvls, jnp.asarray(ops["L2L_mats"]), max_depth)
    return l2p(local, positions, particles_in_leaf, centres, h, is_leaf,
               jnp.asarray(ops["check_pts"]), batch_size=batch_size, field=field)


def _round_up(n, gran):
    return int(((int(n) + gran - 1) // gran) * gran)


@partial(jax.jit, static_argnames=["batch_size", "max_depth", "g"])
def p2l_leveled(local, positions, charges, particles_in_leaf, centres, h, X_list, lvls,
                check_dn_pts, F_dn_lv, max_depth, batch_size=64, g=_green):
    r""":func:`p2l` for PER-LEVEL operators: the target level's decode, NO h factor.

    Parameters
    ----------
    local : (max_boxes, n_s) ndarray
        Downward equivalent densities (accumulated into).
    positions : (N, 3) ndarray
        Particle coordinates.
    charges : (N,) ndarray
        Particle charges.
    particles_in_leaf : (max_boxes, N_max) ndarray of int32
        Per-leaf particle indices, -1 padded.
    centres : (max_boxes, 3) ndarray
        Box centres.
    h : (max_boxes,) ndarray
        Box half-sides.
    X_list : (max_X, 2) ndarray of int32
        (target_leaf, source_leaf) pairs, -1 padded.
    lvls : (max_boxes,) ndarray of int32
        Box levels (select the target level's decode).
    check_dn_pts : (n_s, 3) ndarray
        Downward check surface (``R_EQUIV``, unit box).
    F_dn_lv : (U, d, Vt) tuple
        Level-stacked downward factored decodes, shapes
        ``((max_depth+1, n_s, n_s), (max_depth+1, n_s), (max_depth+1, n_s, n_s))``.
    max_depth : int
        Static tree depth.
    batch_size : int, optional
        Pairs per chunk (static), default 64.
    g : callable, optional
        Kernel function.

    Returns
    -------
    local : (max_boxes, n_s) ndarray
        With the X-list contributions added.
    """
    pil = particles_in_leaf
    valid_p = pil >= 0
    pil_safe = jnp.where(valid_p, pil, 0)
    tb, sb = X_list[:, 0], X_list[:, 1]
    ok = (tb >= 0) & (sb >= 0)
    tb_s = jnp.where(ok, tb, 0)
    sb_s = jnp.where(ok, sb, 0)

    def per_pair(args):
        t_box, s_box, valid = args
        s_idx = pil_safe[s_box]
        s_pos = positions[s_idx]
        s_q = jnp.where(valid_p[s_box], charges[s_idx], 0.0)
        check_phys = check_dn_pts * h[t_box] + centres[t_box]
        d = check_phys[:, None, :] - s_pos[None, :, :]
        K = g(jnp.sqrt(jnp.sum(d * d, axis=-1)))
        check_pot = K @ s_q
        contrib = _decode(check_pot[None, :], _F_at(F_dn_lv, lvls[t_box]))[0]
        return t_box, jnp.where(valid, contrib, 0.0)

    t_box_all, contrib_all = jax.lax.map(per_pair, (tb_s, sb_s, ok), batch_size=batch_size)
    return local.at[t_box_all].add(contrib_all, mode="drop")
