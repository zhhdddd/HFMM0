r"""Operator-set factories (fixed-tree and per-level tables) and the one-shot
:func:`evaluate` pipeline for the KIFMM engine."""



import jax.numpy as jnp

from jaxfmm import core   # shared cubic driver

__all__ = ["make_operators", "evaluate"]

from jaxfmm.kifmm.phases import *


def _make_operators_leveled(ops, V_list, W_list=None, X_list=None, batch_size=64,
                            mem_limit=jnp.inf, field=False):
    r"""Operator-set factory for PER-LEVEL tables (non-homogeneous kernels).

    Consumes the leveled dict from
    :func:`jaxfmm.kifmm.operators.build_operators_leveled`.  Open boundaries
    only; the M2L is the per-pair form (no displacement batching yet) —
    correctness first.

    Parameters
    ----------
    ops : dict
        Leveled operator tables (``leveled=True``).
    V_list : (max_V, 2) ndarray of int32
        Concatenated V-list pairs, -1 padded.
    W_list, X_list : ndarray or None, optional
        Cross-level pair lists for adaptive trees.
    batch_size : int, optional
        Chunk size for the batched maps.
    mem_limit : float, optional
        Working-set bound (unused by the per-pair M2L; kept for signature
        parity).
    field : bool, optional
        Emit gradient-kernel phase closures.

    Returns
    -------
    jaxfmm.core.CubicOperators
        Phase closures for :func:`jaxfmm.core.evaluate`.
    """
    md = ops["max_depth"]
    kern = ops["kernel"]
    g, g_grad = kern.g, kern.g_grad
    equiv_pts = jnp.asarray(ops["equiv_pts"])
    check_pts = jnp.asarray(ops["check_pts"])
    F_up = tuple(jnp.asarray(x) for x in ops["F_up_lv"])
    F_dn = tuple(jnp.asarray(x) for x in ops["F_dn_lv"])
    M2M_K = jnp.asarray(ops["M2M_K_lv"])
    L2L_K = jnp.asarray(ops["L2L_K_lv"])
    M2L_K = jnp.asarray(ops["M2L_K_lv"])

    def s2m_fn(src, tree):
        return p2m_padded_leveled(src.ppts, src.pchg, src.centres, src.h, tree.is_leaf,
                                  tree.lvls, check_pts, F_up, md,
                                  batch_size=batch_size, g=g)

    def m2m_fn(up, tree):
        return m2m_factored_leveled(up, tree.m_id, tree.p_id, tree.lvls, M2M_K, F_up, md)

    def m2l_fn(up, tree):
        return m2l_leveled(up, V_list, tree.m_id, tree.lvls, M2L_K, F_dn, md,
                           batch_size=batch_size)

    def l2l_fn(dn, tree):
        return l2l_factored_leveled(dn, tree.m_id, tree.p_id, tree.lvls, L2L_K, F_dn, md)

    def l2p_fn(dn, src, tree):
        return l2p_padded(dn, src.ppts, src.rev, src.centres, src.h, tree.is_leaf,
                          check_pts, batch_size=batch_size, field=field, g=g, g_grad=g_grad)

    p2l_fn = None
    if X_list is not None:
        def p2l_fn(dn, src, tree):
            return p2l_leveled(dn, src.positions, src.charges, tree.pil, src.centres,
                               src.h, X_list, tree.lvls, equiv_pts, F_dn, md,
                               batch_size=batch_size, g=g)

    m2p_fn = None
    if W_list is not None:
        def m2p_fn(up, src, tree):
            return m2p(up, src.positions, tree.pil, src.centres, src.h, W_list,
                       equiv_pts, batch_size=batch_size, field=field, g=g, g_grad=g_grad)

    return core.CubicOperators(s2m=s2m_fn, m2m=m2m_fn, m2l=m2l_fn, l2l=l2l_fn,
                               l2p=l2p_fn, p2l=p2l_fn, m2p=m2p_fn)


def make_operators(ops, V_list, W_list=None, X_list=None, batch_size=64,
                   m2l_n_pairs="auto", m2l_groups=None, factored=None, mem_limit=jnp.inf,
                   field=False, pbc_op=None, img_cnct=jnp.array([[]])):
    r"""KIFMM operator set for the shared cubic driver (:func:`jaxfmm.core.evaluate`).

    Closures over the SETUP artifacts only (operator tables from
    :func:`jaxfmm.kifmm.operators.build_operators`, the V/W/X lists, the
    optional displacement groups); they delegate to the phase functions
    unchanged, so a fixed tree can be re-evaluated with new charges without
    rebuilding this set.

    Parameters
    ----------
    ops : dict
        Operator tables from :func:`jaxfmm.kifmm.operators.build_operators`
        (or the leveled dict from
        :func:`jaxfmm.kifmm.operators.build_operators_leveled`).
    V_list : (max_V, 2) ndarray of int32
        V-list pairs, -1 padded (unused when ``m2l_groups`` is given).
    W_list, X_list : ndarray or None, optional
        Cross-level pair lists; enable the adaptive M2P/P2L phases.  Leave
        ``None`` on uniform trees.
    batch_size : int, optional
        Chunk-size ceiling for the batched per-box/per-pair maps.
    m2l_n_pairs : int or "auto", optional
        Compacted pair-buffer size for the generic per-pair M2L
        ("auto" derives it via :func:`v_pair_count`).
    m2l_groups : tuple or None, optional
        Displacement-batched M2L precompute: either the flat 4-tuple from
        :func:`m2l_groups_padded` (dispatches to :func:`m2l_batched`) or the
        tier tuple from :func:`m2l_groups_tiered`
        (:func:`m2l_batched_tiered`, which recovers the group padding waste).
        ``None`` selects the generic per-pair :func:`m2l`.
    factored : bool or None, optional
        Select the backward-stable FACTORED decodes (pvfmm cicp15 §2.1).
        ``None`` (default) uses them whenever the ops dict carries the factor
        keys; ``False`` forces the materialised legacy operators.  f64 results
        are identical at P<=6; factored is REQUIRED for the f32 mode (P=3-4)
        and for deep f64 accuracy at high P (see :func:`_decode`).
    mem_limit : float, optional
        Working-set bound (bytes) for the tiered M2L chunking.
    field : bool, optional
        Emit gradient-kernel closures (:math:`-\nabla\phi`).
    pbc_op : ndarray or None, optional
        Lattice operator from :func:`jaxfmm.kifmm.operators.gen_pbc_op_kifmm`;
        enables the periodic far-tail phase.
    img_cnct : ndarray, optional
        Periodic image offset table (image-encoded source boxes in W/X).

    Returns
    -------
    jaxfmm.core.CubicOperators
        Phase closures for :func:`jaxfmm.core.evaluate`.

    Notes
    -----
    S2M/L2P run through the padded-layout variants
    (:func:`p2m_padded`/:func:`l2p_padded`), consuming the
    ``ppts``/``pchg``/``rev`` blocks the driver builds once per call; on the
    lean sorted layout the ``*_lean`` variants run instead.
    """
    if isinstance(ops, dict) and ops.get("leveled"):
        assert pbc_op is None and img_cnct.shape[1] == 0, "leveled kernels: open BC only"
        return _make_operators_leveled(ops, V_list, W_list=W_list, X_list=X_list,
                                       batch_size=batch_size, mem_limit=mem_limit, field=field)
    equiv_pts = jnp.asarray(ops["equiv_pts"])
    check_pts = jnp.asarray(ops["check_pts"])
    # pluggable kernel: non-Laplace operator sets carry their Kernel spec; Laplace (or
    # legacy ops dicts without the key) keeps the module defaults so the default path's
    # jit keys/graphs are UNTOUCHED.
    _kern = ops.get("kernel")
    if not hasattr(_kern, "name"):        # absent (legacy dicts) or degraded by tree-mapping
        _kern = None                      # utilities (cast_operators): treat as Laplace
    _kw = ({} if _kern is None or _kern.name == "laplace"
           else dict(g=_kern.g, g_grad=_kern.g_grad))
    factored = ("F_up" in ops) if factored is None else (factored and "F_up" in ops)
    if factored:                                     # factor triples as the decode arguments
        dec_up = tuple(jnp.asarray(x) for x in ops["F_up"])
        dec_dn = tuple(jnp.asarray(x) for x in ops["F_dn"])
        F_par = tuple(jnp.asarray(x) for x in ops["F_par"])
        M2M_K = jnp.asarray(ops["M2M_K"])
        L2L_K = jnp.asarray(ops["L2L_K"])
    else:                                            # materialised legacy matrices
        dec_up = jnp.asarray(ops["pinv_up"])
        dec_dn = jnp.asarray(ops["pinv_dn"])
        M2M_mats = jnp.asarray(ops["M2M_mats"])
        L2L_mats = jnp.asarray(ops["L2L_mats"])

    # Every per-box/per-pair lax.map whose working set scales with the pil width
    # (S2M, L2P, M2P, P2L: one (width, n_s, 3) distance tensor per mapped element)
    # bounds its CHUNK MEMORY at ~4 GB — a flat batch_size makes 10+ GB chunks on
    # dense-leaf trees (maxleaf 512+).  NOTE the bound is VRAM-scale, NOT mem_limit:
    # mem_limit anchors the near-field batches, and sizing these maps from it
    # throttles them to tiny launch-bound batches.
    def _map_batch(width):
        w = width * int(equiv_pts.shape[0]) * 3 * 8
        return max(16, min(int(batch_size), int(4e9 // max(w, 1))))

    def s2m_fn(src, tree):
        if src.starts is not None:                   # LEAN: sorted-slice arithmetic gathers
            return p2m_lean(src.pos_s, src.q_s, src.starts, src.cnts, tree.pil.shape[1],
                            src.centres, src.h, tree.is_leaf, equiv_pts, check_pts, dec_up,
                            batch_size=_map_batch(tree.pil.shape[1]), **_kw)
        return p2m_padded(src.ppts, src.pchg, src.centres, src.h,
                          tree.is_leaf, equiv_pts, check_pts, dec_up,
                          batch_size=_map_batch(src.ppts.shape[1]), **_kw)

    if factored:
        def m2m_fn(up, tree):
            return m2m_factored(up, tree.m_id, tree.p_id, tree.lvls, M2M_K, F_par,
                                tree.max_depth)
    else:
        def m2m_fn(up, tree):
            return m2m(up, tree.m_id, tree.p_id, tree.lvls, M2M_mats, tree.max_depth)

    if m2l_groups is not None and isinstance(m2l_groups[0], (tuple, list)):
        def m2l_fn(up, tree):                        # tiered displacement-batched M2L
            from jaxfmm.near import ws_limit
            import numpy as _np
            ml_tiers = ws_limit(mem_limit, tree.pil.shape[1]) if _np.isfinite(mem_limit) else mem_limit
            return m2l_batched_tiered(up, m2l_groups, dec_dn, mem_limit=ml_tiers)   # transient-priced tiers -> row working set (0.3.1 budget semantics)
    elif m2l_groups is not None:                     # flat displacement-batched M2L
        def m2l_fn(up, tree):
            return m2l_batched(up, *m2l_groups, dec_dn)
    else:                                            # generic per-pair M2L
        n_pairs = (v_pair_count(V_list, batch_size=batch_size)
                   if m2l_n_pairs == "auto" else m2l_n_pairs)
        M2L_K_table = jnp.asarray(ops["M2L_K_table"])

        def m2l_fn(up, tree):
            return m2l(up, V_list, tree.m_id, M2L_K_table, dec_dn,
                       n_pairs=n_pairs, batch_size=batch_size, lvls=tree.lvls)

    if factored:
        def l2l_fn(dn, tree):
            return l2l_factored(dn, tree.m_id, tree.p_id, tree.lvls, L2L_K, dec_dn,
                                tree.max_depth)
    else:
        def l2l_fn(dn, tree):
            return l2l(dn, tree.m_id, tree.p_id, tree.lvls, L2L_mats, tree.max_depth)

    def l2p_fn(dn, src, tree):
        if src.starts is not None:                   # LEAN: chunk-scan + unique scatter-add
            return l2p_lean(dn, src.pos_s, src.starts, src.cnts, tree.pil.shape[1],
                            src.centres, src.h, tree.is_leaf, check_pts,
                            batch_size=_map_batch(tree.pil.shape[1]), field=field, **_kw)
        return l2p_padded(dn, src.ppts, src.rev, src.centres, src.h, tree.is_leaf,
                          check_pts, batch_size=_map_batch(src.ppts.shape[1]), field=field, **_kw)

    p2l_fn = None
    if X_list is not None:                                  # feeds local expansions -> the
        def p2l_fn(dn, src, tree):                          # L2P gradient covers it, no flag
            if src.starts is not None:                      # LEAN: sorted-slice sources
                return p2l_lean(dn, src.pos_s, src.q_s, src.starts, src.cnts,
                                tree.pil.shape[1], src.centres, src.h, X_list,
                                equiv_pts, dec_dn, batch_size=_map_batch(tree.pil.shape[1]),
                                img_cnct=img_cnct, **_kw)
            return p2l(dn, src.positions, src.charges, tree.pil, src.centres, src.h,
                       X_list, equiv_pts, dec_dn, batch_size=_map_batch(tree.pil.shape[1]),
                       img_cnct=img_cnct, **_kw)

    m2p_fn = None
    if W_list is not None:
        def m2p_fn(up, src, tree):
            if src.starts is not None:                      # LEAN: SORTED-order output —
                return m2p_lean(up, src.pos_s, src.starts, src.cnts,   # the driver folds it
                                tree.pil.shape[1], src.centres, src.h, # in before un-sorting
                                W_list, equiv_pts,
                                batch_size=_map_batch(tree.pil.shape[1]), field=field,
                                img_cnct=img_cnct, **_kw)
            return m2p(up, src.positions, tree.pil, src.centres, src.h,
                       W_list, equiv_pts, batch_size=_map_batch(tree.pil.shape[1]),
                       field=field, img_cnct=img_cnct, **_kw)

    pbc_fn = None
    if pbc_op is not None:
        # periodic far tail at the VIRTUAL ROOT: the cubic tree has no level-0 box, so the
        # 8 level-1 boxes are aggregated into the cell equiv (scale-free M2M_mats, octant =
        # m_id), the lattice operator (gen_pbc_op_kifmm, RAW check potentials) is applied,
        # decoded (pinv_dn: at the cell level the lattice IS the only check contribution),
        # and distributed back with L2L_mats.
        M2M_m = jnp.asarray(ops["M2M_mats"]); L2L_m = jnp.asarray(ops["L2L_mats"])
        pinv_dn_m = jnp.asarray(ops["pinv_dn"]); pbc_mat = jnp.asarray(pbc_op)

        def pbc_fn(dn, up, tree):
            is_l1 = (tree.lvls == 1)
            cell_up = jnp.zeros((pbc_mat.shape[1],), up.dtype)
            idxs, press = [], []
            for k in range(8):
                hit = is_l1 & (tree.m_id == jnp.uint64(k))
                idx = jnp.argmax(hit)
                pres = jnp.any(hit).astype(up.dtype)
                idxs.append(idx); press.append(pres)
                cell_up = cell_up + pres * (M2M_m[k] @ up[idx])
            xi = pinv_dn_m @ (pbc_mat @ cell_up)
            for k in range(8):
                dn = dn.at[idxs[k]].add(press[k] * (L2L_m[k] @ xi))
            return dn

    return core.CubicOperators(s2m=s2m_fn, m2m=m2m_fn, m2l=m2l_fn, l2l=l2l_fn,
                               l2p=l2p_fn, p2l=p2l_fn, m2p=m2p_fn, pbc=pbc_fn)


def evaluate(positions, charges, particles_in_leaf, dir_cnct, V_list,
             m_id, p_id, lvls, is_leaf, domain_extent, pmin, ops,
             W_list=None, X_list=None, batch_size=64,
             m2l_n_pairs="auto", p2p_mixed=False, mem_limit=jnp.inf,
             m2l_groups=None, p2p_bins=None, max_depth=None, factored=None,
             field=False, sorted_meta=None, p2p_tiers=None,
             pbc_op=None, img_cnct=jnp.array([[]])):
    r"""Full KIFMM potential evaluation.

    Back-compat wrapper around the shared cubic driver: builds a
    :class:`jaxfmm.core.CubicTree` + :func:`make_operators` and calls
    :func:`jaxfmm.core.evaluate`.

    Parameters
    ----------
    positions : (N, 3) ndarray
        Particle coordinates.
    charges : (N,) ndarray
        Particle charges.
    particles_in_leaf : (max_boxes, N_max) ndarray of int32
        Per-leaf particle indices, -1 padded (from ``assign_particles``).
    dir_cnct : ndarray
        Precomputed near pair list from :func:`jaxfmm.near.near_pairs`
        (host-side, so this call stays jittable).
    V_list : (max_V, 2) ndarray of int32
        V-list pairs, -1 padded.
    m_id, p_id, lvls, is_leaf : ndarray
        Tree arrays (Morton ids, parent indices, levels, leaf mask).
    domain_extent : float
        Root box edge length.
    pmin : (3,) ndarray
        Root box minimum corner.
    ops : dict
        Operator tables from :func:`jaxfmm.kifmm.operators.build_operators`.
    W_list, X_list : ndarray or None, optional
        Cross-level lists (from ``build_wx_lists``) for the adaptive far
        field; without them the result is exact only on uniform trees.
    batch_size : int, optional
        Chunk size for the batched maps.
    m2l_n_pairs : int or "auto", optional
        Compacted V-pair buffer size for the generic per-pair M2L.
    p2p_mixed : bool, optional
        Run the near field with a float32 kernel and float64 accumulation;
        the far field stays float64 so total accuracy is essentially
        unchanged.
    mem_limit : float, optional
        Working-set bound (bytes) for the near field and tiered M2L.
    m2l_groups : tuple or None, optional
        Displacement-batched M2L precompute (see :func:`make_operators`).
    p2p_bins : tuple or None, optional
        ``(bin_offsets, bin_widths)`` static tuples from
        :func:`jaxfmm.near.near_pairs_binned` (with ``dir_cnct`` then the
        matching flat list); selects the occupancy-binned near field.
    max_depth : int or None, optional
        Static tree depth (Python int).  Pass it to make the whole call
        jit-traceable — the default derives it eagerly via ``int(lvls.max())``,
        which forces a device sync and cannot run under ``jax.jit``.
    factored : bool or None, optional
        Backward-stable factored decodes (see :func:`make_operators`).
    field : bool, optional
        Return the analytic :math:`-\nabla\phi` (N, 3) instead of the
        potential (all phases use the gradient kernel; the equivalent
        densities are point charges, so no new operator tables are needed).
    sorted_meta : tuple or None, optional
        Leaf-major sorted layout ``(perm, rank, starts, cnts, pos_sorted)``
        from :func:`jaxfmm.kifmm.setup`; enables the lean mode.
    p2p_tiers : optional
        Near-field tier metadata for the lean P2P.
    pbc_op : ndarray or None, optional
        Lattice operator for periodic boundaries.
    img_cnct : ndarray, optional
        Periodic image offset table.

    Returns
    -------
    pot : (N,) ndarray of float64
        Total potential: near field = P2P (``dir_cnct``) + M2P (W-list);
        far field = M2L (V-list) + P2L (X-list) → L2L → L2P.
        With ``field=True``: (N, 3) analytic :math:`-\nabla\phi` instead.
    """
    tree = core.CubicTree(m_id, p_id, lvls, is_leaf, particles_in_leaf,
                          domain_extent, pmin,
                          int(lvls.max()) if max_depth is None else max_depth)
    op_set = make_operators(ops, V_list, W_list=W_list, X_list=X_list,
                            batch_size=batch_size, m2l_n_pairs=m2l_n_pairs,
                            m2l_groups=m2l_groups, factored=factored,
                            mem_limit=(float(mem_limit) if mem_limit != jnp.inf else jnp.inf),
                            field=field, pbc_op=pbc_op, img_cnct=img_cnct)
    # LEAN mode by size: above ~2 GB of padded layout, S2M/L2P/P2P run on the LEAF-MAJOR
    # SORTED layout (sorted_meta from setup) with arithmetic slice indices — no
    # (max_boxes, maxleaf, ...) tensor exists (the padded layout is a large multiple of
    # the raw points and drives the eval peak).  Below, materialize: the memory is
    # irrelevant and the padded P2P avoids the per-call sort gathers.  pil.shape is
    # static, so the choice never retraces.
    padded_bytes = particles_in_leaf.size * 4 * positions.dtype.itemsize
    pos_sorted = sorted_meta is not None and len(sorted_meta) > 4 and sorted_meta[4]
    lean = sorted_meta is not None and (pos_sorted or padded_bytes >= 2e9)
    _kern = ops.get("kernel") if isinstance(ops, dict) else None
    if not hasattr(_kern, "name"):
        _kern = None
    return core.evaluate(positions, charges, tree, op_set, dir_cnct, near_kernel=_kern,
                         p2p_bins=p2p_bins, p2p_mixed=p2p_mixed, mem_limit=mem_limit,
                         field=field, materialize_padded=not lean,
                         sorted_meta=sorted_meta if lean else None,
                         p2p_tiers=p2p_tiers if lean else None, img_cnct=img_cnct)
