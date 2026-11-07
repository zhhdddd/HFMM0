r"""Fully-jit moving-geometry KIFMM: tree + lists rebuilt INSIDE jit against frozen
buffer caps, with realized counts certifying every call."""


from functools import partial

import jax
import jax.numpy as jnp

from jaxfmm import core   # shared cubic driver

__all__ = ["fulljit_setup", "compile_fulljit", "fulljit_counts_ok", "fulljit_oversized", "fulljit_caps",
           "fulljit_refit"]

from jaxfmm.kifmm.phases import *
from jaxfmm.kifmm.assemble import make_operators


# ──────────────────────────────────────────────────────────────────────────
# Fully-jit moving-geometry KIFMM: tree + lists recomputed INSIDE jit with
# frozen buffer caps, realized counts certifying every call.
# ──────────────────────────────────────────────────────────────────────────
_MAXI = None


def _compact_capped(pairs, valid, cap, fill):
    r"""Capped in-jit pair-list compaction.

    The eager setup compactions have dynamic shapes; inside jit the valid rows
    are scattered into a fixed ``(cap, 2)`` buffer instead, with overflow rows
    dropped.

    Parameters
    ----------
    pairs : (n, 2) ndarray
        Candidate pair list.
    valid : (n,) bool ndarray
        Mask of rows to keep.
    cap : int
        Static output buffer size.
    fill : int
        Fill value for unused buffer rows.

    Returns
    -------
    out : (cap, 2) ndarray
        Compacted pair buffer.
    count : ndarray scalar
        Realized number of valid pairs — can exceed ``cap``; this is the
        overflow certificate the caller must check.
    """
    dst = jnp.where(valid, jnp.cumsum(valid) - 1, cap)
    out = jnp.full((cap, 2), fill, pairs.dtype).at[dst].set(pairs, mode="drop")
    return out, valid.sum()


def fulljit_setup(positions, p=3, N_max=64, headroom=0.15, max_depth=None, batch_size=64,
                  mem_limit=None, p2p_mixed=True, cap_floor=None):
    r"""Freeze the static caps for the fully-jit moving-geometry KIFMM.

    Probes the tree at the reference ``positions`` and fits every
    data-dependent buffer (box count, depth, pil width, near/V/W/X pair
    counts) with a ``headroom`` fraction on the geometric cap ladder.  Consumed by
    :func:`compile_fulljit`; unlike :func:`jaxfmm.kifmm.setup` (frozen
    geometry, changing charges) the compiled forward REBUILDS tree + lists
    from ``positions`` on every call — moving geometry with zero recompiles
    while the caps hold.

    Parameters
    ----------
    positions : (N, 3) array_like
        Reference geometry used to probe the buffer caps.
    p : int, optional
        Expansion order.
    N_max : int, optional
        Target maximum particles per leaf box.
    headroom : float, optional
        Spare capacity on each realized buffer size, as a fraction
        (:func:`jaxfmm.trees.connectivity._fit_cap`).
    max_depth : int or None, optional
        Tree-depth cap; ``None`` derives it from ``N_max``.
    batch_size : int, optional
        Chunk size for the batched per-box/per-pair maps.
    mem_limit : float or None, optional
        Near-field working-set bound in bytes; ``None`` uses
        :func:`jaxfmm.near.auto_mem_limit`.
    p2p_mixed : bool, optional
        Mixed-precision near field (f32 kernel, f64 accumulation).
    cap_floor : dict or None, optional
        Lower bounds for the caps (:func:`fulljit_caps` of an earlier setup; the guarded
        driver's grow-only rebuilds and ``size_from``).

    Returns
    -------
    s : dict
        ``{"meta": tuple, "arrs": dict}`` — hashable static caps plus the
        operator arrays, consumed by :func:`compile_fulljit`.

    Raises
    ------
    ValueError
        If the U- or W/X-list overflows already at the reference geometry.

    Notes
    -----
    Overflow under motion is detectable, never silent: the forward returns the
    realized counts (:func:`fulljit_counts_ok`); run loops through the guarded
    :func:`jaxfmm.driver.kifmm_fulljit_driver`.

    The W/X cross-level phases are ALWAYS compiled in (with probed caps, at
    least one granule): a uniform probe tree can turn adaptive under motion,
    and phases absent from the executable would silently drop that physics.
    """
    import numpy as np
    from jaxfmm.trees import (build_fmm_tree_auto, build_interaction_lists, build_ulist_adaptive, build_wx_lists)
    from jaxfmm.near import near_pairs, auto_mem_limit
    from jaxfmm.kifmm.operators import build_operators
    from jaxfmm.trees.connectivity import _fit_cap

    positions = jnp.asarray(positions)
    fl = cap_floor or {}
    fit = lambda key, used, gran: max(int(_fit_cap(used, gran, headroom)), fl.get(key, 0))
    out, mb, md, maxleaf = build_fmm_tree_auto(positions, N_max=N_max, max_depth=max_depth,
                                               refine_factor=2)
    (m_id, p_id, lvls, counts, is_leaf, sort_idx, _, dext, pmin, _, _) = out
    pil_cap = fit("pil_cap", maxleaf, 16)
    _, V = build_interaction_lists(m_id, lvls, is_leaf, mb, md)
    U, ovU = build_ulist_adaptive(m_id, lvls, is_leaf, mb, md)
    if bool(ovU):
        raise ValueError("kifmm.fulljit_setup: U-list overflow at the reference geometry")
    W, X, ovWX = build_wx_lists(m_id, lvls, is_leaf, mb, md)
    if bool(ovWX):
        raise ValueError("kifmm.fulljit_setup: W/X-list overflow at the reference geometry")
    dc = np.asarray(near_pairs(U, is_leaf, lvls))
    n_dir = int(((dc[:, 0] >= 0) & (dc[:, 1] >= 0)).sum())
    n_v = int(((np.asarray(V)[:, 0] >= 0) & (np.asarray(V)[:, 1] >= 0)).sum())
    n_w = int(((np.asarray(W)[:, 0] >= 0) & (np.asarray(W)[:, 1] >= 0)).sum())
    n_x = int(((np.asarray(X)[:, 0] >= 0) & (np.asarray(X)[:, 1] >= 0)).sum())
    ops = build_operators(p)
    ops.pop("kernel", None)     # fulljit is Laplace-only; the Kernel spec (a NamedTuple
                                # with a str) must not enter the hand-rolled array split
    meta = dict(p=p, n_s=ops["n_s"], N_max=N_max, mb=int(mb), md=int(md),
                pil_cap=int(pil_cap),
                dir_cap=fit("dir_cap", n_dir, 1024), v_cap=fit("v_cap", n_v, 4096),
                w_cap=fit("w_cap", n_w, 1024), x_cap=fit("x_cap", n_x, 1024),
                batch_size=int(batch_size), p2p_mixed=bool(p2p_mixed),
                mem_limit=float(auto_mem_limit(pil_cap, positions.dtype.itemsize)
                                if mem_limit is None else mem_limit))
    if mem_limit is not None:
        from jaxfmm.near import hint_legacy_mem_limit
        hint_legacy_mem_limit(mem_limit)                 # pre-0.3.1 working-set values are ~6*N_max too small
    arrs = {k: v for k, v in ops.items() if k not in ("p", "n_s")}
    return dict(meta=tuple(sorted(meta.items())), arrs=arrs)


def _kifmm_fulljit_body(arrs, positions, charges, meta_d, field, remat=False):
    r"""Traced body of the moving-geometry forward: in-jit tree + list rebuild
    against the frozen caps in ``meta_d``, then the shared cubic evaluate.
    Returns ``(pot, counts)`` with the realized counts for certification."""
    from jaxfmm.trees import (build_fmm_tree, assign_particles, build_interaction_lists,
                              build_ulist_adaptive, build_wx_lists)
    sg = jax.lax.stop_gradient
    mb, md, N_max = meta_d["mb"], meta_d["md"], meta_d["N_max"]
    out = build_fmm_tree(positions, mb, max_depth=md, N_max=N_max)
    (m_id, p_id, lvls, cnts, is_leaf, sort_idx, _, dext, pmin, ovd, ovb) = out
    m_id, p_id, lvls, cnts, is_leaf, sort_idx = map(sg, (m_id, p_id, lvls, cnts, is_leaf,
                                                         sort_idx))
    leaf = is_leaf & (lvls > 0)
    maxleaf = jnp.where(leaf, cnts, 0).max()
    _, pil = assign_particles(sort_idx, m_id, lvls, cnts, is_leaf, md,
                              N_max=meta_d["pil_cap"])
    pil = sg(pil)
    _, V = build_interaction_lists(m_id, lvls, is_leaf, mb, md)
    U, ovU = build_ulist_adaptive(m_id, lvls, is_leaf, mb, md)
    W, X, ovWX = build_wx_lists(m_id, lvls, is_leaf, mb, md)
    idt = U.dtype
    SENT = jnp.iinfo(idt).max
    boxes = jnp.arange(mb, dtype=idt)
    selfp = jnp.stack([jnp.where(leaf, boxes, SENT)] * 2, 1)          # leaf self-pairs
    cand = jnp.concatenate([U, selfp])                                # (max_U + mb, 2)
    okd = (cand[:, 0] >= 0) & (cand[:, 0] != SENT) & (cand[:, 1] >= 0) & (cand[:, 1] != SENT)
    dir_cnct, n_dir_used = _compact_capped(sg(cand), sg(okd), meta_d["dir_cap"], SENT)
    dir_cnct = dir_cnct.astype(jnp.arange(1).dtype)   # eval_direct's batching pad expects the
                                                      # default int dtype (legacy lists are int64)
    okw = (W[:, 0] >= 0) & (W[:, 1] >= 0)
    okx = (X[:, 0] >= 0) & (X[:, 1] >= 0)
    Wc, n_w_used = _compact_capped(sg(W), sg(okw), meta_d["w_cap"], -1)
    Xc, n_x_used = _compact_capped(sg(X), sg(okx), meta_d["x_cap"], -1)
    n_v_used = ((V[:, 0] >= 0) & (V[:, 1] >= 0)).sum()
    tree = core.CubicTree(m_id, p_id, lvls, is_leaf, pil, dext, pmin, md)
    ops = dict(arrs, p=meta_d["p"], n_s=meta_d["n_s"])
    op_set = make_operators(ops, sg(V), W_list=Wc, X_list=Xc,
                            batch_size=meta_d["batch_size"], m2l_n_pairs=meta_d["v_cap"],
                            mem_limit=meta_d["mem_limit"], field=field)
    pot = core.evaluate(positions, charges, tree, op_set, dir_cnct,
                        p2p_mixed=meta_d["p2p_mixed"], mem_limit=meta_d["mem_limit"],
                        field=field, remat=remat)
    counts = dict(ovd=ovd, ovb=ovb, ovU=ovU, ovWX=ovWX, maxleaf=maxleaf,
                  n_dir_used=n_dir_used, n_v_used=n_v_used, n_w_used=n_w_used,
                  n_x_used=n_x_used)
    return pot, {k: sg(v) for k, v in counts.items()}


@partial(jax.jit, static_argnames=["meta", "field", "remat", "return_counts"])
def _kifmm_fulljit_fwd(arrs, positions, charges, meta, field=False, remat=False,
                       return_counts=False):
    r"""Module-level jitted moving-geometry KIFMM forward (shared across rebuilds).

    ``remat=True`` rematerialises the near-field P2P in the backward pass —
    REQUIRED for ``jax.grad`` at scale: the taped pairwise near-field tensors
    scale with the pair count and OOM otherwise."""
    pot, counts = _kifmm_fulljit_body(arrs, positions, charges, dict(meta), field, remat)
    return (pot, counts) if return_counts else pot


def fulljit_counts_ok(s, counts):
    r"""Host-side verdict on the forward's realized counts vs the frozen caps.

    Parameters
    ----------
    s : dict
        Setup dict from :func:`fulljit_setup` (holds the caps in ``s["meta"]``).
    counts : dict
        Realized counts returned by the forward
        (:func:`_kifmm_fulljit_fwd` with ``return_counts=True``).

    Returns
    -------
    ok : bool
        True when every realized count fits its cap and no builder truncated.
    info : dict
        The realized counts as plain Python values.
    """
    import numpy as np
    m = dict(s["meta"])
    info = {k: np.asarray(v).tolist() for k, v in counts.items()}
    # ovd (leaf > N_max at max_depth) is the NORMAL auto-tuned state: build_fmm_tree_auto
    # accepts maxleaf <= refine_factor*N_max by design -- correctness needs the particles to
    # FIT (pil_cap), which is checked below. ovb (box buffer truncated) IS fatal.
    ok = (not info["ovb"] and not info["ovU"] and not info["ovWX"]
          and info["maxleaf"] <= m["pil_cap"] and info["n_dir_used"] <= m["dir_cap"]
          and info["n_v_used"] <= m["v_cap"] and info["n_w_used"] <= m["w_cap"]
          and info["n_x_used"] <= m["x_cap"])
    return bool(ok), info


def fulljit_caps(s):
    r"""The data-dependent caps frozen in :func:`fulljit_setup` (its ``cap_floor`` currency)."""
    m = dict(s["meta"])
    return {k: int(m[k]) for k in ("pil_cap", "dir_cap", "v_cap", "w_cap", "x_cap")}


def fulljit_oversized(s, info, headroom=0.15, cap_floor=None):
    r"""Lazy-downsize predicate for the moving-geometry KIFMM.

    True when a refit at the realized counts would shrink at least one buffer by
    ``DOWNSIZE_BINS`` cap-ladder bins and grow none (the element analog is
    ``jaxfmm.fem.fulljit_oversized``).

    Parameters
    ----------
    s : dict
        Setup dict from :func:`fulljit_setup`.
    info : dict
        Realized counts from :func:`fulljit_counts_ok`.
    headroom : float, optional
        Spare-capacity fraction of the hypothetical refit.
    cap_floor : dict or None, optional
        Caps the refit would keep regardless (the driver's ``size_from`` floor).

    Returns
    -------
    bool
        Whether re-running :func:`fulljit_setup` would pay off.
    """
    from jaxfmm.trees.connectivity import _caps_oversized
    m, fl = dict(s["meta"]), cap_floor or {}
    return _caps_oversized([(info[u], m[c], g, fl.get(c, 0)) for u, c, g in _CAP_KEYS], headroom)


def fulljit_refit(s, info, headroom=0.15, cap_floor=None):
    r"""The caps a refit at the realised counts ``info`` would freeze (the
    :func:`fulljit_caps` currency, never below ``cap_floor``) -- prices the guarded driver's
    lazy downsizes."""
    from jaxfmm.trees.connectivity import _fit_cap
    fl = cap_floor or {}
    return {c: max(_fit_cap(info[u], g, headroom), fl.get(c, 0)) for u, c, g in _CAP_KEYS}


_CAP_KEYS = (("maxleaf", "pil_cap", 16), ("n_dir_used", "dir_cap", 1024), ("n_v_used", "v_cap", 4096),
             ("n_w_used", "w_cap", 1024), ("n_x_used", "x_cap", 1024))


def compile_fulljit(s, field=False, remat=False, return_counts=False):
    r"""Compile ``fwd(positions, charges) -> pot/H`` with an in-jit tree rebuild.

    The returned forward rebuilds the tree + interaction lists inside jit on
    every call, against the frozen caps from :func:`fulljit_setup`.
    Module-level jit: rebuilds with repeating caps cost no retrace/compile.

    Parameters
    ----------
    s : dict
        Setup dict from :func:`fulljit_setup`.
    field : bool, optional
        If True the forward returns the analytic :math:`-\nabla\phi` (N, 3).
    remat : bool, optional
        Rematerialise the near-field P2P in the backward pass (required for
        ``jax.grad`` at scale; see :func:`_kifmm_fulljit_fwd`).
    return_counts : bool, optional
        Also return the realized counts for :func:`fulljit_counts_ok`.

    Returns
    -------
    callable
        ``fwd(positions, charges)``.

    Notes
    -----
    Silently truncates on cap overflow — prefer the guarded
    :func:`jaxfmm.driver.kifmm_fulljit_driver` for moving-geometry loops.
    """
    meta, arrs = s["meta"], s["arrs"]

    def fwd(positions, charges):
        return _kifmm_fulljit_fwd(arrs, positions, charges, meta, field, remat,
                                  return_counts)
    return fwd
