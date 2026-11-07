r"""One-call KIFMM entry points: tuned setup, solver-loop evaluate, frozen evaluator."""
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

import jaxfmm.core as core
from jaxfmm.kifmm.phases import *
from jaxfmm.kifmm.assemble import evaluate
from jaxfmm.kifmm.operators import build_operators
from jaxfmm.near import (auto_mem_limit)

__all__ = ["setup", "evaluate_setup", "compile_evaluator"]


# ──────────────────────────────────────────────────────────────────────────
# Tuned one-call setup
# ──────────────────────────────────────────────────────────────────────────
def setup(positions, p=3, N_max=64, max_depth=None, batch_size=16384,
          mem_limit=None, p2p_mixed=True, bin_gate=1.2, bin_K=16, lean_threshold=2e9,
          periodic_axes=(), domain=None, kernel=None):
    r"""One-call tuned KIFMM setup: tree plus every precomputed artifact.

    Builds the tree and all precomputed artifacts with the measured-fastest
    configuration, for repeated evaluation with :func:`evaluate_setup` (fixed
    tree, changing charges — the solver loop).

    Parameters
    ----------
    positions : (N, 3) array_like
        Source/evaluation points.
    p : int, optional
        Expansion order (points per cube-face edge).
    N_max : int, optional
        Target maximum number of particles per leaf box.
    max_depth : int or None, optional
        Tree-depth cap; ``None`` derives it from ``N_max``.
    batch_size : int, optional
        Chunk size for the batched per-box/per-pair maps.
    mem_limit : float or None, optional
        Near-field working-set bound in bytes; ``None`` uses
        :func:`jaxfmm.near.auto_mem_limit` (L2-anchored, OOM-safe).
    p2p_mixed : bool, optional
        Run the near field with a float32 kernel in box-local coordinates and
        float64 accumulation; pass ``False`` for a bitwise-f64 near field.
    bin_gate : float, optional
        The occupancy-binned near field is selected only when
        :func:`jaxfmm.near.predicted_p2p_speedup` exceeds this ratio.
    bin_K : int, optional
        Number of occupancy bins for the binned near field.
    lean_threshold : float, optional
        Padded-layout size in bytes above which the lean leaf-major sorted
        layout is used instead of materialising the padded buffers.
    periodic_axes : tuple of int, optional
        Axes (subset of ``(0, 1, 2)``) with periodic boundary conditions.
    domain : optional
        Explicit domain box; ``None`` derives it from ``positions``.
    kernel : jaxfmm.kernels.Kernel or None, optional
        Non-Laplace kernel spec; ``None`` selects the Laplace kernel.

    Returns
    -------
    s : dict
        Setup dict for :func:`evaluate_setup`.  Extra keys: ``maxleaf``,
        ``bin_ratio`` (the predicted P2P speedup that drove the flat/binned
        choice) and ``adaptive``.

    Notes
    -----
    The tuned choices, and when they flip:

    * tight ``pil`` (padded width = the realised max leaf count) and a compact
      near pair list — the padding waste of loose buffers dominates otherwise;
    * flat near field by default; occupancy-binned only when
      :func:`jaxfmm.near.predicted_p2p_speedup` exceeds ``bin_gate`` (binning
      loses on near-uniform occupancy — an unsorted 2-D scatter — and wins on
      clustered/surface clouds; the gate decides per tree);
    * ``p2p_mixed`` (default True): f32 near-field kernel in box-local
      coordinates with f64 accumulation — a faster P2P at ~3e-7 near-field
      error, harmless down to P~8 (pass ``False`` for bitwise-f64);
    * DP-tiered displacement-batched M2L (:func:`m2l_groups_tiered`);
    * W/X cross-level lists only when the tree is actually adaptive (leaves on
      more than one level);
    * ``mem_limit`` from :func:`jaxfmm.near.auto_mem_limit`.

    The box buffer starts compact (~4N/N_max) and retries doubling on
    overflow, so deep adaptive trees do not pay the dense ``8^(d+1)/7`` bound.

    PBC: the periodic far field *projects out* the net cell charge (the
    divergent monopole direction) — a non-neutral cell silently computes the
    neutralised system (jellium convention).  Pass neutral charges, or account
    for the background.
    """
    from jaxfmm.util import validate_points
    validate_points(positions, "positions")
    from jaxfmm.trees import assign_particles

    positions = jnp.asarray(positions)
    periodic = len(periodic_axes) > 0
    # PER-LEVEL operator mode for non-homogeneous kernels: the unit-scale tables +
    # h-rescale shortcut is exact only for degree -1 (Laplace-like) kernels.
    leveled = kernel is not None and kernel.homogeneity != -1.0
    # engine-agnostic part (tree, eager U/V/W/X lists, gated near-field strategy): shared
    # with uniform_fmm.setup.  V stays a LIST of host blocks — m2l_groups_tiered streams
    # them, so the full pair list never exists on device (leveled mode instead keeps the
    # concatenated ARRAY: its per-pair M2L reads V directly).
    g = core.setup_geometry(positions, N_max=N_max, max_depth=max_depth, bin_gate=bin_gate,
                            bin_K=bin_K, periodic_axes=tuple(periodic_axes), domain=domain,
                            v_as_blocks=(not periodic) and (not leveled), want_tiers=True)
    m_id, p_id, lvls, counts = g["m_id"], g["p_id"], g["lvls"], g["counts"]
    is_leaf, sort_idx = g["is_leaf"], g.pop("sort_idx")
    domain_extent, pmin = g["domain_extent"], g["pmin"]
    mb, md, maxleaf, adaptive = g["max_boxes"], g["max_depth"], g["maxleaf"], g["adaptive"]
    V, W, X = g.pop("V"), g["W"], g["X"]
    dir_cnct, p2p_bins, p2p_tiers, bin_ratio = (g["dir_cnct"], g["p2p_bins"],
                                                g["p2p_tiers"], g["bin_ratio"])
    del g

    # LEAF-MAJOR SORTED layout (lean eval at scale): perm sorts particles so each box is a
    # contiguous slice [starts[b], starts[b]+cnts[b]); rank inverts.  pos_sorted (5th,
    # STATIC element): at lean scale on UNIFORM binned trees, the Morton sort already IS
    # the leaf-major order, so assign_particles_lean derives (perm=sort_idx, rank, starts,
    # cnts) WITHOUT materialising pil at all (tens of GB at 1e9 points), the positions are
    # stored PRE-SORTED (+one-tile pad; kills the per-eval sort copy) and pil becomes a
    # width-carrying dummy (m2p/p2l, the only pil consumers, exist on adaptive trees only;
    # pos_sorted FORCES lean in evaluate so the padded path never sees the dummy).
    lean_scale = (mb * maxleaf * 4 * positions.dtype.itemsize >= lean_threshold) and not leveled
    pos_sorted = bool(lean_scale and p2p_tiers is not None)
    # adaptive trees are OK in lean mode: m2p/p2l have sorted-layout variants
    # (m2p_lean/p2l_lean), and the padded fallback at large maxleaf x large N is
    # unaffordable there.
    if pos_sorted:
        from jaxfmm.trees import assign_particles_lean
        perm, rank, starts, cnts = assign_particles_lean(sort_idx, m_id, lvls, counts,
                                                         is_leaf, md)
        sorted_meta = (perm, rank, starts, cnts, True)
        pil = jnp.zeros((1, maxleaf), jnp.int32)      # only its static WIDTH is consumed
        # (the physical permute happens AFTER the m2l grouping frees the host V blocks)
    else:
        _, pil = assign_particles(sort_idx, m_id, lvls, counts, is_leaf, md, N_max=maxleaf)
        pil_np = np.asarray(pil)
        row_cnt = (pil_np >= 0).sum(1).astype(np.int32)
        starts = np.zeros_like(row_cnt)
        np.cumsum(row_cnt[:-1], out=starts[1:])
        perm = pil_np[pil_np >= 0].astype(np.int32)
        rank = np.empty_like(perm)
        rank[perm] = np.arange(perm.shape[0], dtype=np.int32)
        sorted_meta = (jnp.asarray(perm), jnp.asarray(rank),
                       jnp.asarray(starts), jnp.asarray(row_cnt), False)
        del pil_np, perm, rank
    del sort_idx

    if kernel is not None:
        if periodic and (leveled or not kernel.has_pbc):
            raise NotImplementedError(
                f"kifmm.setup: periodic_axes with kernel '{kernel.name}' — no lattice "
                "operator exists for this kernel (Laplace only). Run open-boundary, or "
                "derive a kernel-specific far-field treatment first.")
        if p2p_mixed and kernel.name != "laplace":
            # the f32 box-local near path is Laplace-tuned; keep new kernels exact until
            # their noise floor is characterised
            p2p_mixed = False
    if leveled:
        from jaxfmm.kifmm.operators import build_operators_leveled
        ops = build_operators_leveled(p, md, float(domain_extent), kernel)
    else:
        ops = build_operators(p) if kernel is None else build_operators(p, kernel=kernel)
    if periodic:
        # displacement-batched M2L works under PBC: wrapped pairs fold into the SAME 343
        # classes after unwrapping the displacement (the grouping decodes the 3-col V);
        # the lattice op + image table complete the periodic pieces.
        from jaxfmm.kifmm.operators import gen_pbc_op_kifmm
        from jaxfmm.trees.cubic import cubic_img_table
        m2l_groups = m2l_groups_tiered(V, m_id, ops["M2L_K_table"], lvls=lvls)
        pbc_op, _pbc_info = gen_pbc_op_kifmm(p=p, periodic_axes=tuple(periodic_axes))
        img_cnct = cubic_img_table(domain_extent, tuple(periodic_axes)).astype(positions.dtype)
        V = None                                   # V is DEAD once the tiered groups exist
    else:
        if leveled:
            m2l_groups = None                      # per-pair leveled M2L consumes V directly
        else:
            m2l_groups = m2l_groups_tiered(V, m_id, ops["M2L_K_table"])
            V = None                               # V is DEAD once the tiered groups exist
        pbc_op, img_cnct = None, jnp.array([[]])
                                                   # placeholder shape: m2l_fn never reads
                                                   # it (a dense table would be wasteful)
    # move the operator tables to DEVICE once, AFTER the host consumer (m2l_groups_tiered
    # reads M2L_K_table on the host).  build_operators returns np arrays; leaving them in
    # the setup dict would make every evaluate/compile_evaluator call re-stage them H2D
    # (one copy per table + host-callback staging on every eval).
    ops = {k: (jax.device_put(v) if isinstance(v, np.ndarray) else v) for k, v in ops.items()}
    if pbc_op is not None and isinstance(pbc_op, np.ndarray):
        pbc_op = jax.device_put(pbc_op)
    if pos_sorted:
        # Permute the stored positions into leaf-major order VIA THE HOST, in chunks:
        # a device gather+pad holds 3x N momentarily (the caller's reference pins the
        # original buffer) and a host fancy-index gather holds a full extra temp — both
        # can OOM at ~1e9 points.  Runs after the m2l grouping so the host V blocks are
        # already freed.
        src_h = np.asarray(positions)
        perm_h = np.asarray(sorted_meta[0])
        pos_h = np.empty((src_h.shape[0] + maxleaf, 3), src_h.dtype)
        for lo in range(0, src_h.shape[0], 50_000_000):
            hi = min(lo + 50_000_000, src_h.shape[0])
            pos_h[lo:hi] = src_h[perm_h[lo:hi]]
        pos_h[src_h.shape[0]:] = 0.0
        del src_h, perm_h
        positions = jnp.asarray(pos_h)
        del pos_h
    if mem_limit is None:
        mem_limit = auto_mem_limit(maxleaf, dtype_bytes=positions.dtype.itemsize,
                                   binned=p2p_bins is not None)
        if jax.devices()[0].platform != "cpu":   # CPU: fused kernels never materialise the model
            print(f"[jaxfmm] mem_limit auto: {mem_limit/2**30:.1f} GiB peak-transient budget "
                  f"(speed-optimal; pass mem_limit=<bytes> to cap)")
    else:
        from jaxfmm.near import hint_legacy_mem_limit
        hint_legacy_mem_limit(mem_limit)

    return dict(positions=positions, pil=pil, dir_cnct=dir_cnct, p2p_bins=p2p_bins,
                V=V, W=W, X=X, m_id=m_id, p_id=p_id, lvls=lvls, is_leaf=is_leaf,
                domain_extent=domain_extent, pmin=pmin, ops=ops, m2l_groups=m2l_groups,
                batch_size=batch_size, mem_limit=mem_limit, p2p_mixed=p2p_mixed,
                max_depth=int(md), maxleaf=maxleaf, bin_ratio=float(bin_ratio),
                adaptive=bool(adaptive), sorted_meta=sorted_meta, p2p_tiers=p2p_tiers,
                pbc_op=pbc_op, img_cnct=img_cnct)


def evaluate_setup(s, charges, field=False):
    r"""Evaluate the fixed-tree setup for new charges — the solver-loop entry point.

    Jit-friendly (``s['max_depth']`` is static):
    ``fmm = jax.jit(lambda q: kifmm.evaluate_setup(s, q))``.

    Parameters
    ----------
    s : dict
        Setup dict from :func:`setup` (fixed tree and precomputed artifacts).
    charges : (N,) array_like
        Source charges, in the same order as the ``positions`` passed to
        :func:`setup`.
    field : bool, optional
        If True, return the analytic :math:`-\nabla\phi` of shape (N, 3)
        instead of the potential (e.g. Biot–Savart: three component solves of
        the current density give :math:`\nabla A_i`, with the curl assembled
        from the components).

    Returns
    -------
    (N,) or (N, 3) ndarray
        Potential at every source point, or the field when ``field=True``.
    """
    return evaluate(s["positions"], charges, s["pil"], s["dir_cnct"], s["V"],
                    s["m_id"], s["p_id"], s["lvls"], s["is_leaf"],
                    s["domain_extent"], s["pmin"], s["ops"],
                    W_list=s["W"], X_list=s["X"], batch_size=s["batch_size"],
                    p2p_mixed=s["p2p_mixed"], mem_limit=s["mem_limit"],
                    m2l_groups=s["m2l_groups"], p2p_bins=s["p2p_bins"],
                    max_depth=s["max_depth"], field=field,
                    sorted_meta=s.get("sorted_meta"), p2p_tiers=s.get("p2p_tiers"),
                    pbc_op=s.get("pbc_op"), img_cnct=s.get("img_cnct", jnp.array([[]])))


@partial(jax.jit, static_argnames=["skel", "field"])
def _frozen_fwd(arrs, charges, skel, field):
    r"""Module-level jitted frozen-geometry evaluator (setup-as-arguments)."""
    from jaxfmm.util import join_static
    return evaluate_setup(join_static(skel, arrs), charges, field=field)


def compile_evaluator(s, field=False):
    r"""Frozen-geometry evaluator: one jitted ``fmm(charges) -> potential``.

    The solver-loop form when *only* the charges change: the topology is never
    rebuilt; the first call compiles (persistently cached), warm calls run the
    tuned pipeline directly.

    Parameters
    ----------
    s : dict
        Setup dict from :func:`setup`.
    field : bool, optional
        If True the returned function maps ``charges -> (N, 3)`` analytic
        :math:`-\nabla\phi` instead of the potential.

    Returns
    -------
    callable
        ``fmm(charges) -> (N,)`` potential (or (N, 3) field).

    Notes
    -----
    Setup-as-arguments (not a baked closure): re-compiling after an
    identical-shape setup re-uses the same executable with zero retrace.
    Cross-geometry reuse is limited by design (tight ``pil``, compact pair
    lists and DP M2L tiers are exact per geometry — deliberate performance
    choices); geometries with matching realized shapes do share.
    """
    from jaxfmm.util import split_static
    skel, arrs = split_static(s)
    return lambda charges: _frozen_fwd(arrs, charges, skel, field)
