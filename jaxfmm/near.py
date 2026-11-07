r"""Shared near field (P2P), used by every FMM driver.

The near field is a direct point-point :math:`1/r` sum -- kernel-specific but
representation-agnostic (the same for the KIFMM, the rotation quad-tree, and the uniform
spherical-harmonic path), so it lives here once and is imported by the flexible-tree,
uniform and KIFMM engines.

  eval_direct / eval_direct_binned   -- the P2P appliers (padded per-box points; separate eval points;
                                        optional PBC images, field=True, mixed-f32, occupancy tiers).
  _build_padded                      -- gather (positions, charges) into per-box (nbox, N_max, .) blocks.
  near_pairs / near_pairs_binned     -- the U-list+self near pair list and its occupancy-width tiering.
  predicted_p2p_speedup              -- setup-time gate for whether tiering is worth it.
  auto_mem_limit / autotune_mem_limit-- L2-cache-anchored near-field batch sizing
                                        (mem_limit = PEAK-TRANSIENT byte budget since 0.3.1;
                                        ws_limit derives the minor-stage working-set cap).
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
from functools import lru_cache, partial
from jaxfmm.util import get_gpu_l2_cache_size

@jax.custom_vjp
def _take_sorted(x, idx):
    r"""``x[idx]`` for SORTED ``idx`` (out-of-range sentinels allowed) whose transpose is a
    standalone sorted scatter-add. Reverse mode through a plain gather emits an unsorted atomic
    scatter (contended when many pairs share an eval box); flagging the gather sorted instead
    makes XLA fuse a sorted scatter with the remat recompute, which computed WRONG gradients
    (RTX PRO 6000 MIG slice, jax 0.10.2: shape gradient -1.1e-3 vs finite difference 1.5e-2)."""
    return x[idx]


def _take_sorted_fwd(x, idx):
    return x[idx], (x, idx)


def _take_sorted_bwd(res, ct):
    x, idx = res
    ct = jax.lax.optimization_barrier(ct)                   # keep the scatter out of the recompute fusion
    return jnp.zeros_like(x).at[idx].add(ct, indices_are_sorted=True, mode="drop"), None


_take_sorted.defvjp(_take_sorted_fwd, _take_sorted_bwd)


@partial(jax.jit, static_argnames=['field', 'mem_limit', 'remat', 'mixed', 'return_padded', 'k2', 'k2g',
                                  'kernel_dtype', 'kernel_scale', 'src_fn', 'src_width', 'coincidence_floor',
                                  'batch_rule'])
def eval_direct(padded_pts, padded_chrgs, padded_eval_pts, rev_idcs, dir_cnct, img_cnct = jnp.array([[]]), field = False, mem_limit = jnp.inf, remat = False, mixed = False, centres = None, eval_centres = None, return_padded = False, k2 = None, k2g = None, img_phase = None, kernel_dtype = None, kernel_scale = None, src_fn = None, src_width = None, coincidence_floor = "pair", batch_rule = "pricing"):
    r"""
    Evaluate the near-field potential directly (P2P).

    Parameters
    ----------
    padded_pts : (nbox, w, 3) array
        Padded per-box source points (:func:`_build_padded`).
    padded_chrgs : (nbox, w) array
        Padded per-box charges (0 on padding).
    padded_eval_pts : (nbox, w_e, 3) array
        Padded per-box evaluation points (may be ``padded_pts``).
    rev_idcs : (N,) array
        Flat index of each particle in the padded layout (un-pads the result).
    dir_cnct : (n_pairs, 2) array
        Near box-pair list; the source column may be image-encoded under PBC.
    img_cnct : (n_img, 3) array, optional
        Periodic image offsets; a trailing dimension of 0 disables PBC.
    field : bool
        Evaluate the field (negative gradient) instead of the potential.
    mem_limit : float
        Peak-transient byte budget (0.3.1 semantics). With ``batch_rule="pricing"`` it sets the
        batch (``(batch, w_src, w_eval, 3)`` priced at ~2x live multiplicity); with
        ``"eval_slots"`` (GPU) it only caps the rule's batch at an honest per-pair transient
        (:func:`p2p_batch`). Finite values batch the pair list via ``lax.scan``.
    remat : bool
        Wrap the per-batch P2P body in :func:`jax.checkpoint`, so reverse-mode
        recomputes each batch's pairwise tensor in the backward pass instead of
        stacking the residuals of every batch (the dominant memory cost of the
        near-field gradient).  Only effective together with a finite ``mem_limit``;
        forward behaviour and results are unchanged.
    mixed : bool
        Evaluate the pairwise kernel in float32 (half the memory traffic) while
        ACCUMULATING the per-box scatter-add in the inputs' (float64) dtype, so the
        near-cancelling sum stays accurate.  Complex charges (Bloch, ``img_phase``)
        are reduced as two real f32 channels (re, im) against the shared real weights;
        the per-pair phase multiplies the reduced result.
    centres, eval_centres : (nbox, 3) arrays, optional
        Per-box centres matching ``padded_pts`` (plus ``eval_centres`` if
        ``padded_eval_pts`` uses different boxes).  With ``mixed=True`` they take the
        coordinates BOX-LOCAL before the f32 cast: the f32 mantissa is spent on the
        local point-to-centre offsets and the exact (f64) centre separation is added
        back per pair, so the near-pair distances keep ~f32-machine relative accuracy
        instead of losing ``log2(domain/separation)`` bits to the global cast.
        Without ``centres`` the coordinates are cast globally.
    return_padded : bool
        Return the ``(nbox, w[, 3])`` box-slot accumulator without the ``rev_idcs``
        gather (used by the tier-local lean P2P, which scatters itself).
    k2, k2g : callables, optional
        Pluggable kernel weights (4pi-free; see ``laplace.Kernel``); ``None`` selects
        the Laplace fast path.
    batch_rule : {"pricing", "eval_slots"}, optional
        How the pairs per scan step are chosen (:func:`p2p_batch`): ``"pricing"`` (default,
        point-point passes) or ``"eval_slots"`` (element passes with wide source tiles).
    coincidence_floor : {"pair", "tile", False}, optional
        Guard for source/eval points that coincide only up to rounding (see the
        in-code note): ``"pair"`` (default) zeroes ``d2`` below ``(16 eps)^2`` of a
        per-batch geometric bound ``(max|x_eval| + max|x_src|)^2`` in tile-local
        coordinates (O(w) work, floor <= ~32 ULP of the tile scale); ``"tile"`` uses the
        exact tile maximum of ``d2`` (a second full pass over the tile: measured 1.6-1.7x
        on the element near field at 1e7); ``False`` keeps only the exact ``d2 > 0``
        guard -- correct whenever sources can never land on eval points (element
        quadrature sources vs mesh nodes), and the cheapest.

    Returns
    -------
    pot : (N,) or (N, 3) array
        Near-field potential (or field) per particle, in original particle order
        (padded box-slot layout when ``return_padded=True``).
    """
    if(dir_cnct.size == 0):   # nothing to compute
        return 0
    # Pairs per scan step: p2p_batch (batch_rule="pricing": the transient-priced batch from
    # mem_limit; "eval_slots": the GPU rule for wide-source/narrow-eval element passes).
    _w_src = src_width if src_fn is not None else padded_pts.shape[1]
    _eff_bytes = 8 if padded_eval_pts.itemsize >= 8 else 16
    # complex charges (Bloch) double the pairwise working set: two real channels under mixed,
    # a complex kernel product otherwise
    _chan = 2 if padded_chrgs is not None and jnp.iscomplexobj(padded_chrgs) else 1
    batch_size = p2p_batch(dir_cnct.shape[0], padded_eval_pts.shape[0] * padded_eval_pts.shape[1], _w_src,
                           padded_eval_pts.shape[1], mem_limit, field, _chan, _eff_bytes, batch_rule)
    # accumulator follows the input dtype (f64 stays f64 under mixed); complex charges and
    # Bloch phases promote it to the matching complex dtype
    acc_dtype = jnp.result_type(padded_eval_pts.dtype,
                                *([padded_chrgs.dtype] if _chan == 2 else []),
                                *([img_phase.dtype] if img_phase is not None else []))
    pot_glob = jnp.zeros(padded_eval_pts.shape[:2+field], acc_dtype)
    nbox = padded_pts.shape[0]
    local = mixed and centres is not None                   # box-local f32 (see docstring)
    # kernel_dtype: the pairwise-tensor dtype under mixed (default f32). bf16 works as a
    # drop-in (f32 exponent range); f16 additionally needs kernel_scale="auto", which
    # normalises the box-local coordinates to O(1) so d2 stays clear of f16's subnormals
    # (6e-5), and restores the scale on the result branch-free (potential x ksc, field x
    # ksc^2 -- the kernel is homogeneous). CHARGES ALWAYS STAY f32: q ~ Ms*h^2 carries
    # Ms-class magnitudes that no coordinate rescale touches; only geometry narrows. The
    # coincidence floor adapts through d2.dtype automatically (coarser in f16 by design).
    kdt = jnp.float32 if kernel_dtype is None else kernel_dtype
    if kernel_scale is not None and not local:
        raise NotImplementedError("kernel_scale requires the box-local mixed path")
    if src_fn is not None and kernel_scale == "auto":
        raise NotImplementedError("kernel_scale='auto' needs the stored source extent; pass "
                                  "an explicit scale with src_fn")
    if mixed:                                               # low-dtype kernel inputs; accumulate in pot_glob's dtype
        if local:
            ecen = centres if eval_centres is None else eval_centres
            _ke64 = padded_eval_pts - ecen[:, None, :]
            if src_fn is None:
                _kp64 = padded_pts - centres[:, None, :]
            if kernel_scale == "auto":
                ksc = 1.0 / jnp.maximum(jnp.maximum(jnp.max(jnp.abs(_kp64)),
                                                    jnp.max(jnp.abs(_ke64))), 1e-300)
            elif kernel_scale is not None:
                ksc = kernel_scale
            else:
                ksc = None
            if src_fn is None:
                kpts = (_kp64 if ksc is None else _kp64 * ksc).astype(kdt)
            keval = (_ke64 if ksc is None else _ke64 * ksc).astype(kdt)
        else:
            ksc = None
            if src_fn is None:
                kpts = padded_pts.astype(kdt)
            keval = padded_eval_pts.astype(kdt)
        if src_fn is None:                                  # complex charges keep their dtype: the
            kchg = padded_chrgs if jnp.iscomplexobj(padded_chrgs) else padded_chrgs.astype(jnp.float32)
        kimg = img_cnct.astype(jnp.float32)                 # re/im split happens in the batch body
    else:
        ksc = None
        keval, kimg = padded_eval_pts, img_cnct
        if src_fn is None:
            kpts, kchg = padded_pts, padded_chrgs
    # STRUCTURE-OF-ARRAYS sources: one (nbox, w_src) array per component, each gathered by its
    # single consumer. The (batch, w_src, 3) gather + x/y/z slices let XLA bundle gather and
    # slices into a multi-output fusion and materialise [batch, w_src] planes per step
    # (2026-10-09: 14-35 % slower in the fused regime on A100 / RTX PRO 6000).
    # Exception: wide-tile element passes (the P2P_WIDE_SRC branch of p2p_batch) keep the
    # (batch, w_src, 3) gather -- their small-batch kernel runs up to 2x slower with SoA (shell
    # R9 H100: 249 vs 500 ms per pass), and the wide rule was measured with this layout.
    aos = (batch_rule == "eval_slots" and _w_src > P2P_WIDE_SRC
           and jax.devices()[0].platform == "gpu")
    kpts_c = tuple(kpts[..., d] for d in range(3)) if src_fn is None and not aos else None

    def scan_body(buf, idcs):
        # COMPONENT-WISE pairwise differences (manual fission): the (batch, w_s, w_e, 3)
        # distsvec tensor is never formed. In field mode it would have TWO consumers (the
        # norm chain and the vector product) and XLA stops fusing it at large per-tier
        # pair counts, materialising the full tensor; with one fused reduction per
        # component every big intermediate has a single consumer.
        # eval-side gathers through _take_sorted (idcs[:,0] is sorted, as the output scatter
        # promises): their cotangent scatter-adds were 96 % of a value_and_grad when the eval tree
        # is much coarser than the source tree (~640 pairs per eval box; interior quadrature p3
        # th0.4 N128, 1e5 tets, A100: 1.98 s -> 92 ms vs a 18.5 ms forward)
        kt = _take_sorted(keval, idcs[:,0])                 # (batch, w_eval, 3) SMALL
        partner = idcs[:,1]
        if local:                                           # exact centre separation, f64 -> f32
            cdiff = _take_sorted(ecen, idcs[:,0])
            if(img_cnct.shape[1] != 0): # PBC enabled
                cdiff = cdiff - img_cnct[partner//nbox]     # image offset (folded in f64)
                partner = partner % nbox    # local image position
            _cd = cdiff - centres[partner]
            kt = kt + (_cd if ksc is None else _cd * ksc).astype(kdt)[:,None,:]
        elif(img_cnct.shape[1] != 0): # PBC enabled
            kt = kt - kimg[partner//nbox,None,:]            # image offset
            partner = partner % nbox    # local image position
        if src_fn is not None:
            # LAZY sources (store_quad=False): points/charges formed in-batch from the
            # mesh — nothing (nbox, w, 3)-sized is ever stored. Box-local shift + cast
            # happen here since the precompute above was skipped.
            _kp_raw, _chg_raw = src_fn(partner)             # (batch, w_src, 3), (batch, w_src)
            if local:
                _kp_raw = _kp_raw - centres[partner][:, None, :]
            if ksc is not None:
                _kp_raw = _kp_raw * ksc
            kp = _kp_raw.astype(kdt) if mixed else _kp_raw
            kp_c = [kp[..., d] for d in range(3)]
        elif aos:
            kp = kpts[partner]                              # (batch, w_src, 3)
            kp_c = [kp[..., d] for d in range(3)]
        else:
            kp_c = [x[partner] for x in kpts_c]            # 3 x (batch, w_src) SMALL
        dd = lambda d: kt[:,None,:,d] - kp_c[d][:,:,None]    # (batch, w_src, w_eval) per comp
        d2 = dd(0)**2 + dd(1)**2 + dd(2)**2                      # gradient-safe inverse distance: padded /
        # COINCIDENT source/eval points: any route that reconstructs a separation by
        # arithmetic can leave d2 ~ eps^2 instead of exactly 0 -> 1/sqrt blowup. Two known
        # routes: box-local f32 folding across different trees (three roundings), and the
        # PBC image shift x+L, where a wrap-partner charge (duplicated on both cell faces)
        # lands ONE ULP off its twin instead of on it. The exact `d2 > 0` test below misses
        # both. Relative floor at the tile scale masks them; genuinely distinct points
        # closer than ~16 ULP of the tile scale are beyond the dtype anyway. UNCONDITIONAL: the
        # PBC route has nothing to do with `local`.
        # Floor eps: NEVER coarser than f32's. The floor masks folding-reconstruction
        # rounding; at bf16/f16 kernel dtypes its (16*eps)^2 threshold (1.6e-2 / 2.4e-4 of
        # the tile scale) would zero genuinely close pairs — the LARGEST contributions —
        # measured as rel err 1.3 (bf16) on a film near field. True coincidences cast from
        # the same high-precision value land bitwise equal -> d2 == 0 exactly, which the
        # d2 > 0 guard below already handles at any dtype.
        # Cost: the tile-max form is a second pass over the (w_src, w_eval) tile and was
        # 40% of the ELEMENT near field at 1e7 (47.8 -> 28.9 ms without it, A100 f32);
        # the "pair" bound replaces it with an O(w) per-batch radius sum at no measurable
        # cost. Element callers pass False: quadrature sources sit >= 0.096 h from every
        # vertex for all simplex rules used (deg 1-7), so no route can make them coincide.
        if coincidence_floor:
            _feps = min(jnp.finfo(d2.dtype).eps, jnp.finfo(jnp.float32).eps)
            if coincidence_floor == "tile":
                _scale = jnp.max(d2, axis=(1, 2), keepdims=True)
            elif coincidence_floor == "pair":
                _rs = jnp.sqrt(jnp.max(kp_c[0]**2 + kp_c[1]**2 + kp_c[2]**2, axis=1))
                _scale = ((jnp.sqrt(jnp.max(jnp.sum(kt**2, axis=-1), axis=1)) + _rs)**2)[:, None, None]
            else:
                raise ValueError(f"coincidence_floor must be 'pair', 'tile' or False, got {coincidence_floor!r}")
            d2 = jnp.where(d2 > (16 * _feps)**2 * _scale, d2, 0.0)
        chrgs = _chg_raw if src_fn is not None else kchg.at[partner].get(mode="fill", fill_value=0.0)
        cplx = jnp.iscomplexobj(chrgs)
        if mixed and not cplx:
            chrgs = chrgs.astype(jnp.float32)
        if k2 is not None:                                       # PLUGGABLE kernel (4pi-free
            kw = k2g(d2) if field else k2(d2)                    # weights; see laplace.Kernel)
        else:                                                    # Laplace fast path
            distsnorm = jnp.where(d2>0, 1/jnp.sqrt(jnp.where(d2>0,d2,1.0)), 0.0)  # self pairs (d=0) -> 0, no NaN grad
            kw = distsnorm**3 if field else distsnorm
        if cplx and mixed:
            # Bloch under the f32 kernel path: the pairwise weights are REAL, so reduce the
            # re/im parts as two f32 charge channels sharing one weight tensor (a single
            # consumer keeps the fission intact) and recombine per pair. The complex
            # charges are never cast to f32 -- that would silently drop the imaginary part.
            ch2 = jnp.stack([chrgs.real, chrgs.imag], axis=-1).astype(jnp.float32)  # (batch, w_src, 2)
            if(field):
                w2 = ch2[:, :, None, :] * kw[..., None]                          # (batch, w_src, w_eval, 2)
                p2 = jnp.stack([(w2 * dd(d)[..., None]).sum(axis=1) for d in range(3)], axis=-2)
            else:
                p2 = (kw[..., None] * ch2[:, :, None, :]).sum(axis=1)
            pot = jax.lax.complex(p2[..., 0], p2[..., 1])
        elif(field):
            w3 = chrgs[...,None] * kw
            pot = jnp.stack([(w3 * dd(d)).sum(axis=1) for d in range(3)], axis=-1)
        else:
            pot = (kw*chrgs[...,None]).sum(axis=(1))
        if img_phase is not None:
            # Bloch: e^{ik.R} is constant over a box pair, so it multiplies the REDUCED
            # (batch, w_eval[, 3]) result rather than the (batch, w_src) charges -- same
            # math, and it keeps the reduction real under mixed
            ph = img_phase.at[idcs[:, 1] // nbox].get(mode="fill", fill_value=0.0)
            pot = pot * ph.reshape((-1,) + (1,) * (pot.ndim - 1))
        return buf.at[idcs[:,0]].add(pot.astype(buf.dtype),indices_are_sorted=True), None

    body = jax.checkpoint(scan_body) if remat else scan_body
    if(batch_size < dir_cnct.shape[0]):
        num_pad = batch_size - dir_cnct.shape[0]%batch_size
        fill_value = jnp.asarray(jnp.iinfo(dir_cnct.dtype).max, dir_cnct.dtype)   # match int32
        scandir = jax.lax.pad(dir_cnct, fill_value, ((0,num_pad,0), (0,0,0))).reshape((-1,batch_size,2))
        pot_glob, _ = jax.lax.scan(body,pot_glob,scandir)
    else:
        pot_glob, _ = body(pot_glob, dir_cnct)

    if ksc is not None:                  # undo the coordinate normalisation (homogeneous kernel)
        pot_glob = pot_glob * (ksc**2 if field else ksc)
    if return_padded:                    # (nbox, N_max[, 3]) box-slot accumulator, NO rev
        return pot_glob/(4*jnp.pi)       # gather (tier-local lean P2P scatters it itself)
    if(field):
        return pot_glob.reshape((-1,3))[rev_idcs]/(4*jnp.pi)
    else:
        return pot_glob.flatten()[rev_idcs]/(4*jnp.pi)


def eval_direct_binned(padded_pts, padded_chrgs, padded_eval_pts, rev_idcs, dir_cnct_flat,
                       bin_offsets, bin_widths, img_cnct = jnp.array([[]]), field = False,
                       mem_limit = jnp.inf, remat = False, mixed = False, centres = None,
                       kernel_dtype = None, kernel_scale = None,
                       eval_centres = None, k2 = None, k2g = None):
    r"""
    Occupancy-binned near field (P2P).

    Bit-exact alternative to :func:`eval_direct` that avoids the padding waste of
    computing a full ``N_max x N_max`` tile for every box-pair when leaf occupancy is
    uneven.  The box-pairs are pre-bucketed (by :func:`near_pairs_binned`) into
    ``K = len(bin_widths)`` width tiers stored as contiguous segments of
    ``dir_cnct_flat``; tier ``g`` is processed at the smaller tile width
    ``bin_widths[g]`` (a static multiple of ``N_max/K``), so its cost is
    ``n_pairs_g * w_g^2`` instead of ``n_pairs_g * N_max^2``.

    Parameters
    ----------
    dir_cnct_flat : (n_pairs, 2) array
        Near box-pairs reordered into ``K`` contiguous tier segments
        (:func:`near_pairs_binned`).
    bin_offsets : tuple of int, length K+1
        Cumulative segment offsets.  STATIC (Python tuple).
    bin_widths : tuple of int, length K
        Per-segment tile widths.  STATIC (Python tuple).
    padded_pts, padded_chrgs, padded_eval_pts, rev_idcs, img_cnct, field, mem_limit, \
    remat, mixed, centres, eval_centres, k2, k2g
        As in :func:`eval_direct` (``centres`` makes the f32 cast box-local).

    Returns
    -------
    pot : (N,) or (N, 3) array
        Near-field potential (or field) per particle, in original particle order.

    Notes
    -----
    The segment shapes are baked into the trace, so as long as the setup rounds each
    segment capacity to a stable block the whole pipeline is recompile-free across
    geometries that change over time (the points are padded to a static ``N_max``,
    never the realised ``maxleaf``).

    An in-kernel "lean" mode gathering box tiles from a leaf-major sorted layout via
    computed 2-D indices does not work: XLA only fuses the pairwise tensor into the
    reduction for ROW gathers from a padded 3-D array — the computed-index form
    materialises the (batch, w, w, 3) tensor (much slower and OOM-prone).  The lean
    path instead pads TIER-LOCALLY and calls :func:`eval_direct` per tier: see
    :func:`eval_direct_tiered_lean`.
    """
    if dir_cnct_flat.size == 0:
        return 0
    nbox = padded_pts.shape[0]
    pbc = img_cnct.shape[1] != 0
    local = mixed and centres is not None                   # box-local f32 (see eval_direct)
    if mixed:
        if local:
            ecen = centres if eval_centres is None else eval_centres
            kpts = (padded_pts - centres[:, None, :]).astype(jnp.float32)
            keval = (padded_eval_pts - ecen[:, None, :]).astype(jnp.float32)
        else:
            kpts, keval = padded_pts.astype(jnp.float32), padded_eval_pts.astype(jnp.float32)
        kchg = padded_chrgs.astype(jnp.float32)
        kimg = img_cnct.astype(jnp.float32)
    else:
        kpts, kchg, keval, kimg = padded_pts, padded_chrgs, padded_eval_pts, img_cnct
    pot_glob = jnp.zeros(padded_eval_pts.shape[:2+field], padded_eval_pts.dtype)           # accumulator follows the input dtype (f64 stays f64 under mixed)
    itemsize = padded_pts.itemsize

    def make_body(w):                                              # per-tier scan body at static tile width w
        cols = jnp.arange(w)                                       # sub-column scatter: write only the w real columns
        def scan_body(buf, idcs):                                  # (cheap for the sparse tiers; row-scatter+pad is worse)
            partner = idcs[:,1]
            distsvec = keval[idcs[:,0]][:, None, :w, :]            # (batch, 1, w_eval, 3)
            if local:                                              # exact centre separation, f64 -> f32
                cdiff = ecen[idcs[:,0]]
                if pbc:
                    cdiff = cdiff - img_cnct[partner//nbox]        # image offset (folded in f64)
                    partner = partner % nbox
                distsvec = distsvec + (cdiff - centres[partner]).astype(jnp.float32)[:, None, None, :]
            elif pbc:
                distsvec = distsvec - kimg[partner//nbox, None, None, :]
                partner = partner % nbox
            distsvec = distsvec - kpts[partner][:, :w, None, :]   # (batch, w_src, w_eval, 3)
            chrgs = kchg.at[partner].get(mode="fill", fill_value=0.0)[:, :w]   # (batch, w_src)
            d2 = (distsvec**2).sum(axis=-1)                       # gradient-safe inverse distance (self pairs -> 0)
            # coincidence floor -- UNCONDITIONAL (see eval_direct): covers both the
            # box-local f32 folding and the PBC wrap-partner image shift.
            d2 = jnp.where(d2 > (16 * jnp.finfo(d2.dtype).eps)**2
                       * jnp.max(d2, axis=(1, 2), keepdims=True), d2, 0.0)
            if k2 is not None:                                    # PLUGGABLE kernel: 4pi-free
                distsnorm = k2(d2)                                # weights (see laplace.Kernel);
                w_field = (chrgs[...,None] * k2g(d2)) if field else None   # Laplace path below unchanged
            else:
                distsnorm = jnp.where(d2>0, 1/jnp.sqrt(jnp.where(d2>0,d2,1.0)), 0.0)
                w_field = (chrgs[...,None] * distsnorm**3) if field else None
            if field:
                pot = (w_field[...,None] * distsvec).sum(axis=1)   # (batch, w_eval, 3)
                buf = buf.at[idcs[:,0][:,None], cols[None,:], :].add(pot.astype(buf.dtype), mode="drop")
            else:
                pot = (distsnorm*chrgs[...,None]).sum(axis=1)     # (batch, w_eval)
                buf = buf.at[idcs[:,0][:,None], cols[None,:]].add(pot.astype(buf.dtype), mode="drop")
            return buf, None
        return jax.checkpoint(scan_body) if remat else scan_body

    fill_value = jnp.asarray(jnp.iinfo(dir_cnct_flat.dtype).max, dir_cnct_flat.dtype)
    for g, w in enumerate(bin_widths):                            # K static tiers (Python loop, K static)
        lo, hi = bin_offsets[g], bin_offsets[g+1]
        if hi <= lo:
            continue
        seg = dir_cnct_flat[lo:hi]                                # static-length segment
        n_g = hi - lo
        mem = n_g * w * itemsize        # FUSED kernel: the pairwise tile never materialises (XLA), so
        batch_size = int((mem_limit/mem)*n_g) + 1 if mem_limit < mem else n_g   # rows ARE the honest transient
        body = make_body(w)
        if batch_size < n_g:
            num_pad = batch_size - n_g % batch_size
            scandir = jax.lax.pad(seg, fill_value, ((0,num_pad,0),(0,0,0))).reshape((-1,batch_size,2))
            pot_glob, _ = jax.lax.scan(body, pot_glob, scandir)
        else:
            pot_glob, _ = body(pot_glob, seg)

    if field:
        return pot_glob.reshape((-1,3))[rev_idcs]/(4*jnp.pi)
    else:
        return pot_glob.flatten()[rev_idcs]/(4*jnp.pi)



def eval_direct_tiered_lean(pos_s, q_s, starts, cnts, centres, tiers, img_cnct = jnp.array([[]]),
                            field=False, mem_limit=jnp.inf, mixed=False, remat=False,
                            k2=None, k2g=None):
    r"""LEAN occupancy-binned near field via TIER-LOCAL padding (memory diet).

    Per tier, the boxes' slices are gathered into a compact ``(nb_g, w_g, 3)`` padded
    block and :func:`eval_direct` runs UNCHANGED on it — XLA fuses the pairwise tensor
    for row gathers from a padded 3-D array, but NOT for computed-index gathers from the
    flat layout, so the tier block is materialised deliberately: it is the tier's working
    set, bounded by the fattest tier instead of the whole padded tree.

    Parameters
    ----------
    pos_s : (N + w, 3) array
        Points in LEAF-MAJOR SORTED order (each box a contiguous slice
        ``[starts[b], starts[b]+cnts[b])``, padded by one tile width).
    q_s : (N + w,) array
        Charges in the same sorted order.
    starts, cnts : (nbox,) int32 arrays
        Slice start/count per box.
    centres : (nbox, 3) array or None
        Box centres (box-local f32 folding under ``mixed``).
    tiers : tuple
        Per-tier tuples ``(dir_g, boxes_g, w_g[, has_img])`` from
        :func:`near_pairs_tiered_lean`: pairs remapped to TIER-LOCAL box numbering plus
        the tier's box list.
    img_cnct, field, mem_limit, mixed, remat, k2, k2g
        As in :func:`eval_direct`.

    Returns
    -------
    pot : (N + w,) or (N + w, 3) array
        Potential (or field) in SORTED particle order (caller un-sorts by rank).

    Notes
    -----
    PBC: pass ``img_cnct`` AND build ``tiers`` with the matching ``n_img`` — the tier
    pairs then carry a TIER-LOCAL image encoding (``src = local_box + nb_g*image``)
    which :func:`eval_direct` decodes against the tier block exactly like the global
    path.  Tiers built without ``n_img`` while ``img_cnct`` is non-empty are a contract
    violation and raise.
    """
    pbc = img_cnct.shape[1] != 0
    N_pad = pos_s.shape[0]                                   # includes the one-tile pad
    out = jnp.zeros((N_pad, 3) if field else (N_pad,), pos_s.dtype)
    for tier in tiers:
        if pbc and not (len(tier) >= 4 and tier[3]):
            raise NotImplementedError(
                "eval_direct_tiered_lean: img_cnct given but the tiers carry no image encoding — "
                "build them with near_pairs_tiered_lean(..., n_img=img_cnct.shape[0]). Silently "
                "dropping periodic images is not an option (see ARCHITECTURE §7.4).")
        dir_g, boxes_g, w_g = tier[:3]
        st_g = starts[boxes_g]                               # (nb_g,)
        ct_g = cnts[boxes_g]
        iw = jnp.arange(w_g, dtype=jnp.int32)
        sl = st_g[:, None] + iw[None, :]                     # (nb_g, w_g) in-bounds (padded)
        ok = iw[None, :] < ct_g[:, None]
        ppts_g = pos_s[sl]                                   # compact tier-local padded block
        pchg_g = jnp.where(ok, q_s[sl], 0.0)
        cen_g = centres[boxes_g] if centres is not None else None
        pot_g = eval_direct(ppts_g, pchg_g, ppts_g, None, dir_g,
                            img_cnct=img_cnct if pbc else jnp.array([[]]),
                            field=field, mem_limit=mem_limit, mixed=mixed, remat=remat,
                            centres=cen_g, return_padded=True, k2=k2, k2g=k2g)   # (nb_g, w_g[, 3])
        idx = jnp.where(ok, sl, N_pad)                       # slot -> sorted particle id
        out = out.at[idx].add(pot_g.astype(out.dtype), mode="drop")
    return out


def near_pairs_tiered_lean(dc, counts, maxleaf, K=16, block_budget=4e9, n_img=1):
    r"""Per-tier pair structures for :func:`eval_direct_tiered_lean`.

    Buckets the compact near pairs by occupancy (same tiering rule as
    :func:`near_pairs_binned`), then remaps each tier's pairs to TIER-LOCAL box
    numbering.  Host/NumPy, once per setup.

    Parameters
    ----------
    dc : (n_pairs, 2) array
        Compact near box-pairs (source column may be image-encoded).
    counts : (nbox,) array
        Per-box particle counts.
    maxleaf : int
        Widest leaf occupancy (widest tier width).
    K : int
        Number of occupancy tiers.
    block_budget : float
        Byte budget for a tier's gathered block.
    n_img : int
        Number of periodic images encoded in the source column (1 = open boundaries).

    Returns
    -------
    tiers : tuple
        Tuples ``(dir_g (n_g, 2) int32, boxes_g (nb_g,) int32, w_g int, has_img bool)``.

    Notes
    -----
    A tier whose gathered block ``nb_g * w_g * 16 B`` would exceed ``block_budget`` is
    SPLIT into sub-tiers along the (Morton-sorted) target-box order — sources shared
    across sub-tiers appear in several box lists (small duplication, Morton locality),
    but the eval-time transient is bounded by the budget instead of the fattest tier
    (which alone can dominate device memory at extreme N).

    PBC: with ``n_img > 1`` the pair SOURCE column carries the usual global image
    encoding ``src = box + nbox*image`` (nbox = len(counts)); it is decoded here,
    bucketed by the REAL source box occupancy, and re-encoded TIER-LOCALLY
    (``local_box + nb_g*image``) so :func:`eval_direct`'s ``partner // nbox`` decode
    works unchanged on the tier block.
    """
    dc = np.asarray(dc); cnt = np.asarray(counts)
    dc = dc[(dc[:, 0] >= 0) & (dc[:, 1] >= 0)]
    nbox = cnt.shape[0]
    src_img, src_box = (dc[:, 1] // nbox, dc[:, 1] % nbox) if n_img > 1 else \
                       (np.zeros(len(dc), dc.dtype), dc[:, 1])
    widths = [int(np.ceil(maxleaf * (g + 1) / K)) for g in range(K)]
    widths[-1] = int(maxleaf)
    pair_occ = np.maximum(cnt[dc[:, 0]], cnt[src_box])
    bin_id = np.minimum(np.searchsorted(widths, pair_occ, side="left"), K - 1)
    has_img = n_img > 1
    tiers = []
    for g in range(K):
        gsel = bin_id == g
        if not gsel.any():
            continue
        sel = np.stack([dc[gsel, 0], src_box[gsel]], 1)       # image-free box pairs
        sel_img = src_img[gsel]
        w_g = widths[g]
        nb_cap = max(int(block_budget // (w_g * 16)), 1024)   # boxes per gathered block
        n_sub = max(int(np.ceil(len(np.unique(sel)) / nb_cap)), 1)
        if n_sub > 1:
            order = np.argsort(sel[:, 0], kind="stable")
            parts = np.array_split(order, n_sub)
        else:
            parts = [np.arange(len(sel))]
        for pidx in parts:
            if not len(pidx):
                continue
            part, pimg = sel[pidx], sel_img[pidx]
            boxes_g, inv = np.unique(part, return_inverse=True)
            loc = inv.reshape(part.shape).astype(np.int64)
            loc[:, 1] += pimg.astype(np.int64) * len(boxes_g)  # tier-local image re-encoding
            tiers.append((jnp.asarray(loc.astype(np.int32)),
                          jnp.asarray(boxes_g.astype(np.int32)), w_g, has_img))
    return tuple(tiers)


def _build_padded(positions, charges, pil):
    max_boxes, N_max = pil.shape; N = positions.shape[0]
    valid = pil >= 0
    safe = jnp.where(valid, pil, 0)
    ppts = positions[safe]                                             # (max_boxes, N_max, 3)
    pchg = jnp.where(valid, charges[safe], 0.0)
    flat_pos = (jnp.arange(max_boxes, dtype=jnp.int32)[:, None] * N_max
                + jnp.arange(N_max, dtype=jnp.int32)[None, :])         # (max_boxes, N_max)
    rev = jnp.zeros(N, jnp.int32).at[jnp.where(valid, pil, N)].set(flat_pos, mode="drop")
    return ppts, pchg, rev


def near_pairs(U_list, is_leaf, lvls):
    r"""U-list (leaf-leaf neighbours) + leaf self-pairs -> ``dir_cnct`` for the P2P near field.

    Eager (dynamic shape from ``jnp.where``); build once in setup and pass to the
    evaluator to keep it jittable.

    Parameters
    ----------
    U_list : (n_u, 2) array
        Leaf-leaf neighbour pairs.
    is_leaf : (nbox,) bool array
        Leaf flags.
    lvls : (nbox,) array
        Box levels.

    Returns
    -------
    dir_cnct : (n_u + n_leaf, 2) array
        Near pair list including self-pairs.
    """
    active_leaf = jnp.where(is_leaf & (lvls > 0))[0]
    self_pairs = jnp.stack([active_leaf, active_leaf], axis=1).astype(U_list.dtype)
    return jnp.concatenate([U_list, self_pairs], axis=0)


def near_pairs_binned(dir_cnct, counts, pad_width, K=4, pair_block=4096, caps=None):
    r"""Bucket the near-field box-pairs into ``K`` occupancy/width tiers.

    For :func:`eval_direct_binned`.  Tier ``g`` (0-based) has width
    ``ceil(pad_width*(g+1)/K)`` (the widest is exactly ``pad_width``); a pair is placed
    in the SMALLEST tier whose width covers its LARGER box (``max(occ_t, occ_s)``) so
    the tile always covers both -> numerically equal to :func:`eval_direct`.  Eager
    (NumPy), once per setup.

    Parameters
    ----------
    dir_cnct : (n_pairs, 2) array
        Near pair list (from :func:`near_pairs`).
    counts : (nbox,) array
        Per-box particle counts.
    pad_width : int
        Padded-point width (= the ``pil`` column count fed to :func:`_build_padded`):
        the tree ``N_max`` for the plain uniform grid, or the realised ``maxleaf``
        (which may exceed ``N_max``) for the element strayfield path.  The widest
        tier == ``pad_width`` guarantees coverage.
    K : int
        Number of tiers.
    pair_block : int
        Segment-capacity rounding block when ``caps`` is not given.
    caps : tuple of int, length K, optional
        Frozen per-segment capacities (see Notes).

    Returns
    -------
    dir_cnct_flat : (sum caps, 2) array
        The pairs reordered into ``K`` contiguous segments (sentinel-padded).
    bin_offsets : tuple of int, length K+1
        Cumulative segment offsets (STATIC tuple).
    bin_widths : tuple of int, length K
        Per-segment tile widths (STATIC tuple).

    Notes
    -----
    Recompile-freeness for geometries that change over time: the tier WIDTHS are always
    static (a function of ``pad_width`` and ``K``), so the kernel structure never
    changes for a fixed topology.  The only varying quantity is the per-segment
    capacity.  Pass ``caps`` (a fixed ``len-K`` tuple, e.g. captured from the first
    setup with head-room) to FREEZE the segment shapes across geometries: pairs that
    overflow their natural tier SPILL UP into the next WIDER tier (always correct -- a
    wider tile still covers both boxes; the widest tier ``pad_width`` must hold any
    final overflow, else a ``ValueError``).  If ``caps is None`` each segment is
    instead rounded up to a ``pair_block`` multiple (tight, but the shapes can drift
    across block boundaries as the geometry evolves)."""
    dc = np.asarray(dir_cnct); cnt = np.asarray(counts)
    dc = dc[(dc[:, 0] >= 0) & (dc[:, 1] >= 0)]
    widths = [int(np.ceil(pad_width * (g + 1) / K)) for g in range(K)]   # increasing, widths[-1] == pad_width
    widths[-1] = int(pad_width)
    pair_occ = np.maximum(cnt[dc[:, 0]], cnt[dc[:, 1] % cnt.shape[0]])   # larger box decides the tile width (src may be image-encoded)
    bin_id = np.minimum(np.searchsorted(widths, pair_occ, side="left"), K - 1)   # 0..K-1: smallest tier covering occ
    sentinel = np.iinfo(dc.dtype).max
    segs, offs = [], [0]
    if caps is None:                                             # tight, block-rounded (may drift)
        for g in range(K):
            sel = dc[bin_id == g]
            cap = max(((len(sel) + pair_block - 1) // pair_block) * pair_block, pair_block)
            padded = np.full((cap, 2), sentinel, dc.dtype); padded[:len(sel)] = sel
            segs.append(padded); offs.append(offs[-1] + cap)
    else:                                                        # FIXED caps + spill-to-wider-tier (recompile-free)
        assert len(caps) == K, "caps must have length K"
        carry = np.empty((0, 2), dc.dtype)                       # overflow flows up into wider tiers
        for g in range(K):
            pool = np.concatenate([carry, dc[bin_id == g]], 0)
            cap = int(caps[g]); take = pool[:cap]; carry = pool[cap:]
            if g == K - 1 and len(carry):
                raise ValueError(f"near_pairs_binned: widest tier overflow by {len(carry)} pairs; raise caps[-1]")
            padded = np.full((cap, 2), sentinel, dc.dtype); padded[:len(take)] = take
            segs.append(padded); offs.append(offs[-1] + cap)
    return jnp.asarray(np.concatenate(segs, 0)), tuple(offs), tuple(widths)


def predicted_p2p_speedup(dir_cnct, counts, pad_width, K=4):
    r"""Eager predictor of the binned-P2P tile-work speedup vs the flat near field.

    The flat path computes a ``pad_width x pad_width`` tile per box-pair (cost
    ``n_pairs * pad_width^2``), the binned path a ``w_tier(pair)^2`` tile, so this
    returns

    .. math::

        n_{pairs} \, w_{pad}^2 \Big/ \sum_{pairs} w_{tier}(pair)^2

    with the SAME tier widths :func:`near_pairs_binned` uses.  NumPy, setup-time.

    Parameters
    ----------
    dir_cnct : (n_pairs, 2) array
        Near pair list (from :func:`near_pairs`); the source column may be
        image-encoded under PBC.
    counts : (nbox,) array
        Per-box particle counts.
    pad_width : int
        Padded-point width (the flat path's tile width; see
        :func:`near_pairs_binned`).
    K : int
        Number of tiers (default 4); must match the :func:`near_pairs_binned`
        call being gated.

    Returns
    -------
    speedup : float
        Predicted flat/binned tile-work ratio (>= 1; 1.0 for an empty pair
        list).

    Notes
    -----
    Tile work is the right gate quantity (not ``maxleaf/mean``): a surface/clustered
    cloud is bimodal -- moderate ``maxleaf/mean`` but most pairs fall in small tiers,
    so the tile work drops a lot -- whereas a nearly-uniform DENSE cloud barely shrinks
    and the per-tier overhead makes binning a wash there.  Compare against the setup's
    ``bin_gate``."""
    dc = np.asarray(dir_cnct); cnt = np.asarray(counts)
    dc = dc[(dc[:, 0] >= 0) & (dc[:, 1] >= 0)]
    if dc.shape[0] == 0:
        return 1.0
    widths = np.array([int(np.ceil(pad_width * (g + 1) / K)) for g in range(K)]); widths[-1] = int(pad_width)
    occ = np.maximum(cnt[dc[:, 0]], cnt[dc[:, 1] % cnt.shape[0]])   # src may be image-encoded
    w = widths[np.minimum(np.searchsorted(widths, occ, side="left"), K - 1)]
    binned = float((w.astype(np.float64) ** 2).sum())
    return float(dc.shape[0] * pad_width ** 2 / max(binned, 1.0))


M2L_MAX_BATCHES = 32     # (CPU) scan steps above which the per-batch carry scatter dominates
M2L_WS_BYTES = 64 * 2**20  # (GPU) M2L working set per scan step: measured optimum 2026-10-09 on
                           # H100 (p=3/6/8, three meshes, 1e7) and A100/RTX PRO 6000 (32-67 MiB)
K_M2L = 2                # measured true M2L transient / priced batch budget (XLA temp bytes)


def _tier_bounds(values, n_tiers):
    r"""Exact tiering of non-negative integer ``values`` into at most ``n_tiers`` groups.

    Minimises :math:`\sum_t \max(\text{group}_t)\cdot|\text{group}_t|` (padded slots when
    every item of a group is padded to the group maximum) by dynamic programming over the
    distinct values. Returns the ascending group maxima; the last equals ``values.max()``.
    """
    v, c = np.unique(np.asarray(values), return_counts=True)
    m, T = len(v), min(int(n_tiers), len(v))
    cs = np.concatenate([[0], np.cumsum(c)])
    best = np.full((T + 1, m + 1), np.inf)
    best[0, 0] = 0.0
    arg = np.zeros((T + 1, m + 1), np.int64)
    for k in range(1, T + 1):
        for j in range(1, m + 1):
            cand = best[k - 1, :j] + v[j - 1] * (cs[j] - cs[:j])
            arg[k, j] = int(np.argmin(cand))
            best[k, j] = cand[arg[k, j]]
    cuts, j = [], m
    for k in range(T, 0, -1):
        cuts.append(j)
        j = arg[k, j]
    return [int(v[j - 1]) for j in sorted(cuts)]


P2P_EVAL_DIVISOR = 8        # pairs per scan step >= padded eval slots / 8
P2P_L2_DIVISOR = 1024       # ... and >= device L2 bytes / 1 KiB
P2P_ALIGN = 1024            # step sizes are multiples of this (odd sizes measured up to 1.4x slower)
P2P_WIDE_SRC = 4096         # source points per pair above which the large-batch kernel loses (wide tiles)
P2P_WIDE_BATCH = 1024       # wide-tile batch ...
P2P_WIDE_L2_DIVISOR = 6     # ... capped so that batch * w_src <= L2 bytes / 6
P2P_WIDE_ALIGN = 256


@lru_cache(maxsize=None)
def device_l2_bytes(device_id = 0):
    r"""L2 cache size of CUDA device ``device_id`` in bytes; 40 MiB if it cannot be queried
    (non-CUDA backend, no driver library, or a failed driver call)."""
    try:
        v = int(get_gpu_l2_cache_size(device_id))
        if v > 0:
            return v
    except Exception:
        pass
    return 40 * 2**20


def p2p_batch(n_pairs, n_eval_slots, w_src, w_eval, mem_limit = jnp.inf, field = False, chan = 1, eff_bytes = 8,
              rule = "pricing"):
    r"""Near pairs per :func:`eval_direct` scan step.

    ``rule="pricing"`` (default; CPU always): the transient-priced batch
    ``mem_limit / (w_src * w_eval * 6 * eff_bytes)`` -- with :func:`auto_mem_limit`'s budget
    near-optimal for point-point passes (lumped, interior quadrature, flex: H100 2026-10-09
    sweeps 1.00-1.2x of the best batch), whose optimum stops at ~2 outputs per eval slot per
    step (square 64-wide tiles degrade 1.5-3x above ~100k pairs).

    ``rule="eval_slots"`` (GPU, element passes: wide source tiles, ~32-wide eval tiles):
    ``max(L2 / 1 KiB, n_eval_slots / 8)`` pairs, split into EQUAL steps (no padded
    tail) and rounded up to a multiple of 1024, then capped by ``mem_limit`` at the honest
    per-pair transient (gathered source/eval coordinates + charges and the per-pair output
    row; the pairwise tile is fused, never materialised). Measured 2026-10-09 against
    per-pass batch sweeps on H100/A100/RTX PRO 6000/MIG (element passes 1e5-1e7, R0/R3/R5):
    1.02-1.07x of the best batch on average, 1.23x worst, vs 1.41x / 3.2x for the previous
    byte-priced batch. Two mechanisms set it: below a device- and shape-dependent batch XLA
    stops fusing the source gather into the reduce/scatter kernel and materialises
    ``[batch, w_src]`` arrays every step (the threshold scales with L2: ~16-24k pairs on a
    MIG 1g slice, ~24-41k on H100, ~65-98k on RTX PRO 6000), and every step pays a pass
    over the eval accumulator (~n_eval_slots), which the eval term amortises.

    Wide source tiles (``w_src > 4096``, tet near degree >= 9) take ``min(1024, L2 / (6 w_src))``
    pairs (multiples of 256): there the large-batch fused kernel is 2.2-3.8x slower than the
    small-batch one (H100/A100/RTX PRO 6000, R9 near degree (10, 11)); it takes over at
    ``batch * w_src ~ L2 / 3.3``.

    The element-pass rule is NOT general: on point-point passes the ``n_eval_slots / 8`` term
    lands 2x past the optimum (lumped H100 1e7: P2P 2.4 -> 5.2 ms).

    Parameters
    ----------
    n_pairs : int
        Near pairs of the pass.
    n_eval_slots : int
        Padded eval points (eval boxes x eval width).
    w_src, w_eval : int
        Source / eval tile widths.
    mem_limit : float, optional
        Byte cap.
    field : bool, optional
        Vector output (3 accumulator channels).
    chan : int, optional
        Charge channels (2 for complex charges).
    eff_bytes : int, optional
        Pricing bytes per pairwise element.
    rule : {"pricing", "eval_slots"}, optional
        Batch rule (see above).

    Returns
    -------
    int
        Pairs per scan step (``n_pairs`` for a single step).
    """
    if n_pairs <= 0:
        return 0
    if rule == "pricing" or jax.devices()[0].platform != "gpu":
        mem = n_pairs * w_src * w_eval * 6 * eff_bytes * chan
        return int((mem_limit / mem) * n_pairs) + 1 if mem_limit < mem else n_pairs
    return _gpu_p2p_batch(n_pairs, n_eval_slots, w_src, w_eval, mem_limit, field, chan, device_l2_bytes())


def _gpu_p2p_batch(n_pairs, n_eval_slots, w_src, w_eval, mem_limit, field, chan, l2_bytes):
    r"""The GPU branch of :func:`p2p_batch` (pure; ``l2_bytes`` given)."""
    if w_src > P2P_WIDE_SRC:
        align = P2P_WIDE_ALIGN
        target = min(P2P_WIDE_BATCH, int(l2_bytes) // (P2P_WIDE_L2_DIVISOR * int(w_src)) // align * align)
    else:
        align = P2P_ALIGN
        target = max(int(l2_bytes) // P2P_L2_DIVISOR, int(n_eval_slots) // P2P_EVAL_DIVISOR)
    if np.isfinite(mem_limit):
        per = (16 * (int(w_src) + int(w_eval)) + 8 * int(w_eval) * (3 if field else 1)) * chan
        target = min(target, int(mem_limit // per) // align * align)
    target = max(target, align)
    if target >= n_pairs:
        return int(n_pairs)
    steps = -(-n_pairs // target)
    b = -(-n_pairs // steps)
    return int(min(n_pairs, -(-b // align) * align))


def m2l_budget(ws, mem_limit, p, npairs, itemsize = 8):
    r"""Batch budget for the M2L scan at multipole order ``p`` over ``npairs`` pairs.

    ``ws`` (:func:`ws_limit`) is the right operating point for the row-materialising
    stages, but the M2L scan pays per *batch*: every step scatters into the full locals
    carry, so at a fixed ``ws`` the step count grows with N while the carry outgrows L2
    (unitcube ladder, H100, M2L stage alone, 2026-09-07: element p3 at 9.04e7 tets 564 ms
    in 142 batches vs 245 ms in 9; lumped p4 189 ms in 73 vs 155 in 13). Too FEW batches
    cost as well (1-2 batches: +20-50% at 2e7 tets, and an f32 p=4 ``M2L_mixed`` once lost
    8x to a blanket 1 GiB floor), and the optimum sat at 9-44 batches on every probed
    row, so for ``p <= 4`` the budget is raised from ``ws`` to whatever caps the scan at
    ``M2L_MAX_BATCHES`` steps -- within ~10% of the measured optimum everywhere and
    never below ``ws``. The resulting transient is ``K_M2L * npairs*(p+1)^2*8 / 32``
    bytes = the size of the pair list itself at p=3 (it scales with the persistent M2L
    state, not beyond it); ``mem_limit / K_M2L`` keeps a user-set budget honest. At high
    p the small batches measurably win, so ``ws`` is returned unchanged there.

    GPU (2026-10-09 per-stage sweeps, 1e7 elements, p=3/6/8): the optimum is a fixed
    ~64 MiB working set per step at every p (H100; A100 and RTX PRO 6000 32-67 MiB), not a
    step count -- the rule above sat 1.07-1.24x off at p=3 and a 2-step default wasted a
    nearly full padded batch (2.3x). The GPU branch uses ``M2L_WS_BYTES`` with equal steps.

    Parameters
    ----------
    ws : float
        Minor-stage working set from :func:`ws_limit` (bytes, may be ``inf``).
    mem_limit : float
        Peak-transient byte budget the caller was given (bytes, may be ``inf``).
    p : int
        Multipole order.
    npairs : int
        Number of M2L interaction pairs (rows of ``mpl_cnct``).
    itemsize : int, optional
        Bytes per multipole coefficient (the batcher prices ``2 (p+1)^2 itemsize`` per pair).

    Returns
    -------
    float
        Byte budget to pass to the M2L batcher.
    """
    if jax.devices()[0].platform == "gpu":
        return _gpu_m2l_budget(mem_limit, p, npairs, itemsize)
    if p > 4 or not np.isfinite(ws):
        return ws
    mem_total = 2 * npairs * (p + 1) ** 2 * 4                  # what the batcher prices
    return max(ws, min(mem_total / M2L_MAX_BATCHES, mem_limit / K_M2L))


def _gpu_m2l_budget(mem_limit, p, npairs, itemsize):
    r"""GPU branch of :func:`m2l_budget` (pure): an ``M2L_WS_BYTES`` working set per scan
    step, split into EQUAL steps (the scan pads its last step to a full batch) and made ODD,
    capped at ``mem_limit / K_M2L``; returned as the byte budget for which M2L/M2L_mixed's own pricing
    (``2 (p+1)^2 itemsize`` per pair) yields exactly that batch."""
    per = 2 * (p + 1) ** 2 * int(itemsize)
    target = M2L_WS_BYTES if not np.isfinite(mem_limit) else min(M2L_WS_BYTES, mem_limit / K_M2L)
    b0 = max(1, int(target // per))
    if b0 >= npairs:
        return float("inf")
    steps = -(-int(npairs) // b0)
    b = -(-int(npairs) // steps)
    b += 1 - b % 2          # ODD batch: sizes with large power-of-2 factors run the M2L scan 25-30 %
                            # slower at equal step count (H100 film R5: 51200 pairs 338 ms vs 51281
                            # 264 ms; same on R3 and the cube) -- the opposite of the near field
    return (b - 0.5) * per


def ws_limit(mem_limit, maxleaf):
    r"""Working-set cap for the transient-honest MINOR stages, derived from the budget.

    ``mem_limit`` (0.3.1 semantics) is the peak-transient byte budget; the materialising
    flat P2P prices its ``(batch, w, w, 3)`` tensor at ``6*maxleaf`` bytes per gathered
    source row, so ``mem_limit / (6*maxleaf)`` is the equivalent gathered-row working
    set -- the historical L2-anchored operating point at which the minor stages
    (S2M/M2M/M2L/L2L/L2P and their kifmm counterparts) were tuned.  Routing this value
    to them preserves the measured defaults exactly (bigger is NOT better there: at
    high p the small M2L batches measurably win) while keeping every stage far under
    the budget.

    Parameters
    ----------
    mem_limit : float
        Peak-transient byte budget.
    maxleaf : int
        Widest leaf occupancy of the tree the budget applies to.

    Returns
    -------
    float
        Byte cap for the minor-stage batchers (their ``mem`` models price actual
        transients, so this value is directly comparable there).
    """
    return mem_limit / (6.0 * max(float(maxleaf), 1.0))


def hint_legacy_mem_limit(mem_limit):
    r"""Warn (once per value) when a user-passed ``mem_limit`` looks like pre-0.3.1 units.

    Since 0.3.1, ``mem_limit`` is the PEAK-TRANSIENT byte budget of the evaluation
    (tens of GB on a modern GPU).  Pre-0.3.1 it was the P2P gathered-row working set
    (L2-cache sized, tens of MB) -- a factor ``6*maxleaf`` smaller.  Values below
    128 MiB are almost certainly old-units scripts; warn but never alter behaviour
    (tiny devices may legitimately want tiny budgets)."""
    try:
        ml = float(mem_limit)
    except (TypeError, ValueError):
        return
    if np.isfinite(ml) and 0 < ml < 2.0**27:
        import warnings
        warnings.warn(
            f"mem_limit={ml:.3g} B is below 128 MiB. Since jaxFMM 0.3.1, mem_limit is the "
            f"PEAK MEMORY BUDGET of the evaluation (pre-0.3.1 it was the P2P working-set "
            f"size, ~6*N_max smaller). If this is a pre-0.3.1 value, multiply it by "
            f"6*N_max or drop it to use the auto default.", stacklevel=3)


def auto_mem_limit(maxleaf, dtype_bytes=8, frac=0.3, device_id=0, binned=False):
    r"""Hardware-adaptive ``mem_limit`` (peak-transient byte budget) for the gated stages.

    Parameters
    ----------
    maxleaf : int
        Widest leaf occupancy (sets the per-pair working-set scale).
    dtype_bytes : int
        Bytes per coordinate scalar.
    frac : float
        Fraction of total memory allowed for the OOM bound.
    device_id : int
        CUDA device queried for the L2 size.
    binned : bool
        Size for the occupancy-binned near field instead of the flat one.

    Returns
    -------
    mem_limit : float
        Peak-transient byte budget (0.3.1 semantics).  The FLAT branch returns the
        sqrt(2)-quantized L2-anchored working set scaled by ``6*maxleaf`` -- the actual
        bytes the materialising P2P touches per batch -- so consumers dividing by their
        own transient models reproduce the historical (measured-optimal) batch sizes
        bit-for-bit.

    Notes
    -----
    FLAT near field (``binned=False``): ANCHORS the batch to the device L2 cache via
    :func:`jaxfmm.util.get_gpu_l2_cache_size` -- the gathered-source rows per batch are
    the reuse unit, so sizing them to L2 maximises cache hits (the P2P batch sweep
    peaks at ~L2).  The corresponding TRANSIENT is ``L2 * maxleaf * 6`` (the
    ``(batch,maxleaf,maxleaf,3)`` distance tensor plus d2/norm intermediates at ~2x
    live multiplicity), which is what is returned.  Additionally caps at
    ``frac*TOTAL_VRAM`` for OOM safety.

    BINNED near field (``binned=True``): the tiered kernels are FUSED by XLA (the
    pairwise tile never fully materialises), so their gathered rows ARE the transient;
    anchors to ~0.5% of TOTAL device memory -- the measured runtime plateau -- and is
    returned unscaled.

    Both use TOTAL device memory: a free-VRAM read varies with whatever else is
    resident, silently re-keying trace caches AND throttling performance -- total is
    deterministic across process states.  The sqrt(2) quantization keeps the number of
    distinct cache-key values finite at a bounded (<= 1.19x) batch-size loss; a coarser
    power-of-two quantization measurably hurts."""
    quant = lambda v: float(2.0 ** (np.floor(2.0 * np.log2(max(v, 1.0))) / 2.0))
    if jax.devices()[device_id].platform == "cpu":
        # CPU: NO L2 anchor — small scan batches strangle XLA's CPU loop instead of
        # helping cache reuse. The OOM bound is frac*MemAvailable of transient (the
        # quantized row value scaled like the GPU flat branch, preserving the historical
        # batch sizes under the transient-pricing consumers), capped at 256 MiB PER THREAD:
        # the element near field's optimum grows with the thread count (exclusive zen4,
        # 4.8e5 tets: 64-256 MiB at 1 thread, 1 GiB at 4, 4 GiB at 16 — an unbounded
        # budget costs 1.3-1.8x at 1-4 threads, a fixed small one 1.5x at 16), i.e. the
        # right per-batch work amortises XLA's parallel dispatch, and no cache size
        # predicts it (256 MiB/thread is far above L2/L3). Lumped/point P2P: ~inert
        # (measured 0.87-0.96x at 1-4 threads, 1.2x at 16 for the lumped strayfield).
        try:
            avail = next(float(l.split()[1]) * 1024.0 for l in open("/proc/meminfo")
                         if l.startswith("MemAvailable"))
        except Exception:                                                # non-Linux fallback
            avail = 16e9
        try:
            threads = len(os.sched_getaffinity(0))                       # cgroup/Slurm-aware
        except Exception:
            threads = os.cpu_count() or 1
        val = frac * avail / max(float(maxleaf), 1.0)
        return min(quant(val) * 6.0 * max(float(maxleaf), 1.0), 2.0**28 * threads)
    try:
        cap = float(get_gpu_l2_cache_size(device_id))                    # the speed anchor = L2 cache
    except Exception:                                                    # not CUDA / driver unavailable
        cap = 4e7
    try:
        total = float(jax.devices()[0].memory_stats()["bytes_limit"])
        if binned:
            return quant(max(cap, 0.005 * total))                        # fused kernels: rows = transient
        oom_safe = 0.5 * total / max(maxleaf * 6.0 * (dtype_bytes / 8.0), 1.0)
        val = min(cap, oom_safe)
    except Exception:                                                    # no VRAM stats -> just the L2 anchor
        val = cap
    return quant(val) * 6.0 * max(float(maxleaf), 1.0)                   # rows -> transient bytes


def autotune_mem_limit(positions, pil, dir_cnct, mixed=False, frac=0.3, reps=5):
    r"""Pick the fastest OOM-safe ``mem_limit`` for the P2P near field by measurement.

    Times :func:`eval_direct` on THIS tree (dummy unit charges -- only shapes/dtypes
    matter).  The occupancy optimum is not perfectly predictable from L2 alone, so a
    short sweep around it beats the formula.  Call once at setup.

    Parameters
    ----------
    positions : (N, 3) jnp.ndarray
        Particle positions of THIS tree (their dtype sets the timed precision).
    pil : (max_boxes, N_max) jnp.ndarray of int32
        Particles-in-leaf indices, -1 padded (fed to :func:`_build_padded`).
    dir_cnct : (n_pairs, 2) array
        Near pair list (from :func:`near_pairs`).
    mixed : bool
        Time the mixed-precision variant of :func:`eval_direct` (f32 kernel,
        input-dtype accumulation).
    frac : float
        Fraction of total memory allowed for the OOM bound; sets the sweep's
        upper candidate via :func:`auto_mem_limit`.
    reps : int
        Timed repetitions per candidate (the mean is compared).

    Returns
    -------
    mem_limit : float
        The fastest candidate budget in bytes.
    """
    import time
    maxleaf = pil.shape[1]
    hi = auto_mem_limit(maxleaf, positions.itemsize, frac)
    cands = sorted({hi, hi / 2.0, hi / 4.0, hi / 8.0})
    ppts, pchg, rev = _build_padded(positions, jnp.ones(positions.shape[0], positions.dtype), pil)
    best, best_t = hi, float("inf")
    for ml in cands:
        f = jax.jit(lambda ml=ml: eval_direct(ppts, pchg, ppts, rev, dir_cnct, field=False, mem_limit=ml, mixed=mixed))
        try:
            r = f(); jax.block_until_ready(r); t0 = time.perf_counter()
            for _ in range(reps):
                r = f()
            jax.block_until_ready(r); t = (time.perf_counter() - t0) / reps
        except Exception:
            continue
        if t < best_t:
            best_t, best = t, ml
    return float(best)


