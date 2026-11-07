r"""Analytic translation operators (M2M, M2L, L2L) for real solid-harmonic expansions.

All operators use the "point-and-shoot" scheme: rotate the expansion so the
translation vector becomes the :math:`z`-axis (:func:`rot_azimuth` +
:func:`rot_polar`), translate along :math:`z` where the operator is
:math:`m`-block-diagonal, and rotate back — reducing the naive
:math:`O(p^4)` translation to :math:`O(p^3)`.

The z-translations are triangular index shifts for M2M/L2L, and for M2L the
separable ("folded") form

.. math::

    \mathrm{fct}[n', n] = \frac{(-1)^{n'+|m|}\,(n+n')!}{r^{\,n+n'+1}}
        = \big[(-1)^{n'} r^{-(n'+1)}\big]\;
          T_{\mathrm{unit}}[(m,n'),(m,n)]\;\big[r^{-n}\big],

with :math:`T_{\mathrm{unit}}` a global, :math:`r`-independent matrix
(:func:`_translation_unit`).
"""
import jax
import jax.numpy as jnp
from jaxfmm.kernels.laplace.rotation import rot_azimuth, rot_polar, polar_rot_base
from jaxfmm.util import cart_to_sph
from jax.scipy.special import factorial as fac
from jaxfmm.kernels.laplace.basis import mpl_idx, inv_mpl_idx
from functools import partial

@partial(jax.jit, static_argnames=["mem_limit"])
def M2M(coeffs, oldboxdims, newboxdims, unqs, invs, mem_limit = jnp.inf):
    r"""Translate child multipole expansions to their parents (upward pass).

    Point-and-shoot: rotate each child expansion so the child-to-parent vector
    is :math:`z`, apply the triangular z-shift

    .. math:: C^m_{n+j} \mathrel{+}= \frac{r^j}{j!}\, C^m_n ,

    rotate back, and sum the children of each parent.

    Parameters
    ----------
    coeffs : jnp.ndarray
        Child multipole coefficients, shape ``(n_child_boxes, (p+1)**2)``.
    oldboxdims : jnp.ndarray
        Child box centres, shape ``(n_child_boxes, 3)``.
    newboxdims : jnp.ndarray
        Parent box centres, shape ``(n_parent_boxes, 3)``.
    unqs, invs : list of jnp.ndarray
        Precomputed Wigner tables (``precompute_combs``).
    mem_limit : float, optional
        Memory budget in bytes (static); batches the parent loop via
        ``jax.lax.map`` when exceeded.

    Returns
    -------
    jnp.ndarray
        Parent multipole coefficients, shape ``(n_parent_boxes, (p+1)**2)``.
    """
    p = invs[-1].shape[0]-1
    n_chi = oldboxdims.shape[0]//newboxdims.shape[0]
    coeffs = coeffs.reshape((coeffs.shape[0]//n_chi,n_chi,coeffs.shape[1]))
    oldboxdims = oldboxdims.reshape((-1,n_chi,3))
    mem = invs[-1].size * oldboxdims.shape[0] * n_chi * coeffs.dtype.itemsize
    batch_size = int((mem_limit/mem) * coeffs.shape[0]) + 1 if mem > mem_limit else coeffs.shape[0]

    def M2M_batched(args):
        coeffs, oldboxdims, newboxdims = args
        r, theta, phi = cart_to_sph(oldboxdims - newboxdims)
        ### rotate
        base = polar_rot_base(unqs, theta, p)                       # Wigner-d base, reused fwd+back
        coeffs = rot_azimuth(coeffs, phi)
        coeffs = rot_polar(unqs,invs,theta,coeffs,base=base)
        ### shift in the rotated system
        new_coeffs = coeffs.copy()
        for j in range(1,p+1):
            fact = (r**j) / fac(j)
            ms, ns = inv_mpl_idx(jnp.arange((p-j+1)**2))
            idxs = ms + ns + (ns+j)**2 + j
            new_coeffs = new_coeffs.at[...,idxs].add(fact[...,None] * coeffs[...,:mpl_idx(p-j,p-j)+1])
        ### rotate back
        coeffs = rot_polar(unqs,invs,theta,new_coeffs,True,base=base)
        coeffs = rot_azimuth(coeffs, phi,True)
        return coeffs.sum(axis=(-2))
    return jax.lax.map(M2M_batched,[coeffs,oldboxdims,newboxdims[:,None,:]],batch_size = batch_size)

@partial(jax.jit, static_argnames=["mem_limit"])
def L2L(coeffs_glob, oldboxdims, newboxdims, unqs, invs, mem_limit = jnp.inf):
    r"""Translate parent local expansions to their children (downward pass).

    Adjoint of :func:`M2M`: rotate so the parent-to-child vector is :math:`z`,
    apply the triangular z-shift

    .. math:: C^m_n \mathrel{+}= \frac{r^j}{j!}\, C^m_{n+j} ,

    and rotate back (with the local-expansion normalisation, ``loc=True``).

    Parameters
    ----------
    coeffs_glob : jnp.ndarray
        Parent local coefficients, shape ``(n_parent_boxes, (p+1)**2)``.
    oldboxdims : jnp.ndarray
        Parent box centres, shape ``(n_parent_boxes, 3)``.
    newboxdims : jnp.ndarray
        Child box centres, shape ``(n_child_boxes, 3)``.
    unqs, invs : list of jnp.ndarray
        Precomputed Wigner tables (``precompute_combs``).
    mem_limit : float, optional
        Memory budget in bytes (static); batches via ``jax.lax.map``.

    Returns
    -------
    jnp.ndarray
        Child local coefficients, shape ``(n_child_boxes, (p+1)**2)``.
    """
    p = invs[-1].shape[0]-1
    n_chi = newboxdims.shape[0]//oldboxdims.shape[0]
    coeffs_glob = coeffs_glob[:,None,:]
    newboxdims = newboxdims.reshape((-1,n_chi,3))
    mem = invs[-1].size * newboxdims.shape[0] * n_chi * coeffs_glob.dtype.itemsize
    batch_size = int((mem_limit/mem) * coeffs_glob.shape[0]) + 1 if mem > mem_limit else coeffs_glob.shape[0]

    def L2L_batched(args):
        coeffs, oldboxdims, newboxdims = args
        r ,theta, phi = cart_to_sph(newboxdims - oldboxdims)
        ### rotate
        base = polar_rot_base(unqs, theta, p)                       # Wigner-d base, reused fwd+back
        coeffs = rot_azimuth(coeffs,phi,False,True)
        coeffs = rot_polar(unqs,invs,theta,coeffs,False,True,base=base)
        ### shift in the rotated system
        new_coeffs = coeffs.copy()
        for j in range(1,p+1):
            fact = (r**j) / fac(j)
            ms, ns = inv_mpl_idx(jnp.arange((p-j+1)**2))
            idxs = ms + ns + (ns+j)**2 + j
            new_coeffs = new_coeffs.at[...,:mpl_idx(p-j,p-j)+1].add(fact[...,None] * coeffs[...,idxs])
        ### rotate back
        coeffs = rot_polar(unqs,invs,theta,new_coeffs,True,True,base=base)
        coeffs = rot_azimuth(coeffs,phi,True,True)
        return coeffs
    coeffs = jax.lax.map(L2L_batched,[coeffs_glob,oldboxdims[:,None,:],newboxdims],batch_size = batch_size)
    return coeffs.reshape((-1,coeffs.shape[-1]))

@partial(jax.jit, static_argnames=['mem_limit', 'remat'])
def M2L(mpls, src_centers, trg_centers, mpl_cnct, unqs, invs, img_cnct = jnp.array([[]]), mem_limit = jnp.inf, remat = False, img_phase = None):
    r"""Convert multipole to local expansions for all well-separated pairs at once.

    Per-pair point-and-shoot M2L: rotate the source multipole onto the pair
    axis, apply the z-translation in the folded form (see the module
    docstring), rotate back with the local normalisation, and scatter-add into
    the target boxes.

    Parameters
    ----------
    mpls : jnp.ndarray
        Multipole coefficients of all source boxes, shape
        ``(n_src_boxes, (p+1)**2)``.
    src_centers, trg_centers : jnp.ndarray
        Source / target box centres, shapes ``(n_src_boxes, 3)`` /
        ``(n_trg_boxes, 3)``.
    mpl_cnct : jnp.ndarray
        Well-separated pairs ``(trg, src)``, shape ``(n_pairs, 2)``; padded
        rows carry max-int sentinels and are dropped.
    unqs, invs : list of jnp.ndarray
        Precomputed Wigner tables (``precompute_combs``).
    img_cnct : jnp.ndarray, optional
        Periodic image offsets, shape ``(n_img, 3)``; source ids then encode
        ``box + n_boxes * image``. Shape ``(1, 0)`` for open boundaries.
    mem_limit : float, optional
        Memory budget in bytes (static); the pair list is processed in
        ``jax.lax.scan`` batches when exceeded.

    Returns
    -------
    jnp.ndarray
        Local expansion coefficients per target box, shape
        ``(n_trg_boxes, (p+1)**2)``.
    """
    p = invs[-1].shape[0]-1
    dim = (p + 1) ** 2
    _, ns_deg = inv_mpl_idx(jnp.arange(dim))                # source/output degrees n
    T_unit = _translation_unit(p)                            # global z-translation (jit const)
    sgn_row = jnp.where(ns_deg % 2 == 0, 1.0, -1.0)          # (-1)^n' of the rowscale

    # Bloch: complex accumulator when images carry phases
    _ldt = jnp.result_type(mpls.dtype, img_phase.dtype) if img_phase is not None else mpls.dtype
    locs = jnp.zeros((trg_centers.shape[0], (p + 1) ** 2), _ldt)
    mem = 2 * mpl_cnct.shape[0] * (p+1)**2 * mpls.itemsize
    batch_size = int((mem_limit/mem)*mpl_cnct.shape[0]) + 1 if mem_limit<mem else mpl_cnct.shape[0]
    n_boxs = src_centers.shape[0]

    def scan_body(buf, idcs):
        ok = (idcs[:, 0] < trg_centers.shape[0]) & (idcs[:, 1] < mpls.shape[0] * max(int(img_cnct.shape[0]), 1))
        if(img_cnct.shape[1] > 0):  # both pbc images and real boxes, shift appropriately
            trg_ids = idcs[:,1]%n_boxs
            dvec = trg_centers[idcs[:,0]] - src_centers[trg_ids] - img_cnct[idcs[:,1]//n_boxs]
        else:   # open boundary conditions
            trg_ids = idcs[:,1]
            dvec = trg_centers[idcs[:,0]] - src_centers[trg_ids]
        # padded pairs (max-int sentinels, clip-gathered) can give degenerate dvec; keep it
        # finite or the BACKWARD pass NaNs even though the forward scatter drops those rows
        # (a masked jnp.where does not protect gradients)
        dvec = jnp.where(ok[:, None], dvec, 1.0)
        r, theta, phi = cart_to_sph(dvec)
        coeffs_loc = mpls[trg_ids]

        base = polar_rot_base(unqs, theta, p)                       # Wigner-d base, reused fwd+back
        coeffs_loc = rot_azimuth(coeffs_loc, phi)
        coeffs_loc = rot_polar(unqs,invs,theta,coeffs_loc,base=base)
        ### z-translation, FOLDED form (colscale r^-n -> global T_unit GEMM -> rowscale):
        ### mathematically identical to the classic per-block fac(n+n')/r^(n+n'+1) loop
        ### (one GEMM instead of p+1 block updates), and every intermediate stays
        ### O(q*3^-n) — no fac/r^(n+n'+1)-scale factors, which is what makes the f32
        ### M2L_mixed possible. Differs from the classic loop only in summation order.
        rinv = (1.0 / r)[..., None]
        coeffs_loc = coeffs_loc * rinv ** ns_deg
        coeffs_loc = jnp.einsum("mn,...n->...m", T_unit, coeffs_loc)
        coeffs_loc = coeffs_loc * sgn_row * rinv ** (ns_deg + 1)
        ### rotate back
        coeffs_loc = rot_polar(unqs,invs,theta,coeffs_loc,True,True,base=base)
        coeffs_loc = rot_azimuth(coeffs_loc, phi,True,True)
        if img_phase is not None:                    # Bloch: e^{ik.R} per image pair
            coeffs_loc = coeffs_loc * img_phase.at[idcs[:, 1] // n_boxs].get(
                mode="fill", fill_value=0.0)[:, None]
        # BARRIER before the scatter: XLA (measured on Blackwell/RTX PRO 6000, jax 0.10.2)
        # miscompiles the scatter-add when the fused rotation chain is its operand — at
        # specific coefficient widths ((p+1)^2 = 36/49/121, i.e. p=5/6/10) the fused-scatter
        # kernel computes the operand WRONG by O(4%) while the identical math returned
        # directly is exact (gto4 jobs 2152-2159, standalone repro dev/fmm_p_f32_bug/).
        # The barrier materialises coeffs_loc and blocks the fusion; H100 unaffected either
        # way. This was the whole "f32 breaks at p=5,6,10" defect.
        coeffs_loc = jax.lax.optimization_barrier(coeffs_loc)
        return buf.at[idcs[:,0]].add(coeffs_loc.astype(buf.dtype), indices_are_sorted=True), None

    # remat: reverse-mode through the batched scan otherwise STACKS the
    # per-batch gathered/broadcast multipole intermediates as residuals --
    # measured 69 buffers / 38.7 GB at 2.24M tets (the dominant remaining
    # adjoint cost after the S2M fix). Checkpoint recomputes them per batch
    # in the backward pass; effective only with a finite mem_limit.
    body = jax.checkpoint(scan_body) if remat else scan_body
    if(batch_size<mpl_cnct.shape[0]):
        num_pad = batch_size - mpl_cnct.shape[0]%batch_size
        # lax.pad demands the operand's exact dtype: a plain Python int is int64 under
        # x64, which an f32-built tree's int32 connectivity rejects the moment M2L batches.
        fill_value = mpl_cnct.dtype.type(jnp.iinfo(mpl_cnct.dtype).max)
        scanmpl = jax.lax.pad(mpl_cnct, fill_value, ((0,num_pad,0), (0,0,0))).reshape((-1,batch_size,2))
        locs, _ = jax.lax.scan(body,locs,scanmpl)
    else:
        locs, _ = body(locs,mpl_cnct)

    return locs

@partial(jax.jit, static_argnames=["mem_limit"])
def M2L_mixed(mpls, src_centers, trg_centers, mpl_cnct, unqs, invs, img_cnct=jnp.array([[]]),
              mem_limit=jnp.inf):
    r"""Mixed-precision variant of :func:`M2L` (open boundaries only).

    f32 Wigner rotations + folded z-translation, f64 accumulation. The phase
    noise from the f32 rotations is invisible under low-order truncation and a
    mild accuracy bump at high orders. Select via
    ``eval_potential(..., m2l_mixed=True)``.

    Parameters
    ----------
    mpls, src_centers, trg_centers, mpl_cnct, unqs, invs, img_cnct, mem_limit
        As in :func:`M2L`.

    Returns
    -------
    jnp.ndarray
        Local expansion coefficients per target box (f64), shape
        ``(n_trg_boxes, (p+1)**2)``.

    Notes
    -----
    The folded form is essential here: the classic translation factor
    :math:`(n+n')! / r^{\,n+n'+1}` overflows f32 at deep levels (small
    :math:`r`, high :math:`p`), while folding (colscale :math:`r^{-n}` into the
    coefficients, then the global :math:`T_{\mathrm{unit}}` GEMM, then the
    rowscale) keeps every intermediate :math:`O(q\,3^{-n})`. The pair
    displacement must be formed in f64 BEFORE the cast: box centres are
    near-equal at depth, so a global cast would lose the separation bits.
    """
    dt = jnp.float32
    p = invs[-1].shape[0] - 1
    dim = (p + 1) ** 2
    ms, ns = inv_mpl_idx(jnp.arange(dim))
    T = _translation_unit(p).astype(dt)
    sgn_row = jnp.where(ns % 2 == 0, 1.0, -1.0).astype(dt)
    locs = jnp.zeros((trg_centers.shape[0], dim))
    mem = 2 * mpl_cnct.shape[0] * dim * mpls.itemsize
    batch_size = int((mem_limit / mem) * mpl_cnct.shape[0]) + 1 if mem_limit < mem else mpl_cnct.shape[0]
    n_boxs = src_centers.shape[0]

    def scan_body(buf, idcs):
        ok = (idcs[:, 0] < trg_centers.shape[0]) & (idcs[:, 1] < n_boxs * max(int(img_cnct.shape[0]), 1))
        if img_cnct.shape[1] > 0:            # PBC images: decode + shift in f64, THEN cast
            src_ids = idcs[:, 1] % n_boxs
            dvec = (trg_centers[idcs[:, 0]] - src_centers[src_ids]
                    - img_cnct[idcs[:, 1] // n_boxs]).astype(dt)
        else:
            src_ids = idcs[:, 1]
            dvec = (trg_centers[idcs[:, 0]] - src_centers[src_ids]).astype(dt)   # f64 diff, then cast
        dvec = jnp.where(ok[:, None], dvec, 1.0)     # padded pairs: backward-NaN sanitization
        r, theta, phi = cart_to_sph(dvec)
        C = mpls[src_ids].astype(dt)
        base = polar_rot_base(unqs, theta, p)
        C = rot_azimuth(C, phi)
        C = rot_polar(unqs, invs, theta, C, base=base)
        rinv = (1.0 / r)[..., None]
        C = C * rinv ** ns                                   # colscale r^-n (bounded ~3^-n)
        C = jnp.einsum("mn,...n->...m", T, C)                # shared translation GEMM
        C = C * sgn_row * rinv ** (ns + 1)                   # rowscale (-1)^n' r^-(n'+1)
        C = rot_polar(unqs, invs, theta, C, True, True, base=base)
        C = rot_azimuth(C, phi, True, True)
        C = jax.lax.optimization_barrier(C)          # same scatter-fusion hazard as M2L
        return buf.at[idcs[:, 0]].add(C.astype(buf.dtype), indices_are_sorted=True), None

    if batch_size < mpl_cnct.shape[0]:
        num_pad = batch_size - mpl_cnct.shape[0] % batch_size
        # lax.pad demands the operand's exact dtype: a plain Python int is int64 under
        # x64, which an f32-built tree's int32 connectivity rejects the moment M2L batches.
        fill = mpl_cnct.dtype.type(jnp.iinfo(mpl_cnct.dtype).max)
        scanmpl = jax.lax.pad(mpl_cnct, fill, ((0, num_pad, 0), (0, 0, 0))).reshape((-1, batch_size, 2))
        locs, _ = jax.lax.scan(scan_body, locs, scanmpl)
    else:
        locs, _ = scan_body(locs, mpl_cnct)
    return locs


# ──────────────────────────────────────────────────────────────────────────
# Theta-snapped GROUPED M2L.  The per-pair rotation M2L above is memory-bound: it
# builds the Wigner-d Jacobi base PER PAIR and applies it as batched matvecs.
# Snapping theta to K bin centres makes the rotation matrices SHARED per class
# (snap error ~0.1/K, INDEPENDENT of p: K~256 matches p=3 truncation,
# K~1024-2048 matches p=6), and the z-translation is SEPARABLE:
#     fct[n',n] = (-1)^(n'+|m|) fac(n+n') / r^(n+n'+1)
#               = [(-1)^n' r^-(n'+1)] * T_unit[(m,n'),(m,n)] * [r^-n]
# with T_unit global (r-independent).  The whole M2L then runs as, per cap-tier:
#     phi-rot -> class GEMM -> diag -> ONE shared T_unit GEMM -> diag -> class GEMM -> phi-rot
# with r and phi kept EXACT (only theta is snapped).
# ──────────────────────────────────────────────────────────────────────────
def _dense_rot_mats(unqs, invs, thetas, p, inv, loc):
    r"""Fold the per-degree Wigner blocks into one dense rotation matrix per class.

    Parameters
    ----------
    unqs, invs : list of jnp.ndarray
        Precomputed Wigner tables.
    thetas : jnp.ndarray
        Snapped class angles, shape ``(K,)``.
    p : int
        Expansion order.
    inv, loc : bool
        As in :func:`rot_polar`.

    Returns
    -------
    jnp.ndarray
        Block-diagonal rotation matrices, shape ``(K, (p+1)**2, (p+1)**2)``,
        assembled from ``polar_rot_mats``.
    """
    import numpy as np
    from jaxfmm.kernels.laplace.rotation import polar_rot_mats
    reals, imags = polar_rot_mats(unqs, invs, thetas, p, inv=inv, loc=loc)
    K = thetas.shape[0]; dim = (p + 1) ** 2
    R = np.zeros((K, dim, dim))
    for j in range(p + 1):
        lo_r = int(mpl_idx(0, j)); hi_r = int(mpl_idx(j, j)) + 1
        R[:, lo_r:hi_r, lo_r:hi_r] = np.asarray(reals[j])
        if j > 0:
            lo_i = int(mpl_idx(-j, j)); hi_i = int(mpl_idx(0, j))
            R[:, lo_i:hi_i, lo_i:hi_i] = np.asarray(imags[j])
    return jnp.asarray(R)


def _translation_unit(p):
    r"""Global (r-independent) M2L z-translation matrix :math:`T_{\mathrm{unit}}`.

    :math:`m`-block-diagonal :math:`(n+n')!` with the :math:`(-1)^{|m|}` block
    sign (no separate sign on the :math:`m<0` rows in the sqrt(2) basis). The
    :math:`r`-dependent parts of the translation are the per-pair diagonal
    scales (see the module docstring). jnp-native, hence jit-safe: XLA
    constant-folds it inside a trace.

    Parameters
    ----------
    p : int
        Expansion order.

    Returns
    -------
    jnp.ndarray
        Translation matrix, shape ``((p+1)**2, (p+1)**2)``.
    """
    dim = (p + 1) ** 2
    ms, ns = inv_mpl_idx(jnp.arange(dim))
    # .astype(float): inv_mpl_idx yields int32 degrees and jax's factorial promotes
    # int32 -> FLOAT32 (4e-6 rel. error at 16!), which floored the isolated f64 M2L
    # error at ~4e-7 regardless of p (measured 2026-09-09; now exact to truncation).
    # It is NOT the ~8e-7 floor of the fine element ladders (see ARCHITECTURE §7 3b).
    T = (ms[:, None] == ms[None, :]) * fac((ns[:, None] + ns[None, :]).astype(float))
    return T * (-1.0) ** jnp.abs(ms[:, None])


def m2l_theta_setup(mpl_cnct, boxcenters, eval_boxcenters, unqs, invs, K=256, n_tiers=5):
    r"""Host-side (fixed-topology) setup for the theta-snapped grouped M2L.

    Buckets the valid M2L pairs by snapped-theta class, DP-partitions the
    count-sorted classes into at most ``n_tiers`` capacity tiers (minimising
    padded rows — the theta histogram is strongly skewed, so equal caps waste
    heavily), and precomputes the ``K`` dense rotation matrices (both
    directions) plus :math:`T_{\mathrm{unit}}`.

    Parameters
    ----------
    mpl_cnct : array_like
        Well-separated pairs ``(trg, src)``; -1 or sentinel padding allowed.
        Must be a plain open-boundary list (PBC hierarchies unsupported).
    boxcenters, eval_boxcenters : array_like
        Source / target box centres (concatenated over levels).
    unqs, invs : list of jnp.ndarray
        Precomputed Wigner tables.
    K : int, optional
        Number of theta classes. Pick ``K`` to push the snap error
        (:math:`\sim 0.1/K`) below the truncation error: ~256 for ``p=3``,
        ~1024–2048 for ``p=6``.
    n_tiers : int, optional
        Maximum number of capacity tiers.

    Returns
    -------
    dict
        The ``m2l_grp`` bundle consumed by :func:`M2L_grouped` /
        ``eval_potential(..., m2l_grp=...)``: per-tier class indices ``cls``
        and padded pair buffers ``cnct``, rotation matrices ``RF``/``RB``,
        and ``T``.

    Notes
    -----
    Classes only span :math:`[0, \pi/2]`: the Laplace kernel is z-mirror
    invariant and real solid harmonics transform diagonally under it, so

    .. math:: M2L(\pi - \theta) = S \, M2L(\theta) \, S, \qquad
              S = \mathrm{diag}\big((-1)^{n+m}\big),

    doubling the angular resolution per ``K``. The applier detects reflected
    pairs from :math:`d_z < 0` (no extra storage).
    """
    import numpy as np
    p = invs[-1].shape[0] - 1
    cn = np.asarray(mpl_cnct)
    okm = ((cn[:, 0] >= 0) & (cn[:, 1] >= 0)                       # -1 padding (eager lists)
           & (cn[:, 0] < eval_boxcenters.shape[0])                 # max-int sentinel padding
           & (cn[:, 1] < boxcenters.shape[0]))                     # (jit connectivity)
    cn = cn[okm]
    d = np.asarray(eval_boxcenters)[cn[:, 0]] - np.asarray(boxcenters)[cn[:, 1]]
    th = np.arctan2(np.hypot(d[:, 0], d[:, 1]), np.abs(d[:, 2]))   # |d_z|: z-reflection
    cls = np.clip((th / (0.5 * np.pi / K)).astype(int), 0, K - 1)  # symmetry, see Notes
    counts = np.bincount(cls, minlength=K)
    order = np.argsort(cls, kind="stable")
    starts = np.searchsorted(cls[order], np.arange(K))
    ends = np.append(starts[1:], len(cls))

    csort = np.argsort(counts, kind="stable"); c = counts[csort]; n_g = K
    T_t = min(n_tiers, n_g)                                # DP tier partition (as m2l_groups_tiered)
    INF = float("inf")
    best = np.full((T_t + 1, n_g + 1), INF); best[0, 0] = 0.0
    cut = np.zeros((T_t + 1, n_g + 1), np.int64)
    for k in range(1, T_t + 1):
        for i in range(k, n_g + 1):
            js = np.arange(k - 1, i)
            cost = best[k - 1, js] + (i - js) * c[i - 1]
            j = int(js[np.argmin(cost)])
            best[k, i] = cost[j - (k - 1)]; cut[k, i] = j
    bounds = [n_g]
    for k in range(T_t, 0, -1):
        bounds.append(int(cut[k, bounds[-1]]))
    bounds = bounds[::-1]

    cn_sorted = cn[order]                                  # pairs grouped by class
    sorted_cls = cls[order]
    pos_in_cls = np.arange(len(cls)) - starts[sorted_cls]  # slot within the class
    cls_idxs, cncts = [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi <= lo:
            continue
        sel = csort[lo:hi]; cap = max(int(c[hi - 1]), 1)
        rank = np.full(K, -1, np.int64); rank[sel] = np.arange(len(sel))
        rows = rank[sorted_cls]; m = rows >= 0             # vectorised tier fill (a python
        arr = np.full((len(sel), cap, 2), -1, np.int32)    # per-class loop does not scale)
        arr[rows[m], pos_in_cls[m]] = cn_sorted[m]
        cls_idxs.append(jnp.asarray(sel)); cncts.append(jnp.asarray(arr))

    thetas = (jnp.arange(K) + 0.5) * (0.5 * jnp.pi / K)            # classes span [0, pi/2] (z-reflection)
    return dict(cls=tuple(cls_idxs), cnct=tuple(cncts),
                RF=_dense_rot_mats(unqs, invs, thetas, p, inv=False, loc=False),
                RB=_dense_rot_mats(unqs, invs, thetas, p, inv=True, loc=True),
                T=_translation_unit(p))


@partial(jax.jit, static_argnames=["mem_limit"])
def M2L_grouped(mpls, src_centers, trg_centers, m2l_grp, mem_limit=jnp.inf):
    r"""Theta-snapped grouped M2L applier.

    Same contract as :func:`M2L` (open boundaries only): local expansions at
    every target box. :math:`r` and :math:`\phi` are exact per pair; only the
    polar rotation uses the snapped class matrices.

    Parameters
    ----------
    mpls : jnp.ndarray
        Source multipole coefficients, shape ``(n_src_boxes, (p+1)**2)``.
    src_centers, trg_centers : jnp.ndarray
        Source / target box centres.
    m2l_grp : dict
        Group bundle from :func:`m2l_theta_setup` (or the frozen variant plus
        :func:`m2l_theta_bin`).
    mem_limit : float, optional
        Memory budget in bytes (static); chunks each tier along the cap axis.
        The working set is ~6 live ``(n_classes, cap, dim)`` tensors, so
        unchunked evaluation OOMs at large pair counts.

    Returns
    -------
    jnp.ndarray
        Local expansion coefficients per target box, shape
        ``(n_trg_boxes, (p+1)**2)``.
    """
    dim = m2l_grp["T"].shape[0]
    p = int(dim ** 0.5) - 1
    ms, ns = inv_mpl_idx(jnp.arange(dim))
    flipidx = mpl_idx(-ms, ns)
    locs = jnp.zeros((trg_centers.shape[0], dim))
    sgn_z = jnp.where((ns + jnp.abs(ms)) % 2 == 0, 1.0, -1.0)      # z-reflection rep: (-1)^(n+m)
    for cls_idx, cnct_pad in zip(m2l_grp["cls"], m2l_grp["cnct"]):
        nc, cap, _ = cnct_pad.shape
        mem = nc * cap * dim * mpls.dtype.itemsize * 6             # ~6 live intermediates
        n_chunks = int(mem / mem_limit) + 1 if mem > mem_limit else 1
        cw = -(-cap // n_chunks)                                   # ceil-div chunk width
        for off in range(0, cap, cw):
            blk = cnct_pad[:, off:off + cw]
            ok = blk[..., 0] >= 0
            trg = jnp.where(ok, blk[..., 0], 0)
            src = jnp.where(ok, blk[..., 1], 0)
            dvec = trg_centers[trg] - src_centers[src]
            dvec = jnp.where(ok[..., None], dvec, 1.0)             # padded rows point at the roots
            r, _, phi = cart_to_sph(dvec)                          # (dvec~0): keep r/phi FINITE or
            C = mpls[src]                                          # the BACKWARD pass NaNs (the
            # masked-where does not protect gradients — same trap as eval_direct's safe 1/r)

            mphi = ms[None, None, :] * phi[..., None]
            cphi, sphi = jnp.cos(mphi), jnp.sin(mphi)
            rinv = 1.0 / r[..., None]
            colscale = rinv ** ns                                  # r^-n (source degree)
            rowscale = (jnp.where(ns[None, None, :] % 2 == 0, 1.0, -1.0)
                        * rinv ** (ns + 1))                        # (-1)^n' r^-(n'+1)
            # z-mirrored pairs (theta > pi/2, i.e. dz < 0) use the reflected class conjugated
            # by the diagonal sign S: M2L(d) = S M2L(d') S — doubles the resolution per K
            S = jnp.where(dvec[..., 2:] < 0, sgn_z[None, None, :], 1.0)

            C = C * S                                              # reflect in (Laplace z-mirror)
            C = C * cphi + C[..., flipidx] * sphi                  # phi (exact; rot_azimuth sgns=-1)
            C = jnp.einsum("kmn,kcn->kcm", m2l_grp["RF"][cls_idx], C)  # fwd rot (class GEMM)
            C = C * colscale
            C = jnp.einsum("mn,kcn->kcm", m2l_grp["T"], C)         # shared translation GEMM
            C = C * rowscale
            C = jnp.einsum("kmn,kcn->kcm", m2l_grp["RB"][cls_idx], C)  # inv rot (class GEMM)
            C = C * cphi - C[..., flipidx] * sphi                  # inv phi (rot_azimuth sgns=+1)
            C = C * S                                              # reflect out
            C = jnp.where(ok[..., None], C, 0.0)
            locs = locs.at[trg.reshape(-1)].add(
                jax.lax.optimization_barrier(C).reshape(-1, dim), mode="drop")
    return locs


def m2l_theta_setup_frozen(mpl_cnct_ref, boxcenters, eval_boxcenters, unqs, invs, K=256,
                           cap_gran=256, headroom=0.15):
    r"""Frozen-caps variant of :func:`m2l_theta_setup` for the fully-jit shape-opt path.

    The class-to-tier assignment and per-tier capacities are probed at the
    REFERENCE geometry and frozen (``headroom`` fraction of spare capacity, rounded onto the
    geometric cap ladder like the fulljit connectivity buffers), so the per-pair class
    binning can run INSIDE jit with fixed shapes (:func:`m2l_theta_bin`).

    Parameters
    ----------
    mpl_cnct_ref : array_like
        M2L pair list at the reference geometry.
    boxcenters, eval_boxcenters : array_like
        Source / target box centres at the reference geometry.
    unqs, invs : list of jnp.ndarray
        Precomputed Wigner tables.
    K : int, optional
        Number of theta classes.
    cap_gran : int, optional
        Capacity granularity (rounding unit).
    headroom : float, optional
        Spare capacity per frozen cap, as a fraction (``_fit_cap``).

    Returns
    -------
    dict
        Frozen group bundle: rotation matrices ``RF``/``RB``/``T`` as usual,
        per-tier frozen ``cls`` index arrays and static ``caps``, and the
        ``(K,)`` lookup tables ``tier_of_cls`` / ``row_of_cls`` used by the
        in-jit binner.

    Notes
    -----
    Overflowing a frozen class cap silently DROPS pairs — run moving-geometry
    loops through ``jaxfmm.driver.GuardedShapeEvaluator`` (which checks the
    binner's realised counts every call), or validate manually with
    ``element_quadtree_fulljit_check``.
    """
    import numpy as np
    from jaxfmm.trees.connectivity import _fit_cap          # local: trees imports this package
    base = m2l_theta_setup(mpl_cnct_ref, boxcenters, eval_boxcenters, unqs, invs, K=K)
    caps, tier_of, row_of = [], np.full(K, -1, np.int32), np.full(K, 0, np.int32)
    for t, (cls_idx, cnct) in enumerate(zip(base["cls"], base["cnct"])):
        cls_np = np.asarray(cls_idx)
        used = int(cnct.shape[1])
        cap = _fit_cap(used, cap_gran, headroom)
        caps.append(cap)
        tier_of[cls_np] = t
        row_of[cls_np] = np.arange(len(cls_np))
    return dict(cls=base["cls"], RF=base["RF"], RB=base["RB"], T=base["T"],
                caps=tuple(caps), tier_of_cls=jnp.asarray(tier_of),
                row_of_cls=jnp.asarray(row_of), K=K)


def m2l_theta_bin(mpl_cnct, src_centers, trg_centers, grp, return_counts=False):
    r"""In-jit class binning for the frozen theta-snap group.

    Computes each pair's snapped class from the (current, traced) box centres,
    argsorts, and scatters into the frozen ``(n_classes_tier, cap)`` tier
    buffers (overflow / foreign rows drop). All shapes are static; the class
    ids are combinatorial (``stop_gradient`` upstream, as for the pair lists).

    Parameters
    ----------
    mpl_cnct : jnp.ndarray
        M2L pair list ``(trg, src)``, padded rows negative or out of range.
    src_centers, trg_centers : jnp.ndarray
        Current (traced) box centres.
    grp : dict
        Frozen group bundle from :func:`m2l_theta_setup_frozen`.
    return_counts : bool, optional
        Also return realised occupancies (see Notes).

    Returns
    -------
    tier_cncts : tuple of jnp.ndarray
        Per-tier pair buffers consumed by :func:`M2L_grouped`.
    counts : dict, optional
        Only if ``return_counts=True``: ``theta_tier_max`` — ``(n_tiers,)``
        max pairs in any class of each tier — and ``theta_orphans`` — pairs
        falling in classes that had NO pairs at the reference geometry
        (``tier_of_cls == -1``). All traced scalars, essentially free.

    Notes
    -----
    Orphan rows do not exist in any tier buffer, so a single orphan means
    dropped physics; the guarded driver checks these counts against the frozen
    caps every call.
    """
    K = grp["K"]
    n_e, n_s = trg_centers.shape[0], src_centers.shape[0]
    tgt, src = mpl_cnct[:, 0], mpl_cnct[:, 1]
    ok = (tgt >= 0) & (src >= 0) & (tgt < n_e) & (src < n_s)
    t_s = jnp.where(ok, tgt, 0); s_s = jnp.where(ok, src, 0)
    d = trg_centers[t_s] - src_centers[s_s]
    th = jnp.arctan2(jnp.hypot(d[:, 0], d[:, 1]), jnp.abs(d[:, 2]))   # z-reflection binning
    cls = jnp.clip((th / (0.5 * jnp.pi / K)).astype(jnp.int32), 0, K - 1)
    cls = jnp.where(ok, cls, K)                                        # invalid -> dump class
    order = jnp.argsort(cls)
    cls_s = cls[order]
    pairs_s = jnp.stack([t_s, s_s], 1)[order]
    starts = jnp.searchsorted(cls_s, jnp.arange(K + 1))
    pos = jnp.arange(cls_s.shape[0]) - starts[jnp.clip(cls_s, 0, K)]
    tier_pp = jnp.where(cls_s < K, grp["tier_of_cls"][jnp.clip(cls_s, 0, K - 1)], -1)
    row_pp = grp["row_of_cls"][jnp.clip(cls_s, 0, K - 1)]
    tier_cncts = []
    for t, (cls_idx, cap) in enumerate(zip(grp["cls"], grp["caps"])):
        n_rows = cls_idx.shape[0]
        rows = jnp.where((tier_pp == t) & (pos < cap), row_pp, n_rows)   # OOB -> dropped
        buf = jnp.full((n_rows, cap, 2), -1, mpl_cnct.dtype)
        buf = buf.at[rows, jnp.clip(pos, 0, cap - 1)].set(pairs_s, mode="drop")
        tier_cncts.append(buf)
    if not return_counts:
        return tuple(tier_cncts)
    cls_counts = jnp.diff(starts)                                        # (K,) real classes only
    tier_max = jnp.stack([cls_counts[cls_idx].max() for cls_idx in grp["cls"]])
    orphans = jnp.where(grp["tier_of_cls"] < 0, cls_counts, 0).sum()
    return tuple(tier_cncts), dict(theta_tier_max=tier_max, theta_orphans=orphans)
