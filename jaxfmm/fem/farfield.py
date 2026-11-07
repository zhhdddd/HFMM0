r"""The element-tree engine (boxes sized to hold whole elements): far field via the
flexible FMM, tight low-order near quadrature, Duffy self term."""
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jaxfmm.trees import balanced_tree, balanced_tree_elements, element_aabb, gen_connectivity, reduce_max_lvl, AdaptiveConnectivity
from jaxfmm.fem.quadrature import simplex_quad, element_quad_points
from jaxfmm.kernels.laplace import precompute_combs
from jaxfmm.flex import m2m, l2l, eval_local, eval_direct
from jaxfmm.kernels.laplace import M2L, M2L_grouped, m2l_theta_setup

from jaxfmm.fem.elements import *
from jaxfmm.fem.elements import (_as_elements, _self_correction_geom, _gather_box_quad,
    _coerce_charges)

__all__ = ["element_farfield_setup", "element_farfield_eval", "element_nearfield_eval",
           "element_selfcorrection", "element_eval", "element_compile"]


def _check_harmonic_scale(verts, p, headroom = 1e6):
    r"""Raise on coordinate scales whose solid harmonics leave the dtype's range.

    The recurrences touch :math:`1/r^{p+2}` at element/leaf distances and
    :math:`r^p` at the domain scale. In f32, meter-scale meshes with nm-class
    elements silently produce inf -> NaN fields (measured: a 5 nm mesh in meters
    is all-NaN at p=4-5, finite in nm units). H is scale-invariant and the
    potential scales linearly, so rescaling to O(1) units is always available —
    the neuralmag ``FMMDemagField`` wrapper does it automatically.
    """
    import numpy as _np
    fi = _np.finfo(_np.asarray(verts).dtype)
    v = _np.asarray(verts[: min(len(verts), 65536)], dtype=_np.float64)
    ext = float((v.max(0) - v.min(0)).max())
    # element scale proxy: the domain extent over the cube root of the count —
    # conservative by construction (true element sizes are never smaller than
    # the densest uniform packing estimate by a large factor)
    l_el = ext / max(float(len(v)) ** (1.0 / 3.0), 1.0)
    if l_el > 0 and l_el ** (p + 2) < headroom / fi.max:
        raise ValueError(
            f"Coordinate scale underflows {fi.dtype}: element-scale distances ~{l_el:.2e} "
            f"raise 1/r^{p + 2} beyond the dtype range (silent inf/NaN fields). Rescale "
            f"coordinates to O(1) units (H is scale-invariant; potentials scale linearly) "
            f"or use float64.")
    if ext ** max(p, 1) > fi.max / headroom:
        raise ValueError(
            f"Coordinate scale overflows {fi.dtype}: domain extent {ext:.2e} raises r^{p} "
            f"beyond the dtype range. Rescale coordinates to O(1) units.")


STORE_QUAD_DEVICE_FRACTION = 0.25   # store_quad=None stores the near quadrature up to this device share


def _store_quad_budget():
    r"""Bytes the stored near quadrature may take under ``store_quad=None`` (unbounded off-GPU or
    when the device reports no memory limit)."""
    try:
        return STORE_QUAD_DEVICE_FRACTION * float(jax.devices()[0].memory_stats()["bytes_limit"])
    except Exception:
        return float("inf")


def element_farfield_setup(verts, tets, tris, eval_pts, p = 4, theta = 0.5, N_max = 128, s = 3,
                           N_max_eval = None, s_eval = None,
                           near_deg = None, far_deg = None, duffy_deg = 4, jit_connectivity = True,
                           near_theta2 = None, store_quad = None, near_width_tiers = 3,
                           m2l_snap_K = None, periodic_axes = (), pbc_lvls = 30,
                           pbc_no_monopole = True, domain = None, L0_boxcen = None,
                           L0_boxlen = None, basis_offsets = None, bloch_k = None):
    r"""
    Geometry-fixed setup for the element-tree FEM far + near field.

    Builds the combined element source tree (tets + tris, box :math:`\supseteq` element),
    the vertex evaluation tree, and the asymmetric interaction lists, and precomputes
    per-element quadrature points and weight*measure factors. Everything here is
    independent of the magnetisation, so it is built once per geometry.

    Parameters
    ----------
    verts : jnp.ndarray, shape (N_vert, 3)
        Mesh vertex coordinates.
    tets : jnp.ndarray or None, shape (N_tet, 4)
        Tetrahedra carrying the P0 volume charge; ``None`` disables the family.
    tris : jnp.ndarray or None, shape (N_tri, 3)
        Boundary triangles carrying the P1 surface charge; ``None`` disables the family.
    eval_pts : jnp.ndarray, shape (N_eval, 3)
        Points where the potential/field is wanted (usually ``verts``).
    p : int, optional
        Multipole order; also the volume quadrature degree (surface uses ``p+1``).
    theta : float, optional
        Multipole acceptance criterion.
    N_max : int, optional
        Maximum elements/points per leaf box.
    s : int, optional
        Tree split schedule parameter.
    far_deg : int, tuple or None, optional
        Far-field quadrature degree(s) as ``(vol, surf)``; an int means
        ``(far_deg, far_deg+1)``. ``None`` (default) uses ``(3 if p==4 else p, p+1)``.
        Degrees are honest (the XG tables deliver exactly the requested degree); the
        MAC suppresses moment-quadrature error, so degrees below ``p`` are often
        accuracy-free — measured: deg 3 at p=6 is 4.7x faster at unchanged error on
        the film proxy.
    near_deg : int or None, optional
        Degree of the separate low-order near-field rule (default ``max(2, p//2)``).
    duffy_deg : int, optional
        Degree of the Duffy rule for the self-correction.
    jit_connectivity : bool, optional
        Use the jit, fixed-buffer connectivity (scales to large N) instead of the eager
        exact-pairs reference.
    store_quad : bool or None, optional
        Store the near-rule quadrature points/weights (``True``) or form them per batch from
        ``verts`` + connectivity (``False``, lazy, §7.24: no ``(N, nq, 3)`` arrays, 1.4-4x
        slower near field at 1e7). ``None`` (default): store unless they would take more than
        ``STORE_QUAD_DEVICE_FRACTION`` of the device memory (always store under Bloch phases,
        which the lazy path does not support).
    near_width_tiers : int, optional
        Occupancy-width tiers per element family in the near field: a pair is billed at the
        width of its source box's tier (front-packed ``idcs[:, :w]``), not at the fullest
        box's family count (exact tiering, :func:`jaxfmm.near._tier_bounds`; 1 = one
        pass per family at the family cap).
    m2l_snap_K : int or None, optional
        Enable the theta-snapped grouped M2L with ``K`` theta bins (open BC only).
    periodic_axes : sequence of int, optional
        Axes (0/1/2) with periodic boundary conditions (default open BC). Near
        images enter the direct/M2L lists (``img``), the infinite lattice tail is
        the hierarchical-doubling operator applied at the source-tree root
        (:func:`jaxfmm.pbc.gen_pbc_op` — same machinery and conventions as the
        point engines; 3D requires a cubic cell).
    pbc_lvls : int, optional
        Cap on the lattice-doubling levels (early-stopped; default 30).
    pbc_no_monopole : bool, optional
        Zero the monopole column of the lattice operator (default True).
    domain : (pmin, extent), optional
        Pin the periodic unit cell (corner + extent; extent scalar or length-3).
        Required in spirit for PBC — without it the source AABB is used with a
        warning. ``L0_boxcen``/``L0_boxlen`` are the historical alternative.
    basis_offsets : sequence of (3,) arrays, optional
        Sublattice basis offsets: every periodic image set is the union over
        ``{0} + basis_offsets`` displacements (see ``jaxfmm.pbc.with_basis_offsets``).

    Returns
    -------
    dict
        Setup dict consumed by :func:`element_farfield_eval`,
        :func:`element_nearfield_eval` and :func:`element_selfcorrection`.

    Notes
    -----
    The far field needs the multipoles integrated to order ``p`` (volume degree ``p``, P0
    density; surface degree ``p+1``, P1 density). The **near field** (non-touching pairs
    at near-box distance, smooth :math:`1/r`) only needs a low-order rule, so it uses a
    separate ``near_deg`` -- far fewer points per element -> much cheaper
    ``eval_direct``.

    The far quadrature is never stored: it is folded into per-element **moment tables**
    (``T_tet`` ``(nbox, cap, (p+1)^2)`` leaf-padded, ``T_tri`` ``(N_tri, 3, (p+1)^2)`` in
    tree order with ``tri_p2m_perm``/``tri_p2m_seg``), so P2M is a charge contraction
    (see :func:`_p2m_tables`). The dict also carries ``coincidence_floor`` semantics for
    the near field: ``element_nearfield_eval`` disables the rounding-coincidence guard
    when evaluating at the mesh nodes (``self_correct``), where quadrature sources can
    never land on a target; set ``s["coincidence_floor"]`` to override.
    """
    tets, tris = _as_elements(tets, tris)
    _check_harmonic_scale(verts, p)
    if near_deg is None:
        # Default (3, 4): with HONEST-degree tabulated rules (quad_tables) this reproduces
        # the historical effective near degrees — the old max(2, p//2) formula leaned on
        # the collapsed rules' odd-degree over-delivery (a deg-2 request was a deg-3 rule).
        # Measured (film 1e6): (3,4) = 5.5e-3 vs the historical 5.9e-3, time within noise.
        near_deg = (3, 4)
    # near_deg: int (surface uses deg+1, mirroring the far rules' p vs p+1) or a
    # (vol_deg, surf_deg) tuple — the two families' near demands differ by geometry
    # (bulk is far-limited: vol deg 1 is ~free; films are surface-limited)
    nd_vol, nd_surf = near_deg if isinstance(near_deg, (tuple, list)) \
        else (near_deg, near_deg + 1)
    # Far volume rule: capped at deg 3 for p=4 — the tensor rule jumps 8->27 pts at deg 4
    # while the top moment's quadrature error is buried under MAC truncation (measured
    # 2026-09-02: rel err IDENTICAL to 4 digits on film AND cube at p=4, far field 1.95x
    # faster). p>=5 keeps the exact rule until tested.
    # far rules: default (3 if p==4 else p, p+1); far_deg overrides — int means
    # (far_deg, far_deg+1), mirroring near_deg. The MAC suppresses moment-quadrature
    # error, so capping the far degree below p is often accuracy-free: the p=4 deg-3
    # cap is the measured-identical default (8 vs 27 pts), and far_deg=(3, 4) at p=6
    # measured 4.7x total on the eigensolver-class operator at unchanged error
    # (3.26e-3, H100 film proxy) — now reachable here instead of by patch.
    if far_deg is None:
        fd_vol, fd_surf = (3 if p == 4 else p), p + 1
    else:
        fd_vol, fd_surf = far_deg if isinstance(far_deg, (tuple, list)) \
            else (far_deg, far_deg + 1)
    bvt, wvt = simplex_quad(3, fd_vol)                             # far volume rule (rho is P0)
    bst, wst = simplex_quad(2, fd_surf)                            # far surface rule (sigma is P1)
    bvn, wvn = simplex_quad(3, nd_vol)                              # near volume rule (P0)
    bsn, wsn = simplex_quad(2, nd_surf)                             # near surface rule (P1)
    if near_theta2 is not None:                                    # tiered near field: 1-point
        bvc, wvc = simplex_quad(3, 0)                              # centroid rule for the
        bsc, wsc = simplex_quad(2, 0)                              # box-disjoint outer tier
    n_tet, n_tri = int(tets.shape[0]), int(tris.shape[0])
    if store_quad is None:
        # auto: store unless it would crowd the device; decided from the sizes BEFORE forming
        # anything — forming (N, nq, 3) only to drop it left 16-21 GB holes in the allocator
        # pool that the eval's temporaries could not use (R9 film 1e7: OOM at 48 of 89 GB)
        _nq = n_tet * len(bvn) + n_tri * len(bsn) + (n_tet + n_tri if near_theta2 is not None else 0)
        _bloch = len(periodic_axes) > 0 and bloch_k is not None and np.any(np.asarray(bloch_k) != 0.0)
        store_quad = bool(_bloch) or 4 * _nq * verts.dtype.itemsize <= _store_quad_budget()
    if store_quad:
        tet_qpos_n, tet_vol_n = element_quad_points(verts, tets, bvn)
        tri_qpos_n, tri_area_n = element_quad_points(verts, tris, bsn)
        wm_tet_n = wvn[None, :] * tet_vol_n[:, None]
        wm_tri_n = wsn[None, :] * tri_area_n[:, None]
        if near_theta2 is not None:
            tet_qpos_c, _ = element_quad_points(verts, tets, bvc)
            tri_qpos_c, _ = element_quad_points(verts, tris, bsc)
            wm_tet_c = wvc[None, :] * tet_vol_n[:, None]
            wm_tri_c = wsc[None, :] * tri_area_n[:, None]
    else:                              # lazy: the self-correction forms its points per slab
        tet_qpos_n = wm_tet_n = tri_qpos_n = wm_tri_n = None
    # Duffy self-correction is meaningful only when evaluating at the mesh nodes themselves
    # (touching pairs are (element, own node)); skip it when eval_pts are some other points.
    # The one other structured target set -- the near-rule points of strayfield's diff mode --
    # gets its (energy-level) element self-block from elements._self_energy_geom instead.
    self_correct = eval_pts is verts
    if self_correct:                                              # geometry-fixed precompute
        corr_t = _self_correction_geom(verts, tets, duffy_deg, tet_qpos_n, wm_tet_n, jnp.asarray(bvn),
                                       w_n=jnp.asarray(wvn))
        corr_s = _self_correction_geom(verts, tris, duffy_deg, tri_qpos_n, wm_tri_n, jnp.asarray(bsn),
                                       w_n=jnp.asarray(wsn))
    else:
        corr_t = corr_s = None

    # Tree + connectivity are ALWAYS built in f64: MAC decisions from f32 box geometry
    # mis-route borderline pairs, a p-, theta- and quadrature-independent error floor
    # (measured 1.6e-2 H error at 120k tets / 3.8e-2 at 1e6 for generic m, vs 1.6e-4 with
    # f64-built lists and the identical f32 eval; grows with N). Setup-time cost only —
    # the box geometry is cast back to the input dtype for the eval pipeline below.
    _gdt = verts.dtype
    _build64 = _gdt != jnp.float64
    _x64 = jax.enable_x64() if _build64 else None
    try:
        if _build64:
            _x64.__enter__()
            verts_b = jnp.asarray(np.asarray(verts, np.float64))
            eval_b = verts_b if eval_pts is verts else jnp.asarray(np.asarray(eval_pts, np.float64))
        else:
            verts_b, eval_b = verts, eval_pts
        cT, mnT, mxT = element_aabb(verts_b, tets)
        cS, mnS, mxS = element_aabb(verts_b, tris)
        coms = jnp.concatenate([cT, cS]); emin = jnp.concatenate([mnT, mnS]); emax = jnp.concatenate([mxT, mxS])
        # Per-tree sizing: the SOURCE tree (elements) and the EVAL tree (points) have
        # independent counts and independent row-width economics (a P2P tile is
        # src_row x eval_row) — N_max_eval / s_eval default to the source values but can
        # differ. Under "auto" each tree gets its own fractional-depth schedule (as in
        # gen_hierarchy): leaf occupancy lands on its cap instead of overshooting by up
        # to 2x, which smooths the tree-fill sawtooth over N at fixed integer s.
        _nme = N_max if N_max_eval is None else N_max_eval
        _se = s if s_eval is None else s_eval
        from jaxfmm.trees.flexible import split_schedule, lane_schedule
        s_src = split_schedule(coms.shape[0], N_max, 2) if s == "auto" else s
        # EVAL leaves are the P2P row axis, billed in 32/64-lane multiples: lane_schedule
        # admits an odd root slab count so the occupancy lands within ~10% of N_max_eval
        # (pow2 leaf counts sit anywhere in (N_max/2, N_max], e.g. 39 of 64 on a 2.5e6-vertex
        # film = 39% of the near field paid for empty lanes). Source rows are the reduction
        # axis (cost linear in width) and keep the plain pow2 schedule.
        _se = lane_schedule(eval_b.shape[0], _nme, 2) if _se == "auto" else _se
        src_idcs, _, src_bc, src_bl = balanced_tree_elements(coms, emin, emax, N_max=N_max, s=s_src)
        eval_idcs, eval_rev, eval_bc, eval_bl = balanced_tree(eval_b, N_max=_nme, s=_se)
        src_max_l, eval_max_l = len(src_bc) - 1, len(eval_bc) - 1

        from jaxfmm.util import parse_domain
        L0_boxcen, L0_boxlen = parse_domain(domain, L0_boxcen, L0_boxlen, dtype=verts_b.dtype)
        periodic = len(periodic_axes) > 0
        if periodic and L0_boxlen is None:
            import warnings
            warnings.warn("element_farfield_setup: periodic_axes without a pinned cell — "
                          "using the source AABB as the periodic cell. Pass "
                          "domain=(pmin, extent) to pin it.", stacklevel=2)
        # pin the unit cell as the (virtual) root of BOTH trees: the image displacements
        # and the lattice operator are generated from the SOURCE root box, and a shared
        # root keeps the dual traversal's wrap geometry consistent
        if L0_boxcen is not None:
            src_bc[0] = L0_boxcen[None, :].astype(src_bc[0].dtype)
            eval_bc[0] = L0_boxcen[None, :].astype(eval_bc[0].dtype)
        if L0_boxlen is not None:
            src_bl[0] = L0_boxlen[None, :].astype(src_bl[0].dtype)
            eval_bl[0] = L0_boxlen[None, :].astype(eval_bl[0].dtype)
        if periodic and len(periodic_axes) == 3 and not jnp.all(jnp.isclose(src_bl[0], src_bl[0][0, 0])):
            raise ValueError("3D PBC are currently only supported for cubic level zero boxes. "
                             "Pin a cubic cell with domain=(pmin, extent).")

        if jit_connectivity:    # jit, fixed-buffer (auto-sized) -- jit-compilable, scales to large N
            conn = AdaptiveConnectivity(theta=theta, s=s_src, no_cross_level=False,
                                        periodic_axes=periodic_axes, basis_offsets=basis_offsets)
            sbc, ebc, src_ofs, trg_ofs, mpl_cnct, dir_cnct, lvl_info, img = conn(src_bc, src_bl, eval_bc, eval_bl)
        else:                   # eager reference (exact pairs, no buffers)
            sbc, ebc, src_ofs, trg_ofs, mpl_cnct, dir_cnct, lvl_info, img = \
                gen_connectivity(src_bc, src_bl, eval_bc, eval_bl, theta, s_src, no_cross_level=False,
                                 periodic_axes=periodic_axes, basis_offsets=basis_offsets)
    finally:
        if _build64:
            _x64.__exit__(None, None, None)
    if _build64:
        # back to session dtypes; int64 padding sentinels are iinfo.max, so clip-cast.
        # The clip MUST run in numpy: jax under x64-off truncates int64 operands to int32
        # (wrapping the sentinels to -1) before any jnp op sees them.
        _imax = np.iinfo(np.int32).max
        _di = lambda a: jnp.asarray(np.minimum(np.asarray(a), _imax).astype(np.int32)) \
            if a.dtype == jnp.int64 else a
        _df = lambda a: a.astype(_gdt) if a.dtype == jnp.float64 else a
        _dc = lambda x: [( _di(a) if jnp.issubdtype(a.dtype, jnp.integer) else _df(a)) for a in x] \
            if isinstance(x, list) else (_di(x) if jnp.issubdtype(x.dtype, jnp.integer) else _df(x))
        src_idcs, eval_idcs, eval_rev = _di(src_idcs), _di(eval_idcs), _di(eval_rev)
        mpl_cnct, dir_cnct = _dc(mpl_cnct), _dc(dir_cnct)
        sbc, ebc, img = _dc(sbc), _dc(ebc), _df(img)
        src_bl = _dc(src_bl)
        coms, eval_pts = _df(coms), (verts if eval_pts is verts else _df(eval_b))
    if periodic and len(lvl_info) == 1:
        raise ValueError("element_farfield_setup: no well-separated level under PBC — "
                         "the lattice tail would be dropped. Increase the problem size "
                         "or lower N_max.")
    PBC_op = None
    img_phase = None
    if periodic:
        from jaxfmm.pbc import gen_pbc_op
        PBC_op, _ = gen_pbc_op(src_bl[0], theta, periodic_axes, p, pbc_lvls,
                               pbc_no_monopole, basis_offsets=basis_offsets,
                               bloch_k=bloch_k)
        if pbc_lvls < 0:        # only the non-well-separated level-0 images, no tail
            PBC_op = jnp.zeros(((p + 1) ** 2, (p + 1) ** 2), PBC_op.dtype)
        if bloch_k is not None and np.any(np.asarray(bloch_k) != 0.0):
            # per-image Bloch phase for the tree-level machinery (P2P + M2L)
            img_phase = jnp.exp(1j * (img @ jnp.asarray(bloch_k, img.dtype).reshape(3)))
    # idcs are aggregated to two (usually equal) levels: the finest mpl level (S2M/L2P, can be
    # coarser than the leaves when whole subtrees are well-separated) and the dir level (near field).
    def _reduce(idcs, rev, pts, max_l, lvl, sched):
        return (idcs, rev) if lvl >= max_l else reduce_max_lvl(pts, idcs, sched, max_l, lvl)
    msl, mel = lvl_info[-2][1], lvl_info[-2][0]               # far (multipole) levels
    dsl, del_ = lvl_info[-1][1], lvl_info[-1][0]              # near (direct) levels
    src_far, _ = _reduce(src_idcs, None, coms, src_max_l, msl, s_src)
    eval_far, eval_rev_far = _reduce(eval_idcs, eval_rev, eval_pts, eval_max_l, mel, _se)
    src_near, _ = _reduce(src_idcs, None, coms, src_max_l, dsl, s_src)
    eval_near, eval_rev_near = _reduce(eval_idcs, eval_rev, eval_pts, eval_max_l, del_, _se)
    # Per-FAMILY near source indices + pair lists: a combined row reserves cap slots for
    # BOTH families, so a flat near pass would process a cap*nq_tri-wide tri block per
    # pair even when boxes hold few/no tris. Split into front-packed per-family index
    # arrays with TIGHT caps + per-family pair lists (tri-dead pairs dropped); the near
    # eval runs one plain eval_direct per family.
    n_el_tot = int(tets.shape[0]) + int(tris.shape[0])
    sn = np.asarray(src_near)
    dcn = np.asarray(dir_cnct)
    # sources in dir_cnct are image-ENCODED: src = box + nbox*img_idx (img 0 = the real
    # box); the padding sentinel is n_img*nbox. Keep encoded ids (eval_direct decodes
    # them against ``img``), so live/family tests must use the DECODED box (src % nbox).
    n_img_nf = max(int(np.asarray(img).shape[0]), 1)
    dcn = dcn[(dcn[:, 0] < np.asarray(eval_near).shape[0]) & (dcn[:, 1] < n_img_nf * sn.shape[0])]
    if near_theta2 is not None:
        # Tier split of the near pairs. Coarse (centroid-rule) tier = box spheres STRICTLY
        # disjoint — guarantees no touching-element pair can land there, which keeps the
        # Duffy corrections (built against the FINE near rule) consistent — plus the
        # nested MAC at near_theta2, so the 1-pt quadrature error is separation-bounded.
        _sb = np.asarray(sbc, np.float64)[src_ofs[dsl]:src_ofs[dsl + 1]]
        _eb = np.asarray(ebc, np.float64)[trg_ofs[del_]:trg_ofs[del_ + 1]]
        _slv = np.asarray(src_bl[dsl], np.float64)
        _elv = np.asarray(eval_bl[del_], np.float64)
        _im = np.asarray(img, np.float64)
        _im = _im if _im.size else np.zeros((1, 3))
        _bx, _ix = dcn[:, 1] % sn.shape[0], dcn[:, 1] // sn.shape[0]
        _dc_ = _eb[dcn[:, 0]] - (_sb[_bx] + _im[_ix])
        _d = np.linalg.norm(_dc_, axis=1)
        _hl = (_elv[dcn[:, 0]] + _slv[_bx]) / 2                    # per-axis AABB half-length sums
        # AABB-disjoint with a STRICT gap in some axis: elements live inside the AABBs, so a
        # positive gap guarantees no shared node (face-TOUCHING AABBs can share a vertex on
        # the interface plane — the relative margin keeps 1-ULP cases out of the coarse tier)
        _gap = np.any(np.abs(_dc_) > _hl * (1 + 1e-9), axis=1)
        _r1 = np.linalg.norm(_elv[dcn[:, 0]], axis=1) / 2
        _r2 = np.linalg.norm(_slv[_bx], axis=1) / 2
        _R, _r = np.maximum(_r1, _r2), np.minimum(_r1, _r2)
        _coarse = _gap & (_R + near_theta2 * _r <= near_theta2 * _d)
    else:
        _coarse = np.zeros(dcn.shape[0], bool)
    # Occupancy-width tiers: the family's pairs are split by their source box's family count so
    # each pass's tile is as wide as its tier's fullest box (front-packed idcs[:, :w]), not the
    # whole family's (measured 2026-10-09: the film's tri pass billed 38 % useful slots at one
    # width; 3 tiers recover most of it).
    from jaxfmm.near import _tier_bounds
    fam_near = {}
    for fam, lo, hi in (("tet", 0, int(tets.shape[0])), ("tri", int(tets.shape[0]), n_el_tot)):
        mask = (sn >= lo) & (sn < hi)
        cnt = mask.sum(1)
        cap = max(int(cnt.max()), 1)
        rows, cols = np.nonzero(mask)
        pos = mask.cumsum(1) - 1                                   # front-pack: position within row
        packed = np.full((sn.shape[0], cap), n_el_tot, sn.dtype)   # sentinel -> zero charge
        packed[rows, pos[rows, cols]] = sn[rows, cols]
        nb = sn.shape[0]
        occ = cnt[dcn[:, 1] % nb]
        tiers = []
        for sel in ((occ > 0) & ~_coarse, (occ > 0) & _coarse):      # fine rule, centroid tier
            d_k, o_k, lo_w, lst = dcn[sel], occ[sel], 0, []
            for w in (_tier_bounds(o_k, near_width_tiers) if len(d_k) else []):
                d_t = d_k[(o_k > lo_w) & (o_k <= w)]
                # COMPACT the tier to the source boxes it touches: its pass gathers
                # (len(boxes), w, nq, 3) instead of (nbox, w, nq, 3) — summed over tiers
                # below the single full-width gather (R9 film 1e7: 3 x 19 GB otherwise)
                boxes = np.unique(d_t[:, 1] % nb)
                part = (d_t[:, 1] // nb) * len(boxes) + np.searchsorted(boxes, d_t[:, 1] % nb)
                lst.append((w, jnp.asarray(np.stack([d_t[:, 0], part], 1).astype(d_t.dtype)),
                            jnp.asarray(boxes.astype(np.int32))))
                lo_w = w
            tiers.append(lst)
        fam_near[fam] = (jnp.asarray(packed), tiers[0], tiers[1])
    # hardware-adaptive default budget for the evals (0.3.1 semantics: peak-transient
    # bytes), anchored to the widest near-field source row — mirrors gen_hierarchy's
    # auto default; an unbatched element eval at production sizes requests 100s of GiB
    try:
        from jaxfmm.near import auto_mem_limit
        _w = max(fam_near["tet"][0].shape[1] * (max(len(bvn), 1) if n_tet else 1),
                 fam_near["tri"][0].shape[1] * (max(len(bsn), 1) if n_tri else 1),
                 int(src_far.shape[1]))
        mem_limit_auto = auto_mem_limit(int(_w), verts.dtype.itemsize)
        print(f"[jaxfmm] element mem_limit auto: {mem_limit_auto/2**30:.1f} GiB peak-transient "
              f"budget (pass mem_limit=<bytes> to the evals to override)")
    except Exception:
        mem_limit_auto = jnp.inf
    unqs, invs = precompute_combs(p)
    # Far field: per-element MOMENT TABLES instead of stored far quadrature. The moments
    # are linear in the charges, so the leaf expansion is sum_e rho_e T_e + sum_ek sigma_ek
    # T_ek with geometry-only tables (exact up to rounding) — P2M becomes a bandwidth-bound
    # contraction instead of a per-step regular-basis evaluation at every quad point (which
    # was 83% of the far field / 45% of the whole step at film 1e7 p3, A100 f32; 58 -> 1.4 ms).
    # Storage is (p+1)^2 per tet and 3(p+1)^2 per tri vs 3+1 floats per far quad point:
    # smaller from p>=4 (deg-p rules), ~1.5x larger at p3 for tris only.
    T_tet, tet_p2m_idcs, T_tri, tri_p2m_perm, tri_p2m_seg = _p2m_tables(
        sbc[src_ofs[msl]:src_ofs[msl + 1]], src_far, verts, tets, tris, (bvt, wvt), (bst, wst), p)
    m2l_grp = None
    if m2l_snap_K:                                                # theta-snapped grouped M2L (opt-in)
        if int(np.asarray(img).shape[1]) != 0:
            raise ValueError("m2l_snap_K: the theta-snapped M2L does not support PBC images")
        m2l_grp = m2l_theta_setup(mpl_cnct, sbc, ebc, unqs, invs, K=m2l_snap_K)
    tiered = {} if near_theta2 is None else dict(
        bary_tri_c=jnp.asarray(bsc),
        **(dict(tet_qpos_c=tet_qpos_c, wm_tet_c=wm_tet_c,
                tri_qpos_c=tri_qpos_c, wm_tri_c=wm_tri_c) if store_quad else {}))
    if not store_quad:                       # lazy near sources: rule constants + verts only
        tiered.update(src_verts=verts, bary_tet_n=jnp.asarray(bvn), w_tet_n=jnp.asarray(wvn),
                      w_tri_n=jnp.asarray(wsn),
                      **({"w_tet_c": jnp.asarray(wvc), "w_tri_c": jnp.asarray(wsc),
                          "bary_tet_c": jnp.asarray(bvc)} if near_theta2 is not None else {}))
    return dict(**tiered, p=p, N_tet=int(tets.shape[0]), m2l_grp=m2l_grp,
                T_tet=T_tet, tet_p2m_idcs=tet_p2m_idcs, T_tri=T_tri,
                tri_p2m_perm=tri_p2m_perm, tri_p2m_seg=tri_p2m_seg,
                bary_tri_n=jnp.asarray(bsn), store_quad=store_quad,
                **(dict(tet_qpos_n=tet_qpos_n, wm_tet_n=wm_tet_n,
                        tri_qpos_n=tri_qpos_n, wm_tri_n=wm_tri_n) if store_quad else {}),
                tets=tets, tris=tris, corr_t=corr_t, corr_s=corr_s, self_correct=self_correct,
                src_idcs=src_far, sbc=sbc, src_ofs=src_ofs,
                eval_pts=eval_pts, eval_idcs=eval_far, eval_rev=eval_rev_far, ebc=ebc, trg_ofs=trg_ofs,
                src_idcs_near=src_near, eval_idcs_near=eval_near, eval_rev_near=eval_rev_near,
                tet_idcs_near=fam_near["tet"][0], tri_idcs_near=fam_near["tri"][0],
                **{f"{f}_{k}{sfx}": tuple(t[i] for t in fam_near[f][j])
                   for f in ("tet", "tri") for j, sfx in ((1, ""), (2, "_c"))
                   for k, i in (("near_w", 0), ("dir_cnct", 1), ("near_boxes", 2))},
                mpl_cnct=mpl_cnct, dir_cnct=dir_cnct, lvl_info=lvl_info, img=img, unqs=unqs, invs=invs,
                PBC_op=PBC_op, img_phase=img_phase, mem_limit_auto=mem_limit_auto)



def _p2m_tables(leaf_bc, src_idcs, verts, tets, tris, rule_tet, rule_tri, p, budget = 2**30):
    r"""
    Per-element far-field moment tables about the owning leaf centre.

    ``T_tet[e, K] = sum_q wm_tet[e, q] R_K(x_eq - c_leaf(e))`` and
    ``T_tri[e, k, K] = sum_q bary[q, k] wm_tri[e, q] R_K(x_eq - c_leaf(e))``, so the leaf
    multipoles are ``sum_e rho_e T_tet[e] + sum_ek sigma_ek T_tri[e, k]`` — the far
    quadrature rule is folded into geometry-only tables and never evaluated per step.
    The far quadrature points are formed per chunk and every table is written straight into
    its final (leaf-padded / tree-ordered) layout: the setup peak is the tables themselves
    plus ``budget`` (an all-element ``(N, nq_far, 3)`` array was 20 GB at p=11, 1e7 tets).

    Parameters
    ----------
    rule_tet, rule_tri : tuple of ndarray
        Far rules ``(bary, w)`` from :func:`simplex_quad`.

    Returns
    -------
    T_tet : (nbox, cap_tet, (p+1)^2) array
        Leaf-padded (front-packed) tet table, zero in unused slots.
    tet_p2m_idcs : (nbox, cap_tet) int array
        Tet index per slot (``N_tet`` sentinel -> zero charge under a fill gather).
    T_tri : (N_tri, 3, (p+1)^2) array
        Tri table in leaf (tree) order.
    tri_p2m_perm, tri_p2m_seg : (N_tri,) int arrays
        Tri index and owning leaf per tree-ordered row (``seg`` sorted, for segment_sum).
    """
    si = np.asarray(src_idcs); nbox, cap = si.shape
    n_tet, n_tri = int(tets.shape[0]), int(tris.shape[0])
    box = np.repeat(np.arange(nbox, dtype=np.int32), cap).reshape(nbox, cap)
    valid = si < n_tet + n_tri
    bc = jnp.asarray(leaf_bc, verts.dtype)
    # tets: front-pack per leaf at the TET cap (the mixed row cap reserves tri slots too)
    mt = valid & (si < n_tet)
    cnt = mt.sum(1); cap_t = int(cnt.max()) if n_tet else 0
    rows, cols = np.nonzero(mt)
    packed = np.full((nbox, cap_t), n_tet, np.int32)
    packed[rows, (mt.cumsum(1) - 1)[rows, cols]] = si[rows, cols]
    K = (p + 1)**2
    T_tet = _table_rows(verts, tets, jnp.asarray(packed.reshape(-1)), jnp.repeat(bc, cap_t, axis=0),
                        *rule_tet, p, budget, (nbox, cap_t, K))
    # tris: tree order (box-major = sorted segments), one row per tri
    ms = valid & (si >= n_tet)
    perm, seg = (si[ms] - n_tet).astype(np.int32), box[ms]
    T_tri = _table_rows(verts, tris, jnp.asarray(perm), bc[seg], *rule_tri, p, budget,
                        (len(perm), 3, K))
    return T_tet, jnp.asarray(packed), T_tri, jnp.asarray(perm), jnp.asarray(seg)


def _table_rows(verts, elems, rows, cen, bary, w, p, budget, shape):
    r"""Moment-table rows for element ids ``rows`` (``>= len(elems)``: zero row) about ``cen``,
    reshaped to ``shape``; chunks of ``budget`` bytes of regular-basis transient."""
    nq, K = int(bary.shape[0]), (p + 1)**2
    n = int(rows.shape[0])
    if n == 0 or elems.shape[0] == 0:
        return jnp.zeros(shape, verts.dtype)
    chunk = min(max(int(budget // (3 * max(nq, 1) * K * verts.dtype.itemsize)), 1), n)
    return _table_rows_jit(verts, elems, rows, cen, jnp.asarray(bary, verts.dtype),
                           jnp.asarray(w, verts.dtype), p, chunk, shape)


@partial(jax.jit, static_argnames=("p", "chunk", "shape"))
def _table_rows_jit(verts, elems, rows, cen, bary, w, p, chunk, shape):
    from jaxfmm.kernels.laplace.basis import eval_regular_basis
    from jaxfmm.fem.quadrature import _bary_points, element_measure
    n, tri = rows.shape[0], elems.shape[1] == 3
    live = rows < elems.shape[0]
    rid = jnp.where(live, rows, 0)

    def block(r, c, v):
        ev = verts[elems[r]]                                       # (chunk, k, 3)
        wm = w[None, :] * jnp.where(v, element_measure(ev), 0.0)[:, None]
        R = eval_regular_basis(_bary_points(bary, ev) - c[:, None, :], p)
        return jnp.einsum("qk,eq,eqK->ekK", bary, wm, R) if tri else jnp.einsum("eq,eqK->eK", wm, R)

    def body(i, out):                  # last chunk overlaps its predecessor: idempotent rewrite
        a = jnp.minimum(i * chunk, n - chunk)
        sl = lambda x: jax.lax.dynamic_slice_in_dim(x, a, chunk)
        return jax.lax.dynamic_update_slice_in_dim(out, block(sl(rid), sl(cen), sl(live)), a, 0)

    # result dtype as the table math yields it (f32 geometry under x64 promotes to f64 in the basis)
    dt = jax.eval_shape(block, rid[:chunk], cen[:chunk], live[:chunk]).dtype
    out = jnp.zeros((n,) + ((3,) if tri else ()) + ((p + 1)**2,), dt)
    return jax.lax.fori_loop(0, -(-n // chunk), body, out).reshape(shape)


def element_farfield_eval(s, vol_chrg, surf_nodal, field = False, mem_limit = None):
    r"""
    Evaluate the FEM far field (S2M -> M2M -> M2L -> L2L -> L2P) for current charges.

    Parameters
    ----------
    s : dict
        Setup dict from :func:`element_farfield_setup`.
    vol_chrg : jnp.ndarray or None, shape (N_tet,)
        P0 volume charge density per tet (e.g. :math:`-\nabla\!\cdot M`).
    surf_nodal : jnp.ndarray or None, shape (N_tri, 3)
        P1 nodal surface charge per boundary triangle (:math:`M\cdot n`).
    field : bool, optional
        Return :math:`H=-\nabla u` (vector) instead of the potential ``u``.
    mem_limit : float, optional
        Byte budget for the batched transforms.

    Returns
    -------
    jnp.ndarray, shape (N_eval,) or (N_eval, 3)
        Far-field potential (or field) at ``eval_pts`` (original order).

    Notes
    -----
    Only the far (well-separated) part: the near field is added separately. The inner
    transforms are jit-cached, so warm calls recompute only the charge-dependent work.
    """
    if mem_limit is None:
        mem_limit = s.get("mem_limit_auto", jnp.inf)
    vol_chrg, surf_nodal = _coerce_charges(s, vol_chrg, surf_nodal)
    if len(s["lvl_info"]) == 1:                                  # no well-separated pairs -> no far field
        return jnp.zeros((s["eval_pts"].shape[0], 3) if field else (s["eval_pts"].shape[0],))
    p, msl, mel = s["p"], s["lvl_info"][-2][1], s["lvl_info"][-2][0]
    # 0.3.1 budget semantics: the far-field stages are transient-priced minor stages ->
    # batch them at the derived row working set (jaxfmm.near.ws_limit), not the budget.
    import numpy as _np
    from jaxfmm.near import ws_limit as _wsl, m2l_budget
    ws = _wsl(mem_limit, s["src_idcs"].shape[1]) if _np.isfinite(mem_limit) else mem_limit
    ws_m2l = m2l_budget(ws, mem_limit, p, s["mpl_cnct"].shape[0], s["T_tet"].dtype.itemsize)
    # P2M from the geometry-only moment tables (see _p2m_tables): tets contract against
    # the leaf-padded table, tris are a tree-ordered segment sum (measured layout optimum:
    # 1.4 ms vs 5.8 ms both padded / 2.2 ms both segment-summed, film 1e7).
    nbox = s["src_idcs"].shape[0]
    rho_pad = vol_chrg.at[s["tet_p2m_idcs"]].get(mode="fill", fill_value=0.0)
    C = jnp.einsum("bc,bcK->bK", rho_pad, s["T_tet"]) \
      + jax.ops.segment_sum(jnp.einsum("ek,ekK->eK", surf_nodal[s["tri_p2m_perm"]], s["T_tri"]),
                            s["tri_p2m_seg"], nbox, indices_are_sorted=True)
    C = m2m(C, s["sbc"], s["src_ofs"], msl, s["unqs"], s["invs"], ws)               # M2M up the source tree
    if s.get("img_phase") is not None:                           # Bloch: plain complex M2L only
        L = M2L(C, s["sbc"], s["ebc"], s["mpl_cnct"], s["unqs"], s["invs"], s["img"], ws,
                img_phase=s["img_phase"])
    elif s.get("m2l_grp") is not None:                           # theta-snapped grouped M2L (no PBC)
        L = M2L_grouped(C, s["sbc"], s["ebc"], s["m2l_grp"], ws)
    elif p <= 4 and C.dtype == jnp.float64:                      # mixed M2L below the noise/truncation
        # crossover — F64-ONLY, same dtype gate as the point engine: in f32 the mixed
        # rotations are pure conversion overhead (+31% at p<=3, and the 1 GiB batch floor
        # blows the p=4 far field up ~8x — the 2026-09-02 sweep's p=4 anomaly)
        from jaxfmm.kernels.laplace.transf import M2L_mixed
        L = M2L_mixed(C, s["sbc"], s["ebc"], s["mpl_cnct"], s["unqs"], s["invs"], s["img"], ws_m2l)
    else:
        L = M2L(C, s["sbc"], s["ebc"], s["mpl_cnct"], s["unqs"], s["invs"], s["img"], ws_m2l)
    if s.get("PBC_op") is not None:                              # infinite lattice tail at the root
        L = L.at[0].add((s["PBC_op"] @ C[0]).astype(L.dtype))
    L = l2l(L, s["ebc"], s["trg_ofs"], s["unqs"], s["invs"], s["lvl_info"], ws)   # L2L down the eval tree

    padded_eval = s["eval_pts"].at[s["eval_idcs"]].get(mode="fill", fill_value=0.0)
    leaf_ebc = s["ebc"][s["trg_ofs"][mel]:s["trg_ofs"][mel + 1]]
    return eval_local(L[s["trg_ofs"][mel]:s["trg_ofs"][mel + 1]], padded_eval, s["eval_rev"], leaf_ebc, p, field, ws)


def element_nearfield_eval(s, vol_chrg, surf_nodal, field = False, mem_limit = None,
                           kernel_dtype = None, kernel_scale = None):
    r"""
    Evaluate the FEM near field over the direct (non-well-separated) element pairs.

    Bulk pass: each near source element is represented by its quadrature point-charges and
    the near potential/field is a direct :math:`\sum_q q_q/|x_q-y|` over the ``dir_cnct``
    pairs -- i.e. :func:`jaxfmm.flex.eval_direct` with the element quadrature points as
    the source points.

    Parameters
    ----------
    s : dict
        Setup dict from :func:`element_farfield_setup`.
    vol_chrg : jnp.ndarray or None, shape (N_tet,)
        P0 volume charge density per tet.
    surf_nodal : jnp.ndarray or None, shape (N_tri, 3)
        P1 nodal surface charge per boundary triangle.
    field : bool, optional
        Return the vector field instead of the potential.
    mem_limit : float, optional
        Byte budget for the batched direct evaluation.

    Returns
    -------
    jnp.ndarray, shape (N_eval,) or (N_eval, 3)
        Near-field potential (or field) at ``eval_pts`` (original order).

    Notes
    -----
    This is exact for non-touching pairs (smooth :math:`1/r`, guaranteed by box
    :math:`\supseteq` element); the singular self / touching pairs (``y`` a node of the
    element) still get a plain-Gauss value here and need a Duffy/Sauter-Schwab correction
    (added separately).
    """
    if mem_limit is None:
        mem_limit = s.get("mem_limit_auto", jnp.inf)
    vol_chrg, surf_nodal = _coerce_charges(s, vol_chrg, surf_nodal)
    if jnp.asarray(s["dir_cnct"]).size == 0:                     # no near pairs
        return jnp.zeros((s["eval_pts"].shape[0], 3) if field else (s["eval_pts"].shape[0],))
    padded_eval = s["eval_pts"].at[s["eval_idcs_near"]].get(mode="fill", fill_value=0.0)
    # f32 box-local near kernel, f64 accumulate (mixed=True): the f32 noise sits orders of
    # magnitude below the low-order near-quadrature error, so mixed precision is free here.
    dsl, del_ = s["lvl_info"][-1][1], s["lvl_info"][-1][0]
    cen_s = s["sbc"][s["src_ofs"][dsl]:s["src_ofs"][dsl + 1]]
    cen_e = s["ebc"][s["trg_ofs"][del_]:s["trg_ofs"][del_ + 1]]
    # Coincidence floor (see eval_direct): OFF when evaluating at the mesh nodes — quad
    # sources never land on a vertex (>= 0.096 h for every simplex rule used), so only the
    # exact d2 > 0 guard is needed; 40% of the near field at 1e7. Any other eval set (e.g.
    # the diff mode's own quadrature points) can coincide with sources up to folding
    # rounding and keeps the cheap per-pair bound.
    _cf = s.get("coincidence_floor", False if s["self_correct"] else "pair")
    # One TIGHT pass per element family (see setup: per-family front-packed caps +
    # tri-dead pairs dropped) instead of one pass over the cap-slotted tet+tri concat;
    # falls back to the combined pass for setups predating the split keys.
    if "tet_idcs_near" in s:
        out = 0.0
        if not s.get("store_quad", True):
            # LAZY near sources (store_quad=False): quadrature points/charges are formed
            # per batch inside eval_direct's scan from verts + element connectivity —
            # nothing (N, nq, 3)-sized is stored. Fine + coarse tiers share the closures.
            if s.get("img_phase") is not None:
                raise NotImplementedError("store_quad=False under Bloch-phased PBC images "
                                          "is not implemented (use store_quad=True)")
            _vl, _n_el = s["src_verts"], int(s["N_tet"]) + int(s["tris"].shape[0])

            def _tet_fn(bary, w, packed, off):
                bary = jnp.asarray(bary, _vl.dtype); w = jnp.asarray(w, _vl.dtype)
                def fn(partner):
                    pk = packed.at[partner].get(mode="fill", fill_value=_n_el)
                    loc = pk - off
                    valid = (loc >= 0) & (loc < s["tets"].shape[0])
                    locc = jnp.clip(loc, 0, max(int(s["tets"].shape[0]) - 1, 0))
                    ev = _vl[s["tets"][locc]]                          # (b, cap, 4, 3)
                    qp = jnp.einsum("qk,bckd->bcqd", bary, ev,
                                    precision=jax.lax.Precision.HIGHEST)
                    vol = jnp.abs(jnp.linalg.det(ev[..., 1:, :] - ev[..., :1, :])) / 6
                    chg = (vol_chrg[locc] * vol * valid)[..., None] * w[None, None, :]
                    return qp.reshape(qp.shape[0], -1, 3), chg.reshape(chg.shape[0], -1)
                return fn

            def _tri_fn(bary, w, packed, off):
                bary = jnp.asarray(bary, _vl.dtype); w = jnp.asarray(w, _vl.dtype)
                def fn(partner):
                    pk = packed.at[partner].get(mode="fill", fill_value=_n_el)
                    loc = pk - off
                    valid = (loc >= 0) & (loc < s["tris"].shape[0])
                    locc = jnp.clip(loc, 0, max(int(s["tris"].shape[0]) - 1, 0))
                    ev = _vl[s["tris"][locc]]                          # (b, cap, 3, 3)
                    qp = jnp.einsum("qk,bckd->bcqd", bary, ev,
                                    precision=jax.lax.Precision.HIGHEST)
                    area = jnp.linalg.norm(jnp.cross(ev[..., 1, :] - ev[..., 0, :],
                                                     ev[..., 2, :] - ev[..., 0, :]), axis=-1) / 2
                    sval = jnp.einsum("qk,bck->bcq", bary, surf_nodal[locc])
                    chg = sval * ((area * valid)[..., None] * w[None, None, :])
                    return qp.reshape(qp.shape[0], -1, 3), chg.reshape(chg.shape[0], -1)
                return fn

            # one pass per (family, rule, width tier): the closure sees idcs[:, :w] only
            lz = []
            for fam, fac, off in (("tet", _tet_fn, 0), ("tri", _tri_fn, s["N_tet"])):
                for rule, sfx in (("_n", ""), ("_c", "_c")):
                    if rule == "_c" and "bary_tri_c" not in s:
                        continue
                    bary = s["bary_" + fam + rule]
                    for w, dc, bx in zip(s[fam + "_near_w" + sfx], s[fam + "_dir_cnct" + sfx],
                                         s[fam + "_near_boxes" + sfx]):
                        lz.append((fac(bary, s["w_" + fam + rule], s[fam + "_idcs_near"][bx, :w], off),
                                   w * bary.shape[0], dc, bx))
            _live = sum(1 for _, _, dc, _ in lz if jnp.asarray(dc).size)
            _ml = mem_limit / max(_live, 1)
            _bph = s.get("img_phase")
            for fn, wdt, dc, bx in lz:
                if jnp.asarray(dc).size == 0:
                    continue
                _dummy = jnp.zeros((bx.shape[0], 0, 3), padded_eval.dtype)   # carries nbox only
                out = out + eval_direct(_dummy, None, padded_eval, s["eval_rev_near"], dc,
                                        s["img"], field, _ml, mixed=True, centres=cen_s[bx],
                                        eval_centres=cen_e, img_phase=_bph, kernel_dtype=kernel_dtype,
                                        kernel_scale=kernel_scale, src_fn=fn, src_width=int(wdt),
                                        coincidence_floor=_cf, batch_rule="eval_slots")
            if isinstance(out, float):
                out = jnp.zeros((s["eval_pts"].shape[0], 3) if field else (s["eval_pts"].shape[0],))
            return out
        # one pass per (family, rule, width tier) over the tier's front-packed idcs[:, :w]
        fams = []
        for fam, off in (("tet", 0), ("tri", s["N_tet"])):
            for rule, sfx in (("_n", ""), ("_c", "_c")):
                if rule == "_c" and "bary_tri_c" not in s:
                    continue
                qpos = s[fam + "_qpos" + rule]
                qchg = (vol_chrg[:, None] * s["wm_tet" + rule] if fam == "tet" else
                        jnp.einsum("qk,ek->eq", s["bary_tri" + rule], surf_nodal) * s["wm_tri" + rule])
                for w, dc, bx in zip(s[fam + "_near_w" + sfx], s[fam + "_dir_cnct" + sfx],
                                     s[fam + "_near_boxes" + sfx]):
                    fams.append((qpos, qchg, s[fam + "_idcs_near"][bx, :w], dc, off, bx))
        # the passes are independent scans, so under jit XLA may overlap their batch
        # workspaces — split the transient budget across the live passes
        _live = sum(1 for qp, _, _, dc, _, _ in fams if qp.shape[0] and jnp.asarray(dc).size)
        _ml = mem_limit / max(_live, 1)
        for qpos, qchg, idcs, dc, off, bx in fams:
            if qpos.shape[0] == 0 or jnp.asarray(dc).size == 0:  # family absent / no live pairs
                continue
            sp, sc = _gather_box_quad(qpos, qchg, idcs, off)
            _bph = s.get("img_phase")
            out = out + eval_direct(sp, sc, padded_eval, s["eval_rev_near"], dc, s["img"],
                                    field, _ml, mixed=True, centres=cen_s[bx], eval_centres=cen_e,
                                    img_phase=_bph, kernel_dtype=kernel_dtype,
                                    kernel_scale=kernel_scale, coincidence_floor=_cf, batch_rule="eval_slots")
        if isinstance(out, float):                               # both families empty
            out = jnp.zeros((s["eval_pts"].shape[0], 3) if field else (s["eval_pts"].shape[0],))
        return out
    tet_qchg = vol_chrg[:, None] * s["wm_tet_n"]                                     # low-order near rule
    tri_qchg = jnp.einsum("qk,ek->eq", s["bary_tri_n"], surf_nodal) * s["wm_tri_n"]
    tp, tc = _gather_box_quad(s["tet_qpos_n"], tet_qchg, s["src_idcs_near"], 0)
    sp, sc = _gather_box_quad(s["tri_qpos_n"], tri_qchg, s["src_idcs_near"], s["N_tet"])
    src_pos = jnp.concatenate([tp, sp], axis=1)                  # combined tet + tri quad sources per box
    src_chg = jnp.concatenate([tc, sc], axis=1)
    _bph = s.get("img_phase")
    return eval_direct(src_pos, src_chg, padded_eval, s["eval_rev_near"], s["dir_cnct"], s["img"],
                       field, mem_limit, mixed=True,
                       centres=cen_s, eval_centres=cen_e, img_phase=_bph, coincidence_floor=_cf, batch_rule="eval_slots")


def element_selfcorrection(s, vol_chrg, surf_nodal):
    r"""
    Duffy self/touching correction to the **potential** at the mesh nodes.

    Replaces the inaccurate plain-Gauss value the bulk near field used for each
    (element, own-node) pair with the converged singular integral. Assumes ``eval_pts``
    are the mesh vertices (the node of element ``e`` at local index ``l`` is global
    vertex ``elems[e, l]``).

    Parameters
    ----------
    s : dict
        Setup dict from :func:`element_farfield_setup`.
    vol_chrg : jnp.ndarray or None, shape (N_tet,)
        P0 volume charge density.
    surf_nodal : jnp.ndarray or None, shape (N_tri, 3)
        P1 nodal surface charge density.

    Returns
    -------
    jnp.ndarray, shape (N_eval,)
        Per-node potential correction (already divided by :math:`4\pi`), to add to
        far + near.
    """
    vol_chrg, surf_nodal = _coerce_charges(s, vol_chrg, surf_nodal)
    _cdt = jnp.result_type(vol_chrg.dtype if vol_chrg is not None else jnp.float32,
                           surf_nodal.dtype if surf_nodal is not None else jnp.float32)
    corr = jnp.zeros(s["eval_pts"].shape[0], _cdt)   # complex under Bloch
    corr = corr.at[s["tets"]].add(vol_chrg[:, None] * s["corr_t"].sum(axis=2))           # P0: sum over basis
    corr = corr.at[s["tris"]].add(jnp.einsum("elv,ev->el", s["corr_s"], surf_nodal))     # P1: weight by nodal sigma
    return corr / (4 * jnp.pi)


def element_eval(s, vol_chrg, surf_nodal, field = False, mem_limit = None,
                 kernel_dtype = None, kernel_scale = None):
    r"""
    Total FEM potential at the mesh nodes: far + near + Duffy self-correction.

    For the field (``field=True``) only far + near is returned -- the singular *gradient*
    self-term (a stronger :math:`1/r^2` singularity) is not yet corrected. See the
    component functions.

    Parameters
    ----------
    s : dict
        Setup dict from :func:`element_farfield_setup`.
    vol_chrg : jnp.ndarray or None, shape (N_tet,)
        P0 volume charge density per tet.
    surf_nodal : jnp.ndarray or None, shape (N_tri, 3)
        P1 nodal surface charge per boundary triangle.
    field : bool, optional
        Return :math:`H=-\nabla u` (vector) instead of the potential ``u``.
    mem_limit : float, optional
        Byte budget for the batched transforms (default unlimited).

    Returns
    -------
    jnp.ndarray, shape (N_eval,) or (N_eval, 3)
        Total potential (or field) at ``eval_pts`` (original order).
    """
    out = element_farfield_eval(s, vol_chrg, surf_nodal, field, mem_limit) \
        + element_nearfield_eval(s, vol_chrg, surf_nodal, field, mem_limit,
                                 kernel_dtype=kernel_dtype, kernel_scale=kernel_scale)
    if not field and s["self_correct"]:
        out = out + element_selfcorrection(s, vol_chrg, surf_nodal)
    return out


@partial(jax.jit, static_argnames=["skel", "field", "mem_limit"])
def _elem_frozen_fwd(arrs, vol_chrg, surf_nodal, skel, field, mem_limit):
    r"""Module-level jitted frozen element-tree forward (setup as (skeleton, arrays))."""
    from jaxfmm.util import join_static
    return element_eval(join_static(skel, arrs), vol_chrg, surf_nodal, field, mem_limit)


def element_compile(s, field=False, mem_limit=None):
    r"""
    FROZEN-geometry element-tree evaluator.

    One jitted ``fmm(vol_chrg, surf_nodal) -> potential`` over the
    :func:`element_farfield_setup` dict (parity with :func:`element_quadtree_compile`;
    either charge family may be ``None``). Setup-as-arguments: identical-shape setups
    re-use the executable with zero retrace.

    Parameters
    ----------
    s : dict
        Setup dict from :func:`element_farfield_setup`.
    field : bool, optional
        Return the vector field instead of the potential.
    mem_limit : float, optional
        Byte budget for the batched transforms (default unlimited).

    Returns
    -------
    callable
        ``fmm(vol_chrg, surf_nodal=None) -> (N_eval,) or (N_eval, 3) jnp.ndarray``,
        the jitted frozen-geometry evaluator.
    """
    from jaxfmm.util import split_static
    skel, arrs = split_static(s)
    return lambda vol_chrg, surf_nodal=None: _elem_frozen_fwd(arrs, vol_chrg, surf_nodal,
                                                              skel, field, mem_limit)
