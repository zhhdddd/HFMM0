r"""Shared finite-element source assembly: quadrature point-charges per element,
Duffy self/touching corrections, charge coercion, pair-list compaction."""
import jax
import jax.numpy as jnp
import numpy as np
from jaxfmm.fem.quadrature import duffy_simplex


__all__ = []


def _as_elements(tets, tris):
    r"""
    Normalize the element families, substituting empty index arrays for ``None``.

    ``None`` disables a charge family (volume ``tets`` / surface ``tris``) via a canonical
    zero-length index array — the empty family then flows shape-safely through quadrature,
    trees, charges and evals (0 sources), with no branches in the compiled code.

    Parameters
    ----------
    tets : jnp.ndarray or None, shape (N_tet, 4)
        Tetrahedron vertex indices, or ``None`` to disable the volume family.
    tris : jnp.ndarray or None, shape (N_tri, 3)
        Surface triangle vertex indices, or ``None`` to disable the surface family.

    Returns
    -------
    tets, tris : jnp.ndarray
        The input arrays, with ``None`` replaced by zero-length int32 arrays.

    Notes
    -----
    E.g. a uniformly magnetised body has :math:`\rho = -\nabla\cdot(M_s m) = 0`
    identically, so ``tets=None`` drops the (exactly zero) volume sources for free.
    """
    if tets is None:
        tets = jnp.zeros((0, 4), dtype=jnp.int32)
    if tris is None:
        tris = jnp.zeros((0, 3), dtype=jnp.int32)
    if tets.shape[0] == 0 and tris.shape[0] == 0:
        raise ValueError("at least one element family (tets/tris) must be non-empty")
    return tets, tris


def _self_correction_geom(verts, elems, duffy_deg, qpos_n, wm_n, bary_n, field=False, w_n=None):
    r"""
    Geometric self/touching correction factors per (element, local node).

    The *singular* integral of each P1 basis function over the element against the kernel
    (proper Duffy) MINUS the plain low-order Gauss value the bulk near field already added
    for that pair. Multiply by the nodal charge per step and scatter to the node -> turns
    the inaccurate bulk Gauss into the accurate Duffy value, leaving the rest of the near
    field untouched.

    Parameters
    ----------
    verts : jnp.ndarray, shape (N_vert, 3)
        Vertex coordinates.
    elems : jnp.ndarray, shape (N_elem, k)
        Element vertex indices (``k=4`` tet, ``k=3`` tri).
    duffy_deg : int
        Degree of the Duffy rule for the singular integral.
    qpos_n : jnp.ndarray, shape (N_elem, nq_near, 3), or None
        The near-field quadrature points the bulk pass uses; ``None`` forms them per slab from
        ``bary_n`` and ``w_n`` (lazy near setups, which never store them).
    wm_n : jnp.ndarray, shape (N_elem, nq_near), or None
        The corresponding weight*measure factors (``None`` together with ``qpos_n``).
    bary_n : jnp.ndarray, shape (nq_near, k)
        Barycentric coordinates of the near rule.
    w_n : jnp.ndarray, shape (nq_near,), optional
        Near-rule weights; required when ``qpos_n`` is ``None``.
    field : bool, optional
        Correct the potential (``False``, scalar kernel :math:`1/|x-V_l|`) or the field
        (``True``, vector kernel :math:`(V_l-x)/|x-V_l|^3`, matching :func:`eval_direct`'s
        ``(eval-src)/r^3`` convention).

    Returns
    -------
    corr : jnp.ndarray, shape (N_elem, l, v) or (N_elem, l, v, 3)
        For singular local node ``l`` and basis vertex ``v``,

        .. math:: \int_{elem}\lambda_v\,K(x,V_l)
                  - \sum_q \lambda_v(x_q^{near})\,wm^{near}_q\,K(x_q^{near},V_l)

        with ``K`` the potential/field kernel. (For a P0 density only the ``v``-sum is
        used.)

    Notes
    -----
    The field self-term is only well posed for the **volume** (tet) case: at a tet vertex
    the integral converges (:math:`1/r^2` vs the :math:`r^2` volume Jacobian). For a
    surface (tri) vertex the in-plane corner term is log-divergent (no finite value), so
    the field route does NOT apply the surface correction -- the surface contribution to
    H is taken from the lumped weak gradient instead (see
    :func:`jaxfmm.apps.mag.strayfield.get_lumped_strayfield`).
    """
    k = elems.shape[1]                                   # 4 tet, 3 tri
    bd, Wd, om1 = (jnp.asarray(a) for a in duffy_simplex(k - 1, duffy_deg))
    # Transients scale as N_elem*nq_duffy (up to k*3 of them live at once); one monolithic
    # pass allocates tens of GiB at 1e8 elements. Slab so the largest transient stays ~512 MiB.
    slab = max(1, int(2**29 // max(bd.shape[0] * 3 * verts.dtype.itemsize * (3 if field else 1), 1)))
    if elems.shape[0] > slab:
        return jnp.concatenate(
            [_self_correction_geom(verts, elems[i:i + slab], duffy_deg,
                                   None if qpos_n is None else qpos_n[i:i + slab],
                                   None if wm_n is None else wm_n[i:i + slab], bary_n, field=field, w_n=w_n)
             for i in range(0, elems.shape[0], slab)])
    if qpos_n is None:
        from jaxfmm.fem.quadrature import element_quad_points
        qpos_n, meas = element_quad_points(verts, elems, bary_n)
        wm_n = w_n[None, :] * meas[:, None]
    ev = verts[elems]                                    # (N_elem, k, 3)
    if k == 4:
        geomfac = jnp.abs(jnp.linalg.det(ev[:, 1:] - ev[:, :1])) / 6 * 6          # 6*Vol
    else:
        geomfac = jnp.linalg.norm(jnp.cross(ev[:, 1] - ev[:, 0], ev[:, 2] - ev[:, 0]), axis=-1) / 2 * 2  # 2*Area
    corr = []
    for l in range(k):
        perm = list(range(k)); perm[1], perm[l] = perm[l], perm[1]               # singular node l -> bary index 1
        rv = verts[elems[:, perm]]                                               # (N_elem, k, 3)
        # precision pinned: TF32 duffy points corrupt the correction
        x = jnp.einsum("qk,ekd->eqd", bd, rv, precision=jax.lax.Precision.HIGHEST)  # (N_elem, nq, 3)
        r = jnp.linalg.norm(x - rv[:, 1][:, None], axis=-1)                      # |x-V_l| (N_elem, nq)
        rho = r / om1[None, :]
        node = verts[elems[:, l]]                                                # (N_elem, 3)
        rg = jnp.linalg.norm(qpos_n - node[:, None], axis=-1)                    # (N_elem, nq_near)
        cl = []
        for v in range(k):                                                       # P1 basis of original vertex v
            if field:                                                            # vector kernel (V_l-x)/r^3
                dvec = rv[:, 1][:, None] - x                                      # V_l - x   (N_elem, nq, 3)
                gvec = node[:, None] - qpos_n                                     # V_l - x_near (N_elem, nq_near, 3)
                duffy = geomfac[:, None] * (Wd[None, :, None] * bd[:, perm[v]][None, :, None]
                          * om1[None, :, None] * dvec / (r[..., None] ** 3)).sum(axis=1)
                gauss = (bary_n[:, v][None, :, None] * wm_n[:, :, None]
                          * gvec / (rg[..., None] ** 3)).sum(axis=1)
            else:                                                                # scalar kernel 1/r
                duffy = geomfac * (Wd[None, :] * bd[:, perm[v]][None, :] / rho).sum(axis=1)
                gauss = (bary_n[:, v][None, :] * wm_n / rg).sum(axis=1)
            cl.append(duffy - gauss)
        corr.append(jnp.stack(cl, axis=1))                                       # (N_elem, k[, 3]) over v
    return jnp.stack(corr, axis=1)                                               # (N_elem, l, v[, 3])

def _self_energy_geom(verts, elems, duffy_deg, qpos_n, wm_n, bary_n, outer_deg=None):
    r"""
    Self-energy correction matrices per element for **quadrature-point targets**.

    The diff pairing (``init_strayfield_elements(diff=True)``) evaluates the potential AT
    the near-rule points, so each element's interaction with itself is a plain point sum
    with the coincident pairs dropped -- an O(h) energy floor (unit cube, f64, p8:
    ``N - 1/3`` = -9.2e-3 at h = 0.1 decaying as h^1.0, independent of p/theta/Duffy and
    only weakly of the near rule; surface self-triangles dominate, the volume part is
    O(h^2)). This returns, per element, the exact Galerkin self-interaction of its P1
    density minus the point sum the bulk near field already counts::

        dJ[e, k, l] = int_e int_e lam_k(x) lam_l(y) / |x - y| dy dx
                      - sum_{q != q'} wm_q lam_k(x_q) wm_q' lam_l(x_q') / |x_q - x_q'|

    so that ``1/2 c^T dJ c`` (nodal densities ``c``; a P0 density sums over ``k, l``)
    added to the point-pair energy makes the self-block exact **and keeps it symmetric**
    (the energy gradient stays the exact consistent field). The outer integral uses the
    near rule itself; the inner singular integral is a Duffy rule on the ``k``
    sub-simplices ``(x_q, face_f)`` with the apex at the target. 4pi-free.

    Parameters
    ----------
    verts, elems, duffy_deg, qpos_n, wm_n, bary_n
        As in :func:`_self_correction_geom`.

    Returns
    -------
    dJ : jnp.ndarray, shape (N_elem, k, k)
        Symmetric per-element correction (``k=4`` tet, ``k=3`` tri).
    """
    k = elems.shape[1]
    bd, Wd, om1 = (jnp.asarray(a, verts.dtype) for a in duffy_simplex(k - 1, duffy_deg))
    if outer_deg is None:                                 # outer rule = the near rule itself
        qpos_o, wm_o, bary_o = qpos_n, wm_n, jnp.asarray(bary_n, verts.dtype)
    else:                                                 # decoupled, higher-order outer rule
        from jaxfmm.fem.quadrature import simplex_quad, element_quad_points
        bo, wo = simplex_quad(k - 1, outer_deg)
        qpos_o, meas = element_quad_points(verts, elems, jnp.asarray(bo, verts.dtype))
        wm_o, bary_o = jnp.asarray(wo, verts.dtype)[None, :] * meas[:, None], jnp.asarray(bo, verts.dtype)
    nq, nd = qpos_o.shape[1], bd.shape[0]
    slab = max(1, int(2**29 // max(nq * nd * 3 * verts.dtype.itemsize, 1)))
    if elems.shape[0] > slab:
        return jnp.concatenate(
            [_self_energy_geom(verts, elems[i:i + slab], duffy_deg,
                               qpos_n[i:i + slab], wm_n[i:i + slab], bary_n, outer_deg)
             for i in range(0, elems.shape[0], slab)])
    bary_n = jnp.asarray(bary_n, verts.dtype)
    ev = verts[elems]                                                            # (N, k, 3)
    eye = jnp.eye(k, dtype=verts.dtype)
    I = jnp.zeros(qpos_o.shape[:2] + (k,), verts.dtype)                          # I_l(x_q) (N, nq, k)
    for f in range(k):                                    # sub-simplex (target, face opposite f)
        oth = [j for j in range(k) if j != f]
        # sub-simplex vertices with the singular apex (the target) at bary index 1
        sv = jnp.stack([jnp.broadcast_to(ev[:, None, oth[0]], qpos_o.shape), qpos_o]
                       + [jnp.broadcast_to(ev[:, None, j], qpos_o.shape) for j in oth[1:]],
                       axis=2)                                                   # (N, nq, k, 3)
        # P1 basis of the ORIGINAL element at the sub-simplex vertices -> at the Duffy points
        lam_v = jnp.stack([jnp.broadcast_to(eye[oth[0]], (nq, k)), bary_o]
                          + [jnp.broadcast_to(eye[j], (nq, k)) for j in oth[1:]], axis=1)  # (nq, k, k)
        lam = jnp.einsum("dj,qjl->qdl", bd, lam_v)                                # (nq, nd, k)
        y = jnp.einsum("dj,eqjx->eqdx", bd, sv, precision=jax.lax.Precision.HIGHEST)
        r = jnp.linalg.norm(y - qpos_o[:, :, None], axis=-1)                      # (N, nq, nd)
        if k == 4:
            geomfac = jnp.abs(jnp.linalg.det(sv[:, :, 1:] - sv[:, :, :1]))        # 6 Vol
        else:
            geomfac = jnp.linalg.norm(jnp.cross(sv[:, :, 1] - sv[:, :, 0],
                                                sv[:, :, 2] - sv[:, :, 0]), axis=-1)  # 2 Area
        I = I + geomfac[..., None] * jnp.einsum("d,qdl,eqd->eql", Wd * om1, lam, 1.0 / r)
    J = jnp.einsum("eq,qk,eql->ekl", wm_o, bary_o, I)
    nq = qpos_n.shape[1]
    B = wm_n[:, :, None] * bary_n[None]                                          # (N, nq, k)
    d = jnp.linalg.norm(qpos_n[:, :, None] - qpos_n[:, None, :], axis=-1)        # (N, nq, nq)
    offd = ~jnp.eye(nq, dtype=bool)[None]
    dinv = jnp.where(offd, 1.0 / jnp.where(offd, d, 1.0), 0.0)                   # coincident pairs -> 0, as eval_direct
    G = jnp.einsum("eqk,eqp,epl->ekl", B, dinv, B)
    return 0.5 * (J + jnp.swapaxes(J, 1, 2)) - G


def _gather_box_quad(qpos, qchg, idcs, off):
    r"""
    Per-box element quadrature points/charges for one element family.

    ``idcs`` are global element indices; entries outside this family's ``[off, off+N)``
    block gather zero charge (so families are summed cleanly).

    Notes
    -----
    Mirrors :func:`jaxfmm.flex.get_initial_mpls_elements`'s safe gather: JAX wraps
    negative indices, so out-of-range indices are mapped to a positive sentinel.
    """
    nbox, cap = idcs.shape
    nq, M = qpos.shape[1], qpos.shape[0]
    if M == 0:                                       # disabled family: XLA gathers cannot
        return (jnp.zeros((nbox, 0, 3), qpos.dtype),  # slice a zero-length dim -> 0-width
                jnp.zeros((nbox, 0), qchg.dtype))
    loc = idcs - off
    loc = jnp.where((loc >= 0) & (loc < M), loc, M)
    pos = qpos.at[loc].get(mode="fill", fill_value=0.0).reshape((nbox, cap * nq, 3))
    chg = qchg.at[loc].get(mode="fill", fill_value=0.0).reshape((nbox, cap * nq))
    return pos, chg


def _coerce_charges(s, vol_chrg, surf_nodal):
    r"""
    Coerce ``None`` charges to zeros of the setup's element counts.

    "No volume charge" / "no surface charge" is then spelled the same across every
    element-FMM flavor (families the setup was built without have 0 rows, so the zeros
    cost nothing).
    """
    # flex setups store moment tables (T_tet/T_tri), the quadtree/uniform ones quad weights
    _dt = (s["T_tet"] if "T_tet" in s else s["wm_tet"]).dtype
    if vol_chrg is None:
        vol_chrg = jnp.zeros(s["tets"].shape[0], _dt)
    if surf_nodal is None:
        surf_nodal = jnp.zeros((s["tris"].shape[0], 3), _dt)
    return vol_chrg, surf_nodal



def _compact_pairs(cnct, n_eval, n_src, gran=None):
    r"""
    Drop the sentinel/-1 padding rows of a pair list, optionally re-padding to a granule.

    Frozen geometry doesn't need the stable shapes the padded buffers exist for, and
    padded rows cost gather+compute in M2L/P2P. ``gran`` re-pads the compacted list UP to
    a granularity multiple (-1 rows, dropped by the consumers' ``>=0`` masks): exact-count
    lists never repeat their shape across geometries, granular ones do -- the
    cross-geometry executable reuse of the setup-as-arguments frozen evaluators depends on
    it (cost: less than one granule of masked pair work).
    """
    cn = np.asarray(cnct)
    ok = (cn[:, 0] >= 0) & (cn[:, 1] >= 0) & (cn[:, 0] < n_eval) & (cn[:, 1] < n_src)
    cn = cn[ok]
    if gran is not None:
        n_pad = -len(cn) % gran
        # pad with the POSITIVE-OOB max-int sentinel, NOT -1: negative indices WRAP in
        # jax gathers/scatters, so -1 pad rows double-count the last box's interactions
        # (eval_direct relies on OOB gather-fill/scatter-drop; max-int is its convention)
        cn = np.concatenate([cn, np.full((n_pad, 2), np.iinfo(cn.dtype).max, cn.dtype)])
    return jnp.asarray(cn)


def _quantize_snap_grp(grp, gran=256, gran_cls=4):
    r"""
    Granularity-pad the theta-snap tier buffers along BOTH axes (see :func:`_compact_pairs`).

    Pads the pair capacity (``gran``) and the per-tier CLASS count (``gran_cls`` -- a
    single theta class migrating between tiers under geometry drift otherwise re-keys the
    executable). Padding classes use id 0 with all-sentinel pair rows: M2L_grouped's
    ``>= 0`` mask zeroes them, so they cost one rotation GEMM row and contribute nothing.
    """
    if grp is None:
        return None
    cls_q, cncts = [], []
    for ci, cn in zip(grp["cls"], grp["cnct"]):
        nc, w = cn.shape[0], cn.shape[1]
        ncq = nc + (-nc % gran_cls)
        wq = w + (-w % gran)
        cls_q.append(jnp.pad(jnp.asarray(ci), (0, ncq - nc), constant_values=0))
        cncts.append(jnp.pad(cn, ((0, ncq - nc), (0, wq - w), (0, 0)), constant_values=-1))
    return dict(grp, cls=tuple(cls_q), cnct=tuple(cncts))
