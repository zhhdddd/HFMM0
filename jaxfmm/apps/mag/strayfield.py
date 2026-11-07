r"""Micromagnetic stray-field computation on P1-FEM tetrahedral meshes via the FMM.

Maps a nodal magnetisation to magnetic charges — element densities for the element-based
FMM (:func:`gen_element_charges`), or vertex-lumped point charges optionally paired with
an analytic on-mesh near-field correction (:func:`gen_chrgs_and_corr`) — evaluates the
scalar potential with the point FMM, and recovers the field :math:`\mathbf{H}` at the
mesh nodes. Includes the analytic uniformly-magnetised-cuboid reference solution
(:func:`anahexa`).
"""
import jax
import jax.numpy as jnp
import numpy as np
from jaxfmm import *
from functools import partial
from jaxfmm.fem.mesh_util import FACE_IDCS, OPP_IDCS, get_tet_volumes, reorder_mesh_rcm, check_min_dist
import warnings

RCM_EXACT_LIMIT = 2**24   # tets; above this the RCM ordering diagnostic uses the N^(2/3) estimate

__all__ = ["init_strayfield", "eval_strayfield", "gen_element_charges", "gen_lumped_charges", "get_tris",
           "get_lumped_strayfield", "filter_periodic_boundary_tris", "gen_chrgs_and_corr", "gen_lumped_ops", "fulljit_strayfield_driver", "anahexa", "anahexa0", "gen_correction", "matinv_3x3", "init_strayfield_elements", "eval_strayfield_elements", "compile_strayfield_elements", "plan_strayfield"]

@jax.jit
def matinv_3x3(T):
    r"""Invert a batch of 3x3 matrices with Cramer's rule.

    Parameters
    ----------
    T : jnp.ndarray
        Array of shape (N, 3, 3).

    Returns
    -------
    A : jnp.ndarray
        Inverse matrices, shape (N, 3, 3).
    dets : jnp.ndarray
        Determinants, shape (N,).
    """
    A = jnp.zeros_like(T)

    ### the first-row cofactors double as the determinant expansion
    tmp = T[:,1,1]*T[:,2,2] - T[:,1,2]*T[:,2,1]
    dets = T[:,0,0]*tmp
    A = A.at[:,0,0].set(tmp)

    tmp = -(T[:,1,0]*T[:,2,2] - T[:,1,2]*T[:,2,0])
    A = A.at[:,0,1].set(tmp)
    dets += T[:,0,1]*tmp

    tmp = T[:,1,0]*T[:,2,1] - T[:,1,1]*T[:,2,0]
    A = A.at[:,0,2].set(tmp)
    dets += T[:,0,2]*tmp

    A = A.at[:,1,0].set(-(T[:,0,1]*T[:,2,2] - T[:,0,2]*T[:,2,1]))
    A = A.at[:,2,0].set( (T[:,0,1]*T[:,1,2] - T[:,0,2]*T[:,1,1]))
    A = A.at[:,1,1].set( (T[:,0,0]*T[:,2,2] - T[:,0,2]*T[:,2,0]))
    A = A.at[:,2,1].set(-(T[:,0,0]*T[:,1,2] - T[:,0,2]*T[:,1,0]))
    A = A.at[:,1,2].set(-(T[:,0,0]*T[:,2,1] - T[:,0,1]*T[:,2,0]))
    A = A.at[:,2,2].set( (T[:,0,0]*T[:,1,1] - T[:,0,1]*T[:,1,0]))

    return A / dets[:,None,None], dets

@jax.jit
def gen_element_charges(m, Ms, Ms_surf, verts, tets, tris):
    r"""P1-FEM magnetic charge densities for the element-based FMM.

    Computes the densities the element FMM consumes (volume P0 + surface P1) from a nodal
    magnetisation ``m`` — no point-charge lumping and no analytic correction; the element
    pipeline itself handles the near field (far field + near quadrature + Duffy
    self-integrals).

    The volume density :math:`\rho = -\nabla\cdot(M_s\,\mathbf{m})` is constant per
    tetrahedron (P1 ``m`` gives a P0 divergence), computed via the divergence theorem
    from the face area vectors — no matrix inverse (see the inline note; identical to
    the affine-inverse form by the P1 gradient identity). The surface density
    :math:`\sigma = M_s^{\mathrm{surf}}\,(\mathbf{n}\cdot\mathbf{m})` is P1 nodal per
    boundary triangle, with the normal convention
    :math:`\mathbf{n} = (v_0 - v_2)\times(v_1 - v_2)` — pass ``tris`` (and the sign of
    ``Ms_surf``) from :func:`get_tris`.

    Parameters
    ----------
    m : jnp.ndarray
        Nodal magnetisation of shape (N_vert, 3).
    Ms : float or jnp.ndarray
        Saturation magnetisation, scalar or per-tetrahedron of shape (N_tet,).
    Ms_surf : float or jnp.ndarray
        Surface saturation magnetisation, scalar or per-triangle of shape (N_tri,).
    verts : jnp.ndarray
        Vertex coordinates of shape (N_vert, 3).
    tets : jnp.ndarray or None
        Tetrahedron vertex indices of shape (N_tet, 4); ``None`` disables volume charges.
    tris : jnp.ndarray or None
        Boundary-triangle vertex indices of shape (N_tri, 3); ``None`` disables surface
        charges.

    Returns
    -------
    rho : jnp.ndarray
        Volume charge density of shape (N_tet,).
    sigma : jnp.ndarray
        Surface nodal charge density of shape (N_tri, 3).

    Notes
    -----
    Fully vectorised (no scan), jit-able, and linear in ``m``, so the magnetisation
    gradient has a cheap adjoint. A disabled charge family yields a zero-length density,
    matching the element setups' family toggles; zero-length index arrays flow through
    the regular math unchanged.
    """
    if tets is None:
        rho = jnp.zeros((0,), dtype=m.dtype)
    else:
        # divergence theorem, no matrix inverse: div m = 2 * sum_v a_v . (m_v - m_3) / det6
        # with a_v the outward face area vectors (P1 gradient identity T[v,:] = 2 a_v/det6).
        # Well-conditioned on m-DIFFERENCES near divergence-free states, one signed scalar
        # division instead of the 3x3 inverse (which blows up on degenerating tets — the
        # shape-opt path deforms elements), and a cleaner verts-adjoint.
        e = verts[tets[:, :3]] - verts[tets[:, 3], None]
        an = 0.5 * jnp.stack([jnp.cross(e[:, 1], e[:, 2], axis=-1),
                              jnp.cross(e[:, 2], e[:, 0], axis=-1),
                              jnp.cross(e[:, 0], e[:, 1], axis=-1)], axis=1)     # (T,3v,3d)
        det6 = jnp.einsum("td,td->t", e[:, 0], jnp.cross(e[:, 1], e[:, 2], axis=-1))
        dm = m[tets[:, :3]] - m[tets[:, 3], None]
        rho = -Ms * 2.0 * jnp.einsum("tvd,tvd->t", an, dm) / det6               # (N_tet,) volume density
    if tris is None:
        sigma = jnp.zeros((0, 3), dtype=m.dtype)
    else:
        n = jnp.cross(verts[tris[:, 0]] - verts[tris[:, 2]], verts[tris[:, 1]] - verts[tris[:, 2]], axis=-1)
        n = n / jnp.linalg.norm(n, axis=-1)[:, None]                            # unit outward normal
        sn = jnp.einsum('td,tvd->tv', n, m[tris])                               # (N_tri, 3) = n . m_v
        sigma = (Ms_surf[:, None] if jnp.ndim(Ms_surf) else Ms_surf) * sn
    return rho, sigma


def _scan_blocked(carry, elems, scale, body, mem_limit, bytes_per_elem):
    r"""Blocked ``lax.scan`` of a carry-accumulating per-element body.

    The shared batching wrapper of the eager assembly sweeps: pads ``elems`` (and a
    per-element ``scale``, when given) with out-of-range indices / zeros — degenerate,
    exactly-zero-contribution elements — reshapes into blocks sized to keep the fused
    per-block intermediates within ``mem_limit`` bytes, and scans ``body(carry, xs)``
    over them. ``scale`` may be None (body takes elems only), a scalar (passed through
    unblocked) or a per-element array (blocked and zero-padded alongside).
    """
    n = elems.shape[0]
    mem = n * bytes_per_elem
    batch = int((mem_limit / mem) * n) + 1 if mem_limit < mem else n
    per_elem = scale is not None and jnp.ndim(scale) and scale.shape[0] == n
    if batch >= n:
        xs = elems if scale is None or not per_elem else (elems, scale)
        return body(carry, xs)[0]
    pad = batch - n % batch
    # fill typed to the operand: under x64 a Python int is int64 and lax.pad rejects it
    # against int32 element indices (ARCHITECTURE 7.34)
    eb = jax.lax.pad(elems, elems.dtype.type(jnp.iinfo(elems.dtype).max),
                     ((0, pad, 0), (0, 0, 0))).reshape((-1, batch, elems.shape[1]))
    if per_elem:
        sb = jax.lax.pad(scale, 0.0, [(0, pad, 0)]).reshape((-1, batch))
        return jax.lax.scan(body, carry, (eb, sb))[0]
    return jax.lax.scan(body, carry, eb)[0]

def _scatter_blocked(q, elems, scale, per_elem_fn, mem_limit, bytes_per_elem):
    r"""Blocked scatter-add of per-element nodal contributions.

    Accumulates ``q.at[blk].add(per_elem_fn(blk, scale_blk))`` over element blocks sized
    so the fused per-block intermediates stay within ``mem_limit`` bytes. Blocks are
    padded with all-zero-index elements, which are degenerate (zero measure/area) and
    contribute exactly zero — no validity masks needed. ``scale`` (per-element Ms) rides
    along blocked and zero-padded; scalars pass through unblocked.
    """
    n = elems.shape[0]
    n_blk = 1 if mem_limit == jnp.inf else max(1, -(-int(n * bytes_per_elem) // int(mem_limit)))
    if n_blk == 1:
        return q.at[elems].add(per_elem_fn(elems, scale).astype(q.dtype))
    blk = -(-n // n_blk)
    pad = n_blk * blk - n

    def _blocked(a):
        ab = jnp.concatenate([a, jnp.zeros((pad,) + a.shape[1:], a.dtype)])
        return ab.reshape((n_blk, blk) + a.shape[1:])

    eb = _blocked(elems)
    per_elem = jnp.ndim(scale) and scale.shape[0] == n       # (1,)-collapsed stays scalar-like
    sb = _blocked(scale) if per_elem else None

    def body(qc, xs):
        e_blk, s_blk = xs if per_elem else (xs, scale)
        return qc.at[e_blk].add(per_elem_fn(e_blk, s_blk).astype(qc.dtype)), None

    return jax.lax.scan(body, q, (eb, sb) if per_elem else eb)[0]


@partial(jax.jit, static_argnames=['mem_limit'])
def gen_lumped_charges(m, Ms, Ms_surf, verts, tets, tris, mem_limit = jnp.inf):
    r"""Fast vertex-lumped nodal magnetic charges (no analytic correction).

    The per-timestep charge path for dynamic simulations. Bit-comparable to
    ``gen_chrgs_and_corr(compute_corr=False)`` but built on the divergence theorem
    instead of the affine inverse:

    .. math:: \rho V = -\tfrac{M_s}{3}\sum_{v<3} (A\hat n)_v \cdot (\mathbf{m}_v - \mathbf{m}_3),

    with :math:`(A\hat n)_v` the outward area vector of the face opposite vertex ``v`` —
    three cross products and three dot products on the *differences* of ``m``
    (well-conditioned for near-divergence-free states), no matrix inverse, no
    determinant, and orientation-independent by construction. Volume lumping assigns
    :math:`\rho V/4` per node; surface lumping :math:`\sigma A/3` per node (same
    convention as :func:`gen_chrgs_and_corr`).

    Parameters
    ----------
    m : jnp.ndarray
        Nodal magnetisation of shape (N_vert, 3).
    Ms : float or jnp.ndarray
        Saturation magnetisation, scalar or per-tetrahedron of shape (N_tet,).
    Ms_surf : float or jnp.ndarray
        Surface saturation magnetisation, scalar or per-triangle of shape (N_tri,).
    verts : jnp.ndarray
        Vertex coordinates of shape (N_vert, 3).
    tets : jnp.ndarray or None
        Tetrahedron vertex indices of shape (N_tet, 4); ``None`` disables volume charges.
    tris : jnp.ndarray or None
        Boundary-triangle vertex indices of shape (N_tri, 3); ``None`` disables surface
        charges.
    mem_limit : float, optional
        Byte budget for the fused per-block intermediates, which otherwise grow with the
        full element count; blocks are scanned with zero-padded (degenerate,
        exactly-zero-contribution) elements. Default: unlimited.

    Returns
    -------
    q : jnp.ndarray
        Lumped nodal charges of shape (N_vert,).
    """
    q = jnp.zeros(verts.shape[0], m.dtype)
    itemsize = jnp.dtype(m.dtype).itemsize
    if tets is not None:
        def vol_q(tet_blk, Ms_blk):
            ev = verts[tet_blk]                               # (B,4,3)
            e = ev[:, :3] - ev[:, 3:4]
            an0 = 0.5 * jnp.cross(e[:, 1], e[:, 2], axis=-1)  # ~outward area vecs (sign via det)
            an1 = 0.5 * jnp.cross(e[:, 2], e[:, 0], axis=-1)
            an2 = 0.5 * jnp.cross(e[:, 0], e[:, 1], axis=-1)
            vol6 = jnp.einsum("td,td->t", e[:, 0], jnp.cross(e[:, 1], e[:, 2], axis=-1))
            dm = m[tet_blk[:, :3]] - m[tet_blk[:, 3], None]
            rhoV = -(jnp.sign(vol6) / 3.0) * (jnp.einsum("td,td->t", an0, dm[:, 0])
                                              + jnp.einsum("td,td->t", an1, dm[:, 1])
                                              + jnp.einsum("td,td->t", an2, dm[:, 2]))
            return (Ms_blk * rhoV)[:, None] / 4.0             # broadcast to the 4 nodes

        q = _scatter_blocked(q, tets, Ms, vol_q, mem_limit, 40 * itemsize)
    if tris is not None:
        def surf_q(tri_blk, sf_blk):
            tv = verts[tri_blk]
            n2 = jnp.cross(tv[:, 0] - tv[:, 2], tv[:, 1] - tv[:, 2], axis=-1)   # 2*A*n_hat
            smn = jnp.einsum("td,tvd->tv", n2, m[tri_blk])                      # 2A*(n.m)_v
            sf = sf_blk[:, None] if jnp.ndim(sf_blk) else sf_blk
            return sf * smn / 6.0                             # sigma * A/3 per node

        q = _scatter_blocked(q, tris, Ms_surf, surf_q, mem_limit, 30 * itemsize)
    return q


@partial(jax.jit, static_argnames=['mem_limit'])
def gen_correction(m, Ms, Ms_surf, pts, tets, tris, mem_limit = jnp.inf, **kwargs):
    r"""Analytic on-mesh near-field correction for the vertex-lumped P1-FEM charges.

    At each element's own vertices, adds the exact self-element potential and subtracts
    the potential of the element's lumped point charges — repairing the lumped
    representation exactly where it is worst. The exact potential of the P0 volume / P1
    surface density over each face is assembled from analytic per-edge terms of the form

    .. math::

        \log\frac{R^+ + l^+}{R^- + l^-}, \qquad
        \arctan\frac{P_0\, l^+}{R_0^2 + |d|\,R^+} - \arctan\frac{P_0\, l^-}{R_0^2 + |d|\,R^-},

    i.e. the closed-form potential of a polynomial density over a plane polygon. The
    per-element charge densities are recomputed locally (cheap next to the edge terms).

    Parameters
    ----------
    m : jnp.ndarray
        Nodal magnetisation of shape (N_vert, 3).
    Ms : float or jnp.ndarray
        Saturation magnetisation, scalar or per-tetrahedron of shape (N_tet,).
    Ms_surf : float or jnp.ndarray
        Surface saturation magnetisation, scalar or per-triangle of shape (N_tri,).
    pts : jnp.ndarray
        Vertex coordinates of shape (N_vert, 3).
    tets : jnp.ndarray or None
        Tetrahedron vertex indices of shape (N_tet, 4); ``None`` skips the volume part.
    tris : jnp.ndarray or None
        Boundary-triangle vertex indices of shape (N_tri, 3); ``None`` skips the surface
        part.
    mem_limit : float, optional
        Byte budget for the per-batch intermediates; elements are processed in a scan of
        padded batches when exceeded. Default: unlimited.

    Returns
    -------
    corrpot : jnp.ndarray
        Correction potential of shape (N_vert,), to be added to the FMM potential of the
        lumped charges.
    """
    corrpot = jnp.zeros(m.shape[0], m.dtype)   # follows m: complex under Bloch

    ### volume computations
    if(tets is not None):
        Ms_is_scalar = (jnp.size(Ms) < 2)

        def scan_body(buf, xs):
            if(Ms_is_scalar):
                tet_loc = xs
                Ms_loc = Ms
            else:
                tet_loc, Ms_loc = xs
            corrpot_loc = buf

            ### canonicalize orientation: the lumping (rho*det/24) and the analytic corr
            ### (outward face normals via FACE_IDCS) both assume detsvol < 0; meshes with
            ### mixed tet orientation are valid input, so swap two vertices of the
            ### positively-oriented tets.
            ec = pts[tet_loc[:,:3]] - pts[tet_loc[:,3],None]
            det0 = jnp.einsum('id,id->i', ec[:,0], jnp.cross(ec[:,1], ec[:,2], axis=-1))
            tet_loc = jnp.where((det0 > 0)[:,None], tet_loc[:,jnp.array([1,0,2,3])], tet_loc)

            ### compute volume charge density (divergence theorem — no matrix inverse;
            ### canonical orientation gives detsvol < 0, the signed division keeps signs)
            ec2 = pts[tet_loc[:,:3]] - pts[tet_loc[:,3],None]
            cr12 = jnp.cross(ec2[:,1], ec2[:,2], axis=-1)
            anv = 0.5 * jnp.stack([cr12, jnp.cross(ec2[:,2], ec2[:,0], axis=-1),
                                   jnp.cross(ec2[:,0], ec2[:,1], axis=-1)], axis=1)
            detsvol = jnp.einsum("td,td->t", ec2[:,0], cr12)                    # NOTE: detsvol < 0
            dmv = m[tet_loc[:,:3]] - m[tet_loc[:,3],None]
            volchrgs = -Ms_loc * 2.0 * jnp.einsum("tvd,tvd->t", anv, dmv) / detsvol

            ### compute quantities for all faces - tets must be ordered correctly!
            nvec = jnp.cross(pts[tet_loc[:,FACE_IDCS[:,1]]]-pts[tet_loc[:,FACE_IDCS[:,0]]],pts[tet_loc[:,FACE_IDCS[:,2]]]-pts[tet_loc[:,FACE_IDCS[:,0]]],axis=-1)
            nvec /= jnp.linalg.norm(nvec,axis=-1)[...,None]

            ### compute quantities for all edges
            rvecminus = pts[tet_loc[:,FACE_IDCS[:,:]]]          # edge starting point
            rvecplus = pts[tet_loc[:,FACE_IDCS[:,(1,2,0)]]]     # edge endpoint

            lvec = rvecplus - rvecminus       # [tets, faces, edges, coords]
            l = jnp.linalg.norm(lvec,axis=-1) # [tets, faces, edges]
            lvec /= l[...,None]

            pvecplus = rvecplus - nvec[:,:,None,:] * (nvec[:,:,None,:]*rvecplus).sum(axis=-1)[...,None]    # [tets, faces, edges, coords]
            pvecminus = rvecminus - nvec[:,:,None,:] * (nvec[:,:,None,:]*rvecminus).sum(axis=-1)[...,None] # [tets, faces, edges, coords]
            uvec = jnp.cross(lvec,nvec[:,:,None,:],axis=-1)                                                # [tets, edges, coords]

            ### quantities that depend on the eval points
            pvec = pts[tet_loc[:,OPP_IDCS]]
            d = ((pvec[:,:,None,:] - rvecplus)*nvec[:,:,None,:]).sum(axis=-1) # [tets, faces, edges]
            pvec -= nvec*(pvec*nvec).sum(axis=-1)[...,None]                   # [tets, faces, coords]

            lplus = ((pvecplus - pvec[:,:,None,:]) * lvec).sum(axis=-1)   # [tets, faces, edges]
            lminus = ((pvecminus - pvec[:,:,None,:]) * lvec).sum(axis=-1) # [tets, faces, edges]

            P0 = ((pvecplus - pvec[:,:,None,:])*uvec).sum(axis=-1) # [tets, faces, edges]
            Pplus_2 = P0**2 + lplus**2                             # [tets, faces, edges]
            Pminus_2 = P0**2 + lminus**2                           # [tets, faces, edges]

            R0_2 = P0**2 + d**2                 # [tets, faces, edges]
            Rplus = jnp.sqrt(Pplus_2 + d**2)    # [tets, faces, edges]
            Rminus = jnp.sqrt(Pminus_2 + d**2)  # [tets, faces, edges]

            logterm = jnp.log((Rplus + lplus)/(Rminus + lminus)) # [tets, faces, edges]
            tanterm = jnp.atan(P0 * lplus / (R0_2 + jnp.abs(d) * Rplus)) - jnp.atan(P0 * lminus / (R0_2 + jnp.abs(d) * Rminus)) # [tets, faces, edges]
            total = 0.5 * (d * (jnp.abs(d) * tanterm - P0*logterm)).sum(axis=-1)    # [tets, faces]

            term = volchrgs[:,None] * total

            ### turn the volume charge density into point-charges
            volchrgs = detsvol/(4*6) * volchrgs   # charge density times element volume divided across 4 point-charges

            ### subtract the inaccurate potential created by the other points in the tetrahedron
            dists = jnp.linalg.norm(pts[tet_loc[:,OPP_IDCS]][:,None,:,:]  - pts[tet_loc[:,OPP_IDCS]][:,:,None,:],axis=-1)
            dists = jnp.where(dists>1e-12,1/dists,0)
            term += (volchrgs[:,None,None] * dists).sum(axis=-1)
            corrpot_loc = corrpot_loc.at[tet_loc[:,OPP_IDCS]].add(term)

            ### sum up point-charges on the same vertices
            return corrpot_loc, None

        corrpot = _scan_blocked(corrpot, tets, None if Ms_is_scalar else Ms, scan_body,
                                mem_limit, 24 * pts.itemsize)

    ### surface charges
    if(tris is not None):
        Ms_surf_is_scalar = (jnp.size(Ms_surf) < 2)

        def scan_body_surf(buf, xs):
            if(Ms_surf_is_scalar):
                tri_loc = xs
                Ms_surf_loc = Ms_surf
            else:
                tri_loc, Ms_surf_loc = xs
            corrpot_loc = buf

            ### compute surface charge density
            ### numerics: ALL positions are localized to the triangle's third vertex.
            ### The density parametrization sigma(x) = Cvec.x + C and the linterm below
            ### are affine in ABSOLUTE position; with f32 coordinates at mesh scales
            ### (1e-5 m positions, 1e-7 m cells) the absolute-coordinate anchor loses
            ### ~5 of 7 significant digits to cancellation — measured: the same star
            ### evaluated at two lattice translations differed by O(1) x its value, and
            ### a pure cyclic vertex rotation shifted it ~50%. In local coordinates the
            ### identical math is translation-invariant and the anchor term vanishes.
            org = pts[tri_loc[:,2]]                                # [tris, coords]
            ptl = pts[tri_loc] - org[:,None,:]                     # [tris, 3, coords], vertex 2 -> 0
            T = jnp.zeros((tri_loc.shape[0],tri_loc.shape[1],3))
            T = T.at[:,:2,:].set(ptl[:,:2])
            T = T.at[:,2,:].set(jnp.cross(T[:,0],T[:,1],axis=-1))  # guarantee invertible T by choosing normal vector
            nvecnorm = jnp.linalg.norm(T[:,2,:],axis=-1)
            nvec = (T[:,2,:]/nvecnorm[:,None])[:,None,:]
            T, _ = matinv_3x3(T) # NOTE: determinant != surface area
            surfchrgs = Ms_surf_loc[:,None] * (nvec*m[tri_loc]).sum(axis=-1)

            chrgdiff = surfchrgs.at[:,:2].set(surfchrgs[:,:2] - surfchrgs[:,2,None])    # choosing the third component this way has advantages for later!
            Cvec = jnp.einsum('ikj, ik -> ij',T,chrgdiff)
            C = surfchrgs[:,2]                                     # local frame: vertex 2 at the origin

            ### computing quantities related to the edges
            jrange = jnp.arange(3)
            rvecminus = ptl[:,jrange][:,None,:,:]      # [tris, pts_in_tris, edges, coords]
            rvecplus = ptl[:,(jrange+1)%3][:,None,:,:] # [tris, pts_in_tris, edges, coords]
            nvec = nvec[:,None,:,:]

            lvec = rvecplus - rvecminus       # [tris, pts_in_tris, edges, coords]
            l = jnp.linalg.norm(lvec,axis=-1) # [tris, pts_in_tris, edges]
            lvec /= l[...,None]

            pvecplus = rvecplus - nvec * (nvec*rvecplus).sum(axis=-1)[...,None]    # [tris, pts_in_tris, edges, coords]
            pvecminus = rvecminus - nvec * (nvec*rvecminus).sum(axis=-1)[...,None] # [tris, pts_in_tris, edges, coords]
            uvec = jnp.cross(lvec,nvec,axis=-1) # [tris, pts_in_tris, edges, coords]

            ### quantities that depend on the eval points
            rvec = ptl[:,:,None,:]             # [tris, pts_in_tris, edges, coords] (local frame)
            pvec = rvec - nvec * (rvec*nvec).sum(axis=-1)[...,None]   # [tris, pts_in_tris, edges, coords]

            lplus = ((pvecplus - pvec) * lvec).sum(axis=-1)   # [tris, pts_in_tris, edges]
            lminus = ((pvecminus - pvec) * lvec).sum(axis=-1) # [tris, pts_in_tris, edges]

            P0 = ((pvecplus - pvec)*uvec).sum(axis=-1) # [tris, pts_in_tris, edges]
            Pplus = jnp.sqrt(P0**2 + lplus**2)
            Pminus = jnp.sqrt(P0**2 + lminus**2)

            logterm = jnp.nan_to_num(jnp.log((Pplus + lplus)/(Pminus + lminus)),nan=0,posinf=0,neginf=0) # [tris, pts_in_tris, edges]
            temp = P0 * logterm # [tris, pts_in_tris, edges]
            constterm = temp.sum(axis=-1) # [tris, pts_in_tris]
            linterm = (temp[...,None] * (pvec + nvec * (nvec * rvecminus).sum(axis=-1)[...,None]) + 0.5 * uvec * (P0**2 * logterm + lplus * Pplus - lminus * Pminus)[...,None]).sum(axis=-2)   # [tris, pts_in_tris, coords]

            term = C[:,None] * constterm + (Cvec[:,None,:] * linterm).sum(axis=-1)

            ### transform charge density into point charges 
            surfchrgs *= nvecnorm[...,None]/6 # scale charges with area

            ### subtract wrong self-interactions
            srcs = ptl[:,(jnp.arange(2)[None,:] + jrange[:,None]+1)%3]
            trg = ptl[:,(2+jrange+1)%3]
            term -= (surfchrgs[...,(jnp.arange(2)[None,:] + jrange[:,None]+1)%3]/(jnp.linalg.norm(srcs - trg[:,:,None,:], axis=-1))).sum(axis=(-1))
            corrpot_loc = corrpot_loc.at[tri_loc].add(term)

            ### place charges
            return corrpot_loc, None

        corrpot = _scan_blocked(corrpot, tris, None if Ms_surf_is_scalar else Ms_surf,
                                scan_body_surf, mem_limit, 3**3 * 3 * pts.itemsize)

    return corrpot/(4*jnp.pi)

def gen_chrgs_and_corr(m, Ms, Ms_surf, pts, tets, tris, compute_corr = True, mem_limit = jnp.inf, **kwargs):
    r"""Vertex-lumped P1-FEM charges plus (optionally) the analytic near-field correction.

    A thin composition of :func:`gen_lumped_charges` and :func:`gen_correction`.

    Parameters
    ----------
    m, Ms, Ms_surf, pts, tets, tris
        As in :func:`gen_lumped_charges`.
    compute_corr : bool, optional
        If False, skip the analytic correction and return zeros for it.
    mem_limit : float, optional
        Byte budget forwarded to both stages. Default: unlimited.

    Returns
    -------
    chrgs : jnp.ndarray
        Lumped nodal charges of shape (N_vert,).
    corrpot : jnp.ndarray
        On-mesh correction potential of shape (N_vert,) (zeros if ``compute_corr`` is
        False).
    """
    chrgs = gen_lumped_charges(m, Ms, Ms_surf, pts, tets, tris, mem_limit = mem_limit)
    if(not compute_corr):
        return chrgs, jnp.zeros(m.shape[0], m.dtype)
    return chrgs, gen_correction(m, Ms, Ms_surf, pts, tets, tris, mem_limit = mem_limit)


def gen_lumped_ops(verts, tets, tris, Ms, Ms_surf, mem_limit = jnp.inf):
    r"""Precompute the static per-element operators of the lumped stray-field sweeps.

    The three per-evaluation mesh sweeps (charge lumping, analytic vertex correction,
    weak gradient) are all *linear in the nodal input* with purely geometric
    coefficients: the charge/correction share a single per-tet scalar
    :math:`\rho V = \mathrm{msfac}\,\sum_v \mathbf{a}_v\cdot(\mathbf{m}_v-\mathbf{m}_3)`
    (area vectors :math:`\mathbf{a}_v`), the correction adds
    :math:`\rho V\, G_v` at the tet's own vertices, and the weak-gradient payload is
    :math:`2\sum_v \mathbf{a}_v (u_v - u_3)` (the signed determinant cancels). This
    function evaluates the expensive geometry — including all analytic edge terms
    (logs/arctans) of :func:`gen_correction` — ONCE, so the per-step sweeps reduce to
    gather → tiny linear map → scatter.

    Parameters
    ----------
    verts : jnp.ndarray
        Vertex coordinates of shape (N_vert, 3).
    tets : jnp.ndarray or None
        Tetrahedron vertex indices of shape (N_tet, 4) (positively oriented, as
        enforced by :func:`init_strayfield`).
    tris : jnp.ndarray or None
        Boundary-triangle vertex indices of shape (N_tri, 3).
    Ms : float or jnp.ndarray
        Saturation magnetisation, scalar or per-tetrahedron.
    Ms_surf : float or jnp.ndarray
        Surface saturation magnetisation, scalar or per-triangle.
    mem_limit : float, optional
        Byte budget for the (one-time) geometry intermediates; elements are processed
        in blocks. Default: unlimited.

    Returns
    -------
    lops : dict
        ``"vol"``: (N_tet, 15) pack ``[a_0 a_1 a_2 (9) | det (1) | msfac (1) | G (4)]``
        with ``G`` in the tets' own column order (correction factor per vertex,
        :math:`1/4\pi` included). ``"surf"``: (N_tri, 13) pack
        ``[Ms_surf·n_hat (3) | A/3 (1) | K (9)]`` with ``K[trg, src]`` mapping the
        per-vertex surface density to the correction potential. Keys absent when the
        element family is absent. Memory: 60 B/tet + 52 B/tri (f32).
    """
    lops = {}
    fpi = 4 * jnp.pi
    itemsize = jnp.dtype(verts.dtype).itemsize
    SIGMA = jnp.array([1, 0, 2, 3])

    if tets is not None:
        Ms_per = (jnp.size(Ms) > 1)

        @jax.jit
        def vol_blk(tet_loc, Ms_loc):
            ev = verts[tet_loc]
            e = ev[:, :3] - ev[:, 3:4]
            an = 0.5 * jnp.stack([jnp.cross(e[:, 1], e[:, 2], axis=-1),
                                  jnp.cross(e[:, 2], e[:, 0], axis=-1),
                                  jnp.cross(e[:, 0], e[:, 1], axis=-1)], axis=1)   # (B,3v,3d)
            det = jnp.einsum("td,td->t", e[:, 0], jnp.cross(e[:, 1], e[:, 2], axis=-1))
            msfac = -(jnp.sign(det) / 3.0) * Ms_loc

            # canonical frame (det < 0) — same convention as gen_correction
            swap = det > 0
            tet_c = jnp.where(swap[:, None], tet_loc[:, SIGMA], tet_loc)

            nvec = jnp.cross(verts[tet_c[:, FACE_IDCS[:, 1]]] - verts[tet_c[:, FACE_IDCS[:, 0]]],
                             verts[tet_c[:, FACE_IDCS[:, 2]]] - verts[tet_c[:, FACE_IDCS[:, 0]]], axis=-1)
            nvec /= jnp.linalg.norm(nvec, axis=-1)[..., None]
            rvecminus = verts[tet_c[:, FACE_IDCS[:, :]]]
            rvecplus = verts[tet_c[:, FACE_IDCS[:, (1, 2, 0)]]]
            lvec = rvecplus - rvecminus
            lvec /= jnp.linalg.norm(lvec, axis=-1)[..., None]
            pvecplus = rvecplus - nvec[:, :, None, :] * (nvec[:, :, None, :] * rvecplus).sum(axis=-1)[..., None]
            pvecminus = rvecminus - nvec[:, :, None, :] * (nvec[:, :, None, :] * rvecminus).sum(axis=-1)[..., None]
            uvec = jnp.cross(lvec, nvec[:, :, None, :], axis=-1)
            pvec = verts[tet_c[:, OPP_IDCS]]
            d = ((pvec[:, :, None, :] - rvecplus) * nvec[:, :, None, :]).sum(axis=-1)
            pvec = pvec - nvec * (pvec * nvec).sum(axis=-1)[..., None]
            lplus = ((pvecplus - pvec[:, :, None, :]) * lvec).sum(axis=-1)
            lminus = ((pvecminus - pvec[:, :, None, :]) * lvec).sum(axis=-1)
            P0 = ((pvecplus - pvec[:, :, None, :]) * uvec).sum(axis=-1)
            R0_2 = P0**2 + d**2
            Rplus = jnp.sqrt(P0**2 + lplus**2 + d**2)
            Rminus = jnp.sqrt(P0**2 + lminus**2 + d**2)
            logterm = jnp.log((Rplus + lplus) / (Rminus + lminus))
            tanterm = (jnp.atan(P0 * lplus / (R0_2 + jnp.abs(d) * Rplus))
                       - jnp.atan(P0 * lminus / (R0_2 + jnp.abs(d) * Rminus)))
            total = 0.5 * (d * (jnp.abs(d) * tanterm - P0 * logterm)).sum(axis=-1)  # (B,4 faces)

            pp = verts[tet_c[:, OPP_IDCS]]
            dists = jnp.linalg.norm(pp[:, None, :, :] - pp[:, :, None, :], axis=-1)
            invsum = jnp.where(dists > 1e-12, 1 / jnp.where(dists > 1e-12, dists, 1), 0).sum(axis=-1)

            Gcan = (6.0 / jnp.abs(det))[:, None] * total - 0.25 * invsum            # (B,4) face axis
            colc = jnp.broadcast_to(OPP_IDCS[None, :], Gcan.shape)                  # canonical column
            colo = jnp.where(swap[:, None], SIGMA[colc], colc)                      # original column
            G = jnp.zeros_like(Gcan).at[jnp.arange(Gcan.shape[0])[:, None], colo].set(Gcan) / fpi
            return jnp.concatenate([an.reshape(-1, 9), det[:, None], msfac[:, None], G], axis=1)

        n = tets.shape[0]
        blk = n if not np.isfinite(mem_limit) else max(1, min(n, int(mem_limit // (200 * itemsize))))
        parts = []
        for i in range(0, n, blk):
            Ms_loc = Ms[i:i + blk] if Ms_per else jnp.asarray(Ms, verts.dtype)
            parts.append(vol_blk(tets[i:i + blk], Ms_loc))
        lops["vol"] = jax.block_until_ready(jnp.concatenate(parts) if len(parts) > 1 else parts[0])

    if tris is not None:
        Msf_per = (jnp.ndim(Ms_surf) > 0 and jnp.size(Ms_surf) > 1)

        @jax.jit
        def surf_blk(tri_loc, Msf_loc):
            T = jnp.zeros((tri_loc.shape[0], 3, 3), verts.dtype)
            T = T.at[:, :2, :].set(verts[tri_loc[:, :2]] - verts[tri_loc[:, 2, None]])
            T = T.at[:, 2, :].set(jnp.cross(T[:, 0], T[:, 1], axis=-1))
            nvecnorm = jnp.linalg.norm(T[:, 2, :], axis=-1)
            nvec = (T[:, 2, :] / nvecnorm[:, None])[:, None, :]
            Tinv, _ = matinv_3x3(T)

            jrange = jnp.arange(3)
            rvecminus = verts[tri_loc[:, jrange]][:, None, :, :]
            rvecplus = verts[tri_loc[:, (jrange + 1) % 3]][:, None, :, :]
            nv = nvec[:, None, :, :]
            lvec = rvecplus - rvecminus
            lvec /= jnp.linalg.norm(lvec, axis=-1)[..., None]
            pvecplus = rvecplus - nv * (nv * rvecplus).sum(axis=-1)[..., None]
            pvecminus = rvecminus - nv * (nv * rvecminus).sum(axis=-1)[..., None]
            uvec = jnp.cross(lvec, nv, axis=-1)
            rvec = verts[tri_loc][:, :, None, :]
            pvec = rvec - nv * (rvec * nv).sum(axis=-1)[..., None]
            lplus = ((pvecplus - pvec) * lvec).sum(axis=-1)
            lminus = ((pvecminus - pvec) * lvec).sum(axis=-1)
            P0 = ((pvecplus - pvec) * uvec).sum(axis=-1)
            Pplus = jnp.sqrt(P0**2 + lplus**2)
            Pminus = jnp.sqrt(P0**2 + lminus**2)
            logterm = jnp.nan_to_num(jnp.log((Pplus + lplus) / (Pminus + lminus)), nan=0, posinf=0, neginf=0)
            temp = P0 * logterm
            constterm = temp.sum(axis=-1)                                            # (B,3 trg)
            linterm = (temp[..., None] * (pvec + nv * (nv * rvecminus).sum(axis=-1)[..., None])
                       + 0.5 * uvec * (P0**2 * logterm + lplus * Pplus - lminus * Pminus)[..., None]
                       ).sum(axis=-2)                                                # (B,3 trg,3 d)

            # inverse distances between the 3 vertices for the point-charge subtraction
            tv = verts[tri_loc]
            dmat = jnp.linalg.norm(tv[:, None, :, :] - tv[:, :, None, :], axis=-1)
            dinv = jnp.where(dmat > 1e-12, 1 / jnp.where(dmat > 1e-12, dmat, 1), 0)  # (B,trg,src)

            # basis evaluation of the linear map density -> correction potential
            eye = jnp.eye(3, dtype=verts.dtype)
            def one_basis(sb):                                                        # sb: (3,) unit density
                chrgdiff = sb.at[:2].add(-sb[2])
                Cvec = jnp.einsum("tkj,k->tj", Tinv, chrgdiff)                        # (B,3)
                C = sb[2] - (Cvec * verts[tri_loc[:, 2]]).sum(axis=1)
                term = C[:, None] * constterm + (Cvec[:, None, :] * linterm).sum(axis=-1)
                sub = ((sb * nvecnorm[:, None] / 6)[:, None, :] * dinv).sum(axis=-1)  # diag(dinv)=0
                return term - sub
            K = jnp.stack([one_basis(eye[b]) for b in range(3)], axis=-1) / fpi       # (B,trg,src)
            nS = Msf_loc[:, None] * nvec[:, 0] if jnp.ndim(Msf_loc) else Msf_loc * nvec[:, 0]
            return jnp.concatenate([nS, (nvecnorm / 6)[:, None], K.reshape(-1, 9)], axis=1)

        n = tris.shape[0]
        blk = n if not np.isfinite(mem_limit) else max(1, min(n, int(mem_limit // (150 * itemsize))))
        parts = []
        for i in range(0, n, blk):
            Msf_loc = Ms_surf[i:i + blk] if Msf_per else jnp.asarray(Ms_surf, verts.dtype)
            parts.append(surf_blk(tris[i:i + blk], Msf_loc))
        lops["surf"] = jax.block_until_ready(jnp.concatenate(parts) if len(parts) > 1 else parts[0])
    return lops


def _lumped_ops_chrgs_corr(m, lops, tets, tris, n_vert, mem_limit):
    r"""Fused charges + vertex-correction sweep from precomputed operators (one pass
    per element family; payload channel 0 = lumped charge, channel 1 = correction)."""
    itemsize = jnp.dtype(m.dtype).itemsize
    q2 = jnp.zeros((n_vert, 2), m.dtype)
    if tets is not None and "vol" in lops:
        def vol_fn(tet_blk, p):
            an = p[:, :9].reshape(-1, 3, 3)
            rhoV = p[:, 10] * jnp.einsum("tvd,tvd->t",
                                         an, m[tet_blk[:, :3]] - m[tet_blk[:, 3], None])
            return jnp.stack([jnp.broadcast_to((rhoV / 4)[:, None], tet_blk.shape),
                              rhoV[:, None] * p[:, 11:15]], axis=-1)
        q2 = _scatter_blocked(q2, tets, lops["vol"], vol_fn, mem_limit, 60 * itemsize)
    if tris is not None and "surf" in lops:
        def surf_fn(tri_blk, p):
            s = jnp.einsum("td,tvd->tv", p[:, :3], m[tri_blk])
            return jnp.stack([p[:, 3, None] * s,
                              jnp.einsum("tij,tj->ti", p[:, 4:13].reshape(-1, 3, 3), s)], axis=-1)
        q2 = _scatter_blocked(q2, tris, lops["surf"], surf_fn, mem_limit, 40 * itemsize)
    return q2[:, 0], q2[:, 1]


def _lumped_ops_adjoint(w0, w1, lops, tets, tris, n_vert, mem_limit):
    r"""Transpose of :func:`_lumped_ops_chrgs_corr`: nodal cotangents ``w0`` (charge
    channel) and ``w1`` (correction channel) back to magnetisation space. The exact
    adjoint of the fused assembly, written with the same packs and blocking — the
    energy-consistent field is ``-adjoint(u + corr/2, c/2) / Vlump``."""
    itemsize = jnp.dtype(w0.dtype).itemsize
    g = jnp.zeros((n_vert, 3), w0.dtype)
    if tets is not None and "vol" in lops:
        def vol_fn(tet_blk, p):
            an = p[:, :9].reshape(-1, 3, 3)
            s = w0[tet_blk].sum(axis=1) / 4.0 + jnp.einsum("tv,tv->t",
                                                           p[:, 11:15], w1[tet_blk])
            gv = (p[:, 10] * s)[:, None, None] * an           # nodes 0..2 of d rhoV/d m
            return jnp.concatenate([gv, -gv.sum(axis=1, keepdims=True)], axis=1)
        g = _scatter_blocked(g, tets, lops["vol"], vol_fn, mem_limit, 60 * itemsize)
    if tris is not None and "surf" in lops:
        def surf_fn(tri_blk, p):
            t = p[:, 3, None] * w0[tri_blk] + jnp.einsum("tij,ti->tj",
                                                         p[:, 4:13].reshape(-1, 3, 3),
                                                         w1[tri_blk])
            return t[:, :, None] * p[:, None, :3]             # t_v * (Ms_surf n_hat)
        g = _scatter_blocked(g, tris, lops["surf"], surf_fn, mem_limit, 40 * itemsize)
    return g


def _lumped_ops_grad(u, lops_vol, tets, Vlump, mem_limit):
    r"""Weak gradient from precomputed operators: payload :math:`2\sum_v a_v (u_v-u_3)`
    per tet (signed det cancels), normalised by the static lumped volume."""
    itemsize = jnp.dtype(u.dtype).itemsize
    H = jnp.zeros((Vlump.shape[0], 3), u.dtype)
    def g_fn(tet_blk, p):
        an = p[:, :9].reshape(-1, 3, 3)
        Ht = 2.0 * jnp.einsum("tv,tvd->td", u[tet_blk[:, :3]] - u[tet_blk[:, 3], None], an)
        return jnp.broadcast_to(Ht[:, None, :], tet_blk.shape + (3,))
    H = _scatter_blocked(H, tets, lops_vol, g_fn, mem_limit, 60 * itemsize)
    return H / (24.0 * Vlump[:, None])


def _geom_rescale(verts, kwargs):
    r"""Normalize the geometry to O(1) units (see ``auto_rescale``).

    Computes the characteristic scale from the pinned cell (``domain`` /
    ``L0_boxlen``) or the vertex AABB and divides every length-bearing input by
    it: verts, domain, L0_boxcen/L0_boxlen, basis_offsets, eval_pts;
    ``bloch_k`` carries units 1/length and is multiplied instead. The Laplace
    stray field H is exactly invariant under this rescale (the demag kernel is
    homogeneous of degree -1 and H is dimensionless in Ms); the scalar
    potential picks up one factor of the scale, restored at the eval seams via
    the stored ``geom_scale``. Inputs already in O(1) units get a scale near 1
    and the rescale degrades to a no-op.
    """
    dom = kwargs.get("domain")
    if dom is not None:
        sc = float(np.max(np.asarray(dom[1])))
    elif kwargs.get("L0_boxlen") is not None:
        sc = float(np.max(np.asarray(kwargs["L0_boxlen"])))
    else:
        v = np.asarray(verts[: min(len(verts), 65536)], dtype=np.float64)
        sc = float((v.max(0) - v.min(0)).max())
    if not np.isfinite(sc) or sc <= 0:
        return verts, kwargs, 1.0
    kwargs = dict(kwargs)
    if dom is not None:
        kwargs["domain"] = (jnp.asarray(dom[0]) / sc, jnp.asarray(dom[1]) / sc)
    for key in ("L0_boxcen", "L0_boxlen", "eval_pts"):
        if kwargs.get(key) is not None:
            kwargs[key] = jnp.asarray(kwargs[key]) / sc
    if kwargs.get("basis_offsets") is not None:
        kwargs["basis_offsets"] = [jnp.asarray(o) / sc for o in kwargs["basis_offsets"]]
    if kwargs.get("bloch_k") is not None:
        kwargs["bloch_k"] = jnp.asarray(kwargs["bloch_k"]) * sc
    return verts / sc, kwargs, sc


def init_strayfield_elements(verts, tets = None, Ms = None, tris = None, Ms_surf = None,
                             eval_pts = None, err_tol = None, p = None, theta = None,
                             N_max = None, diff = False,
                             drop_periodic_surface_charges = True, **kwargs):
    r"""Element-based (non-lumped) stray-field setup: charges INSIDE the elements.

    The higher-accuracy alternative to :func:`init_strayfield`: the P0 volume /
    P1 surface charge densities are integrated per element (Gauss quadrature +
    Duffy self terms) instead of being lumped onto vertices, so there is no
    vertex-rule near-field error and **arbitrary evaluation points are supported —
    including near or inside the mesh**, where the lumped pipeline refuses to
    evaluate. Costlier per evaluation than the lumped path (more source points);
    accuracy is set by ``p``/``theta``/``duffy_deg`` of the element engine.

    Parameters
    ----------
    verts : (N_vert, 3) jnp.ndarray
        Vertex coordinates.
    tets : (N_tet, 4) jnp.ndarray of int, optional
        Tetrahedra (positively oriented, cf. :func:`jaxfmm.fem.mesh_util.fix_tet_orientations`);
        ``None`` disables volume charges.
    Ms : (N_tet,) or (1,) jnp.ndarray, optional
        Saturation magnetization per tetrahedron (or scalar).
    tris : (N_tri, 3) jnp.ndarray of int, optional
        Boundary triangles; ``None`` auto-extracts them from ``tets``/``Ms``
        (:func:`get_tris`). Pass an empty array to disable surface charges.
    Ms_surf : (N_tri,) or (1,) jnp.ndarray, optional
        Surface saturation magnetization; ``None`` auto-extracts with ``tris``.
    eval_pts : (N_eval, 3) jnp.ndarray, optional
        Evaluation points; default ``None`` evaluates at ``verts`` (on-mesh).
        The target set must be large enough to form a tree with a far field
        (realistic sensor planes/grids are fine; a handful of isolated probe
        points is currently not supported by the element engine).
    p, theta, N_max : int, float, int, optional
        Element-engine far-field order, MAC parameter and leaf capacity
        (defaults ``4, 0.5, 128``); further options (``near_deg``, ``duffy_deg``,
        ``s``, ...) pass through to :func:`jaxfmm.fem.element_farfield_setup`.
    diff : bool, optional
        Build for ``field_mode='diff'`` (default False): the evaluation set becomes the
        engine's own near-rule quadrature points (incompatible with ``eval_pts``), so
        ``H_i = -dE/dm_i / (Ms V)_i`` pairs the discrete energy exactly in the near zone
        (asym ~3e-5). The element self-blocks of that energy are the exact Galerkin
        self-interaction (``_diff_dJ_t``/``_diff_dJ_s``), not the coincidence-zeroed
        point sum -- without it the energy carries an O(h) floor (unit cube
        ``N - 1/3 = -9.2e-3`` at h = 0.1). What remains is the near-rule quadrature
        error of TOUCHING neighbour pairs, still O(h) but ~10x smaller (+8e-4 at
        h = 0.1, f64, any p/theta); the pointwise ``'grad'`` field averaged with the
        lumped volumes converges as h^1.8 instead. Use ``'diff'`` where energy
        consistency matters (energy-stable time stepping, eigensolves), ``'grad'`` for
        the field itself.
    drop_periodic_surface_charges : bool, optional
        With ``periodic_axes``, remove boundary triangles lying on periodic cell
        faces (default True) — same convention as :func:`init_strayfield`: a
        periodic continuation has no material boundary there, so a uniformly
        magnetized fully-periodic medium carries no charges (**H = 0**, the
        infinite-medium Maxwell solution, not Lorentz −M/3).

    Returns
    -------
    dict
        Setup for :func:`eval_strayfield_elements` /
        :func:`compile_strayfield_elements` (element trees, interaction lists,
        quadrature factors, plus the mesh/material arrays).

    Notes
    -----
    **Periodic boundaries.** Pass ``periodic_axes`` (and ``domain=(pmin, extent)``
    to pin the cell; ``basis_offsets`` for sublattices) — forwarded to
    :func:`jaxfmm.fem.element_farfield_setup`. Element sources are quadrature
    points strictly inside elements, so there are no zero-distance image pairs and
    no vertex-correction port is needed. Cut-out cell meshes still evaluate at
    duplicated periodic-face *targets* with one-sided near quadrature — interior
    nodes are clean (same caveat class as the lumped path, milder mechanism).
    """
    from jaxfmm.util import validate_points
    from jaxfmm.fem import element_farfield_setup
    validate_points(verts, "verts")
    auto_rescale = kwargs.pop("auto_rescale", True)
    geom_scale = 1.0
    if auto_rescale:
        verts, kwargs, geom_scale = _geom_rescale(verts, kwargs)
        if eval_pts is not None:
            eval_pts = jnp.asarray(eval_pts) / geom_scale
    if tets is not None:
        validate_points(tets, "tets", n_cols=4, integer=True)
        if jnp.any(get_tet_volumes(verts, tets) < 0):
            raise ValueError("Inconsistent tetrahedron node ordering (negative volumes). "
                             "Consider using jaxfmm.fem.mesh_util.fix_tet_orientations.")
    if tris is None or Ms_surf is None:
        if tets is None or Ms is None:
            raise ValueError("Must set either tets, tris or both (and the corresponding "
                             "saturation magnetization).")
        tris, Ms_surf = get_tris(verts, tets, Ms)
    if tris is not None and tris.size < 1:
        tris, Ms_surf = None, None
    periodic_axes = kwargs.get("periodic_axes", ())
    if drop_periodic_surface_charges and len(periodic_axes) > 0 and tris is not None:
        from jaxfmm.util import parse_domain
        _cen, _len = parse_domain(kwargs.get("domain", None),
                                  kwargs.get("L0_boxcen", None), kwargs.get("L0_boxlen", None))
        tris, Ms_surf, n_drop = filter_periodic_boundary_tris(
            verts, tris, Ms_surf, periodic_axes, _cen, _len)
        if n_drop:
            print(f"init_strayfield_elements: dropped {n_drop} surface triangles on periodic "
                  f"cell faces (axes {tuple(periodic_axes)}) — no material boundary under "
                  f"periodic continuation.")
    # err_tol ladder (jaxfmm.fem.tolerances): resolve tolerance -> concrete parameters ONCE,
    # here at setup — moving-geometry/shape-opt drivers built on this setup inherit a FROZEN
    # config (they consume the resolved values, never re-derive). Explicit kwargs always win.
    from jaxfmm.fem.tolerances import resolve, DEFAULT_TOL, QUALITY_FLOOR
    import warnings as _w
    _x64 = jnp.asarray(verts).dtype == jnp.float64
    _ov = dict(p=p, theta=theta, N_max=N_max,
               **{k: kwargs.pop(k) for k in ("s", "near_deg", "near_theta2", "duffy_deg",
                                             "far_deg")
                  if k in kwargs})
    _ov = {k: v for k, v in _ov.items() if v is not None}
    # perf-only knobs: override the rung values but do NOT trigger the ladder bypass
    _perf = {k: kwargs.pop(k) for k in ("N_max_eval", "s_eval") if k in kwargs}
    if err_tol is None and _ov:
        # Explicit accuracy parameters WITHOUT err_tol: legacy semantics. The ladder must
        # not fill the remaining knobs from its fast rung -- an explicit p=6 caller would
        # silently get the fast rung's near_deg=(1, 3) near field (measured: the PBC
        # duplicated-lattice gate moved 4.6e-2 -> 5.2e-2 exactly this way). Unset knobs
        # fall through to element_farfield_setup's own defaults, as before 0.3.x.
        params, _info = dict(_ov), dict(tol=None, clamped=False)
    else:
        params, _info = resolve(DEFAULT_TOL if err_tol is None else float(err_tol), _x64, _ov)
    params.update(_perf)
    if err_tol is not None and _info["clamped"]:
        _w.warn(f"init_strayfield_elements: err_tol={err_tol:g} is tighter than the best "
                f"admissible rung ({_info['tol']:g}"
                + ("" if _x64 else "; float64 inputs unlock the tighter f64 rungs") + ").",
                stacklevel=2)
    if _info["tol"] is not None and tets is not None and tets.shape[0]:
        # mesh-quality proxy vs the calibration suite (vol/edge_max^3, 1 = regular tet):
        # the err_tol indication is calibrated on median quality >= ~0.2 — below it the
        # near/quadrature error can floor ABOVE the requested tolerance (measured: 3.3:1
        # aspect cells floor at 4-7e-3 up to R3; only the (5,6)+ surface rules of R4+
        # get below it — see fem/tolerances.py).
        import numpy as _np
        _sub = _np.asarray(tets[:: max(1, tets.shape[0] // 200_000)])
        _ev = _np.asarray(verts, _np.float64)[_sub]
        _vol = _np.abs(_np.linalg.det(_ev[:, 1:] - _ev[:, :1])) / 6
        _em = _np.max([_np.linalg.norm(_ev[:, a] - _ev[:, b], axis=1)
                       for a, b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))], axis=0)
        _qall = _vol / _em ** 3 * 6 * _np.sqrt(2)
        _q = float(_np.median(_qall))
        _q1 = float(_np.percentile(_qall, 1))
        if _q < QUALITY_FLOOR:
            _w.warn(f"init_strayfield_elements: mesh quality proxy {_q:.2f} is below the "
                    f"err_tol calibration range (>= {QUALITY_FLOOR}); the error indication "
                    f"may not be reached on this mesh (high-aspect/sliver elements).",
                    stacklevel=2)
        elif _q1 < QUALITY_FLOOR:
            # tail statistic: a LOCAL defect cluster evades the median (measured: a
            # 4-incident-tet gmsh anomaly, 1st percentile 0.178 in a 0.40-median mesh,
            # carried an entire eigensolver stationarity-gate failure; QUALITY_FLOOR/2
            # missed it). Not comparable across mesh families: a PERFECT cubic Kuhn tet
            # scores sqrt(2)/(3 sqrt(3)) = 0.272 on this proxy and drops below 0.2 at
            # cell aspect >= 1.5, so anisotropic structured meshes fire by construction
            # (and rightly: 3.3:1 cells floor at 4-7e-3 on every rung up to R3).
            _w.warn(f"init_strayfield_elements: mesh quality 1st percentile {_q1:.2f} "
                    f"(median {_q:.2f}) — sliver tail present; per-node quantities "
                    f"(gates, torque residuals) may be dominated by it. Structured "
                    f"(Kuhn-split) meshes score <= 0.27 here and reach this at cell "
                    f"aspect >= 1.5 — expected, not a mesh defect.",
                    stacklevel=2)
    _diff_split = None
    if diff:
        # field_mode='diff' pairs the discrete energy at the engine's own NEAR-RULE
        # quadrature points, E = 1/2 sum_pt q_pt u(x_pt): the near zone (which dominates
        # the pairing) is then EXACTLY symmetric and the far asymmetry is MAC-suppressed.
        # Measured on a 10^3 cube (f64, p=5/theta=0.5): asym 3.5e-5, vs 5.2e-2 for the
        # vertex/centroid collocation of the original eigensolver reference and 7.8e-3
        # for far-rule points. The evaluation set is fixed here at setup.
        if eval_pts is not None:
            raise ValueError("diff=True builds its own evaluation set (the near-rule "
                             "quadrature points) and is incompatible with separate "
                             "eval_pts.")
        if tets is None:
            raise ValueError("diff=True requires tetrahedra.")
        from jaxfmm.fem.quadrature import simplex_quad, element_quad_points
        _nd = params.get("near_deg", None)
        if _nd is None:
            _nd = (3, 4)                    # mirror element_farfield_setup's default
        _ndv, _nds = _nd if isinstance(_nd, (tuple, list)) else (_nd, _nd + 1)
        # identical construction to the setup's own near rule (same simplex_quad /
        # element_quad_points on the same rescaled verts -> bitwise the same points)
        _bvn, _ = simplex_quad(3, _ndv)
        _qt, _ = element_quad_points(verts, tets, jnp.asarray(_bvn))
        _parts = [_qt.reshape(-1, 3)]
        if tris is not None:
            _bsn, _ = simplex_quad(2, _nds)
            _qs, _ = element_quad_points(verts, tris, jnp.asarray(_bsn))
            _parts.append(_qs.reshape(-1, 3))
        eval_pts = jnp.concatenate(_parts, axis=0)
        _diff_split = int(_qt.shape[0]) * int(_qt.shape[1])
    s = element_farfield_setup(verts, tets, tris, verts if eval_pts is None else eval_pts,
                               **params, **kwargs)
    s["err_tol_resolved"] = _info["tol"]
    s["params_resolved"] = dict(params)      # resolved setup knobs (rung + overrides), metadata only
    s.update(_mag_verts=verts, _mag_tets=tets, _mag_tris=tris, _mag_Ms=Ms,
             _mag_Ms_surf=Ms_surf, _mag_mesh_eval=(eval_pts is None),
             geom_scale=geom_scale)
    if diff:
        import numpy as _np
        _ev = _np.asarray(verts, _np.float64)[_np.asarray(tets)]
        _vol = _np.abs(_np.einsum("td,td->t", _ev[:, 0] - _ev[:, 3],
                                  _np.cross(_ev[:, 1] - _ev[:, 3], _ev[:, 2] - _ev[:, 3]))) / 6.0
        _Msv = (_np.broadcast_to(_np.asarray(Ms, _np.float64).reshape(-1), _vol.shape)
                if Ms is not None else _np.ones_like(_vol))
        _msv = _np.zeros(verts.shape[0])
        _np.add.at(_msv, _np.asarray(tets).reshape(-1), _np.repeat(_Msv * _vol / 4.0, 4))
        # Self-energy correction: at the near-rule points an element's interaction with
        # ITSELF is a plain point sum with the coincident pairs dropped (element_farfield_setup
        # only Duffy-corrects vertex targets) -- an O(h) energy floor (unit cube N - 1/3 =
        # -9.2e-3 at h = 0.1, ~h^1.0, p/theta-independent; the surface self-triangles
        # dominate). _self_energy_geom returns the exact symmetric Galerkin self-block
        # minus that point sum; its gradient is added in _element_diff_field.
        from jaxfmm.fem.elements import _self_energy_geom
        _dd = params.get("duffy_deg", 4)
        _dJt = _self_energy_geom(verts, tets, _dd, _qt, s["wm_tet_n"], _bvn).sum(axis=(1, 2))
        _dJs = (_self_energy_geom(verts, tris, _dd, _qs, s["wm_tri_n"], _bsn)
                if tris is not None else None)
        # zero-guard like the lumped diff: no moment -> no torque -> field value irrelevant
        s.update(_diff_split=_diff_split,
                 _diff_msv=jnp.asarray(_np.where(_msv > 0, _msv, 1.0), verts.dtype),
                 _diff_dJ_t=_dJt, _diff_dJ_s=_dJs)
    return s


_MAG_KEYS = ("_mag_verts", "_mag_tets", "_mag_tris", "_mag_Ms", "_mag_Ms_surf",
             "_mag_mesh_eval")
_DIFF_KEYS = ("_diff_split", "_diff_msv", "_diff_dJ_t", "_diff_dJ_s")


def _elements_charges(m, s):
    r"""P0/P1 charge densities for the element pipeline from a magnetization state."""
    return gen_element_charges(m, s["_mag_Ms"] if s["_mag_Ms"] is not None else 1.0,
                               s["_mag_Ms_surf"], s["_mag_verts"], s["_mag_tets"],
                               s["_mag_tris"])


def _element_diff_field(s, u_all, m):
    r"""Energy-consistent H: ``-Q^T u / (Ms V)_lumped`` at the near-rule pairing.

    ``E = 1/2 q^T G q`` with ``q = Q m`` the engine's own near-rule point charges and
    ``G`` the (symmetric) point kernel gives ``dE/dm = Q^T u`` with ``u`` the potential
    at those same points — one forward FMM plus one transposed assembly sweep, never
    differentiating through the FMM. Pairing at the near-rule points makes the dominant
    near zone exactly symmetric (measured asym 3.5e-5 vs 5.2e-2 for vertex/centroid
    collocation); the residual is the far/tier skew of the FMM's approximation of G.
    The element SELF-blocks of G are replaced by the exact Galerkin self-interaction
    (``E += 1/2 rho^T dJ_t rho + 1/2 sum_tri sig^T dJ_s sig``, see
    :func:`jaxfmm.fem.elements._self_energy_geom`) -- the point sum alone is an O(h)
    energy floor. Conjugations follow the Bloch convention of the complex path (no-ops
    for real m).
    """
    ns = s["_diff_split"]
    tris = s["_mag_tris"]
    Ms = s["_mag_Ms"] if s["_mag_Ms"] is not None else 1.0
    u_t = u_all[:ns].reshape(-1, s["wm_tet_n"].shape[1])
    rho0, sig0 = gen_element_charges(m, Ms, s["_mag_Ms_surf"], s["_mag_verts"],
                                     s["_mag_tets"], tris)
    _4pi = 4 * jnp.pi                                   # u carries 1/(4 pi); dJ is 4pi-free
    if tris is None:
        def _q(mm):
            rho, _ = gen_element_charges(mm, Ms, s["_mag_Ms_surf"], s["_mag_verts"],
                                         s["_mag_tets"], None)
            return rho[:, None] * s["wm_tet_n"], rho
        cot = (jnp.conj(u_t), jnp.conj(s["_diff_dJ_t"] * rho0) / _4pi)
    else:
        u_s = u_all[ns:].reshape(-1, s["wm_tri_n"].shape[1])

        def _q(mm):
            rho, sig = gen_element_charges(mm, Ms, s["_mag_Ms_surf"], s["_mag_verts"],
                                           s["_mag_tets"], tris)
            return (rho[:, None] * s["wm_tet_n"],
                    jnp.einsum("qk,ek->eq", s["bary_tri_n"], sig) * s["wm_tri_n"], rho, sig)
        cot = (jnp.conj(u_t), jnp.conj(u_s), jnp.conj(s["_diff_dJ_t"] * rho0) / _4pi,
               jnp.conj(jnp.einsum("ekl,el->ek", s["_diff_dJ_s"], sig0)) / _4pi)
    (g,) = jax.linear_transpose(_q, m)(cot)
    return -jnp.conj(g) / s["_diff_msv"][:, None]


def _element_grad_ok(meta, field_mode):
    r"""Guard: the weak-gradient field mode is only defined for on-mesh evaluation."""
    if field_mode not in ("direct", "grad", "diff"):
        raise ValueError(f"field_mode must be 'direct', 'grad' or 'diff', got {field_mode!r}.")
    if field_mode == "diff" and meta.get("_diff_split") is None:
        raise ValueError("field_mode='diff' needs a setup built with "
                         "init_strayfield_elements(diff=True): it evaluates the potential "
                         "at the mesh vertices AND the tet centroids.")
    if field_mode == "grad" and not meta["_mag_mesh_eval"]:
        raise ValueError("field_mode='grad' needs on-mesh evaluation (build without "
                         "eval_pts): the weak gradient is defined on the mesh. Use "
                         "field_mode='direct' for separate targets.")


def eval_strayfield_elements(m, potential = False, field_mode = None, mem_limit = None,
                             kernel_dtype = None, kernel_scale = None, **s):
    r"""Stray field (or potential) from the element-based setup.

    Parameters
    ----------
    m : (N_vert, 3) jnp.ndarray
        Normalized magnetization at the vertices.
    potential : bool, optional
        Return the scalar potential ``u`` instead of ``H`` (default False).
    field_mode : {'grad', 'direct', 'diff'}, optional
        ``None`` (default) resolves to ``'grad'`` on mesh evaluation — the weak-gradient
        H, the same recovery route as the lumped path — ``'diff'`` on a setup built with
        ``init_strayfield_elements(diff=True)`` (the variationally consistent
        H_i = -dE/dm_i / (Ms V)_i, the element analogue of the lumped ``mode='diff'``;
        one forward FMM at the near-rule pairing points plus a transposed assembly,
        measured 1.7x 'grad'), and ``'direct'`` for separate ``eval_pts``. Pass
        ``'direct'`` explicitly for the direct-kernel H on the mesh (escapes the
        weak-gradient floor, ~3x the cost).
        How ``H`` is obtained (ignored if ``potential``). ``'direct'`` is the
        analytic field :math:`-\nabla G` of the charges — accurate at interior/off-mesh
        points, but the tangential surface field is log-divergent so on-mesh boundary
        nodes are quadrature-regularised. ``'grad'`` evaluates the (high-accuracy element)
        potential at the mesh nodes and takes its weak gradient
        (:func:`get_lumped_strayfield`) — the physically meaningful, surface-robust field;
        on-mesh only.
    mem_limit : float, optional
        Peak-transient byte budget (0.3.1 semantics). Default ``None`` uses the
        setup's hardware-adaptive budget (``mem_limit_auto``); pass ``jnp.inf``
        explicitly for the historical unbatched behaviour.
    **s
        The setup dict from :func:`init_strayfield_elements`.

    Returns
    -------
    (N_eval, 3) or (N_eval,) jnp.ndarray
        ``H`` at the evaluation points (or ``u`` for ``potential=True``).
    """
    from jaxfmm.fem import element_eval
    if field_mode is None:      # default: the weak-gradient H on the mesh (the same
        # DG0->CG1 recovery route as the lumped path); 'diff' on a diff=True setup (its
        # concatenated evaluation set exists for exactly that); 'direct' for separate
        # targets or an explicit request (direct-kernel H, ~3x cost, escapes the
        # weak-grad floor)
        field_mode = ("diff" if s.get("_diff_split") is not None
                      else "grad" if s["_mag_mesh_eval"] else "direct")
    grad = (not potential) and field_mode == "grad"
    diff = (not potential) and field_mode == "diff"
    _element_grad_ok(s, field_mode if not potential else "direct")
    rho, sig = _elements_charges(m, s)
    core = {k: v for k, v in s.items() if k not in _MAG_KEYS and k not in _DIFF_KEYS}
    out = element_eval(core, rho, sig, field=(not potential) and not (grad or diff),
                       mem_limit=mem_limit,
                       kernel_dtype=kernel_dtype, kernel_scale=kernel_scale)
    if grad:
        return get_lumped_strayfield(out, s["_mag_verts"], s["_mag_tets"])
    if diff:
        return _element_diff_field(s, out, m)
    return out * s.get("geom_scale", 1.0) if potential else out


def compile_strayfield_elements(s, potential = False, field_mode = None, mem_limit = None,
                                kernel_dtype = None, kernel_scale = None):
    r"""Frozen-geometry element stray-field evaluator: one jitted ``f(m) -> H``.

    Parameters
    ----------
    s : dict
        Setup from :func:`init_strayfield_elements`.
    potential : bool, optional
        Compile the scalar-potential evaluator instead (default False).
    field_mode : {'grad', 'direct', 'diff'}, optional
        ``None`` (default) resolves to ``'grad'`` on mesh evaluation — the weak-gradient
        H, the same recovery route as the lumped path — ``'diff'`` on a setup built with
        ``init_strayfield_elements(diff=True)`` (the variationally consistent
        H_i = -dE/dm_i / (Ms V)_i, the element analogue of the lumped ``mode='diff'``;
        one forward FMM at the near-rule pairing points plus a transposed assembly,
        measured 1.7x 'grad'), and ``'direct'`` for separate ``eval_pts``. Pass
        ``'direct'`` explicitly for the direct-kernel H on the mesh (escapes the
        weak-gradient floor, ~3x the cost).
        How ``H`` is obtained (ignored if ``potential``); see
        :func:`eval_strayfield_elements`. ``'direct'`` is the analytic field;
        ``'grad'`` is the surface-robust weak gradient of the potential (on-mesh only).
        ``None`` (default) resolves as there: ``'grad'`` on mesh, ``'direct'`` off mesh.
    mem_limit : float, optional
        Near-field batch budget in bytes (static).

    Returns
    -------
    callable
        Jitted ``f(m) -> H`` (or ``u``); geometry and interaction lists are
        frozen, only the magnetization changes between calls.
    """
    meta = {k: s[k] for k in _MAG_KEYS}
    meta["_diff_split"] = s.get("_diff_split")
    if field_mode is None:
        field_mode = ("diff" if meta["_diff_split"] is not None
                      else "grad" if meta["_mag_mesh_eval"] else "direct")
    grad = (not potential) and field_mode == "grad"
    diff = (not potential) and field_mode == "diff"
    _element_grad_ok(meta, field_mode if not potential else "direct")
    # The WHOLE setup (mag geometry + core) must reach the jit as ARGUMENTS: a closure
    # capture bakes every array as a constant of the outer trace — XLA then constant-folds
    # gathers over them into multi-GB executables (4.69 GB captured at 1e7; the CUBIN
    # cannot even load under XLA_PYTHON_CLIENT_MEM_FRACTION=0.95). Nesting the frozen
    # element_compile inside an outer jit re-captures its arrays the same way, so the
    # split happens here, once, at the top level.
    from jaxfmm.util import split_static
    skel, arrs = split_static(s)
    _gs = s.get("geom_scale", 1.0)
    return partial(_strayfield_elem_frozen, arrs=arrs, skel=skel, potential=potential,
                   grad=grad, diff=diff, gs=_gs, mem_limit=mem_limit,
                   kernel_dtype=kernel_dtype, kernel_scale=kernel_scale)


@partial(jax.jit, static_argnames=["skel", "potential", "grad", "diff", "gs", "mem_limit",
                                   "kernel_dtype", "kernel_scale"])
def _strayfield_elem_frozen(m, arrs, skel, potential, grad, gs, mem_limit,
                            kernel_dtype=None, kernel_scale=None, diff=False):
    r"""Module-level jitted frozen element stray field (setup as (skeleton, arrays))."""
    from jaxfmm.util import join_static
    from jaxfmm.fem import element_eval
    s = join_static(skel, arrs)
    rho, sig = _elements_charges(m, s)
    core = {k: v for k, v in s.items() if k not in _MAG_KEYS and k not in _DIFF_KEYS}
    out = element_eval(core, rho, sig, field=(not potential) and not (grad or diff),
                       mem_limit=mem_limit,
                       kernel_dtype=kernel_dtype, kernel_scale=kernel_scale)
    if grad:
        return get_lumped_strayfield(out, s["_mag_verts"], s["_mag_tets"])
    if diff:
        return _element_diff_field(s, out, m)
    return out * gs if potential else out


def fulljit_strayfield_driver(verts, tets, Ms, tris = None, Ms_surf = None,
                              field_mode = None, targets = None, **driver_kwargs):
    r"""Guarded moving-geometry stray field: certified ``f(verts, m) -> H`` (or ``u``).

    The magnetics front door to :func:`jaxfmm.driver.element_quadtree_fulljit_driver`:
    both the geometry AND the magnetisation are traced call arguments, so rigid-body
    motion (or any deformation) plus an evolving ``m`` run recompile-free while the
    frozen buffer caps certify (rebuilds happen transparently on overflow and cost no
    retrace when the refit caps repeat). Uses the ELEMENT pipeline
    (:func:`gen_element_charges`): charges sit inside the elements, so — unlike the
    lumped pipeline — arbitrary ``targets`` including close ones are valid.

    Parameters
    ----------
    verts : jnp.ndarray
        Reference vertex positions of shape (N_vert, 3) (caps are fitted here).
    tets : jnp.ndarray
        Tetrahedron vertex indices of shape (N_tet, 4).
    Ms : float or jnp.ndarray
        Saturation magnetisation, scalar or per-tetrahedron.
    tris, Ms_surf : optional
        Boundary triangles / surface Ms; extracted via :func:`get_tris` when omitted.
    field_mode : {'grad', 'direct', 'potential'}, optional
        ``None`` (default) resolves to ``'grad'`` on mesh evaluation and ``'direct'``
        when separate ``targets`` are given.
        ``'direct'``: analytic field (volume self-term only — surface-node H
        is better served by ``'grad'``, see §7 of the architecture notes).
        ``'grad'``: potential forward + mass-lumped weak gradient at the CURRENT verts
        (on-mesh only; returns a plain callable, not the driver object).
        ``'potential'``: scalar potential.
    targets : jnp.ndarray, optional
        Fixed external evaluation points (e.g. a sensor plane). ``field_mode`` must be
        ``'direct'`` or ``'potential'`` with targets.
    **driver_kwargs
        Forwarded to :func:`jaxfmm.driver.element_quadtree_fulljit_driver`
        (``err_tol``, ``p``, ``theta``, ``N_max``, ``headroom``, ``size_from``, ``mem_limit``,
        ...). Without ``err_tol`` and accuracy knobs the interior rung I1 is used.

    Returns
    -------
    :class:`jaxfmm.driver.GuardedShapeEvaluator` or callable
        ``f(verts, m) -> H/u`` — certified and auto-rebuilding. For
        ``field_mode='grad'`` a thin callable wrapping the potential driver (exposed
        as ``f.driver``).
    """
    from jaxfmm.driver import element_quadtree_fulljit_driver
    if tris is None or Ms_surf is None:
        tris, Ms_surf = get_tris(verts, tets, Ms)
    if field_mode is None:
        field_mode = "direct" if targets is not None else "grad"
    if field_mode == "grad" and targets is not None:
        raise ValueError("field_mode='grad' is on-mesh only (the weak gradient lives on "
                         "the mesh); use field_mode='direct' for external targets.")

    def charge_fn(v, m):
        return gen_element_charges(m, Ms, Ms_surf, v, tets, tris)

    drv = element_quadtree_fulljit_driver(verts, tets, tris, charge_fn,
                                          field = (field_mode == "direct"),
                                          targets = targets, **driver_kwargs)
    if field_mode != "grad":
        return drv

    def f(v, m):
        return get_lumped_strayfield(drv(v, m), v, tets)
    f.driver = drv
    return f


# selection table: (max err_tol, path, p, theta) — from measured discretisation floors
# on well-shaped meshes. First row whose threshold >= err_tol wins.
_LUMPED_PLAN = dict(p=3, theta=0.85)       # lumped opt-in: the per-timestep vertex-charge config


def plan_strayfield(verts, tets=None, Ms=None, tris=None, Ms_surf=None, err_tol=None,
                    eval_pts=None, potential=False, field_mode="grad", pipeline="element",
                    verbose=True, **kwargs):
    r"""Build a stray-field evaluator ``f(m) -> H`` from one accuracy knob.

    The one-call front door for the stray field (mirrors :func:`jaxfmm.plan` for the point
    FMM). The default is the **element pipeline** (:func:`init_strayfield_elements`):
    charges integrated inside the elements, with ``err_tol`` resolved on the calibrated
    rung ladder of :mod:`jaxfmm.fem.tolerances` — ``None`` selects the fast rung R0
    (indication 7.5e-3), tighter values the cheapest rung indicating at or below them.
    Rungs below 2.1e-3 need float64 inputs; on float32 inputs the request clamps to the
    best f32 rung with a warning.

    ``pipeline='lumped'`` opts into vertex-lumped charges (:func:`init_strayfield`,
    ``p=3``, ``theta=0.85``; ``err_tol`` is ignored). It is faster, but its charge model
    shifts the physics measurably (standard problem 4: a -4.7 ps phase lead), so it is
    never a default. Separate ``eval_pts`` always use the element pipeline (the lumped
    vertex correction only exists on the mesh).

    ``field_mode`` chooses how ``H`` is obtained (ignored if ``potential``):

    * ``'grad'`` (default): the weak gradient of the potential — the physically meaningful,
      **surface-robust** field, and the same object for both pipelines (lumped:
      :func:`get_lumped_strayfield`; element: the element potential then its weak
      gradient). On-mesh only. This is the default because surface nodes are common in
      micromagnetics and the direct field misbehaves there.
    * ``'diff'`` (lumped only): the energy-consistent gradient ``H = -dE/dm/V`` via autodiff.
    * ``'direct'`` (element only): the analytic field :math:`-\nabla G` of the charges —
      more accurate at interior/off-mesh points, but the tangential surface field is
      log-divergent so on-mesh boundary nodes are quadrature-regularised.

    Off-mesh ``eval_pts`` use ``'direct'`` regardless (the weak gradient is defined on the
    mesh). In ``'grad'`` mode the field carries the weak-gradient operator floor
    (mesh-dependent, order ``5e-3``, shrinking with refinement); for a field accurate to a
    tight ``err_tol`` at interior/off-mesh points use ``'direct'`` (element ``'direct'``
    interior error was ~7e-5 where ``'grad'`` was ~5e-3).

    ``err_tol`` is a calibration-suite indication on reasonable-quality meshes, not a
    certified per-call bound: it cannot beat the mesh (refine rather than lower
    ``err_tol``) nor the working precision.

    Parameters
    ----------
    verts, tets, Ms, tris, Ms_surf
        Mesh and material arrays, as for :func:`init_strayfield`.
    err_tol : float, optional
        Target error indication of the element pipeline (default ``None`` -> fast rung).
    eval_pts : (N_eval, 3) jnp.ndarray, optional
        Off-mesh evaluation points (element pipeline only). Default: at ``verts``.
    potential : bool, optional
        Return the scalar potential ``u`` instead of ``H`` (default False).
    field_mode : {'grad', 'diff', 'direct'}, optional
        How the field is obtained (default ``'grad'``); see above.
    pipeline : {'element', 'lumped'}, optional
        Charge model (default ``'element'``).
    verbose : bool, optional
        Print the chosen pipeline and config (default True).
    **kwargs
        Forwarded to the underlying ``init_strayfield``/``init_strayfield_elements``
        (explicit ``p``/``theta``/... override the rung values).

    Returns
    -------
    callable
        ``f(m) -> H`` (or ``u`` if ``potential``). Carries a ``.info`` dict
        (``path``, ``p``, ``theta``, ``err_tol`` = the resolved rung indication or ``None``,
        ``field_mode``).
    """
    if pipeline not in ("element", "lumped"):
        raise ValueError(f"pipeline must be 'element' or 'lumped', got {pipeline!r}")
    path = "element" if eval_pts is not None else pipeline

    if path == "lumped":
        if field_mode == "direct":
            warnings.warn("plan_strayfield: field_mode='direct' needs the element pipeline; "
                          "the lumped pipeline uses the weak gradient.",
                          RuntimeWarning, stacklevel=2)
        lump_mode = "diff" if field_mode == "diff" else "lump"
        cfg = {**_LUMPED_PLAN, **{k: kwargs.pop(k) for k in ("p", "theta") if k in kwargs}}
        h = init_strayfield(verts, tets, Ms, tris, Ms_surf, jit_cnct=True, **cfg, **kwargs)
        f = lambda m: eval_strayfield(m, potential=potential, mode=lump_mode, **h)
        resolved, tol = ("diff" if lump_mode == "diff" else "grad"), None
    else:
        if field_mode == "diff":
            raise ValueError("field_mode='diff' (energy gradient) is only implemented for the "
                             "lumped pipeline; use 'grad' or 'direct' for the element path.")
        fm = "direct" if eval_pts is not None else field_mode   # off-mesh -> analytic field
        s = init_strayfield_elements(verts, tets=tets, Ms=Ms, tris=tris, Ms_surf=Ms_surf,
                                     eval_pts=eval_pts, err_tol=err_tol, **kwargs)
        f = compile_strayfield_elements(s, potential=potential, field_mode=fm)
        cfg, resolved, tol = s["params_resolved"], fm, s["err_tol_resolved"]

    f.info = {"path": path, "p": cfg.get("p"), "theta": cfg.get("theta"), "err_tol": tol,
              "field_mode": resolved}
    if verbose:
        print(f"plan_strayfield: {path} pipeline (p={f.info['p']}, theta={f.info['theta']}, "
              f"indication={tol}, field_mode={resolved})", flush=True)
    return f



GET_TRIS_BYTES_PER_TET = 200      # measured working-set scale of get_tris (faces x temporaries)

def _available_bytes():
    r"""Conservative memory bound for :func:`get_tris`'s pre-flight check: the minimum of
    host MemAvailable and (when jax defaults to an accelerator) the device budget."""
    avail = float("inf")
    try:
        avail = next(float(l.split()[1]) * 1024.0 for l in open("/proc/meminfo")
                     if l.startswith("MemAvailable"))
    except Exception:
        pass
    try:
        dev = jax.devices()[0]
        if dev.platform != "cpu":
            avail = min(avail, float(dev.memory_stats()["bytes_limit"]))
    except Exception:
        pass
    return avail

def get_tris(verts, tets, Ms):
    r"""Extract the charged surface triangles of a volume mesh.

    Every face where the saturation magnetisation jumps carries a surface charge;
    contributions of the two adjacent tetrahedra cancel wherever Ms is continuous, so
    only faces with a net jump are returned.

    Parameters
    ----------
    verts : jnp.ndarray
        Vertex coordinates of shape (N_vert, 3).
    tets : jnp.ndarray
        Tetrahedron vertex indices of shape (N_tet, 4).
    Ms : jnp.ndarray
        Per-tetrahedron saturation magnetisation of shape (N_tet,).

    Returns
    -------
    tris : jnp.ndarray
        Surface triangles of shape (N_tri, 3), ordered so the effective Ms jump is
        positive under the normal convention
        :math:`\mathbf{n} = (v_0 - v_2)\times(v_1 - v_2)`.
    Ms_surf : jnp.ndarray
        Effective surface magnetisation of shape (N_tri,), collapsed to shape (1,) when
        uniform.
    """
    # pre-flight: the extraction materialises ~GET_TRIS_BYTES_PER_TET bytes per tet
    # (4 faces x index/normal/unique temporaries). A 1e9-tet mesh needs ~200 GB — refuse
    # loudly instead of OOM-killing the host: the surface is usually known upstream.
    projected = tets.shape[0] * GET_TRIS_BYTES_PER_TET
    limit = 0.75 * _available_bytes()
    if projected > limit:
        raise MemoryError(
            f"get_tris: extracting the boundary of {tets.shape[0]:,} tets needs "
            f"~{projected/2**30:.0f} GiB (working set), above the available "
            f"~{limit/2**30:.0f} GiB. Pass the surface directly instead "
            f"(init_strayfield(..., tris=, Ms_surf=) — mesh generators usually know it), "
            f"or run get_tris once on a machine with enough memory and cache the result.")
    tets = jnp.sort(tets, axis=-1)
    perm = jnp.array([[0,1,2],
                      [0,1,3],
                      [0,2,3],
                      [1,2,3]])
    notin = jnp.array([3,2,1,0])

    tris = tets[:,perm]
    nvecs = jnp.cross(verts[tris[:,:,2]] - verts[tris[:,:,0]], verts[tris[:,:,1]] - verts[tris[:,:,0]], axis=-1)
    Ms_contrib = jnp.sign(((verts[tets[:,notin]] - verts[tris[:,:,0]]) * nvecs).sum(axis=-1)) * Ms[:,None] # account for nvec orientation
    Ms_contrib = Ms_contrib.flatten()
    tris = tris.reshape((-1,3))

    uq_tris, uq_inv = jnp.unique(tris,axis=0,return_inverse=True)
    Ms_surf = jnp.zeros(uq_tris.shape[0])
    Ms_surf = jnp.add.at(Ms_surf, uq_inv, Ms_contrib, inplace=False)

    ### only take nonzero Ms_surf, check if it can be expressed as a scalar
    keep = jnp.abs(Ms_surf) > 1e-8
    uq_tris = uq_tris[keep]
    Ms_surf = Ms_surf[keep]
    flip = (Ms_surf < 0)
    uq_tris = uq_tris.at[flip,:].set(jnp.flip(uq_tris[flip],axis=-1))
    Ms_surf = Ms_surf.at[flip].multiply(-1)
    if(jnp.allclose(Ms_surf,Ms_surf[0])):
        Ms_surf = Ms_surf[0,None]

    return uq_tris, Ms_surf

@partial(jax.jit, static_argnames=['mem_limit'])
def get_lumped_strayfield(u, pts, tets, mem_limit = jnp.inf, **kwargs):
    r"""Turn a CG1 potential into a mass-lumped CG1 field.

    Differentiates the nodal (CG1) potential ``u`` to the per-tetrahedron constant (DG0)
    field :math:`\mathbf{H} = -\nabla u` and projects it back onto the vertices by a
    volume-weighted average over the adjacent tetrahedra.

    Parameters
    ----------
    u : jnp.ndarray
        Nodal potential of shape (N_vert,).
    pts : jnp.ndarray
        Vertex coordinates of shape (N_vert, 3).
    tets : jnp.ndarray
        Tetrahedron vertex indices of shape (N_tet, 4).
    mem_limit : float, optional
        Byte budget for the per-batch intermediates; tetrahedra are processed in a scan
        of padded batches when exceeded. Default: unlimited.

    Returns
    -------
    Hlump : jnp.ndarray
        Mass-lumped nodal field of shape (N_vert, 3).
    """
    def scan_body(buf, tet_loc):
        Hlump, fac = buf

        ### DG0 field times volume, via face area vectors: the P1 gradient identity
        ### T[v,:] = 2 a_v/det6 makes (nabla u)*det6 = 2 sum_v a_v (u_v - u_3) — the
        ### det6 of the volume weighting cancels the one in the gradient, so no
        ### matrix inverse and no division at all (robust on degenerating tets).
        e = pts[tet_loc[:,:3]] - pts[tet_loc[:,3],None]
        cr12 = jnp.cross(e[:,1], e[:,2], axis=-1)
        an = 0.5 * jnp.stack([cr12, jnp.cross(e[:,2], e[:,0], axis=-1),
                              jnp.cross(e[:,0], e[:,1], axis=-1)], axis=1)
        det6 = jnp.einsum("td,td->t", e[:,0], cr12)
        du = u[tet_loc[:,:3]] - u[tet_loc[:,3],None]
        H = 2.0 * jnp.einsum("tv,tvd->td", du, an)          # = (DG0 grad) * det6

        ### project back onto CG1
        Hlump = Hlump.at[tet_loc].add(H[:,None,:])  # add contributions to nodes
        fac = fac.at[tet_loc].add(jnp.abs(det6)[:,None])    # add to total adjacent volume

        return (Hlump, fac), None

    buf = (jnp.zeros((pts.shape[0],3), u.dtype), jnp.zeros((pts.shape[0])))   # field follows u (complex under Bloch)
    Hlump, fac = _scan_blocked(buf, tets, None, scan_body, mem_limit, 3**2 * pts.itemsize)
    Hlump /= fac[:,None]    # divide by total adjacent volume for weighted average
    return Hlump

def filter_periodic_boundary_tris(verts, tris, Ms_surf, periodic_axes, L0_boxcen=None,
                                  L0_boxlen=None, rtol=5e-3, naln=0.9):
    r"""Drop boundary triangles lying on the periodic cell faces.

    A truly periodic medium has no material boundary across a periodic direction, so the
    :math:`\mathbf{M}\cdot\mathbf{n}` surface charges of those cut faces must not exist —
    each face's charge would otherwise double-count against the neighbouring image's
    opposite face, and undropped wall charges make the periodic field diverge near the
    cell boundary. Two mesh conventions are covered: torus-topology meshes (nodes
    identified across the cell) produce no such faces and this is a no-op; cut-out block
    meshes produce real cell-face triangles and this removes exactly those. Faces normal
    to non-periodic axes are always kept (a film periodic in x, y keeps its top/bottom
    charges).

    A triangle is identified as a periodic-wall face when its **normal aligns with the
    periodic axis** (``|n . axis| >= naln``) *and* its centroid lies within
    ``rtol * cell_extent`` of that axis' min/max plane. The normal test is what makes
    this robust to real (gmsh / ``from_msh``) meshes whose periodic side faces are not
    bit-exactly planar: a wall face keeps its axis-aligned normal under small coordinate
    noise, whereas a corrugated top/bottom face near the boundary has a normal pointing
    out of plane and is correctly kept even under a generous centroid tolerance. (An
    exact all-vertices-on-plane test with a tight tolerance silently leaks wall charges
    on such meshes.)

    Parameters
    ----------
    verts : jnp.ndarray
        Vertex coordinates of shape (N_vert, 3).
    tris : jnp.ndarray or None
        Boundary-triangle vertex indices of shape (N_tri, 3).
    Ms_surf : float or jnp.ndarray or None
        Surface saturation magnetisation; filtered only when per-triangle.
    periodic_axes : sequence of int
        Axes (0/1/2) that are periodic.
    L0_boxcen, L0_boxlen : array_like, optional
        Centre and edge lengths of the periodic cell (default: the mesh bounding box).
    rtol : float, optional
        Relative tolerance for the centroid-on-plane test (fraction of cell extent).
        The default tolerates the coordinate noise of conforming periodic meshes; the
        normal-alignment test guards against dropping genuine boundary faces.
    naln : float, optional
        Minimum ``|normal . axis|`` (cosine) for a face to count as axis-normal. The
        default 0.9 (~26 deg) admits noisy wall faces while excluding top/bottom faces.

    Returns
    -------
    tris_kept : jnp.ndarray or None
        Remaining triangles (``None`` if none remain).
    Ms_surf_kept : float or jnp.ndarray or None
        Correspondingly filtered surface magnetisation.
    n_dropped : int
        Number of triangles removed.
    """
    if tris is None or len(periodic_axes) == 0:
        return tris, Ms_surf, 0
    v = np.asarray(verts)
    lo = np.asarray(L0_boxcen) - np.asarray(L0_boxlen) / 2 if L0_boxcen is not None else v.min(0)
    hi = np.asarray(L0_boxcen) + np.asarray(L0_boxlen) / 2 if L0_boxcen is not None else v.max(0)
    tv = v[np.asarray(tris)]                                   # (N_tri, 3, 3)
    cen = tv.mean(axis=1)                                      # (N_tri, 3) centroids
    n = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])     # (N_tri, 3) face normals
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    drop = np.zeros(tv.shape[0], bool)
    for ax in periodic_axes:
        tol = rtol * max(hi[ax] - lo[ax], 1e-30)
        near = (np.abs(cen[:, ax] - lo[ax]) < tol) | (np.abs(cen[:, ax] - hi[ax]) < tol)
        drop |= near & (np.abs(n[:, ax]) >= naln)
    if not drop.any():
        return tris, Ms_surf, 0
    keep = jnp.asarray(~drop)
    tris_k = tris[keep]
    Ms_k = Ms_surf[keep] if (Ms_surf is not None and np.ndim(Ms_surf) > 0
                             and np.shape(Ms_surf)[0] == tv.shape[0]) else Ms_surf
    if tris_k.shape[0] == 0:
        tris_k, Ms_k = None, None
    return tris_k, Ms_k, int(drop.sum())


def init_strayfield(verts, tets = None, Ms = None, tris = None , Ms_surf = None,
                    drop_periodic_surface_charges = True, precompute_ops = False,
                    fold = None, **kwargs):
    r"""Generate the FMM hierarchy for a (P1-FEM) mesh.

    Any additional FMM options are forwarded to :func:`jaxfmm.trees.gen_hierarchy`.

    Parameters
    ----------
    verts : jnp.ndarray
        Vertex coordinates of shape (N_vert, 3).
    tets : jnp.ndarray, optional
        Integer array of shape (N_tet, 4) with tetrahedron vertex indices, i.e.
        ``verts[tets[i]]`` gives the corner positions of tetrahedron ``i``. The default
        (None) disables volume charges. The indices must follow a consistent ordering
        (see :func:`jaxfmm.fem.mesh_util.fix_tet_orientations`).
    Ms : jnp.ndarray, optional
        Float array of shape (N_tet,) or (1,) with the saturation magnetization of the
        individual tetrahedra or the entire volume.
    tris : jnp.ndarray, optional
        Integer array of shape (N_tri, 3) with surface-triangle vertex indices. The
        default (None) automatically determines the required surface triangles for the
        given ``tets`` and ``Ms``. To disable surface charges, pass an empty array
        (``tris = jnp.array([])``). Internally, the normal vector of triangle ``i`` is
        defined as :math:`(v_{t_{i0}} - v_{t_{i2}})\times(v_{t_{i1}} - v_{t_{i2}})`, so
        the ordering of ``tris`` and the signs of ``Ms_surf`` must be chosen accordingly.
    Ms_surf : jnp.ndarray, optional
        Float array of shape (N_tri,) or (1,) with the saturation magnetization on the
        individual surface triangles or the entire surface. The default (None)
        automatically determines it for the given ``tets`` and ``Ms``.
    drop_periodic_surface_charges : bool, optional
        With periodic axes, remove the boundary triangles lying on periodic cell faces
        (default True); see Notes.
    fold : jnp.ndarray, optional
        Integer array of shape (N_vert,) mapping every vertex to the representative
        vertex of its periodic DOF class (a representative maps to itself, so the map is
        idempotent). Supply it for a periodically folded FEM mesh whose duplicate partner
        vertices are passed unfolded; ``None`` (default) disables folding. See Notes.
    **kwargs
        Forwarded to :func:`jaxfmm.trees.gen_hierarchy` (e.g. ``periodic_axes``,
        ``L0_boxcen``/``L0_boxlen``, ``eval_pts``).

    Returns
    -------
    dict
        Hierarchy information to be fed to :func:`eval_strayfield`.

    Notes
    -----
    **Periodic boundaries.** Pass ``periodic_axes`` (and optionally
    ``domain=(pmin, extent)`` (or ``L0_boxcen``/``L0_boxlen``) to pin the unit cell). With
    ``drop_periodic_surface_charges=True`` the boundary triangles lying on periodic cell
    faces are removed (:func:`filter_periodic_boundary_tris`): a periodic continuation
    has no material boundary there. This implies a convention: a uniformly magnetized
    fully-periodic medium carries no charges at all, i.e. :math:`\mathbf{H} = 0` (the
    Maxwell solution for the infinite medium; codes using a Lorentz-cavity convention
    report :math:`\langle\mathbf{H}\rangle = -\mathbf{M}/3` instead).

    **Folded (cut-out cell) meshes — pass** ``fold``. On a cut-out cell mesh the duplicated
    min/max face vertices are *the same point* under periodic continuation, which corrupts
    them two ways: each partner sees the other's charge at zero distance (caught only by the
    ULP-relative P2P coincidence guard, which a one-ULP wrap can defeat), and the analytic
    vertex correction covers just that vertex's half of the true DOF star. The damage is
    local but severe and silent — measured up to :math:`80\,M_s` on a 2.2M-DOF cell while
    every bulk percentile through p99.99 stayed clean. Passing ``fold`` sums the charges
    *and* the correction of each DOF class onto its representative, leaves the duplicates
    chargeless and gathers the (lattice-periodic) potential back, which is exact and removes
    both effects; the same mesh then peaks at :math:`0.94\,M_s`. Torus-topology meshes have
    no duplicates and need no fold.

    **Separate targets.** Passing ``eval_pts`` evaluates at off-mesh targets *without*
    the vertex correction — valid only at distance (enforced: minimum target-mesh
    distance > 1.5 * max edge length, else ``ValueError``). For targets near or inside
    the mesh, use the element-based pipeline (:func:`init_strayfield_elements` /
    :func:`eval_strayfield_elements`), which places quadrature charges inside the
    elements and supports arbitrary targets.
    """
    from jaxfmm.util import validate_points
    from jaxfmm.fem.farfield import _check_harmonic_scale
    validate_points(verts, "verts")
    auto_rescale = kwargs.pop("auto_rescale", True)
    geom_scale = 1.0
    if auto_rescale:
        verts, kwargs, geom_scale = _geom_rescale(verts, kwargs)
    _check_harmonic_scale(verts, kwargs.get("p", 3))
    if tets is not None:
        validate_points(tets, "tets", n_cols=4, integer=True)
    if(tets is not None):
        # blocked: the whole-mesh gather (verts[tets], f64) is 16*8 B/tet — a 28 GiB
        # device transient at 2.2e8 tets that OOMed setups the eval itself fits
        neg = False
        for lo in range(0, tets.shape[0], 16_000_000):
            neg = neg or bool(jnp.any(get_tet_volumes(verts, tets[lo:lo + 16_000_000]) < 0))
        if(neg):
            raise ValueError("Inconsistent tetrahedron node ordering (negative volumes). Consider using jaxfmm.fem.mesh_util.fix_tet_orientations to reorder tets.")
        
        # ordering diagnostic: exact RCM reference below RCM_EXACT_LIMIT tets; above it the
        # full vertex adjacency alone needs O(100 GB) host RAM (it OOM-killed a 1e9-tet setup
        # merely to phrase a warning), so use the structural locality scale of a well-ordered
        # 3D mesh instead: RCM bandwidth ~ N_vert^(2/3) (exact for structured grids, the
        # bandwidth lower bound in general).
        meandistfrac = jnp.mean(jnp.ptp(tets,axis=1))
        if tets.shape[0] <= RCM_EXACT_LIMIT:
            meandistfrac /= jnp.mean(jnp.ptp(reorder_mesh_rcm(verts, tets)[1],axis=1))
        else:
            meandistfrac /= verts.shape[0] ** (2.0 / 3.0)
        if(meandistfrac > 10):
            warnings.warn("Inefficient vertex ordering (dist_frac = %.1f > 10), which can significantly decrease performance. Consider using jaxfmm.fem.mesh_util.reorder_mesh_rcm for a more efficient mesh reordering."%meandistfrac,
                          category=RuntimeWarning,
                          stacklevel=2)

    if(tris is None or Ms_surf is None):
        if(tets is None or Ms is None):
            raise ValueError("Must set either tets, tris or both (and the corresponding saturation magnetization).")
        tris, Ms_surf = get_tris(verts, tets, Ms)
    if(tris is not None and tris.size < 1):
        tris, Ms_surf = None, None

    periodic_axes = kwargs.get("periodic_axes", ())
    if drop_periodic_surface_charges and len(periodic_axes) > 0:
        from jaxfmm.util import parse_domain
        _cen, _len = parse_domain(kwargs.get("domain", None),          # accept both cell forms
                                  kwargs.get("L0_boxcen", None), kwargs.get("L0_boxlen", None))
        tris, Ms_surf, n_drop = filter_periodic_boundary_tris(
            verts, tris, Ms_surf, periodic_axes, _cen, _len)
        if n_drop:
            print(f"init_strayfield: dropped {n_drop} surface triangles on periodic cell faces "
                  f"(axes {tuple(periodic_axes)}) — no material boundary under periodic continuation.")

    hier = gen_hierarchy(verts, **kwargs)
    hier["geom_scale"] = geom_scale
    if(hier["eval_pts"] is not hier["pts"]):
        padded_pts = hier["pts"].at[hier["idcs"][2][0]].get(mode="fill",fill_value=0)
        padded_eval_pts = hier["eval_pts"].at[hier["idcs"][3][0]].get(mode="fill",fill_value=jnp.inf)
        min_dist = check_min_dist(padded_pts,padded_eval_pts,hier["dir_cnct"],hier["mem_limit"])
        elems = tets if tets is not None else tris
        ev = verts[elems]
        max_edge = jnp.max(jnp.linalg.norm(ev[:,None,:,:] - ev[:,:,None,:],axis=-1))
        if(min_dist < 1.5*max_edge):
            # NOT a warning: the vertex-lumped charges are a valid representation only at
            # distance (multipole convergence) or AT the vertices themselves (where the
            # analytic self-correction repairs the near field). For targets this close
            # there is no correction that can fix the lumping error.
            raise ValueError(
                "Separate evaluation points too close to the mesh for the vertex-lumped "
                "strayfield (min_dist = %.2e < 1.5*max_edge = %.2e): the analytic "
                "correction only exists ON the mesh vertices, so near-mesh accuracy is "
                "unrecoverable in this representation. Use the element-based strayfield "
                "instead (init_strayfield_elements / eval_strayfield_elements), which "
                "places quadrature charges inside the elements and supports arbitrary "
                "targets." % (min_dist, max_edge))

    hier["tets"] = tets
    hier["tris"] = tris
    hier["Ms"] = Ms
    hier["Ms_surf"] = Ms_surf
    hier["mesh_eval"] = (hier["eval_pts"] is hier["pts"])
    if fold is not None:
        fold = jnp.asarray(fold)
        if fold.shape != (verts.shape[0],):
            raise ValueError(f"fold must have one entry per vertex ({verts.shape[0]}), got "
                             f"{fold.shape}: it maps each vertex to the representative vertex "
                             "of its periodic DOF class.")
        if not bool(jnp.all(fold[fold] == fold)):
            raise ValueError("fold must be idempotent (fold[fold] == fold): every entry has to "
                             "be a representative vertex, which is its own representative.")
    hier["fold"] = fold
    if fold is not None and kwargs.get("bloch_k") is not None:
        # Bloch: the fold identifies duplicate vertices with their periodic
        # masters, but the Bloch field is only periodic UP TO e^{ik.L} across
        # the wrap — fold a duplicate's charge onto its master with the wrap
        # phase e^{ik.(x_master - x_v)} (unity for interior classes) and gather
        # the potential back with the conjugate. The pair stays exactly
        # adjoint, so the mode="diff" symmetry argument is unchanged.
        _kb = jnp.asarray(kwargs["bloch_k"], jnp.float64).reshape(3)
        hier["fold_phase"] = jnp.exp(1j * ((verts[fold] - verts) @ _kb))
    if tets is not None:                                            # nodal lumped volume V_i = int phi_i = sum_{tet>i} V_tet/4
        # blocked like the orientation check above: same 16*8 B/tet gather
        Vl = jnp.zeros(verts.shape[0])
        MsVl = jnp.zeros(verts.shape[0])
        Ms_a = jnp.asarray(Ms)
        for lo in range(0, tets.shape[0], 16_000_000):
            tb = tets[lo:lo + 16_000_000]
            ev = verts[tb]; Vt = jnp.abs(jnp.linalg.det(ev[:, 1:] - ev[:, :1])) / 6.0
            Vl = Vl.at[tb].add(jnp.broadcast_to((Vt / 4)[:, None], tb.shape))
            Msb = Ms_a[lo:lo + 16_000_000] if (Ms_a.ndim and Ms_a.shape[0] == tets.shape[0]) else Ms_a
            MsVt = jnp.broadcast_to(Msb, Vt.shape) * Vt
            MsVl = MsVl.at[tb].add(jnp.broadcast_to((MsVt / 4)[:, None], tb.shape))
        hier["Vlump"] = Vl
        # nodal lumped MOMENT (Ms V)_i — the mode="diff" normaliser: the energy is
        # quadratic in Ms (charges AND potential), so the effective field is
        # -dE/dm_i / (Ms V)_i, linear in Ms like the lumped field
        hier["MsVlump"] = MsVl
    if precompute_ops:
        from jaxfmm.near import ws_limit
        _w = hier["idcs"][0][0].shape[1]
        _ml = ws_limit(hier["mem_limit"], _w) if np.isfinite(hier["mem_limit"]) else jnp.inf
        hier["lops"] = gen_lumped_ops(verts, hier["tets"], hier["tris"], Ms, hier["Ms_surf"],
                                      mem_limit = _ml)
    return hier

@partial(jax.jit, static_argnames=['field', 'lvl_info', 'mem_limit', 'potential', 'src_ofs', 'trg_ofs', 'mesh_eval', 'mode'])
def eval_strayfield(m, potential = False, mesh_eval = True, mode = "lump", **hierarchy):
    r"""Compute the stray field for a hierarchy generated by :func:`init_strayfield`.

    Besides the hierarchy information, only ``m`` needs to be passed here.

    Parameters
    ----------
    m : jnp.ndarray
        Normalized magnetization of shape (N_vert, 3).
    potential : bool, optional
        Compute the scalar potential instead of the stray field itself.
    mesh_eval : bool, optional
        Whether evaluation points coincide with source points (default True). Must be
        False for hierarchies built with separate ``eval_pts`` (enforced): the vertex
        correction only exists on the mesh vertices.
    mode : {"lump", "diff"}, optional
        How the on-mesh field is obtained (only for ``mesh_eval`` and ``not potential``).
        ``"lump"`` (default) mass-lumps the spatially-differentiated DG0 field
        (:func:`get_lumped_strayfield`). ``"diff"`` instead returns the
        variationally-consistent effective field

        .. math:: H_i = -\frac{1}{(M_s V)_i}\frac{\partial E}{\partial m_i}, \qquad
                  E = \frac{1}{2}\sum_i c_i u_i,

        with :math:`E` the discrete demag energy (charge times potential, no spatial
        gradient) and :math:`(M_s V)_i` the nodal lumped moment (E is quadratic in
        :math:`M_s`, so this normalisation makes H linear in it, like ``"lump"``) — evaluated in closed form as
        the transposed charge/correction assembly applied to the potential (charges and
        correction are linear in ``m`` and the potential operator is symmetric), so the
        cost matches ``"lump"``: one forward FMM plus one assembly-shaped sweep, no
        differentiation through the FMM. The two agree in the volume mean; ``"diff"``
        is the exact energy gradient (better at the surface, where H jumps) and is the
        right choice when energy-force consistency matters — in particular for low-α
        dynamics, where the non-gradient residual of ``"lump"`` can pump grid-scale
        modes faster than damping removes them. ``"diff"`` requires tetrahedra.
    **hierarchy
        Hierarchy information from :func:`init_strayfield`.

    Returns
    -------
    jnp.ndarray
        Stray field H (mass-lumped or energy-consistent) of shape (N_eval, 3), or the
        scalar potential u of shape (N_eval,).
    """
    if(mesh_eval and hierarchy["eval_pts"].shape != hierarchy["pts"].shape):
        # trace-time shape check: the corrected on-mesh pipeline pairs the vertex
        # correction with the potential AT THE VERTICES — meaningless for separate targets.
        raise ValueError(
            "mesh_eval=True, but this hierarchy was built with separate eval_pts. The "
            "vertex correction is only defined on the mesh vertices; evaluate with "
            "mesh_eval=False (targets well away from the mesh), or use the element-based "
            "strayfield (init_strayfield_elements) for near/off-mesh targets.")
    # 0.3.2: the assembly sweeps (charges / vertex correction / weak gradient) batch at
    # the derived working set, not the budget — measured 1.7x faster at the historical
    # chunking (their "largest array" mem models understate the true working set, so
    # near-unbatched sweeps thrash). The FMM keeps the budget (it splits internally).
    from jaxfmm.near import ws_limit
    lops = hierarchy.pop("lops", None)                               # precomputed sweep operators
    _w = hierarchy["idcs"][0][0].shape[1]
    _ml = ws_limit(hierarchy["mem_limit"], _w) if np.isfinite(hierarchy["mem_limit"]) \
        else hierarchy["mem_limit"]
    _hs = {**hierarchy, "mem_limit": _ml}

    def _assemble(mm, compute_corr):
        if lops is not None:
            return _lumped_ops_chrgs_corr(mm, lops, hierarchy["tets"], hierarchy["tris"],
                                          hierarchy["pts"].shape[0], _ml)
        return gen_chrgs_and_corr(mm, compute_corr = compute_corr, **_hs)

    # Periodic DOF fold: duplicate partner vertices of a cut-out cell mesh are the SAME
    # point under periodic continuation, so their charges (and corrections) belong on one
    # representative. Scattering them there leaves the duplicates chargeless — removing the
    # zero-distance image pairs the P2P guard would otherwise have to catch — and the
    # potential is gathered back because it is lattice-periodic. Both operations are exact;
    # `_scatter` and its transpose `_gather` are identity without a fold.
    _fold = hierarchy.get("fold", None)
    _fphase = hierarchy.get("fold_phase", None)
    if _fold is None:
        _scatter = _gather = lambda a: a
    elif _fphase is None:
        _scatter = lambda a: jnp.zeros_like(a).at[_fold].add(a)
        _gather = lambda a: a[_fold]
    else:  # Bloch: wrap-phased fold (see init_strayfield); adjoint pair
        _scatter = lambda a: jnp.zeros(a.shape, jnp.result_type(a.dtype, _fphase.dtype)).at[_fold].add(_fphase * a)
        _gather = lambda a: jnp.conj(_fphase) * a[_fold]

    if(mesh_eval and not potential and mode == "diff"):              # energy-consistent field
        if(hierarchy["tets"] is None):
            raise ValueError("mode='diff' requires tetrahedra.")
        # Closed-form gradient of E = 1/2 c.(G c + corr): both c and corr are LINEAR
        # in m, and G (the FMM potential operator) is symmetric, so
        #     dE/dm = A^T (u + corr/2) + C^T c / 2
        # with A^T, C^T the transposed assembly maps — one forward FMM plus a
        # transposed (gather/scatter) assembly, never differentiating through the
        # FMM itself. Exact up to the skew part of the FMM approximation of G.
        c, cr = _assemble(m, True)
        # With a fold the quadratic form becomes E = 1/2 c^T P^T(G P c + P corr) with P the
        # scatter: G -> P^T G P (still symmetric) and corr -> P^T P corr, so the adjoint sees
        # the gathered potential, correction and charge. Without a fold all three are no-ops.
        u = _gather(eval_potential(_scatter(c), **hierarchy))
        cr, c = _gather(_scatter(cr)), _gather(_scatter(c))
        if lops is not None:                                         # hand-written blocked adjoint
            g = _lumped_ops_adjoint((u + 0.5 * cr).astype(m.dtype), (0.5 * c).astype(m.dtype),
                                    lops, hierarchy["tets"], hierarchy["tris"],
                                    hierarchy["pts"].shape[0], _ml)
        else:                                                        # legacy sweeps: vjp of the
            _, pull = jax.vjp(lambda mm: _assemble(mm, True), m)     # (linear) assembly only
            (g,) = pull(((u + 0.5 * cr).astype(c.dtype), (0.5 * c).astype(cr.dtype)))
        # normalise by the nodal lumped MOMENT (Ms V)_i, not the volume: E is quadratic
        # in Ms, so this makes H linear in Ms (consistent with "lump"). Zero-guard for
        # nonmagnetic stars (no moment -> no torque -> field value irrelevant).
        MsVl = hierarchy.get("MsVlump", hierarchy["Vlump"])
        return -g / jnp.where(MsVl > 0, MsVl, 1.0)[:, None]          # H_i = -dE/dm_i / (Ms V)_i

    chrgs, corr = _assemble(m, mesh_eval)
    if(not mesh_eval):
        # off-mesh targets: fold the sources, but there is nothing to gather back
        res = eval_potential(_scatter(chrgs), field = not potential, **hierarchy)
        return res * hierarchy.get("geom_scale", 1.0) if potential else res
    else:
        if(hierarchy["tets"] is None):
            raise ValueError("Cannot compute lumped strayfield without tetrahedra.")
        u = _gather(eval_potential(_scatter(chrgs), **hierarchy) + _scatter(corr))
        if(potential):
            return u * hierarchy.get("geom_scale", 1.0)
        # blocked at the ws_limit working set, NOT merely capped by mem_limit: unbatched, the
        # weak gradient's 9-floats-per-tet transient sits on top of the live hierarchy (H100,
        # 1.9e8 tets: GPU peak 10.6 -> 19.2 GB at equal step time; 1e9 tets: OOM)
        if lops is not None:
            return _lumped_ops_grad(u, lops["vol"], hierarchy["tets"], hierarchy["Vlump"], _ml)
        H = get_lumped_strayfield(u, **_hs)
    return H

def anahexa0(x,y,z,a,b):
    r"""Stray field of a uniformly charged rectangle (building block of :func:`anahexa`).

    Field at ``(x, y, z)`` of the rectangle :math:`[-a, a]\times[-b, b]` in the
    :math:`z = 0` plane carrying unit surface charge density, summed in closed form
    over the four corners:

    .. math:: h_x \propto \log(r - t), \quad h_y \propto \log(r - s), \quad
              h_z \propto \arctan\frac{s\,t}{r\,z}.

    Parameters
    ----------
    x, y, z : (N,) jnp.ndarray
        Evaluation-point coordinates.
    a, b : float
        Half-edge lengths of the rectangle along x and y.

    Returns
    -------
    h : jnp.ndarray
        Field of shape (N, 3) (factor :math:`1/4\pi` included).
    """
    h = jnp.zeros((x.shape[0],3))
    for i in range(2):
        for j in range(2):
            s = x - (-1)**i * a
            t = y - (-1)**j * b
            r = jnp.sqrt(s*s + t*t + z*z)

            h = h.at[:,0].add((-1)**(i+j)/(4*jnp.pi) * jnp.log(r - t))
            h = h.at[:,1].add((-1)**(i+j)/(4*jnp.pi) * jnp.log(r - s))
            h = h.at[:,2].add((-1)**(i+j)/(4*jnp.pi) * jnp.arctan((s*t)/(r*z)))
    return h

def anahexa(pts, a, b, c):
    r"""Analytic stray field of a uniformly z-magnetized cuboid.

    Cuboid of edge lengths ``(a, b, c)`` centred at the origin with unit magnetization
    along z: the field is the difference of the two charged-face contributions
    (:func:`anahexa0`). Serves as an analytic reference solution.

    Parameters
    ----------
    pts : jnp.ndarray
        Evaluation points of shape (N, 3).
    a, b, c : float
        Cuboid edge lengths along x, y, z.

    Returns
    -------
    jnp.ndarray
        Stray field of shape (N, 3).

    References
    ----------
    G. Akoun and J.-P. Yonnet, "3D analytical calculation of the forces exerted between
    two cuboidal magnets", IEEE Trans. Magn. 20, 1962-1964 (1984).
    """
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    z1 = z - c/2
    z2 = z + c/2
    h1 = anahexa0(x,y,z1,a/2,b/2)
    h2 = anahexa0(x,y,z2,a/2,b/2)

    return h1 - h2
