r"""Tetrahedral-mesh utilities: RCM vertex reordering, orientation fixing, simple mesh
generation, near-field distance diagnostics, and quantised mesh padding for stable
array shapes across remeshing."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from functools import partial

FACE_IDCS = jnp.array([
    [0, 1, 3],
    [1, 2, 3],
    [2, 0, 3],
    [0, 2, 1]   # base face
])
r"""Face node ordering jaxFMM adopts for tetrahedra (standard convention)."""

OPP_IDCS = jnp.array([2,0,1,3])
r"""Vertices opposite the four faces defined in :data:`FACE_IDCS`."""

def reorder_mesh_rcm(verts, idcs):
    r"""
    Reorder mesh vertices with Reverse Cuthill-McKee (RCM) to improve data locality.

    Builds the vertex-vertex adjacency from the element connectivity, permutes the
    vertices by the RCM ordering, and re-sorts the elements by their minimum vertex
    index.

    Parameters
    ----------
    verts : jnp.ndarray, shape (N_vert, 3)
        Vertex coordinates.
    idcs : jnp.ndarray, shape (N_elem, N_nodes)
        Vertex indices defining the mesh elements.

    Returns
    -------
    reordered_verts : jnp.ndarray, shape (N_vert, 3)
        Vertices in RCM order.
    reordered_idcs : jnp.ndarray, shape (N_elem, N_nodes)
        Element connectivity re-indexed to the new vertex order.
    """
    verts = np.asarray(verts)
    idcs = np.asarray(idcs)
    N_vert = verts.shape[0]
    N_elem, nodes_per_elem = idcs.shape
    
    ### generate element-vertex and vertex-vertex adjacency matrices
    row_indices = np.repeat(np.arange(N_elem), nodes_per_elem)
    col_indices = idcs.flatten()
    data = np.ones(len(col_indices), dtype=bool)
    C = csr_matrix((data, (row_indices, col_indices)), shape=(N_elem, N_vert))
    A = C.T @ C
    
    ### compute RCM permutation and reorder vertices
    perm = reverse_cuthill_mckee(A, symmetric_mode=True)
    reordered_verts = verts[perm]
    
    ### reindex with new vertex ordering and sort by minimum index
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(N_vert)
    reordered_idcs = inv_perm[idcs]
    minvert = np.min(reordered_idcs, axis=1)
    reordered_idcs = reordered_idcs[np.argsort(minvert)]
    
    return jnp.array(reordered_verts), jnp.array(reordered_idcs)

@jax.jit
def fix_tet_orientations(verts, tets):
    r"""
    Reorder tetrahedron indices so faces extracted with :data:`FACE_IDCS` point outward.

    Tets with negative signed volume get their first two vertices swapped.

    Parameters
    ----------
    verts : jnp.ndarray, shape (N_vert, 3)
        Vertex coordinates.
    tets : jnp.ndarray, shape (N_elem, 4)
        Vertex indices defining the tetrahedra.

    Returns
    -------
    jnp.ndarray, shape (N_elem, 4)
        Reordered tetrahedron indices with outward-facing normals when indexed
        with :data:`FACE_IDCS`.
    """
    is_negative = get_tet_volumes(verts, tets) < 0
    swapped_tets = jnp.stack([tets[:, 1], tets[:, 0], tets[:, 2], tets[:, 3]], axis=1)
    fixed_tets = jnp.where(is_negative[:, None], swapped_tets, tets)
    return fixed_tets

@jax.jit
def get_tet_volumes(verts, tets):
    r"""
    Signed tetrahedron volumes (times 6).

    Parameters
    ----------
    verts : jnp.ndarray, shape (N_vert, 3)
        Vertex coordinates.
    tets : jnp.ndarray, shape (N_elem, 4)
        Vertex indices defining the tetrahedra.

    Returns
    -------
    jnp.ndarray, shape (N_elem,)
        Signed scalar triple products (positive for :data:`FACE_IDCS`-outward
        orientation).
    """
    cross_prod = jnp.cross(verts[tets[:,2]]-verts[tets[:,0]],verts[tets[:,3]]-verts[tets[:,0]],axis=-1)
    volumes = jnp.sum((verts[tets[:,1]] - verts[tets[:,0]]) * cross_prod, axis=1)
    return volumes

@partial(jax.jit, static_argnames=['mem_limit'])
def check_min_dist(padded_pts, padded_eval_pts, dir_cnct, mem_limit = jnp.inf):
    r"""
    Closest source-eval distance over the near-field pair list.

    Helps estimate the worst-case near-field quadrature error. Batches the pair list
    when the pairwise distance tensor would exceed ``mem_limit``.

    Parameters
    ----------
    padded_pts : jnp.ndarray, shape (n_box, cap, 3)
        Padded per-box source points.
    padded_eval_pts : jnp.ndarray, shape (n_box_e, cap_e, 3)
        Padded per-box evaluation points.
    dir_cnct : jnp.ndarray, shape (n_pairs, 2)
        Near-field (eval box, source box) pair list.
    mem_limit : float, optional
        Byte budget for one batch of the pairwise distance tensor.

    Returns
    -------
    jnp.ndarray, scalar
        Minimum distance (``inf`` if there are no near-field pairs).
    """
    if(dir_cnct.size == 0):   # no near-field evaluation
        return jnp.inf
    # 0.3.1 budget semantics: price the materialised (batch, w_src, w_eval, 3) distance
    # tensor (~2x live multiplicity) so mem_limit caps the actual transient.
    mem = dir_cnct.shape[0] * padded_pts.shape[1] * padded_eval_pts.shape[1] * 6 * padded_pts.itemsize
    batch_size = int((mem_limit/mem)*dir_cnct.shape[0]) + 1 if mem_limit<mem else dir_cnct.shape[0]
    mindist = jnp.array(jnp.inf)

    def scan_body(buf, idcs):
        distsvec = padded_eval_pts.at[idcs[:,0],None,:,:].get(mode="fill",fill_value=jnp.inf) - padded_pts.at[idcs[:,1],:,None,:].get(mode="fill",fill_value=0)
        distsmin = jnp.min(jnp.linalg.norm(distsvec,axis=-1))
        return jnp.where(buf < distsmin, buf, distsmin), None

    if(batch_size < dir_cnct.shape[0]):
        num_pad = batch_size - dir_cnct.shape[0]%batch_size
        fill_value = dir_cnct.dtype.type(jnp.iinfo(dir_cnct.dtype).max)   # typed: x64 + int32 lists
        scandir = jax.lax.pad(dir_cnct, fill_value, ((0,num_pad,0), (0,0,0))).reshape((-1,batch_size,2))
        mindist, _ = jax.lax.scan(scan_body, mindist, scandir)
    else:
        mindist, _ = scan_body(mindist, dir_cnct)

    return mindist

def gen_box_mesh(nx, ny, nz, Lx=1.0, Ly=1.0, Lz=1.0):
    r"""
    Generate a structured tetrahedral mesh of the box :math:`[0,L_x]\times[0,L_y]\times[0,L_z]`.

    Each of the ``nx * ny * nz`` grid cells is split into six tetrahedra;
    orientations are fixed for consistent outward normals
    (:func:`fix_tet_orientations`).

    Parameters
    ----------
    nx, ny, nz : int
        Number of cells along each axis.
    Lx, Ly, Lz : float, optional
        Box edge lengths (default: unit box).

    Returns
    -------
    verts : jnp.ndarray, shape ((nx+1)*(ny+1)*(nz+1), 3)
        Vertex coordinates.
    tets : jnp.ndarray, shape (6*nx*ny*nz, 4)
        Tetrahedron vertex indices.
    """
    gx = jnp.linspace(0.0, Lx, nx + 1)
    gy = jnp.linspace(0.0, Ly, ny + 1)
    gz = jnp.linspace(0.0, Lz, nz + 1)
    X, Y, Z = jnp.meshgrid(gx, gy, gz, indexing="ij")
    verts = jnp.stack([X, Y, Z], -1).reshape(-1, 3)
    cube_elem = jnp.array([[[0,0,0], [1,0,0], [1,1,0], [1,1,1]],
                           [[0,0,0], [1,0,0], [1,0,1], [1,1,1]],
                           [[0,0,0], [0,1,0], [1,1,0], [1,1,1]],
                           [[0,0,0], [0,1,0], [0,1,1], [1,1,1]],
                           [[0,0,0], [0,0,1], [1,0,1], [1,1,1]],
                           [[0,0,0], [0,0,1], [0,1,1], [1,1,1]]])
    cells = jnp.stack(jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), jnp.arange(nz),
                                   indexing="ij"), -1).reshape(-1, 1, 1, 3)
    tets = (cells + cube_elem[None]).reshape(-1, 4, 3)
    conv = jnp.array([(ny + 1) * (nz + 1), nz + 1, 1])
    tets = tets @ conv
    return verts, fix_tet_orientations(verts, tets)


def gen_unitcube_mesh(N):
    r"""
    Generate a unit-cube tetrahedral mesh with ``N+1`` vertices along each edge.

    Each grid cell is split into six tets; orientations are fixed for consistent
    outward normals.

    Parameters
    ----------
    N : int
        Number of cells along each edge.

    Returns
    -------
    verts : jnp.ndarray, shape ((N+1)**3, 3)
        Vertex coordinates in :math:`[0, 1]^3`.
    tets : jnp.ndarray, shape (6*N**3, 4)
        Tetrahedron vertex indices.
    """
    verts = jnp.mgrid[:N+1,:N+1,:N+1].T.reshape((-1,3))*1.0/N
    cube_elem = jnp.array([[[0,0,0], [1,0,0], [1,1,0], [1,1,1]],
                        [[0,0,0], [1,0,0], [1,0,1], [1,1,1]],
                        [[0,0,0], [0,1,0], [1,1,0], [1,1,1]],
                        [[0,0,0], [0,1,0], [0,1,1], [1,1,1]],
                        [[0,0,0], [0,0,1], [1,0,1], [1,1,1]],
                        [[0,0,0], [0,0,1], [0,1,1], [1,1,1]]])
    tets = (jnp.mgrid[:N,:N,:N].T[...,None,None,:] + cube_elem[None,None,None,...]).reshape((-1,4,3))
    conv = jnp.array([1,N+1,(N+1)**2])
    tets = tets@conv

    tets = fix_tet_orientations(verts, tets)    # reorder for consistent normals
    return verts, tets

def bucket_cap(n, base=256, growth=1.25, prev=None):
    r"""
    Quantised capacity: the smallest ``base * growth**k >= n`` (used by :func:`pad_mesh`).

    Parameters
    ----------
    n : int
        Required capacity.
    base : int, optional
        Smallest bucket.
    growth : float, optional
        Geometric bucket growth factor.
    prev : int or None, optional
        Previous cap, enabling downsize hysteresis (see Notes).

    Returns
    -------
    int
        The quantised capacity.

    Notes
    -----
    ``prev`` adds DOWNSIZE HYSTERESIS: a still-sufficient previous cap is kept unless it
    is more than one bucket step above what is needed, and then it shrinks by ONE step
    per call — transient mesh shrinks don't force a recompile, sustained shrinks converge
    to the tight cap gradually. Growth (``prev`` too small) is immediate.
    """
    k = 0 if n <= base else int(np.ceil(np.log(n / base) / np.log(growth)))
    cap = base if k == 0 else int(np.ceil(base * growth ** k))
    if prev is None or prev < cap:
        return cap                                   # first call / grow: immediate
    one_up = base if k == 0 else int(np.ceil(base * growth ** (k + 1)))
    if prev <= one_up:
        return prev                                  # within one step: keep (hysteresis)
    # far oversized: shrink ONE step per call (floor+eps: prev is itself a CEILed bucket
    # value, so ceil(log(prev)) overshoots its own index and would reproduce prev)
    kp = int(np.floor(np.log(prev / base) / np.log(growth) + 1e-9))
    return max(cap, int(np.ceil(base * growth ** (kp - 1))))


def pad_mesh(verts, tets, tris, growth=1.25, base=256, prev_caps=None):
    r"""
    Pad a tet mesh (+ surface tris) to quantised capacities for stable array shapes.

    The shape-churn fix for remesh/shape-opt loops: every new vert/tet/tri COUNT otherwise
    forces a full recompile of the jitted forward, while a repeated bucket hits the
    in-process/persistent compilation cache.

    Parameters
    ----------
    verts : array_like, shape (N_vert, 3)
        Vertex coordinates.
    tets : array_like, shape (N_tet, 4)
        Tetrahedron vertex indices.
    tris : array_like, shape (N_tri, 3)
        Surface triangle vertex indices.
    growth : float, optional
        Geometric bucket growth factor (see Notes).
    base : int, optional
        Smallest bucket.
    prev_caps : tuple of int or None, optional
        The previous call's ``caps``, enabling downsize hysteresis (see
        :func:`bucket_cap`): shrinking meshes step down one bucket at a time instead
        of recompiling on every transient shrink.

    Returns
    -------
    verts_p : jnp.ndarray, shape (cap_v, 3)
        Padded vertices.
    tets_p : jnp.ndarray, shape (cap_t, 4)
        Padded tetrahedra.
    tris_p : jnp.ndarray, shape (cap_s, 3)
        Padded triangles.
    n_verts_real : int
        Number of real (unpadded) vertices.
    caps : tuple of int
        The realised ``(cap_v, cap_t, cap_s)``, to pass back as ``prev_caps``.

    Notes
    -----
    Padding = 4 dedicated template vertices forming a TINY real tet at the domain corner,
    with padded tets/tris replicating its connectivity: finite geometry (no NaNs in the
    Duffy factors), padded charge far below FMM truncation, and re-derived surfaces drop
    the padded faces automatically (they repeat, so never count==1). Downstream, mask
    outputs / losses to the REAL slots: ``u[:n_verts_real]``.

    Pick ``growth`` ABOVE the per-remesh size growth for in-run cache hits (padding
    inflates the eval cost by at most that factor, far cheaper than recompiles). Even
    below that, bucketing coarsens the compilation-cache key space, so RERUNS of an
    optimisation with drifting remesh trajectories still hit the cache. Bucketed-vs-raw
    output differs at truncation level (the padded tree splits differently; both are
    within FMM accuracy).
    """
    v, t, s = np.asarray(verts), np.asarray(tets), np.asarray(tris)
    nv, nt, ns = len(v), len(t), len(s)
    pv, pt, ps = prev_caps if prev_caps is not None else (None, None, None)
    corner = v.min(0)
    tmpl_v = corner[None] + 1e-3 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    cv = bucket_cap(nv + 4, base=base, growth=growth, prev=pv)
    vpad = np.concatenate([v, tmpl_v, np.repeat(tmpl_v[:1], cv - nv - 4, axis=0)])
    ti = nv + np.array([0, 1, 2, 3])
    ct = bucket_cap(nt, base=base, growth=growth, prev=pt)
    tpad = np.concatenate([t, np.repeat(ti[None], ct - nt, axis=0)]) if ct > nt else t
    si = nv + np.array([0, 1, 2])
    cs = bucket_cap(ns, base=base, growth=growth, prev=ps)
    spad = np.concatenate([s, np.repeat(si[None], cs - ns, axis=0)]) if cs > ns else s
    return (jnp.asarray(vpad), jnp.asarray(tpad.astype(t.dtype)),
            jnp.asarray(spad.astype(s.dtype)), nv, (cv, ct, cs))
