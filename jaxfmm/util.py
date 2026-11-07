r"""Assorted utilities shared across the library.

Hardware queries, the persistent compilation cache, generic setup splitting for
setup-as-arguments jit, entry-door input validation, gradient-safe coordinate
transforms, and debug/inspection/VTK output helpers.
"""
import jax
import ctypes
from ctypes import util
import jax.numpy as jnp
import jax.scipy as jsp
from time import perf_counter
import numpy as np
import os
from functools import partial

__all__ = ["parse_domain", "split_static", "join_static", "enable_compilation_cache", "get_gpu_l2_cache_size", "gen_stats", "gen_hierarchy_vtk", "gen_wellsep_vtk", "gen_pts_vtk", "time_function", "gen_multipole_dist", "get_locs", "eval_mpls", "plot_sparsity", "make_pyvista_mesh", "write_mesh_pv", "binom"]


def parse_domain(domain, L0_boxcen=None, L0_boxlen=None, dtype=None):
    r"""Resolve the unit-cell specification shared by every engine.

    ``domain=(pmin, extent)`` — the corner-plus-extent convention of
    :func:`jaxfmm.kifmm.setup` / :func:`jaxfmm.uniform.setup`; ``extent`` may be a
    scalar (cubic cell) or a length-3 vector.  The flexible engine's historical
    ``L0_boxcen``/``L0_boxlen`` (centre + side lengths) remain accepted; passing both
    forms is ambiguous and raises.

    Parameters
    ----------
    domain : (pmin, extent) or None
        Cell corner (length-3) and extent (scalar or length-3).
    L0_boxcen, L0_boxlen : array_like, optional
        Centre/side-length form (mutually exclusive with ``domain``).
    dtype : jnp dtype, optional
        Result dtype (default: infer / float).

    Returns
    -------
    (L0_boxcen, L0_boxlen) : (jnp.ndarray, jnp.ndarray) or (None, None)
        Centre and side lengths of the root box, or ``(None, None)`` when neither
        form was given.
    """
    import jax.numpy as jnp
    if domain is not None:
        if L0_boxcen is not None or L0_boxlen is not None:
            raise ValueError("pass either domain=(pmin, extent) or L0_boxcen/L0_boxlen, not both")
        pmin, extent = domain
        pmin = jnp.asarray(pmin, dtype=dtype)
        lens = jnp.broadcast_to(jnp.asarray(extent, dtype=dtype), (3,))
        return pmin + lens / 2, lens
    if L0_boxcen is None and L0_boxlen is None:
        return None, None
    cen = jnp.asarray(L0_boxcen, dtype=dtype) if L0_boxcen is not None else None
    lens = jnp.asarray(L0_boxlen, dtype=dtype) if L0_boxlen is not None else None
    return cen, lens

def get_gpu_l2_cache_size(device_id=0):
    r"""
    L2 cache size in bytes of a CUDA device (via the driver API).

    Parameters
    ----------
    device_id : int
        CUDA device index.

    Returns
    -------
    int
        L2 cache size in bytes.
    """
    libname = util.find_library('cuda')
    if libname is None:
        raise OSError("get_gpu_l2_cache_size: CUDA driver library not found")
    cuda = ctypes.CDLL(libname)
    device = ctypes.c_int()
    attribute_value = ctypes.c_int()
    L2_CACHE_ATTR = 38
    # every driver call returns a CUresult; a silent failure would leave attribute_value at 0
    # and size every L2-anchored batch to ~1 element
    for call, args in ((cuda.cuInit, (0,)), (cuda.cuDeviceGet, (ctypes.byref(device), device_id)),
                       (cuda.cuDeviceGetAttribute, (ctypes.byref(attribute_value), L2_CACHE_ATTR, device))):
        rc = call(*args)
        if rc != 0:
            raise OSError(f"get_gpu_l2_cache_size: CUDA driver call {call.__name__} failed (CUresult {rc})")
    if attribute_value.value <= 0:
        raise OSError("get_gpu_l2_cache_size: driver reported a non-positive L2 size")
    return attribute_value.value
def enable_compilation_cache(path=None):
    r"""Enable the persistent XLA compilation cache with the project's canonical settings.

    Call this from EVERY entry point (conftest, scripts, notebooks) instead of setting
    the jax options directly: the options MUST be consistent across all writers of one
    cache directory.

    Parameters
    ----------
    path : str, optional
        Cache directory; defaults to ``.jax_cache`` next to the package.

    Notes
    -----
    Configuration traps this function exists to centralise:

    * ``max_size`` (LRU) requires the ``filelock`` package or every READ fails
      (warning only);
    * entries written WITHOUT ``max_size`` lack the ``-atime`` LRU sidecars and poison
      the bookkeeping for every later ``max_size`` writer (all WRITES fail with
      FileNotFoundError on a legacy entry) — mixed writer configs are how this recurs;
      wipe the dir once after any config change;
    * the default ``min_compile_time_secs`` (1s) silently skips the many small
      executables that dominate fresh-process setup, so it is set to 0 here.

    The cache directory is disposable — wipe it whenever in doubt."""
    import os
    import jax as _jax
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            ".jax_cache")
    _jax.config.update("jax_compilation_cache_dir", path)
    _jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    _jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    _jax.config.update("jax_compilation_cache_max_size", 10 * 1024 ** 3)   # 10 GB LRU cap


# ──────────────────────────────────────────────────────────────────────────
# Generic setup splitting for setup-as-arguments jit
# ──────────────────────────────────────────────────────────────────────────
_TOK_ARR, _TOK_DICT, _TOK_TUP, _TOK_LIST = "\x00A", "\x00D", "\x00T", "\x00L"


def split_static(obj):
    r"""Split a setup structure into a hashable skeleton and the flat tuple of arrays.

    Handles nested dicts/tuples/lists of arrays and static scalars.  The pair is meant
    for use as (static_argnames-keyed meta, runtime pytree) of a MODULE-LEVEL jitted
    function -- so re-built setups with repeating array shapes re-use the same
    executable with zero retrace/compile.  Rejoin with :func:`join_static`.

    Parameters
    ----------
    obj : dict, tuple, list, array or hashable
        Setup structure: arbitrarily nested dicts/tuples/lists whose leaves are
        arrays (jax or numpy) or hashable statics.

    Returns
    -------
    skel : hashable
        Statics + array placeholders.
    arrs : tuple of arrays
        The arrays in traversal order.
    """
    import numpy as _np
    import jax as _jax
    arrs = []

    def rec(o):
        if isinstance(o, (_jax.Array, _np.ndarray)):
            arrs.append(o)
            return _TOK_ARR
        if isinstance(o, dict):
            return (_TOK_DICT,) + tuple((k, rec(o[k])) for k in sorted(o))
        if isinstance(o, tuple) and hasattr(o, "_fields") and not any(
                isinstance(v, (_jax.Array, _np.ndarray)) for v in o):
            return o                     # array-free NamedTuple (e.g. a Kernel spec): keep
                                         # ATOMIC so type + accessors survive the round trip
        if isinstance(o, tuple):
            return (_TOK_TUP,) + tuple(rec(v) for v in o)
        if isinstance(o, list):
            return (_TOK_LIST,) + tuple(rec(v) for v in o)
        return o                                        # hashable static leaf

    skel = rec(obj)
    return skel, tuple(arrs)


def join_static(skel, arrs):
    r"""Inverse of :func:`split_static`.

    Parameters
    ----------
    skel : hashable
        Skeleton from :func:`split_static` (statics + array placeholders).
    arrs : tuple of arrays
        The arrays in traversal order.

    Returns
    -------
    dict, tuple, list, array or hashable
        The reassembled setup structure.
    """
    it = iter(arrs)

    def rec(sk):
        if sk == _TOK_ARR:
            return next(it)
        if isinstance(sk, tuple) and sk and sk[0] == _TOK_DICT:
            return {k: rec(v) for k, v in sk[1:]}
        if isinstance(sk, tuple) and sk and sk[0] == _TOK_TUP:
            return tuple(rec(v) for v in sk[1:])
        if isinstance(sk, tuple) and sk and sk[0] == _TOK_LIST:
            return [rec(v) for v in sk[1:]]
        return sk

    return rec(skel)


def validate_points(pts, name="positions", n_cols=3, integer=False):
    r"""Friendly entry-door validation.

    Raise early with a readable message instead of a deep tracer error: ``pts`` must be
    an (N, ``n_cols``) array, floating for coordinates / integer for connectivity, with
    N > 0.

    Parameters
    ----------
    pts : array-like
        Candidate array.
    name : str
        Name used in error messages.
    n_cols : int
        Required column count.
    integer : bool
        Require an integer dtype (connectivity) instead of floating (coordinates).
    """
    shape = getattr(pts, "shape", None)
    if shape is None or len(shape) != 2 or shape[1] != n_cols:
        hint = " (a transposed array? pass pts.T)" if (shape and len(shape) == 2
                                                       and shape[0] == n_cols) else ""
        raise ValueError(f"{name}: expected an (N, {n_cols}) array, got shape {shape}{hint}")
    if shape[0] == 0:
        raise ValueError(f"{name}: got 0 rows")
    import numpy as _np
    kind = _np.dtype(getattr(pts, "dtype", float)).kind
    if integer and kind not in "iu":
        raise ValueError(f"{name}: expected an integer index array, got dtype {pts.dtype}")
    if not integer and kind != "f":
        raise ValueError(f"{name}: expected a floating-point array, got dtype {pts.dtype} "
                         f"(cast with jnp.asarray(x, dtype=float))")


# -- generic coordinate helpers (formerly jaxfmm.laplace.rotation) --
@jax.jit
def cart_to_sph(rvec):
    r"""
    Convert cartesian to spherical coordinates, GRADIENT-SAFE on the polar axis.

    Parameters
    ----------
    rvec : (..., 3) array
        Cartesian displacement vectors.

    Returns
    -------
    r, theta, phi : arrays
        Radius, polar angle, azimuth.

    Notes
    -----
    :math:`\sqrt{x^2+y^2}` and :math:`\operatorname{atan2}(y, x)` have NaN derivatives
    at :math:`x = y = 0`, and axis-aligned displacements are COMMON on structured
    geometries (regular grids give exact box-centre alignment), so a naive
    implementation NaNs every backward pass through the rotation M2L/M2M/L2L there.
    The guarded form returns ``phi = 0`` with ZERO phi-sensitivity at the axis --
    which is the CORRECT limit: the :math:`m \neq 0` azimuthal components vanish as
    :math:`s^{|m|}` on the axis, so their gauge derivative contributes nothing (same
    masked-where pattern as the safe :math:`1/r` in ``eval_direct``).

    The guard extends to ``rvec = 0`` EXACTLY (``r``/``theta`` frozen at 0 with zero
    sensitivity): duplicated coordinates on structured meshes let a median split
    produce a child whose tight AABB coincides with its parent's, giving an M2M
    displacement of exactly zero (observed on surface-only quad clouds — a flat-slab
    boundary NaN'd every shape gradient through one such pair). Zero sensitivity is
    again the CORRECT limit, not a clamp: coincident AABBs are pinned by the same
    extremal points, so the displacement stays identically zero under any vertex
    perturbation and the true derivative through it vanishes.
    """
    xy = rvec[...,0]**2 + rvec[...,1]**2
    on_axis = xy == 0.0
    xy_safe = jnp.where(on_axis, 1.0, xy)
    s_xy = jnp.where(on_axis, 0.0, jnp.sqrt(xy_safe))
    z2 = rvec[...,2]**2
    at_origin = (xy + z2) == 0.0
    r = jnp.where(at_origin, 0.0, jnp.sqrt(jnp.where(at_origin, 1.0, xy + z2)))
    theta = jnp.arctan2(s_xy, jnp.where(at_origin, 1.0, rvec[...,2]))
    phi = jnp.arctan2(jnp.where(on_axis, 0.0, rvec[...,1]),
                      jnp.where(on_axis, 1.0, rvec[...,0]))
    return r, theta, phi


# -- debug / inspection helpers --


def gen_stats(pts, eval_pts, idcs, boxcenters, eval_boxcenters, mpl_cnct, dir_cnct, img_cnct, lvl_info, s, periodic_axes, p, theta, print_stats = False, **kwargs):
    r"""Generate hierarchy stats.

    Use the hierarchy information from ``gen_hierarchy`` as input (splat the dict).

    Parameters
    ----------
    pts : (N, 3) jnp.ndarray
        Source points.
    eval_pts : (M, 3) jnp.ndarray
        Evaluation points.
    idcs : list of tuple
        The hierarchy's four ``(sort_idcs, rev_idcs)`` pairs (source/eval at the
        multipole and direct levels).
    boxcenters : (n_box_src, 3) jnp.ndarray
        Source box centres (all levels, concatenated).
    eval_boxcenters : (n_box_trg, 3) jnp.ndarray
        Evaluation box centres (may be ``boxcenters`` for symmetric trees).
    mpl_cnct : (n_mpl, 2) jnp.ndarray
        Well-separated (M2L) box pairs.
    dir_cnct : (n_dir, 2) jnp.ndarray
        Near-field (direct) box pairs.
    img_cnct : jnp.ndarray
        PBC image shift vectors.
    lvl_info : list of tuple
        Per-traversal-step ``(trg_lvl, src_lvl)`` pairs.
    s : int
        Split schedule (2**s children per box).
    periodic_axes : tuple of int
        Axes with periodic boundaries (empty for open BC).
    p : int
        Multipole order.
    theta : float
        Multipole acceptance criterion.
    print_stats : bool, optional
        Also pretty-print the stats.
    **kwargs
        Ignored (lets the whole ``gen_hierarchy`` dict be splatted in).

    Returns
    -------
    stats : dict
        Interaction counts, compression ratio and memory estimates; optionally printed.
    """
    stats = {"p": p,
             "theta": theta,
             "max_l_src": lvl_info[-1][1],
             "max_l_trg": lvl_info[-1][0],
             "num_pts": pts.shape[0],
             "num_eval_pts": eval_pts.shape[0],
             "num_child": 2**s,
             "pts_per_box": idcs[2][0].shape[1],
             "eval_pts_per_box": idcs[3][0].shape[1]}
    periodic = len(periodic_axes)>0
    mem_hierarch = idcs[0][0].nbytes + idcs[0][1].nbytes
    if(idcs[1] is not idcs[0]):
        mem_hierarch += idcs[1][0].nbytes + idcs[1][1].nbytes
    if(idcs[2] is not idcs[0]):
        mem_hierarch += idcs[2][0].nbytes + idcs[2][1].nbytes
    if((idcs[3] is not idcs[1]) and (idcs[3] is not idcs[2])):
        mem_hierarch += idcs[3][0].nbytes + idcs[3][1].nbytes
    mem_hierarch = sum([elem.nbytes for elem in (boxcenters, mpl_cnct, dir_cnct, img_cnct)])
    if(boxcenters is not eval_boxcenters):
        mem_hierarch += eval_boxcenters.nbytes
    
    stats["M2L_num"] = mpl_cnct.size//2
    stats["lvl_info"] = lvl_info
    stats["direct_num"] = dir_cnct.size//2
    stats["compression_ratio"] = dir_cnct.shape[0] * idcs[3][0].shape[1] * idcs[2][0].shape[1] / (pts.shape[0]*eval_pts.shape[0])
    stats["mem_hierarch"] = mem_hierarch
    stats["mem_coeffs"] = 4*2*((p+1)**2)*((2**s)**(stats["max_l_src"]+1)-1)/7 # 4 bytes, 2 arrays, (p+1)**2 coeffs and nboxes
    stats["mem_pts_chrgs"] = (pts.nbytes*4)/3
    if(eval_pts is not pts):
        stats["mem_pts_chrgs"] += eval_pts.nbytes
    if(print_stats):
        print("---------------------FMM Hierarchy Stats---------------------")
        print("p = %i, theta = %.2f, %i children per box"%(p, theta, stats["num_child"]))
        print("%i sources, %i targets"%(stats["num_pts"],stats["num_eval_pts"]))
        print("%i sources per leaf, %i targets per leaf"%(stats["pts_per_box"], stats["eval_pts_per_box"]))
        print("max src lvl: %i, max trg lvl: %i"%(stats["max_l_src"], stats["max_l_trg"]))
        if(periodic): 
            print("Periodic boundary on axes: ", periodic_axes)
        print("")
        print("Total interactions: %11i"%(stats["M2L_num"]+stats["direct_num"]))
        print("      of which M2L: %11i"%(stats["M2L_num"]))
        print("      of which P2P: %11i"%(stats["direct_num"]))
        print("Compression (vs dir.): %.2e"%(stats["compression_ratio"]))
        print("")
        print("Memory used by the hierarchy:            %.2e Bytes"%stats["mem_hierarch"])
        print("Memory used by mpl + local coeffs (p=%i): %.2e Bytes"%(p, stats["mem_coeffs"])) 
        print("Memory used by the points + charges:     %.2e Bytes"%stats["mem_pts_chrgs"])
        print("-------------------------------------------------------------")
    return stats

def make_pyvista_mesh(boxcenters, boxlens):
    r"""
    Generate a pyvista hexahedron mesh of boxes with centers ``boxcenters`` and
    diagonals ``boxlens``.

    Parameters
    ----------
    boxcenters : (n_box, 3) array_like
        Box centres.
    boxlens : (n_box, 3) array_like
        Box edge lengths (full extents per axis).

    Returns
    -------
    pyvista.UnstructuredGrid
        Hexahedron mesh with one cell per box.
    """
    import pyvista as pv  # heavy optional dep: only needed for plotting
    pv_shifts = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
    pts = np.zeros((boxcenters.shape[0],8,3))
    for i in range(pv_shifts.shape[0]):
        pts[:,i,:] = boxcenters + boxlens/2 * pv_shifts[i,None,:]
    pts = pts.reshape((-1,3))
    cells = np.arange(pts.shape[0],dtype=np.int32).reshape((-1,8))
    return pv.UnstructuredGrid({pv.CellType.HEXAHEDRON: cells}, pts)

def gen_hierarchy_vtk(boxcenters, boxlens, src_ofs, eval_boxcenters=None, eval_boxlens=None, trg_ofs=None, dir="hierarchy", **kwargs):
    r"""
    Output a series of vtk files showing the FMM hierarchy on every level.

    Does not show virtual PBC images.

    Parameters
    ----------
    boxcenters : (n_box_src, 3) jnp.ndarray
        Source box centres (all levels, concatenated).
    boxlens : (n_box_src, 3) jnp.ndarray
        Source box edge lengths.
    src_ofs : sequence of int
        Per-level offsets into the source box arrays.
    eval_boxcenters : (n_box_trg, 3) jnp.ndarray or None, optional
        Evaluation box centres (default: same as the sources).
    eval_boxlens : (n_box_trg, 3) jnp.ndarray or None, optional
        Evaluation box edge lengths.
    trg_ofs : sequence of int or None, optional
        Per-level offsets into the evaluation box arrays.
    dir : str, optional
        Output directory (created if missing).
    **kwargs
        Ignored (lets the whole hierarchy dict be splatted in).
    """
    if(eval_boxcenters is None or eval_boxlens is None or trg_ofs is None):
        eval_boxcenters = boxcenters
        eval_boxlens = boxlens
        trg_ofs = src_ofs

    eval_max_l = len(trg_ofs)-1
    max_l = len(src_ofs)-1
    if not os.path.exists(dir):
        os.makedirs(dir)
    for l in range(max_l):
        mesh = make_pyvista_mesh(boxcenters[src_ofs[l]:src_ofs[l+1]],boxlens[src_ofs[l]:src_ofs[l+1]])
        mesh.cell_data["src_level"] = np.ones(mesh.points.shape[0]//8)*l
        mesh.save("%s/src_level_%i.vtk"%(dir,l))
    
    if(eval_boxcenters is not boxcenters or eval_boxlens is not boxlens):
        for l in range(eval_max_l):
            mesh = make_pyvista_mesh(eval_boxcenters[trg_ofs[l]:trg_ofs[l+1]],eval_boxlens[trg_ofs[l]:trg_ofs[l+1]])
            mesh.cell_data["eval_level"] = np.ones(mesh.points.shape[0]//8)*l
            mesh.save("%s/eval_level_%i.vtk"%(dir,l))

def gen_pts_vtk(pts, eval_pts, dir="hierarchy", **kwargs):
    r"""
    Output the hierarchy points as vtk.

    Parameters
    ----------
    pts : (N, 3) jnp.ndarray
        Source points.
    eval_pts : (M, 3) jnp.ndarray
        Evaluation points.
    dir : str, optional
        Output directory (created if missing).
    **kwargs
        Ignored (lets the whole hierarchy dict be splatted in).
    """
    import pyvista as pv  # heavy optional dep: only needed for plotting
    if not os.path.exists(dir):
        os.makedirs(dir)
    point_cloud = pv.PolyData(np.array(pts))
    point_cloud.save("%s/src_pts.vtk"%(dir))
    point_cloud = pv.PolyData(np.array(eval_pts))
    point_cloud.save("%s/eval_pts.vtk"%(dir))

def gen_wellsep_vtk(id, boxcenters, boxlens, eval_boxcenters, eval_boxlens, mpl_cnct, dir_cnct, src_ofs, trg_ofs, img_cnct, s, pbc_ws, dir="hierarchy", **kwargs):
    r"""
    Output a vtk file showing all the boxes considered for potential calculation,
    given an ``id`` of a box on the highest level.

    Includes virtual PBC images.

    Parameters
    ----------
    id : int
        Index of the evaluation box on the finest target level.
    boxcenters : (n_box_src, 3) jnp.ndarray
        Source box centres (all levels, concatenated).
    boxlens : (n_box_src, 3) jnp.ndarray
        Source box edge lengths.
    eval_boxcenters : (n_box_trg, 3) jnp.ndarray
        Evaluation box centres.
    eval_boxlens : (n_box_trg, 3) jnp.ndarray
        Evaluation box edge lengths.
    mpl_cnct : (n_mpl, 2) jnp.ndarray
        Well-separated (M2L) box pairs.
    dir_cnct : (n_dir, 2) jnp.ndarray
        Near-field (direct) box pairs (finest level, offset-free).
    src_ofs : sequence of int
        Per-level offsets into the source box arrays.
    trg_ofs : sequence of int
        Per-level offsets into the evaluation box arrays.
    img_cnct : jnp.ndarray
        PBC image shift vectors.
    s : int
        Split schedule (2**s children per box).
    pbc_ws : list
        Per-super-level PBC well-separated image data (empty for open BC).
    dir : str, optional
        Output directory (created if missing).
    **kwargs
        Ignored (lets the whole hierarchy dict be splatted in).
    """
    import pyvista as pv  # heavy optional dep: only needed for plotting
    if not os.path.exists(dir):
        os.makedirs(dir)
    n_chi = 2**s
    n_l_src = len(src_ofs) - 1
    n_l_trg = len(trg_ofs) - 1
    nbox_src = boxcenters.shape[0]
    periodic = len(pbc_ws)>0
    meshlist = []

    ### periodic far-field images
    L = boxlens[0]
    for l, img_cnct_ws in enumerate(pbc_ws):
        mesh = make_pyvista_mesh(boxcenters[0] - img_cnct_ws[1],jnp.tile(L,(img_cnct_ws[1].shape[0],1)))
        mesh.cell_data["eval_level"] = jnp.ones(mesh.points.shape[0]//8)*(-l)
        mesh.cell_data["src_level"] = jnp.ones(mesh.points.shape[0]//8)*(-l)
        meshlist.append(mesh)
        L += jnp.ptp(img_cnct_ws[0],axis=0)

    ### M2L
    if(mpl_cnct.size > 0):
        for i in range(n_l_trg):
            loc_id = id//(n_chi**(n_l_src-i-1)) + src_ofs[i]
            extract = mpl_cnct[(mpl_cnct[:,0] == loc_id),1]
            if(extract.size>0):
                if(periodic):
                    mesh = make_pyvista_mesh(boxcenters[extract%nbox_src] + img_cnct[extract//nbox_src], boxlens[extract%nbox_src])
                else:
                    mesh = make_pyvista_mesh(boxcenters[extract], boxlens[extract])
                mesh.cell_data["eval_level"] = jnp.ones(mesh.points.shape[0]//8) * i
                src_level = (extract[:,None]%nbox_src>=jnp.array(src_ofs)[None,:]).sum(axis=1) - 1
                mesh.cell_data["src_level"] = jnp.ones(mesh.points.shape[0]//8) * src_level
                meshlist.append(mesh)

    ### direct
    nbox_src_dir = src_ofs[-1] - src_ofs[-2]
    if(dir_cnct.size > 0):
        extract = dir_cnct[(dir_cnct[:,0] == id),1] # NOTE: the dir_cnct has no offset currently, so we add it manually
        boxcenters_loc = boxcenters[src_ofs[-2]:src_ofs[-1]]
        boxlens_loc = boxlens[src_ofs[-2]:src_ofs[-1]]
        if(extract.size>0):
            if(periodic):
                mesh = make_pyvista_mesh(boxcenters_loc[extract%nbox_src_dir] + img_cnct[extract//nbox_src_dir], boxlens_loc[extract%nbox_src_dir])
            else:
                mesh = make_pyvista_mesh(boxcenters_loc[extract], boxlens_loc[extract])
            mesh.cell_data["eval_level"] = jnp.ones(mesh.points.shape[0]//8) * n_l_trg   # NOTE: we do not subtract 1 to mark the direct interactions
            mesh.cell_data["src_level"] = jnp.ones(mesh.points.shape[0]//8) * n_l_src
            meshlist.append(mesh)

    ### the id box itself
    mesh = make_pyvista_mesh(eval_boxcenters[id+trg_ofs[-2]][None,:],eval_boxlens[id+trg_ofs[-2]][None,:]*1.001)   # the multiplication fixes pyvista getting confused for eval_pts = pts
    mesh.cell_data["eval_level"] = jnp.ones(1) * (n_l_trg + 1)
    mesh.cell_data["src_level"] = jnp.ones(1) * (n_l_src + 1)
    meshlist.append(mesh)

    mergedmesh = pv.merge(meshlist)
    mergedmesh.save("%s/wellsep_%i.vtk"%(dir,id))

@partial(jax.jit, static_argnames=['p'])
def eval_mpls(mpls, eval_pts, boxcenters, p):
    r"""
    Evaluate multipole expansions ``mpls`` at ``eval_pts`` around ``boxcenters``.

    Parameters
    ----------
    mpls : (n_box, (p+1)**2) jnp.ndarray
        Multipole coefficients per expansion.
    eval_pts : (n_eval, 3) jnp.ndarray
        Evaluation points (broadcast against ``boxcenters``; must lie outside the
        expansion sphere).
    boxcenters : (3,) jnp.ndarray
        Expansion centre.
    p : int
        Multipole order (static).

    Returns
    -------
    (n_box, n_eval) jnp.ndarray
        Potential values (with the :math:`1/4\pi` factor applied).
    """
    from jaxfmm.kernels.laplace import eval_singular_basis  # local: avoids util<->laplace cycle
    sing = eval_singular_basis(eval_pts - boxcenters[None,:],p)
    res = (mpls[:, None, :] * sing[None]).sum(axis=-1)                 # (n_box, n_eval)
    return res / (4*jnp.pi)

@partial(jax.jit, static_argnames=['p'])
def get_locs(padded_pts, padded_chrgs, boxcenters, p):
    r"""
    Obtain local expansions of the padded point charges directly (P2L).

    Parameters
    ----------
    padded_pts : (n_box, n_pad, 3) jnp.ndarray
        Source points per box (padded).
    padded_chrgs : (n_box, n_pad) jnp.ndarray
        Charges per box (zero on padding).
    boxcenters : (n_box, 3) jnp.ndarray
        Expansion centres (must be well away from the sources).
    p : int
        Expansion order (static).

    Returns
    -------
    (n_box, (p+1)**2) jnp.ndarray
        Local expansion coefficients per box.
    """
    from jaxfmm.kernels.laplace import eval_singular_basis  # local: avoids util<->laplace cycle
    dist = padded_pts - boxcenters[:,None]
    sing = eval_singular_basis(dist,p)
    return (sing * padded_chrgs[...,None]).sum(axis=1)

def binom(x, y):
  r"""Binomial coefficient :math:`\binom{x}{y}` via ``gammaln`` (real-valued, differentiable).

  Parameters
  ----------
  x : float or jnp.ndarray
      Upper argument (need not be integer).
  y : float or jnp.ndarray
      Lower argument (need not be integer; broadcastable against ``x``).

  Returns
  -------
  jnp.ndarray
      The binomial coefficients, shape of the broadcast inputs.
  """
  return jnp.exp(jsp.special.gammaln(x + 1) - jsp.special.gammaln(y + 1) - jsp.special.gammaln(x - y + 1))

def gen_multipole_dist(m, n, eps = 0.5):
    r"""
    Generate a point charge distribution corresponding to a specific multipole moment.

    Parameters
    ----------
    m : int
        Multipole order, ``-n <= m <= n`` (negative m selects the rotated/sine
        real component).
    n : int
        Multipole degree.
    eps : float, optional
        Point spacing of the discrete distribution (default 0.5); larger values
        reduce the finite-difference error of the represented moment.

    Returns
    -------
    pts : (K, 3) jnp.ndarray
        Point positions.
    chrgs : (K,) jnp.ndarray
        Point charges.

    References
    ----------
    Majic, Matt (2022). Point charge representations of multipoles.
    European Journal of Physics 43. doi:10.1088/1361-6404/ac578b
    """
    if(m == 0):   # axial
        k = jnp.arange(-n, n+1, 2)
        chrgs = (-1)**((n-k)/2) * binom(n, (n-k)/2.0) / (jsp.special.factorial(n) * (2*eps)**n)
        pts = jnp.zeros((k.shape[0],3))
        pts = pts.at[:,2].set(k*eps)
    else:         # (stacked) bracelet
        rotate = m < 0
        m = abs(m)      # we work with the real basis and rotate later
        knum = n-m+1
        jnum = 2*m
        j = jnp.tile(jnp.arange(jnum),knum)
        k = jnp.repeat(jnp.arange(-n+m,n-m+1,2),jnum)
        phi = (j-0.5) * jnp.pi/m if rotate else j * jnp.pi/m
        pts = jnp.array([eps*jnp.cos(phi), eps*jnp.sin(phi), k*eps]).T
        chrgs = 4**(m-1) * jsp.special.factorial(m-1) / ((2*eps)**n * jsp.special.factorial(n-m)) * (-1)**((n-m-k)/2 + j) * binom(n-m,(n-m-k)/2)
    return pts, chrgs

def time_function(func, nruns = 10, print_times = True, **kwargs):
    r"""
    Time ``func``, obtaining the best of ``nruns`` runs and optionally printing the
    results.  Function args are supplied as additional kwargs.

    Parameters
    ----------
    func : callable
        Function to time (its result is ``block_until_ready``'d).
    nruns : int, optional
        Number of runs (default 10); the first includes compilation.
    print_times : bool, optional
        Print the compile and best runtimes.
    **kwargs
        Forwarded to ``func``.

    Returns
    -------
    (res, comp, best)
        Function result, first-run (compile) time, and best runtime in seconds.
    """
    best = jnp.inf
    for i in range(nruns):
        T0 = perf_counter()
        res = jax.block_until_ready(func(**kwargs))
        T1 = perf_counter()
        best = min(T1-T0,best)
        if(i==0):
            comp = best
    if(print_times):
        print("%s compilation: %.2e s"%(func.__name__,comp))
        print("%s runtime: %.2e s"%(func.__name__,best))
    return res, comp, best

def plot_sparsity(mpl_cnct, dir_cnct, trg_ofs, src_ofs, show_plot = True, **kwargs):
    r"""
    Plot the sparsity pattern of the interactions in a given hierarchy.

    Gives an indication of how well the FMM compresses the number of interactions
    compared to a direct evaluation.

    Parameters
    ----------
    mpl_cnct : (n_mpl, 2) jnp.ndarray
        Well-separated (M2L) box pairs.
    dir_cnct : (n_dir, 2) jnp.ndarray
        Near-field (direct) box pairs (finest level, offset-free).
    trg_ofs : sequence of int
        Per-level offsets into the evaluation box arrays.
    src_ofs : sequence of int
        Per-level offsets into the source box arrays.
    show_plot : bool, optional
        Call ``plt.show()`` before returning.
    **kwargs
        Ignored (lets the whole hierarchy dict be splatted in).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axs : np.ndarray of matplotlib.axes.Axes
        The two axes (near / far field).
    """
    import matplotlib.pyplot as plt  # heavy optional dep: only needed for plotting
    s = 0.5
    fig, axs = plt.subplots(1,2)
    if(mpl_cnct.size < 1):
        mpl_cnct = np.array([[[np.nan,np.nan]]])
    if(dir_cnct.size < 1):
        dir_cnct = np.array([[[np.nan,np.nan]]])
    data = [(dir_cnct[...,0]+trg_ofs[-2],dir_cnct[...,1]+src_ofs[-2]), (mpl_cnct[...,0],mpl_cnct[...,1])]

    titles = ["Near Field Connectivity", "Far Field Connectivity"]
    fig.set_size_inches(14,6)
    for arr, title, ax in zip(data, titles, axs):
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.set_title(title, y=-0.1)
        ax.vlines(jnp.array(trg_ofs)[1:-1]-0.5,0,trg_ofs[-1],'lightgray')
        ax.hlines(jnp.array(src_ofs)[1:-1]-0.5,0,src_ofs[-1],'lightgray')
        ax.scatter(arr[0],arr[1],s=s,zorder=500)
        ax.set_xlabel("Target ID")
        ax.set_ylabel("Source ID")
        ax.set_xlim(trg_ofs[0]-1,trg_ofs[-1])
        ax.set_ylim(src_ofs[0]-1,src_ofs[-1])
        ax.yaxis.set_inverted(True)

    if(show_plot):
        plt.show()
    return fig, axs

def write_mesh_pv(verts, tets, point_data = {}, cell_data = {}, fname = "mesh.vtk"):
    r"""
    Write a tet-mesh file with pyvista, optionally including cell-/point-data.

    Parameters
    ----------
    verts : (N, 3) jnp.ndarray
        Vertex positions.
    tets : (T, 4) jnp.ndarray
        Tetrahedron vertex indices (int).
    point_data : dict, optional
        Per-vertex arrays to attach, keyed by name.
    cell_data : dict, optional
        Per-tet arrays to attach, keyed by name.
    fname : str, optional
        Output file name (default ``"mesh.vtk"``).

    Returns
    -------
    pyvista.UnstructuredGrid
        The saved mesh.
    """
    import pyvista as pv  # heavy optional dep: only needed for plotting
    points = np.array(verts)
    cells = np.array(jnp.insert(tets,0,jnp.full(tets.shape[0],fill_value=4),axis=1))
    celltypes = np.full(cells.shape[0],fill_value=pv.CellType.TETRA, dtype=jnp.uint8)
    mesh = pv.UnstructuredGrid(cells,celltypes,points)
    for key, item in point_data.items():
        mesh.point_data[key] = item
    for key, item in cell_data.items():
        mesh.cell_data[key] = item
    mesh.save(fname)
    return mesh
