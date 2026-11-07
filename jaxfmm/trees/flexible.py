r"""Perfectly-balanced median-split cuboid tree (the "flexible" tree).

Boxes are split at the particle median along their longest axis, so every level
is perfectly balanced: each box at level :math:`l` holds the same number of
particles (up to one unit of padding), and a box has :math:`2^{s_l}` children.
The split count per level may be a uniform ``s`` or a per-level tuple
(fractional depth, see :func:`split_schedule`).
"""
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from math import log2, ceil

__all__ = ["balanced_tree", "balanced_tree_elements", "element_aabb", "get_max_l",
           "split_schedule", "lane_schedule", "children_of", "splits_of", "reduce_max_lvl"]

def get_max_l(N_tot, N_max, s = 3):
    r"""Number of levels in the hierarchy.

    Parameters
    ----------
    N_tot : int
        Total number of particles.
    N_max : int
        Maximum number of particles per leaf box.
    s : int or tuple of int, optional
        Splits per level; a tuple's length IS the level count.

    Returns
    -------
    int
        Number of levels below the root,
        :math:`\lceil \log_2(N_{tot}/N_{max}) / s \rceil` (clamped at 0).
    """
    if isinstance(s, tuple):
        return len(s)
    max_l = int(ceil(log2(N_tot/N_max)/s))
    return 0 if max_l < 0 else max_l

def splits_of(s, max_l):
    r"""Normalise the split spec to a per-level tuple.

    Parameters
    ----------
    s : int or tuple of int
        An int becomes the uniform tuple ``(s,) * max_l``; a tuple
        ``(s_1, ..., s_max_l)`` gives level ``l`` exactly ``2**s_l`` children.
    max_l : int
        Number of levels.

    Returns
    -------
    tuple of int
        Per-level split counts.

    Notes
    -----
    The tuple form provides FRACTIONAL depth: the total binary split count is
    no longer forced to a multiple of ``s``, so the leaf occupancy can land on
    ``N_max`` instead of overshooting by up to ``2**(s-1)``.
    """
    return s if isinstance(s, tuple) else (int(s),) * max_l

def _level_entry(e):
    r"""Normalise one schedule entry to ``(m, s_l)``: an int ``s_l`` is ``(1, s_l)``."""
    return (int(e[0]), int(e[1])) if isinstance(e, tuple) else (1, int(e))

def children_of(s, max_l):
    r"""Per-level child counts of a schedule: ``m * 2**s_l`` per level.

    Parameters
    ----------
    s : int or tuple
        Split spec; a tuple entry may be an int ``s_l`` (``2**s_l`` children) or a
        pair ``(m, s_l)`` (an ``m``-way slab split followed by ``s_l`` binary
        splits, ``m * 2**s_l`` children — see :func:`lane_schedule`).
    max_l : int
        Number of levels.

    Returns
    -------
    tuple of int
        Children per box at each level.
    """
    return tuple(m * 2**k for m, k in map(_level_entry, splits_of(s, max_l)))

def split_schedule(N_tot, N_max, s = 2, coarse_first = True):
    r"""Per-level split schedule with the smallest power-of-two leaf count.

    Distributes :math:`S = \lceil \log_2(N_{tot}/N_{max}) \rceil` binary splits
    into levels of ``s`` splits plus one remainder level, so the leaf count is
    the smallest power of two holding ``N_tot / N_max``.

    Parameters
    ----------
    N_tot : int
        Total number of particles.
    N_max : int
        Maximum number of particles per leaf box.
    s : int, optional
        Splits per full level.
    coarse_first : bool, optional
        Place the remainder level coarsest (cheapest: fewest boxes there).

    Returns
    -------
    tuple of int
        Schedule usable as the ``s`` argument of :func:`balanced_tree` and the
        connectivity/driver stack; equals the uniform schedule when the total
        divides ``s``.
    """
    S = max(int(ceil(log2(N_tot / N_max))), 0)
    full, rem = divmod(S, s)
    sched = (int(s),) * full + ((rem,) if rem else ())
    if coarse_first and rem:
        sched = (rem,) + (int(s),) * full
    return sched

def lane_schedule(N_tot, N_max, s = 2, odd_max = 15):
    r"""Split schedule whose leaf count ``m * 2**S`` (odd ``m``) lands the leaf
    occupancy just under ``N_max``.

    A power-of-two leaf count fixes the occupancy at ``N_tot / 2**S``, i.e. the
    fill ``occupancy / N_max`` is set by the fractional part of ``log2(N_tot)`` and
    can be as low as 50%. Since the P2P eval rows are billed in 32/64-lane
    multiples (an eval row of 39 costs exactly a row of 64), that is up to 2x of
    the near field paid for empty slots. Admitting one ``m``-way slab split at the
    root (``m`` odd, ``<= odd_max``) makes the admissible leaf counts
    ``{m * 2**k}``. With every odd ``m <= 15`` admissible these cover each integer
    8..16 per octave, so the worst consecutive ratio is 9/8 and the fill is
    ``>= 8/9`` (88.9%). That needs ``m = 15`` admissible, i.e. ``k >= 9`` by the depth
    rule below, so ``N_tot >~ 15 * 2**9 * N_max`` (~1e6 at ``N_max = 128``); smaller
    trees admit fewer ``m`` and fall back towards the binary fill. Measured minimum
    fill at ``N_max = 128``: 66.7% for 1e4-1e5, 80.0% for 1e5-1e6, 88.9% above
    (worst case just above a power of two: ``N = 2**20 + 2`` -> ``9 * 2**k`` leaves).

    An odd root is admitted only with ``k >= 2 * ceil(log2 m) + 1`` bisections
    below it: the slabs need ~``2 log2 m`` splits of their long axes before the
    leaves are isotropic again, and a shallow slab tree has no level whose boxes
    are well separated from anything in the domain (a 729-point box gave a
    depth-1 six-slab eval tree with NO far level). Below that size the plain
    binary schedule is returned — fill is irrelevant there.

    Parameters
    ----------
    N_tot : int
        Total number of particles.
    N_max : int
        Leaf occupancy cap (the lane target: 32 or 64).
    s : int, optional
        Binary splits per full level.
    odd_max : int, optional
        Largest odd slab count admitted at the root; ``1`` reproduces
        :func:`split_schedule`.

    Returns
    -------
    tuple
        Schedule for :func:`balanced_tree`: the root entry is ``(m, s_0)`` when
        ``m > 1`` (``m * 2**s_0`` children), all other entries ints.
    """
    L_min = max(int(ceil(N_tot / N_max)), 1)
    best = None
    for m in range(1, odd_max + 1, 2):
        k = max(int(ceil(log2(L_min / m))), 0)
        L = m * 2**k
        if m > 1 and k < 2 * int(ceil(log2(m))) + 1:
            continue
        if best is None or L < best[0]:
            best = (L, m, k)
    _, m, S = best
    sched = list(split_schedule(N_max * 2**S, N_max, s, coarse_first=True))   # S binary splits
    if m == 1:
        return tuple(sched)
    if not sched:
        return ((m, 0),)
    sched[0] = (m, sched[0])
    return tuple(sched)

def _nanminmax_rows(vals_min, vals_max, idcs, max_slots = 2**26):
    r"""Per-row (per-box) NaN-ignoring min/max of ``vals[idcs]``, evaluated in sections.

    Parameters
    ----------
    vals_min, vals_max : jnp.ndarray
        Value arrays gathered per box; may be the same array (points) or
        per-element AABB corners.
    idcs : jnp.ndarray
        Per-box index rows, shape ``(n_boxes, width)``; out-of-range entries
        gather NaN fill.
    max_slots : int, optional
        Upper bound on the flat padded slot count of any single fused
        gather+reduce kernel.

    Returns
    -------
    mins, maxs : jnp.ndarray
        Per-row NaN-ignoring minima / maxima.

    Notes
    -----
    A single fused gather+nan-reduce whose flat padded slot count exceeds
    ~``2**27`` SILENTLY CORRUPTS on GPU (reads zeros past the boundary, in f32
    and f64 alike, while the materialised gather alone is exact). Sectioning
    keeps every fused kernel far below that boundary; results are bit-identical
    because min/max are exactly associative.
    """
    nb, w = idcs.shape
    rs = max(1, int(max_slots // max(w, 1)))
    cs = int(min(w, max_slots))
    mins, maxs = [], []
    for lo in range(0, nb, rs):
        sub = idcs[lo:lo + rs]
        mn = mx = None
        for cl in range(0, w, cs):
            gmin = vals_min.at[sub[:, cl:cl + cs]].get(mode="fill", fill_value=jnp.nan)
            gmax = gmin if vals_max is vals_min else \
                vals_max.at[sub[:, cl:cl + cs]].get(mode="fill", fill_value=jnp.nan)
            m1, m2 = jnp.nanmin(gmin, axis=1), jnp.nanmax(gmax, axis=1)
            mn = m1 if mn is None else jnp.fmin(mn, m1)   # fmin/fmax ignore NaN blocks
            mx = m2 if mx is None else jnp.fmax(mx, m2)
        mins.append(mn); maxs.append(mx)
    return jnp.concatenate(mins), jnp.concatenate(maxs)

@partial(jax.jit, static_argnames = ["max_l", "s", "N_max"])
def balanced_tree(pts, max_l = None, s = 3, N_max = 128):
    r"""Generate a balanced :math:`2^n`-tree hierarchy over a point cloud.

    Each level splits every box at the particle median along its longest axis
    (:math:`s_l` binary splits per level), so all leaves hold equally many
    particles.

    Parameters
    ----------
    pts : jnp.ndarray
        Particle positions, shape ``(N, 3)``.
    max_l : int, optional
        Number of levels (static). Depends only on the input size and
        ``N_max``, so it is computed internally by default via
        ``get_max_l(N, N_max, s)``; pass explicitly only to override
        (e.g. to force fewer levels).
    s : int or tuple of int, optional
        Splits per level (static): an int gives ``2**s`` children per level, a
        per-level tuple from :func:`split_schedule` gives fractional depth
        (identical results to the int path when all entries are equal).
    N_max : int, optional
        Maximum particles per leaf (static; used only when ``max_l`` is None).

    Returns
    -------
    idcs : jnp.ndarray
        Per-leaf particle indices, shape ``(n_leaves, leaf_size)``; padded
        entries are out of range (``>= N``).
    rev_idcs : jnp.ndarray
        Flat inverse permutation (undoes the leaf sorting).
    boxcenters : list of jnp.ndarray
        Per level, the box centres, shape ``(n_boxes_l, 3)``.
    boxlens : list of jnp.ndarray
        Per level, the box side lengths (tight AABBs), shape
        ``(n_boxes_l, 3)``.
    """
    if max_l is None:
        max_l = get_max_l(pts.shape[0], N_max, s)
    splits = splits_of(s, max_l)
    assert len(splits) == max_l, "per-level split tuple must have max_l entries"
    idcs = jnp.arange(pts.shape[0])[None,:]

    def _sorted_axis(idcs):
        # padded values get converted to NaNs - NOTE: due to the argpartition implementation,
        # we need to make sure that all the NaNs have the same sign
        pts_sorted = pts.at[idcs].get(mode="fill",fill_value=-jnp.nan**2)
        mn_, mx_ = _nanminmax_rows(pts, pts, idcs)   # SECTIONED extents (the fused one-shot
        axis_to_split = jnp.argmax(mx_ - mn_, axis=1)  # nan-reduce corrupts past ~2**27 slots)
        return pts_sorted[jnp.arange(axis_to_split.shape[0]),:,axis_to_split]

    def _bisect(idcs):
        splitpos = idcs.shape[1]//2   # split position in the middle
        needpad = idcs.shape[1]%2     # modulo tells us if padding must be inserted
        idcs = idcs[jnp.arange(idcs.shape[0])[:,None],jnp.argpartition(_sorted_axis(idcs),splitpos,axis=1)] # splitting, the NaNs introduced below get transported to the beginning
        # padding so the next array has the correct shape
        idcs = jax.lax.pad(idcs,pts.shape[0],[(0,0,0),(0,needpad,0)])   # pad at the end with out of range values
        return idcs.reshape((-1,idcs.shape[1]//2))                      # now that we padded, we can safely reshape this

    def _slab(idcs, m):
        # m-way slab split (lane_schedule): full sort along the longest axis, then m
        # equal-count chunks; NaN-padded slots sort LAST so the padding stays in the
        # final chunk and each chunk keeps the out-of-range-at-the-end convention
        idcs = jnp.take_along_axis(idcs, jnp.argsort(_sorted_axis(idcs), axis=1), axis=1)
        needpad = (-idcs.shape[1]) % m
        idcs = jax.lax.pad(idcs, idcs.dtype.type(pts.shape[0]), [(0,0,0),(0,needpad,0)])
        return idcs.reshape((-1, idcs.shape[1]//m))

    for m, k in map(_level_entry, splits):     # per level: optional slab split, then k bisections
        if m > 1:
            idcs = _slab(idcs, m)
        for _ in range(k):
            idcs = _bisect(idcs)

    idcs = jnp.sort(idcs,axis=1)    # sorting is good for locality
    rev_idcs = jnp.argsort(idcs.flatten())[:pts.shape[0]]     # reverse sorting indices, to undo the sorting
    # leaf extents via the SECTIONED gather+nan-reduce (see _nanminmax_rows), then coarsen
    # hierarchically: parent min/max = min/max over child min/max — bit-identical to a
    # per-level particle re-reduction (min/max associativity) and O(N) instead of
    # O(N * levels).
    minc, maxc = _nanminmax_rows(pts, pts, idcs)
    chi = children_of(splits, max_l)
    boxcenters, boxlens = [None]*(max_l+1), [None]*(max_l+1)

    for l in range(max_l,-1,-1):
        boxlens[l] = maxc - minc    # in principle we only need to save the norm of this, but it is nice to have for visualizations
        boxcenters[l] = minc + boxlens[l]/2
        if(l > 0):
            minc = jnp.nanmin(minc.reshape((-1,chi[l-1],3)),axis=1)
            maxc = jnp.nanmax(maxc.reshape((-1,chi[l-1],3)),axis=1)
    return idcs, rev_idcs, boxcenters, boxlens

def element_aabb(verts, elems):
    r"""Per-element centroid and axis-aligned bounding box of a mesh.

    Parameters
    ----------
    verts : array_like
        Vertex coordinates, shape ``(N_vert, 3)``.
    elems : array_like
        Vertex indices per element, shape ``(N_elem, k)`` (``k=4`` tets,
        ``k=3`` tris).

    Returns
    -------
    coms, emin, emax : ndarray
        Each of shape ``(N_elem, 3)``: the centroid (= vertex mean, exact for
        simplices) and the min/max corner of each element's bounding box.
    """
    ev = verts[elems]                                   # (N_elem, k, 3)
    return ev.mean(axis=1), ev.min(axis=1), ev.max(axis=1)

@partial(jax.jit, static_argnames = ["max_l", "s", "N_max"])
def balanced_tree_elements(coms, emin, emax, max_l = None, s = 3, N_max = 128):
    r"""Balanced :math:`2^s`-tree for finite elements (rather than points).

    Elements are sorted into boxes by their centre of mass ``coms`` (reusing
    :func:`balanced_tree`, so the split/locality logic is shared), but every
    box is then sized to enclose the full extent of its elements
    (box :math:`\supseteq` element) using the per-element AABB corners
    ``emin`` / ``emax``.

    Parameters
    ----------
    coms : jnp.ndarray
        Element centres of mass (the sort key), shape ``(N_elem, 3)``.
    emin, emax : jnp.ndarray
        Per-element AABB corners, shape ``(N_elem, 3)`` (see
        :func:`element_aabb`).
    max_l : int, optional
        Number of levels (static); as in :func:`balanced_tree`.
    s : int or tuple of int, optional
        Splits per level (static).
    N_max : int, optional
        Maximum elements per leaf (static).

    Returns
    -------
    idcs, rev_idcs, boxcenters, boxlens
        Same layout as :func:`balanced_tree`, with boxes enclosing the
        elements.

    Notes
    -----
    The "no element pokes out" property is what keeps the near/far split
    clean: two boxes that pass the well-separatedness test then contain no
    element pair that is actually near. Box extents are built bottom-up —
    leaf boxes from the element AABBs, coarser boxes as the union of their
    children — so the element extent enters only at the leaf and propagates
    up.
    """
    if max_l is None:
        max_l = get_max_l(coms.shape[0], N_max, s)
    chi = children_of(s, max_l)
    idcs, rev_idcs, _, _ = balanced_tree(coms, max_l, s, N_max)   # sort elements by COM
    # leaf box extents from element AABBs — SECTIONED gather+nan-reduce (see
    # _nanminmax_rows for the fused-kernel corruption boundary)
    bmin, bmax = _nanminmax_rows(emin, emax, idcs)
    boxcenters, boxlens = [None]*(max_l+1), [None]*(max_l+1)
    for l in range(max_l, -1, -1):
        boxlens[l] = bmax - bmin
        boxcenters[l] = bmin + boxlens[l]/2
        if l > 0:                                       # parent extent = union of child extents
            bmin = jnp.nanmin(bmin.reshape((-1, chi[l-1], 3)), axis=1)
            bmax = jnp.nanmax(bmax.reshape((-1, chi[l-1], 3)), axis=1)
    return idcs, rev_idcs, boxcenters, boxlens

def reduce_max_lvl(pts, idcs, s, old_max_l, new_max_l):
    r"""Reduce the maximum level of an already-built balanced tree.

    Folds sibling leaves together so the leaf level becomes ``new_max_l``,
    trimming the exactly-known accumulated padding.

    Parameters
    ----------
    pts : jnp.ndarray
        Particle positions, shape ``(N, 3)``.
    idcs : jnp.ndarray
        Leaf index array of the original tree.
    s : int or tuple of int
        Split spec of the original tree.
    old_max_l, new_max_l : int
        Original and reduced leaf levels.

    Returns
    -------
    idcs : jnp.ndarray
        Coarsened per-leaf particle indices.
    rev_idcs : jnp.ndarray
        Flat inverse permutation for the coarsened leaves.
    """
    n_merge = int(np.prod(children_of(s, old_max_l)[new_max_l:old_max_l]))   # children folded away per new leaf
    idcs = idcs.reshape((idcs.shape[0]//n_merge,-1))
    trim = idcs.shape[1] - (idcs.size - pts.shape[0])//idcs.shape[0]
    idcs = jnp.sort(idcs,axis=-1)[:,:trim] # we know exactly how much padding is accumulated!
    rev_idcs = jnp.argsort(idcs.flatten())[:pts.shape[0]] # also fix the reverse sorting
    return idcs, rev_idcs
