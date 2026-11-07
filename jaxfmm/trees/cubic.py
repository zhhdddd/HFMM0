r"""2:1-balanced cubic (uniform-grid) octree builder.

Particles are assigned Morton codes at ``max_depth`` — code
:math:`m = \mathrm{interleave}(i_x, i_y, i_z)` on the :math:`2^d` grid — and the
tree is coarsened bottom-up while enforcing the 2:1 balance constraint (adjacent
leaves differ by at most one level).

Primary entry point: :func:`build_fmm_tree`; boxes are packed fine-to-coarse
(finest boxes first) in fixed-size ``max_boxes`` buffers, with inactive padding
entries identified by ``lvls == 0``. Deep trees require
``jax.config.update("jax_enable_x64", True)`` (uint64 Morton codes); 32-bit mode
supports ``max_depth <= 9``.
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

__all__ = [
    "build_fmm_tree",
    "build_fmm_tree_auto",
    "assign_particles",
    "morton_to_coords",
    "get_26_neighbors",
    "box_geometry", "assign_particles_lean", "cubic_img_table", "get_26_neighbors_periodic"]

# ─── Balance-constraint lookup table (built once at import time) ──────────────

def _build_neighbor_tables():
    r"""Build the balance-constraint lookup tables (computed once at import time).

    Returns
    -------
    _ADJ_CHILD_MAP : jnp.ndarray
        ``(26, 4)`` int32 — child indices of each neighbour that touch the
        centre box.
    _ADJ_CHILD_MASK : jnp.ndarray
        ``(26, 8)`` bool — boolean version of ``_ADJ_CHILD_MAP``.
    _NEIGH_OFFSETS : jnp.ndarray
        ``(26, 3)`` int32 — ``(dx, dy, dz)`` for each of the 26 neighbours.

    Notes
    -----
    All tables use the same row ordering: ``dx`` outer, ``dy`` middle, ``dz``
    inner, centre excluded.
    """
    dx_all, dy_all, dz_all = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')
    dx = np.delete(dx_all.flatten(), 13)
    dy = np.delete(dy_all.flatten(), 13)
    dz = np.delete(dz_all.flatten(), 13)

    adj_table = np.full((26, 4), -1, dtype=np.int32)
    adj_mask  = np.zeros((26, 8), dtype=bool)
    for i in range(26):
        valid = []
        for c in range(8):
            cx, cy, cz = c & 1, (c >> 1) & 1, (c >> 2) & 1
            if ((dx[i] == -1 and cx == 1) or (dx[i] == 1 and cx == 0) or dx[i] == 0) and \
               ((dy[i] == -1 and cy == 1) or (dy[i] == 1 and cy == 0) or dy[i] == 0) and \
               ((dz[i] == -1 and cz == 1) or (dz[i] == 1 and cz == 0) or dz[i] == 0):
                valid.append(c)
        adj_table[i, :len(valid)] = valid
        adj_mask[i, valid] = True

    offsets = np.stack([dx, dy, dz], axis=1).astype(np.int32)
    return jnp.array(adj_table), jnp.array(adj_mask), jnp.array(offsets)

_ADJ_CHILD_MAP, _ADJ_CHILD_MASK, _NEIGH_OFFSETS = _build_neighbor_tables()

# ─── Morton code utilities ─────────────────────────────────────────────────────

@jax.jit
def _expand_bits(v):
    if not jax.config.jax_enable_x64:                        # single precision: uint32 Morton, <=10 bits/axis
        v = v.astype(jnp.uint32) & jnp.uint32(0x3FF)         # (valid for max_depth<=9 -> 3*9=27-bit Morton fits uint32)
        v = (v | (v << 16)) & jnp.uint32(0xFF0000FF)
        v = (v | (v <<  8)) & jnp.uint32(0x0300F00F)
        v = (v | (v <<  4)) & jnp.uint32(0x030C30C3)
        v = (v | (v <<  2)) & jnp.uint32(0x09249249)
        return v
    v = v.astype(jnp.uint64)
    v = (v | (v << 32)) & jnp.uint64(0x1F00000000FFFF)
    v = (v | (v << 16)) & jnp.uint64(0x1F0000FF0000FF)
    v = (v | (v <<  8)) & jnp.uint64(0x100F00F00F00F00F)
    v = (v | (v <<  4)) & jnp.uint64(0x10C30C30C30C30C3)
    v = (v | (v <<  2)) & jnp.uint64(0x1249249249249249)
    return v

@jax.jit
def _compact_bits(v):
    if not jax.config.jax_enable_x64:                        # inverse of the uint32 Morton expand
        v = v.astype(jnp.uint32) & jnp.uint32(0x09249249)
        v = (v | (v >>  2)) & jnp.uint32(0x030C30C3)
        v = (v | (v >>  4)) & jnp.uint32(0x0300F00F)
        v = (v | (v >>  8)) & jnp.uint32(0xFF0000FF)
        v = (v | (v >> 16)) & jnp.uint32(0x3FF)
        return v
    v = v & jnp.uint64(0x1249249249249249)
    v = (v | (v >>  2)) & jnp.uint64(0x10C30C30C30C30C3)
    v = (v | (v >>  4)) & jnp.uint64(0x100F00F00F00F00F)
    v = (v | (v >>  8)) & jnp.uint64(0x1F0000FF0000FF)
    v = (v | (v >> 16)) & jnp.uint64(0x1F00000000FFFF)
    v = (v | (v >> 32)) & jnp.uint64(0x1FFFFF)
    return v

@jax.jit
def _morton_3d(x, y, z):
    return _expand_bits(x) | (_expand_bits(y) << 1) | (_expand_bits(z) << 2)

@jax.jit
def _morton_to_grid(morton_code):
    return _compact_bits(morton_code), _compact_bits(morton_code >> 1), _compact_bits(morton_code >> 2)

@jax.jit
def morton_to_coords(morton_codes, lvls, domain_extent, pmin):
    r"""Physical centre coordinates for an array of boxes.

    Parameters
    ----------
    morton_codes : jnp.ndarray
        Morton codes at each box's own level, shape ``(n,)``.
    lvls : jnp.ndarray
        Box levels, shape ``(n,)``.
    domain_extent : float
        Edge length of the cubic root domain.
    pmin : jnp.ndarray
        Lower corner of the root domain, shape ``(3,)``.

    Returns
    -------
    jnp.ndarray
        Box centre coordinates, shape ``(n, 3)``.
    """
    coords = jnp.stack(_morton_to_grid(morton_codes), axis=-1).astype(jnp.float64)
    coords = coords * domain_extent / (2 ** lvls[:, None])
    coords = coords + domain_extent * 0.5 ** (lvls[:, None] + 1) + pmin
    return coords


def box_geometry(m_id, lvls, domain_extent, pmin):
    r"""Physical box centres and half-sides for the cubic octree.

    Shared by the KIFMM and uniform spherical-harmonic drivers.

    Parameters
    ----------
    m_id, lvls, domain_extent, pmin
        As in :func:`morton_to_coords`.

    Returns
    -------
    centres : jnp.ndarray
        Box centre coordinates, shape ``(max_boxes, 3)``.
    h : jnp.ndarray
        Box half-sides ``domain_extent * 0.5**(l+1)``, shape ``(max_boxes,)``.
    """
    centres = morton_to_coords(m_id, lvls, domain_extent, pmin)
    h = domain_extent * 0.5 ** (lvls.astype(jnp.float64) + 1.0)
    return centres, h

# ─── 26-neighbor lookup ────────────────────────────────────────────────────────

@jax.jit
def get_26_neighbors(morton_codes, level):
    r"""Morton codes of all 26 same-level face/edge/corner neighbours.

    Parameters
    ----------
    morton_codes : jnp.ndarray
        Box Morton codes, shape ``(N,)``.
    level : jnp.ndarray
        Box levels, shape ``(N,)``.

    Returns
    -------
    neighbor_codes : jnp.ndarray
        Neighbour Morton codes, shape ``(N, 26)``, uint64.
    valid_mask : jnp.ndarray
        Shape ``(N, 26)``, bool; ``False`` for out-of-domain neighbours.
    """
    grid_size = (2 ** level)[:, None]          # (N, 1)
    x, y, z = _morton_to_grid(morton_codes)    # (N,) each

    dx = _NEIGH_OFFSETS[:, 0]   # (26,) — precomputed constants
    dy = _NEIGH_OFFSETS[:, 1]
    dz = _NEIGH_OFFSETS[:, 2]

    nx = x[:, None].astype(jnp.int32) + dx[None, :]   # (N, 26)
    ny = y[:, None].astype(jnp.int32) + dy[None, :]
    nz = z[:, None].astype(jnp.int32) + dz[None, :]

    valid_mask = (nx >= 0) & (nx < grid_size) & \
                 (ny >= 0) & (ny < grid_size) & \
                 (nz >= 0) & (nz < grid_size)

    nx_safe = jnp.clip(nx, 0, grid_size - 1).astype(jnp.uint64)
    ny_safe = jnp.clip(ny, 0, grid_size - 1).astype(jnp.uint64)
    nz_safe = jnp.clip(nz, 0, grid_size - 1).astype(jnp.uint64)

    return _morton_3d(nx_safe, ny_safe, nz_safe), valid_mask


# image-combo table aligned with cubic_img_table's row order (row 0 = zero image)
import numpy as _np
_IMG_COMBOS = _np.stack(_np.meshgrid(*([_np.arange(-1, 2)] * 3), indexing="ij"),
                        -1).reshape(-1, 3)
_IMG_COMBOS[[0, 13]] = _IMG_COMBOS[[13, 0]]

def get_26_neighbors_periodic(morton_codes, level, periodic_axes=()):
    r"""Periodic (toroidal) variant of :func:`get_26_neighbors`.

    Along ``periodic_axes`` the grid wraps (``mod 2**level``) instead of
    rejecting out-of-domain neighbours, and each neighbour carries an image
    index identifying the wrap direction.

    Parameters
    ----------
    morton_codes : jnp.ndarray
        Box Morton codes, shape ``(N,)``.
    level : jnp.ndarray
        Box levels, shape ``(N,)``.
    periodic_axes : tuple of int, optional
        Axes (0, 1 and/or 2) that wrap.

    Returns
    -------
    neighbor_codes : jnp.ndarray
        Shape ``(N, 26)``, uint64.
    valid_mask : jnp.ndarray
        Shape ``(N, 26)``, bool; ``False`` only for out-of-domain along
        NON-periodic axes.
    img_idx : jnp.ndarray
        Shape ``(N, 26)``, int32 — flattened image index into
        :func:`cubic_img_table`; 0 = no wrap (the zero offset, matching the
        flexible engine's "image 0 = no shift" convention consumed by
        ``eval_direct``).
    """
    grid_size = (2 ** level)[:, None].astype(jnp.int32)       # (N, 1)
    x, y, z = _morton_to_grid(morton_codes)

    out_codes, out_valid, out_img = [], [], []
    for ax, (c, dax) in enumerate(((x, _NEIGH_OFFSETS[:, 0]),
                                   (y, _NEIGH_OFFSETS[:, 1]),
                                   (z, _NEIGH_OFFSETS[:, 2]))):
        n = c[:, None].astype(jnp.int32) + dax[None, :]        # (N, 26)
        if ax in periodic_axes:
            img = jnp.floor_divide(n, grid_size)               # -1/0/+1 wrap direction
            n = jnp.mod(n, grid_size)
            v = jnp.ones_like(n, dtype=bool)
        else:
            img = jnp.zeros_like(n)
            v = (n >= 0) & (n < grid_size)
            n = jnp.clip(n, 0, grid_size - 1)
        out_codes.append(n.astype(jnp.uint64))
        out_valid.append(v)
        out_img.append(img)
    codes = _morton_3d(*out_codes)
    valid = out_valid[0] & out_valid[1] & out_valid[2]
    # flatten (ix,iy,iz) in {-1,0,1}^3 to [0,27) with the ZERO image remapped to index 0
    flat = (out_img[0] + 1) * 9 + (out_img[1] + 1) * 3 + (out_img[2] + 1)
    img_idx = jnp.where(flat == 13, 0, jnp.where(flat == 0, 13, flat)).astype(jnp.int32)
    return codes, valid, img_idx


def cubic_img_table(domain_extent, periodic_axes=()):
    r"""Image offset table for the cubic engines' wrapped pairs.

    Parameters
    ----------
    domain_extent : float
        Edge length of the root box (the periodic unit cell).
    periodic_axes : tuple of int, optional
        Unused for the table values; kept for signature symmetry.

    Returns
    -------
    jnp.ndarray
        Shape ``(27, 3)``: row ``img_idx`` (from
        :func:`get_26_neighbors_periodic`) is the physical shift of the SOURCE
        box's periodic image (in the convention of ``near.eval_direct``'s
        ``img_cnct``). Row 0 is the zero offset; rows of non-periodic axes are
        present but unused.
    """
    import numpy as np
    combos = np.stack(np.meshgrid(*([np.arange(-1, 2)] * 3), indexing="ij"), -1).reshape(-1, 3)
    tbl = combos.astype(np.float64) * float(domain_extent)
    tbl[[0, 13]] = tbl[[13, 0]]                               # zero image -> row 0
    return jnp.asarray(tbl)


# ─── Internal helper ───────────────────────────────────────────────────────────

@jax.jit
def _compact(arr, scatter_indices, placeholder=0):
    r"""Scatter ``arr`` into a placeholder-filled copy, dropping out-of-range indices."""
    out = jnp.full_like(arr, placeholder)
    return out.at[scatter_indices].set(arr, mode="drop")

# ─── Main tree builder ─────────────────────────────────────────────────────────

def _check_32bit_depth(max_depth):
    r"""Refuse tree depths that overflow uint32 Morton codes in 32-bit mode.

    32-bit mode (``jax_enable_x64=False``) uses uint32 Morton codes: 3 bits per
    level plus up to 4 level bits in the interaction lists'
    ``(lvl << 3*max_depth) | morton`` box ids — depth 10+ SILENTLY corrupts the
    tree, so refuse it loudly.
    """
    if not jax.config.jax_enable_x64 and max_depth > 9:
        raise ValueError(
            f"32-bit mode (jax_enable_x64=False) supports max_depth <= 9, got {max_depth}"
            " (uint32 Morton codes; enable x64 for deeper trees or raise N_max)")


@partial(jax.jit, static_argnames=['max_boxes', 'max_depth', 'N_max', 'periodic_axes'])
def build_fmm_tree(positions, max_boxes, max_depth=15, N_max=128, domain=None,
                   periodic_axes=()):
    r"""Build a 2:1-balanced cubic octree in fixed-size buffers.

    Assigns Morton codes at ``max_depth``, then coarsens bottom-up: a parent
    absorbs its children when its count is at most ``N_max`` AND the 2:1
    balance constraint holds (every adjacent child of every same-level
    neighbour is a leaf).

    Parameters
    ----------
    positions : jnp.ndarray
        Particle positions, shape ``(N, 3)``; NaN rows are treated as inactive
        particles.
    max_boxes : int
        Static — size of every output array (buffer capacity).
    max_depth : int, optional
        Static — maximum octree depth; coarsening stops at level 1.
    N_max : int, optional
        Static — maximum particles per leaf.
    domain : tuple, optional
        ``(pmin, extent)`` pins the root box (periodic unit cell) instead of
        the tight bounding box.
    periodic_axes : tuple of int, optional
        Static — axes along which the 2:1 balance wraps periodically.

    Returns
    -------
    m_id : jnp.ndarray
        ``(max_boxes,)`` uint64 — Morton code of each box at its own level.
    p_id : jnp.ndarray
        ``(max_boxes,)`` int32 — index of the parent box; level-1 boxes point
        to themselves.
    lvls : jnp.ndarray
        ``(max_boxes,)`` int32 — level of each box; 0 marks inactive padding.
    counts : jnp.ndarray
        ``(max_boxes,)`` int32 — number of source particles in the box.
    is_leaf : jnp.ndarray
        ``(max_boxes,)`` bool — True for leaf boxes.
    sort_idx : jnp.ndarray
        ``(N,)`` — permutation sorting particles by Morton code.
    rev_sort_idx : jnp.ndarray
        ``(N,)`` — inverse of ``sort_idx``.
    domain_extent : jnp.ndarray
        Scalar — edge length of the (cubic) bounding domain.
    pmin : jnp.ndarray
        ``(3,)`` — lower corner of the bounding domain.
    max_depth_exceeded : jnp.ndarray
        Scalar bool — True if any leaf at ``max_depth`` has count > ``N_max``.
    max_boxes_exceeded : jnp.ndarray
        Scalar bool — True if the buffer was too small.

    Notes
    -----
    Boxes are packed fine-to-coarse: finest boxes (highest level) come first.
    """
    _check_32bit_depth(max_depth)                     # max_depth static: trace-time check
    MAX_UINT64 = ~jnp.array(0, dtype=jnp.uint64)
    all_boxids = jnp.arange(max_boxes, dtype=jnp.uint64)

    # ── Phase 1: Morton assignment at max_depth ───────────────────────────────

    active = ~jnp.isnan(positions[:, 0])
    pmax   = jnp.nanmax(positions, axis=0)
    pmin   = jnp.nanmin(positions, axis=0)
    positions = jnp.where(active[:, None], positions, pmin)

    grid_size     = 2 ** max_depth
    domain_extent = jnp.max(pmax - pmin)
    if domain is not None:                            # pinned root box (periodic cell)
        pmin = jnp.asarray(domain[0], positions.dtype)
        domain_extent = jnp.asarray(domain[1], positions.dtype)

    scaled      = (positions - pmin) / domain_extent * grid_size
    grid_coords = jnp.clip(scaled.astype(jnp.uint64), 0, grid_size - 1)
    part_mortons = _morton_3d(grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2])

    # XLA bug with jnp.argsort on uint64 (x64 mode) is fixed in JAX >= 0.6.2.
    N = positions.shape[0]
    iota = jnp.arange(N, dtype=jnp.int32)
    sort_idx = jnp.argsort(part_mortons).astype(jnp.int32)
    sorted_mortons = part_mortons[sort_idx]
    # Inverse permutation via scatter — faster to compile and execute than a second sort.
    rev_sort_idx = jnp.zeros_like(iota).at[sort_idx].set(iota, unique_indices=True)
    sorted_active   = active[sort_idx]

    # Find unique boxes at max_depth via run-length encoding of sorted Mortons
    code_changes = jnp.concatenate([jnp.array([False]),
                                    sorted_mortons[1:] != sorted_mortons[:-1]])
    node_ids = jnp.cumsum(code_changes)   # box-ID for each (sorted) particle

    counts = jax.ops.segment_sum(sorted_active.astype(jnp.int32),
                                 node_ids, num_segments=max_boxes)
    m_id   = jax.ops.segment_max(sorted_mortons, node_ids, num_segments=max_boxes)
    lvls   = (counts > 0).astype(jnp.int32) * max_depth

    # ── Phase 2: Bottom-up coarsening with 2:1 balance ───────────────────────

    p_id    = jnp.zeros(max_boxes, dtype=jnp.int32)
    is_leaf = counts > 0

    # Track the highest box index ever used (for overflow detection)
    max_box_count = node_ids[-1].astype(jnp.int32)

    def _coarsen_step(i, val, check_balance):
        r"""One coarsening step: merge level ``max_depth - i`` into ``max_depth - i - 1``.

        ``check_balance=False`` skips the 2:1 check (safe at ``i=0`` because
        every finest-level box is a leaf, so the balance condition is
        trivially met).
        """
        m_id, p_id, lvls, counts, is_leaf, max_box_count = val

        on_prev_lvl    = (lvls == max_depth - i)
        ofs            = (lvls >= max_depth - i).sum().astype(jnp.int32)
        prev_mortons   = jnp.where(on_prev_lvl, m_id, MAX_UINT64)
        prev_counts    = jnp.where(on_prev_lvl, counts, 0)

        parent_mortons = prev_mortons // jnp.uint64(8)
        code_changes   = jnp.concatenate([jnp.array([False]),
                                          parent_mortons[1:] > parent_mortons[:-1]])
        parent_mortons = jnp.where(on_prev_lvl, parent_mortons, 0)

        this_node_ids  = (jnp.cumsum(code_changes) + ofs).astype(jnp.int32)
        max_box_count  = jnp.maximum(max_box_count, this_node_ids[-1])

        this_counts  = jax.ops.segment_sum(prev_counts, this_node_ids, num_segments=max_boxes)
        this_mortons = jax.ops.segment_max(parent_mortons, this_node_ids, num_segments=max_boxes)

        m_id   = m_id   + this_mortons
        lvls   = lvls   + (this_counts > 0).astype(jnp.int32) * jnp.int32(max_depth - i - 1)
        counts = counts + this_counts
        p_id   = jnp.where(on_prev_lvl, this_node_ids, p_id)

        count_ok          = this_counts <= N_max
        delete_children   = count_ok[p_id] & on_prev_lvl
        mark_parent_leaf  = count_ok & (this_counts > 0)

        if check_balance:
            # Balance constraint: a parent may only absorb its children if every
            # same-level neighbor's adjacent children are all leaves. Under PBC the
            # 2:1 balance must hold ACROSS the periodic boundary too — the wrapped
            # U/W/X partition proofs rely on it (open-BC balancing lets leaf levels
            # jump >1 across the wrap, silently corrupting the near/far split).
            if len(periodic_axes) > 0:
                neighs, valid_mask, _ = get_26_neighbors_periodic(m_id, lvls, periodic_axes)
            else:
                neighs, valid_mask = get_26_neighbors(m_id, lvls)

            this_mortons_search = jnp.where(counts > 0, this_mortons, MAX_UINT64)
            insids = jnp.searchsorted(this_mortons_search, neighs,
                                      side='right', method='scan_unrolled') - 1
            valid_mask = valid_mask & (neighs == this_mortons_search[insids])

            child_ids = jnp.zeros((max_boxes, 8), dtype=jnp.uint64)
            child_ids = child_ids.at[p_id, m_id % jnp.uint64(8)].set(
                all_boxids, unique_indices=True, indices_are_sorted=True)

            # _ADJ_CHILD_MASK[d, c] = True iff child c of neighbor in direction d touches the box.
            children_are_leaves = ~valid_mask[:, :, None] | is_leaf[child_ids[insids]]
            balance_ok = jnp.all(
                children_are_leaves | ~_ADJ_CHILD_MASK[None, :, :],
                axis=(1, 2)
            )

            delete_children  = delete_children  & balance_ok[p_id]
            mark_parent_leaf = mark_parent_leaf & balance_ok

        keep            = ~delete_children
        target_indices  = (jnp.cumsum(keep) - 1).astype(jnp.int32)
        scatter_indices = jnp.where(keep, target_indices, jnp.int32(max_boxes))

        m_id   = _compact(m_id,   scatter_indices)
        lvls   = _compact(lvls,   scatter_indices)
        counts = _compact(counts, scatter_indices)

        # Remap EVERY box's parent pointer through this step's renumbering, not
        # just the level being merged: deeply-refined boxes (adaptive trees)
        # persist across many coarsening steps and their parent index changes at
        # each compaction.  Parents are never deleted (only absorbed children are),
        # so target_indices[p_id] is valid for all boxes.
        p_id = target_indices[p_id.astype(jnp.int32)]
        p_id = _compact(p_id, scatter_indices)

        is_leaf = is_leaf | mark_parent_leaf
        is_leaf = _compact(is_leaf, scatter_indices)

        return m_id, p_id, lvls, counts, is_leaf, max_box_count

    def body_fun(i, val):
        return _coarsen_step(i, val, check_balance=True)

    state = (m_id, p_id, lvls, counts, is_leaf, max_box_count)
    # i=0: all finest-level boxes are leaves → balance constraint trivially satisfied
    state = _coarsen_step(0, state, check_balance=False)
    # i=1..max_depth-2: full balance check required
    m_id, p_id, lvls, counts, is_leaf, max_box_count = jax.lax.fori_loop(
        1, max_depth - 1, body_fun, state)

    # Level-1 boxes (the coarsest) have no parent in the tree; point to themselves
    lvl1 = lvls == 1
    p_id = jnp.where(lvl1, jnp.arange(max_boxes, dtype=jnp.int32), p_id)

    # ── Overflow flags ────────────────────────────────────────────────────────
    max_depth_exceeded = jnp.any(is_leaf & (lvls == max_depth) & (counts > N_max))
    max_boxes_exceeded = max_box_count >= max_boxes

    return (m_id, p_id, lvls, counts, is_leaf,
            sort_idx, rev_sort_idx,
            domain_extent, pmin,
            max_depth_exceeded, max_boxes_exceeded)


def build_fmm_tree_auto(positions, N_max=64, max_depth=None, refine_factor=2,
                        depth_cap=10, max_boxes=None, domain=None, periodic_axes=()):
    r""":func:`build_fmm_tree` with a compact box buffer and an auto depth (host-side, eager).

    Parameters
    ----------
    positions : array_like
        Particle positions, shape ``(N, 3)``.
    N_max : int, optional
        Maximum particles per leaf.
    max_depth : int, optional
        Explicit depth; disables the auto refinement.
    refine_factor : float, optional
        Refine while the realised max leaf count exceeds
        ``refine_factor * N_max``.
    depth_cap : int, optional
        Upper bound on the auto depth.
    max_boxes : int, optional
        Initial buffer size; doubles on overflow.
    domain : tuple, optional
        Pinned root box, as in :func:`build_fmm_tree`.
    periodic_axes : tuple of int, optional
        As in :func:`build_fmm_tree`.

    Returns
    -------
    out : tuple
        The full :func:`build_fmm_tree` tuple.
    max_boxes : int
        The buffer size that sufficed.
    max_depth : int
        The realised depth.
    maxleaf : int
        The realised maximum leaf occupancy.

    Notes
    -----
    * The buffer starts compact (``~4N/N_max``, or ``max_boxes``) and DOUBLES
      on the overflow flag instead of the dense ``8**(d+1)/7`` bound — boxes
      are packed, so ``max_boxes`` is only an upper bound, and the dense
      sizing host-OOMs at large depths.
    * When ``max_depth`` is None it starts from the volume-filling heuristic
      ``ceil(log8(N / N_max))`` and REFINES while the realised max leaf count
      exceeds ``refine_factor * N_max`` (capped at ``depth_cap``): the
      heuristic under-refines clustered / quasi-2D clouds (thin films),
      leaving the finest leaves overfull with P2P cost ~ ``maxleaf**2``.
    """
    import math
    positions = jnp.asarray(positions)
    N = positions.shape[0]
    if not jax.config.jax_enable_x64:
        depth_cap = min(depth_cap, 9)                  # uint32 Morton ceiling (see
    md_given = max_depth is not None                   # _check_32bit_depth)
    md = max_depth if md_given else max(2, math.ceil(math.log(max(N / max(N_max, 1), 1.0))
                                                     / math.log(8)))
    mb = int(max_boxes) if max_boxes is not None else max(4 * N // N_max, 512) + 4096
    while True:
        for _ in range(10):
            out = build_fmm_tree(positions, mb, max_depth=md, N_max=N_max, domain=domain,
                              periodic_axes=periodic_axes)
            if not bool(out[10]):                          # max_boxes_exceeded
                break
            mb *= 2
        else:
            raise ValueError("build_fmm_tree_auto: box buffer kept overflowing; "
                             "check max_depth/N_max")
        lvls_n, counts_n, leaf_n = (np.asarray(out[i]) for i in (2, 3, 4))
        maxleaf = int(counts_n[leaf_n & (lvls_n > 0)].max())
        if md_given or maxleaf <= refine_factor * N_max or md >= depth_cap:
            return out, mb, md, maxleaf
        md += 1


# ─── Particle assignment ───────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=['max_depth', 'N_max'])
def assign_particles(sort_idx, m_id, lvls, counts, is_leaf, max_depth, N_max=None):
    r"""Assign each particle to its leaf box.

    Parameters
    ----------
    sort_idx : jnp.ndarray
        ``(N,)`` int32 — argsort of particles by Morton code (from
        :func:`build_fmm_tree`).
    m_id : jnp.ndarray
        ``(max_boxes,)`` uint64 — Morton code of each box.
    lvls : jnp.ndarray
        ``(max_boxes,)`` int32 — level of each box (0 = inactive).
    counts : jnp.ndarray
        ``(max_boxes,)`` int32 — particle count per box.
    is_leaf : jnp.ndarray
        ``(max_boxes,)`` bool.
    max_depth : int
        Static.
    N_max : int, optional
        Static — max particles per leaf; if given, also returns
        ``particles_in_leaf`` of shape ``(max_boxes, N_max)``.

    Returns
    -------
    leaf_of_particle : jnp.ndarray
        ``(N,)`` int32 — index into the box arrays of the leaf box containing
        each particle.
    particles_in_leaf : jnp.ndarray
        ``(max_boxes, N_max_or_1)`` int32 — for each leaf box, the sorted
        original particle indices it contains, padded with -1. Shape is
        ``(max_boxes, 1)`` when ``N_max`` is None.
    """
    N = sort_idx.shape[0]
    max_boxes = m_id.shape[0]
    MAX_UINT64 = ~jnp.array(0, dtype=jnp.uint64)

    # --- Step 1: scaled max_depth Morton code for each leaf box -------------- #
    # Leaf at level l with Morton code M → first max_depth cell: M << 3*(max_depth-l)
    level_diff = (jnp.int32(max_depth) - lvls).astype(jnp.uint64)
    scaled_mc = jnp.where(
        is_leaf & (lvls > 0),
        m_id << (jnp.uint64(3) * level_diff),
        MAX_UINT64,          # non-leaves and inactive sort to end
    )  # (max_boxes,) uint64

    # --- Step 2: sort leaves by first-particle Morton code ------------------- #
    leaf_order = jnp.argsort(scaled_mc)  # (max_boxes,) — leaves first, non-leaves last

    # --- Step 3: cumulative leaf counts → sorted particle ranges ------------- #
    leaf_counts = jnp.where(is_leaf & (lvls > 0), counts, jnp.int32(0))
    sorted_leaf_counts   = leaf_counts[leaf_order]          # (max_boxes,) int32
    sorted_leaf_cumcounts = jnp.cumsum(sorted_leaf_counts)  # (max_boxes,) int32

    # --- Step 4: assign sorted particle k → leaf rank via searchsorted ------- #
    # rank r satisfies: cumcounts[r-1] <= k < cumcounts[r]
    leaf_rank = jnp.searchsorted(
        sorted_leaf_cumcounts,
        jnp.arange(N, dtype=jnp.int32),
        side='right',
    ).astype(jnp.int32)  # (N,)

    # Box index of the leaf for sorted particle k
    leaf_of_sorted = leaf_order[leaf_rank].astype(jnp.int32)  # (N,)

    # --- Step 5: scatter back to original particle order -------------------- #
    leaf_of_particle = (
        jnp.zeros(N, dtype=jnp.int32)
        .at[sort_idx]
        .set(leaf_of_sorted, unique_indices=True)
    )  # (N,)

    # --- Step 6: particles_in_leaf ------------------------------------------ #
    if N_max is None:
        return leaf_of_particle, jnp.full((max_boxes, 1), jnp.int32(-1))

    # Position of sorted particle k within its leaf (0-based)
    start_by_rank = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), sorted_leaf_cumcounts[:-1]]
    )  # (max_boxes,)
    pos_in_leaf = (
        jnp.arange(N, dtype=jnp.int32) - start_by_rank[leaf_rank]
    )  # (N,), values 0..counts[leaf]-1

    particles_in_leaf = (
        jnp.full((max_boxes, N_max), jnp.int32(-1))
        .at[leaf_of_sorted, pos_in_leaf]
        .set(sort_idx, mode='drop')  # drop writes where pos_in_leaf >= N_max
    )  # (max_boxes, N_max)

    return leaf_of_particle, particles_in_leaf


@partial(jax.jit, static_argnames=['max_depth'])
def assign_particles_lean(sort_idx, m_id, lvls, counts, is_leaf, max_depth):
    r"""Leaf-major sorted layout WITHOUT materialising ``particles_in_leaf``.

    The dense ``(max_boxes, maxleaf)`` buffer of :func:`assign_particles`
    dominates memory at large ``N`` — this is the memory-diet lever for the
    lean eval path. The Morton sort already places each leaf's particles
    contiguously, so ``perm`` IS ``sort_idx`` and only box-sized bookkeeping is
    needed.

    Parameters
    ----------
    sort_idx, m_id, lvls, counts, is_leaf, max_depth
        As in :func:`assign_particles`.

    Returns
    -------
    perm : jnp.ndarray
        ``(N,)`` int32 — ``positions[perm]`` is leaf-major sorted.
    rank : jnp.ndarray
        ``(N,)`` int32 — inverse of ``perm``.
    starts : jnp.ndarray
        ``(max_boxes,)`` int32 — box ``b`` owns slice
        ``[starts[b], starts[b]+cnts[b])``.
    cnts : jnp.ndarray
        ``(max_boxes,)`` int32 — per-box particle counts.
    """
    N = sort_idx.shape[0]
    max_boxes = m_id.shape[0]
    MAX_UINT = ~jnp.array(0, dtype=m_id.dtype)
    level_diff = (jnp.int32(max_depth) - lvls).astype(m_id.dtype)
    scaled_mc = jnp.where(is_leaf & (lvls > 0),
                          m_id << (jnp.asarray(3, m_id.dtype) * level_diff), MAX_UINT)
    leaf_order = jnp.argsort(scaled_mc).astype(jnp.int32)  # leaves in Morton order
    leaf_counts = jnp.where(is_leaf & (lvls > 0), counts, jnp.int32(0))
    sorted_leaf_counts = leaf_counts[leaf_order]
    cum = jnp.cumsum(sorted_leaf_counts)
    start_by_rank = jnp.concatenate([jnp.zeros(1, cum.dtype), cum[:-1]]).astype(jnp.int32)
    starts = jnp.zeros(max_boxes, jnp.int32).at[leaf_order].set(start_by_rank)
    rank = jnp.zeros(N, jnp.int32).at[sort_idx].set(jnp.arange(N, dtype=jnp.int32),
                                                    unique_indices=True)
    return sort_idx.astype(jnp.int32), rank, starts, leaf_counts

