r"""Equivalent- and check-surface quadrature points for the KIFMM cube surfaces.

pvfmm-style KIFMM discretises each surface as a cube with a :math:`p \times p`
regular grid on each face.  In units of the box half-side, the equivalent
surface sits at radius ``R_EQUIV`` = 1.05 (just outside the unit box, half-side
1) and the check surface at ``R_CHECK`` = 2.95 (far field).

The number of unique surface points for expansion order :math:`p` is

.. math:: n_\mathrm{surf}(p) = 6p^2 - 12p + 8,

e.g. 152 points per surface at :math:`p = 6` (the pvfmm default).
"""

import numpy as np

__all__ = ["n_surf", "surface_pts", "R_EQUIV", "R_CHECK"]

# Surface radii (in units of box half-side), matching pvfmm defaults.
R_EQUIV: float = 1.05   # equivalent surface — just outside the unit box
R_CHECK: float = 2.95   # check surface — far field


def n_surf(p: int) -> int:
    r"""Number of unique surface points for a :math:`p \times p` grid per face.

    Parameters
    ----------
    p : int
        Points per cube-face edge (expansion order).

    Returns
    -------
    int
        :math:`6p^2 - 12p + 8` — shared edge/corner points counted once.
    """
    return 6 * p * p - 12 * p + 8


def surface_pts(p: int, r: float = 1.0) -> np.ndarray:
    r"""Generate quadrature points on a cube surface with half-side ``r``.

    Parameters
    ----------
    p : int
        Points per edge (must be >= 2).
    r : float, optional
        Half-side of the cube (default 1.0, the unit box).

    Returns
    -------
    pts : (n_surf(p), 3) ndarray of float64
        Surface points, ordered by face then row-major within each face.
        Duplicate edge/corner points are removed; the returned ordering is
        deterministic (``np.unique`` with ``return_index``, then sort).

    Raises
    ------
    ValueError
        If ``p < 2``.
    """
    if p < 2:
        raise ValueError(f"p must be ≥ 2, got {p}")
    t = np.linspace(-r, r, p)
    # (p, p) grids for the two "free" coordinates on each face
    g0, g1 = np.meshgrid(t, t, indexing="ij")  # (p, p) each
    g0 = g0.reshape(-1)
    g1 = g1.reshape(-1)
    face_const = r * np.ones(p * p)

    # 6 faces: +x, -x, +y, -y, +z, -z
    faces = [
        np.stack([+face_const, g0, g1], axis=1),   # +x
        np.stack([-face_const, g0, g1], axis=1),   # -x
        np.stack([g0, +face_const, g1], axis=1),   # +y
        np.stack([g0, -face_const, g1], axis=1),   # -y
        np.stack([g0, g1, +face_const], axis=1),   # +z
        np.stack([g0, g1, -face_const], axis=1),   # -z
    ]
    pts = np.vstack(faces)                          # (6*p^2, 3)

    # Deduplicate: round to 14 significant figures to avoid float noise.
    pts_key = np.round(pts / max(r, 1e-300), 14)
    _, first_occ = np.unique(pts_key, axis=0, return_index=True)
    return pts[np.sort(first_occ)]                  # (n_surf(p), 3), stable order
