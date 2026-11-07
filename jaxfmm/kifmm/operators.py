r"""Precomputed KIFMM operator matrices (Laplace kernel, pvfmm-style convention).

All operators are computed for a UNIT CHILD box (half-side 1); the parent box
has half-side 2, with child centres at :math:`(\pm 1, \pm 1, \pm 1)`.

Surface radii, in units of the box half-side (see :mod:`jaxfmm.kifmm.surfaces`):

* ``R_EQUIV`` = 1.05 — "small", just outside the unit box;
* ``R_CHECK`` = 2.95 — "large", well outside the unit box.

Upward pass (multipole): the upward equivalent surface is the small one and
the upward check surface the large one, with

.. math:: \mathrm{pinv\_up} = K(\mathrm{check\_up}, \mathrm{equiv\_up})^{+}.

Downward pass (local expansion): the roles are SWAPPED — the downward
equivalent surface is the large one (it represents the incoming far field) and
the downward check surface the small one, with
:math:`\mathrm{pinv\_dn} = K(\mathrm{check\_dn}, \mathrm{equiv\_dn})^{+}
= \mathrm{pinv\_up}^{T}`.

L2P evaluates :math:`K(\mathrm{eval}, \mathrm{equiv\_dn} + c)\,\xi` at points
inside the box.  Since the downward equivalent radius (2.95) exceeds the box
half-side (1), which exceeds the eval radius, the maximum principle in the
ball of radius 1.05 guarantees accuracy everywhere inside.

Precomputed operators (see :func:`build_operators` for the full key list):

* ``pinv_up``/``pinv_dn`` : (n_s, n_s) check-potential → equivalent-density
  decodes (materialised legacy form; the factored triples ``F_up``/``F_dn``/
  ``F_par`` are the preferred backward-stable form).
* ``M2M_mats`` : (8, n_s, n_s), child ``k``:
  :math:`\sigma_\mathrm{parent} \mathrel{+}= \mathrm{M2M}_k\,\sigma_\mathrm{child}`
  with :math:`\mathrm{M2M}_k = \mathrm{pinv\_parent}\,
  K(\mathrm{parent\_check}, \mathrm{child\_equiv\_up})`.
* ``M2L_K_table`` : (7, 7, 7, n_s, n_s) RAW kernel matrices (factored,
  backward-stable form — ``pinv_dn`` NOT folded in):
  ``M2L_K_table[dx+3, dy+3, dz+3]`` :math:`= K(\mathrm{check\_dn} + d,
  \mathrm{equiv\_up})` with :math:`d = 2(dx, dy, dz)` in half-side units and
  :math:`(dx, dy, dz) \in [-3, 3]^3`.  Adjacent entries
  (:math:`\|d\|_\infty \le 1`) are zero.  Applied as

  .. math::
     \mathrm{local}[t] = \mathrm{pinv\_dn} \sum_{s \in V(t)}
     \mathrm{M2L\_K}[\mathrm{disp}(t, s)]\, \mathrm{equiv}[s].

* ``L2L_mats`` : (8, n_s, n_s), child ``k``:
  :math:`\xi_\mathrm{child} \mathrel{+}= \mathrm{L2L}_k\,\xi_\mathrm{parent}`
  with :math:`\mathrm{L2L}_k = \mathrm{pinv\_dn}\,
  K(\mathrm{child\_check\_dn}, \mathrm{parent\_equiv\_dn})`.
"""

import numpy as np
from typing import Dict

from jaxfmm.kifmm.surfaces import surface_pts, R_EQUIV, R_CHECK, n_surf
from jaxfmm.kernels.laplace import laplace_G_np
from jaxfmm.kernels.laplace.kernel import LAPLACE

__all__ = ["build_operators", "cast_operators", "child_offsets",
           "gen_pbc_op_kifmm", "expand_m2l_table", "build_operators_leveled"]

# Child-octant centres in PARENT coordinates (parent half-side = 2, child half-side = 1).
# Morton ordering: bit 0 = x, bit 1 = y, bit 2 = z.
child_offsets = np.array(
    [(-1, -1, -1), (1, -1, -1), (-1, 1, -1), (1, 1, -1),
     (-1, -1,  1), (1, -1,  1), (-1, 1,  1), (1, 1,  1)],
    dtype=np.float64,
)   # (8, 3)  — child centres at (±1, ±1, ±1) relative to parent centre


def _pinv_factors(A: np.ndarray, eps_rel: float = 1e-7):
    r"""Damped (Tikhonov) pseudoinverse as factors.

    Parameters
    ----------
    A : (n, n) ndarray
        Matrix to invert (the ill-conditioned check→equivalent kernel matrix).
    eps_rel : float, optional
        Relative damping: :math:`\varepsilon = \mathrm{eps\_rel} \cdot s_\max`.

    Returns
    -------
    U, d, Vt : ndarray
        Factors with :math:`A^{+} = V^T \operatorname{diag}(d)\, U^T` and
        :math:`d_i = s_i / (s_i^2 + \varepsilon^2)`.

    Notes
    -----
    Keeping the factors and applying them sequentially (orthogonal matvec →
    diagonal scaling → orthogonal matvec) is the BACKWARD-STABLE form of the
    KIFMM decode (pvfmm, cicp15 §2.1): materialising the product spans the
    cond-sized dynamic range, and its cancellation noise limits f64 to ~9
    digits at high P and breaks f32 entirely (e.g. f32 P=3 far field: 3.7e-2
    materialised vs 8.2e-4 factored).
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    eps = eps_rel * s[0]
    return U, s / (s * s + eps * eps), Vt


def _pinv(A: np.ndarray, eps_rel: float = 1e-7) -> np.ndarray:
    r"""Backward-stable damped (Tikhonov) pseudoinverse, materialised.

    Legacy form — prefer the :func:`_pinv_factors` triple (see there).

    Parameters
    ----------
    A : (n, n) ndarray
        Matrix to invert.
    eps_rel : float, optional
        Relative damping (see :func:`_pinv_factors`).

    Returns
    -------
    (n, n) ndarray
        The damped pseudoinverse.

    Notes
    -----
    Instead of the hard truncated-SVD inverse
    :math:`\operatorname{diag}(1/s_i \text{ for } s_i > \text{thresh}, 0)`,
    this uses the smooth filter :math:`s_i / (s_i^2 + \varepsilon^2)` with
    :math:`\varepsilon = \mathrm{eps\_rel} \cdot s_\max`.  Damping (rather
    than sharply truncating) small singular values gives far
    better-conditioned KIFMM operators (e.g. :math:`\|A^{+}\|` roughly 1000x
    smaller) at essentially the same accuracy; norm-controlled operators are
    numerically more robust and keep the equivalent densities small.
    """
    U, s_inv, Vt = _pinv_factors(A, eps_rel)
    return (Vt.T * s_inv) @ U.T


def cast_operators(ops: Dict, dtype) -> Dict:
    r"""Cast every float64 array in an operator dict to ``dtype``.

    Factor tuples included — e.g.
    ``cast_operators(build_operators(3), np.float32)`` for the
    single-precision KIFMM mode (viable at P=3-4 with the factored decode).

    Parameters
    ----------
    ops : dict
        Operator dict from :func:`build_operators`.
    dtype : numpy dtype
        Target floating dtype.

    Returns
    -------
    dict
        New dict with all float64 members cast; other members unchanged.
    """
    def c(v):
        if isinstance(v, tuple):
            return tuple(c(x) for x in v)
        a = np.asarray(v)
        return np.asarray(a, dtype) if a.dtype == np.float64 else v
    return {k: c(v) for k, v in ops.items()}


def _m2l_symmetry_classes(M2L_K_table, equiv_pts, check_pts):
    r"""Octahedral symmetry compression of the M2L table (pvfmm sec 3.1).

    The Laplace kernel is invariant under the 48-element cube group (axis
    permutations × sign flips), and the KIFMM cube surfaces are closed under
    it, so every displacement's kernel matrix is a row/column PERMUTATION of
    its class representative (keyed by the sorted :math:`|d|` multiset):
    343 matrices reduce to 20 distinct ones (~17x table memory at high P).

    Parameters
    ----------
    M2L_K_table : (7, 7, 7, n_s, n_s) ndarray
        Full raw M2L kernel table.
    equiv_pts, check_pts : (n_s, 3) ndarray
        The two cube surfaces (needed to derive the point permutations).

    Returns
    -------
    dict
        ``M2L_cls`` : (343,) int32 class id per flattened displacement;
        ``M2L_K_class`` : (20, n_s, n_s) class representatives;
        ``M2L_perm_eq`` / ``M2L_perm_ch`` : (343, n_s) int32 permutations with
        ``K[d] == K_class[cls[d]][perm_ch[d], :][:, perm_eq[d]]``
        (the identity the reconstruction/appliers use; disk caches store ONLY
        the class form).

    Raises
    ------
    ValueError
        If the reconstruction is not exact to roundoff — i.e. the kernel is
        not invariant under the cube group.
    """
    n_s = M2L_K_table.shape[-1]
    K = M2L_K_table.reshape(343, n_s, n_s)
    disps = np.array([(i, j, k) for i in range(-3, 4) for j in range(-3, 4)
                      for k in range(-3, 4)])
    reps = {}
    cls = np.zeros(343, np.int32)
    perm_eq = np.zeros((343, n_s), np.int32)
    perm_ch = np.zeros((343, n_s), np.int32)
    K_class = []

    def surf_perm(pts, axperm, signs):
        # K[d][i,j] = K[dref][idx(g^-1 c_i), idx(g^-1 e_j)] for d = g(dref): the permutation
        # must be the INVERSE group action (an involutory test element would not detect a
        # direction mix-up here, so keep this explicit)
        q = pts[:, axperm] * signs[None, :]
        m = np.argmin(np.linalg.norm(q[:, None] - pts[None], axis=-1), axis=1)   # idx(g pt_i)
        assert len(np.unique(m)) == len(pts)
        inv = np.empty_like(m)
        inv[m] = np.arange(len(m))
        return inv

    from itertools import permutations, product
    for i, d in enumerate(disps):
        key = tuple(sorted(np.abs(d)))
        if key not in reps:
            reps[key] = (len(K_class), i, d)
            K_class.append(K[i])
        c, _, dref = reps[key]
        cls[i] = c
        # find the group element g with g(dref) == d, then perms of the surfaces under g
        found = False
        for axp in permutations(range(3)):
            if found:
                break
            for sg in product((1.0, -1.0), repeat=3):
                dd = np.asarray(dref)[list(axp)] * np.asarray(sg)
                if np.array_equal(dd.astype(int), d):
                    perm_eq[i] = surf_perm(np.asarray(equiv_pts), list(axp), np.asarray(sg))
                    perm_ch[i] = surf_perm(np.asarray(check_pts), list(axp), np.asarray(sg))
                    found = True
                    break
        assert found, d
    out = {"M2L_cls": cls, "M2L_K_class": np.stack(K_class),
           "M2L_perm_eq": perm_eq, "M2L_perm_ch": perm_ch}
    # build-time self-check: the reconstruction is exact to ROUNDOFF only (the table
    # entries are computed from differently-arranged float coordinates than their permuted
    # class representatives, so bitwise equality does NOT hold — ~1 ulp)
    for i in range(343):
        nrm = np.linalg.norm(K[i])
        if nrm > 0:
            rec = out["M2L_K_class"][cls[i]][np.ix_(perm_ch[i], perm_eq[i])]
            if np.linalg.norm(K[i] - rec) > 1e-12 * nrm:      # a REAL raise, not an assert:
                raise ValueError(                              # python -O strips asserts and
                    f"M2L symmetry-class reconstruction failed for displacement "
                    f"{tuple(disps[i])}: this kernel is not invariant under the cube "
                    f"group (tensor-valued or anisotropic kernels need a discovery/"
                    f"fallback generalisation)")
    return out


def expand_m2l_table(ops):
    r"""Reconstruct the full M2L table from the symmetry-class form.

    Parameters
    ----------
    ops : dict
        Operator dict carrying the class-form keys ``M2L_K_class``,
        ``M2L_cls``, ``M2L_perm_eq`` and ``M2L_perm_ch``
        (see :func:`_m2l_symmetry_classes`).

    Returns
    -------
    (7, 7, 7, n_s, n_s) ndarray
        The full raw M2L kernel table.
    """
    Kc = np.asarray(ops["M2L_K_class"])
    cls = np.asarray(ops["M2L_cls"])
    pe, pc = np.asarray(ops["M2L_perm_eq"]), np.asarray(ops["M2L_perm_ch"])
    n_s = Kc.shape[-1]
    K = np.stack([Kc[cls[i]][np.ix_(pc[i], pe[i])] for i in range(343)])
    return K.reshape(7, 7, 7, n_s, n_s)


def build_operators(p: int = 6, eps_rel: float = 1e-7, kernel=LAPLACE) -> Dict[str, np.ndarray]:
    r"""Precompute all KIFMM operator matrices for a 3D kernel.

    Parameters
    ----------
    p : int, optional
        Expansion order (points per cube-face edge, default 6).
    eps_rel : float, optional
        Relative damping for the backward-stable (Tikhonov) surface
        pseudoinverses (:math:`\varepsilon = \mathrm{eps\_rel} \cdot s_\max`
        in :math:`s_i/(s_i^2+\varepsilon^2)`).  The default 1e-7 preserves
        full float64 accuracy (~2e-6 at p=6, same as a hard SVD truncation)
        while giving ~100x better-conditioned operators; larger values (e.g.
        1e-6) damp more (better conditioned, slightly less accurate).
    kernel : jaxfmm.kernels.Kernel, optional
        Kernel spec (default Laplace).  Phases read ``g``/``g_grad`` from it;
        caches key on ``kernel.name``.

    Returns
    -------
    dict
        Operator tables:

        * ``pinv_up`` : (n_s, n_s) upward pseudoinverse
          :math:`K(\mathrm{check\_up}, \mathrm{equiv\_up})^{+}`;
        * ``pinv_dn`` : (n_s, n_s) downward pseudoinverse
          :math:`= \mathrm{pinv\_up}^T`;
        * ``F_up``/``F_dn``/``F_par`` : factored decode triples ``(U, d, Vt)``
          — the backward-stable form :func:`jaxfmm.kifmm.make_operators` uses
          by default; enables f32 at P=3-4 and deep f64 accuracy at high P;
        * ``equiv_pts`` : (n_s, 3) ``R_EQUIV`` surface (upward equiv /
          downward check);
        * ``check_pts`` : (n_s, 3) ``R_CHECK`` surface (upward check /
          downward equiv);
        * ``M2M_mats``/``L2L_mats`` : (8, n_s, n_s) materialised child/parent
          translation operators (legacy consumers);
        * ``M2M_K``/``L2L_K`` : (8, n_s, n_s) RAW kernels for the factored
          M2M/L2L;
        * ``M2L_K_table`` : (7, 7, 7, n_s, n_s) V-list RAW kernel matrices
          (factored form; apply ``pinv_dn`` once per target box);
        * ``M2L_cls``/``M2L_K_class``/``M2L_perm_eq``/``M2L_perm_ch`` :
          octahedral class compression, present only for cube-group-invariant
          kernels (see Notes);
        * ``kernel`` : the Kernel spec;
        * ``p``, ``n_s`` : expansion order and surface point count.

    Notes
    -----
    The octahedral class compression is emitted only when
    ``kernel.symmetric_Oh``; the builder VERIFIES the symmetry and raises
    otherwise.  Non-invariant kernels keep the raw table — larger but always
    correct (the volume module's disk-cache bundling handles the absent class
    form).
    """
    n_s = n_surf(p)
    equiv_pts = surface_pts(p, R_EQUIV)   # (n_s, 3) at radius 1.05 — upward equiv / downward check
    check_pts = surface_pts(p, R_CHECK)   # (n_s, 3) at radius 2.95 — upward check / downward equiv

    # ── Pseudoinverses (factored triples + materialised legacy forms) ──────
    Kce = kernel.G_np(check_pts, equiv_pts)   # (n_s, n_s): K(check_up, equiv_up)
    F_up = _pinv_factors(Kce, eps_rel)                   # upward decode, factored
    pinv_up = (F_up[2].T * F_up[1]) @ F_up[0].T          # materialised (legacy consumers)

    Kec = kernel.G_np(equiv_pts, check_pts)   # (n_s, n_s): K(check_dn, equiv_dn)
    F_dn = _pinv_factors(Kec, eps_rel)                   # = pinv_up.T analytically; recomputed for stability
    pinv_dn = (F_dn[2].T * F_dn[1]) @ F_dn[0].T

    # ── M2M: 8 child-to-parent upward operators ────────────────────────────
    # Parent box has half-side 2: check at R_CHECK*2, equiv at R_EQUIV*2.
    # Child k (half-side 1) centred at child_offsets[k] = (±1, ±1, ±1).
    parent_check = check_pts * 2.0               # (n_s, 3) at radius 5.9
    parent_equiv = equiv_pts * 2.0               # (n_s, 3) at radius 2.1
    Kcc_parent = kernel.G_np(parent_check, parent_equiv)
    F_par = _pinv_factors(Kcc_parent, eps_rel)           # parent upward decode, factored
    pinv_parent = (F_par[2].T * F_par[1]) @ F_par[0].T   # — 2*pinv_up by scale invariance

    M2M_mats = np.empty((8, n_s, n_s), dtype=np.float64)
    M2M_K = np.empty((8, n_s, n_s), dtype=np.float64)    # RAW child->parent-check kernels
    for k in range(8):
        # Child upward equiv in parent coordinates: radius R_EQUIV centred at child_offsets[k]
        child_equiv_up = equiv_pts + child_offsets[k]     # (n_s, 3)
        K_pk = kernel.G_np(parent_check, child_equiv_up)  # K(parent_check, child_equiv_up)
        M2M_K[k] = K_pk
        M2M_mats[k] = pinv_parent @ K_pk                  # materialised (legacy consumers)

    # ── M2L table: 7×7×7 displacement grid (FACTORED, pvfmm-style) ─────────
    # For same-level unit boxes (half-side 1), target at displacement d (grid units).
    # 1 grid unit = 2 half-sides, so target centre = (2*dx, 2*dy, 2*dz) in half-side units.
    #
    # SOURCE upward equiv:  equiv_pts  at (0,0,0), radius R_EQUIV=1.05
    # TARGET downward check: equiv_pts + d_vec,   radius R_EQUIV=1.05 centred at target
    #
    # We store the RAW kernel matrices K_d = K(equiv_pts + d_vec, equiv_pts) only
    # (NOT folded with pinv_dn).  At apply time the V-list sum gives the target's
    # downward CHECK potential, and pinv_dn is applied ONCE per target box to get
    # the local (downward-equivalent) density:
    #     check_pot[t] = Σ_{s∈V(t)} K_{disp(t,s)} @ equiv[s]
    #     local[t]     = pinv_dn @ check_pot[t]
    # This is the backward-stable factored form (pinv applied to a physical check
    # potential rather than folded into every operator); it is also exactly the
    # convolution structure used for FFT-accelerated M2L.
    M2L_K_table = np.zeros((7, 7, 7, n_s, n_s), dtype=np.float64)
    for dxi in range(7):
        for dyi in range(7):
            for dzi in range(7):
                dx, dy, dz = dxi - 3, dyi - 3, dzi - 3
                if max(abs(dx), abs(dy), abs(dz)) <= 1:
                    continue   # adjacent — U-list territory
                d_vec = np.array([2.0 * dx, 2.0 * dy, 2.0 * dz])
                target_check_dn = equiv_pts + d_vec
                M2L_K_table[dxi, dyi, dzi] = kernel.G_np(target_check_dn, equiv_pts)

    # ── L2L: 8 parent-to-child downward operators ─────────────────────────
    # Parent (half-side 2) downward equiv: check_pts * 2.0, radius R_CHECK*2=5.9.
    # Child k (half-side 1) centred at child_offsets[k]:
    #   downward check: equiv_pts + child_offsets[k], radius R_EQUIV=1.05
    #   downward equiv: check_pts + child_offsets[k], radius R_CHECK=2.95
    #
    # L2L_k = pinv_dn @ K(child_check_dn, parent_equiv_dn)
    parent_equiv_dn = check_pts * 2.0   # same array as parent_check used in M2M — parent's downward equiv

    L2L_mats = np.empty((8, n_s, n_s), dtype=np.float64)
    L2L_K = np.empty((8, n_s, n_s), dtype=np.float64)    # RAW parent->child-check kernels
    for k in range(8):
        child_check_dn = equiv_pts + child_offsets[k]    # (n_s, 3) downward check
        K_ck = kernel.G_np(child_check_dn, parent_equiv_dn)
        L2L_K[k] = K_ck
        L2L_mats[k] = pinv_dn @ K_ck                     # materialised (legacy consumers)

    return {
        "pinv_up"   : pinv_up,
        "pinv_dn"   : pinv_dn,
        "F_up"      : F_up,       # factored decodes (U, d, Vt): the backward-stable
        "F_dn"      : F_dn,       # form make_operators uses by default; enables f32
        "F_par"     : F_par,      # at P=3-4 and deep f64 accuracy at high P
        "M2M_K"     : M2M_K,      # raw kernels for the factored M2M/L2L
        "L2L_K"     : L2L_K,
        "equiv_pts" : equiv_pts,
        "check_pts" : check_pts,
        "M2M_mats"  : M2M_mats,
        "M2L_K_table" : M2L_K_table,   # RAW kernel matrices; pinv_dn applied per box
        "kernel"    : kernel,          # the Kernel spec: phases read g/g_grad, caches key on .name
        # octahedral class compression only for cube-group-invariant kernels (the builder
        # VERIFIES and raises otherwise); non-invariant kernels keep the raw table — larger
        # but always correct (volume's disk-cache bundling handles the absent class form).
        **(_m2l_symmetry_classes(M2L_K_table, equiv_pts, check_pts) if kernel.symmetric_Oh else {}),
        "L2L_mats"  : L2L_mats,
        "p"         : p,
        "n_s"       : n_s,
    }


def gen_pbc_op_kifmm(p: int = 6, p_harm: int = 10, periodic_axes=(0, 1, 2),
                     pbc_lvls: int = 30, eps_rel: float = 1e-7):
    r"""KIFMM lattice operator for periodic boundaries.

    A dense ``(n_s, n_s)`` matrix mapping the unit CELL's upward-equivalent
    density to the raw downward-check potentials representing ALL periodic
    images beyond the wrapped 27-neighbourhood (:math:`\|d\|_\infty \ge 2` in
    cell units along the periodic axes).

    Parameters
    ----------
    p : int, optional
        KIFMM expansion order (surface grid).
    p_harm : int, optional
        Harmonic order of the lattice bridge (default 10, roughly
        referee-grade for the :math:`\|d\|_\infty \ge 2` shell; the
        ``eps_rel`` pinv floor, ~1e-7 at p=6, usually dominates).
    periodic_axes : tuple of int, optional
        Axes with periodic boundary conditions.
    pbc_lvls : int, optional
        Number of lattice doubling levels for the harmonic operator.
    eps_rel : float, optional
        Pseudoinverse damping (kept for interface symmetry).

    Returns
    -------
    op : (n_s, n_s) ndarray
        The lattice operator (raw check potentials; see Notes).
    info : dict
        ``dict(levels, p_harm)``.

    Raises
    ------
    ValueError
        If ``p_harm > 12`` (the rotation-based harmonic machinery is untested
        past ~10 and produces NaN well beyond that), or if the assembled
        operator is non-finite.

    Notes
    -----
    Assembled as a HARMONIC-BASIS BRIDGE, not by doubling in the surface
    basis:

    .. math:: \mathrm{op} = B_0 + E \cdot \mathrm{PBC}_\mathrm{harm} \cdot M,

    where :math:`M` (multipoles of unit charges at the equivalent points) and
    :math:`E` (check-surface potentials of unit local coefficients) are
    identity-column pushes through the production
    ``get_initial_mpls``/``eval_local``, :math:`\mathrm{PBC}_\mathrm{harm}` is
    :func:`jaxfmm.pbc.gen_pbc_op` at theta=0.77 — whose image pattern
    (num_images=1, mul=3) is EXACTLY the cubic 2:1 MAC's 27-neighbourhood
    split, and whose conditionally-convergent shell terms vanish STRUCTURALLY
    under its mirror mask — and :math:`B_0` is the exact level-0 shell.

    Why not doubling in the surface basis: it does not converge numerically —
    the aggregated supercell density's dipole content grows ~:math:`9^l` while
    the shell kernels' dipole cancellation is only rounding-level, so the
    increment noise grows ~:math:`3^l` and the iteration stalls far from
    convergence.  The harmonic detour is the structural fix; the net-charge
    divergence is handled by the harmonic operator's ``pbc_no_monopole`` and
    additionally projected off the input here (charged cells need an Ewald
    background regardless).

    The operator outputs RAW downward-CHECK potentials, to be ADDED to the
    M2L phase's check-potential accumulation before the pipeline's single
    per-box ``pinv_dn``.  Folding the pinv into the lattice operator amplifies
    the harmonic truncation's off-range components by ~:math:`\|A^{+}\|`,
    producing O(1) errors — riding the shared pinv keeps that component small
    relative to the tree M2L signal.

    Geometry is in CELL half-side units (cell = :math:`[-1, 1]^3`, centre 0).
    """
    import jax
    import jax.numpy as jnp
    try:
        from jax.experimental import enable_x64          # removed in newer jax
    except ImportError:
        from contextlib import contextmanager
        @contextmanager
        def enable_x64():                                # eager-only (setup): toggle the global flag
            prev = bool(jax.config.jax_enable_x64)
            jax.config.update("jax_enable_x64", True)
            try:
                yield
            finally:
                jax.config.update("jax_enable_x64", prev)
    from jaxfmm.pbc import gen_pbc_op
    from jaxfmm.flex import get_initial_mpls, eval_local

    if not jax.config.jax_enable_x64:
        with enable_x64():                       # the harmonic machinery needs f64 (p_harm=10
            return gen_pbc_op_kifmm(p, p_harm, periodic_axes, pbc_lvls, eps_rel)   # overflows f32)

    if p_harm > 12:
        raise ValueError("gen_pbc_op_kifmm: p_harm > 12 — the rotation-based harmonic "
                         "machinery is untested past ~10 and produces NaN at ~22.")
    n_s = n_surf(p)
    equiv = surface_pts(p, R_EQUIV)
    check = surface_pts(p, R_CHECK)

    # harmonic lattice operator on the cell (theta=0.77 -> the cubic 27-neighbourhood split)
    L = jnp.array([[2.0, 2.0, 2.0]])
    # deep tail ONLY (|d|_inf >= 5): the level-0 shell is handled EXACTLY below — the
    # KIFMM surfaces stick out to r~1.82 vs the level-0 shell at distance 4 (harmonic
    # ratio 0.91, hopeless); from level 1 the ratio is <= 0.36 (fine at p_harm=10).
    PBC_h, lst = gen_pbc_op(L, 0.77, periodic_axes, p_harm, pbc_lvls, True, first_shell_lvl=1)
    assert lst[0][0].shape[0] == 3 ** len(periodic_axes), "theta=0.77 must give the 27-set"
    dim = (p_harm + 1) ** 2

    cen = jnp.zeros((1, 3))
    eq = jnp.asarray(equiv)[None]                       # (1, n_s, 3)
    # M: (dim, n_s) — column i = cell multipole of a unit charge at equiv point i
    eyeS = jnp.eye(n_s)                                 # python loop, not vmap (see pbc._m2l_op)
    M = np.stack([np.asarray(get_initial_mpls(eq, eyeS[i][None], cen, p_harm, jnp.inf)[0])
                  for i in range(n_s)], axis=1)
    # E: (n_s, dim) — row j = potential of a unit local coefficient at DOWNWARD-check
    # point j (= the R_EQUIV surface; the local expansion is valid there, r=1.05 < 3)
    ck = jnp.asarray(equiv)[None]
    rev = jnp.arange(n_s)
    eyeD = jnp.eye(dim)
    E = np.stack([np.asarray(eval_local(eyeD[c][None], ck, rev, cen, p_harm, False, jnp.inf))
                  for c in range(dim)], axis=1)
    # NO pinv_dn here: the operator outputs RAW downward-CHECK potentials, added to the
    # m2l phase's check-potential accumulation before the pipeline's single per-box
    # pinv_dn (see Notes — folding the pinv in amplifies the truncation error).
    # EXACT level-0 shell (|d|_inf in 2..4): plain kernel-matrix sums, production accuracy
    per = [np.arange(-1, 2) if ax in periodic_axes else np.array([0]) for ax in range(3)]
    nw = np.stack(np.meshgrid(*per, indexing="ij"), -1).reshape(-1, 3)
    parents = nw[np.any(nw != 0, axis=1)]
    shell0 = (3 * parents[:, None, :] + nw[None, :, :]).reshape(-1, 3)
    B0 = np.zeros((n_s, n_s))
    for dd in shell0:
        B0 += laplace_G_np(equiv, equiv + 2.0 * dd)       # check_dn pts <- image equiv_up pts
    op = B0 + np.asarray(E) @ np.asarray(PBC_h) @ np.asarray(M)
    op = op @ (np.eye(n_s) - np.ones((n_s, n_s)) / n_s)   # neutral input projection
    if not np.isfinite(op).all():
        raise ValueError(
            "gen_pbc_op_kifmm: operator non-finite. Diagnostics: "
            f"E {bool(np.isfinite(np.asarray(E)).all())}, "
            f"PBC_h {bool(np.isfinite(np.asarray(PBC_h)).all())}, M {bool(np.isfinite(np.asarray(M)).all())}, "
            f"bad entries {int((~np.isfinite(op)).sum())}/{op.size}")
    return op, dict(levels=len(lst), p_harm=p_harm)


def build_operators_leveled(p, max_depth, domain_extent, kernel, eps_rel=1e-7):
    r"""Per-level KIFMM operator tables for NON-HOMOGENEOUS kernels.

    The plain :func:`build_operators` set is assembled on UNIT surfaces and
    rescaled by box size at runtime — exact only for degree :math:`-1`
    homogeneous kernels (Laplace).  For a general kernel the operators depend
    on the PHYSICAL box size, so one set is assembled per level.  Trick:
    unit-geometry assembly with the SCALED kernel
    :math:`G_s(x, y) = G(s x, s y)` equals physical assembly at scale
    :math:`s` — the whole existing builder is reused verbatim per level
    (level :math:`l` half-side
    :math:`s_l = \mathrm{domain\_extent} \cdot 0.5^{l+1}`, the
    :mod:`jaxfmm.kifmm` convention).

    Parameters
    ----------
    p : int
        Expansion order.
    max_depth : int
        Tree depth; one operator set per level ``0..max_depth``.
    domain_extent : float
        Root box edge length (sets the physical scale per level).
    kernel : jaxfmm.kernels.Kernel
        The (non-homogeneous) kernel spec.
    eps_rel : float, optional
        Pseudoinverse damping (see :func:`build_operators`).

    Returns
    -------
    dict
        The per-level-INDEXED members the leveled kifmm phases consume:
        ``F_up_lv``/``F_dn_lv`` (stacked factored decodes),
        ``M2M_K_lv``/``L2L_K_lv`` ((max_depth, 8, n_s, n_s), indexed by PARENT
        level; assembled at the CHILD scale like the unit pair),
        ``M2L_K_lv`` ((max_depth+1, 343, n_s, n_s)), plus
        ``equiv_pts``/``check_pts`` (unit; phases scale physically),
        ``kernel``, ``max_depth`` and ``leveled=True``.

    Notes
    -----
    Memory: :math:`(\mathrm{max\_depth}+1) \cdot 343 \cdot n_s^2` float64 —
    ~70 MB at p=4, ~500 MB at p=6.
    """
    per_level = []
    for l in range(max_depth + 1):
        s = float(domain_extent) * 0.5 ** (l + 1)
        kern_s = kernel._replace(
            name=f"{kernel.name}[s={s:.3e}]",
            G_np=(lambda x, y, _s=s: kernel.G_np(x * _s, y * _s)))
        per_level.append(build_operators(p, eps_rel=eps_rel, kernel=kern_s))
    def stack(key):
        return np.stack([o[key] for o in per_level])
    def stack_factors(key):
        return tuple(np.stack([np.asarray(o[key][i]) for o in per_level]) for i in range(3))
    return {
        "leveled"   : True,
        "max_depth" : int(max_depth),
        "kernel"    : kernel,
        "equiv_pts" : per_level[0]["equiv_pts"],
        "check_pts" : per_level[0]["check_pts"],
        "F_up_lv"   : stack_factors("F_up"),     # decode at level l (parent decode for M2M)
        "F_dn_lv"   : stack_factors("F_dn"),
        # parent level pl couples children at pl+1: the unit (parent=2, child=1) pair
        # scaled by the CHILD half-side -> take the child level's assembly
        "M2M_K_lv"  : np.stack([per_level[pl + 1]["M2M_K"] for pl in range(max_depth)]),
        "L2L_K_lv"  : np.stack([per_level[cl]["L2L_K"] for cl in range(1, max_depth + 1)]),
        "M2L_K_lv"  : stack("M2L_K_table").reshape(max_depth + 1, 343,
                                                   per_level[0]["M2L_K_table"].shape[-2],
                                                   per_level[0]["M2L_K_table"].shape[-1]),
    }
