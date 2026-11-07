r"""The ``err_tol`` ladder: calibrated element-path parameter rungs.

``tol`` is the CALIBRATION-SUITE MEAN relative H error of the rung — an *indication*
of the expected relative magnetisation-field error on a reasonable-quality mesh, not
a bound. Suite (2026-09-12 calibration, Musica H100, comfort mode): structured film
nz=1 (1.5e6 tets), structured cube (1.0e6), gmsh sphere shell (1.4e6), gmsh eggcarton
(1.9e5); band-limited random m (seeds 7, 8) against an f64 p=10/theta=0.4/(10,11)/
duffy-12 reference (unitcube floor 8e-7 — the earlier p=7/duffy-4 reference carried a
4.4e-4 singular-rule error and could not label anything below ~1e-3). Raw rows:
``dev/thesis/bench_strayfield/plotdata/calib_ladder/`` (CALROW, jobs 1727359-63);
t_H = unitcube 9.6e6 tets, H100, comfort mode (``data_params_element`` front rows).
Labels and table re-measured at 0.5.0 (17325bb, 2026-10-18, jobs 1852300-04, same suite and
reference, rows ``plotdata/calib_050/``): R0-R8 within 0.98-1.05 of the 09-12 means, R9 0.83
(reference floor); R7-R9 relabelled.

====  ====  =========================  =======  =======  =======  =====
rung  dt    parameters                 mean     worst    white    t_H
====  ====  =========================  =======  =======  =======  =====
R0    f32   p3  th0.77 (1,3)           7.5e-3   1.6e-2   1.8e-2    11 ms
R1    f32   p4  th0.85 (3,4) tier      3.9e-3   7.4e-3   4.8e-3    27
R2    f32   p4  th0.70 (3,4) tier      2.1e-3   4.3e-3   3.7e-3    30
R3    f64   p6  th0.70 (3,4)  duffy 8  4.3e-4   7.0e-4   1.0e-3    54
R4    f64   p7  th0.70 (5,6)  duffy 8  1.4e-4   2.8e-4   1.9e-4   102
R5    f64   p8  th0.70 (5,6)  duffy 12 6.2e-5   1.1e-4   1.2e-4   123
R6    f64   p8  th0.55 (5,9)  duffy 12 1.2e-5   1.8e-5   8.6e-5   192
R7    f64   p10 th0.65 (7,9)  duffy 12 9.2e-6   1.7e-5   1.7e-5   304
R8    f64   p11 th0.60 (7,9)  duffy 12 3.7e-6   7.8e-6   1.2e-5   368
R9    f64   p12 th0.60 (10,11) duffy 12 1.1e-6  1.6e-6   2.2e-6   651
====  ====  =========================  =======  =======  =======  =====

(N_max=128 except R3/R5/R6 at 256, s="auto", N_max_eval=32.) Spacing follows the measured front law
t ∝ err^-0.45 (~2.7x per decade): 1e-2 … 1e-6 in roughly half-decade steps (R7 exists so that a
round err_tol=1e-5 does not jump to R8); R9 sits at the reference floor (its label is
reference-limited, not resolved). White-noise (rough) fields run up to 2-3x the indication
at the coarse rungs.

Measured constraints behind the choices:

* **f32 ends at R2.** The element f32 method floor is 1-3e-3; every rung below is f64
  (``requires_x64``) and clamps with a warning on f32 inputs.
* **Near tier (near_theta2=1.0) only on R1/R2.** It is a no-op on the (1,3) rule (R0:
  52.6 vs 53.8 ms at 1e7) and saves 7-23 % on (3,4) at <= 2 % suite error (R1/R2, white
  +10-50 %); from p5 on it FLOORS the error at ~2e-3 regardless of p (cube: R4-old
  9.7e-4 tiered vs 2.4e-4 untiered) — the old f64 rungs p5/th0.7 and p6/th0.5 were both
  tier-floored and are gone.
* **duffy_deg (self-term rule) 8 from R3, 12 from R5.** The default 4 floors the nz=1
  film at 1.9-2.0e-3 (R3 rule at duffy 4 vs 8: 1.98e-3 vs 5.4e-4) and the unitcube
  potential at 4.4e-4. Setup-only cost (per-element constant); t_H identical.
* **Surface degree >= 6 from R4.** The anisotropic nz=2 film (quality proxy 0.14, below
  ``QUALITY_FLOOR`` — excluded from the labels, reported here) floors at 4.4-6.7e-3 on
  every (3,4)/(4,5) rung and drops only with the surface rule: (5,6) 9.0e-4, (5,9)
  3.5e-4, (7,9) 1.2e-4, (10,11) 1.7e-6. Flat cells are a near-quadrature limit, not a
  hard floor.
* **N_max=128, 256 on R3/R5/R6**: not an accuracy knob (errors within 4 %). Re-swept after
  the 2026-10 batching rules (step at 1e7, film/shell/cube, N 128 -> 256): R3 -6/-7/+1 %,
  R5 -5/-6/+1 %, R6 -7/-7/-5 %; R0 +4/+1/+12 %, R4 0/-2/+4 %, and the wide-tile rules
  (7,9)/(10,11) +24..+111 % (R7/R8). The 2.1e6 sweep favours 256 on every rung but R9 -- the
  optimum moves with N. N_max_eval=32 (lane-quantised P2P rows) and s="auto"
  (occupancy-pinned depth) are perf knobs a caller may override without leaving the ladder.
"""

# (tol, requires_x64, params) — ordered fast -> accurate; the resolver picks the
# FIRST rung whose tol <= err_tol (i.e. the cheapest rung indicating at or below
# the requested error).
_T = dict(N_max=128, s="auto", N_max_eval=32)
_T256 = dict(_T, N_max=256)
LADDER = (
    (7.5e-3, False, dict(p=3,  theta=0.77, near_deg=(1, 3),   near_theta2=None, **_T)),
    (3.9e-3, False, dict(p=4,  theta=0.85, near_deg=(3, 4),   near_theta2=1.0,  **_T)),
    (2.1e-3, False, dict(p=4,  theta=0.7,  near_deg=(3, 4),   near_theta2=1.0,  **_T)),
    (4.3e-4, True,  dict(p=6,  theta=0.7,  near_deg=(3, 4),   near_theta2=None, duffy_deg=8,  **_T256)),
    (1.4e-4, True,  dict(p=7,  theta=0.7,  near_deg=(5, 6),   near_theta2=None, duffy_deg=8,  **_T)),
    (6.2e-5, True,  dict(p=8,  theta=0.7,  near_deg=(5, 6),   near_theta2=None, duffy_deg=12, **_T256)),
    (1.2e-5, True,  dict(p=8,  theta=0.55, near_deg=(5, 9),   near_theta2=None, duffy_deg=12, **_T256)),
    (9.2e-6, True,  dict(p=10, theta=0.65, near_deg=(7, 9),   near_theta2=None, duffy_deg=12, **_T)),
    (3.7e-6, True,  dict(p=11, theta=0.6,  near_deg=(7, 9),   near_theta2=None, duffy_deg=12, **_T)),
    (1.1e-6, True,  dict(p=12, theta=0.6,  near_deg=(10, 11), near_theta2=None, duffy_deg=12, **_T)),
)

DEFAULT_TOL = 1e-2          # no err_tol and no knobs: element R0 (speed tier), interior I1

QUALITY_FLOOR = 0.2         # calibration-suite median-quality proxy lower edge


# Interior-quadrature (point-tree) ladder: (tol, requires_x64, params) for element_quadtree_setup /
# element_quadtree_fulljit_setup / element_quadtree_fulljit_driver. tol = suite-mean relative H error,
# same suite, fields and reference as LADDER minus the nz=2 film (2026-10-10, Musica H100, jobs
# 1811943-47, rows dev/thesis/bench_strayfield/plotdata/calib_interior/); t_H = unitcube 2.1e6, H100.
# Re-measured at 0.5.0 (17325bb, 2026-10-18, jobs 1852300-04, rows plotdata/calib_050/): every rung
# within 0.99-1.01 of the 10-10 means; I1 and I4 relabelled by rounding.
#
#   rung  dt   parameters          mean    worst   white   film nz=2   t_H
#   I0    f32  p2 th0.77 N128 d6   3.2e-2  6.2e-2  2.3e-2  2.2e-2     19 ms
#   I1    f32  p3 th0.70 N256 d6   7.9e-3  1.5e-2  6.8e-3  7.8e-3     35
#   I2    f32  p3 th0.60 N256 d6   4.9e-3  1.1e-2  4.0e-3  7.3e-3     44
#   I3    f32  p3 th0.50 N256 d6   2.6e-3  6.1e-3  2.2e-3  7.0e-3     60
#   I4    f32  p4 th0.50 N128 d12  8.0e-4  1.7e-3  8.2e-4  5.9e-3    103
#   I5    f32  p4 th0.40 N128 d12  3.2e-4  7.3e-4  4.3e-4  5.9e-3    152
#   I6    f32  p4 th0.30 N128 d12  1.2e-4  2.3e-4  3.1e-4  5.9e-3    282
#   I7    f64  p5 th0.35 N128 d12  4.8e-5  9.0e-5  1.0e-4  9.1e-4    539
#   I8    f64  p6 th0.35 N256 d12  1.8e-5  4.0e-5  4.1e-5  2.9e-4    929
#
# duffy_deg (self-element rule) 3 floors the nz=1 film at 1.1e-2 on every rung; 6 equals 12 at
# p3 th0.5 on all geometries but costs film1 2.3x at p4 th0.4 and 4x at p5 th0.35, hence 6 up to I3
# and 12 above. In the guarded driver the self-correction is re-formed per geometry change: d12 vs d6
# is +5..15 % forward time and up to 2.6x peak memory (A100, 1e5-5e5 tets). The nz=2 film (quality
# below QUALITY_FLOOR, not in the means) floors at 5.9e-3 for p4 and 9e-4 for p5: point quadrature of
# the near field on flat cells. Dominated and dropped: p2 th0.85, p5 th0.45 (I6's error, 13 % slower),
# p5 th0.25 (worse than I8 on film1, 24 % slower). f32 matches f64 at I6 (mean 1.18e-4); p7, and p6 with
# N_max=128, at th<=0.45 exceed 80 GB at 2.1e6.
_D6, _D12 = dict(duffy_deg=6), dict(duffy_deg=12)
INTERIOR_LADDER = (
    (3.2e-2, False, dict(p=2, theta=0.77, N_max=128, **_D6)),
    (7.9e-3, False, dict(p=3, theta=0.7,  N_max=256, **_D6)),
    (4.9e-3, False, dict(p=3, theta=0.6,  N_max=256, **_D6)),
    (2.6e-3, False, dict(p=3, theta=0.5,  N_max=256, **_D6)),
    (8.0e-4, False, dict(p=4, theta=0.5,  N_max=128, **_D12)),
    (3.2e-4, False, dict(p=4, theta=0.4,  N_max=128, **_D12)),
    (1.2e-4, False, dict(p=4, theta=0.3,  N_max=128, **_D12)),
    (4.8e-5, True,  dict(p=5, theta=0.35, N_max=128, **_D12)),
    (1.8e-5, True,  dict(p=6, theta=0.35, N_max=256, **_D12)),
)

INTERIOR_LEGACY = dict(p=3, theta=0.8, N_max=128, duffy_deg=6)   # fill for explicit knobs without err_tol


def interior_params(err_tol, x64, caller, **knobs):
    r"""Interior-path parameters: the ``INTERIOR_LADDER`` rung for ``err_tol`` (explicit knobs win).

    Without ``err_tol`` the semantics follow the per-element path: no knobs -> the rung for
    :data:`DEFAULT_TOL` (I1); any explicit knob -> the ladder is bypassed and the unset knobs
    take :data:`INTERIOR_LEGACY` (the ladder never fills around explicit parameters).

    Parameters
    ----------
    err_tol : float or None
        Requested error indication (``None``: see above).
    x64 : bool
        Whether the geometry is float64 (f64-only rungs are skipped otherwise, with a warning).
    caller : str
        Name used in the clamp warning.
    **knobs
        ``p``, ``theta``, ``N_max``, ``duffy_deg`` as passed (``None`` = unset).

    Returns
    -------
    dict
        Concrete ``p``, ``theta``, ``N_max``, ``duffy_deg``.
    """
    set_ = {k: v for k, v in knobs.items() if v is not None}
    if err_tol is None and set_:
        return {**INTERIOR_LEGACY, **set_}
    out, info = resolve(DEFAULT_TOL if err_tol is None else float(err_tol), x64, set_,
                        ladder=INTERIOR_LADDER)
    if err_tol is not None and info["clamped"]:
        import warnings
        warnings.warn(f"{caller}: err_tol={err_tol:g} is tighter than the best admissible interior rung "
                      f"({info['tol']:g}" + ("" if x64 else "; float64 inputs unlock the tighter f64 rungs")
                      + ").", stacklevel=3)
    return out


def resolve(err_tol, x64, overrides, ladder = LADDER):
    r"""Pick the cheapest rung indicating ``<= err_tol``; apply explicit overrides.

    Parameters
    ----------
    err_tol : float
        Requested error indication (suite-mean relative H error).
    x64 : bool
        Whether the inputs are float64 (f64-only rungs are skipped otherwise,
        with the clamp reported in the returned info).
    overrides : dict
        Explicitly passed setup kwargs — they always win over rung values.
    ladder : tuple, optional
        :data:`LADDER` (element path, default) or :data:`INTERIOR_LADDER`.

    Returns
    -------
    params : dict
        Setup kwargs for :func:`jaxfmm.fem.element_farfield_setup`.
    info : dict
        ``tol`` (the rung's calibrated indication), ``clamped`` (True when an
        f64-only rung was requested with f32 inputs and the best f32 rung was
        substituted).
    """
    chosen, clamped = None, False
    for tol, needs_x64, params in ladder:
        if tol <= err_tol and not (needs_x64 and not x64):
            chosen = (tol, params)
            break
    if chosen is None:                      # tighter than any admissible rung
        for tol, needs_x64, params in reversed(ladder):
            if not (needs_x64 and not x64):
                chosen = (tol, params)
                clamped = True
                break
    tol, params = chosen
    out = dict(params)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out, dict(tol=tol, clamped=clamped or tol > err_tol)
