r"""
jaxfmm.tune — measured near/far balancing ("autotune") with a persistent wisdom store.

Three-tier tuning philosophy:
  1. the analytic gates built into the setups (predicted_p2p_speedup, depth refinement,
     the ~0.1/K snap rule) — always on, no measurement;
  2. THIS MODULE: probe a small candidate grid at setup time, timing a WARM frozen
     evaluation and MEASURING the accuracy of each candidate against a sampled direct
     reference — so accuracy-changing knobs (the flexible tree's theta MAC!) are traded
     honestly: a candidate must stay within ``rel_slack`` of the reference accuracy;
  3. the wisdom file: results keyed by (model, N-bucket, aspect-bucket, p, dtype, device)
     are persisted, so probing happens once per problem class per machine (FFTW-style).

The probe cost is dominated by first-ever compiles of the candidate shapes; the
persistent compilation cache (jaxfmm.util.enable_compilation_cache) makes re-probes and
wisdom-guided reruns cheap.
"""
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

__all__ = ["tune_kifmm", "tune_uniform", "tune_flex", "wisdom_path",
           "plan", "AutoEvaluator"]

_WISDOM_VERSION = 1


def wisdom_path():
    r"""Path of the persistent wisdom (tuning-results) file, next to the package.

    Returns
    -------
    path : str
        Absolute path of ``.jaxfmm_wisdom.json`` in the directory one level above
        the installed package (the repository root).
    """
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        ".jaxfmm_wisdom.json")


def _wisdom_key(model, N, aspect, p, dtype):
    nb = int(np.round(np.log2(max(N, 1))))                       # N bucket: powers of two
    ab = int(np.clip(np.round(np.log2(max(aspect, 1.0))), 0, 10))  # aspect bucket
    dev = jax.devices()[0].device_kind.replace(" ", "_")
    return f"v{_WISDOM_VERSION}|{model}|N2^{nb}|A2^{ab}|p{p}|{dtype}|{dev}"


def _load_wisdom():
    try:
        with open(wisdom_path()) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_wisdom(w):
    try:
        with open(wisdom_path(), "w") as f:
            json.dump(w, f, indent=1, sort_keys=True)
    except Exception:
        pass


def _aspect(pos):
    ext = np.asarray(pos).max(0) - np.asarray(pos).min(0)
    return float(ext.max() / max(ext.min(), 1e-30))


def _probe(build, candidates, pos, q, rel_slack, n_ref=500, verbose=True, ref=None):
    r"""Generic tuning probe over a candidate grid.

    ``build(cand) -> frozen fmm(q)`` per candidate; time a warm call and measure rel
    error vs a sampled direct sum; pick the fastest candidate whose error stays within
    ``rel_slack`` of the ANCHOR accuracy — the ``ref`` candidate's measured error
    (i.e. what the library default delivers) if given, else the best accuracy seen.  So
    accuracy-changing knobs (theta!) can buy speed only inside the default's accuracy
    class.

    Returns
    -------
    (cand, fz, warm_ms, rel, report)
        Winning candidate dict, its frozen evaluator, warm time, measured relative
        error, and the full probe report.
    """
    from jaxfmm.flex import eval_potential_direct
    rng = np.random.default_rng(0)
    idx = rng.choice(pos.shape[0], min(n_ref, pos.shape[0]), replace=False)
    uref = np.asarray(eval_potential_direct(pos, q, pos[idx]))
    results = []
    for cand in candidates:
        try:
            fz = build(cand)
            u = np.asarray(fz(q))                                  # compile + run
            jax.block_until_ready(fz(q))
            t0 = time.perf_counter(); jax.block_until_ready(fz(q))
            t1 = time.perf_counter(); jax.block_until_ready(fz(q))
            warm_ms = min(time.perf_counter() - t1, t1 - t0) * 1e3
            rel = float(np.linalg.norm(u[idx] - uref) / np.linalg.norm(uref))
            results.append((cand, warm_ms, rel, fz))
            if verbose:
                print(f"  tune: {cand} -> {warm_ms:7.1f} ms  rel {rel:.2e}", flush=True)
        except Exception as e:                                     # candidate infeasible (OOM, ...)
            if verbose:
                print(f"  tune: {cand} -> FAILED ({type(e).__name__})", flush=True)
    if not results:
        raise RuntimeError("autotune: every candidate failed")
    anchor = min(r for _, _, r, _ in results)                      # fallback: best accuracy
    if ref is not None:
        for c, _, r, _ in results:
            if all(c.get(k) == v for k, v in ref.items()):
                anchor = r
                break
    feasible = [r for r in results if r[2] <= max(anchor * rel_slack, anchor + 1e-16)]
    cand, warm_ms, rel, fz = min(feasible, key=lambda r: r[1])
    report = [(c, t, r) for c, t, r, _ in results]
    return cand, fz, warm_ms, rel, report


def tune_kifmm(pos, q, p=3, candidates=None, rel_slack=1.5, force=False, verbose=True):
    r"""Autotune the KIFMM near/far balance (``N_max``) at this cloud.

    Consults/updates the wisdom file.

    Parameters
    ----------
    pos : (N, 3) array
        Point cloud (sources = targets).
    q : (N,) array
        Charges used by the probe (warm timing AND measured accuracy vs a sampled
        direct sum).
    p : int
        Expansion order.
    candidates : list of dict, optional
        Candidate grid, each dict holding ``N_max``; ``None`` probes
        ``N_max in (32, 64, 128)``.
    rel_slack : float
        Accuracy slack: a candidate is feasible only while its measured relative
        error stays within ``rel_slack`` of the reference candidate's
        (``N_max=64``, the library default).
    force : bool
        Re-probe even when the wisdom file already holds this problem class.
    verbose : bool
        Print per-candidate probe results.

    Returns
    -------
    (params, frozen_fmm, report)
        Winning parameter dict, frozen evaluator, and the probe report.
    """
    from jaxfmm import kifmm
    pos = jnp.asarray(pos)
    key = _wisdom_key("kifmm", pos.shape[0], _aspect(pos), p, str(pos.dtype))
    w = _load_wisdom()
    if not force and key in w:
        params = w[key]
        s = kifmm.setup(pos, p=p, N_max=params["N_max"])
        return params, kifmm.compile_evaluator(s), [("wisdom", params)]
    candidates = candidates or [{"N_max": n} for n in (32, 64, 128)]

    def build(cand):
        s = kifmm.setup(pos, p=p, N_max=cand["N_max"])
        return kifmm.compile_evaluator(s)

    cand, fz, warm_ms, rel, report = _probe(build, candidates, pos, q, rel_slack, verbose=verbose,
                                            ref={"N_max": 64})
    w[key] = {**cand, "warm_ms": warm_ms, "rel": rel}
    _save_wisdom(w)
    return w[key], fz, report


def tune_uniform(pos, q, p=3, candidates=None, rel_slack=1.5, force=False, verbose=True):
    r"""Autotune the cubic precomputed-s.h. model (``N_max``).

    Parameters
    ----------
    pos : (N, 3) array
        Point cloud (sources = targets).
    q : (N,) array
        Charges used by the probe (warm timing AND measured accuracy vs a sampled
        direct sum).
    p : int
        Expansion order.
    candidates : list of dict, optional
        Candidate grid, each dict holding ``N_max``; ``None`` probes
        ``N_max in (32, 64, 128)``.
    rel_slack : float
        Accuracy slack: a candidate is feasible only while its measured relative
        error stays within ``rel_slack`` of the reference candidate's
        (``N_max=64``, the library default).
    force : bool
        Re-probe even when the wisdom file already holds this problem class.
    verbose : bool
        Print per-candidate probe results.

    Returns
    -------
    (params, frozen_fmm, report)
        Winning parameter dict, frozen evaluator, and the probe report.
    """
    import jaxfmm.uniform as uf
    from jaxfmm.trees import (build_fmm_tree_auto, assign_particles, build_interaction_lists,
                              build_ulist_adaptive)
    from jaxfmm.near import near_pairs, auto_mem_limit
    pos = jnp.asarray(pos)
    key = _wisdom_key("uniform", pos.shape[0], _aspect(pos), p, str(pos.dtype))
    w = _load_wisdom()

    def build(cand):
        out, mb, md, maxleaf = build_fmm_tree_auto(pos, N_max=cand["N_max"], refine_factor=2)
        (m_id, p_id, lvls, counts, is_leaf, sort_idx, _, de, pmin, _, _) = out
        _, pil = assign_particles(sort_idx, m_id, lvls, counts, is_leaf, md, N_max=maxleaf)
        _, V = build_interaction_lists(m_id, lvls, is_leaf, mb, md)
        U, _ = build_ulist_adaptive(m_id, lvls, is_leaf, mb, md)
        dc = np.asarray(near_pairs(U, is_leaf, lvls))
        dc = jnp.asarray(dc[(dc[:, 0] >= 0) & (dc[:, 1] >= 0)].astype(np.int64))
        ops = uf.precompute_operators(p, float(de), md)
        lr = uf.m2l_level_ranges(lvls, md)
        ml = auto_mem_limit(maxleaf, pos.dtype.itemsize)
        return jax.jit(lambda qq: uf.uniform_eval(pos, qq, m_id, p_id, lvls, is_leaf, pil, V,
                                                  dc, de, pmin, ops, md, level_ranges=lr,
                                                  p2p_mixed=True, mem_limit=ml))

    if not force and key in w:
        params = w[key]
        return params, build(params), [("wisdom", params)]
    candidates = candidates or [{"N_max": n} for n in (32, 64, 128)]
    cand, fz, warm_ms, rel, report = _probe(build, candidates, pos, q, rel_slack, verbose=verbose,
                                            ref={"N_max": 64})
    w[key] = {**cand, "warm_ms": warm_ms, "rel": rel}
    _save_wisdom(w)
    return w[key], fz, report


def tune_flex(pos, q, p=3, snap=True, candidates=None, rel_slack=1.5, force=False,
              verbose=True):
    r"""Autotune the flexible tree.

    Tunes ``N_max``, the theta MAC (an ACCURACY parameter: the measured-rel constraint
    keeps candidates within ``rel_slack`` of the best accuracy, so theta can only buy
    speed inside the same accuracy class) AND the split count ``s`` (2^s children per
    box — its main effect is bringing the realised leaf occupancy close to ``N_max``).

    STAGED probe: ``s`` in {2,3,4} at the reference (theta, N_max) first (occupancy is
    ~orthogonal to the MAC), then the theta x N_max grid at the winning ``s`` — 12
    probes instead of the full 27.  ``snap`` adds the theta-snapped grouped M2L (K by
    the 0.1/K rule).

    Parameters
    ----------
    pos : (N, 3) array
        Point cloud (sources = targets).
    q : (N,) array
        Charges used by the probe (warm timing AND measured accuracy vs a sampled
        direct sum).
    p : int
        Expansion order.
    snap : bool
        Add the theta-snapped grouped M2L (group count ``K = 256`` for ``p <= 4``,
        else ``1024``); snapped and unsnapped results are stored under separate
        wisdom keys.
    candidates : list of dict, optional
        Candidate grid, each dict holding ``N_max``, ``theta`` and optionally
        ``s``; ``None`` runs the staged default probe described above.
    rel_slack : float
        Accuracy slack: a candidate is feasible only while its measured relative
        error stays within ``rel_slack`` of the reference candidate's
        (``N_max=64, theta=0.8``, the library default).
    force : bool
        Re-probe even when the wisdom file already holds this problem class.
    verbose : bool
        Print per-candidate probe results.

    Returns
    -------
    (params, frozen_fmm, report)
        Winning parameter dict, frozen evaluator, and the probe report.
    """
    import jaxfmm.flex as f
    from jaxfmm.trees import gen_hierarchy
    from jaxfmm.kernels.laplace import m2l_theta_setup
    from jaxfmm.near import auto_mem_limit
    pos = jnp.asarray(pos)
    model = "snap" if snap else "flex"
    key = _wisdom_key(model, pos.shape[0], _aspect(pos), p, str(pos.dtype))
    w = _load_wisdom()
    K = 256 if p <= 4 else 1024

    def build(cand):
        hier = gen_hierarchy(pos, eval_pts=pos, p=p, N_max=cand["N_max"],
                             theta=cand["theta"], s=cand.get("s", 3), jit_cnct=True)
        hier.pop("mem_limit", None)
        nb_e = int(hier["eval_boxcenters"].shape[0]); nb_s = int(hier["boxcenters"].shape[0])
        for k2 in ("mpl_cnct", "dir_cnct"):                       # frozen: compact the lists
            cn = np.asarray(hier[k2])
            okm = (cn[:, 0] >= 0) & (cn[:, 1] >= 0) & (cn[:, 0] < nb_e) & (cn[:, 1] < nb_s)
            hier[k2] = jnp.asarray(cn[okm])
        if snap:
            hier["m2l_grp"] = m2l_theta_setup(hier["mpl_cnct"], hier["boxcenters"],
                                              hier["eval_boxcenters"], hier["unqs"],
                                              hier["invs"], K=K)
        ml = auto_mem_limit(cand["N_max"], pos.dtype.itemsize)   # leaf width = the working-set scale
        return jax.jit(lambda qq: f.eval_potential(qq, **hier, mem_limit=ml))

    if not force and key in w:
        params = w[key]
        return params, build(params), [("wisdom", params)]
    if candidates is None:                                  # stage 1: split count at reference
        s_cands = [{"N_max": 64, "theta": 0.8, "s": si} for si in (2, 3, 4, "auto")]
        s_best, _, _, _, rep1 = _probe(build, s_cands, pos, q, rel_slack, verbose=verbose,
                                       ref={"s": 3})
        candidates = [{"N_max": n, "theta": t, "s": s_best["s"]}   # stage 2: MAC x leaf grid
                      for n in (32, 64, 128) for t in (0.6, 0.8, 1.0)]
    else:
        rep1 = []
    cand, fz, warm_ms, rel, report = _probe(build, candidates, pos, q, rel_slack, verbose=verbose,
                                            ref={"N_max": 64, "theta": 0.8})
    w[key] = {**cand, "warm_ms": warm_ms, "rel": rel}
    _save_wisdom(w)
    return w[key], fz, rep1 + report


# ────────────────────────────────────────────────────────────────────────────────────────
# Tolerance-driven front-end: the user chooses ONE number — err_tol — and the library
# picks/verifies the perf/accuracy knobs. Expert knobs remain as overrides.
# ────────────────────────────────────────────────────────────────────────────────────────

_TOL_VERSION = 2                    # bump to stale wisdom recorded before behaviour changes
_F32_TOL_FLOOR = 1.5e-3            # measured f32 certification floor; below needs f64
_F32_DEFAULT_TOL = 5e-3            # plan()'s x64-off default, safely above the floor
_MIXED_P2P_TOL = 1e-6              # below: exact f64 P2P (the mixed P2P floors at ~3e-7)


def _tol_key(engine, N, aspect, err_tol, dtype, field):
    nb = int(np.round(np.log2(max(N, 1))))
    ab = int(np.clip(np.round(np.log2(max(_np_aspect_safe(aspect), 1.0))), 0, 10))
    tb = int(np.floor(np.log10(err_tol)))                   # tolerance decade bucket
    dev = jax.devices()[0].device_kind.replace(" ", "_")
    return f"tol{_TOL_VERSION}|{engine}|N2^{nb}|A2^{ab}|tol1e{tb}|{dtype}|f{int(field)}|{dev}"


def _np_aspect_safe(a):
    return a if np.isscalar(a) else float(a)


def _referee(pts, chrgs, sel, field, chunk=4_000_000):
    r"""Sampled ground truth by chunked direct summation.

    f64 accumulation of f32 chunk partials: the ACTUAL error of the actual
    configuration on the actual data — the certification primitive of the
    tolerance-driven API (kernel-independent by design)."""
    import jax.numpy as jnp

    @jax.jit
    def one(tc, src, q):
        d = tc[:, None, :] - src[None, :, :]
        r2 = jnp.sum(d * d, axis=-1)
        if field:
            w = jnp.where(r2 > 1e-12, q[None, :] * r2 ** -1.5, 0.0)
            return jnp.einsum("ts,tsd->td", w, d)
        return jnp.where(r2 > 1e-12, q[None, :] * r2 ** -0.5, 0.0).sum(axis=1)

    trg = pts[sel]
    N = pts.shape[0]
    acc = None
    for lo in range(0, N, chunk):
        part = np.asarray(one(trg, pts[lo:lo + chunk], chrgs[lo:lo + chunk]), np.float64)
        acc = part if acc is None else acc + part
    return acc / (4 * np.pi)


class AutoEvaluator:
    r"""
    Tolerance-driven frozen evaluator (from :func:`plan`): call it like ``H = fmm(chrgs)``.

    The FIRST call certifies against a sampled direct referee on the ACTUAL charges:
    if the measured error exceeds ``err_tol``, the configuration steps along the
    measured accuracy frontier (p up, theta fixed at the tuned 0.85) and re-certifies —
    then the certified configuration is written to the wisdom store, so later plans of
    the same problem class skip the search. ``.report`` holds the certification history;
    ``.recertify(chrgs)`` re-runs the check (e.g. every k steps of a dynamics loop whose
    charge distribution drifts).
    """

    def __init__(self, pts, err_tol, field, engine, n_ref, seed, mem_limit, verbose,
                 expert, verify):
        self.pts = pts
        self.err_tol, self.field, self.engine = float(err_tol), bool(field), engine
        self.n_ref, self.seed = int(n_ref), int(seed)
        self.mem_limit, self.verbose, self.verify = mem_limit, verbose, verify
        self.expert = dict(expert)
        self.report = []
        self._fn = None
        self.params = None
        x64 = jax.config.jax_enable_x64
        if not x64 and self.err_tol < _F32_TOL_FLOOR:
            raise ValueError(
                f"err_tol={err_tol:g} is below the measured float32 accuracy floor "
                f"(~{_F32_TOL_FLOOR:g}). Enable double precision (JAX_ENABLE_X64=1 / "
                f"jax.config.update('jax_enable_x64', True)) for tighter tolerances.")
        # frontier descent ladder: p rises within a theta tier, theta drops between
        # tiers (the theta-MAC term saturates pure p-stepping for tight tolerances)
        # (x64 tail measured on 2e4 random pts, exact P2P: (10,.4) 2.0e-8, (12,.4) 1.5e-9,
        # (12,.3) 2.8e-11, (16,.3) 9.0e-14 — ends at the f64 floor)
        self._ladder = ([(3, .85), (4, .85), (5, .85), (5, .7), (6, .7), (7, .7),
                         (7, .55), (8, .55), (8, .4), (10, .4), (12, .4), (12, .3),
                         (16, .3)] if x64 else
                        [(3, .85), (4, .85), (5, .85), (5, .7)])
        self._step = 0
        key = _tol_key(engine, pts.shape[0], _aspect(pts), self.err_tol,
                       str(pts.dtype), self.field)
        self._key = key
        w = _load_wisdom()
        hit = w.get(key)
        if self.expert:                                    # expert override: no search
            self.params = {"p": 3, "theta": 0.85, "N_max": 128, **self.expert}
            self._search = False
        elif hit:
            self.params = dict(hit)
            self._search = False                           # wisdom-certified previously
            if self.verbose:
                print(f"plan: wisdom hit {key} -> {self.params}", flush=True)
        else:
            self.params = {"p": 3, "theta": 0.85, "N_max": 128}
            self._search = True

    # -- engine builders -------------------------------------------------------------
    def _build(self):
        from jaxfmm.util import split_static, join_static
        if self.engine == "flex":
            from jaxfmm.trees import gen_hierarchy
            from jaxfmm.flex import eval_potential
            h = gen_hierarchy(self.pts, N_max=self.params["N_max"], s="auto",
                              p=self.params["p"], theta=self.params["theta"],
                              jit_cnct=True)
            h = dict(h)
            if h["eval_pts"] is not h["pts"]:
                h["eval_pts"] = h["pts"]
            if self.mem_limit is not None:
                h["mem_limit"] = self.mem_limit
            skel, arrs = split_static(h)
            field = self.field
            p2p_mixed = False if self.err_tol < _MIXED_P2P_TOL else None   # None: flex default

            @jax.jit
            def ff(arrs_, q):
                return eval_potential(q, **join_static(skel, arrs_), field=field,
                                      p2p_mixed=p2p_mixed)

            return lambda q: ff(arrs, q)
        elif self.engine == "kifmm":
            from jaxfmm import kifmm
            kw = {} if self.mem_limit is None else {"mem_limit": self.mem_limit}
            if self.err_tol < _MIXED_P2P_TOL:
                kw["p2p_mixed"] = False                 # same ~3e-7 mixed-P2P floor as flex
            s = kifmm.setup(self.pts, p=self.params["p"],
                            N_max=self.params.get("N_max", 512), **kw)
            return kifmm.compile_evaluator(s, field=self.field)
        raise ValueError(f"unknown engine {self.engine!r} (flex | kifmm)")

    # -- certification ---------------------------------------------------------------
    def _certify(self, chrgs, out):
        rng = np.random.default_rng(self.seed)
        sel = np.sort(rng.choice(self.pts.shape[0],
                                 min(self.n_ref, self.pts.shape[0]), replace=False))
        ref = _referee(self.pts, chrgs, jax.numpy.asarray(sel), self.field)
        got = np.asarray(out)[sel]
        rel = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-300))
        self.report.append((dict(self.params), rel))
        if self.verbose:
            print(f"plan: certify {self.params} -> rel {rel:.3e} "
                  f"(tol {self.err_tol:g})", flush=True)
        return rel

    def __call__(self, chrgs):
        while True:
            if self._fn is None:
                self._fn = self._build()
            out = self._fn(chrgs)
            jax.block_until_ready(out)
            if self._fn is not None and not self.verify:
                return out
            if getattr(self, "_certified", False):
                return out
            rel = self._certify(chrgs, out)
            if rel <= self.err_tol or not self._search:
                if rel > self.err_tol and self.verbose:
                    print(f"plan: WARNING measured rel {rel:.2e} misses err_tol "
                          f"{self.err_tol:g} (expert/frozen config kept)", flush=True)
                self._certified = True
                if self._search or not self.expert:
                    w = _load_wisdom()
                    w[self._key] = dict(self.params)
                    _save_wisdom(w)
                return out
            if self._step + 1 >= len(self._ladder):       # frontier exhausted
                best_cfg, best = min(self.report, key=lambda t: t[1])
                raise RuntimeError(
                    f"plan: cannot certify err_tol={self.err_tol:g}: best measured "
                    f"{best:.2e} at p={best_cfg['p']}, theta={best_cfg['theta']}, and the "
                    f"search ladder ends at (p, theta) = {self._ladder[-1]}"
                    + ("" if jax.config.jax_enable_x64 else
                       " (float32 ladder; enable x64 for tighter tolerances)") + ".")
            self._step += 1
            pp, tt = self._ladder[self._step]
            self.params = dict(self.params, p=pp, theta=tt)
            self._fn = None                                # rebuild at the next rung

    def recertify(self, chrgs):
        r"""Re-run the referee on current charges (dynamics drift check); returns rel."""
        out = self._fn(chrgs)
        jax.block_until_ready(out)
        return self._certify(chrgs, out)


def plan(pts, err_tol=None, field=False, engine="flex", n_ref=256, seed=0,
         mem_limit=None, verbose=True, verify=True, **expert):
    r"""
    ONE-KNOB frozen evaluator: ``fmm = plan(pts, err_tol=1e-3); H = fmm(chrgs)``.

    Parameters
    ----------
    pts : (N, 3) array
        Point cloud.
    err_tol : float, optional
        Target GLOBAL relative L2 error of the evaluation (certified on the first
        call against a sampled direct summation of the actual charges — the algebraic
        error of the fast method, NOT any upstream discretization error).  Tolerances
        below the float32 floor (~1.5e-3) require x64 mode and raise otherwise.
        Default ``None`` resolves by precision mode: ``1e-3`` with x64 enabled,
        ``5e-3`` without (safely above the float32 floor).
    field : bool
        Evaluate the field (negative gradient) instead of the potential.
    engine : {"flex", "kifmm"}
        Engine choice.
    n_ref : int
        Referee sample size.
    seed : int
        Referee sampling seed.
    mem_limit : float, optional
        Override the automatic batch budget.
    verbose : bool
        Print certification progress.
    verify : bool
        ``False`` skips certification entirely (trust wisdom/defaults).
    **expert
        Expert overrides (``p=; theta=; N_max=``): skip the search but are still
        measured (a warning is printed if they miss the tolerance).

    Returns
    -------
    AutoEvaluator
        Frozen callable evaluator; all perf/accuracy knobs are chosen internally
        (wisdom-store warm start, measured frontier stepping on a miss) and remembered
        per problem class.
    """
    pts = jax.numpy.asarray(pts)
    if err_tol is None:
        err_tol = 1e-3 if jax.config.jax_enable_x64 else _F32_DEFAULT_TOL
    return AutoEvaluator(pts, err_tol, field, engine, n_ref, seed, mem_limit,
                         verbose, expert, verify)


