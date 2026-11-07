r"""
Guaranteed-correct moving-geometry FMM drivers.

The fully-jitted shape-differentiable forwards trade recompiles for FROZEN buffer caps
(connectivity ``W``/``n_mpl``/``n_dir``, theta-snap class tiers, the frozen uniform-grid
topology). Frozen caps cannot raise inside jit -- on overflow they silently drop interaction
pairs: plausible-looking, wrong physics. :class:`GuardedShapeEvaluator` makes that impossible:

* every call retrieves the REALISED occupancies computed inside the jitted forward anyway
  (``return_counts=True`` -- a few traced scalars, zero cost, no recompile);
* the result is accepted only if every count fits its frozen cap (and the frozen level
  structure still holds); otherwise the result is REJECTED, the setup is rebuilt at the
  current geometry (caps re-fitted with a fractional headroom onto a geometric cap ladder and
  never below the previous caps, so later rebuilds mostly land on already-compiled shapes),
  and the step re-runs;
* hence every value (and gradient) a driver returns is certified adequate -- the steady state
  keeps the fulljit promise of zero recompiles, and a rebuild happens exactly when the
  alternative would have been silent truncation.

The other variant families need no driver: the eager per-call setups rebuild their lists per
geometry (``gen_connectivity_jit`` raises on overflow, ``AdaptiveConnectivity`` grows), and the
frozen-geometry compiled evaluators (charges-only closures) bake lists that are EXACT for the
geometry they were compacted from.

Concrete factories: :func:`element_quadtree_fulljit_driver` (capped flexible quad-tree,
optionally theta-snapped) and :func:`element_uniform_diff_driver` (frozen uniform-grid
topology, staleness detected as particles leaving their assigned Morton cells).
"""
import math
import statistics
import time
from collections import OrderedDict

import jax
import jax.numpy as jnp

__all__ = ["GuardedShapeEvaluator", "element_quadtree_fulljit_driver",
           "element_uniform_diff_driver", "kifmm_fulljit_driver"]


def _cap_max(a, b):
    r"""Elementwise maximum of two cap dicts (``None`` = no floor)."""
    return dict(b) if a is None else {k: max(a.get(k, 0), v) for k, v in b.items()}


def _covers(a, b):
    r"""Every cap of ``a`` holds the corresponding cap of ``b``."""
    return all(a[k] >= v for k, v in b.items())


def _log2_size(caps, ref):
    r"""Mean log2 cap ratio ``caps / ref`` -- the padded-size proxy of the step-time model."""
    return sum(math.log2(caps[k] / ref[k]) for k in ref) / len(ref)


# Compiled cap tuples, PROCESS-wide like the jit cache that holds their executables:
# (cache group, call mode) -> OrderedDict(static key -> {"caps": dict, "t": [settled step s]}).
_KNOWN = {}
KNOWN_PER_GROUP = 8          # LRU bound per (group, mode)
STEP_PER_DOUBLING = 0.31     # step-time model for unmeasured caps: 2x caps = +31 % (§7.47)
MAX_SWITCH_RATIO = 4.0       # never switch onto a known tuple with any cap > 4x the tight fit


class GuardedShapeEvaluator:
    r"""
    Watertight wrapper around a frozen-caps differentiable forward.

    Call the driver like a function: ``out = driver(verts, *charge_args)`` -- certified,
    auto-rebuilding; ``charge_args`` are traced extras forwarded to the forward's
    ``charge_fn`` (e.g. a per-call magnetisation).
    For optimisation loops use :meth:`value_and_grad`, which threads the counts through
    ``jax.value_and_grad`` (``has_aux``) so the gradient step carries its own certificate.
    ``driver.raw`` exposes the current ``verts -> (out, counts)`` -- results obtained that
    way are certified only if YOU check the counts with ``driver.ok_fn(driver.setup, counts)``.

    Parameters
    ----------
    verts0 : array
        Reference geometry the first setup is probed at.
    build_fn : callable
        ``build_fn(verts) -> setup`` -- eager setup with caps fitted at ``verts``.
    raw_factory : callable
        ``raw_factory(setup) -> (verts -> (out, counts))`` -- the READY-TO-CALL forward
        returning the result and its realised in-jit occupancies.  The factory owns the
        jit strategy: route through a module-level jitted function of (setup arrays,
        verts) with the static bits as ``static_argnames``, so rebuilds whose buffer
        shapes repeat cost NO retrace/recompile.  The driver deliberately does NOT wrap
        the callable in ``jax.jit`` -- an outer jit would inline and re-trace the inner
        one, defeating exactly that reuse.
    ok_fn : callable
        ``ok_fn(setup, counts) -> (ok, info)`` -- host-side cap verdict.
    max_attempts : int
        Rebuild-and-retry budget per call (a rebuild fits the caps at the offending
        geometry, so the retry passes unless the caller mutates ``verts`` concurrently).
    vg_builder : callable, optional
        ``vg_builder(setup, loss_fn, geom_fn, charge_args_fn) -> (x, *args) ->
        ((loss, counts), grad)``
        for a retrace-free gradient path (same module-level-jit idea); without it
        :meth:`value_and_grad` falls back to a per-rebuild
        ``jax.jit(jax.value_and_grad(...))`` closure (correct, but re-traces after
        every rebuild).
    oversized_fn : callable, optional
        ``oversized_fn(setup, info) -> bool`` -- reports whether a refit would shrink the
        caps by a worthwhile step (enables lazy downsizing); with ``caps_fn`` it is called as
        ``oversized_fn(setup, info, cap_floor)`` with the ``size_from`` floor.
    downsize_after : int
        Minimum number of consecutive oversized certified calls before a lazy downsize
        (anti-flicker); with ``key_fn``/``refit_fn`` the downsize must also pay for itself
        (Notes).
    caps_fn : callable, optional
        ``caps_fn(setup) -> dict`` of the frozen caps. Enables cap floors: ``build_fn`` is then
        called as ``build_fn(verts, cap_floor)`` (``cap_floor`` a dict or ``None``); a
        rejection rebuild passes the current caps (grow-only, so no other cap shrinks into
        the next rejection) and only a lazy downsize refits without a floor.
    size_from : array or sequence of arrays, optional
        Extra geometries the initial caps must also hold (elementwise maximum), e.g. the
        heaviest pose of a scan (an MFM tip over the sample centre); needs ``caps_fn``.
        These caps are a PERMANENT floor: lazy downsizes never go below them.
    key_fn : callable, optional
        ``key_fn(setup) -> hashable`` -- the static part that keys the compiled executable
        (shapes, caps, level structure). Enables compile-aware rebuilds (Notes); needs
        ``caps_fn``.
    refit_fn : callable, optional
        ``refit_fn(setup, info, cap_floor) -> dict`` -- the caps a refit at the realised
        counts ``info`` would choose (prices lazy downsizes).
    cache_group : hashable, optional
        Identity of the compiled forward family (factory, static config, mesh shapes,
        ``charge_fn``): drivers sharing it share the record of compiled cap tuples, as they
        share the jit cache. ``None``: private to this driver.

    Notes
    -----
    **Compile-aware rebuilds** (``key_fn``). The driver remembers the cap tuples compiled in
    this process (LRU, :data:`KNOWN_PER_GROUP` per group and call mode), with their measured
    step times, and the cost of a fresh compile (first call minus settled step) and of an
    eager probe (``build_fn``). A rebuild onto a remembered tuple is a *switch* (probe only),
    otherwise a *compile*. Both directions are priced like ski rental -- pay the one-off cost
    once the accumulated per-step loss of not paying exceeds it -- which bounds either choice
    at ~2x the hindsight optimum:

    * rejection: the tight grow-only fit if it is known; else the smallest remembered tuple
      that holds it (no compile), while the excess step time over the tight fit accumulates
      until it pays for compiling the tight fit; else compile the tight fit;
    * lazy downsize (after ``downsize_after`` oversized calls): onto the smallest remembered
      tuple that holds the refit (cost: a probe), or onto the refit itself (cost: probe +
      compile), once the accumulated saving pays that cost.

    Unmeasured step times come from the padded-size model (+:data:`STEP_PER_DOUBLING` per
    doubling of the caps). ``events`` lists ``(call, kind)`` per rebuild, kind in
    ``{"compile", "switch"}`` (``+"-down"`` for lazy downsizes).
    """

    def __init__(self, verts0, build_fn, raw_factory, ok_fn, max_attempts=3, vg_builder=None,
                 oversized_fn=None, downsize_after=32, caps_fn=None, size_from=None, key_fn=None,
                 refit_fn=None, cache_group=None):
        self.build_fn, self.raw_factory, self.ok_fn = build_fn, raw_factory, ok_fn
        self.vg_builder = vg_builder
        self.oversized_fn, self.refit_fn = oversized_fn, refit_fn
        self.key_fn = key_fn if caps_fn is not None else None
        self.cache_group = object() if cache_group is None else cache_group   # unique, never reused
        self.downsize_after = int(downsize_after) if downsize_after else 0
        self.max_attempts = int(max_attempts)
        self.n_rebuilds = self.n_downsizes = self.n_compiles = self.n_switches = 0
        self.events, self.n_calls = [], 0
        self._low_streak, self._saving = 0, 0.0
        self._downsize_stuck = False
        self._grow = None                                    # renting a fat tuple: target + excess
        self._pre_down = None                                # caps before the last lazy downsize
        self._pending = None                                 # first-call time of a fresh compile
        self._mode = "fwd"
        self.compile_s = self.probe_s = None
        self._clock = time.perf_counter
        self.generation = 0
        self.caps_fn = caps_fn
        floor = self.floor0 = None
        if size_from is not None:
            if caps_fn is None:
                raise ValueError("size_from needs caps_fn (a factory with cap floors)")
            poses = [size_from] if getattr(size_from, "ndim", 0) == 2 else list(size_from)
            for v in poses:
                floor = _cap_max(floor, caps_fn(build_fn(v, None)))
        self.floor0 = floor                                  # size_from caps: downsizes keep them
        self.setup = self._build(verts0, floor)
        self.raw = raw_factory(self.setup)

    # -- bookkeeping -------------------------------------------------------------------
    def _build(self, verts, floor):
        t0 = self._clock()
        s = self.build_fn(verts) if self.caps_fn is None else self.build_fn(verts, floor)
        self.probe_s = self._clock() - t0
        return s

    def _known(self):
        return _KNOWN.setdefault((self.cache_group, self._mode), OrderedDict())

    def _t(self, caps):
        r"""Step time at ``caps`` in the current mode: measured median, else the size model
        scaled from the current tuple's measurement; ``None`` before any measurement."""
        known, cur = self._known(), self.caps_fn(self.setup)
        e = next((e for e in known.values() if e["caps"] == caps and len(e["t"]) >= 3), None)
        if e is not None:
            return statistics.median(e["t"])
        e = known.get(self.key_fn(self.setup))
        if e is None or len(e["t"]) < 3:
            return None
        return statistics.median(e["t"]) * max(0.5, 1 + STEP_PER_DOUBLING * _log2_size(caps, cur))

    def _record(self, dt):
        r"""Book one certified call of ``dt`` seconds against the current tuple."""
        self.n_calls += 1
        if self.key_fn is None:
            return
        known, key = self._known(), self.key_fn(self.setup)
        if key not in known:                                 # first call on it: trace + compile
            known[key] = {"caps": self.caps_fn(self.setup), "t": []}
            self._pending = dt
            while len(known) > KNOWN_PER_GROUP:
                known.popitem(last=False)
            return
        known.move_to_end(key)
        t = known[key]["t"]
        t.append(dt)
        del t[:-32]
        if self._pending is not None and len(t) >= 3:
            self.compile_s, self._pending = max(self._pending - statistics.median(t), 0.0), None

    def _install(self, setup, kind):
        known = self._known() if self.key_fn is not None else {}
        compiled = self.key_fn is not None and self.key_fn(setup) not in known
        kind = ("compile" if compiled or self.key_fn is None else "switch") + kind
        self.setup = setup
        self.raw = self.raw_factory(setup)
        self.generation += 1
        self.n_rebuilds += 1
        self.n_compiles += kind.startswith("compile")
        self.n_switches += kind.startswith("switch")
        self.events.append((self.n_calls, kind))
        self._low_streak, self._saving = 0, 0.0

    def _rebuild(self, verts, grow=True):
        r"""Rejection rebuild (``grow``) -- or the legacy unpriced downsize (``grow=False``)."""
        if not grow:
            self._grow = None
            self._install(self._build(verts, self.floor0), "")
            return
        cur = self.caps_fn(self.setup) if self.caps_fn is not None else None
        tight = self._build(verts, cur if cur is not None else self.floor0)
        self._grow = None
        if self.key_fn is None:
            self._install(tight, "")
            return
        need, pre, known = self.caps_fn(tight), self._pre_down, self._known()
        self._pre_down = None
        if pre is not None and _covers(pre, need) and any(e["caps"] == pre for e in known.values()):
            cand = self._build(verts, pre)                   # first regrow after a downsize: back
            if self.key_fn(cand) in known:                   # onto the caps that held the whole
                self._install(cand, "")                      # previous stretch, no ladder climb
                return
        if self.key_fn(tight) not in known:
            fits = [e["caps"] for e in known.values()
                    if _covers(e["caps"], need) and all(e["caps"][k] <= MAX_SWITCH_RATIO * need[k] for k in need)]
            if fits:
                big = min(fits, key=lambda c: _log2_size(c, need))
                cand = self._build(verts, big)
                if self.key_fn(cand) in known:               # level structure may still differ
                    self._install(cand, "")
                    self._grow = {"target": need, "excess": 0.0}
                    return
        self._install(tight, "")

    # -- per certified call ------------------------------------------------------------
    def _certified(self, info, verts, dt):
        self._record(dt)
        if self._grow is not None and self.compile_s is not None:
            t_now, t_tgt = self._t(self.caps_fn(self.setup)), self._t(self._grow["target"])
            if t_now is not None and t_tgt is not None:
                self._grow["excess"] += max(t_now - t_tgt, 0.0)
                if self._grow["excess"] >= self.compile_s + self.probe_s:
                    target = _cap_max(self.floor0, self._grow["target"])
                    self._grow = None
                    s = self._build(verts, target)
                    if self.key_fn(s) != self.key_fn(self.setup):
                        self._install(s, "")
                    return
        self._maybe_downsize(info, verts)

    def _maybe_downsize(self, info, verts):
        r"""Lazily downsize fat caps left behind by a transient.

        A transient through a pathological configuration leaves fat caps behind (padded
        compute every call).  When the factory's ``oversized_fn`` reports that a refit
        would shrink some buffer by a worthwhile step (the fulljit factories: >= two cap-ladder
        bins, and grow none) for ``downsize_after`` CONSECUTIVE certified calls -- and, when
        compile-aware, once the accumulated saving pays the switch or compile (class Notes) --
        rebuild at the current geometry, never below the ``size_from`` floor."""
        if self.oversized_fn is None or not self.downsize_after or self._downsize_stuck:
            return
        if not self._oversized(info):
            self._low_streak, self._saving = 0, 0.0
            return
        self._low_streak += 1
        if self.key_fn is None or self.refit_fn is None:     # unpriced: the streak alone
            if self._low_streak >= self.downsize_after:
                self._downsize(verts, self.floor0, info)
            return
        cur, want = self.caps_fn(self.setup), self.refit_fn(self.setup, info, self.floor0)
        smaller = [e["caps"] for e in self._known().values()
                   if _covers(e["caps"], want) and _covers(cur, e["caps"]) and e["caps"] != cur]
        target = min(smaller, key=lambda c: _log2_size(c, want)) if smaller else want
        t_now, t_tgt = self._t(cur), self._t(target)
        if t_now is None or t_tgt is None or (not smaller and self.compile_s is None):
            return
        self._saving += max(t_now - t_tgt, 0.0)
        # a downsize is a round trip: this probe, and the rejected step + probe of the regrow
        cost = 2 * self.probe_s + t_now + (0.0 if smaller else self.compile_s)
        if self._low_streak >= self.downsize_after and self._saving >= cost:
            self._downsize(verts, target if smaller else self.floor0, info)

    def _downsize(self, verts, floor, info):
        self._grow = None
        before = self._pre_down = self.caps_fn(self.setup) if self.caps_fn is not None else None
        self._install(self._build(verts, floor), "-down")
        self.n_downsizes += 1
        if self._oversized(info) and (before is None or self.caps_fn(self.setup) == before):
            # the rebuild reproduced the same caps: the gap is probe-vs-runtime
            # count drift (the capped traversal truncates candidates in a different
            # order than the auto-buffer probe), not real slack -- stop trying
            # until a rejection rebuild signals an actual regime change
            self._downsize_stuck = True

    def _oversized(self, info):
        if self.caps_fn is None:
            return self.oversized_fn(self.setup, info)
        return self.oversized_fn(self.setup, info, self.floor0)

    def __call__(self, verts, *charge_args):
        self._mode = "fwd"
        for _ in range(self.max_attempts):
            t0 = self._clock()
            out, counts = self.raw(verts, *charge_args)
            ok, info = self.ok_fn(self.setup, counts)        # host sync on the counts
            if ok:
                self._certified(info, verts, self._clock() - t0)
                return out
            self._rebuild(verts)
            self._downsize_stuck = False
        raise RuntimeError(f"buffer caps still inadequate after {self.max_attempts} rebuilds "
                           f"(last counts: {info}) -- geometry changing between attempts?")

    def value_and_grad(self, loss_fn, geom_fn=None, charge_args_fn=None):
        r"""
        Guarded ``jax.value_and_grad`` through the forward.

        Parameters
        ----------
        loss_fn : callable
            ``loss_fn(out, x, *args) -> scalar``.  Receives ``x`` so regularisers on
            the design variables stay inside the differentiated function (a closure
            over ``x`` would silently drop their gradient).
        geom_fn : callable, optional
            ``geom_fn(x) -> verts`` maps design parameters to the geometry (identity
            when ``None``, i.e. ``x`` IS ``verts``).
        charge_args_fn : callable, optional
            ``charge_args_fn(x) -> tuple`` extracts the traced ``charge_fn`` extras
            (e.g. the magnetisation) from the design variable — the gradient then
            flows through them as well. E.g. ``x = (verts, m)`` with
            ``geom_fn=lambda x: x[0]``, ``charge_args_fn=lambda x: (x[1],)`` yields
            ``grad = (dL/dverts, dL/dm)``. ``None``: no extras.

        Returns
        -------
        vg : callable
            ``vg(x, *args) -> (loss, grad_x)`` where every returned pair was computed
            with certified-adequate buffers (rejected steps rebuild + re-run
            transparently).

        Notes
        -----
        ``loss_fn`` and ``geom_fn`` must be stable function objects (defined once) --
        with a ``vg_builder`` they key the shared jit cache (and the record of compiled
        cap tuples).
        """
        state = {"gen": -1, "vg": None}
        mode = ("vg", id(loss_fn), id(geom_fn), id(charge_args_fn))

        def _vg_for_current_setup():
            if self.vg_builder is not None:                  # retrace-free (module-level jit)
                return self.vg_builder(self.setup, loss_fn, geom_fn, charge_args_fn)
            raw = self.raw_factory(self.setup)               # fallback: re-traces per rebuild

            def f(x, *args):
                cargs = charge_args_fn(x) if charge_args_fn is not None else ()
                out, counts = raw(geom_fn(x) if geom_fn is not None else x, *cargs)
                return loss_fn(out, x, *args), counts

            return jax.jit(jax.value_and_grad(f, has_aux=True))

        def vg(x, *args):
            self._mode = mode
            info = None
            for _ in range(self.max_attempts):
                if state["gen"] != self.generation:
                    state["vg"] = _vg_for_current_setup()
                    state["gen"] = self.generation
                t0 = self._clock()
                (L, counts), g = state["vg"](x, *args)
                ok, info = self.ok_fn(self.setup, counts)
                if ok:
                    self._certified(info, geom_fn(x) if geom_fn is not None else x, self._clock() - t0)
                    return L, g
                self._rebuild(geom_fn(x) if geom_fn is not None else x)
                self._downsize_stuck = False
            raise RuntimeError(f"buffer caps still inadequate after {self.max_attempts} "
                               f"rebuilds (last counts: {info})")

        return vg


def element_quadtree_fulljit_driver(verts, tets, tris, charge_fn, p=None, theta=None, N_max=None,
                                    s=3, duffy_deg=None, headroom=0.15, m2l_snap_K=None, field=False,
                                    mem_limit=None, remat=False, max_attempts=3,
                                    downsize_after=32, targets=None, err_tol=None, size_from=None):
    r"""
    Guarded fully-jit differentiable quad-tree forward (``verts -> u`` or ``H``).

    The recommended way to run shape optimisation.  Wraps
    :func:`element_quadtree_fulljit_setup` / ``_eval`` with the certificate from
    :func:`fulljit_counts_ok` -- connectivity caps, frozen theta-class tiers (including
    orphan classes) and level-structure drift are all rejected, never truncated.

    Parameters
    ----------
    verts : (n_verts, 3) array
        Reference vertex positions (the first setup is fitted here).
    tets, tris : integer arrays
        Element connectivity.
    charge_fn : callable
        ``charge_fn(verts, *charge_args) -> (rho, sig)`` volume/surface charge
        densities; ``charge_args`` are the traced extras passed at call time
        (``driver(verts, *charge_args)``).
    p, theta, N_max, s, duffy_deg, m2l_snap_K, targets, err_tol
        Forwarded to :func:`element_quadtree_fulljit_setup`; ``err_tol`` picks an
        :data:`jaxfmm.fem.tolerances.INTERIOR_LADDER` rung once, and every rebuild reuses it
        (no ``err_tol`` and no accuracy knobs: rung I1).
    headroom : float
        Spare capacity on the frozen buffer caps, as a fraction of the probed counts (caps are
        then rounded onto a geometric 19 %-step ladder, so recurring rebuilds reuse compiled
        shapes).
    size_from : array or sequence of arrays, optional
        Extra vertex positions (same topology) the initial caps must also hold, e.g. the
        heaviest pose of a scan; avoids the rebuild cascade of caps fitted at a light pose.
    field, mem_limit, remat : see the frozen evaluators; ``mem_limit=None`` (default)
        resolves to the L2-anchored auto budget (:func:`jaxfmm.near.auto_mem_limit`)
        Forwarded to the compiled forward.
    max_attempts, downsize_after : int
        Guard policy (see :class:`GuardedShapeEvaluator`).

    Returns
    -------
    GuardedShapeEvaluator
        Certified, auto-rebuilding ``verts -> u/H`` evaluator.
    """
    from jaxfmm.fem.tolerances import interior_params
    _r = interior_params(err_tol, jnp.asarray(verts).dtype == jnp.float64, "element_quadtree_fulljit_driver",
                         p=p, theta=theta, N_max=N_max, duffy_deg=duffy_deg)
    p, theta, N_max, duffy_deg = _r["p"], _r["theta"], _r["N_max"], _r["duffy_deg"]
    if mem_limit is None:                       # auto near-field budget, like the sibling drivers
        from jaxfmm.near import auto_mem_limit
        mem_limit = float(auto_mem_limit(N_max, jnp.asarray(verts).dtype.itemsize))
    import jaxfmm.fem as efmm

    def build(v, cap_floor=None):
        return efmm.element_quadtree_fulljit_setup(v, tets, tris, p=p, theta=theta, N_max=N_max,
                                                   s=s, duffy_deg=duffy_deg, headroom=headroom,
                                                   m2l_snap_K=m2l_snap_K, targets=targets,
                                                   cap_floor=cap_floor)

    def raw_factory(setup):                          # module-level jit: rebuilds with repeating
        return efmm.element_quadtree_fulljit_compile(  # buffer shapes cost no retrace/compile
            setup, charge_fn, field=field, mem_limit=mem_limit, remat=remat,
            return_counts=True)

    def vg_builder(setup, loss_fn, geom_fn, charge_args_fn=None):
        meta, arrs = efmm._fulljit_split(setup)

        def vg_fn(x, *args):
            return efmm._fulljit_vg(arrs, x, tuple(args), meta, charge_fn, loss_fn, geom_fn,
                                    field, mem_limit, remat, charge_args_fn=charge_args_fn)
        return vg_fn

    def oversized(setup, info, cap_floor=None):
        return efmm.fulljit_oversized(setup, info, headroom=headroom, cap_floor=cap_floor)

    def refit(setup, info, cap_floor=None):
        return efmm.fulljit_refit(setup, info, headroom=headroom, cap_floor=cap_floor)

    v0 = jnp.asarray(verts)
    group = ("element_quadtree", id(charge_fn), v0.shape, str(v0.dtype), tuple(jnp.shape(tets)),
             tuple(jnp.shape(tris)), p, theta, N_max, s, duffy_deg, headroom, m2l_snap_K, field,
             float(mem_limit), remat, None if targets is None else tuple(jnp.shape(targets)))
    return GuardedShapeEvaluator(verts, build, raw_factory, efmm.fulljit_counts_ok,
                                 max_attempts=max_attempts, vg_builder=vg_builder,
                                 oversized_fn=oversized, downsize_after=downsize_after,
                                 caps_fn=efmm.fulljit_caps, size_from=size_from,
                                 key_fn=lambda setup: efmm._fulljit_split(setup)[0],
                                 refit_fn=refit, cache_group=group)


def element_uniform_diff_driver(verts, tets, tris, charge_fn, p=3, N_max=64, duffy_deg=3,
                                bin_p2p=True, K=16, bin_ratio=1.2, max_depth=None,
                                mem_limit=None, p2p_mixed=False, remat=False, max_attempts=3):
    r"""
    Guarded shape-differentiable uniform-grid forward (``verts -> u``).

    The frozen Morton topology is EXACT while every particle stays inside its assigned
    leaf cell of the frozen lattice; the in-jit staleness counts
    (:func:`uniform_diff_counts_ok`) detect the first particle that leaves (or exits the
    frozen root box), reject the step and rebuild.

    Parameters
    ----------
    verts : (n_verts, 3) array
        Reference vertex positions (the frozen topology is fitted here).
    tets, tris : integer arrays
        Element connectivity.
    charge_fn : callable
        ``charge_fn(verts) -> (rho, sig)`` volume/surface charge densities.
    p, N_max, duffy_deg, bin_p2p, K, bin_ratio, max_depth
        Forwarded to :func:`jaxfmm.fem.element_uniform_diff_setup`.
    mem_limit, p2p_mixed, remat
        Forwarded to the compiled forward.
    max_attempts : int
        Guard policy (see :class:`GuardedShapeEvaluator`).

    Returns
    -------
    GuardedShapeEvaluator
        Certified, auto-rebuilding ``verts -> u`` evaluator.
    """
    import jaxfmm.fem as efmm

    def build(v):
        return efmm.element_uniform_diff_setup(v, tets, tris, p=p, N_max=N_max,
                                               duffy_deg=duffy_deg, bin_p2p=bin_p2p, K=K,
                                               bin_ratio=bin_ratio, max_depth=max_depth)

    def raw_factory(setup):                          # module-level jit (see quadtree factory)
        return efmm.element_uniform_diff_compile(setup, charge_fn, mem_limit=mem_limit,
                                                 p2p_mixed=p2p_mixed, remat=remat,
                                                 return_counts=True)

    return GuardedShapeEvaluator(verts, build, raw_factory, efmm.uniform_diff_counts_ok,
                                 max_attempts=max_attempts)


def kifmm_fulljit_driver(positions, charge_fn, p=3, N_max=64, headroom=0.15, max_depth=None,
                         batch_size=64, mem_limit=None, p2p_mixed=True, field=False,
                         max_attempts=3, downsize_after=32, size_from=None):
    r"""
    Guarded fully-jit moving-geometry KIFMM (``positions -> pot/H``).

    The tree and every interaction list are rebuilt INSIDE the compiled forward from the
    current ``positions`` (frozen caps from :func:`jaxfmm.kifmm.fulljit_setup`), so
    arbitrary point motion runs with zero recompiles while the caps hold -- and every
    returned result is certified against the realized in-jit counts (tree buffer/depth
    overflow, pil width, near/V/W/X pair counts), reject+rebuild on violation.
    ``charge_fn(positions) -> charges`` is applied outside the jit boundary (pass
    ``lambda pos: q`` for fixed charges).

    Parameters
    ----------
    positions : (N, 3) array
        Reference point positions (the first setup's caps are fitted here).
    charge_fn : callable
        ``charge_fn(positions) -> (N,) charges``, applied outside the jit boundary.
    p, N_max, headroom, max_depth, batch_size, mem_limit, p2p_mixed
        Forwarded to :func:`jaxfmm.kifmm.fulljit_setup`.
    field : bool
        Evaluate the field (negative gradient) instead of the potential.
    max_attempts, downsize_after : int
        Guard policy (see :class:`GuardedShapeEvaluator`).
    size_from : array or sequence of arrays, optional
        Extra positions the initial caps must also hold (see :class:`GuardedShapeEvaluator`).

    Returns
    -------
    GuardedShapeEvaluator
        Certified, auto-rebuilding ``positions -> pot/H`` evaluator.

    Notes
    -----
    Gradients through this forward (d pot / d positions) are structurally available
    (integer paths are stop_gradient'd) but not performance-validated -- benchmark
    against the flexible quad-tree diff path before adopting for optimisation loops.
    """
    from jaxfmm import kifmm

    def build(pos, cap_floor=None):
        return kifmm.fulljit_setup(pos, p=p, N_max=N_max, headroom=headroom,
                                   max_depth=max_depth, batch_size=batch_size,
                                   mem_limit=mem_limit, p2p_mixed=p2p_mixed, cap_floor=cap_floor)

    def raw_factory(setup):
        fwd = kifmm.compile_fulljit(setup, field=field, return_counts=True)
        return lambda pos: fwd(pos, charge_fn(pos))

    def oversized(setup, info, cap_floor=None):
        return kifmm.fulljit_oversized(setup, info, headroom=headroom, cap_floor=cap_floor)

    def refit(setup, info, cap_floor=None):
        return kifmm.fulljit_refit(setup, info, headroom=headroom, cap_floor=cap_floor)

    return GuardedShapeEvaluator(positions, build, raw_factory, kifmm.fulljit_counts_ok,
                                 max_attempts=max_attempts,
                                 oversized_fn=oversized, downsize_after=downsize_after,
                                 caps_fn=kifmm.fulljit_caps, size_from=size_from,
                                 key_fn=lambda setup: setup["meta"], refit_fn=refit)
