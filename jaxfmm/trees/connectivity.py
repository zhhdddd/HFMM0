r"""Flexible-tree connectivity and interaction-list setup.

The interaction lists for the perfectly-balanced median-split tree
(``balanced_tree`` in ``trees/flexible.py``): eager ``gen_connectivity``, the
jit fixed-buffer ``AdaptiveConnectivity`` / ``gen_connectivity_jit`` /
``connectivity_static`` / ``_connectivity_core``, and the ``gen_hierarchy``
setup that ties tree + connectivity together for the harmonic FMM.

A source/eval box pair with centre distance :math:`d` and radii
:math:`r_1, r_2` is well separated (a multipole pair) when the multipole
acceptance criterion holds:

.. math:: R + \theta\, r \le \theta\, d, \qquad
          R = \max(r_1, r_2),\; r = \min(r_1, r_2),

with :math:`d > 0` required (zero-extent boxes would otherwise pass at zero
distance). Non-well-separated pairs descend the tree; at the leaf level they
become direct (near-field) pairs.
"""
import math
import jax
import jax.numpy as jnp
from functools import partial
from jaxfmm.pbc import gen_img_connectivity, gen_pbc_op, with_basis_offsets
from jaxfmm.kernels.laplace import precompute_combs
from jaxfmm.trees.flexible import balanced_tree, get_max_l, reduce_max_lvl

def gen_connectivity(boxcenters, boxlens, eval_boxcenters, eval_boxlens, theta = 0.75, s = 3, no_cross_level = False, periodic_axes = (), basis_offsets = None):
    r"""Compute connectivity information for a given hierarchy (eager, host-driven).

    Dual descent of the source and eval trees: at each step, well-separated
    candidate pairs (see the module docstring) are recorded as multipole (M2L)
    pairs and dropped; the rest descend, subdividing whichever tree currently
    has the larger boxes (cross-level traversal). The remaining
    non-well-separated leaf pairs become the direct list.

    For a data-dependent buffer-free build; see :func:`gen_connectivity_jit`
    for the jit-compilable fixed-buffer drop-in.

    Parameters
    ----------
    boxcenters, boxlens : list of jnp.ndarray
        Per-level source box centres / side lengths.
    eval_boxcenters, eval_boxlens : list of jnp.ndarray
        Per-level eval box centres / side lengths (may be the same objects).
    theta : float, optional
        Well-separatedness parameter (accuracy knob).
    s : int or tuple, optional
        Split spec; kept for back-compat only — per-level child counts are
        derived from the box counts (balanced trees are full).
    no_cross_level : bool, optional
        Disable cross-level comparisons (synchronized descent).
    periodic_axes : tuple of int, optional
        Periodic dimensions; adds image candidates from
        ``gen_img_connectivity``.

    Returns
    -------
    boxcenters : jnp.ndarray
        Concatenated source box centres up to the deepest mpl level.
    eval_boxcenters : jnp.ndarray
        Concatenated eval box centres.
    src_ofs, trg_ofs : tuple of int
        Per-level offsets into the concatenated centre arrays.
    mpl_cnct : jnp.ndarray
        M2L pairs ``(trg, src)``, trg-sorted; src ids encode
        ``box + nbox_tot * image`` under PBC.
    dir_cnct : jnp.ndarray
        Direct (leaf) pairs ``(trg, src)``, trg-sorted, LOCAL leaf indices.
    lvl_info : tuple of tuple
        The traversed ``(l_eval, l_src)`` sequence; ``[-2]`` is the finest mpl
        level, ``[-1]`` the direct (terminal) level.
    img_cnct : jnp.ndarray
        Periodic image offsets (``(1, 0)`` when open).
    """
    n_l, eval_n_l = len(boxcenters), len(eval_boxcenters)  # number of levels of the source and eval hierarchy
    # per-level child counts DERIVED from the box counts (balanced trees are full), so a
    # per-level split schedule (flexible.split_schedule) needs no API change; `s` is kept
    # for back-compat only
    chi_s = [boxcenters[l+1].shape[0] // boxcenters[l].shape[0] for l in range(n_l-1)]
    chi_e = [eval_boxcenters[l+1].shape[0] // eval_boxcenters[l].shape[0] for l in range(eval_n_l-1)]
    periodic = len(periodic_axes) > 0
    if(periodic):
        img_cnct = with_basis_offsets(gen_img_connectivity(boxlens[0], theta, periodic_axes)[0],
                                      basis_offsets)
    else:
        img_cnct = jnp.array([[]])

    n_img = img_cnct.shape[0]
    l_src, l_eval = 0, 0
    mpl_cnct, lvl_info = [], []
    nbox_tot = sum([boxcenter.shape[0] for boxcenter in boxcenters])  # this is the value we need to pad the periodic interactions
    non_wellseps = jnp.arange(n_img)[None,:]
    trg_ofs = tuple(jnp.cumsum(jnp.array([0]+[tbox.shape[0] for tbox in eval_boxcenters])).tolist())
    src_ofs = tuple(jnp.cumsum(jnp.array([0]+[sbox.shape[0] for sbox in boxcenters])).tolist())
    keepgoing = True
    while(keepgoing):
        evalend, srcend = (l_eval >= (eval_n_l-1)), (l_src >= (n_l-1))
        keepgoing = not (evalend and srcend)    # we keep going until we reach the final level in both hierarchies
        nbox, nbox_eval = boxcenters[l_src].shape[0], eval_boxcenters[l_eval].shape[0]

        if(periodic):
            rhs = boxcenters[l_src][non_wellseps%nbox] + img_cnct.at[non_wellseps//nbox].get(mode="fill",fill_value=jnp.nan)
        else:
            rhs = boxcenters[l_src].at[non_wellseps].get(mode="fill",fill_value=jnp.nan)
        d = jnp.linalg.norm(eval_boxcenters[l_eval][:,None,:] - rhs,axis=-1)  # distances between boxcenters

        r1 = jnp.linalg.norm(eval_boxlens[l_eval][:,None,:],axis=-1)/2  # eval box radii
        r2 = jnp.linalg.norm(boxlens[l_src][non_wellseps%nbox if periodic else non_wellseps],axis=-1)/2 # source box radii
        tmp = r1 > r2
        R = jnp.where(tmp,r1,r2)     # R = max(r1,r2)
        r = jnp.where(tmp,r2,r1)     # r = min(r1,r2)

        # d > 0: a pair can never be well-separated at zero distance. Occupancy-1 leaves have
        # ZERO-extent tight AABBs (R = r = 0), which otherwise makes even the SELF-pair pass
        # the MAC -> M2L at dvec = 0 -> 1/r = inf -> NaN.
        # Point-boxes at any d > 0 are legitimately ws (their expansions are exact).
        ws = (R + theta*r <= theta*d) & (d > 0)     # well-separatedness criterion, NaNs always return False
        non_ws = ~ws & ~jnp.isnan(d)                # need to check again to exclude NaNs (padding)
        ws_nums = ws.sum(axis=1)          # number of well-separated boxes for each box
        non_ws_nums = non_ws.sum(axis=1)  # number of non-well-separated boxes for each box

        if(ws_nums.sum() > 0):       # do not save empty cnct information
            trgid = jnp.repeat(jnp.arange(nbox_eval),ws_nums) + trg_ofs[l_eval]
            srcid = non_wellseps[ws]
            srcid = jnp.where(srcid>=nbox,(srcid%nbox + (nbox_tot*(srcid//nbox))),srcid)
            srcid += src_ofs[l_src]
            pairs = jnp.stack((trgid,srcid),axis=1)
            mpl_cnct.append(pairs)
            lvl_info.append((l_eval, l_src))

        if(not (evalend and srcend)):
            to_pad_non_ws = jnp.max(non_ws_nums) - non_ws_nums    # how much padding per box must be inserted for non-well-separated boxes
            non_wellsep_padding = jnp.repeat(jnp.cumsum(non_ws_nums),to_pad_non_ws)   # the correct indices for the insert below
            non_wellseps = jnp.insert(non_wellseps[non_ws],non_wellsep_padding,n_img*nbox).reshape((nbox_eval,-1))    # we overwrite the old values here
            if(non_wellseps.shape[1] == 0):   # we are already done, quit out
                break

        r1mean, r2mean = jnp.mean(r1), jnp.mean(r2)
        if((not srcend) and ((r1mean < 1.05 * r2mean) or evalend or no_cross_level)): # advance the tree with the larger mean radius
            nc = chi_s[l_src]   # children of the src level being subdivided
            l_src += 1          # src_boxes >= eval_boxes -> increase src lvl
            non_wellseps = jnp.repeat(non_wellseps,nc,axis=1)*nc + jnp.tile(jnp.arange(nc),non_wellseps.shape[1])   # indices change
        if((not evalend) and ((r1mean >= 1.05 * r2mean) or srcend or no_cross_level)):
            non_wellseps = jnp.repeat(non_wellseps,chi_e[l_eval],axis=0)    # size of non-wellseps grows
            l_eval += 1             # eval boxes > src boxes -> increase eval lvl

    lvl_info.append((l_eval, l_src))

    ### stack everything together and cut unneeded information
    boxcenters = jnp.concatenate(boxcenters[:lvl_info[-1][1]+1])
    eval_boxcenters = jnp.concatenate(eval_boxcenters[:lvl_info[-1][0]+1])
    src_ofs = src_ofs[:lvl_info[-1][1]+2]
    trg_ofs = trg_ofs[:lvl_info[-1][0]+2]

    if(len(mpl_cnct)>0):
        mpl_cnct = jnp.concatenate(mpl_cnct,axis=0)
        mpl_cnct = mpl_cnct[jnp.argsort(mpl_cnct[:,0])] # sort along axis 0
    else:
        mpl_cnct = jnp.array([[]],dtype=jnp.int32)

    if(non_ws.sum()>0): # if nonempty
        trgid = jnp.repeat(jnp.arange(nbox_eval),non_ws_nums)   # dir ids stay LOCAL (no level offsets)
        srcid = non_wellseps[non_ws]
        dir_cnct = jnp.stack((trgid,srcid),axis=1)
        dir_cnct = dir_cnct[jnp.argsort(dir_cnct[:,0])] # sort along axis 0
    else:
        dir_cnct = jnp.array([[]],dtype=jnp.int32)

    return boxcenters, eval_boxcenters, src_ofs, trg_ofs, mpl_cnct, dir_cnct, tuple(lvl_info), img_cnct

# traversal step actions: which tree(s) subdivide after a step
_NONE, _SRC, _EVAL, _BOTH = 0, 1, 2, 3

def _conn_steps(max_l, cross_level):
    r"""Static traversal sequence of ``(l_eval, l_src, action)`` for symmetric trees.

    Synchronized descent (``cross_level=False``) advances both trees together
    each step. Cross-level alternates advancing src / eval based on the
    relative box size — for a balanced symmetric tree the mean radius is
    :math:`\sim R_0 / 2^l`, so the decision (advance src iff
    ``l_src <= l_eval``) is deterministic, giving the alternating sequence
    ``(0,0), (0,1), (1,1), (1,2), ..., (max_l, max_l)``.
    """
    if max_l == 0:
        return ((0, 0, _NONE),)
    if not cross_level:
        return tuple((l, l, _BOTH) for l in range(max_l)) + ((max_l, max_l, _NONE),)
    steps = []; le = ls = 0
    while not (le == max_l and ls == max_l):
        adv_src = (ls < max_l) and (ls <= le or le == max_l)
        adv_eval = (le < max_l) and (ls > le or ls == max_l)
        action = _BOTH if (adv_src and adv_eval) else (_SRC if adv_src else _EVAL)
        steps.append((le, ls, action))
        if action in (_SRC, _BOTH): ls += 1
        if action in (_EVAL, _BOTH): le += 1
    steps.append((max_l, max_l, _NONE))
    return tuple(steps)

def _conn_steps_radii(r1mean, r2mean, cross_level):
    r"""Static traversal sequence for the asymmetric case (eval tree != src tree).

    Derived on the host from the concrete per-level mean box radii ``r1mean``
    (eval, indexed by ``l_eval``) and ``r2mean`` (src, indexed by ``l_src``).
    Replicates the data-dependent descent in :func:`gen_connectivity` (advance
    src iff ``r1mean[le] < 1.05*r2mean[ls]`` etc.) but with level-wide means
    instead of the running candidate-set mean. The step *order* only affects
    how aggressively coarse well-separated pairs are compressed, never
    correctness/accuracy: the ws criterion is enforced before every mpl record
    and the loop always terminates at ``(eval_max, src_max)``, so any
    reasonable order yields a valid connectivity.
    """
    eval_max, src_max = len(r1mean) - 1, len(r2mean) - 1
    steps = []; le = ls = 0
    while True:
        evalend, srcend = (le >= eval_max), (ls >= src_max)
        if evalend and srcend:
            steps.append((le, ls, _NONE)); break
        adv_src = (not srcend) and ((r1mean[le] < 1.05 * r2mean[ls]) or evalend or not cross_level)
        adv_eval = (not evalend) and ((r1mean[le] >= 1.05 * r2mean[ls]) or srcend or not cross_level)
        action = _BOTH if (adv_src and adv_eval) else (_SRC if adv_src else _EVAL)
        steps.append((le, ls, action))
        if adv_src: ls += 1
        if adv_eval: le += 1
    return tuple(steps)

def _step_widths(steps, n_chi, W, n_img, nbox_s, nbox_e):
    r"""Per-step candidate widths (``width_in``) and non-ws caps (``keep``), static.

    ``nbox_s``/``nbox_e`` are the per-level source / eval box counts (equal in
    the symmetric case). ``keep``/``width_in`` follow the *source*
    subdivision.
    """
    width_in, keep = [], []
    w = n_img                                       # step 0: n_img images of box 0
    for (le, ls, action) in steps:
        width_in.append(w)
        k = min(W, nbox_s[ls] * n_img); keep.append(k)
        if action in (_SRC, _BOTH):                 # child count of the level being subdivided
            w = min(k * (nbox_s[ls + 1] // nbox_s[ls]), nbox_s[ls + 1] * n_img)
        else:
            w = k
    return tuple(width_in), tuple(keep)

# largest pair-buffer ROW count the fixed-buffer connectivity may index with int32 (x64
# off). ROWS are the binding constraint (index VALUES in _compact's cumsum/scatter must
# fit int32); a (rows, 2) buffer with >2^31 ELEMENTS is fine. Margin 2^24 below 2^31.
INT32_CNCT_LIMIT = 2**31 - 2**24

def _cnct_size_report(nbox_s, nbox_e, steps, width_in, keep, n_mpl, n_dir, nbox_tot, n_img):
    r"""Element-count summary of a fixed-buffer connectivity (static, host-side)."""
    step_elems = max(nbox_e[le] * width_in[i] for i, (le, ls, a) in enumerate(steps))
    return dict(n_mpl=int(n_mpl), n_dir=int(n_dir), step_elems=int(step_elems),
                src_id_max=int(nbox_tot) * int(n_img),
                worst=int(max(n_mpl, n_dir, step_elems, nbox_tot * n_img)))

def predict_cnct_sizes(N_src, N_max, s, W=None, no_cross_level=False, n_img=1):
    r"""Predict the fixed-buffer connectivity sizes for a symmetric tree.

    Works from the point count alone (no tree build; mirrors
    :func:`gen_hierarchy`'s schedule resolution and
    :func:`connectivity_static`'s sizing). Used to detect int32 overflow
    *before* building a deep tree — at extreme ``N / N_max`` ratios the padded
    ``(n_dir, 2)`` buffer alone can exceed ``2**31`` rows.

    Parameters
    ----------
    N_src : int
        Number of source points.
    N_max : int
        Maximum points per leaf.
    s : int, tuple or "auto"
        Split spec.
    W : int, optional
        Non-well-separated cap per box (defaults as in
        :func:`connectivity_static`).
    no_cross_level : bool, optional
        Synchronized descent.
    n_img : int, optional
        Number of periodic images.

    Returns
    -------
    dict
        The size report of :func:`_cnct_size_report` (worst count in
        ``"worst"``).
    """
    from jaxfmm.trees.flexible import lane_schedule, children_of, get_max_l
    sched = lane_schedule(N_src, N_max, 2) if s == "auto" else s
    max_l = get_max_l(N_src, N_max, sched)
    if isinstance(sched, tuple):
        nbox, nb = [1], 1
        for c in children_of(sched, max_l):
            nb *= c; nbox.append(nb)
        nbox = tuple(nbox[:max_l + 1])
    else:
        nbox = tuple(2 ** (sched * l) for l in range(max_l + 1))
    steps = _conn_steps(max_l, not no_cross_level)
    if W is None:
        W = 48 if no_cross_level else 256
    width_in, keep = _step_widths(steps, 0, W, n_img, nbox, nbox)
    n_mpl = sum(nbox[le] * width_in[i] for i, (le, ls, a) in enumerate(steps))
    n_dir = nbox[steps[-1][0]] * keep[-1]
    return _cnct_size_report(nbox, nbox, steps, width_in, keep, n_mpl, n_dir, sum(nbox), n_img)

@partial(jax.jit, static_argnames=["s", "n_mpl", "n_dir", "src_ofs", "eval_ofs", "steps", "widths", "nbox_tot"])
def _connectivity_core(bc_s, bl_s, bc_e, bl_e, theta, s, n_mpl, n_dir, src_ofs, eval_ofs,
                       steps, widths, img_cnct, nbox_tot):
    r"""JIT core of the fixed-buffer connectivity.

    Handles symmetric OR asymmetric (eval != src) trees, open OR periodic
    boundaries, synchronized OR cross-level descent. ``steps`` is the static
    ``(l_eval, l_src, action)`` traversal. At each step the eval boxes
    ``bc_e[l_eval]`` are checked against candidate src boxes ``bc_s[l_src]``;
    candidate values encode ``v = box + nbox_{l_src} * image``.

    Records well-separated pairs into ``mpl_cnct`` ``(n_mpl, 2)`` (trg id =
    eval box + ``eval_ofs[l_eval]``; src id = box + ``src_ofs[l_src]`` +
    ``nbox_tot * image``) and, at the terminal step, the remaining non-ws as
    ``dir_cnct`` ``(n_dir, 2)`` LOCAL leaf pairs (src id = box +
    ``nbox_leaf * image``). In the symmetric case the caller passes
    ``bc_s is bc_e`` etc. Both buffers are iinfo-max padded and trg-sorted.

    Returns
    -------
    mpl_cnct, dir_cnct : jnp.ndarray
        Fixed-size padded pair buffers.
    n_mpl_used, n_dir_used : jnp.ndarray
        Realised pair counts.
    max_nws : jnp.ndarray
        Maximum non-well-separated candidates per box (the ``W`` check).
    """
    idt = jnp.arange(1).dtype                       # default int dtype (int32/int64 per x64 config)
    SENT = int(jnp.iinfo(idt).max)                  # consumer's padding sentinel (M2L/eval_direct drop it)
    nbox_s, nbox_e, width_in, keep = widths         # per-level child factors derive from these
    mpl_t, mpl_s = [], []
    dir_t = dir_s = None
    max_nws = jnp.zeros((), idt)
    cand = jnp.arange(img_cnct.shape[0], dtype=idt)[None, :]   # (1, n_img): images of box 0
    for i, (le, ls, action) in enumerate(steps):
        nb_e, nb_s = nbox_e[le], nbox_s[ls]
        valid = cand >= 0
        v = jnp.where(valid, cand, 0)
        box = v % nb_s; image = v // nb_s           # decode src box + periodic image
        src_pos = bc_s[ls][box] + img_cnct[image]
        d = jnp.linalg.norm(bc_e[le][:, None, :] - src_pos, axis=-1)
        r1 = jnp.linalg.norm(bl_e[le], axis=-1)[:, None] / 2
        r2 = jnp.linalg.norm(bl_s[ls][box], axis=-1) / 2
        R = jnp.maximum(r1, r2); rr = jnp.minimum(r1, r2)
        ws = valid & (R + theta * rr <= theta * d) & (d > 0)   # d>0: zero-extent (occupancy-1)
        nws = valid & (~ws)                                    # boxes must not ws their own image
        max_nws = jnp.maximum(max_nws, nws.sum(axis=1).max())
        trg_g = jnp.arange(nb_e, dtype=idt)[:, None] + eval_ofs[le]
        src_g = box.astype(idt) + src_ofs[ls] + nbox_tot * image.astype(idt)
        mpl_t.append(jnp.where(ws, trg_g, SENT).ravel())
        mpl_s.append(jnp.where(ws, src_g, SENT).ravel())
        if action == _NONE:                         # terminal (l_eval==l_src==max_l): dir
            dir_t = jnp.where(nws, jnp.arange(nb_e, dtype=idt)[:, None], SENT).ravel()
            dir_s = jnp.where(nws, v.astype(idt), SENT).ravel()
            break
        # compact non-ws to keep[i] per eval box (sort non-ws first), then subdivide
        order = jnp.argsort(~nws, axis=1)
        k = keep[i]
        cand_k = jnp.where(jnp.take_along_axis(nws, order, axis=1)[:, :k],
                           jnp.take_along_axis(cand, order, axis=1)[:, :k], -1)
        if action in (_SRC, _BOTH):                 # subdivide source boxes (columns)
            nc = nbox_s[ls + 1] // nbox_s[ls]       # children of src level ls
            child = jnp.where(cand_k[..., None] < 0, -1, cand_k[..., None] * nc + jnp.arange(nc))
            cand_k = child.reshape(nb_e, k * nc)
        w = width_in[i + 1]
        cand_k = (cand_k[:, :w] if cand_k.shape[1] >= w
                  else jnp.pad(cand_k, ((0, 0), (0, w - cand_k.shape[1])), constant_values=-1))
        if action in (_EVAL, _BOTH):                # subdivide eval boxes (rows inherit candidates)
            cand_k = jnp.repeat(cand_k, nbox_e[le + 1] // nbox_e[le], axis=0)
        cand = cand_k

    def _compact(tg, sc, n):
        good = tg != SENT
        dst = jnp.where(good, jnp.cumsum(good) - 1, n)
        out = jnp.full((n, 2), SENT, idt).at[dst].set(jnp.stack([tg, sc], 1), mode="drop")
        return out, good.sum()
    mpl_cnct, n_mpl_used = _compact(jnp.concatenate(mpl_t), jnp.concatenate(mpl_s), n_mpl)
    dir_cnct, n_dir_used = _compact(dir_t, dir_s, n_dir)
    return mpl_cnct, dir_cnct, n_mpl_used, n_dir_used, max_nws

def connectivity_static(boxcenters, boxlens, eval_boxcenters, eval_boxlens,
                        theta=0.77, s=3, W=None, n_mpl=None, n_dir=None, periodic_axes=(),
                        no_cross_level=False, basis_offsets=None):
    r"""Host-side static bundle of the fixed-buffer connectivity.

    Everything in :func:`gen_connectivity_jit` that depends only on the tree
    *shape* (per-level box counts) and, for asymmetric trees, the per-level
    mean box radii — but NOT on the box *positions*. Computed once (eagerly)
    it lets a caller freeze the traversal (``steps``), offsets, buffer sizes
    and ``lvl_info`` and then call :func:`_connectivity_core` repeatedly
    *inside one jit* on changing geometry with no recompile (the positions
    enter only the traced core). Used by :func:`gen_connectivity_jit` and by
    the fully-jit shape-opt path.

    Parameters
    ----------
    boxcenters, boxlens, eval_boxcenters, eval_boxlens : list of jnp.ndarray
        Per-level box geometry (pass the same objects for symmetric trees);
        only the shapes (and, asymmetric, the mean radii) are used here.
    theta : float, optional
        Well-separatedness parameter.
    s : int or tuple, optional
        Split spec (back-compat; shape is derived from the box counts).
    W : int, optional
        Cap on non-well-separated neighbours per box (default 48
        synchronized / 256 cross-level).
    n_mpl, n_dir : int, optional
        Fixed buffer sizes (auto-sized from the traversal when None).
    periodic_axes : tuple of int, optional
        Periodic dimensions.
    no_cross_level : bool, optional
        ``False`` (default) uses the cross-level (alternating src/eval)
        descent; ``True`` uses synchronized levels (only valid when
        eval == src).

    Returns
    -------
    dict
        Static metadata: traversal ``steps``, per-level box counts and
        widths, offsets, buffer sizes, ``img_cnct``, ``lvl_info``.

    Notes
    -----
    ``lvl_info`` here assumes mpl pairs exist (the dense-cloud case);
    :func:`gen_connectivity_jit` overrides it with the direct-only form when
    the realised mpl count is zero.
    """
    symmetric = eval_boxcenters is boxcenters
    if not symmetric:
        no_cross_level = False                      # asymmetric trees need cross-level
    if W is None:
        W = 48 if no_cross_level else 256           # overflow raises -> increase if needed
    src_max, eval_max = len(boxcenters) - 1, len(eval_boxcenters) - 1
    n_chi = (boxcenters[-1].shape[0] // boxcenters[-2].shape[0]) if src_max > 0 else 2 ** (s if isinstance(s, int) else s[-1])
    if len(periodic_axes) > 0:
        img_cnct = with_basis_offsets(gen_img_connectivity(boxlens[0], theta, periodic_axes)[0], basis_offsets)
    else:
        img_cnct = jnp.zeros((1, 3))                # open BC: one zero image offset
    n_img = int(img_cnct.shape[0])
    nbox_s = tuple(int(b.shape[0]) for b in boxcenters)
    nbox_e = tuple(int(b.shape[0]) for b in eval_boxcenters)
    if symmetric:
        steps = _conn_steps(src_max, not no_cross_level)
    else:
        r1mean = tuple(float(jnp.mean(jnp.linalg.norm(bl, axis=-1) / 2)) for bl in eval_boxlens)
        r2mean = tuple(float(jnp.mean(jnp.linalg.norm(bl, axis=-1) / 2)) for bl in boxlens)
        steps = _conn_steps_radii(r1mean, r2mean, not no_cross_level)
    width_in, keep = _step_widths(steps, n_chi, W, n_img, nbox_s, nbox_e)
    src_ofs = tuple(int(x) for x in (jnp.cumsum(jnp.array([0] + list(nbox_s)))))
    eval_ofs = tuple(int(x) for x in (jnp.cumsum(jnp.array([0] + list(nbox_e)))))
    nbox_tot = int(src_ofs[-1])
    if n_mpl is None:
        n_mpl = int(sum(nbox_e[le] * width_in[i] for i, (le, ls, a) in enumerate(steps)))
    if n_dir is None:
        n_dir = int(nbox_e[steps[-1][0]] * keep[-1])
    # int32 safety guard: with x64 off, _connectivity_core indexes with int32 and XLA
    # buffers are int32-addressed — a silent wraparound past 2^31 corrupts pair lists.
    # (At N_max=64 / 1e9 pts the padded n_dir alone exceeds 2^31; gen_hierarchy avoids
    # this by auto-doubling N_max, see predict_cnct_sizes.)
    if not jax.config.jax_enable_x64:
        rep = _cnct_size_report(nbox_s, nbox_e, steps, width_in, keep, n_mpl, n_dir, nbox_tot, n_img)
        if rep["worst"] > INT32_CNCT_LIMIT:
            raise ValueError(
                f"connectivity buffers exceed the int32-safe limit with x64 disabled: "
                f"{ {k: v for k, v in rep.items() if v > INT32_CNCT_LIMIT} } > {INT32_CNCT_LIMIT}. "
                f"Remedies: larger N_max (fewer leaves; gen_hierarchy auto-bumps unless disabled), "
                f"smaller W / explicit n_mpl/n_dir, or enable x64.")
    base = tuple((le, ls) for (le, ls, a) in steps)
    return dict(symmetric=symmetric, W=int(W), n_chi=n_chi, src_max=src_max, eval_max=eval_max,
                img_cnct=img_cnct, n_img=n_img, nbox_s=nbox_s, nbox_e=nbox_e, steps=steps,
                width_in=width_in, keep=keep, src_ofs=src_ofs, eval_ofs=eval_ofs, nbox_tot=nbox_tot,
                n_mpl=int(n_mpl), n_dir=int(n_dir), lvl_info=base + (base[-1],), periodic_axes=periodic_axes)

def gen_connectivity_jit(boxcenters, boxlens, eval_boxcenters, eval_boxlens,
                         theta=0.77, s=3, W=None, n_mpl=None, n_dir=None, periodic_axes=(),
                         no_cross_level=False, return_info=False, basis_offsets=None):
    r"""JIT-compilable, fixed-buffer drop-in for :func:`gen_connectivity`.

    Open or periodic, synchronized or cross-level, symmetric (eval == src) or
    asymmetric (eval != src). Bounds the work with ``W`` non-well-separated
    neighbours per box (overflow raises) and returns ``mpl_cnct``/``dir_cnct``
    in fixed-size buffers (``n_mpl``/``n_dir``, iinfo-max padded, trg-sorted).
    The expensive traversal runs entirely inside ``jax.jit``; the static
    metadata (offsets, step sequence, ``lvl_info``) follows from the
    balanced-tree shape (symmetric: a deterministic level rule; asymmetric:
    the per-level mean box radii, computed on the host).

    Parameters
    ----------
    boxcenters, boxlens, eval_boxcenters, eval_boxlens : list of jnp.ndarray
        Per-level box geometry (pass the same objects for symmetric trees).
    theta : float, optional
        Well-separatedness parameter.
    s : int or tuple, optional
        Split spec (back-compat; shape is derived from the box counts).
    W : int, optional
        Cap on non-well-separated neighbours per box.
    n_mpl, n_dir : int, optional
        Fixed buffer sizes (auto-sized from the traversal when None).
    periodic_axes : tuple of int, optional
        Periodic dimensions.
    no_cross_level : bool, optional
        ``False`` (default) uses the cross-level (alternating src/eval)
        descent; ``True`` uses synchronized levels (only valid when
        eval == src).
    return_info : bool, optional
        Also return realised counts / overflow info instead of raising.

    Returns
    -------
    tuple
        The same 8-tuple as :func:`gen_connectivity`; with
        ``return_info=True``, ``(result, info)``.
    """
    # symmetric (eval==src) is detected by object identity (gen_hierarchy / AdaptiveConnectivity
    # pass the *same* box arrays when pts is eval_pts). Equal shapes alone are not enough:
    # balanced trees of equal N always share per-level box counts, so two distinct point sets
    # would otherwise be misclassified. Symmetric uses the validated deterministic step rule;
    # asymmetric derives the steps from per-level mean radii (and forces cross-level).
    st = connectivity_static(boxcenters, boxlens, eval_boxcenters, eval_boxlens,
                             theta, s, W, n_mpl, n_dir, periodic_axes, no_cross_level, basis_offsets=basis_offsets)
    symmetric, src_max, eval_max = st["symmetric"], st["src_max"], st["eval_max"]
    img_cnct, steps, nbox_tot = st["img_cnct"], st["steps"], st["nbox_tot"]
    src_ofs, eval_ofs = st["src_ofs"], st["eval_ofs"]
    nbox_s, nbox_e, width_in, keep = st["nbox_s"], st["nbox_e"], st["width_in"], st["keep"]
    W, n_mpl, n_dir = st["W"], st["n_mpl"], st["n_dir"]
    mpl_cnct, dir_cnct, n_mpl_used, n_dir_used, max_nws = _connectivity_core(
        tuple(boxcenters), tuple(boxlens), tuple(eval_boxcenters), tuple(eval_boxlens),
        theta, s, int(n_mpl), int(n_dir), src_ofs, eval_ofs,
        steps, (nbox_s, nbox_e, width_in, keep), img_cnct, nbox_tot)
    n_mpl_used, n_dir_used, max_nws = int(n_mpl_used), int(n_dir_used), int(max_nws)
    overflow = (max_nws > W) or (n_mpl_used > n_mpl) or (n_dir_used > n_dir)
    if overflow and not return_info:
        raise ValueError(f"connectivity buffer overflow: W {max_nws}/{W}, mpl "
                         f"{n_mpl_used}/{n_mpl}, dir {n_dir_used}/{n_dir}. Increase W/n_mpl/n_dir "
                         f"(or use AdaptiveConnectivity).")
    # static metadata. lvl_info only needs: the finest mpl level at [-2], the dir
    # (terminal) level at [-1], and len>1 iff there are any mpl pairs (else eval_potential
    # takes the direct-only path). Intermediate entries are unused by the consumer, so
    # listing all step levels is harmless; whether mpl exists depends on the data (e.g. PBC
    # has mpl pairs at coarse levels that open BC does not). The terminal step (eval_max,
    # src_max) is duplicated so [-2]==[-1]==terminal, mirroring gen_connectivity.
    src_bc_cat = jnp.concatenate(list(boxcenters))
    eval_bc_cat = src_bc_cat if symmetric else jnp.concatenate(list(eval_boxcenters))
    if n_mpl_used > 0:
        base = tuple((le, ls) for (le, ls, a) in steps)
        lvl_info = base + (base[-1],)
    else:
        lvl_info = ((eval_max, src_max),)
    img_out = img_cnct if len(periodic_axes) > 0 else jnp.array([[]])
    result = (src_bc_cat, eval_bc_cat, src_ofs, eval_ofs, mpl_cnct, dir_cnct, lvl_info, img_out)
    if return_info:
        info = {"n_mpl_used": n_mpl_used, "n_dir_used": n_dir_used, "max_nws": max_nws,
                "overflow": overflow, "W": W, "n_mpl": n_mpl, "n_dir": n_dir}
        return result, info
    return result

def _fit_buffer(used, gran, headroom):
    r"""Smallest multiple of ``gran`` holding ``used``, plus ``headroom`` spare steps."""
    base = ((int(used) + gran - 1) // gran) * gran        # ceil(used/gran)*gran >= used
    return base + headroom * gran

CAP_BIN = 2 ** 0.25          # frozen-cap ladder step (+19 %), see _fit_cap
DOWNSIZE_BINS = 2            # a lazy downsize must drop some cap by >= this many bins (>= 29 %)


def _fit_cap(used, gran, headroom):
    r"""Frozen moving-geometry cap for a realised count: ``used * (1 + headroom)`` rounded UP onto
    the geometric ladder ``gran * ceil(CAP_BIN**k)``, k >= 0; ``headroom`` is a FRACTION.

    Rebuilds then land on caps they have already compiled for: across a 41 x 41 MFM raster
    (270k tets) the counts vary 2x and take ~110 distinct ``gran`` multiples per buffer, i.e.
    nearly every rejection rebuild recompiled (20-25 s each), but only ~5 ladder bins.
    (:func:`_fit_buffer` stays the tight fit for one-shot static hierarchies, which pay every
    padded slot on every evaluation.)"""
    need = max(int(used), 1) * (1.0 + headroom)
    k = max(0, math.ceil(math.log(need / gran) / math.log(CAP_BIN)))
    while gran * math.ceil(CAP_BIN ** k) < need:          # guard the float log
        k += 1
    return gran * math.ceil(CAP_BIN ** k)


def _caps_oversized(dims, headroom):
    r"""Lazy-downsize verdict over ``dims = [(used, cap, gran[, floor]), ...]`` from a CERTIFIED
    call: a refit (never below ``floor``, e.g. the caps of a declared ``size_from`` pose) would
    shrink some cap by >= ``DOWNSIZE_BINS`` ladder bins and grow none. Smaller shrinks barely pay
    (2x-oversized caps cost +31 % per MFM force step) and would oscillate."""
    shrink = False
    for used, cap, gran, *fl in dims:
        want = max(_fit_cap(used, gran, headroom), fl[0] if fl else 0)
        if want > cap:
            return False
        shrink = shrink or want * CAP_BIN ** DOWNSIZE_BINS <= cap * (1 + 1e-9)
    return shrink

class AdaptiveConnectivity:
    r"""Stateful driver around :func:`gen_connectivity_jit` with adaptive buffer sizes.

    Adapts the fixed buffer sizes (``W``, ``n_mpl``, ``n_dir``) in discrete
    (granularity) steps across calls — for moving-geometry / time-stepping
    runs where the exact interaction counts drift.

    On each call it runs the connectivity with the current buffers and
    inspects the realised counts:

    * if a buffer was **too small** (overflow), it grows that buffer to fit
      (rounded up to the granularity, plus ``headroom`` spare steps) and
      **reruns** — the truncated result is rejected;
    * otherwise it **keeps** the result and, if a buffer is comfortably
      oversized, **downsizes** it for the *next* call (no rerun).

    Each distinct buffer size compiles the jit core once (then JAX caches it),
    so after the sizes settle the steady state is recompile-free. ``n_reruns``
    counts the (rare, transient) overflow reruns.

    Parameters
    ----------
    theta, s, periodic_axes, no_cross_level
        Forwarded to :func:`gen_connectivity_jit`.
    W, n_mpl, n_dir : int, optional
        Initial buffer sizes (``None`` resolves on the first call).
    mpl_gran, dir_gran, W_gran : int, optional
        Buffer granularities.
    headroom : int, optional
        Spare granularity steps kept above the realised counts.
    """
    def __init__(self, theta=0.77, s=3, periodic_axes=(), no_cross_level=False,
                 W=None, n_mpl=None, n_dir=None,
                 mpl_gran=4096, dir_gran=1024, W_gran=16, headroom=1, basis_offsets=None):
        self.theta, self.s = theta, s
        self.periodic_axes, self.no_cross_level = periodic_axes, no_cross_level
        self.basis_offsets = basis_offsets
        self.W, self.n_mpl, self.n_dir = W, n_mpl, n_dir      # None -> resolved on first call
        self.mpl_gran, self.dir_gran, self.W_gran = mpl_gran, dir_gran, W_gran
        self.headroom = headroom
        self.n_reruns = 0

    def __call__(self, boxcenters, boxlens, eval_boxcenters=None, eval_boxlens=None):
        ebc = boxcenters if eval_boxcenters is None else eval_boxcenters
        ebl = boxlens if eval_boxlens is None else eval_boxlens
        # None buffers resolve to a-priori worst-case sizes inside the first call; the
        # consumers (M2L/eval_direct) pay for every padded slot, so a one-shot caller
        # (gen_hierarchy) must not be handed that first result — rerun once at the
        # fitted sizes instead of only downsizing for the next call.
        refit = self.n_mpl is None or self.n_dir is None or self.W is None
        while True:
            res, info = gen_connectivity_jit(
                boxcenters, boxlens, ebc, ebl, theta=self.theta, s=self.s,
                W=self.W, n_mpl=self.n_mpl, n_dir=self.n_dir,
                periodic_axes=self.periodic_axes, no_cross_level=self.no_cross_level,
                return_info=True, basis_offsets=self.basis_offsets)
            self.W, self.n_mpl, self.n_dir = info["W"], info["n_mpl"], info["n_dir"]
            if info["overflow"]:                              # too small -> grow + rerun
                if info["max_nws"] > self.W:
                    self.W = _fit_buffer(info["max_nws"], self.W_gran, self.headroom)
                if info["n_mpl_used"] > self.n_mpl:
                    self.n_mpl = _fit_buffer(info["n_mpl_used"], self.mpl_gran, self.headroom)
                if info["n_dir_used"] > self.n_dir:
                    self.n_dir = _fit_buffer(info["n_dir_used"], self.dir_gran, self.headroom)
                self.n_reruns += 1
                continue
            # fit -> keep result; lazily downsize for the next call
            self.n_mpl = min(self.n_mpl, _fit_buffer(info["n_mpl_used"], self.mpl_gran, self.headroom))
            self.n_dir = min(self.n_dir, _fit_buffer(info["n_dir_used"], self.dir_gran, self.headroom))
            self.W = min(self.W, _fit_buffer(info["max_nws"], self.W_gran, self.headroom))
            if refit and (self.n_mpl < info["n_mpl"] or self.n_dir < info["n_dir"] or self.W < info["W"]):
                refit = False                                 # fitted sizes hold the realized counts -> rerun fits
                self.n_reruns += 1
                continue
            return res

def gen_hierarchy(pts, eval_pts = None, N_max = 128, theta = 0.77, s = 3, p = 3, periodic_axes = (), pbc_lvls = 30, pbc_no_monopole = True, domain = None, L0_boxcen = None, L0_boxlen = None, no_cross_level = False, debug_info = False, mem_limit = None, jit_cnct = True, W = None, n_mpl = None, n_dir = None, basis_offsets = None, bloch_k = None, auto_rescale = False, **kwargs):
    r"""Generate the balanced tree and connectivity for the FMM.

    The one-stop setup for the flexible-tree engine: builds the balanced
    source (and optionally eval) tree, computes the connectivity (eager or
    jit fixed-buffer), resolves the level paddings, and assembles the
    hierarchy dict consumed by ``eval_potential``.

    Parameters
    ----------
    pts : jnp.ndarray
        Positions of the N point charges, shape ``(N, 3)``.
    eval_pts : jnp.ndarray, optional
        Positions of the evaluation points, shape ``(N_eval, 3)``. Defaults
        to ``None``, where ``eval_pts = pts``.
    N_max : int, optional
        Maximum allowed number of point charges per box.
    theta : float, optional
        Well-separatedness parameter, determines accuracy.
    s : int, tuple or "auto", optional
        Splits per level and box (each box has ``2**s`` children); a
        per-level tuple or ``"auto"`` (see :func:`~jaxfmm.trees.flexible.split_schedule`)
        gives fractional depth.
    p : int, optional
        Maximum expansion order.
    periodic_axes : tuple of int, optional
        Which dimensions (0, 1 and/or 2) are periodic.
    pbc_lvls : int, optional
        Cap on the number of virtual doubling levels for the periodic far
        field (the assembly stops early once the lattice operator is
        converged — the default 30 is a safety ceiling, effective levels are
        typically 10–20). If negative, only the non-well-separated boxes on
        level zero are added.
    pbc_no_monopole : bool, optional
        Disable the contribution from distant monopoles for PBC. True by
        default, to negate possible errors introduced by an almost-zero total
        charge.
    domain : (pmin, extent), optional
        Unit-cell specification in the corner-plus-extent convention shared with
        :func:`jaxfmm.kifmm.setup` / :func:`jaxfmm.uniform.setup` (``extent`` scalar
        or length-3).  Preferred over ``L0_boxcen``/``L0_boxlen`` (passing both
        raises).  Under ``periodic_axes`` this pins the periodic cell.
    L0_boxcen, L0_boxlen : jnp.ndarray, optional
        Centre / side lengths of the level-zero source box (useful for PBC).
        The default (None) generates the smallest possible axis-aligned box.
    no_cross_level : bool, optional
        For ``pts is eval_pts``, disable cross-level comparisons.
    debug_info : bool, optional
        Include further information in the hierarchy (box lengths, PBC
        shifts).
    mem_limit : float, optional
        Peak-transient device-memory budget in bytes for the evaluation (0.3.1
        semantics: the largest tensor a batched stage may materialise; tens of GB
        on a modern GPU). The default (None) picks the hardware-adaptive optimum
        (:func:`jaxfmm.near.auto_mem_limit`); minor stages derive their smaller
        working-set caps from it internally (:func:`jaxfmm.near.ws_limit`).
    jit_cnct : bool, optional
        Use the jit fixed-buffer connectivity (default since 0.3.3: its transients are
        bounded and it carries the int32 N_max auto-bump; ``False`` selects the eager
        exact-list builder — kept as the reference implementation / debugging fallback,
        with setup transients that GROW with N and no auto-bump protection)
        (:func:`gen_connectivity_jit` via :class:`AdaptiveConnectivity`).
    W, n_mpl, n_dir : int, optional
        Explicit fixed buffer sizes for the jit connectivity (raises on
        overflow instead of adapting).

    Returns
    -------
    dict
        Full hierarchy information (points, sorting indices, box centres,
        connectivity, Wigner tables, PBC operator, ...).
    """
    fmm_geom_scale = 1.0
    if auto_rescale:
        # Normalize the geometry to O(1) units (opt-in: consumers that mix their
        # own geometry with the stored hierarchy must keep frames themselves).
        # Exact: eval_potential restores the caller's units via fmm_geom_scale
        # (potential 1/s, field 1/s^2 for the Laplace kernel); bloch_k carries
        # units 1/length and scales inversely.
        import numpy as _rnp
        if domain is not None:
            _sc = float(_rnp.max(_rnp.asarray(domain[1])))
        elif L0_boxlen is not None:
            _sc = float(_rnp.max(_rnp.asarray(L0_boxlen)))
        else:
            _v = _rnp.asarray(pts[: min(len(pts), 65536)], dtype=_rnp.float64)
            _sc = float((_v.max(0) - _v.min(0)).max())
        if _rnp.isfinite(_sc) and _sc > 0:
            fmm_geom_scale = _sc
            pts = pts / _sc
            if eval_pts is not None:
                eval_pts = eval_pts / _sc
            if domain is not None:
                domain = (jnp.asarray(domain[0]) / _sc, jnp.asarray(domain[1]) / _sc)
            if L0_boxcen is not None:
                L0_boxcen = jnp.asarray(L0_boxcen) / _sc
            if L0_boxlen is not None:
                L0_boxlen = jnp.asarray(L0_boxlen) / _sc
            if basis_offsets is not None:
                basis_offsets = [jnp.asarray(o) / _sc for o in basis_offsets]
            if bloch_k is not None:
                bloch_k = jnp.asarray(bloch_k) * _sc
    from jaxfmm.util import validate_points, parse_domain
    validate_points(pts, "pts")
    L0_boxcen, L0_boxlen = parse_domain(domain, L0_boxcen, L0_boxlen, dtype=pts.dtype)
    if len(periodic_axes) > 0 and L0_boxlen is None:
        import warnings
        warnings.warn("gen_hierarchy: periodic_axes without a unit cell — using the point-cloud "
                      "AABB as the periodic cell. Pass domain=(pmin, extent) to pin it.",
                      stacklevel=2)
    if eval_pts is not None:
        validate_points(eval_pts, "eval_pts")
    if(eval_pts is None):
        eval_pts = pts
    periodic = len(periodic_axes) > 0
    sz_eps = 1.01   # safety factor for checking if coordinates are inside level zero boxes

    # int32 overflow auto-bump (symmetric, auto-sized jit connectivity): at extreme
    # N/N_max ratios the padded fixed buffers exceed int32-safe sizes. Predict from the
    # point count alone and double N_max until it fits — keeps small-N_max speedups
    # wherever they are representable. Disable with nmax_bump=False.  PERIODIC: the near
    # field replicates each box across n_img images, so the pair buffers grow ~n_img;
    # predict with the real image count (scale-invariant for a cubic root box — use
    # L0_boxlen when given) so the bump covers PBC too. The in-build check (line ~466)
    # and predict_cnct_sizes share _cnct_size_report, so a passing prediction guarantees
    # a passing check.
    import warnings
    _nmax_autobump = (kwargs.pop("nmax_bump", True) and jit_cnct and eval_pts is pts
                      and W is None and n_mpl is None and n_dir is None and not jax.config.jax_enable_x64)
    N_max_req = N_max
    if _nmax_autobump:
        # CHEAP FAST-START from an idealized uniform tree (open-BC image count): lets the
        # common case skip a doomed small-N_max build. Only a LOWER BOUND — clustered
        # geometry refines deeper and PBC replicates across images, so the actual in-build
        # int32 check in the loop below is the authoritative correctness backstop.
        while (predict_cnct_sizes(pts.shape[0], N_max, s)["worst"] > INT32_CNCT_LIMIT
               and N_max < pts.shape[0]):
            N_max *= 2

    while True:   # int32 connectivity-overflow auto-bump BACKSTOP: keys off the REAL built
        # tree (unlike the idealized-uniform pre-predict above, which under-shoots for
        # clustered geometry). On the in-build int32 check firing, double N_max and rebuild.
        # Geometry-agnostic (open/periodic, uniform/clustered); disabled -> the check just raises.
        if s == "auto":                     # per-tree schedules with an odd slab root: leaf
            from jaxfmm.trees.flexible import lane_schedule    # fill >= 89% at every N (the P2P
            s_src = lane_schedule(pts.shape[0], N_max, 2)      # eval-row axis is billed in 32/64
            s_eval = (s_src if eval_pts is pts                 # -lane multiples, ARCHITECTURE 7.30)
                      else lane_schedule(eval_pts.shape[0], N_max, 2))
        else:
            s_src = s_eval = s
        max_l = get_max_l(pts.shape[0], N_max, s_src)
        eval_max_l = get_max_l(eval_pts.shape[0], N_max, s_eval)
        idcs, rev_idcs, boxcenters, boxlens = balanced_tree(pts, max_l, s_src)

        # geometry canary: every box centre must be finite and inside the root box. A fused
        # gather+nan-reduce miscompilation can silently ZERO centres past a flat-padded-slot
        # boundary (worked around by sectioning, see trees.flexible._nanminmax_rows) — the
        # exact trigger lives inside XLA, so guard the invariant itself in case another
        # program shape re-awakens it.
        lo = boxcenters[0][0] - boxlens[0][0] * 0.5001
        hi = boxcenters[0][0] + boxlens[0][0] * 0.5001
        for _l, _c in enumerate(boxcenters):
            bad = ~(jnp.isfinite(_c).all(axis=1) & (_c >= lo).all(axis=1) & (_c <= hi).all(axis=1))
            if bool(bad.any()):
                raise RuntimeError(f"corrupt box centres at level {_l}: {int(bad.sum())} boxes "
                                   f"outside the root box / non-finite (first: box "
                                   f"{int(jnp.argmax(bad))}). This indicates a compiler-level "
                                   f"reduction bug (see an XLA fused gather+nan-reduce miscompile past flat slot 2^27; sectioned reduction avoids it")

        old_boxcen, old_boxlen = boxcenters[0].copy(), boxlens[0].copy()
        if(L0_boxcen is not None):  # overwrite with specified value
            boxcenters[0] = L0_boxcen[None,:]
        if(L0_boxlen is not None):  # overwrite with specified value
            boxlens[0] = L0_boxlen[None,:]
        if(jnp.any((old_boxcen[0] - old_boxlen[0]/2 - boxcenters[0]) < -sz_eps*boxlens[0]/2) or jnp.any((old_boxcen[0] + old_boxlen[0]/2 - boxcenters[0]) > sz_eps*boxlens[0]/2)):
            raise ValueError("Level zero box does not contain all source points.")

        if(pts is eval_pts):
            eval_idcs, eval_boxcenters, eval_boxlens = idcs, boxcenters, boxlens
        else:
            no_cross_level = False  # we must deal with cross-level contributions
            eval_idcs, _, eval_boxcenters, eval_boxlens = balanced_tree(eval_pts, eval_max_l, s_eval)

            if(periodic):
                if(jnp.any((eval_boxcenters[0] - eval_boxlens[0]/2 - boxcenters[0]) < -sz_eps*boxlens[0]/2) or jnp.any((eval_boxcenters[0] + eval_boxlens[0]/2 - boxcenters[0]) > sz_eps*boxlens[0]/2)):
                    raise ValueError("All evaluation points must be contained in the level zero source box for PBC.")
                eval_boxcenters[0], eval_boxlens[0] = boxcenters[0], boxlens[0]  # for PBC, we need identical level zero boxes

        try:
            if jit_cnct:
                if W is None and n_mpl is None and n_dir is None:   # auto-size the fixed buffers (robust; compiles cached per N)
                    conn = AdaptiveConnectivity(theta=theta, s=s_src, periodic_axes=periodic_axes, no_cross_level=no_cross_level, basis_offsets=basis_offsets)
                    boxcenters, eval_boxcenters, src_ofs, trg_ofs, mpl_cnct, dir_cnct, lvl_info, img_cnct = \
                        conn(boxcenters, boxlens, eval_boxcenters, eval_boxlens)
                else:                                               # explicit buffer sizes -> single jit call (raises on overflow)
                    boxcenters, eval_boxcenters, src_ofs, trg_ofs, mpl_cnct, dir_cnct, lvl_info, img_cnct = \
                        gen_connectivity_jit(boxcenters, boxlens, eval_boxcenters, eval_boxlens, theta, s_src, W, n_mpl, n_dir, periodic_axes, no_cross_level, basis_offsets=basis_offsets)
            else:
                boxcenters, eval_boxcenters, src_ofs, trg_ofs, mpl_cnct, dir_cnct, lvl_info, img_cnct = gen_connectivity(boxcenters, boxlens, eval_boxcenters, eval_boxlens, theta, s_src, no_cross_level, periodic_axes, basis_offsets=basis_offsets)
        except ValueError as e:
            if _nmax_autobump and "int32-safe limit" in str(e) and N_max < pts.shape[0]:
                N_max *= 2   # real tree overflowed int32; rebuild deeper-capped and retry
                continue
            raise
        break

    if N_max != N_max_req:
        warnings.warn(f"gen_hierarchy: N_max={N_max_req} would overflow int32 connectivity "
                      f"buffers at N={pts.shape[0]:.3g}" + (" (periodic)" if periodic else "")
                      + f"; auto-bumped to N_max={N_max}. Pass N_max>={N_max} to skip the "
                      f"rebuild retries, or nmax_bump=False / enable x64 to override.")

    ### adapt the sorting idcs to the "true" max levels - the if statements prevent unnecessary work
    only_dir = (len(lvl_info) == 1)
    idcs_list = [(idcs, rev_idcs) if only_dir else reduce_max_lvl(pts,idcs,s_src,max_l,lvl_info[-2][1])] # src padding at max mpl level

    if(only_dir or (pts is eval_pts and lvl_info[-2][0]==lvl_info[-2][1])):   # eval padding at max mpl level
        idcs_list.append(idcs_list[0])
    else:
        idcs_list.append(reduce_max_lvl(eval_pts,eval_idcs,s_eval,eval_max_l,lvl_info[-2][0]))

    if(only_dir or (lvl_info[-2][1]==lvl_info[-1][1])):                       # src padding at max dir level
        idcs_list.append(idcs_list[0])
    else:
        idcs_list.append(reduce_max_lvl(pts,idcs,s_src,max_l,lvl_info[-1][1]))

    if(only_dir or (lvl_info[-2][0]==lvl_info[-1][0])):                       # eval padding at max dir level
        idcs_list.append(idcs_list[1])
    elif(pts is eval_pts and lvl_info[-1][0]==lvl_info[-1][1]):
        idcs_list.append(idcs_list[2])
    else:
        idcs_list.append(reduce_max_lvl(eval_pts,eval_idcs,s_eval,eval_max_l,lvl_info[-1][0]))

    if(periodic):
        PBC_op, pbc_ws = gen_pbc_op(boxlens[0], theta, periodic_axes, p, pbc_lvls, pbc_no_monopole, basis_offsets=basis_offsets, bloch_k=bloch_k)
        if(pbc_lvls < 0): # optionally, only add non-ws boxes on level 0
            PBC_op = jnp.zeros(((p+1)**2,(p+1)**2))
        if(len(periodic_axes)==3 and not jnp.all(jnp.isclose(boxlens[0],boxlens[0][0,0]))):
            raise ValueError("3D PBC are currently only supported for cubic level zero boxes. "
                             "Pin a cubic cell with domain=(pmin, extent).")
    else:   # for open boundary conditions, we get no local expansion on level 0
        PBC_op = jnp.zeros(((p+1)**2,(p+1)**2))
        pbc_ws = []

    unqs, invs = precompute_combs(p)

    if mem_limit is not None:
        from jaxfmm.near import hint_legacy_mem_limit
        hint_legacy_mem_limit(mem_limit)             # pre-0.3.1 working-set values are ~6*N_max too small
    if mem_limit is None:
        # hardware-adaptive default in BUDGET units on every platform (0.3.2 fix: the GPU
        # branch previously stored the RAW L2 size — the pre-0.3.1 working-set unit —
        # which under-supplied the budget-pricing consumers by 6*N_max, ~9x slower P2P).
        try:
            from jaxfmm.near import auto_mem_limit
            mem_limit = auto_mem_limit(N_max, pts.dtype.itemsize)
            if jax.devices()[0].platform != "cpu":   # CPU: fused kernels never materialise the model
                print(f"[jaxfmm] mem_limit auto: {mem_limit/2**30:.1f} GiB peak-transient budget "
                      f"(speed-optimal; pass mem_limit=<bytes> to cap)")
        except Exception:
            mem_limit = jnp.inf
            print("Warning: mem_limit could not be determined automatically, setting mem_limit = jnp.inf.")

    hierarchy = {"pts": pts,
                 "eval_pts": eval_pts,
                 "idcs": idcs_list,
                 "boxcenters": boxcenters,
                 "eval_boxcenters": eval_boxcenters,
                 "src_ofs": src_ofs,
                 "trg_ofs": trg_ofs,
                 "mpl_cnct": mpl_cnct,
                 "dir_cnct": dir_cnct,
                 "unqs": unqs, "invs": invs,
                 "lvl_info": lvl_info,
                 "img_cnct": img_cnct,
                 "s": s_src,   # RESOLVED schedule (s="auto" -> per-level tuple): the string
                               # is not a JAX type and the dict gets **-splatted into jitted
                               # consumers (eval_potential); the tuple is a valid int pytree
                 "p": p,
                 "theta": theta,
                 "periodic_axes": periodic_axes,
                 "pbc_lvls": pbc_lvls if periodic else 0,
                 "pbc_no_monopole": pbc_no_monopole,
                 "PBC_op": PBC_op,
                 "mem_limit": mem_limit
                }
    import numpy as _bnp
    if bloch_k is not None and _bnp.any(_bnp.asarray(bloch_k) != 0.0):
        # per-image Bloch phase e^{ik.d} for the tree-level image machinery (P2P + M2L);
        # the lattice tail's phases were folded into PBC_op above
        hierarchy["img_phase"] = jnp.exp(1j * (img_cnct @ jnp.asarray(bloch_k, img_cnct.dtype).reshape(3)))
    # m-independent leaf-gathered geometry, hoisted out of the per-eval graph (see
    # pad_geometry: avoids XLA constant-folding stalls inside closed-over solver graphs)
    hierarchy["pad_geom"] = pad_geometry(pts, eval_pts, idcs_list)
    hierarchy["fmm_geom_scale"] = fmm_geom_scale
    if(debug_info):
        hierarchy["boxlens"] = jnp.concatenate(boxlens)
        hierarchy["eval_boxlens"] = jnp.concatenate(eval_boxlens)
        hierarchy["pbc_ws"] = pbc_ws
    return hierarchy

@jax.jit
def pad_geometry(pts, eval_pts, idcs):
    r"""The m-independent half of :func:`handle_padding`: leaf-gathered GEOMETRY arrays.

    Computed ONCE at setup (``gen_hierarchy`` stores it as ``hierarchy["pad_geom"]``) so
    per-evaluation graphs never gather the geometry. This matters inside closed-over
    solver graphs (``diffeqsolve``): there ``pts``/``idcs`` are compile-time constants,
    and a per-eval gather on constants gets CONSTANT-FOLDED by XLA — >1 s per fold on a
    multi-million-point mesh, repeated across the solver unroll, i.e. a multi-minute
    compile stall. Materialising the padded geometry up front removes the folds (and the
    per-eval gather cost) entirely; the charge gathers stay per-eval but act on traced
    values, which never fold.

    Returns
    -------
    tuple of jnp.ndarray
        ``(padded_pts, loc_padded_eval_pts, dir_padded_pts, dir_padded_eval_pts)``,
        zero-filled at padding, with identical index objects shared, not re-gathered.
    """
    padded_pts = pts.at[idcs[0][0]].get(mode="fill", fill_value=0.0)
    loc_padded_eval_pts = padded_pts if idcs[1] is idcs[0] else eval_pts.at[idcs[1][0]].get(mode="fill", fill_value=0.0)
    dir_padded_pts = padded_pts if idcs[2] is idcs[0] else pts.at[idcs[2][0]].get(mode="fill", fill_value=0.0)
    if(idcs[3] is idcs[2]):
        dir_padded_eval_pts = dir_padded_pts
    elif(idcs[3] is idcs[1]):
        dir_padded_eval_pts = loc_padded_eval_pts
    else:
        dir_padded_eval_pts = eval_pts.at[idcs[3][0]].get(mode="fill", fill_value=0.0)
    return padded_pts, loc_padded_eval_pts, dir_padded_pts, dir_padded_eval_pts


def handle_padding(pts, chrgs, eval_pts, idcs, pad_geom = None):
    r"""Generate the padded (leaf-gathered) arrays required for the potential evaluation.

    Parameters
    ----------
    pts : jnp.ndarray
        Source positions, shape ``(N, 3)``.
    chrgs : jnp.ndarray
        Source charges, shape ``(N,)``.
    eval_pts : jnp.ndarray
        Evaluation positions, shape ``(N_eval, 3)``.
    idcs : list of tuple
        The four ``(idcs, rev_idcs)`` paddings from ``gen_hierarchy`` (src /
        eval at the mpl and dir levels); identical objects are reused instead
        of re-gathered.
    pad_geom : tuple, optional
        Precomputed geometry padding from :func:`pad_geometry` (stored by
        ``gen_hierarchy`` as ``hierarchy["pad_geom"]``). When given, only the
        charge gathers run per call — see :func:`pad_geometry` for why this is
        essential inside solver graphs at scale.

    Returns
    -------
    tuple of jnp.ndarray
        ``(padded_pts, padded_chrgs, loc_padded_eval_pts, dir_padded_pts,
        dir_padded_chrgs, dir_padded_eval_pts)``, zero-filled at padding.
    """
    if pad_geom is None:
        pad_geom = pad_geometry(pts, eval_pts, idcs)
    padded_pts, loc_padded_eval_pts, dir_padded_pts, dir_padded_eval_pts = pad_geom
    padded_chrgs = chrgs.at[idcs[0][0]].get(mode="fill",fill_value=0.0)
    dir_padded_chrgs = padded_chrgs if idcs[2] is idcs[0] else chrgs.at[idcs[2][0]].get(mode="fill",fill_value=0.0)
    return padded_pts, padded_chrgs, loc_padded_eval_pts, dir_padded_pts, dir_padded_chrgs, dir_padded_eval_pts
