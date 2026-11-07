import argparse
import gc
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from HFMM import setup, evaluate
from jaxfmm.util import split_static, join_static

def direct(positions, charges, k):
    result = np.empty(len(positions), dtype=np.complex128)
    for start in range(0, len(positions), 256):
        x = positions[start:start + 256]
        r = np.linalg.norm(x[:, None, :] - positions[None, :, :], axis=-1)
        weights = np.where(r > 0, np.exp(1j * k * r) / np.where(r > 0, r, 1.), 0.)
        result[start:start + len(x)] = weights @ charges
    return result

def relative_error(actual, reference):
    norm = np.linalg.norm(reference)
    return float(np.linalg.norm(actual - reference) / (norm if norm else 1.))

def main():
    parser = argparse.ArgumentParser(description="Check HFMM accuracy and timing.")
    parser.add_argument("--n", type=int, nargs="+", default=[256, 640])
    parser.add_argument("--k", type=float, nargs="+", default=[0., 1., 3.])
    parser.add_argument("--p", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--leaf-size", type=int, default=16)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clustered", action="store_true")
    args = parser.parse_args()
    if min(args.n) < 2 or args.runs < 1:
        parser.error("N must be >=2 and runs must be >=1")
    print(f"JAX {jax.__version__} | {jax.devices()}", flush=True)
    print(f"{'N':>6} {'k':>6} {'p':>3} {'rel.error':>10} {'setup(s)':>9} {'eval(ms)':>9}", flush=True)
    cases = 0
    for n in args.n:
        rng = np.random.default_rng(args.seed + n)
        positions = rng.uniform(0., 1., (n, 3))
        if args.clustered:
            positions[:n // 2] = rng.uniform(.1, .25, (n // 2, 3))
        charges = rng.normal(size=n)
        q = jax.block_until_ready(jnp.asarray(charges, dtype=jnp.float64))
        for k in args.k:
            reference = direct(positions, charges, k)
            baseline = None
            for p in sorted(set(args.p)):
                start = perf_counter()
                state = setup(positions, k, p=p, N_max=args.leaf_size)
                skeleton, arrays = split_static(state)
                arrays = jax.block_until_ready(tuple(jax.device_put(a) for a in arrays))
                setup_time = perf_counter() - start
                compiled = jax.jit(lambda a, q: evaluate(join_static(skeleton, a), q))
                compiled(arrays, q).block_until_ready()
                elapsed = float("inf")
                for _ in range(args.runs):
                    start = perf_counter()
                    result = compiled(arrays, q).block_until_ready()
                    elapsed = min(elapsed, perf_counter() - start)
                actual = np.asarray(result)
                assert actual.dtype == np.complex128 and np.isfinite(actual).all()
                errors = np.array([relative_error(a, b) for a, b in
                                   ((actual, reference), (actual.real, reference.real), (actual.imag, reference.imag))])
                assert max(errors) < (5e-2 if p <= 3 else 5e-3), errors
                np.testing.assert_allclose(compiled(arrays, -.5 * q), -.5 * actual, rtol=1e-10, atol=1e-10)
                if k == 0:
                    assert np.count_nonzero(actual.imag) == 0
                if baseline is not None:
                    assert np.all(errors < np.maximum(.8 * baseline, 2e-6)), errors
                baseline = errors if baseline is None else baseline
                print(f"{n:6d} {k:6g} {p:3d} {errors[0]:10.2e} {setup_time:9.3f} {elapsed * 1000:9.3f}", flush=True)
                cases += 1
                jax.clear_caches()
                gc.collect()
    print(f"PASS: {cases} cases", flush=True)

if __name__ == "__main__":
    main()
