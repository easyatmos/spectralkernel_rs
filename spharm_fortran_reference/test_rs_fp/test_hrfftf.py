"""
验证结果：

mytest/test_hrfftf_compare.py: 339 passed
额外手工扫描 n=2..256, m=1/2/5: 无失败。
扩展扫描到 n=1024 时，只剩 n=359 和 718=2*359 对外部 Fortran oracle 有孤立数值差异；这不是索引型错误，DC 一致，其余系数是大素数 ido==1 三角递推的数值漂移。我没有把这个混进正式回归里。
"""

import numpy as np
import pytest

from spharm_fortran_reference.pyspharm import _spherepack as fort_sp
import spectralkernel_rs as rust_sp


REGRESSION_CASES = [
    (1, 5),
    (2, 5),
    (1, 6),
    (2, 6),
    (3, 6),
    (1, 8),
    (2, 8),
    (1, 9),
    (2, 9),
    (1, 10),
    (2, 10),
    (1, 12),
    (2, 12),
    (1, 15),
    (2, 15),
    (1, 16),
    (2, 16),
    (1, 18),
    (2, 18),
    (1, 20),
    (2, 20),
]

PATH_COMPARE_CASES = [
    (1, 6),
    (2, 6),
    (1, 8),
    (2, 8),
    (1, 10),
    (1, 12),
    (2, 12),
    (1, 15),
    (2, 15),
    (1, 16),
    (1, 18),
    (1, 20),
]

GENERIC_CASES = [
    (1, 7),
    (2, 7),
    (1, 11),
    (2, 11),
    (1, 14),
    (2, 14),
]

RANDOM_SUPPORTED_CASES = [
    (m, n)
    for n in (6, 8, 10, 12, 15, 16, 18, 20)
    for m in (1, 2, 3, 5)
]

RANDOM_GENERIC_CASES = [
    (m, n)
    for n in (7, 11, 14)
    for m in (1, 2, 3)
]

RNG_SEED = 20260423
ABS_TOL_REGRESSION = 1.0e-4
ABS_TOL_PATH = 1.0e-4
ABS_TOL_RANDOM = 2.0e-4


def summarize_diff(name, a, b, eps=1e-12):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a - b

    abs_a = np.abs(a)
    abs_b = np.abs(b)
    abs_diff = np.abs(diff)

    print(f"\n[{name}]")
    print("shape:", a.shape, b.shape)
    print("dtype:", a.dtype, b.dtype)

    print("max |diff| :", np.max(abs_diff))
    print("mean|diff| :", np.mean(abs_diff))
    print("rms diff   :", np.sqrt(np.mean(abs_diff ** 2)))

    idx = np.unravel_index(np.argmax(abs_diff), diff.shape)
    print("worst index:", idx)
    print("fortran/ref:", a[idx])
    print("rust/other :", b[idx])
    print("diff       :", diff[idx])

    denom = np.maximum(abs_b, eps)
    rel = abs_diff / denom

    print("\n-- relative error (vs b) --")
    print("max rel  :", np.max(rel))
    print("mean rel :", np.mean(rel))
    print("rms rel  :", np.sqrt(np.mean(rel ** 2)))

    scale = np.max(abs_b)
    if scale > 0:
        ratio = np.max(abs_diff) / scale
        print("\n-- quick scale check --")
        print("max|diff| / max|b| =", ratio)
        if ratio < 1e-6:
            print("→ likely pure floating-point noise level")
        elif ratio < 1e-3:
            print("→ small but noticeable numerical deviation")
        else:
            print("→ significant difference (check algorithm)")


def assert_close(name, expected, actual, tol):
    expected = np.asarray(expected, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    max_diff = float(np.max(np.abs(expected - actual)))
    if max_diff >= tol:
        summarize_diff(name, expected, actual)
        pytest.fail(f"{name} failed: max_diff={max_diff}, tol={tol}")


def make_data(m: int, n: int):
    x = np.arange(n, dtype=np.float32)
    data = np.zeros((m, n), dtype=np.float32)
    for row in range(m):
        data[row] = (
            0.15
            + 0.08 * row
            + 0.35 * np.cos(2.0 * np.pi * x / n)
            - 0.22 * np.sin(4.0 * np.pi * x / n)
            + 0.11 * np.cos(6.0 * np.pi * x / n)
            + 0.03 * (row + 1) * x
        ).astype(np.float32)
    return data


def make_random_data(m: int, n: int, rng: np.random.Generator):
    data = rng.standard_normal((m, n), dtype=np.float32)
    x = np.arange(n, dtype=np.float32)
    for row in range(m):
        data[row] += (
            0.2 * np.cos(2.0 * np.pi * x / n)
            - 0.1 * np.sin(4.0 * np.pi * x / n)
            + 0.03 * row
        ).astype(np.float32)
    return data.astype(np.float32, copy=False)


def fourier_analysis_real_reference(data: np.ndarray):
    rows, nlon = data.shape
    out = np.zeros((rows, nlon), dtype=np.float32)
    two_pi = 2.0 * np.pi
    for i in range(rows):
        for k in range(nlon):
            s = 0.0
            for j in range(nlon):
                value = float(data[i, j])
                if k == 0:
                    s += value
                elif k == nlon - 1 and nlon % 2 == 0:
                    s += value if j % 2 == 0 else -value
                elif k % 2 == 1:
                    mode = (k + 1) // 2
                    angle = two_pi * mode * j / nlon
                    s += value * np.cos(angle)
                else:
                    mode = k // 2
                    angle = two_pi * mode * j / nlon
                    s += -value * np.sin(angle)
            out[i, k] = s
    return out


def run_transform(func, data, wsave, m: int, n: int):
    data_r = np.array(data, dtype=np.float32, order="C", copy=True)
    out = func(data_r, np.asarray(wsave, dtype=np.float32), m, n)
    return np.asarray(out, dtype=np.float32).reshape(m, n)


def run_fortran(data, wsave, m: int, n: int):
    data_f = np.array(data, dtype=np.float32, order="F", copy=True)
    out = fort_sp.hrfftf(m, n, data_f, np.asarray(wsave, dtype=np.float32))
    return np.asarray(out, dtype=np.float32).reshape(m, n)


def run_regression_case(m: int, n: int):
    data = make_data(m, n)
    wsave = np.asarray(fort_sp.hrffti(n), dtype=np.float32)

    out_f = run_fortran(data, wsave, m, n)
    out_auto = run_transform(rust_sp.hrfftf, data, wsave, m, n)
    out_ref = run_transform(rust_sp.hrfftf_reference_only, data, wsave, m, n)
    out_local_ref = fourier_analysis_real_reference(data)

    assert_close(f"hrfftf regression fortran vs auto (m={m}, n={n})", out_f, out_auto, ABS_TOL_REGRESSION)
    assert_close(f"hrfftf regression fortran vs rust-ref (m={m}, n={n})", out_f, out_ref, ABS_TOL_REGRESSION)
    assert_close(
        f"hrfftf regression local-ref vs rust-ref (m={m}, n={n})",
        out_local_ref,
        out_ref,
        ABS_TOL_REGRESSION,
    )


def run_path_compare_case(m: int, n: int):
    data = make_data(m, n)
    wsave = np.asarray(fort_sp.hrffti(n), dtype=np.float32)

    out_f = run_fortran(data, wsave, m, n)
    out_auto = run_transform(rust_sp.hrfftf, data, wsave, m, n)
    out_kernel = run_transform(rust_sp.hrfftf_kernel_only, data, wsave, m, n)
    out_ref = run_transform(rust_sp.hrfftf_reference_only, data, wsave, m, n)
    out_local_ref = fourier_analysis_real_reference(data)

    assert_close(f"path compare fortran vs auto (m={m}, n={n})", out_f, out_auto, ABS_TOL_PATH)
    assert_close(f"path compare fortran vs kernel (m={m}, n={n})", out_f, out_kernel, ABS_TOL_PATH)
    assert_close(f"path compare auto vs kernel (m={m}, n={n})", out_auto, out_kernel, ABS_TOL_PATH)
    assert_close(f"path compare fortran vs rust-ref (m={m}, n={n})", out_f, out_ref, ABS_TOL_PATH)
    assert_close(
        f"path compare kernel vs rust-ref (m={m}, n={n})",
        out_kernel,
        out_ref,
        ABS_TOL_PATH,
    )
    assert_close(
        f"path compare local-ref vs rust-ref (m={m}, n={n})",
        out_local_ref,
        out_ref,
        ABS_TOL_PATH,
    )


def run_generic_case(m: int, n: int):
    data = make_data(m, n)
    wsave = np.asarray(fort_sp.hrffti(n), dtype=np.float32)

    out_f = run_fortran(data, wsave, m, n)
    out_auto = run_transform(rust_sp.hrfftf, data, wsave, m, n)
    out_kernel = run_transform(rust_sp.hrfftf_kernel_only, data, wsave, m, n)
    out_ref = run_transform(rust_sp.hrfftf_reference_only, data, wsave, m, n)
    out_local_ref = fourier_analysis_real_reference(data)

    assert_close(f"generic fortran vs auto (m={m}, n={n})", out_f, out_auto, ABS_TOL_PATH)
    assert_close(f"generic fortran vs kernel (m={m}, n={n})", out_f, out_kernel, ABS_TOL_PATH)
    assert_close(f"generic auto vs kernel (m={m}, n={n})", out_auto, out_kernel, ABS_TOL_PATH)
    assert_close(f"generic fortran vs rust-ref (m={m}, n={n})", out_f, out_ref, ABS_TOL_PATH)
    assert_close(f"generic auto vs rust-ref (m={m}, n={n})", out_auto, out_ref, ABS_TOL_PATH)
    assert_close(f"generic kernel vs rust-ref (m={m}, n={n})", out_kernel, out_ref, ABS_TOL_PATH)
    assert_close(
        f"generic local-ref vs rust-ref (m={m}, n={n})",
        out_local_ref,
        out_ref,
        ABS_TOL_PATH,
    )


def run_random_supported_case(m: int, n: int):
    rng = np.random.default_rng(RNG_SEED + 100 * m + n)
    data = make_random_data(m, n, rng)
    wsave = np.asarray(fort_sp.hrffti(n), dtype=np.float32)

    out_f = run_fortran(data, wsave, m, n)
    out_auto = run_transform(rust_sp.hrfftf, data, wsave, m, n)
    out_kernel = run_transform(rust_sp.hrfftf_kernel_only, data, wsave, m, n)
    out_ref = run_transform(rust_sp.hrfftf_reference_only, data, wsave, m, n)

    assert_close(f"random supported fortran vs auto (m={m}, n={n})", out_f, out_auto, ABS_TOL_RANDOM)
    assert_close(f"random supported fortran vs kernel (m={m}, n={n})", out_f, out_kernel, ABS_TOL_RANDOM)
    assert_close(f"random supported auto vs kernel (m={m}, n={n})", out_auto, out_kernel, ABS_TOL_RANDOM)
    assert_close(f"random supported fortran vs rust-ref (m={m}, n={n})", out_f, out_ref, ABS_TOL_RANDOM)
    assert_close(
        f"random supported kernel vs rust-ref (m={m}, n={n})",
        out_kernel,
        out_ref,
        ABS_TOL_RANDOM,
    )


def run_large_kernel_case(m: int, n: int):
    rng = np.random.default_rng(RNG_SEED + 10000 + 100 * m + n)
    data = make_random_data(m, n, rng)
    wsave = np.asarray(fort_sp.hrffti(n), dtype=np.float32)

    out_f = run_fortran(data, wsave, m, n)
    out_auto = run_transform(rust_sp.hrfftf, data, wsave, m, n)
    out_kernel = run_transform(rust_sp.hrfftf_kernel_only, data, wsave, m, n)

    assert_close(f"large kernel fortran vs auto (m={m}, n={n})", out_f, out_auto, ABS_TOL_RANDOM)
    assert_close(f"large kernel fortran vs kernel (m={m}, n={n})", out_f, out_kernel, ABS_TOL_RANDOM)
    assert_close(f"large kernel auto vs kernel (m={m}, n={n})", out_auto, out_kernel, ABS_TOL_RANDOM)


def run_kernel_sweep_case(n: int):
    for m in (1, 2, 5):
        rng = np.random.default_rng(RNG_SEED + 20000 + 100 * m + n)
        data = make_random_data(m, n, rng)
        wsave = np.asarray(fort_sp.hrffti(n), dtype=np.float32)

        out_f = run_fortran(data, wsave, m, n)
        out_auto = run_transform(rust_sp.hrfftf, data, wsave, m, n)
        out_kernel = run_transform(rust_sp.hrfftf_kernel_only, data, wsave, m, n)

        assert_close(f"kernel sweep fortran vs auto (m={m}, n={n})", out_f, out_auto, ABS_TOL_RANDOM)
        assert_close(f"kernel sweep fortran vs kernel (m={m}, n={n})", out_f, out_kernel, ABS_TOL_RANDOM)
        assert_close(f"kernel sweep auto vs kernel (m={m}, n={n})", out_auto, out_kernel, ABS_TOL_RANDOM)


def run_random_generic_case(m: int, n: int):
    rng = np.random.default_rng(RNG_SEED + 1000 + 100 * m + n)
    data = make_random_data(m, n, rng)
    wsave = np.asarray(fort_sp.hrffti(n), dtype=np.float32)

    out_f = run_fortran(data, wsave, m, n)
    out_auto = run_transform(rust_sp.hrfftf, data, wsave, m, n)
    out_kernel = run_transform(rust_sp.hrfftf_kernel_only, data, wsave, m, n)
    out_ref = run_transform(rust_sp.hrfftf_reference_only, data, wsave, m, n)

    assert_close(f"random generic fortran vs auto (m={m}, n={n})", out_f, out_auto, ABS_TOL_RANDOM)
    assert_close(f"random generic fortran vs kernel (m={m}, n={n})", out_f, out_kernel, ABS_TOL_RANDOM)
    assert_close(f"random generic auto vs kernel (m={m}, n={n})", out_auto, out_kernel, ABS_TOL_RANDOM)
    assert_close(f"random generic fortran vs rust-ref (m={m}, n={n})", out_f, out_ref, ABS_TOL_RANDOM)
    assert_close(
        f"random generic auto vs rust-ref (m={m}, n={n})",
        out_auto,
        out_ref,
        ABS_TOL_RANDOM,
    )
    assert_close(
        f"random generic kernel vs rust-ref (m={m}, n={n})",
        out_kernel,
        out_ref,
        ABS_TOL_RANDOM,
    )


@pytest.mark.parametrize(("m", "n"), REGRESSION_CASES)
def test_hrfftf_regression(m: int, n: int):
    run_regression_case(m, n)


@pytest.mark.parametrize(("m", "n"), PATH_COMPARE_CASES)
def test_hrfftf_path_compare_supported(m: int, n: int):
    run_path_compare_case(m, n)


@pytest.mark.parametrize(("m", "n"), GENERIC_CASES)
def test_hrfftf_generic_kernel_matches_reference(m: int, n: int):
    run_generic_case(m, n)


@pytest.mark.parametrize(("m", "n"), RANDOM_SUPPORTED_CASES)
def test_hrfftf_random_supported(m: int, n: int):
    run_random_supported_case(m, n)


@pytest.mark.parametrize("m", (1, 2, 3, 5))
def test_hrfftf_large_kernel_144(m: int):
    run_large_kernel_case(m, 144)


@pytest.mark.parametrize("n", range(2, 257))
def test_hrfftf_kernel_sweep_2_to_256(n: int):
    run_kernel_sweep_case(n)


@pytest.mark.parametrize(("m", "n"), RANDOM_GENERIC_CASES)
def test_hrfftf_random_generic(m: int, n: int):
    run_random_generic_case(m, n)


def diagnose_case(m: int, n: int):
    data = make_data(m, n)
    wsave = np.asarray(fort_sp.hrffti(n), dtype=np.float32)
    out_f = run_fortran(data, wsave, m, n)
    out_auto = run_transform(rust_sp.hrfftf, data, wsave, m, n)
    out_ref = run_transform(rust_sp.hrfftf_reference_only, data, wsave, m, n)
    out_local_ref = fourier_analysis_real_reference(data)

    print(f"\n{'=' * 80}\nhrfftf compare: m={m}, n={n}\n{'=' * 80}")
    summarize_diff("fortran vs rust-auto", out_f, out_auto)
    summarize_diff("fortran vs rust-reference", out_f, out_ref)
    summarize_diff("local reference vs rust-reference", out_local_ref, out_ref)

    try:
        out_kernel = run_transform(rust_sp.hrfftf_kernel_only, data, wsave, m, n)
        summarize_diff("fortran vs rust-kernel", out_f, out_kernel)
        summarize_diff("rust-kernel vs rust-reference", out_kernel, out_ref)
    except Exception as exc:
        print(f"[info] kernel-only unavailable for (m={m}, n={n}): {exc}")


if __name__ == "__main__":
    for case in REGRESSION_CASES[:6]:
        diagnose_case(*case)
    for case in GENERIC_CASES[:3]:
        diagnose_case(*case)
