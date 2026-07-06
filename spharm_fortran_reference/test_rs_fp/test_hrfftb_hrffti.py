import numpy as np

from spharm_fortran_reference.pyspharm import _spherepack as fort_sp
import spectralkernel_rs as rust_sp


REGRESSION_CASES = [
    (2, 2),
    (2, 3),
    (1, 4),
    (2, 5),
    (1, 6),
    (2, 6),
    (3, 6),
    (1, 8),
    (2, 8),
    (3, 8),
    (1, 10),
    (2, 10),
    (1, 12),
    (2, 12),
    (3, 12),
    (1, 14),
    (2, 14),
    (1, 15),
    (2, 15),
    (1, 16),
    (2, 16),
    (1, 18),
    (2, 18),
    (1, 20),
    (2, 20),
]

REFERENCE_FOCUS_CASES = [(1, 6), (1, 8), (1, 12), (1, 15), (1, 16), (1, 18), (1, 20)]
PATH_COMPARE_CASES = [
    (1, 6),
    (2, 6),
    (1, 8),
    (1, 10),
    (1, 12),
    (2, 12),
    (1, 14),
    (1, 15),
    (2, 15),
    (1, 16),
    (1, 18),
    (1, 20),
]
RANDOM_CASES = [(m, n) for n in (6, 8, 10, 12, 14, 15, 16, 18, 20) for m in (1, 2, 3, 5)]
RNG_SEED = 20260415
STABLE_RANDOM_CASES = [(m, n) for (m, n) in RANDOM_CASES if not (n % 2 == 1 and n > 5)]
EXPLORATORY_RANDOM_CASES = [(m, n) for (m, n) in RANDOM_CASES if (n % 2 == 1 and n > 5)]
EVEN_KERNEL_DIAGNOSTIC_CASES = [(m, n) for (m, n) in RANDOM_CASES if n in (6, 8, 10, 12, 14, 16, 18, 20)]
ODD_KERNEL_DIAGNOSTIC_CASES = [(m, n) for (m, n) in RANDOM_CASES if n in (15,)]


# def summarize_diff(name, a, b):
#     a = np.asarray(a)
#     b = np.asarray(b)
#     diff = a - b
#     print(f"\n[{name}]")
#     print("shape:", a.shape, b.shape)
#     print("dtype:", a.dtype, b.dtype)
#     print("max |diff| :", np.max(np.abs(diff)))
#     print("mean|diff| :", np.mean(np.abs(diff)))
#     print("rms diff   :", np.sqrt(np.mean(np.abs(diff) ** 2)))
#     idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
#     print("worst index:", idx)
#     print("fortran    :", a[idx])
#     print("rust       :", b[idx])
#     print("diff       :", diff[idx])
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

    # --- 原有误差 ---
    print("max |diff| :", np.max(abs_diff))
    print("mean|diff| :", np.mean(abs_diff))
    print("rms diff   :", np.sqrt(np.mean(abs_diff ** 2)))

    idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    print("worst index:", idx)
    print("fortran    :", a[idx])
    print("rust       :", b[idx])
    print("diff       :", diff[idx])

    # --- 新增：量级 ---
    # print("\n-- magnitude of a --")
    # print("max |a| :", np.max(abs_a))
    # print("mean|a| :", np.mean(abs_a))
    # print("rms  |a|:", np.sqrt(np.mean(abs_a ** 2)))

    # print("\n-- magnitude of b --")
    # print("max |b| :", np.max(abs_b))
    # print("mean|b| :", np.mean(abs_b))
    # print("rms  |b|:", np.sqrt(np.mean(abs_b ** 2)))

    # --- 相对误差（整体）---
    denom = np.maximum(abs_b, eps)
    rel = abs_diff / denom

    print("\n-- relative error (vs b) --")
    print("max rel  :", np.max(rel))
    print("mean rel :", np.mean(rel))
    print("rms rel  :", np.sqrt(np.mean(rel ** 2)))

    # --- 一个快速判断 ---
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


def make_coeffs(m: int, n: int):
    coeffs = np.zeros((m, n), dtype=np.float32)
    for row in range(m):
        coeffs[row, 0] = 1.0 + 0.25 * row
        if n >= 3:
            coeffs[row, 1] = 0.5 * (row + 1)
            coeffs[row, 2] = -0.125 * (row + 1)
        if n >= 5:
            coeffs[row, 3] = 0.2 * (row + 1)
            coeffs[row, 4] = 0.1 * (row + 2)
        if n % 2 == 0:
            coeffs[row, -1] = (-1.0) ** row * 0.3
    return coeffs


def make_random_coeffs(m: int, n: int, rng: np.random.Generator):
    coeffs = rng.standard_normal((m, n), dtype=np.float32)
    if n >= 2:
        coeffs[:, 0] += 0.5
    return coeffs.astype(np.float32, copy=False)


def run_transform(func, coeffs, wsave, m: int, n: int):
    coeffs_r = np.array(coeffs, dtype=np.float32, order="C", copy=True)
    out = func(coeffs_r, np.asarray(wsave, dtype=np.float32), m, n)
    return np.asarray(out).reshape(m, n)


def run_case(m: int, n: int):
    coeffs = make_coeffs(m, n)
    wsave = fort_sp.hrffti(n)
    coeffs_f = np.array(coeffs, dtype=np.float32, order="F", copy=True)
    coeffs_r = np.array(coeffs, dtype=np.float32, order="C", copy=True)

    out_f = fort_sp.hrfftb(m, n, coeffs_f, np.asarray(wsave, dtype=np.float32))
    out_r = rust_sp.hrfftb(coeffs_r, np.asarray(wsave, dtype=np.float32), m, n)
    out_r_c = np.asarray(out_r).reshape(m, n)

    print(f"\n{'=' * 80}\nhrfftb: m={m}, n={n}\n{'=' * 80}")
    if m == 1 and n in (6, 8, 12, 15, 16, 18, 20):
        print("fortran raw row:", np.asarray(out_f).reshape(m, n)[0])
        print("rust raw 1d   :", np.asarray(out_r))
        print("rust C row    :", out_r_c[0])
        print("fortran odd idx:", np.asarray(out_f).reshape(m, n)[0][::2])
        print("fortran even idx:", np.asarray(out_f).reshape(m, n)[0][1::2])
        print("rust odd idx   :", out_r_c[0][::2])
        print("rust even idx  :", out_r_c[0][1::2])
    print("note: hrfftb correctness is validated against Rust C-order reshape only")
    summarize_diff("hrfftb fortran vs rust (C reshape)", out_f, out_r_c)
    max_diff = float(np.max(np.abs(np.asarray(out_f) - out_r_c)))
    assert max_diff < 1.0e-5, f"hrfftb regression failed for (m={m}, n={n}), max_diff={max_diff}"


def run_reference_case(m: int, n: int):
    coeffs = make_coeffs(m, n)
    wsave = np.asarray(fort_sp.hrffti(n), dtype=np.float32)
    coeffs_f = np.array(coeffs, dtype=np.float32, order="F", copy=True)
    coeffs_r = np.array(coeffs, dtype=np.float32, order="C", copy=True)

    out_f = fort_sp.hrfftb(m, n, coeffs_f, wsave)
    out_r = rust_sp.hrfftb(coeffs_r, wsave, m, n)
    out_r_c = np.asarray(out_r).reshape(m, n)

    print(f"\n{'-' * 80}\nreference-focus hrfftb: m={m}, n={n}\n{'-' * 80}")
    print("fortran row 0:", np.asarray(out_f).reshape(m, n)[0])
    print("rust row 0   :", out_r_c[0])
    summarize_diff("reference-focus fortran vs rust (C reshape)", out_f, out_r_c)
    max_diff = float(np.max(np.abs(np.asarray(out_f) - out_r_c)))
    assert max_diff < 1.0e-5, f"hrfftb reference-focus failed for (m={m}, n={n}), max_diff={max_diff}"


def run_path_compare_case(m: int, n: int):
    coeffs = make_coeffs(m, n)
    wsave = np.asarray(fort_sp.hrffti(n), dtype=np.float32)
    coeffs_f = np.array(coeffs, dtype=np.float32, order="F", copy=True)

    out_f = fort_sp.hrfftb(m, n, coeffs_f, wsave)
    out_auto = run_transform(rust_sp.hrfftb, coeffs, wsave, m, n)
    out_kernel = run_transform(rust_sp.hrfftb_kernel_only, coeffs, wsave, m, n)
    out_ref = run_transform(rust_sp.hrfftb_reference_only, coeffs, wsave, m, n)

    print(f"\n{'*' * 80}\npath-compare hrfftb: m={m}, n={n}\n{'*' * 80}")
    summarize_diff("fortran vs rust-auto", out_f, out_auto)
    summarize_diff("fortran vs rust-kernel", out_f, out_kernel)
    summarize_diff("fortran vs rust-reference", out_f, out_ref)
    summarize_diff("rust-kernel vs rust-reference", out_kernel, out_ref)


def run_random_cases(cases, *, enforce_assert: bool, label: str):
    rng = np.random.default_rng(RNG_SEED)
    print(f"\n{'#' * 80}\n{label} seed={RNG_SEED}\n{'#' * 80}")
    for m, n in RANDOM_CASES:
        coeffs = make_random_coeffs(m, n, rng)
        if (m, n) not in cases:
            continue

        wsave = np.asarray(fort_sp.hrffti(n), dtype=np.float32)
        coeffs_f = np.array(coeffs, dtype=np.float32, order="F", copy=True)
        out_f = fort_sp.hrfftb(m, n, coeffs_f, wsave)
        out_auto = run_transform(rust_sp.hrfftb, coeffs, wsave, m, n)
        max_diff = float(np.max(np.abs(np.asarray(out_f) - out_auto)))
        rms_diff = float(np.sqrt(np.mean((np.asarray(out_f) - out_auto) ** 2)))
        print(f"random case (m={m}, n={n}) max_diff={max_diff:.8e} rms_diff={rms_diff:.8e}")
        if enforce_assert:
            assert max_diff < 1.0e-4, f"random regression failed for (m={m}, n={n}), max_diff={max_diff}"


def run_kernel_random_cases(cases, *, enforce_assert: bool, label: str):
    rng = np.random.default_rng(RNG_SEED)
    print(f"\n{'%' * 80}\n{label} seed={RNG_SEED}\n{'%' * 80}")
    worst_by_n = {}
    for m, n in RANDOM_CASES:
        coeffs = make_random_coeffs(m, n, rng)
        if (m, n) not in cases:
            continue

        wsave = np.asarray(fort_sp.hrffti(n), dtype=np.float32)
        coeffs_f = np.array(coeffs, dtype=np.float32, order="F", copy=True)
        out_f = fort_sp.hrfftb(m, n, coeffs_f, wsave)
        out_kernel = run_transform(rust_sp.hrfftb_kernel_only, coeffs, wsave, m, n)
        max_diff = float(np.max(np.abs(np.asarray(out_f) - out_kernel)))
        rms_diff = float(np.sqrt(np.mean((np.asarray(out_f) - out_kernel) ** 2)))
        print(f"kernel random case (m={m}, n={n}) max_diff={max_diff:.8e} rms_diff={rms_diff:.8e}")
        worst_by_n[n] = max(worst_by_n.get(n, 0.0), max_diff)
        if enforce_assert:
            assert max_diff < 1.0e-4, f"kernel random regression failed for (m={m}, n={n}), max_diff={max_diff}"

    for n in sorted(worst_by_n):
        print(f"kernel random summary n={n} worst_max_diff={worst_by_n[n]:.8e}")


def run_random_regression():
    run_random_cases(
        STABLE_RANDOM_CASES,
        enforce_assert=True,
        label="stable random hrfftb regression",
    )


def run_kernel_random_diagnostics():
    run_kernel_random_cases(
        EVEN_KERNEL_DIAGNOSTIC_CASES,
        enforce_assert=False,
        label="even kernel-only hrfftb diagnostics",
    )
    run_kernel_random_cases(
        ODD_KERNEL_DIAGNOSTIC_CASES,
        enforce_assert=False,
        label="odd kernel-only hrfftb diagnostics",
    )


def run_exploratory_path_compare_case(m: int, n: int):
    coeffs = make_random_coeffs(m, n, np.random.default_rng(RNG_SEED + 99))
    wsave = np.asarray(fort_sp.hrffti(n), dtype=np.float32)
    coeffs_f = np.array(coeffs, dtype=np.float32, order="F", copy=True)

    out_f = fort_sp.hrfftb(m, n, coeffs_f, wsave)
    out_auto = run_transform(rust_sp.hrfftb, coeffs, wsave, m, n)
    out_kernel = run_transform(rust_sp.hrfftb_kernel_only, coeffs, wsave, m, n)
    out_ref = run_transform(rust_sp.hrfftb_reference_only, coeffs, wsave, m, n)

    print(f"\n{'!' * 80}\nexploratory path-compare hrfftb: m={m}, n={n}\n{'!' * 80}")
    summarize_diff("fortran vs rust-auto", out_f, out_auto)
    summarize_diff("fortran vs rust-kernel", out_f, out_kernel)
    summarize_diff("fortran vs rust-reference", out_f, out_ref)
    summarize_diff("rust-kernel vs rust-reference", out_kernel, out_ref)
    print("fortran row 0:", np.asarray(out_f).reshape(m, n)[0])
    print("rust auto row0:", out_auto[0])
    print("rust kernel row0:", out_kernel[0])
    print("rust ref row0:", out_ref[0])


def run_exploratory_random_regression():
    run_random_cases(
        EXPLORATORY_RANDOM_CASES,
        enforce_assert=False,
        label="exploratory random hrfftb diagnostics",
    )


def make_single_mode_coeffs(n: int, mode_index: int, amplitude: float = 1.0):
    coeffs = np.zeros((1, n), dtype=np.float32)
    coeffs[0, mode_index] = np.float32(amplitude)
    return coeffs


def run_single_mode_cross_check(n: int, mode_index: int, amplitude: float = 1.0):
    coeffs = make_single_mode_coeffs(n, mode_index, amplitude)
    wsave = np.asarray(fort_sp.hrffti(n), dtype=np.float32)
    coeffs_f = np.array(coeffs, dtype=np.float32, order="F", copy=True)

    out_f = np.asarray(fort_sp.hrfftb(1, n, coeffs_f, wsave), dtype=np.float32).reshape(1, n)
    out_auto = run_transform(rust_sp.hrfftb, coeffs, wsave, 1, n)
    out_ref = run_transform(rust_sp.hrfftb_reference_only, coeffs, wsave, 1, n)

    print(f"\n{'~' * 80}\nsingle-mode hrfftb: n={n}, mode_index={mode_index}, amplitude={amplitude}\n{'~' * 80}")
    print("coeff row      :", coeffs[0])
    print("fortran row    :", out_f[0])
    print("rust auto row  :", out_auto[0])
    print("rust ref row   :", out_ref[0])
    summarize_diff("single-mode fortran vs rust-auto", out_f, out_auto)
    summarize_diff("single-mode fortran vs rust-ref", out_f, out_ref)


def run_single_mode_suite():
    for n in (8, 16):
        mode_indices = [0, 1, 2, 3]
        if n % 2 == 0:
            mode_indices.append(n - 1)
        for mode_index in mode_indices:
            run_single_mode_cross_check(n, mode_index)


def make_physical_rows(m: int, n: int):
    x = np.arange(n, dtype=np.float32)
    out = np.zeros((m, n), dtype=np.float32)
    for row in range(m):
        out[row] = (
            0.35
            + 0.1 * row
            + 0.6 * np.cos(2.0 * np.pi * x / n)
            - 0.4 * np.sin(4.0 * np.pi * x / n)
            + 0.2 * np.cos(6.0 * np.pi * x / n)
        ).astype(np.float32)
    return out


def run_fortran_semantic_synthesis_check(m: int, n: int):
    data = make_physical_rows(m, n)
    wsave = np.asarray(fort_sp.hrffti(n), dtype=np.float32)
    coeffs_f = np.asarray(fort_sp.hrfftf(m, n, np.array(data, order="F", copy=True), wsave), dtype=np.float32).reshape(m, n)

    out_f = np.asarray(
        fort_sp.hrfftb(m, n, np.array(coeffs_f, dtype=np.float32, order="F", copy=True), wsave),
        dtype=np.float32,
    ).reshape(m, n)
    out_auto = run_transform(rust_sp.hrfftb, coeffs_f, wsave, m, n)
    out_kernel = run_transform(rust_sp.hrfftb_kernel_only, coeffs_f, wsave, m, n)

    print(f"\n{'=' * 80}\nfortran-semantic hrfftb: m={m}, n={n}\n{'=' * 80}")
    print("physical row 0      :", data[0])
    print("fortran coeff row 0 :", coeffs_f[0])
    print("fortran synth row 0 :", out_f[0])
    print("rust auto synth row0:", out_auto[0])
    print("rust kern synth row0:", out_kernel[0])
    summarize_diff("fortran-semantic fortran vs rust-auto", out_f, out_auto)
    summarize_diff("fortran-semantic fortran vs rust-kernel", out_f, out_kernel)


def run_fortran_semantic_suite():
    for case in [(1, 8), (1, 16), (2, 16), (2, 144)]:
        run_fortran_semantic_synthesis_check(*case)


if __name__ == "__main__":
    for case in REGRESSION_CASES:
        run_case(*case)
    for case in REFERENCE_FOCUS_CASES:
        run_reference_case(*case)
    for case in PATH_COMPARE_CASES:
        run_path_compare_case(*case)
    run_random_regression()
    run_kernel_random_diagnostics()
    run_exploratory_path_compare_case(1, 15)
    run_exploratory_random_regression()
    run_single_mode_suite()
    run_fortran_semantic_suite()
