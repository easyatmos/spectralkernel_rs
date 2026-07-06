import math
import numpy as np

import spectralkernel_rs as rust_sp
from spharm_fortran_reference.pyspharm import _spherepack as fort_sp


# =========================================================
# 从 spharm.py 抽出的辅助函数
# =========================================================
def getspecindx(ntrunc: int):
    indexn = np.indices((ntrunc + 1, ntrunc + 1))[1, :, :]
    indexm = np.indices((ntrunc + 1, ntrunc + 1))[0, :, :]
    indices = np.nonzero(np.greater(indexn, indexm - 1).flatten())
    indxn = np.take(indexn.flatten(), indices)
    indxm = np.take(indexm.flatten(), indices)
    return np.atleast_1d(np.squeeze(indxm)), np.atleast_1d(np.squeeze(indxn))


def infer_ntrunc_from_nspec(nspec: int) -> int:
    ntrunc = int(-1.5 + 0.5 * math.sqrt(9.0 - 8.0 * (1.0 - nspec)))
    if (ntrunc + 1) * (ntrunc + 2) // 2 != nspec:
        raise ValueError(f"Invalid spectral size: {nspec}")
    return ntrunc


def multsmoothfact_from_backend(sp, dataspec, smooth):
    out = np.asarray(sp.multsmoothfact(dataspec, smooth))
    if np.asarray(dataspec).ndim == 1:
        return np.atleast_1d(np.squeeze(out))
    return out


# =========================================================
# 通用工具
# =========================================================
def summarize_diff(name, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a - b

    print(f"\n[{name}]")
    print("shape:", a.shape, b.shape)
    print("dtype:", a.dtype, b.dtype)
    print("max |diff| :", np.max(np.abs(diff)))
    print("mean|diff| :", np.mean(np.abs(diff)))
    print("rms diff   :", np.sqrt(np.mean(np.abs(diff) ** 2)))

    idx = np.argmax(np.abs(diff))
    print("worst index :", idx)
    print("A value     :", a.reshape(-1)[idx])
    print("B value     :", b.reshape(-1)[idx])
    print("diff        :", diff.reshape(-1)[idx])


def summarize_theory_error(name, actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    diff = actual - expected

    print(f"\n[{name}]")
    print("max |actual-expected| :", np.max(np.abs(diff)))
    print("mean|actual-expected| :", np.mean(np.abs(diff)))
    print("rms diff              :", np.sqrt(np.mean(np.abs(diff) ** 2)))

    idx = np.argmax(np.abs(diff))
    print("worst index :", idx)
    print("actual      :", actual.reshape(-1)[idx])
    print("expected    :", expected.reshape(-1)[idx])
    print("diff        :", diff.reshape(-1)[idx])


def make_random_dataspec(ntrunc: int, seed=123, scale=1e-1, dtype=np.complex64):
    ncoeff = (ntrunc + 1) * (ntrunc + 2) // 2
    rng = np.random.default_rng(seed)
    real = rng.normal(scale=scale, size=ncoeff)
    imag = rng.normal(scale=scale, size=ncoeff)
    out = (real + 1j * imag).astype(dtype)
    out[0] = dtype(1.0 + 0.0j)
    return out


def make_single_mode_dataspec(ntrunc: int, m: int, n: int, value=1.0 + 0.5j, dtype=np.complex64):
    ncoeff = (ntrunc + 1) * (ntrunc + 2) // 2
    dataspec = np.zeros(ncoeff, dtype=dtype)
    indxm, indxn = getspecindx(ntrunc)
    idx = np.where((indxm == m) & (indxn == n))[0]
    if len(idx) != 1:
        raise ValueError(f"Cannot find unique spectral index for (m={m}, n={n})")
    dataspec[idx[0]] = dtype(value)
    return dataspec


def make_smooth_identity(nlat: int, dtype=np.float32):
    return np.ones(nlat, dtype=dtype)


def make_smooth_zero(nlat: int, dtype=np.float32):
    return np.zeros(nlat, dtype=dtype)


def make_smooth_linear_decay(nlat: int, dtype=np.float32):
    """
    n=0 保留 1，高波数线性衰减到 0
    """
    if nlat == 1:
        return np.ones(1, dtype=dtype)
    x = np.linspace(1.0, 0.0, nlat, dtype=np.float64)
    return x.astype(dtype)


def make_smooth_gaussian_like(nlat: int, alpha=8.0, dtype=np.float32):
    """
    一个比较常见的平滑因子例子：exp(-alpha*(n/(nlat-1))^2)
    """
    if nlat == 1:
        return np.ones(1, dtype=dtype)
    n = np.arange(nlat, dtype=np.float64)
    x = n / float(nlat - 1)
    s = np.exp(-alpha * x * x)
    return s.astype(dtype)


# =========================================================
# multsmoothfact 理论公式
# =========================================================
def expected_multsmoothfact_theory(dataspec, smooth):
    """
    multsmoothfact 的理论行为：
    每个谱系数根据其总波数 n，乘以 smooth[n]。
    """
    dataspec = np.asarray(dataspec)
    ntrunc = infer_ntrunc_from_nspec(dataspec.shape[0])
    _, indxn = getspecindx(ntrunc)

    if smooth.ndim != 1:
        raise ValueError("smooth must be rank-1")
    if smooth.shape[0] < ntrunc + 1:
        raise ValueError("smooth length too small for dataspec truncation")

    factor = np.asarray(smooth, dtype=np.float64)[indxn]
    return (dataspec.astype(np.complex128) * factor.astype(np.float64)).astype(np.complex128)


# =========================================================
# 1) 随机谱场：Fortran vs Rust + 理论比较
# =========================================================
def check_multsmoothfact_random(
    ntrunc_list=(0, 1, 3, 5, 8, 12),
    smooth_kind="gaussian",
    seed=123
):
    print("\n" + "=" * 80)
    print("MULTSMOOTHFACT COMPARISON ON RANDOM SPECTRA")
    print("=" * 80)

    for ntrunc in ntrunc_list:
        print(f"\n-------------------- ntrunc = {ntrunc} --------------------")

        dataspec = make_random_dataspec(ntrunc=ntrunc, seed=seed + ntrunc, scale=1e-1)
        nlat = ntrunc + 1

        if smooth_kind == "identity":
            smooth = make_smooth_identity(nlat)
        elif smooth_kind == "zero":
            smooth = make_smooth_zero(nlat)
        elif smooth_kind == "linear":
            smooth = make_smooth_linear_decay(nlat)
        else:
            smooth = make_smooth_gaussian_like(nlat, alpha=8.0)

        out_f = multsmoothfact_from_backend(fort_sp, dataspec, smooth)
        out_r = multsmoothfact_from_backend(rust_sp, dataspec, smooth)

        summarize_diff("multsmoothfact(fortran) vs multsmoothfact(rust)", out_f, out_r)

        expected = expected_multsmoothfact_theory(dataspec, smooth)
        summarize_theory_error("multsmoothfact(fortran) vs theory", out_f, expected)
        summarize_theory_error("multsmoothfact(rust) vs theory", out_r, expected)


# =========================================================
# 2) 单模态检查
# =========================================================
def check_multsmoothfact_single_modes():
    print("\n" + "=" * 80)
    print("MULTSMOOTHFACT COMPARISON ON SINGLE MODES")
    print("=" * 80)

    cases = [
        # (ntrunc, m, n, value)
        (0, 0, 0, 1.0 + 0.0j),
        (1, 0, 1, 1.0 + 0.0j),
        (3, 1, 1, 2.0 - 0.5j),
        (5, 0, 2, -1.2 + 0.3j),
        (5, 2, 3, 0.75 + 0.25j),
        (8, 3, 5, -0.4 + 1.1j),
        (10, 4, 7, 0.2 - 0.9j),
    ]

    for ntrunc, m, n, value in cases:
        print(f"\n-------------------- ntrunc={ntrunc}, m={m}, n={n} --------------------")

        dataspec = make_single_mode_dataspec(ntrunc, m, n, value=value)

        # 给一个能明显看出缩放的 smooth
        smooth = make_smooth_linear_decay(ntrunc + 1)

        out_f = multsmoothfact_from_backend(fort_sp, dataspec, smooth)
        out_r = multsmoothfact_from_backend(rust_sp, dataspec, smooth)
        expected = expected_multsmoothfact_theory(dataspec, smooth)

        summarize_diff("multsmoothfact(fortran) vs multsmoothfact(rust)", out_f, out_r)
        summarize_theory_error("multsmoothfact(fortran) vs theory", out_f, expected)
        summarize_theory_error("multsmoothfact(rust) vs theory", out_r, expected)

        indxm, indxn = getspecindx(ntrunc)
        idx = np.where((indxm == m) & (indxn == n))[0][0]

        print("\n[mode detail]")
        print("input coeff        =", dataspec[idx])
        print("smooth[n]          =", smooth[n])
        print("expected output    =", expected[idx])
        print("fortran output     =", out_f[idx])
        print("rust output        =", out_r[idx])


# =========================================================
# 3) 特殊 smooth 检查
# =========================================================
def check_multsmoothfact_special_smooths(ntrunc=8, seed=202):
    print("\n" + "=" * 80)
    print("MULTSMOOTHFACT SPECIAL SMOOTH CHECKS")
    print("=" * 80)

    dataspec = make_random_dataspec(ntrunc=ntrunc, seed=seed, scale=1e-1)
    nlat = ntrunc + 1

    cases = [
        ("identity", make_smooth_identity(nlat)),
        ("zero", make_smooth_zero(nlat)),
        ("linear", make_smooth_linear_decay(nlat)),
        ("gaussian", make_smooth_gaussian_like(nlat, alpha=8.0)),
    ]

    for name, smooth in cases:
        print(f"\n-------------------- smooth = {name} --------------------")

        out_f = multsmoothfact_from_backend(fort_sp, dataspec, smooth)
        out_r = multsmoothfact_from_backend(rust_sp, dataspec, smooth)
        expected = expected_multsmoothfact_theory(dataspec, smooth)

        summarize_diff("fortran vs rust", out_f, out_r)
        summarize_theory_error("fortran vs theory", out_f, expected)
        summarize_theory_error("rust vs theory", out_r, expected)

        if name == "identity":
            summarize_theory_error("identity check (fortran output vs input)", out_f, dataspec)
            summarize_theory_error("identity check (rust output vs input)", out_r, dataspec)

        if name == "zero":
            summarize_theory_error("zero check (fortran output vs 0)", out_f, np.zeros_like(out_f))
            summarize_theory_error("zero check (rust output vs 0)", out_r, np.zeros_like(out_r))


# =========================================================
# 4) 组合性质检查：两次平滑应等价于因子逐点相乘
# =========================================================
def check_multsmoothfact_composition(ntrunc_list=(1, 3, 5, 8, 12), seed=456):
    print("\n" + "=" * 80)
    print("MULTSMOOTHFACT COMPOSITION CHECK")
    print("=" * 80)

    for ntrunc in ntrunc_list:
        print(f"\n-------------------- ntrunc = {ntrunc} --------------------")

        dataspec = make_random_dataspec(ntrunc=ntrunc, seed=seed + ntrunc, scale=1e-1)
        nlat = ntrunc + 1

        smooth1 = make_smooth_linear_decay(nlat)
        smooth2 = make_smooth_gaussian_like(nlat, alpha=4.0)
        smooth12 = (smooth1.astype(np.float64) * smooth2.astype(np.float64)).astype(np.float32)

        # Fortran
        out_f_2step = multsmoothfact_from_backend(
            fort_sp,
            multsmoothfact_from_backend(fort_sp, dataspec, smooth1),
            smooth2,
        )
        out_f_1step = multsmoothfact_from_backend(fort_sp, dataspec, smooth12)

        # Rust
        out_r_2step = multsmoothfact_from_backend(
            rust_sp,
            multsmoothfact_from_backend(rust_sp, dataspec, smooth1),
            smooth2,
        )
        out_r_1step = multsmoothfact_from_backend(rust_sp, dataspec, smooth12)

        summarize_theory_error("fortran composition: 2-step vs 1-step", out_f_2step, out_f_1step)
        summarize_theory_error("rust composition: 2-step vs 1-step", out_r_2step, out_r_1step)
        summarize_diff("fortran vs rust (2-step)", out_f_2step, out_r_2step)
        summarize_diff("fortran vs rust (1-step)", out_f_1step, out_r_1step)


# =========================================================
# 5) 单位因子打印：看每个谱系数对应哪个 smooth[n]
# =========================================================
def inspect_multsmoothfact_factors(ntrunc=5):
    print("\n" + "=" * 80)
    print("INSPECT MULTSMOOTHFACT FACTORS")
    print("=" * 80)

    indxm, indxn = getspecindx(ntrunc)
    smooth = make_smooth_linear_decay(ntrunc + 1)

    print(f"{'idx':>4} {'m':>4} {'n':>4} {'smooth[n]':>16}")
    for i in range(len(indxn)):
        print(f"{i:4d} {int(indxm[i]):4d} {int(indxn[i]):4d} {float(smooth[indxn[i]]):16.8f}")


# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    check_multsmoothfact_random(
        ntrunc_list=(0, 1, 3, 5, 8, 12),
        smooth_kind="gaussian",
        seed=123,
    )

    check_multsmoothfact_single_modes()

    check_multsmoothfact_special_smooths(
        ntrunc=8,
        seed=202,
    )

    check_multsmoothfact_composition(
        ntrunc_list=(1, 3, 5, 8, 12),
        seed=456,
    )

    inspect_multsmoothfact_factors(ntrunc=5)
