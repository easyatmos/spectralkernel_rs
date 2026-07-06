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


def invlap_from_backend(sp, dataspec, rsphere: float):
    out = np.asarray(sp.invlap(dataspec, rsphere))
    if np.asarray(dataspec).ndim == 1:
        return np.atleast_1d(np.squeeze(out))
    return out


def lap_from_backend(sp, dataspec, rsphere: float):
    out = np.asarray(sp.lap(dataspec, rsphere))
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


def make_random_dataspec_no_mean(ntrunc: int, seed=123, scale=1e-1, dtype=np.complex64):
    """
    n=0 常数项置零，便于检查 invlap 与 lap 的可逆关系。
    """
    out = make_random_dataspec(ntrunc=ntrunc, seed=seed, scale=scale, dtype=dtype)
    out[0] = dtype(0.0 + 0.0j)
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


# =========================================================
# invlap 理论公式
# =========================================================
def expected_invlap_theory(dataspec, rsphere: float):
    """
    理论上：
        lap(Y_n^m) = -n(n+1)/rsphere^2 * Y_n^m
    因此对 n>=1：
        invlap(Y_n^m) = -rsphere^2/[n(n+1)] * Y_n^m

    对 n=0 常数模态不可逆，通常返回 0。
    """
    dataspec = np.asarray(dataspec)
    ntrunc = infer_ntrunc_from_nspec(dataspec.shape[0])
    _, indxn = getspecindx(ntrunc)

    out = np.zeros(dataspec.shape, dtype=np.complex128)
    mask = indxn > 0
    factor = np.zeros_like(indxn, dtype=np.float64)
    factor[mask] = -(rsphere ** 2) / (indxn[mask] * (indxn[mask] + 1)).astype(np.float64)

    out[mask] = dataspec.astype(np.complex128)[mask] * factor[mask]
    out[~mask] = 0.0 + 0.0j
    return out


# =========================================================
# 1) 随机谱场：Fortran vs Rust + 理论比较
# =========================================================
def check_invlap_random(ntrunc_list=(0, 1, 3, 5, 8, 12), rsphere=6.3712e6, seed=123):
    print("\n" + "=" * 80)
    print("INVLAP COMPARISON ON RANDOM SPECTRA")
    print("=" * 80)

    for ntrunc in ntrunc_list:
        print(f"\n-------------------- ntrunc = {ntrunc} --------------------")

        dataspec = make_random_dataspec(ntrunc=ntrunc, seed=seed + ntrunc, scale=1e-1)

        inv_f = invlap_from_backend(fort_sp, dataspec, rsphere)
        inv_r = invlap_from_backend(rust_sp, dataspec, rsphere)

        summarize_diff("invlap(fortran) vs invlap(rust)", inv_f, inv_r)

        expected = expected_invlap_theory(dataspec, rsphere)
        summarize_theory_error("invlap(fortran) vs theory", inv_f, expected)
        summarize_theory_error("invlap(rust) vs theory", inv_r, expected)


# =========================================================
# 2) 单模态检查
# =========================================================
def check_invlap_single_modes(rsphere=6.3712e6):
    print("\n" + "=" * 80)
    print("INVLAP COMPARISON ON SINGLE MODES")
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

        inv_f = invlap_from_backend(fort_sp, dataspec, rsphere)
        inv_r = invlap_from_backend(rust_sp, dataspec, rsphere)
        expected = expected_invlap_theory(dataspec, rsphere)

        summarize_diff("invlap(fortran) vs invlap(rust)", inv_f, inv_r)
        summarize_theory_error("invlap(fortran) vs theory", inv_f, expected)
        summarize_theory_error("invlap(rust) vs theory", inv_r, expected)

        indxm, indxn = getspecindx(ntrunc)
        idx = np.where((indxm == m) & (indxn == n))[0][0]

        print("\n[mode detail]")
        print("input coeff        =", dataspec[idx])

        if n == 0:
            print("expected output    = 0 (constant mode is not invertible)")
        else:
            factor = -(rsphere ** 2) / (n * (n + 1))
            print("expected factor    =", factor)
            print("expected output    =", expected[idx])

        print("fortran output     =", inv_f[idx])
        print("rust output        =", inv_r[idx])


# =========================================================
# 3) 常数模态专项检查
# =========================================================
def check_invlap_constant_mode(rsphere_list=(1.0, 6.3712e6, 7.0e6)):
    print("\n" + "=" * 80)
    print("INVLAP CONSTANT MODE CHECK")
    print("=" * 80)

    for rsphere in rsphere_list:
        print(f"\n-------------------- rsphere = {rsphere} --------------------")

        dataspec = np.zeros(1, dtype=np.complex64)
        dataspec[0] = np.complex64(3.5 + 0.0j)

        inv_f = invlap_from_backend(fort_sp, dataspec, rsphere)
        inv_r = invlap_from_backend(rust_sp, dataspec, rsphere)

        print("input constant coeff =", dataspec[0])
        print("fortran output       =", inv_f[0])
        print("rust output          =", inv_r[0])
        print("expected             =", 0.0 + 0.0j)


# =========================================================
# 4) 组合检查：lap(invlap(x)) ≈ x（对 n=0 置零后的场）
# =========================================================
def check_lap_invlap_composition(ntrunc_list=(1, 3, 5, 8, 12), rsphere=6.3712e6, seed=456):
    print("\n" + "=" * 80)
    print("COMPOSITION CHECK: lap(invlap(x)) ~= x")
    print("=" * 80)

    for ntrunc in ntrunc_list:
        print(f"\n-------------------- ntrunc = {ntrunc} --------------------")

        dataspec = make_random_dataspec_no_mean(ntrunc=ntrunc, seed=seed + ntrunc, scale=1e-1)

        rec_f = lap_from_backend(fort_sp, invlap_from_backend(fort_sp, dataspec, rsphere), rsphere)
        rec_r = lap_from_backend(rust_sp, invlap_from_backend(rust_sp, dataspec, rsphere), rsphere)

        summarize_theory_error("fortran lap(invlap(x)) vs x", rec_f, dataspec)
        summarize_theory_error("rust lap(invlap(x)) vs x", rec_r, dataspec)
        summarize_diff("fortran vs rust reconstructed x", rec_f, rec_r)


# =========================================================
# 5) 组合检查：invlap(lap(x)) ≈ x（对 n=0 置零后的场）
# =========================================================
def check_invlap_lap_composition(ntrunc_list=(1, 3, 5, 8, 12), rsphere=6.3712e6, seed=789):
    print("\n" + "=" * 80)
    print("COMPOSITION CHECK: invlap(lap(x)) ~= x")
    print("=" * 80)

    for ntrunc in ntrunc_list:
        print(f"\n-------------------- ntrunc = {ntrunc} --------------------")

        dataspec = make_random_dataspec_no_mean(ntrunc=ntrunc, seed=seed + ntrunc, scale=1e-1)

        rec_f = invlap_from_backend(fort_sp, lap_from_backend(fort_sp, dataspec, rsphere), rsphere)
        rec_r = invlap_from_backend(rust_sp, lap_from_backend(rust_sp, dataspec, rsphere), rsphere)

        summarize_theory_error("fortran invlap(lap(x)) vs x", rec_f, dataspec)
        summarize_theory_error("rust invlap(lap(x)) vs x", rec_r, dataspec)
        summarize_diff("fortran vs rust reconstructed x", rec_f, rec_r)


# =========================================================
# 6) 多半径检查
# =========================================================
def check_invlap_rsphere_scaling(ntrunc=6, rsphere_list=(1.0, 2.0, 10.0, 6.3712e6), seed=321):
    print("\n" + "=" * 80)
    print("INVLAP RSPHERE SCALING CHECK")
    print("=" * 80)

    dataspec = make_random_dataspec(ntrunc=ntrunc, seed=seed, scale=1e-1)

    for rsphere in rsphere_list:
        print(f"\n-------------------- rsphere = {rsphere} --------------------")

        inv_f = invlap_from_backend(fort_sp, dataspec, rsphere)
        inv_r = invlap_from_backend(rust_sp, dataspec, rsphere)
        expected = expected_invlap_theory(dataspec, rsphere)

        summarize_diff("invlap(fortran) vs invlap(rust)", inv_f, inv_r)
        summarize_theory_error("invlap(fortran) vs theory", inv_f, expected)
        summarize_theory_error("invlap(rust) vs theory", inv_r, expected)


# =========================================================
# 7) 打印理论因子
# =========================================================
def inspect_invlap_factors(ntrunc=5, rsphere=6.3712e6):
    print("\n" + "=" * 80)
    print("INSPECT INVLAPLACIAN FACTORS")
    print("=" * 80)

    indxm, indxn = getspecindx(ntrunc)
    factor = np.zeros_like(indxn, dtype=np.float64)
    mask = indxn > 0
    factor[mask] = -(rsphere ** 2) / (indxn[mask] * (indxn[mask] + 1)).astype(np.float64)

    print(f"{'idx':>4} {'m':>4} {'n':>4} {'factor':>20}")
    for i in range(len(indxn)):
        print(f"{i:4d} {int(indxm[i]):4d} {int(indxn[i]):4d} {factor[i]:20.12e}")


# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    check_invlap_random(
        ntrunc_list=(0, 1, 3, 5, 8, 12),
        rsphere=6.3712e6,
        seed=123,
    )

    check_invlap_single_modes(rsphere=6.3712e6)

    check_invlap_constant_mode(rsphere_list=(1.0, 6.3712e6, 7.0e6))

    check_lap_invlap_composition(
        ntrunc_list=(1, 3, 5, 8, 12),
        rsphere=6.3712e6,
        seed=456,
    )

    check_invlap_lap_composition(
        ntrunc_list=(1, 3, 5, 8, 12),
        rsphere=6.3712e6,
        seed=789,
    )

    check_invlap_rsphere_scaling(
        ntrunc=6,
        rsphere_list=(1.0, 2.0, 10.0, 6.3712e6),
        seed=321,
    )

    inspect_invlap_factors(ntrunc=5, rsphere=6.3712e6)
