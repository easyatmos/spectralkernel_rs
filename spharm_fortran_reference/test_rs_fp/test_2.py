import math
import numpy as np

import spectralkernel_rs as rust_sp
from spharm_fortran_reference.pyspharm import _spherepack as fort_sp


# =========================================================
# 从 spharm.py 单独抽出来的辅助函数
# =========================================================
def getspecindx(ntrunc: int):
    """
    返回每个谱系数对应的 (m, n)。
    顺序与 pyspharm/spherepack 约定一致。
    """
    indexn = np.indices((ntrunc + 1, ntrunc + 1))[1, :, :]
    indexm = np.indices((ntrunc + 1, ntrunc + 1))[0, :, :]
    indices = np.nonzero(np.greater(indexn, indexm - 1).flatten())
    indxn = np.take(indexn.flatten(), indices)
    indxm = np.take(indexm.flatten(), indices)
    return np.atleast_1d(np.squeeze(indxm)), np.atleast_1d(np.squeeze(indxn))


def infer_ntrunc_from_nspec(nspec: int) -> int:
    """
    从谱系数长度反推 ntrunc。
    nspec = (ntrunc+1)(ntrunc+2)/2
    """
    ntrunc = int(-1.5 + 0.5 * math.sqrt(9.0 - 8.0 * (1.0 - nspec)))
    if (ntrunc + 1) * (ntrunc + 2) // 2 != nspec:
        raise ValueError(f"Invalid spectral size: {nspec}")
    return ntrunc


def lap_from_backend(sp, dataspec, rsphere: float):
    """
    对应 spharm.py 中的底层调用:
        _spherepack.lap(dataspec, rsphere)
    """
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

    # 给常数项一个固定值，便于检查
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


def expected_lap_theory(dataspec, rsphere: float):
    """
    理论上，球面 Laplacian 在谱空间应为：
        lap(Y_n^m) = -n(n+1)/rsphere^2 * Y_n^m
    因此每个谱系数应乘以相同的 n-dependent 因子。
    """
    dataspec = np.asarray(dataspec)
    ntrunc = infer_ntrunc_from_nspec(dataspec.shape[0])
    _, indxn = getspecindx(ntrunc)

    factor = -(indxn * (indxn + 1)).astype(np.float64) / (rsphere ** 2)
    return (dataspec.astype(np.complex128) * factor.astype(np.float64)).astype(np.complex128)


# =========================================================
# 1) 随机谱场：Fortran vs Rust 直接比较
# =========================================================
def check_lap_random(ntrunc_list=(0, 1, 3, 5, 8, 12), rsphere=6.3712e6, seed=123):
    print("\n" + "=" * 80)
    print("LAP COMPARISON ON RANDOM SPECTRA")
    print("=" * 80)

    for ntrunc in ntrunc_list:
        print(f"\n-------------------- ntrunc = {ntrunc} --------------------")

        dataspec = make_random_dataspec(ntrunc=ntrunc, seed=seed + ntrunc, scale=1e-1)

        lap_f = lap_from_backend(fort_sp, dataspec, rsphere)
        lap_r = lap_from_backend(rust_sp, dataspec, rsphere)

        summarize_diff("lap(fortran) vs lap(rust)", lap_f, lap_r)

        # 理论检查
        expected = expected_lap_theory(dataspec, rsphere)
        summarize_theory_error("lap(fortran) vs theory", lap_f, expected)
        summarize_theory_error("lap(rust) vs theory", lap_r, expected)


# =========================================================
# 2) 单模态检查：更容易定位错误
# =========================================================
def check_lap_single_modes(rsphere=6.3712e6):
    print("\n" + "=" * 80)
    print("LAP COMPARISON ON SINGLE MODES")
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

        lap_f = lap_from_backend(fort_sp, dataspec, rsphere)
        lap_r = lap_from_backend(rust_sp, dataspec, rsphere)
        expected = expected_lap_theory(dataspec, rsphere)

        summarize_diff("lap(fortran) vs lap(rust)", lap_f, lap_r)
        summarize_theory_error("lap(fortran) vs theory", lap_f, expected)
        summarize_theory_error("lap(rust) vs theory", lap_r, expected)

        # 把非零模式直接打印出来，便于人工检查
        indxm, indxn = getspecindx(ntrunc)
        idx = np.where((indxm == m) & (indxn == n))[0][0]
        factor = -(n * (n + 1)) / (rsphere ** 2)

        print("\n[mode detail]")
        print("input coeff        =", dataspec[idx])
        print("expected factor    =", factor)
        print("expected output    =", expected[idx])
        print("fortran output     =", lap_f[idx])
        print("rust output        =", lap_r[idx])

        # 常数项额外看一下：n=0 时 lap 应为 0
        if n == 0:
            print("constant-mode check: expected exactly/approximately 0")


# =========================================================
# 3) 常数项专项检查
# =========================================================
def check_lap_constant_mode(rsphere_list=(1.0, 6.3712e6, 7.0e6)):
    print("\n" + "=" * 80)
    print("LAP CONSTANT MODE CHECK")
    print("=" * 80)

    for rsphere in rsphere_list:
        print(f"\n-------------------- rsphere = {rsphere} --------------------")

        dataspec = np.zeros(1, dtype=np.complex64)
        dataspec[0] = np.complex64(3.5 + 0.0j)

        lap_f = lap_from_backend(fort_sp, dataspec, rsphere)
        lap_r = lap_from_backend(rust_sp, dataspec, rsphere)

        print("input constant coeff =", dataspec[0])
        print("fortran output       =", lap_f[0])
        print("rust output          =", lap_r[0])
        print("expected             =", 0.0 + 0.0j)


# =========================================================
# 4) 多半径检查
# =========================================================
def check_lap_rsphere_scaling(ntrunc=6, rsphere_list=(1.0, 2.0, 10.0, 6.3712e6), seed=321):
    print("\n" + "=" * 80)
    print("LAP RSPHERE SCALING CHECK")
    print("=" * 80)

    dataspec = make_random_dataspec(ntrunc=ntrunc, seed=seed, scale=1e-1)

    for rsphere in rsphere_list:
        print(f"\n-------------------- rsphere = {rsphere} --------------------")

        lap_f = lap_from_backend(fort_sp, dataspec, rsphere)
        lap_r = lap_from_backend(rust_sp, dataspec, rsphere)
        expected = expected_lap_theory(dataspec, rsphere)

        summarize_diff("lap(fortran) vs lap(rust)", lap_f, lap_r)
        summarize_theory_error("lap(fortran) vs theory", lap_f, expected)
        summarize_theory_error("lap(rust) vs theory", lap_r, expected)


# =========================================================
# 5) 逐模态打印缩放因子（便于调试 Rust）
# =========================================================
def inspect_lap_factors(ntrunc=5, rsphere=6.3712e6):
    print("\n" + "=" * 80)
    print("INSPECT LAPLACIAN FACTORS")
    print("=" * 80)

    indxm, indxn = getspecindx(ntrunc)
    factor = -(indxn * (indxn + 1)).astype(np.float64) / (rsphere ** 2)

    print(f"{'idx':>4} {'m':>4} {'n':>4} {'factor':>20}")
    for i in range(len(indxn)):
        print(f"{i:4d} {int(indxm[i]):4d} {int(indxn[i]):4d} {factor[i]:20.12e}")


# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    check_lap_random(
        ntrunc_list=(0, 1, 3, 5, 8, 12),
        rsphere=6.3712e6,
        seed=123,
    )

    check_lap_single_modes(rsphere=6.3712e6)

    check_lap_constant_mode(rsphere_list=(1.0, 6.3712e6, 7.0e6))

    check_lap_rsphere_scaling(
        ntrunc=6,
        rsphere_list=(1.0, 2.0, 10.0, 6.3712e6),
        seed=321,
    )

    inspect_lap_factors(ntrunc=5, rsphere=6.3712e6)
