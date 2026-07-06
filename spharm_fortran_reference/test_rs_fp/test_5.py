import math
import numpy as np

import spectralkernel_rs as rust_sp
from spharm_fortran_reference.pyspharm import _spherepack as fort_sp


# =========================================================
# 辅助函数
# =========================================================
def getspecindx(ntrunc: int):
    """
    与 spharm.py 中一致：返回每个一维谱系数对应的 (m, n)
    """
    indexn = np.indices((ntrunc + 1, ntrunc + 1))[1, :, :]
    indexm = np.indices((ntrunc + 1, ntrunc + 1))[0, :, :]
    indices = np.nonzero(np.greater(indexn, indexm - 1).flatten())
    indxn = np.take(indexn.flatten(), indices)
    indxm = np.take(indexm.flatten(), indices)
    return np.asarray(indxm).reshape(-1), np.asarray(indxn).reshape(-1)


def infer_ntrunc_from_nspec(nspec: int) -> int:
    ntrunc = int(-1.5 + 0.5 * math.sqrt(9.0 - 8.0 * (1.0 - nspec)))
    if (ntrunc + 1) * (ntrunc + 2) // 2 != nspec:
        raise ValueError(f"Invalid spectral size: {nspec}")
    return ntrunc


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


def summarize_exact_match(name, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    print(f"\n[{name}]")
    print("allclose:", np.allclose(a, b))
    print("array_equal:", np.array_equal(a, b))
    print("max |diff| :", np.max(np.abs(a - b)))


# =========================================================
# 构造测试数据
# =========================================================
def make_random_ab(nlat: int, nt: int = 1, seed: int = 123, scale: float = 1e-1):
    """
    构造形状与 onedtotwod/twodtooned 兼容的二维谱实虚部数组:
    a.shape = b.shape = (nlat, nlat, nt)
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(scale=scale, size=(nlat, nlat, nt)).astype(np.float32)
    b = rng.normal(scale=scale, size=(nlat, nlat, nt)).astype(np.float32)
    return a, b


def zero_out_invalid_triangle(a, b):
    """
    可选：把明显不属于三角截断区的部分清零，便于更干净地检查。
    不强依赖 backend 的内部约定，只做一个保守清零：
    对 m > n 的项置零。
    """
    a = np.array(a, copy=True)
    b = np.array(b, copy=True)

    nlat = a.shape[0]
    for m in range(nlat):
        for n in range(nlat):
            if m > n:
                a[m, n, :] = 0.0
                b[m, n, :] = 0.0
    return a, b


def make_random_dataspec(ntrunc: int, nt: int = 1, seed: int = 123, scale: float = 1e-1):
    ncoeff = (ntrunc + 1) * (ntrunc + 2) // 2
    rng = np.random.default_rng(seed)
    real = rng.normal(scale=scale, size=(ncoeff, nt))
    imag = rng.normal(scale=scale, size=(ncoeff, nt))
    dataspec = (real + 1j * imag).astype(np.complex64)
    dataspec[0, :] = np.complex64(1.0 + 0.0j)
    return dataspec


def make_single_mode_dataspec(ntrunc: int, m: int, n: int, nt: int = 1, value=1.0 + 0.5j):
    ncoeff = (ntrunc + 1) * (ntrunc + 2) // 2
    dataspec = np.zeros((ncoeff, nt), dtype=np.complex64)
    indxm, indxn = getspecindx(ntrunc)
    idx = np.where((indxm == m) & (indxn == n))[0]
    if len(idx) != 1:
        raise ValueError(f"Cannot find unique spectral index for (m={m}, n={n})")
    dataspec[idx[0], :] = np.complex64(value)
    return dataspec


# =========================================================
# 理论打包函数：只按 getspecindx 的顺序抽取 a,b -> complex
# =========================================================
def expected_twodtooned_theory(a, b, ntrunc: int):
    """
    理论参考：
    一维复谱系数第 k 个对应 (m,n)=getspecindx(k)，
    取 complex(a[m,n], b[m,n])。
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        raise ValueError("a and b must have same shape")
    if a.ndim != 3:
        raise ValueError("a and b must have shape (nlat, nlat, nt)")

    nlat, nlat2, nt = a.shape
    if nlat != nlat2:
        raise ValueError("a,b first two dimensions must be equal")

    indxm, indxn = getspecindx(ntrunc)
    out = np.empty((len(indxn), nt), dtype=np.complex128)
    for k, (m, n) in enumerate(zip(indxm, indxn)):
        out[k, :] = a[m, n, :].astype(np.float64) + 1j * b[m, n, :].astype(np.float64)
    return out


# =========================================================
# 1) twodtooned: Fortran vs Rust + 理论
# =========================================================
def check_twodtooned_random(nlat_list=(3, 4, 6, 8, 12), nt_list=(1, 2, 4), seed=123):
    print("\n" + "=" * 80)
    print("TWODTOONED COMPARISON ON RANDOM a,b")
    print("=" * 80)

    for nlat in nlat_list:
        for nt in nt_list:
            print(f"\n-------------------- nlat={nlat}, nt={nt} --------------------")

            a, b = make_random_ab(nlat=nlat, nt=nt, seed=seed + 10 * nlat + nt)
            a, b = zero_out_invalid_triangle(a, b)

            for ntrunc in (0, min(1, nlat - 1), min(3, nlat - 1), nlat - 1):
                print(f"\n  >>> ntrunc = {ntrunc}")

                out_f = fort_sp.twodtooned(a, b, ntrunc)
                out_r = rust_sp.twodtooned(a, b, ntrunc)
                expected = expected_twodtooned_theory(a, b, ntrunc)

                summarize_diff("twodtooned(fortran) vs twodtooned(rust)", out_f, out_r)
                summarize_diff("twodtooned(fortran) vs theory", out_f, expected)
                summarize_diff("twodtooned(rust) vs theory", out_r, expected)


# =========================================================
# 2) 单模态 a,b 检查
# =========================================================
def check_twodtooned_single_ab():
    print("\n" + "=" * 80)
    print("TWODTOONED SINGLE-ENTRY CHECK")
    print("=" * 80)

    cases = [
        # (nlat, ntrunc, m, n, aval, bval)
        (3, 2, 0, 0, 1.0, 0.0),
        (4, 3, 0, 1, 2.0, -0.5),
        (6, 5, 1, 1, -1.2, 0.3),
        (6, 5, 2, 3, 0.75, 0.25),
        (8, 7, 3, 5, -0.4, 1.1),
    ]

    for nlat, ntrunc, m, n, aval, bval in cases:
        print(f"\n-------------------- nlat={nlat}, ntrunc={ntrunc}, m={m}, n={n} --------------------")

        a = np.zeros((nlat, nlat, 1), dtype=np.float32)
        b = np.zeros((nlat, nlat, 1), dtype=np.float32)
        a[m, n, 0] = np.float32(aval)
        b[m, n, 0] = np.float32(bval)

        out_f = fort_sp.twodtooned(a, b, ntrunc)
        out_r = rust_sp.twodtooned(a, b, ntrunc)
        expected = expected_twodtooned_theory(a, b, ntrunc)

        summarize_diff("fortran vs rust", out_f, out_r)
        summarize_diff("fortran vs theory", out_f, expected)
        summarize_diff("rust vs theory", out_r, expected)

        indxm, indxn = getspecindx(ntrunc)
        idx = np.where((indxm == m) & (indxn == n))[0]
        if len(idx) == 1:
            k = idx[0]
            print("\n[mode detail]")
            print("expected coeff =", expected[k, 0])
            print("fortran coeff  =", out_f[k, 0] if out_f.ndim == 2 else out_f[k])
            print("rust coeff     =", out_r[k, 0] if out_r.ndim == 2 else out_r[k])


# =========================================================
# 3) onedtotwod -> twodtooned 组合检查
# =========================================================
def check_roundtrip_oned_to_twod(nlat_list=(3, 4, 6, 8, 12), nt_list=(1, 2, 4), seed=456):
    print("\n" + "=" * 80)
    print("ROUNDTRIP CHECK: onedtotwod -> twodtooned")
    print("=" * 80)

    for nlat in nlat_list:
        ntrunc = nlat - 1
        for nt in nt_list:
            print(f"\n-------------------- nlat={nlat}, ntrunc={ntrunc}, nt={nt} --------------------")

            dataspec = make_random_dataspec(ntrunc=ntrunc, nt=nt, seed=seed + 10 * nlat + nt)

            # Fortran roundtrip
            a_f, b_f = fort_sp.onedtotwod(dataspec, nlat)
            rec_f = fort_sp.twodtooned(a_f, b_f, ntrunc)

            # Rust roundtrip
            a_r, b_r = rust_sp.onedtotwod(dataspec, nlat)
            rec_r = rust_sp.twodtooned(a_r, b_r, ntrunc)

            summarize_diff("fortran roundtrip recovered vs input", rec_f, dataspec)
            summarize_diff("rust roundtrip recovered vs input", rec_r, dataspec)
            summarize_diff("fortran recovered vs rust recovered", rec_f, rec_r)

            summarize_diff("onedtotwod a: fortran vs rust", a_f, a_r)
            summarize_diff("onedtotwod b: fortran vs rust", b_f, b_r)


# =========================================================
# 4) twodtooned -> onedtotwod 组合检查
# =========================================================
def check_roundtrip_twod_to_oned(nlat_list=(3, 4, 6, 8, 12), nt_list=(1, 2, 4), seed=789):
    print("\n" + "=" * 80)
    print("ROUNDTRIP CHECK: twodtooned -> onedtotwod")
    print("=" * 80)

    for nlat in nlat_list:
        for nt in nt_list:
            print(f"\n-------------------- nlat={nlat}, nt={nt} --------------------")

            a, b = make_random_ab(nlat=nlat, nt=nt, seed=seed + 10 * nlat + nt)
            a, b = zero_out_invalid_triangle(a, b)

            for ntrunc in (0, min(1, nlat - 1), min(3, nlat - 1), nlat - 1):
                print(f"\n  >>> ntrunc = {ntrunc}")

                # Fortran
                spec_f = fort_sp.twodtooned(a, b, ntrunc)
                aa_f, bb_f = fort_sp.onedtotwod(spec_f, nlat)

                # Rust
                spec_r = rust_sp.twodtooned(a, b, ntrunc)
                aa_r, bb_r = rust_sp.onedtotwod(spec_r, nlat)

                summarize_diff("spec fortran vs rust", spec_f, spec_r)
                summarize_diff("a_recovered fortran vs rust", aa_f, aa_r)
                summarize_diff("b_recovered fortran vs rust", bb_f, bb_r)

                # 只检查有效谱三角区是否恢复
                indxm, indxn = getspecindx(ntrunc)
                max_da_f = 0.0
                max_db_f = 0.0
                max_da_r = 0.0
                max_db_r = 0.0
                for m, n in zip(indxm, indxn):
                    max_da_f = max(max_da_f, abs(float(aa_f[m, n, 0] - a[m, n, 0])))
                    max_db_f = max(max_db_f, abs(float(bb_f[m, n, 0] - b[m, n, 0])))
                    max_da_r = max(max_da_r, abs(float(aa_r[m, n, 0] - a[m, n, 0])))
                    max_db_r = max(max_db_r, abs(float(bb_r[m, n, 0] - b[m, n, 0])))

                print("\n[valid triangular region recovery]")
                print("fortran max |a_rec-a| =", max_da_f)
                print("fortran max |b_rec-b| =", max_db_f)
                print("rust    max |a_rec-a| =", max_da_r)
                print("rust    max |b_rec-b| =", max_db_r)


# =========================================================
# 5) 索引表打印
# =========================================================
def inspect_twodtooned_index_map(ntrunc=5):
    print("\n" + "=" * 80)
    print("INSPECT TWODTOONED INDEX MAP")
    print("=" * 80)

    indxm, indxn = getspecindx(ntrunc)
    print(f"{'k':>4} {'m':>4} {'n':>4}")
    for k in range(len(indxn)):
        print(f"{k:4d} {int(indxm[k]):4d} {int(indxn[k]):4d}")


# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    check_twodtooned_random(
        nlat_list=(3, 4, 6, 8, 12),
        nt_list=(1, 2, 4),
        seed=123,
    )

    check_twodtooned_single_ab()

    check_roundtrip_oned_to_twod(
        nlat_list=(3, 4, 6, 8, 12),
        nt_list=(1, 2, 4),
        seed=456,
    )

    check_roundtrip_twod_to_oned(
        nlat_list=(3, 4, 6, 8, 12),
        nt_list=(1, 2, 4),
        seed=789,
    )

    inspect_twodtooned_index_map(ntrunc=5)