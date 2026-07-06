import math
import numpy as np

import spectralkernel_rs as rust_sp
from spharm_fortran_reference.pyspharm import _spherepack as fort_sp


# =========================================================
# 辅助函数
# =========================================================
def getspecindx(ntrunc: int):
    """
    始终返回 1D 数组。
    """
    indexn = np.indices((ntrunc + 1, ntrunc + 1))[1, :, :]
    indexm = np.indices((ntrunc + 1, ntrunc + 1))[0, :, :]
    indices = np.nonzero(np.greater(indexn, indexm - 1).flatten())
    indxn = np.take(indexn.flatten(), indices).reshape(-1)
    indxm = np.take(indexm.flatten(), indices).reshape(-1)
    return indxm, indxn


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


def print_nonzero_complex(label, arr, tol=1e-7, limit=20):
    arr = np.asarray(arr).reshape(-1)
    idx = np.where(np.abs(arr) > tol)[0]
    print(f"\n[{label}] nonzero count = {len(idx)}")
    for k in idx[:limit]:
        print(f"  k={k}, value={arr[k]!r}")


# =========================================================
# 构造测试数据
# =========================================================
def make_random_vector_harmonics(nlat: int, nt: int = 1, seed: int = 123, scale: float = 1e-1):
    """
    br, bi, cr, ci 的测试输入。
    形状统一取 (nlat, nlat, nt)。
    """
    rng = np.random.default_rng(seed)
    br = rng.normal(scale=scale, size=(nlat, nlat, nt)).astype(np.float32)
    bi = rng.normal(scale=scale, size=(nlat, nlat, nt)).astype(np.float32)
    cr = rng.normal(scale=scale, size=(nlat, nlat, nt)).astype(np.float32)
    ci = rng.normal(scale=scale, size=(nlat, nlat, nt)).astype(np.float32)
    return br, bi, cr, ci


def zero_out_invalid_triangle4(br, bi, cr, ci):
    """
    保守做法：m > n 的区域清零。
    """
    br = np.array(br, copy=True)
    bi = np.array(bi, copy=True)
    cr = np.array(cr, copy=True)
    ci = np.array(ci, copy=True)

    nlat = br.shape[0]
    for m in range(nlat):
        for n in range(nlat):
            if m > n:
                br[m, n, :] = 0.0
                bi[m, n, :] = 0.0
                cr[m, n, :] = 0.0
                ci[m, n, :] = 0.0
    return br, bi, cr, ci


# def make_random_vrtdivspec(ntrunc: int, nt: int = 1, seed: int = 123, scale: float = 1e-1):
#     ncoeff = (ntrunc + 1) * (ntrunc + 2) // 2
#     rng = np.random.default_rng(seed)

#     vrt_real = rng.normal(scale=scale, size=(ncoeff, nt))
#     vrt_imag = rng.normal(scale=scale, size=(ncoeff, nt))
#     div_real = rng.normal(scale=scale, size=(ncoeff, nt))
#     div_imag = rng.normal(scale=scale, size=(ncoeff, nt))

#     vrtspec = (vrt_real + 1j * vrt_imag).astype(np.complex64)
#     divspec = (div_real + 1j * div_imag).astype(np.complex64)

#     vrtspec[0, :] = np.complex64(1.0 + 0.0j)
#     divspec[0, :] = np.complex64(2.0 + 0.0j)
#     return vrtspec, divspec
def make_random_vrtdivspec(ntrunc: int, nt: int = 1, seed: int = 123, scale: float = 1e-1):
    ncoeff = (ntrunc + 1) * (ntrunc + 2) // 2
    rng = np.random.default_rng(seed)

    vrt_real = rng.normal(scale=scale, size=(ncoeff, nt))
    vrt_imag = rng.normal(scale=scale, size=(ncoeff, nt))
    div_real = rng.normal(scale=scale, size=(ncoeff, nt))
    div_imag = rng.normal(scale=scale, size=(ncoeff, nt))

    vrtspec = (vrt_real + 1j * vrt_imag).astype(np.complex64)
    divspec = (div_real + 1j * div_imag).astype(np.complex64)

    # n=0 模态置零
    vrtspec[0, :] = np.complex64(0.0 + 0.0j)
    divspec[0, :] = np.complex64(0.0 + 0.0j)
    return vrtspec, divspec


def make_single_mode_vrtdivspec(ntrunc: int, m: int, n: int, nt: int = 1,
                                vrt_value=1.0 + 0.5j, div_value=-0.25 + 0.75j):
    ncoeff = (ntrunc + 1) * (ntrunc + 2) // 2
    vrtspec = np.zeros((ncoeff, nt), dtype=np.complex64)
    divspec = np.zeros((ncoeff, nt), dtype=np.complex64)

    indxm, indxn = getspecindx(ntrunc)
    idx = np.where((indxm == m) & (indxn == n))[0]
    if len(idx) != 1:
        raise ValueError(f"Cannot find unique spectral index for (m={m}, n={n})")

    vrtspec[idx[0], :] = np.complex64(vrt_value)
    divspec[idx[0], :] = np.complex64(div_value)
    return vrtspec, divspec


# =========================================================
# 简单双参考：
# 假设 twodtooned_vrtdiv 的打包顺序与 getspecindx 对应。
# 这里只做“弱参考”：
#   vrtspec ~ 0.5*(br + i*bi) / rsphere
#   divspec ~ 0.5*(cr + i*ci) / rsphere
# 如果你的 Fortran 实现显示别的缩放，可再调这里。
# =========================================================
def expected_twodtooned_vrtdiv_weak_theory(br, bi, cr, ci, ntrunc: int, rsphere: float):
    br = np.asarray(br)
    bi = np.asarray(bi)
    cr = np.asarray(cr)
    ci = np.asarray(ci)

    nlat, nlat2, nt = br.shape
    if nlat != nlat2:
        raise ValueError("br/bi/cr/ci first two dims must be equal")

    indxm, indxn = getspecindx(ntrunc)
    out_vrt = np.empty((indxn.size, nt), dtype=np.complex128)
    out_div = np.empty((indxn.size, nt), dtype=np.complex128)

    for k, (m, n) in enumerate(zip(indxm, indxn)):
        m = int(m)
        n = int(n)
        out_vrt[k, :] = 0.5 * (br[m, n, :].astype(np.float64) + 1j * bi[m, n, :].astype(np.float64)) / rsphere
        out_div[k, :] = 0.5 * (cr[m, n, :].astype(np.float64) + 1j * ci[m, n, :].astype(np.float64)) / rsphere

    return out_vrt, out_div


# =========================================================
# 1) twodtooned_vrtdiv: Fortran vs Rust
# =========================================================
def check_twodtooned_vrtdiv_random(
    nlat_list=(3, 4, 6, 8, 12),
    nt_list=(1, 2, 4),
    rsphere=6.3712e6,
    seed=123,
    check_weak_theory=False,
):
    print("\n" + "=" * 80)
    print("TWODTOONED_VRTDIV COMPARISON ON RANDOM br,bi,cr,ci")
    print("=" * 80)

    for nlat in nlat_list:
        for nt in nt_list:
            print(f"\n-------------------- nlat={nlat}, nt={nt} --------------------")

            br, bi, cr, ci = make_random_vector_harmonics(
                nlat=nlat, nt=nt, seed=seed + 10 * nlat + nt
            )
            br, bi, cr, ci = zero_out_invalid_triangle4(br, bi, cr, ci)

            for ntrunc in (0, min(1, nlat - 1), min(3, nlat - 1), nlat - 1):
                print(f"\n  >>> ntrunc = {ntrunc}")

                vrt_f, div_f = fort_sp.twodtooned_vrtdiv(br, bi, cr, ci, ntrunc, rsphere)
                vrt_r, div_r = rust_sp.twodtooned_vrtdiv(br, bi, cr, ci, ntrunc, rsphere)

                summarize_diff("vrtspec: fortran vs rust", vrt_f, vrt_r)
                summarize_diff("divspec: fortran vs rust", div_f, div_r)

                if check_weak_theory:
                    vrt_t, div_t = expected_twodtooned_vrtdiv_weak_theory(
                        br, bi, cr, ci, ntrunc, rsphere
                    )
                    summarize_diff("vrtspec: fortran vs weak theory", vrt_f, vrt_t)
                    summarize_diff("divspec: fortran vs weak theory", div_f, div_t)


# =========================================================
# 2) 单模态 br/bi/cr/ci 检查
# =========================================================
def check_twodtooned_vrtdiv_single_entries(rsphere=6.3712e6):
    print("\n" + "=" * 80)
    print("TWODTOONED_VRTDIV SINGLE-ENTRY CHECK")
    print("=" * 80)

    cases = [
        # (nlat, ntrunc, m, n, br, bi, cr, ci)
        (3, 2, 0, 0, 1.0, 0.0, 2.0, 0.0),
        (4, 3, 0, 1, 2.0, -0.5, -1.0, 0.25),
        (6, 5, 1, 1, -1.2, 0.3, 0.4, -0.7),
        (6, 5, 2, 3, 0.75, 0.25, -0.6, 1.1),
        (8, 7, 3, 5, -0.4, 1.1, 0.2, -0.9),
    ]

    for nlat, ntrunc, m, n, br0, bi0, cr0, ci0 in cases:
        print(f"\n-------------------- nlat={nlat}, ntrunc={ntrunc}, m={m}, n={n} --------------------")

        br = np.zeros((nlat, nlat, 1), dtype=np.float32)
        bi = np.zeros((nlat, nlat, 1), dtype=np.float32)
        cr = np.zeros((nlat, nlat, 1), dtype=np.float32)
        ci = np.zeros((nlat, nlat, 1), dtype=np.float32)

        br[m, n, 0] = np.float32(br0)
        bi[m, n, 0] = np.float32(bi0)
        cr[m, n, 0] = np.float32(cr0)
        ci[m, n, 0] = np.float32(ci0)

        vrt_f, div_f = fort_sp.twodtooned_vrtdiv(br, bi, cr, ci, ntrunc, rsphere)
        vrt_r, div_r = rust_sp.twodtooned_vrtdiv(br, bi, cr, ci, ntrunc, rsphere)

        summarize_diff("vrtspec: fortran vs rust", vrt_f, vrt_r)
        summarize_diff("divspec: fortran vs rust", div_f, div_r)

        print_nonzero_complex("vrtspec fortran", vrt_f)
        print_nonzero_complex("vrtspec rust", vrt_r)
        print_nonzero_complex("divspec fortran", div_f)
        print_nonzero_complex("divspec rust", div_r)


# =========================================================
# 3) onedtotwod_vrtdiv -> twodtooned_vrtdiv 回环检查
# =========================================================
def check_roundtrip_vrtdiv_oned_to_twod(
    nlat_list=(3, 4, 6, 8, 12),
    nt_list=(1, 2, 4),
    rsphere=6.3712e6,
    seed=456,
):
    print("\n" + "=" * 80)
    print("ROUNDTRIP CHECK: onedtotwod_vrtdiv -> twodtooned_vrtdiv")
    print("=" * 80)

    for nlat in nlat_list:
        ntrunc = nlat - 1
        for nt in nt_list:
            print(f"\n-------------------- nlat={nlat}, ntrunc={ntrunc}, nt={nt} --------------------")

            vrtspec, divspec = make_random_vrtdivspec(
                ntrunc=ntrunc, nt=nt, seed=seed + 10 * nlat + nt
            )

            # Fortran roundtrip
            br_f, bi_f, cr_f, ci_f = fort_sp.onedtotwod_vrtdiv(vrtspec, divspec, nlat, rsphere)
            vrt_rec_f, div_rec_f = fort_sp.twodtooned_vrtdiv(br_f, bi_f, cr_f, ci_f, ntrunc, rsphere)

            # Rust roundtrip
            br_r, bi_r, cr_r, ci_r = rust_sp.onedtotwod_vrtdiv(vrtspec, divspec, nlat, rsphere)
            vrt_rec_r, div_rec_r = rust_sp.twodtooned_vrtdiv(br_r, bi_r, cr_r, ci_r, ntrunc, rsphere)

            summarize_diff("vrtspec fortran roundtrip vs input", vrt_rec_f, vrtspec)
            summarize_diff("divspec fortran roundtrip vs input", div_rec_f, divspec)

            summarize_diff("vrtspec rust roundtrip vs input", vrt_rec_r, vrtspec)
            summarize_diff("divspec rust roundtrip vs input", div_rec_r, divspec)

            summarize_diff("vrtspec recovered: fortran vs rust", vrt_rec_f, vrt_rec_r)
            summarize_diff("divspec recovered: fortran vs rust", div_rec_f, div_rec_r)

            summarize_diff("br: fortran vs rust", br_f, br_r)
            summarize_diff("bi: fortran vs rust", bi_f, bi_r)
            summarize_diff("cr: fortran vs rust", cr_f, cr_r)
            summarize_diff("ci: fortran vs rust", ci_f, ci_r)


# =========================================================
# 4) twodtooned_vrtdiv -> onedtotwod_vrtdiv 回环检查
# =========================================================
def check_roundtrip_vrtdiv_twod_to_oned(
    nlat_list=(3, 4, 6, 8, 12),
    nt_list=(1, 2, 4),
    rsphere=6.3712e6,
    seed=789,
):
    print("\n" + "=" * 80)
    print("ROUNDTRIP CHECK: twodtooned_vrtdiv -> onedtotwod_vrtdiv")
    print("=" * 80)

    for nlat in nlat_list:
        for nt in nt_list:
            print(f"\n-------------------- nlat={nlat}, nt={nt} --------------------")

            br, bi, cr, ci = make_random_vector_harmonics(
                nlat=nlat, nt=nt, seed=seed + 10 * nlat + nt
            )
            br, bi, cr, ci = zero_out_invalid_triangle4(br, bi, cr, ci)

            for ntrunc in (0, min(1, nlat - 1), min(3, nlat - 1), nlat - 1):
                print(f"\n  >>> ntrunc = {ntrunc}")

                vrt_f, div_f = fort_sp.twodtooned_vrtdiv(br, bi, cr, ci, ntrunc, rsphere)
                vrt_r, div_r = rust_sp.twodtooned_vrtdiv(br, bi, cr, ci, ntrunc, rsphere)

                br_f2, bi_f2, cr_f2, ci_f2 = fort_sp.onedtotwod_vrtdiv(vrt_f, div_f, nlat, rsphere)
                br_r2, bi_r2, cr_r2, ci_r2 = rust_sp.onedtotwod_vrtdiv(vrt_r, div_r, nlat, rsphere)

                summarize_diff("vrtspec: fortran vs rust", vrt_f, vrt_r)
                summarize_diff("divspec: fortran vs rust", div_f, div_r)

                summarize_diff("br recovered: fortran vs rust", br_f2, br_r2)
                summarize_diff("bi recovered: fortran vs rust", bi_f2, bi_r2)
                summarize_diff("cr recovered: fortran vs rust", cr_f2, cr_r2)
                summarize_diff("ci recovered: fortran vs rust", ci_f2, ci_r2)

                # 只检查有效三角区
                indxm, indxn = getspecindx(ntrunc)
                max_errs_f = {"br": 0.0, "bi": 0.0, "cr": 0.0, "ci": 0.0}
                max_errs_r = {"br": 0.0, "bi": 0.0, "cr": 0.0, "ci": 0.0}

                for m, n in zip(indxm, indxn):
                    m = int(m)
                    n = int(n)
                    max_errs_f["br"] = max(max_errs_f["br"], abs(float(br_f2[m, n, 0] - br[m, n, 0])))
                    max_errs_f["bi"] = max(max_errs_f["bi"], abs(float(bi_f2[m, n, 0] - bi[m, n, 0])))
                    max_errs_f["cr"] = max(max_errs_f["cr"], abs(float(cr_f2[m, n, 0] - cr[m, n, 0])))
                    max_errs_f["ci"] = max(max_errs_f["ci"], abs(float(ci_f2[m, n, 0] - ci[m, n, 0])))

                    max_errs_r["br"] = max(max_errs_r["br"], abs(float(br_r2[m, n, 0] - br[m, n, 0])))
                    max_errs_r["bi"] = max(max_errs_r["bi"], abs(float(bi_r2[m, n, 0] - bi[m, n, 0])))
                    max_errs_r["cr"] = max(max_errs_r["cr"], abs(float(cr_r2[m, n, 0] - cr[m, n, 0])))
                    max_errs_r["ci"] = max(max_errs_r["ci"], abs(float(ci_r2[m, n, 0] - ci[m, n, 0])))

                print("\n[valid triangular region recovery]")
                print("fortran max |br_rec-br| =", max_errs_f["br"])
                print("fortran max |bi_rec-bi| =", max_errs_f["bi"])
                print("fortran max |cr_rec-cr| =", max_errs_f["cr"])
                print("fortran max |ci_rec-ci| =", max_errs_f["ci"])
                print("rust    max |br_rec-br| =", max_errs_r["br"])
                print("rust    max |bi_rec-bi| =", max_errs_r["bi"])
                print("rust    max |cr_rec-cr| =", max_errs_r["cr"])
                print("rust    max |ci_rec-ci| =", max_errs_r["ci"])


# =========================================================
# 5) 单模态谱 -> 二维数组 定位检查
# =========================================================
def check_onedtotwod_vrtdiv_single_modes(rsphere=6.3712e6):
    print("\n" + "=" * 80)
    print("ONEDTOTWOD_VRTDIV SINGLE-MODE CHECK")
    print("=" * 80)

    cases = [
        (3, 2, 0, 0),
        (4, 3, 0, 1),
        (6, 5, 1, 1),
        (6, 5, 2, 3),
        (8, 7, 3, 5),
    ]

    for nlat, ntrunc, m, n in cases:
        print(f"\n-------------------- nlat={nlat}, ntrunc={ntrunc}, m={m}, n={n} --------------------")

        vrtspec, divspec = make_single_mode_vrtdivspec(
            ntrunc=ntrunc,
            m=m,
            n=n,
            nt=1,
            vrt_value=1.0 + 0.5j,
            div_value=-0.25 + 0.75j,
        )

        br_f, bi_f, cr_f, ci_f = fort_sp.onedtotwod_vrtdiv(vrtspec, divspec, nlat, rsphere)
        br_r, bi_r, cr_r, ci_r = rust_sp.onedtotwod_vrtdiv(vrtspec, divspec, nlat, rsphere)

        summarize_diff("br: fortran vs rust", br_f, br_r)
        summarize_diff("bi: fortran vs rust", bi_f, bi_r)
        summarize_diff("cr: fortran vs rust", cr_f, cr_r)
        summarize_diff("ci: fortran vs rust", ci_f, ci_r)

        print("\n[nonzero entries: Fortran]")
        nz = np.where(np.abs(br_f.reshape(-1)) > 1e-7)[0]
        print("br:", nz[:20])
        nz = np.where(np.abs(bi_f.reshape(-1)) > 1e-7)[0]
        print("bi:", nz[:20])
        nz = np.where(np.abs(cr_f.reshape(-1)) > 1e-7)[0]
        print("cr:", nz[:20])
        nz = np.where(np.abs(ci_f.reshape(-1)) > 1e-7)[0]
        print("ci:", nz[:20])

        print("\n[nonzero entries: Rust]")
        nz = np.where(np.abs(br_r.reshape(-1)) > 1e-7)[0]
        print("br:", nz[:20])
        nz = np.where(np.abs(bi_r.reshape(-1)) > 1e-7)[0]
        print("bi:", nz[:20])
        nz = np.where(np.abs(cr_r.reshape(-1)) > 1e-7)[0]
        print("cr:", nz[:20])
        nz = np.where(np.abs(ci_r.reshape(-1)) > 1e-7)[0]
        print("ci:", nz[:20])


# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    check_twodtooned_vrtdiv_random(
        nlat_list=(3, 4, 6, 8, 12),
        nt_list=(1, 2, 4),
        rsphere=6.3712e6,
        seed=123,
        check_weak_theory=False,   # 先只看 Fortran vs Rust
    )

    check_twodtooned_vrtdiv_single_entries(
        rsphere=6.3712e6
    )

    check_roundtrip_vrtdiv_oned_to_twod(
        nlat_list=(3, 4, 6, 8, 12),
        nt_list=(1, 2, 4),
        rsphere=6.3712e6,
        seed=456,
    )

    check_roundtrip_vrtdiv_twod_to_oned(
        nlat_list=(3, 4, 6, 8, 12),
        nt_list=(1, 2, 4),
        rsphere=6.3712e6,
        seed=789,
    )

    check_onedtotwod_vrtdiv_single_modes(
        rsphere=6.3712e6
    )