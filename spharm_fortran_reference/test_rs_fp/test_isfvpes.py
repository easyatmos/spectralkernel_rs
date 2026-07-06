import numpy as np

from spharm_fortran_reference.pyspharm import _spherepack as fort_sp
import spectralkernel_rs as rust_sp


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

    # --- 新增：量级 ---
    print("\n-- magnitude of a --")
    print("max |a| :", np.max(abs_a))
    print("mean|a| :", np.mean(abs_a))
    print("rms  |a|:", np.sqrt(np.mean(abs_a ** 2)))

    print("\n-- magnitude of b --")
    print("max |b| :", np.max(abs_b))
    print("mean|b| :", np.mean(abs_b))
    print("rms  |b|:", np.sqrt(np.mean(abs_b ** 2)))

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


def summarize_relative_error(name, a, b, eps=1e-12):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a - b
    denom = np.maximum(np.abs(b), eps)
    rel = np.abs(diff) / denom
    print(f"\n[{name} relative error]")
    print("max rel   :", np.max(rel))
    print("mean rel  :", np.mean(rel))
    print("rms rel   :", np.sqrt(np.mean(rel ** 2)))


def calc_vhses_sizes(nlat: int, nlon: int):
    if nlon % 2:
        n1 = min(nlat, (nlon + 1) // 2)
    else:
        n1 = min(nlat, (nlon + 2) // 2)
    n2 = (nlat + 1) // 2
    lvhses = n1 * n2 * (2 * nlat - n1 + 1) + nlon + 15
    lwork = 3 * max(n1 - 2, 0) * (2 * nlat - n1 - 1) // 2 + 5 * n2 * nlat
    ldwork = 4 * nlat * (nlat + 1) + 1
    return lvhses, lwork, ldwork


def calc_lwork(nlat: int, nlon: int, nt: int, isym: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    if isym == 0:
        return nlat * ((2 * nt + 1) * nlon + 4 * l1 * nt + 1)
    return (2 * nt + 1) * l2 * nlon + nlat * (4 * l1 * nt + 1)


def make_coeffs(nlat: int, nt: int, nlon: int):
    mmax = min(nlat, (nlon + 1) // 2)
    as_ = np.zeros((nlat, nlat, nt), dtype=np.float32)
    bs = np.zeros((nlat, nlat, nt), dtype=np.float32)
    av = np.zeros((nlat, nlat, nt), dtype=np.float32)
    bv = np.zeros((nlat, nlat, nt), dtype=np.float32)
    for k in range(nt):
        for m in range(mmax):
            for n in range(m, nlat):
                seq = (m * nlat + n) * nt + k
                as_[m, n, k] = 0.05 + 0.002 * seq
                bs[m, n, k] = -0.02 + 0.003 * seq
                av[m, n, k] = 0.04 - 0.0015 * seq
                bv[m, n, k] = -0.015 + 0.001 * seq
    return as_, bs, av, bv


def build_vector_coeffs(as_, bs, av, bv, nlat: int, nlon: int, nt: int):
    mmax = min(nlat, (nlon + 1) // 2)
    br = np.zeros_like(as_)
    bi = np.zeros_like(bs)
    cr = np.zeros_like(av)
    ci = np.zeros_like(bv)
    fnn = np.zeros((nlat,), dtype=np.float32)
    for n in range(1, nlat):
        fn = np.float32(n)
        fnn[n] = -np.sqrt(fn * (fn - np.float32(1.0)))
    for k in range(nt):
        for n in range(1, nlat):
            br[0, n, k] = -fnn[n] * av[0, n, k]
            bi[0, n, k] = -fnn[n] * bv[0, n, k]
            cr[0, n, k] = fnn[n] * as_[0, n, k]
            ci[0, n, k] = fnn[n] * bs[0, n, k]
        for m in range(1, mmax):
            for n in range(m, nlat):
                br[m, n, k] = -fnn[n] * av[m, n, k]
                bi[m, n, k] = -fnn[n] * bv[m, n, k]
                cr[m, n, k] = fnn[n] * as_[m, n, k]
                ci[m, n, k] = fnn[n] * bs[m, n, k]
    return br, bi, cr, ci


def run_case(nlat: int, nlon: int, nt: int, isym: int = 0):
    print(f"\n{'=' * 80}\nisfvpes: nlat={nlat}, nlon={nlon}, nt={nt}, isym={isym}\n{'=' * 80}")
    lvhses, init_lwork, ldwork = calc_vhses_sizes(nlat, nlon)
    wvhses, ierr0 = fort_sp.vhsesi(nlat, nlon, lvhses, init_lwork, ldwork)
    assert ierr0 == 0, ("vhsesi failed", nlat, nlon, ierr0)

    as_, bs, av, bv = make_coeffs(nlat, nt, nlon)
    lwork = calc_lwork(nlat, nlon, nt, isym)
    wvhses = np.asarray(wvhses, dtype=np.float32)

    br, bi, cr, ci = build_vector_coeffs(as_, bs, av, bv, nlat, nlon, nt)
    ityp = {0: 0, 1: 3, 2: 6}[isym]
    vh_v_f, vh_w_f, vh_ierr_f = fort_sp.vhses(nlon, br, bi, cr, ci, wvhses, lwork, ityp=ityp)
    vh_v_r, vh_w_r, vh_ierr_r = rust_sp.vhses_ityp(br, bi, cr, ci, ityp, wvhses, lwork)
    assert vh_ierr_f == 0, ("fortran vhses failed", nlat, nlon, nt, isym, vh_ierr_f)
    assert vh_ierr_r == 0, ("rust vhses failed", nlat, nlon, nt, isym, vh_ierr_r)
    summarize_diff("isfvpes low-level vhses v", vh_v_f, vh_v_r)
    summarize_diff("isfvpes low-level vhses w", vh_w_f, vh_w_r)

    if isym == 0:
        v_f, w_f, ierr_f = fort_sp.isfvpes(nlon, as_, bs, av, bv, wvhses, lwork)
        v_r, w_r, ierr_r = rust_sp.isfvpes(nlon, as_, bs, av, bv, wvhses, lwork)
    else:
        v_f, w_f, ierr_f = fort_sp.isfvpes(nlon, as_, bs, av, bv, wvhses, lwork, isym=isym)
        v_r, w_r, ierr_r = rust_sp.isfvpes_isym(nlon, isym, as_, bs, av, bv, wvhses, lwork)

    assert ierr_f == 0, ("fortran isfvpes failed", nlat, nlon, nt, isym, ierr_f)
    assert ierr_r == 0, ("rust isfvpes failed", nlat, nlon, nt, isym, ierr_r)
    summarize_diff("isfvpes v", v_f, v_r)
    summarize_diff("isfvpes w", w_f, w_r)
    # summarize_relative_error("isfvpes v", v_f, v_r)
    # summarize_relative_error("isfvpes w", w_f, w_r)


if __name__ == "__main__":
    for case in [(4, 8, 1, 0), (5, 8, 2, 0), (5, 8, 1, 1), (5, 8, 1, 2), (73, 144, 1, 0)]:
        run_case(*case)
