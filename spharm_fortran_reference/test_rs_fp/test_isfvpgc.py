import numpy as np

from spharm_fortran_reference.pyspharm import _spherepack as fort_sp
import spectralkernel_rs as rust_sp


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


def calc_vhsgc_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    lzz1 = 2 * nlat * imid
    mmax = min(nlat, (nlon + 1) // 2)
    labc = 3 * max(mmax - 2, 0) * (2 * nlat - mmax + 1) // 2
    lvhsgc = 2 * (lzz1 + labc) + nlon + 15
    ldwork = 2 * nlat * (nlat + 1) + 1
    return lvhsgc, ldwork


def calc_lwork(nlat: int, nlon: int, nt: int, isym: int):
    l1 = min(nlat, (nlon + 1) // 2)
    l2 = (nlat + 1) // 2
    if isym == 0:
        return nlat * (2 * nt * nlon + max(6 * l2, nlon) + 4 * l1 * nt + 1)
    return l2 * (2 * nt * nlon + max(6 * nlat, nlon)) + nlat * (4 * l1 * nt + 1)


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
                as_[m, n, k] = 0.07 + 0.003 * seq
                bs[m, n, k] = -0.05 + 0.004 * seq
                av[m, n, k] = 0.02 - 0.002 * seq
                bv[m, n, k] = -0.03 + 0.0015 * seq
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
    print(f"\n{'=' * 80}\nisfvpgc: nlat={nlat}, nlon={nlon}, nt={nt}, isym={isym}\n{'=' * 80}")
    lvhsgc, ldwork = calc_vhsgc_sizes(nlat, nlon)
    wvhsgc, ierr0 = fort_sp.vhsgci(nlat, nlon, lvhsgc, ldwork)
    assert ierr0 == 0, ("vhsgci failed", nlat, nlon, ierr0)

    as_, bs, av, bv = make_coeffs(nlat, nt, nlon)
    lwork = calc_lwork(nlat, nlon, nt, isym)
    wvhsgc = np.asarray(wvhsgc, dtype=np.float32)

    br, bi, cr, ci = build_vector_coeffs(as_, bs, av, bv, nlat, nlon, nt)
    ityp = {0: 0, 1: 3, 2: 6}[isym]
    vh_v_f, vh_w_f, vh_ierr_f = fort_sp.vhsgc(nlon, br, bi, cr, ci, wvhsgc, lwork, ityp=ityp)
    vh_v_r, vh_w_r, vh_ierr_r = rust_sp.vhsgc_ityp(br, bi, cr, ci, ityp, wvhsgc, lwork)
    assert vh_ierr_f == 0, ("fortran vhsgc failed", nlat, nlon, nt, isym, vh_ierr_f)
    assert vh_ierr_r == 0, ("rust vhsgc failed", nlat, nlon, nt, isym, vh_ierr_r)
    summarize_diff("isfvpgc low-level vhsgc v", vh_v_f, vh_v_r)
    summarize_diff("isfvpgc low-level vhsgc w", vh_w_f, vh_w_r)

    if isym == 0:
        v_f, w_f, ierr_f = fort_sp.isfvpgc(nlon, as_, bs, av, bv, wvhsgc, lwork)
        v_r, w_r, ierr_r = rust_sp.isfvpgc(nlon, as_, bs, av, bv, wvhsgc, lwork)
    else:
        v_f, w_f, ierr_f = fort_sp.isfvpgc(nlon, as_, bs, av, bv, wvhsgc, lwork, isym=isym)
        v_r, w_r, ierr_r = rust_sp.isfvpgc_isym(nlon, isym, as_, bs, av, bv, wvhsgc, lwork)

    assert ierr_f == 0, ("fortran isfvpgc failed", nlat, nlon, nt, isym, ierr_f)
    assert ierr_r == 0, ("rust isfvpgc failed", nlat, nlon, nt, isym, ierr_r)
    summarize_diff("isfvpgc v", v_f, v_r)
    summarize_diff("isfvpgc w", w_f, w_r)
    summarize_relative_error("isfvpgc v", v_f, v_r)
    summarize_relative_error("isfvpgc w", w_f, w_r)


if __name__ == "__main__":
    for case in [(4, 8, 1, 0), (5, 8, 2, 0), (5, 8, 1, 1), (5, 8, 1, 2), (73, 144, 1, 0)]:
        run_case(*case)
