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


def calc_vhsgs_sizes(nlat: int, nlon: int):
    n1 = min(nlat, (nlon + 1) // 2)
    imid = (nlat + 1) // 2
    lmn = nlat * (nlat + 1) // 2
    lvhsgs_synth = n1 * imid * (2 * nlat - n1 + 1) // 2 * 2 + nlon + 15
    lvhsgs_init = 2 * imid * lmn + nlon + 15
    lvhsgs = max(lvhsgs_synth, lvhsgs_init)
    ldwork = (3 * nlat * (nlat + 3) + 2) // 2
    return lvhsgs, ldwork


def calc_lwork(nlat: int, nlon: int, nt: int, ityp: int):
    imid = (nlat + 1) // 2
    mmax = min(nlat, (nlon + 1) // 2)
    if ityp <= 2:
        return (2 * nt + 1) * nlat * nlon + nlat * (4 * nt * mmax + 1)
    return (2 * nt + 1) * imid * nlon + nlat * (4 * nt * mmax + 1)


def make_coeffs(nlat: int, nt: int):
    br = np.zeros((nlat, nlat, nt), dtype=np.float32)
    bi = np.zeros((nlat, nlat, nt), dtype=np.float32)
    cr = np.zeros((nlat, nlat, nt), dtype=np.float32)
    ci = np.zeros((nlat, nlat, nt), dtype=np.float32)
    for k in range(nt):
        for m in range(nlat):
            for n in range(m, nlat):
                seq = (m * nlat + n) * nt + k
                br[m, n, k] = 0.05 + 0.011 * seq
                bi[m, n, k] = -0.02 + 0.007 * seq
                cr[m, n, k] = 0.03 - 0.009 * seq
                ci[m, n, k] = -0.01 + 0.005 * seq
    return br, bi, cr, ci


def run_case(nlat: int, nlon: int, nt: int, ityp: int = 0):
    print(f"\n{'=' * 80}\nvlapgs: nlat={nlat}, nlon={nlon}, nt={nt}, ityp={ityp}\n{'=' * 80}")
    lvhsgs, ldwork = calc_vhsgs_sizes(nlat, nlon)
    wvhsgs, ierr0 = fort_sp.vhsgsi(nlat, nlon, lvhsgs, ldwork)
    assert ierr0 == 0, ("vhsgsi failed", nlat, nlon, ierr0)

    br, bi, cr, ci = make_coeffs(nlat, nt)
    if ityp in (1, 4, 7):
        cr.fill(0.0)
        ci.fill(0.0)
    if ityp in (2, 5, 8):
        br.fill(0.0)
        bi.fill(0.0)

    lwork = calc_lwork(nlat, nlon, nt, ityp)
    wvhsgs = np.asarray(wvhsgs, dtype=np.float32)

    if ityp == 0:
        vlap_f, wlap_f, ierr_f = fort_sp.vlapgs(nlon, br, bi, cr, ci, wvhsgs, lwork)
        vlap_r, wlap_r, ierr_r = rust_sp.vlapgs(nlon, br, bi, cr, ci, wvhsgs, lwork)
    else:
        vlap_f, wlap_f, ierr_f = fort_sp.vlapgs(nlon, br, bi, cr, ci, wvhsgs, lwork, ityp=ityp)
        vlap_r, wlap_r, ierr_r = rust_sp.vlapgs_ityp(nlon, br, bi, cr, ci, ityp, wvhsgs, lwork)

    assert ierr_f == 0, ("fortran vlapgs failed", nlat, nlon, nt, ityp, ierr_f)
    assert ierr_r == 0, ("rust vlapgs failed", nlat, nlon, nt, ityp, ierr_r)
    summarize_diff("vlapgs vlap", vlap_f, vlap_r)
    summarize_diff("vlapgs wlap", wlap_f, wlap_r)


if __name__ == "__main__":
    for ityp in range(9):
        for case in [(4, 4, 1), (5, 8, 2), (73, 144, 1)]:
            run_case(*case, ityp=ityp)
