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


def calc_vhagc_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    lzz1 = 2 * nlat * imid
    mmax = min(nlat, (nlon + 1) // 2)
    labc = 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2
    lvhagc = 2 * (lzz1 + labc) + nlon + imid + 15
    ldwork = 2 * nlat * (nlat + 1) + 1
    return lvhagc, ldwork


def calc_shsgc_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    lshsgc = nlat * (2 * l2 + 3 * l1 - 2) + 3 * l1 * (1 - l1) // 2 + nlon + 15
    ldwork = nlat * (nlat + 4)
    return lshsgc, ldwork


def calc_vhagc_lwork(nlat: int, nlon: int, nt: int, ityp: int):
    imid = (nlat + 1) // 2
    if ityp <= 2:
        return nlat * (4 * nlon * nt + 6 * imid)
    return imid * (4 * nlon * nt + 6 * nlat)


def calc_sfvpgc_lwork(nlat: int, nlon: int, nt: int, isym: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    if isym == 0:
        return nlat * ((nt * nlon + max(3 * l2, nlon)) + 2 * l1 * nt + 1)
    return l2 * (nt * nlon + max(3 * nlat, nlon)) + nlat * (2 * l1 * nt + 1)


def make_vector_grid(nlat: int, nlon: int, nt: int):
    theta, _wts, ierr = fort_sp.gaqd(nlat)
    assert ierr == 0
    theta = np.asarray(theta, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    v = np.zeros((nlat, nlon, nt), dtype=np.float32)
    w = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        phase = k + 1
        v[:, :, k] = np.sin(theta)[:, None] * np.cos(phase * lon)[None, :] + 0.15 * np.cos(2.0 * theta)[:, None] * np.sin((phase + 1) * lon)[None, :]
        w[:, :, k] = np.cos(theta)[:, None] * np.sin(phase * lon)[None, :] + 0.10 * np.sin(3.0 * theta)[:, None] * np.cos((phase + 2) * lon)[None, :]
    return v, w


def run_case(nlat: int, nlon: int, nt: int, isym: int = 0):
    v, w = make_vector_grid(nlat, nlon, nt)
    lvhagc, ldwork_vhagc = calc_vhagc_sizes(nlat, nlon)
    wvhagc, ierr0 = fort_sp.vhagci(nlat, nlon, lvhagc, ldwork_vhagc)
    assert ierr0 == 0
    ityp = {0: 0, 1: 3, 2: 6}[isym]
    lwork_vhagc = calc_vhagc_lwork(nlat, nlon, nt, ityp)
    br, bi, cr, ci, ierr1 = fort_sp.vhagc(v, w, np.asarray(wvhagc, dtype=np.float32), lwork_vhagc, ityp=ityp)
    assert ierr1 == 0
    lshsgc, ldwork_shsgc = calc_shsgc_sizes(nlat, nlon)
    wshsgc, ierr2 = fort_sp.shsgci(nlat, nlon, lshsgc, ldwork_shsgc)
    assert ierr2 == 0
    lwork = calc_sfvpgc_lwork(nlat, nlon, nt, isym)
    if isym == 0:
        sf_f, vp_f, ierr_f = fort_sp.sfvpgc(nlon, br, bi, cr, ci, np.asarray(wshsgc, dtype=np.float32), lwork)
        sf_r, vp_r, ierr_r = rust_sp.sfvpgc(nlon, br, bi, cr, ci, np.asarray(wshsgc, dtype=np.float32), lwork)
    else:
        sf_f, vp_f, ierr_f = fort_sp.sfvpgc(nlon, br, bi, cr, ci, np.asarray(wshsgc, dtype=np.float32), lwork, isym=isym)
        sf_r, vp_r, ierr_r = rust_sp.sfvpgc_isym(nlon, isym, br, bi, cr, ci, np.asarray(wshsgc, dtype=np.float32), lwork)
    print(f"sfvpgc case nlat={nlat} nlon={nlon} nt={nt} isym={isym} ierr_f={ierr_f} ierr_r={ierr_r} lwork={lwork} wshsgc_len={len(wshsgc)}")
    assert ierr_f == 0
    assert ierr_r == 0
    summarize_diff(f"sfvpgc sf isym={isym}", sf_f, sf_r)
    summarize_diff(f"sfvpgc vp isym={isym}", vp_f, vp_r)


if __name__ == "__main__":
    for case in [(4, 8, 1, 0), (5, 8, 2, 0), (5, 8, 1, 1), (5, 8, 1, 2), (73, 144, 1, 0)]:
        run_case(*case)
