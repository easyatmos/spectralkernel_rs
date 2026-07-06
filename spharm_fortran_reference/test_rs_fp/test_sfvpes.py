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


def calc_vhaes_sizes(nlat: int, nlon: int):
    n1 = min(nlat, (nlon + 1) // 2)
    n2 = (nlat + 1) // 2
    lvhaes = n1 * n2 * (2 * nlat - n1 + 1) + nlon + 15
    lwork = 3 * max(n1 - 2, 0) * (2 * nlat - n1 - 1) // 2 + 5 * n2 * nlat
    ldwork = 2 * (nlat + 1)
    return lvhaes, lwork, ldwork


def calc_shses_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    mmax = min(nlat, (nlon + 2) // 2)
    lpimn = (imid * mmax * (2 * nlat - mmax + 1)) // 2
    lshses = lpimn + nlon + 15
    lwork = 5 * nlat * imid + 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2
    ldwork = 2 * (nlat + 1)
    return lshses, lwork, ldwork


def calc_vhaes_lwork(nlat: int, nlon: int, nt: int, ityp: int):
    imid = (nlat + 1) // 2
    if ityp <= 2:
        return nlat * (4 * nlon * nt + 6 * imid)
    return imid * (4 * nlon * nt + 6 * nlat)


def calc_sfvpes_lwork(nlat: int, nlon: int, nt: int, isym: int):
    imid = (nlat + 1) // 2
    if isym == 0:
        return nlat * ((nt + 1) * nlon + 2 * imid * nt + 1)
    return imid * ((nt + 1) * nlon + 2 * nlat * nt) + nlat


def find_valid_sfvpes_lwork(nlat: int, nlon: int, nt: int, isym: int, br, bi, cr, ci, wshses):
    lwork = calc_sfvpes_lwork(nlat, nlon, nt, isym)
    while True:
        if isym == 0:
            _, _, ierr = fort_sp.sfvpes(nlon, br, bi, cr, ci, np.asarray(wshses, dtype=np.float32), lwork)
        else:
            _, _, ierr = fort_sp.sfvpes(nlon, br, bi, cr, ci, np.asarray(wshses, dtype=np.float32), lwork, isym=isym)
        if ierr == 0:
            return lwork
        lwork += 1


def make_vector_grid(nlat: int, nlon: int, nt: int):
    theta = np.linspace(0.0, np.pi, nlat, dtype=np.float32)
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
    lvhaes, init_lwork_vhaes, ldwork_vhaes = calc_vhaes_sizes(nlat, nlon)
    wvhaes, ierr0 = fort_sp.vhaesi(nlat, nlon, lvhaes, init_lwork_vhaes, ldwork_vhaes)
    assert ierr0 == 0
    ityp = {0: 0, 1: 3, 2: 6}[isym]
    lwork_vhaes = calc_vhaes_lwork(nlat, nlon, nt, ityp)
    br, bi, cr, ci, ierr1 = fort_sp.vhaes(v, w, np.asarray(wvhaes, dtype=np.float32), lwork_vhaes, ityp=ityp)
    assert ierr1 == 0
    lshses, init_lwork, ldwork = calc_shses_sizes(nlat, nlon)
    wshses, ierr2 = fort_sp.shsesi(nlat, nlon, lshses, init_lwork, ldwork)
    assert ierr2 == 0
    lwork = find_valid_sfvpes_lwork(nlat, nlon, nt, isym, br, bi, cr, ci, wshses)
    if isym == 0:
        sf_f, vp_f, ierr_f = fort_sp.sfvpes(nlon, br, bi, cr, ci, np.asarray(wshses, dtype=np.float32), lwork)
        sf_r, vp_r, ierr_r = rust_sp.sfvpes(nlon, br, bi, cr, ci, np.asarray(wshses, dtype=np.float32), lwork)
    else:
        sf_f, vp_f, ierr_f = fort_sp.sfvpes(nlon, br, bi, cr, ci, np.asarray(wshses, dtype=np.float32), lwork, isym=isym)
        sf_r, vp_r, ierr_r = rust_sp.sfvpes_isym(nlon, isym, br, bi, cr, ci, np.asarray(wshses, dtype=np.float32), lwork)
    print(f"sfvpes case nlat={nlat} nlon={nlon} nt={nt} isym={isym} ierr_f={ierr_f} ierr_r={ierr_r} lwork={lwork} wshses_len={len(wshses)}")
    assert ierr_f == 0
    assert ierr_r == 0
    summarize_diff(f"sfvpes sf isym={isym}", sf_f, sf_r)
    summarize_diff(f"sfvpes vp isym={isym}", vp_f, vp_r)


def run_lwork_regression_case(nlat: int, nlon: int, nt: int, isym: int = 0):
    v, w = make_vector_grid(nlat, nlon, nt)
    lvhaes, init_lwork_vhaes, ldwork_vhaes = calc_vhaes_sizes(nlat, nlon)
    wvhaes, ierr0 = fort_sp.vhaesi(nlat, nlon, lvhaes, init_lwork_vhaes, ldwork_vhaes)
    assert ierr0 == 0
    ityp = {0: 0, 1: 3, 2: 6}[isym]
    lwork_vhaes = calc_vhaes_lwork(nlat, nlon, nt, ityp)
    br, bi, cr, ci, ierr1 = fort_sp.vhaes(v, w, np.asarray(wvhaes, dtype=np.float32), lwork_vhaes, ityp=ityp)
    assert ierr1 == 0
    lshses, init_lwork, ldwork = calc_shses_sizes(nlat, nlon)
    wshses, ierr2 = fort_sp.shsesi(nlat, nlon, lshses, init_lwork, ldwork)
    assert ierr2 == 0
    valid_lwork = find_valid_sfvpes_lwork(nlat, nlon, nt, isym, br, bi, cr, ci, wshses)
    wrong_lwork = max(valid_lwork - 1, 1)
    _, _, ierr_f = fort_sp.sfvpes(nlon, br, bi, cr, ci, np.asarray(wshses, dtype=np.float32), wrong_lwork)
    _, _, ierr_r = rust_sp.sfvpes(nlon, br, bi, cr, ci, np.asarray(wshses, dtype=np.float32), wrong_lwork)
    print(f"sfvpes regression case nlat={nlat} nlon={nlon} nt={nt} isym={isym} ierr_f={ierr_f} ierr_r={ierr_r} wrong_lwork={wrong_lwork} valid_lwork={valid_lwork}")
    assert ierr_f == 10
    assert ierr_r == 10


if __name__ == "__main__":
    for case in [(4, 8, 1, 0), (5, 8, 2, 0), (5, 8, 1, 1), (5, 8, 1, 2), (73, 144, 1, 0)]:
        run_case(*case)
    run_lwork_regression_case(4, 8, 1, 0)
