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


def calc_vhags_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    lmn = nlat * (nlat + 1) // 2
    mmax = min(nlat, (nlon + 1) // 2)
    idz = mmax * (2 * nlat - mmax + 1) // 2
    lzimn = idz * imid
    lvhags = max(2 * (imid * lmn) + nlon + 15, 2 * lzimn + nlon + 15)
    ldwork = (nlat * (3 * nlat + 9) + 2) // 2
    return lvhags, ldwork


def calc_shsgs_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    lshsgs = nlat * (3 * (l1 + l2) - 2) + (l1 - 1) * (l2 * (2 * nlat - l1) - 3 * l1) // 2 + nlon + 15
    ldwork = nlat * (nlat + 4)
    return lshsgs, ldwork


def calc_vhags_lwork(nlat: int, nlon: int, nt: int, ityp: int):
    imid = (nlat + 1) // 2
    if ityp <= 2:
        return nlat * (4 * nlon * nt + 6 * imid)
    return imid * (4 * nlon * nt + 6 * nlat)


def calc_sfvpgs_lwork(nlat: int, nlon: int, nt: int, isym: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    if isym == 0:
        return nlat * nlon * (nt + 1) + nlat * (2 * l1 * nt + 1)
    return l2 * nlon * (nt + 1) + nlat * (2 * l1 * nt + 1)


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


def crop_half_sphere(data, nlat: int, isym: int):
    if isym == 0:
        return np.asarray(data)
    idv = (nlat + 1) // 2
    return np.asarray(data)[:idv, ...]


def run_case(nlat: int, nlon: int, nt: int, isym: int = 0):
    v, w = make_vector_grid(nlat, nlon, nt)
    lvhags, ldwork_vhags = calc_vhags_sizes(nlat, nlon)
    wvhags, ierr0 = fort_sp.vhagsi(nlat, nlon, lvhags, ldwork_vhags)
    assert ierr0 == 0
    ityp = {0: 0, 1: 3, 2: 6}[isym]
    lwork_vhags = calc_vhags_lwork(nlat, nlon, nt, ityp)
    br, bi, cr, ci, ierr1 = fort_sp.vhags(v, w, np.asarray(wvhags, dtype=np.float32), lwork_vhags, ityp=ityp)
    assert ierr1 == 0
    lshsgs, ldwork = calc_shsgs_sizes(nlat, nlon)
    init_lwork = 4 * nlat * (nlat + 2) + 2
    wshsgs, ierr2 = fort_sp.shsgsi(nlat, nlon, lshsgs, init_lwork, ldwork)
    assert ierr2 == 0
    lwork = calc_sfvpgs_lwork(nlat, nlon, nt, isym)
    if isym == 0:
        sf_f, vp_f, ierr_f = fort_sp.sfvpgs(nlon, br, bi, cr, ci, np.asarray(wshsgs, dtype=np.float32), lwork)
        sf_r, vp_r, ierr_r = rust_sp.sfvpgs(nlon, br, bi, cr, ci, np.asarray(wshsgs, dtype=np.float32), lwork)
    else:
        sf_f, vp_f, ierr_f = fort_sp.sfvpgs(nlon, br, bi, cr, ci, np.asarray(wshsgs, dtype=np.float32), lwork, isym=isym)
        sf_r, vp_r, ierr_r = rust_sp.sfvpgs_isym(nlon, isym, br, bi, cr, ci, np.asarray(wshsgs, dtype=np.float32), lwork)
    print(f"sfvpgs case nlat={nlat} nlon={nlon} nt={nt} isym={isym} ierr_f={ierr_f} ierr_r={ierr_r} lwork={lwork} wshsgs_len={len(wshsgs)}")
    assert ierr_f == 0
    assert ierr_r == 0
    summarize_diff(f"sfvpgs sf isym={isym}", crop_half_sphere(sf_f, nlat, isym), sf_r)
    summarize_diff(f"sfvpgs vp isym={isym}", crop_half_sphere(vp_f, nlat, isym), vp_r)


if __name__ == "__main__":
    for case in [(4, 8, 1, 0), (5, 8, 2, 0), (5, 8, 1, 1), (5, 8, 1, 2), (73, 144, 1, 0)]:
        run_case(*case)
