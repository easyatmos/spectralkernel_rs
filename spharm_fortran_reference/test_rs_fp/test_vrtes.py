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
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    lshses = (l1 * l2 * (2 * nlat - l1 + 1)) // 2 + nlon + 15
    lwork = 3 * max(l1 - 2, 0) * (2 * nlat - l1 - 1) // 2 + 5 * l2 * nlat
    ldwork = nlat + 1
    return lshses, lwork, ldwork


def make_vec_grid(nlat: int, nlon: int, nt: int):
    theta = np.linspace(0.0, np.pi, nlat, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    v = np.zeros((nlat, nlon, nt), dtype=np.float32)
    w = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        v[:, :, k] = (
            np.cos(theta)[:, None] * np.cos((k + 1) * lon)[None, :]
            + 0.2 * np.sin(theta)[:, None] * np.sin((k + 2) * lon)[None, :]
        )
        w[:, :, k] = (
            np.sin(theta)[:, None] * np.sin((k + 1) * lon)[None, :]
            + 0.15 * np.cos(theta)[:, None] * np.cos((k + 3) * lon)[None, :]
        )
    return v, w


def run_case(nlat: int, nlon: int, nt: int, isym: int = 0):
    v, w = make_vec_grid(nlat, nlon, nt)
    lvhaes, init_lwork, ldwork = calc_vhaes_sizes(nlat, nlon)
    wvhaes, ierr0 = fort_sp.vhaesi(nlat, nlon, lvhaes, init_lwork, ldwork)
    assert ierr0 == 0, ("vhaesi failed", nlat, nlon, ierr0)
    lwork_vhaes = (2 * nt + 1) * nlat * nlon
    _br, _bi, cr, ci, ierr1 = fort_sp.vhaes(v, w, np.asarray(wvhaes, dtype=np.float32), lwork_vhaes)
    assert ierr1 == 0, ("vhaes failed", nlat, nlon, nt, ierr1)
    lshses, init_lwork_shses, ldwork_shses = calc_shses_sizes(nlat, nlon)
    wshses, ierr2 = fort_sp.shsesi(nlat, nlon, lshses, init_lwork_shses, ldwork_shses)
    assert ierr2 == 0, ("shsesi failed", nlat, nlon, ierr2)
    l2 = (nlat + 1) // 2
    l1 = min(nlat, (nlon + 2) // 2)
    if isym == 0:
        lwork_vrtes = nlat * ((nt + 1) * nlon + 2 * nt * l1 + 1)
    else:
        lwork_vrtes = (nt + 1) * l2 * nlon + nlat * (2 * nt * l1 + 1)
    vort_f, ierr_f = fort_sp.vrtes(nlon, cr, ci, np.asarray(wshses, dtype=np.float32), lwork_vrtes, isym=isym)
    vort_r, ierr_r = rust_sp.vrtes(nlon, cr, ci, np.asarray(wshses, dtype=np.float32), lwork_vrtes, isym=isym)
    assert ierr_f == 0, ("fortran vrtes failed", nlat, nlon, nt, isym, ierr_f)
    assert ierr_r == 0, ("rust vrtes failed", nlat, nlon, nt, isym, ierr_r)
    summarize_diff("vrtes vort", vort_f, vort_r)


if __name__ == "__main__":
    for case in [(3, 4, 1, 0), (4, 4, 1, 0), (5, 8, 2, 0), (73, 144, 1, 0)]:
        run_case(*case)
