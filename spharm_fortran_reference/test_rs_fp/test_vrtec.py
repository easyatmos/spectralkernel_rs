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
    idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    print("worst index:", idx)
    print("lhs        :", a[idx])
    print("rhs        :", b[idx])
    print("diff       :", diff[idx])


def calc_vhaec_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 1) // 2)
    l2 = (nlat + 1) // 2
    lvhaec = 4 * nlat * l2 + 3 * max(l1 - 2, 0) * (2 * nlat - l1 - 1) + nlon + 15
    ldwork = 2 * (nlat + 1)
    return lvhaec, ldwork


def calc_shsec_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    lshsec = 2 * nlat * l2 + 3 * (max(l1 - 2, 0) * (2 * nlat - l1 - 1)) // 2 + nlon + 15
    ldwork = 2 * (nlat + 1)
    return lshsec, ldwork


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


def run_case(nlat: int, nlon: int, nt: int):
    v, w = make_vec_grid(nlat, nlon, nt)

    lvhaec, ldwork_vhaeci = calc_vhaec_sizes(nlat, nlon)
    wvhaec, ierr0 = fort_sp.vhaeci(nlat, nlon, lvhaec, ldwork_vhaeci)
    assert ierr0 == 0, ("vhaeci failed", nlat, nlon, ierr0)

    lwork_vhaec = nlat * (4 * nlon * nt + max(6 * ((nlat + 1) // 2), nlon))
    _br, _bi, cr, ci, ierr1 = fort_sp.vhaec(v, w, np.asarray(wvhaec, dtype=np.float32), lwork_vhaec)
    assert ierr1 == 0, ("vhaec failed", nlat, nlon, nt, ierr1)

    lshsec, ldwork_shseci = calc_shsec_sizes(nlat, nlon)
    wshsec, ierr2 = fort_sp.shseci(nlat, nlon, lshsec, ldwork_shseci)
    assert ierr2 == 0, ("shseci failed", nlat, nlon, ierr2)

    l2 = (nlat + 1) // 2
    l1 = min(nlat, (nlon + 2) // 2)
    lwork_vrtec = nlat * (nt * nlon + max(3 * l2, nlon) + 2 * nt * l1 + 1)
    vort_f, ierr_f = fort_sp.vrtec(nlon, cr, ci, np.asarray(wshsec, dtype=np.float32), lwork_vrtec)
    vort_r, ierr_r = rust_sp.vrtec(cr, ci, np.asarray(wshsec, dtype=np.float32), lwork_vrtec)

    assert ierr_f == 0, ("fortran vrtec failed", nlat, nlon, nt, ierr_f)
    assert ierr_r == 0, ("rust vrtec failed", nlat, nlon, nt, ierr_r)

    print(f"\n{'=' * 80}\nvrtec: nlat={nlat}, nlon={nlon}, nt={nt}\n{'=' * 80}")
    summarize_diff("vrtec vort: fortran vs rust", vort_f, vort_r)


if __name__ == "__main__":
    for case in [(3, 4, 1), (4, 4, 1), (5, 8, 2), (73, 144, 1)]:
        run_case(*case)
