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


def calc_vhaes_sizes(nlat: int, nlon: int):
    n1 = min(nlat, (nlon + 1) // 2)
    n2 = (nlat + 1) // 2
    lvhaes = n1 * n2 * (2 * nlat - n1 + 1) + nlon + 15
    lwork = 3 * (max(n1 - 2, 0) * (2 * nlat - n1 - 1)) // 2 + 5 * n2 * nlat
    ldwork = 2 * (nlat + 1)
    return lvhaes, lwork, ldwork


def calc_shses_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    mmax = min(nlat, (nlon + 2) // 2)
    lpimn = (imid * mmax * (2 * nlat - mmax + 1)) // 2
    lshses = lpimn + nlon + 15
    l1 = min(nlat, (nlon + 2) // 2)
    lwork = nlat * ((2 * 1 + 1) * nlon + 2 * l1 * 1 + 1)
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
            + 0.15 * np.sin(theta)[:, None] * np.sin((k + 2) * lon)[None, :]
        )
        w[:, :, k] = (
            np.sin(theta)[:, None] * np.sin((k + 1) * lon)[None, :]
            + 0.1 * np.cos(theta)[:, None] * np.cos((k + 3) * lon)[None, :]
        )
    return v, w


def run_case(nlat: int, nlon: int, nt: int):
    v, w = make_vec_grid(nlat, nlon, nt)

    lvhaes, lwork_vhaesi, ldwork_vhaesi = calc_vhaes_sizes(nlat, nlon)
    wvhaes, ierr0 = fort_sp.vhaesi(nlat, nlon, lvhaes, lwork_vhaesi, ldwork_vhaesi)
    assert ierr0 == 0, ("vhaesi failed", nlat, nlon, ierr0)

    lwork_vhaes = (2 * nt + 1) * nlat * nlon
    br, bi, _cr, _ci, ierr1 = fort_sp.vhaes(v, w, np.asarray(wvhaes, dtype=np.float32), lwork_vhaes)
    assert ierr1 == 0, ("vhaes failed", nlat, nlon, nt, ierr1)

    lshses, lwork_shsesi, ldwork_shsesi = calc_shses_sizes(nlat, nlon)
    wshses, ierr2 = fort_sp.shsesi(nlat, nlon, lshses, lwork_shsesi, ldwork_shsesi)
    assert ierr2 == 0, ("shsesi failed", nlat, nlon, ierr2)

    lwork_igrades = nlat * ((nt + 1) * nlon + 2 * nt * min(nlat, (nlon + 2) // 2) + 1)
    sf_f, ierr_f = fort_sp.igrades(nlon, br, bi, np.asarray(wshses, dtype=np.float32), lwork_igrades)
    sf_r, ierr_r = rust_sp.igrades(br, bi, np.asarray(wshses, dtype=np.float32), lwork_igrades)

    assert ierr_f == 0, ("fortran igrades failed", nlat, nlon, nt, ierr_f)
    assert ierr_r == 0, ("rust igrades failed", nlat, nlon, nt, ierr_r)

    print(f"\n{'=' * 80}\nigrades: nlat={nlat}, nlon={nlon}, nt={nt}\n{'=' * 80}")
    summarize_diff("igrades sf: fortran vs rust", sf_f, sf_r)


if __name__ == "__main__":
    for case in [(3, 4, 1), (4, 4, 1), (5, 8, 2), (73, 144, 1)]:
        run_case(*case)
