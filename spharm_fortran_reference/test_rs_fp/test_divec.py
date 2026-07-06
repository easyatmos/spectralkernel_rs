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
    imid = (nlat + 1) // 2
    lzz1 = 2 * nlat * imid
    mmax = min(nlat, (nlon + 1) // 2)
    labc = 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2
    lvhaec = 2 * (lzz1 + labc) + nlon + 15
    ldwork = 2 * nlat + 2
    return lvhaec, ldwork


def calc_shsec_sizes(nlat: int, nlon: int):
    n1 = min(nlat, (nlon + 2) // 2)
    n2 = (nlat + 1) // 2
    lshsec = 2 * nlat * n2 + 3 * ((n1 - 2) * (2 * nlat - n1 - 1)) // 2 + nlon + 15
    ldwork = nlat + 1
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


def run_case(nlat: int, nlon: int, nt: int, isym: int = 0):
    v, w = make_vec_grid(nlat, nlon, nt)

    lvhaec, ldwork_vhaeci = calc_vhaec_sizes(nlat, nlon)
    wvhaec, ierr0 = fort_sp.vhaeci(nlat, nlon, lvhaec, ldwork_vhaeci)
    assert ierr0 == 0, ("vhaeci failed", nlat, nlon, ierr0)

    lwork_vhaec = nlat * (4 * nt * nlon + max(6 * ((nlat + 1) // 2), nlon))
    br, bi, _cr, _ci, ierr1 = fort_sp.vhaec(v, w, np.asarray(wvhaec, dtype=np.float32), lwork_vhaec)
    assert ierr1 == 0, ("vhaec failed", nlat, nlon, nt, ierr1)

    lshsec, ldwork_shseci = calc_shsec_sizes(nlat, nlon)
    wshsec, ierr2 = fort_sp.shseci(nlat, nlon, lshsec, ldwork_shseci)
    assert ierr2 == 0, ("shseci failed", nlat, nlon, ierr2)

    l2 = (nlat + 1) // 2
    l1 = min(nlat, (nlon + 2) // 2)
    if isym == 0:
        lwork_divec = nlat * (nt * nlon + max(3 * l2, nlon) + 2 * nt * l1 + 1)
    else:
        lwork_divec = l2 * (nt * nlon + max(3 * nlat, nlon)) + nlat * (2 * nt * l1 + 1)

    dv_f, ierr_f = fort_sp.divec(nlon, br, bi, np.asarray(wshsec, dtype=np.float32), lwork_divec, isym=isym)
    dv_r, ierr_r = rust_sp.divec(nlon, br, bi, np.asarray(wshsec, dtype=np.float32), lwork_divec, isym=isym)

    assert ierr_f == 0, ("fortran divec failed", nlat, nlon, nt, isym, ierr_f)
    assert ierr_r == 0, ("rust divec failed", nlat, nlon, nt, isym, ierr_r)

    print(f"\n{'=' * 80}\ndivec: nlat={nlat}, nlon={nlon}, nt={nt}, isym={isym}\n{'=' * 80}")
    summarize_diff("divec dv: fortran vs rust", dv_f, dv_r)


if __name__ == "__main__":
    for case in [(3, 4, 1, 0), (4, 4, 1, 0), (5, 8, 2, 0), (73, 144, 1, 0)]:
        run_case(*case)
