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


def calc_shagc_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    lshagc = nlat * (2 * l2 + 3 * l1 - 2) + 3 * l1 * (1 - l1) // 2 + nlon + 15
    ldwork = nlat * (nlat + 4)
    return lshagc, ldwork


def calc_shsgc_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    lshsgc = nlat * (2 * l2 + 3 * l1 - 2) + 3 * l1 * (1 - l1) // 2 + nlon + 15
    ldwork = nlat * (nlat + 4)
    return lshsgc, ldwork


def make_scalar_grid(nlat: int, nlon: int, nt: int):
    theta = np.linspace(0.0, np.pi, nlat, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    sf = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        sf[:, :, k] = (
            np.cos(theta)[:, None] * np.cos((k + 1) * lon)[None, :]
            + 0.2 * np.sin(theta)[:, None] * np.sin((k + 2) * lon)[None, :]
            + 0.05 * np.cos(2.0 * theta)[:, None]
        )
    return sf


def run_case(nlat: int, nlon: int, nt: int, isym: int = 0):
    sf_true = make_scalar_grid(nlat, nlon, nt)
    xlmbda = np.zeros((nt,), dtype=np.float32)

    lshagc, ldwork_shagci = calc_shagc_sizes(nlat, nlon)
    wshagc, ierr0 = fort_sp.shagci(nlat, nlon, lshagc, ldwork_shagci)
    assert ierr0 == 0, ("shagci failed", nlat, nlon, ierr0)

    l2 = (nlat + 1) // 2
    if isym == 0:
        lwork_shagc = nlat * (nt * nlon + max(3 * l2, nlon))
        a, b, ierr1 = fort_sp.shagc(sf_true, np.asarray(wshagc, dtype=np.float32), lwork_shagc)
    else:
        lwork_shagc = l2 * (nt * nlon + max(3 * nlat, nlon))
        a, b, ierr1 = fort_sp.shagc(sf_true, np.asarray(wshagc, dtype=np.float32), lwork_shagc, isym=isym)
    assert ierr1 == 0, ("shagc failed", nlat, nlon, nt, isym, ierr1)

    a = a.copy()
    b = b.copy()
    a[0, 0, :] = 0.0
    b[0, 0, :] = 0.0

    lshsgc, ldwork_shsgci = calc_shsgc_sizes(nlat, nlon)
    wshsgc, ierr2 = fort_sp.shsgci(nlat, nlon, lshsgc, ldwork_shsgci)
    assert ierr2 == 0, ("shsgci failed", nlat, nlon, ierr2)

    l1 = min(nlat, (nlon + 2) // 2)
    if isym == 0:
        lwork_islapgc = nlat * (2 * nt * nlon + max(6 * l2, nlon) + 2 * l1 * nt + 1)
        sf_f, p_f, ierr_f = fort_sp.islapgc(nlon, xlmbda, a, b, np.asarray(wshsgc, dtype=np.float32), lwork_islapgc)
        sf_r, p_r, ierr_r = rust_sp.islapgc(nlon, xlmbda, a, b, np.asarray(wshsgc, dtype=np.float32), lwork_islapgc)
    else:
        lwork_islapgc = l2 * (2 * nt * nlon + max(6 * nlat, nlon)) + nlat * (2 * l1 * nt + 1)
        sf_f, p_f, ierr_f = fort_sp.islapgc(nlon, xlmbda, a, b, np.asarray(wshsgc, dtype=np.float32), lwork_islapgc, isym=isym)
        sf_r, p_r, ierr_r = rust_sp.islapgc_isym(nlon, xlmbda, a, b, isym, np.asarray(wshsgc, dtype=np.float32), lwork_islapgc)

    assert ierr_f == 0, ("fortran islapgc failed", nlat, nlon, nt, isym, ierr_f)
    assert ierr_r == 0, ("rust islapgc failed", nlat, nlon, nt, isym, ierr_r)
    summarize_diff("islapgc sf", sf_f, sf_r)
    summarize_diff("islapgc pertrb", p_f, p_r)


if __name__ == "__main__":
    for case in [(3, 4, 1, 0), (4, 4, 1, 0), (5, 8, 2, 0), (73, 144, 1, 0)]:
        run_case(*case)
