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


def calc_shaes_sizes(nlat: int, nlon: int):
    n1 = min(nlat, (nlon + 2) // 2)
    n2 = (nlat + 1) // 2
    lshaes = (n1 * n2 * (2 * nlat - n1 + 1)) // 2 + nlon + 15
    lwork = 5 * nlat * n2 + (3 * ((n1 - 2) * (2 * nlat - n1 - 1))) // 2
    ldwork = nlat + 1
    return lshaes, lwork, ldwork


def calc_shses_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    lshses = (l1 * l2 * (2 * nlat - l1 + 1)) // 2 + nlon + 15
    lwork = 5 * nlat * l2 + (3 * ((l1 - 2) * (2 * nlat - l1 - 1))) // 2
    ldwork = nlat + 1
    return lshses, lwork, ldwork


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
    sf = make_scalar_grid(nlat, nlon, nt)
    lshaes, lwork_shaesi, ldwork_shaesi = calc_shaes_sizes(nlat, nlon)
    wshaes, ierr0 = fort_sp.shaesi(nlat, nlon, lshaes, lwork_shaesi, ldwork_shaesi)
    assert ierr0 == 0, ("shaesi failed", nlat, nlon, ierr0)
    l2 = (nlat + 1) // 2
    if isym == 0:
        lwork_shaes = nlat * (2 * nt * nlon + max(6 * l2, nlon))
        a, b, ierr1 = fort_sp.shaes(sf, np.asarray(wshaes, dtype=np.float32), lwork_shaes)
    else:
        lwork_shaes = l2 * (2 * nt * nlon + max(6 * nlat, nlon))
        a, b, ierr1 = fort_sp.shaes(sf, np.asarray(wshaes, dtype=np.float32), lwork_shaes, isym=isym)
    assert ierr1 == 0, ("shaes failed", nlat, nlon, nt, isym, ierr1)
    lshses, init_lwork, ldwork = calc_shses_sizes(nlat, nlon)
    wshses, ierr2 = fort_sp.shsesi(nlat, nlon, lshses, init_lwork, ldwork)
    assert ierr2 == 0, ("shsesi failed", nlat, nlon, ierr2)
    l1 = min(nlat, (nlon + 2) // 2)
    if isym == 0:
        lwork_slapes = (nt + 1) * nlat * nlon + nlat * (2 * nt * l1 + 1)
    else:
        lwork_slapes = (nt + 1) * l2 * nlon + nlat * (2 * nt * l1 + 1)
    slap_f, ierr_f = fort_sp.slapes(nlon, a, b, np.asarray(wshses, dtype=np.float32), lwork_slapes, isym=isym)
    slap_r, ierr_r = rust_sp.slapes(nlon, a, b, np.asarray(wshses, dtype=np.float32), lwork_slapes, isym=isym)
    assert ierr_f == 0, ("fortran slapes failed", nlat, nlon, nt, isym, ierr_f)
    assert ierr_r == 0, ("rust slapes failed", nlat, nlon, nt, isym, ierr_r)
    summarize_diff("slapes slap", slap_f, slap_r)


if __name__ == "__main__":
    for case in [(3, 4, 1, 0), (4, 4, 1, 0), (5, 8, 2, 0), (73, 144, 1, 0)]:
        run_case(*case)
