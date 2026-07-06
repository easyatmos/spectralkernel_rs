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


def calc_shags_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    lshags = nlat * (3 * (l1 + l2) - 2) + (l1 - 1) * (l2 * (2 * nlat - l1) - 3 * l1) // 2 + nlon + 15
    lwork = 4 * nlat * (nlat + 2) + 2
    ldwork = nlat * (nlat + 4)
    return lshags, lwork, ldwork


def calc_shsgs_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    lshsgs = nlat * (3 * (l1 + l2) - 2) + (l1 - 1) * (l2 * (2 * nlat - l1) - 3 * l1) // 2 + nlon + 15
    ldwork = nlat * (nlat + 4)
    return lshsgs, ldwork


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
    lshags, lwork_shagsi, ldwork_shagsi = calc_shags_sizes(nlat, nlon)
    wshags, ierr0 = fort_sp.shagsi(nlat, nlon, lshags, lwork_shagsi, ldwork_shagsi)
    assert ierr0 == 0, ("shagsi failed", nlat, nlon, ierr0)
    l2 = (nlat + 1) // 2
    if isym == 0:
        lwork_shags = nlat * (nt * nlon + max(3 * l2, nlon))
        a, b, ierr1 = fort_sp.shags(sf, np.asarray(wshags, dtype=np.float32), lwork_shags)
    else:
        lwork_shags = l2 * (nt * nlon + max(3 * nlat, nlon))
        a, b, ierr1 = fort_sp.shags(sf, np.asarray(wshags, dtype=np.float32), lwork_shags, isym=isym)
    assert ierr1 == 0, ("shags failed", nlat, nlon, nt, isym, ierr1)
    lshsgs, ldwork_shsgsi = calc_shsgs_sizes(nlat, nlon)
    init_lwork_shsgsi = 4 * nlat * (nlat + 2) + 2
    wshsgs, ierr2 = fort_sp.shsgsi(nlat, nlon, lshsgs, init_lwork_shsgsi, ldwork_shsgsi)
    assert ierr2 == 0, ("shsgsi failed", nlat, nlon, ierr2)
    l1 = min(nlat, (nlon + 2) // 2)
    if isym == 0:
        lwork_slapgs = (nt + 1) * nlat * nlon + nlat * (2 * nt * l1 + 1)
    else:
        lwork_slapgs = (nt + 1) * l2 * nlon + nlat * (2 * nt * l1 + 1)
    slap_f, ierr_f = fort_sp.slapgs(nlon, a, b, np.asarray(wshsgs, dtype=np.float32), lwork_slapgs, isym=isym)
    slap_r, ierr_r = rust_sp.slapgs(nlon, a, b, np.asarray(wshsgs, dtype=np.float32), lwork_slapgs, isym=isym)
    assert ierr_f == 0, ("fortran slapgs failed", nlat, nlon, nt, isym, ierr_f)
    assert ierr_r == 0, ("rust slapgs failed", nlat, nlon, nt, isym, ierr_r)
    summarize_diff("slapgs slap", slap_f, slap_r)


if __name__ == "__main__":
    for case in [(3, 4, 1, 0), (4, 4, 1, 0), (5, 8, 2, 0), (73, 144, 1, 0)]:
        run_case(*case)
