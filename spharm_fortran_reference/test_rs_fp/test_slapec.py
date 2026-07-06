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


def calc_shaec_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    lshaec = 2 * nlat * l2 + 3 * max(l1 - 2, 0) * (2 * nlat - l1 - 1) // 2 + nlon + 15
    ldwork = 2 * (nlat + 1)
    return lshaec, ldwork


def calc_shsec_sizes(nlat: int, nlon: int):
    n1 = min(nlat, (nlon + 2) // 2)
    n2 = (nlat + 1) // 2
    lshsec = 2 * nlat * n2 + 3 * ((n1 - 2) * (2 * nlat - n1 - 1)) // 2 + nlon + 15
    ldwork = nlat + 1
    return lshsec, ldwork


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
    lshaec, ldwork_shaeci = calc_shaec_sizes(nlat, nlon)
    wshaec, ierr0 = fort_sp.shaeci(nlat, nlon, lshaec, ldwork_shaeci)
    assert ierr0 == 0, ("shaeci failed", nlat, nlon, ierr0)
    l2 = (nlat + 1) // 2
    if isym == 0:
        lwork_shaec = nlat * (2 * nt * nlon + max(6 * l2, nlon))
        a, b, ierr1 = fort_sp.shaec(sf, np.asarray(wshaec, dtype=np.float32), lwork_shaec)
    else:
        lwork_shaec = l2 * (2 * nt * nlon + max(6 * nlat, nlon))
        a, b, ierr1 = fort_sp.shaec(sf, np.asarray(wshaec, dtype=np.float32), lwork_shaec, isym=isym)
    assert ierr1 == 0, ("shaec failed", nlat, nlon, nt, isym, ierr1)
    lshsec, ldwork_shseci = calc_shsec_sizes(nlat, nlon)
    wshsec, ierr2 = fort_sp.shseci(nlat, nlon, lshsec, ldwork_shseci)
    assert ierr2 == 0, ("shseci failed", nlat, nlon, ierr2)
    l1 = min(nlat, (nlon + 2) // 2)
    if isym == 0:
        lwork_slapec = nlat * (2 * nt * nlon + max(6 * l2, nlon) + 2 * nt * l1 + 1)
    else:
        lwork_slapec = l2 * (2 * nt * nlon + max(6 * nlat, nlon)) + nlat * (2 * nt * l1 + 1)
    slap_f, ierr_f = fort_sp.slapec(nlon, a, b, np.asarray(wshsec, dtype=np.float32), lwork_slapec, isym=isym)
    slap_r, ierr_r = rust_sp.slapec(nlon, a, b, np.asarray(wshsec, dtype=np.float32), lwork_slapec, isym=isym)
    assert ierr_f == 0, ("fortran slapec failed", nlat, nlon, nt, isym, ierr_f)
    assert ierr_r == 0, ("rust slapec failed", nlat, nlon, nt, isym, ierr_r)
    summarize_diff("slapec slap", slap_f, slap_r)


if __name__ == "__main__":
    for case in [(3, 4, 1, 0), (4, 4, 1, 0), (5, 8, 2, 0), (73, 144, 1, 0)]:
        run_case(*case)
