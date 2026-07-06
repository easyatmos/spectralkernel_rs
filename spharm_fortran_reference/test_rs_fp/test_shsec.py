import numpy as np

import spectralkernel_rs as rust_sp
from spharm_fortran_reference.pyspharm import _spherepack as fort_sp


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
    print("fortran    :", a[idx])
    print("rust       :", b[idx])
    print("diff       :", diff[idx])


def calc_shsec_sizes(nlat: int, nlon: int):
    n1 = min(nlat, (nlon + 2) // 2)
    n2 = (nlat + 1) // 2
    lshsec = 2 * nlat * n2 + 3 * ((n1 - 2) * (2 * nlat - n1 - 1)) // 2 + nlon + 15
    ldwork = nlat + 1
    return lshsec, ldwork


def make_coeffs(nlat: int, nt: int):
    a = np.zeros((nlat, nlat, nt), dtype=np.float32)
    b = np.zeros((nlat, nlat, nt), dtype=np.float32)
    for k in range(nt):
        for m in range(nlat):
            for n in range(m, nlat):
                a[m, n, k] = (m + 1) * 0.1 + (n + 1) * 0.03 + k * 0.07
                b[m, n, k] = (m + 1) * 0.05 - (n + 1) * 0.02 + k * 0.04
    return a, b


def run_case(nlat: int, nlon: int, nt: int):
    lshsec, ldwork = calc_shsec_sizes(nlat, nlon)
    wshsec, ierr0 = rust_sp.shseci(nlat, nlon, lshsec, ldwork)
    assert ierr0 == 0
    a, b = make_coeffs(nlat, nt)
    lwork = nt * nlat * nlon + max(nlat * nlon, 3 * nlat * ((nlat + 1) // 2))
    g_f, ierr_f = fort_sp.shsec(nlon, a, b, np.asarray(wshsec, dtype=np.float32), lwork)
    g_r, ierr_r = rust_sp.shsec(a, b, np.asarray(wshsec, dtype=np.float32), lwork)
    print(f"\n{'=' * 80}\nshsec: nlat={nlat}, nlon={nlon}, nt={nt}\n{'=' * 80}")
    print("ierror fortran =", ierr_f)
    print("ierror rust    =", ierr_r)
    summarize_diff("shsec g", g_f, g_r)


if __name__ == "__main__":
    for case in [(3, 4, 1), (4, 4, 1), (5, 8, 2), (73, 144, 1)]:
        run_case(*case)
