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


def calc_shsgc_sizes(nlat: int, nlon: int):
    l = min((nlon + 2) // 2, nlat)
    late = (nlat + (nlat % 2)) // 2
    lshsgc = nlat * (2 * late + 3 * l - 2) + 3 * l * (1 - l) // 2 + nlon + 15
    ldwork = nlat * (nlat + 4)
    return lshsgc, ldwork


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
    lshsgc, ldwork = calc_shsgc_sizes(nlat, nlon)
    wshsgc, ierr0 = rust_sp.shsgci(nlat, nlon, lshsgc, ldwork)
    assert ierr0 == 0
    a, b = make_coeffs(nlat, nt)
    n2 = (nlat + 1) // 2
    lwork = nlat * (nt * nlon + max(3 * n2, nlon))
    g_f, ierr_f = fort_sp.shsgc(nlon, a, b, np.asarray(wshsgc, dtype=np.float32), lwork)
    g_r, ierr_r = rust_sp.shsgc(a, b, np.asarray(wshsgc, dtype=np.float32), lwork)
    print(f"\n{'=' * 80}\nshsgc: nlat={nlat}, nlon={nlon}, nt={nt}\n{'=' * 80}")
    print("ierror fortran =", ierr_f)
    print("ierror rust    =", ierr_r)
    summarize_diff("shsgc g", g_f, g_r)


if __name__ == "__main__":
    for case in [(3, 4, 1), (4, 4, 1), (5, 8, 2), (73, 144, 1)]:
        run_case(*case)
