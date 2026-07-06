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


def calc_shagc_sizes(nlat: int, nlon: int):
    l = min((nlon + 2) // 2, nlat)
    late = (nlat + (nlat % 2)) // 2
    lshagc = nlat * (2 * late + 3 * l - 2) + 3 * l * (1 - l) // 2 + nlon + 15
    ldwork = nlat * (nlat + 4)
    return lshagc, ldwork


def calc_shsgc_sizes(nlat: int, nlon: int):
    return calc_shagc_sizes(nlat, nlon)


def calc_vhagc_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    lzz1 = 2 * nlat * imid
    mmax = min(nlat, (nlon + 1) // 2)
    labc = 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2
    lvhagc = 2 * (lzz1 + labc) + nlon + imid + 15
    ldwork = 2 * nlat * (nlat + 1) + 1
    return lvhagc, ldwork


def calc_vhsgc_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    lzz1 = 2 * nlat * imid
    mmax = min(nlat, (nlon + 1) // 2)
    labc = 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2
    lvhsgc = 2 * (lzz1 + labc) + nlon + 15
    ldwork = 2 * nlat * (nlat + 1) + 1
    return lvhsgc, ldwork


def run_scalar_case(name, rust_func, fort_func, nlat, nlon, size_func):
    out_len, ldwork = size_func(nlat, nlon)
    w_f, ierr_f = fort_func(nlat, nlon, out_len, ldwork)
    w_r, ierr_r = rust_func(nlat, nlon, out_len, ldwork)
    print(f"\n{'=' * 80}\n{name}: nlat={nlat}, nlon={nlon}\n{'=' * 80}")
    print("output len =", out_len)
    print("ldwork     =", ldwork)
    print("ierror fortran =", ierr_f)
    print("ierror rust    =", ierr_r)
    summarize_diff(name, w_f, w_r)


def run_vector_case(name, rust_func, fort_func, nlat, nlon, size_func):
    out_len, ldwork = size_func(nlat, nlon)
    w_f, ierr_f = fort_func(nlat, nlon, out_len, ldwork)
    w_r, ierr_r = rust_func(nlat, nlon, out_len, ldwork)
    print(f"\n{'=' * 80}\n{name}: nlat={nlat}, nlon={nlon}\n{'=' * 80}")
    print("output len =", out_len)
    print("ldwork     =", ldwork)
    print("ierror fortran =", ierr_f)
    print("ierror rust    =", ierr_r)
    summarize_diff(name, w_f, w_r)


def check_gaussian_computed_init():
    cases = [(3, 4), (4, 4), (5, 8), (8, 8), (9, 16), (73, 144),]
    for nlat, nlon in cases:
        run_scalar_case("shagci", rust_sp.shagci, fort_sp.shagci, nlat, nlon, calc_shagc_sizes)
        run_scalar_case("shsgci", rust_sp.shsgci, fort_sp.shsgci, nlat, nlon, calc_shsgc_sizes)
        run_vector_case("vhagci", rust_sp.vhagci, fort_sp.vhagci, nlat, nlon, calc_vhagc_sizes)
        run_vector_case("vhsgci", rust_sp.vhsgci, fort_sp.vhsgci, nlat, nlon, calc_vhsgc_sizes)


if __name__ == "__main__":
    check_gaussian_computed_init()
