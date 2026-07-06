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


def calc_shags_sizes(nlat: int, nlon: int):
    n1 = min(nlat, (nlon + 2) // 2)
    n2 = (nlat + 1) // 2
    lshags = nlat * (3 * (n1 + n2) - 2) + (n1 - 1) * (n2 * (2 * nlat - n1) - 3 * n1) // 2 + nlon + 15
    lwork = 4 * nlat * (nlat + 2) + 2
    ldwork = nlat * (nlat + 4)
    return lshags, lwork, ldwork


def calc_shsgs_sizes(nlat: int, nlon: int):
    return calc_shags_sizes(nlat, nlon)


def calc_vhags_sizes(nlat: int, nlon: int):
    lvhags = (nlat + 1) * (nlat + 1) * nlat // 2 + nlon + 15
    ldwork = (3 * nlat * (nlat + 3) + 2) // 2
    return lvhags, ldwork


def calc_vhsgs_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    lmn = nlat * (nlat + 1) // 2
    lvhsgs = 2 * (imid * lmn) + nlon + 15
    ldwork = (3 * nlat * (nlat + 3) + 2) // 2
    return lvhsgs, ldwork


def summarize_gaussian_segments(name, nlat, nlon, wf, wr):
    if name in ("shagsi", "shsgsi"):
        l = min((nlon + 2) // 2, nlat)
        late = (nlat + 1) // 2
        labc = l * (l - 1) // 2 + (nlat - l) * (l - 1)
        pcount = l * (2 * nlat - l + 1) // 2
        i1 = 0
        i2 = i1 + nlat
        i3 = i2 + nlat * late
        i4 = i3 + nlat * late
        i5 = i4 + labc
        i6 = i5 + labc
        i7 = i6 + labc
        i8 = i7 + nlon + 15
        summarize_diff(f"{name} wts", wf[i1:i2], wr[i1:i2])
        summarize_diff(f"{name} p0n", wf[i2:i3], wr[i2:i3])
        summarize_diff(f"{name} p1n", wf[i3:i4], wr[i3:i4])
        summarize_diff(f"{name} abel", wf[i4:i5], wr[i4:i5])
        summarize_diff(f"{name} bbel", wf[i5:i6], wr[i5:i6])
        summarize_diff(f"{name} cbel", wf[i6:i7], wr[i6:i7])
        summarize_diff(f"{name} hrffti", wf[i7:i8], wr[i7:i8])
        summarize_diff(f"{name} pmnf", wf[i8:i8 + late * pcount], wr[i8:i8 + late * pcount])
    elif name in ("vhagsi", "vhsgsi"):
        imid = (nlat + 1) // 2
        lmn = nlat * (nlat + 1) // 2
        block = imid * lmn
        summarize_diff(f"{name} vb", wf[:block], wr[:block])
        summarize_diff(f"{name} wb", wf[block:2 * block], wr[block:2 * block])
        summarize_diff(f"{name} hrffti", wf[2 * block:], wr[2 * block:])


def run_scalar_case(name, rust_func, fort_func, nlat, nlon, size_func):
    out_len, lwork, ldwork = size_func(nlat, nlon)
    w_f, ierr_f = fort_func(nlat, nlon, out_len, lwork, ldwork)
    w_r, ierr_r = rust_func(nlat, nlon, out_len, lwork, ldwork)
    print(f"\n{'=' * 80}\n{name}: nlat={nlat}, nlon={nlon}\n{'=' * 80}")
    print("output len =", out_len)
    print("lwork      =", lwork)
    print("ldwork     =", ldwork)
    print("ierror fortran =", ierr_f)
    print("ierror rust    =", ierr_r)
    summarize_diff(name, w_f, w_r)
    summarize_gaussian_segments(name, nlat, nlon, np.asarray(w_f).reshape(-1), np.asarray(w_r).reshape(-1))


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
    summarize_gaussian_segments(name, nlat, nlon, np.asarray(w_f).reshape(-1), np.asarray(w_r).reshape(-1))


def check_gaussian_stored_init():
    cases = [
        (3, 4),
        (4, 4),
        (5, 8),
        (8, 8),
        (9, 16),
        (73, 144),
    ]

    for nlat, nlon in cases:
        run_scalar_case("shagsi", rust_sp.shagsi, fort_sp.shagsi, nlat, nlon, calc_shags_sizes)
        run_scalar_case("shsgsi", rust_sp.shsgsi, fort_sp.shsgsi, nlat, nlon, calc_shsgs_sizes)
        run_vector_case("vhagsi", rust_sp.vhagsi, fort_sp.vhagsi, nlat, nlon, calc_vhags_sizes)
        run_vector_case("vhsgsi", rust_sp.vhsgsi, fort_sp.vhsgsi, nlat, nlon, calc_vhsgs_sizes)


if __name__ == "__main__":
    check_gaussian_stored_init()
