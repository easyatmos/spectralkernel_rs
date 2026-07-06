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


def calc_shaec_sizes(nlat: int, nlon: int):
    n1 = min(nlat, (nlon + 2) // 2)
    n2 = (nlat + 1) // 2
    lshaec = 2 * nlat * n2 + 3 * ((n1 - 2) * (2 * nlat - n1 - 1)) // 2 + nlon + 15
    ldwork = nlat + 1
    return lshaec, ldwork


def calc_shsec_sizes(nlat: int, nlon: int):
    return calc_shaec_sizes(nlat, nlon)


def calc_vhaec_sizes(nlat: int, nlon: int):
    n1 = min(nlat, (nlon + 1) // 2)
    n2 = (nlat + 1) // 2
    lvhaec = 4 * nlat * n2 + 3 * max(n1 - 2, 0) * (2 * nlat - n1 - 1) + nlon + 15
    ldwork = 2 * (nlat + 1)
    return lvhaec, ldwork


def calc_vhsec_sizes(nlat: int, nlon: int):
    return calc_vhaec_sizes(nlat, nlon)


def run_case(name, rust_func, fort_func, nlat, nlon, size_func):
    out_len, ldwork = size_func(nlat, nlon)
    w_f, ierr_f = fort_func(nlat, nlon, out_len, ldwork)
    w_r, ierr_r = rust_func(nlat, nlon, out_len, ldwork)

    print(f"\n{'=' * 80}\n{name}: nlat={nlat}, nlon={nlon}\n{'=' * 80}")
    print("output len =", out_len)
    print("ldwork     =", ldwork)
    print("ierror fortran =", ierr_f)
    print("ierror rust    =", ierr_r)
    summarize_diff(name, w_f, w_r)
    wf = np.asarray(w_f).reshape(-1)
    wr = np.asarray(w_r).reshape(-1)
    print("prefix hrffti split:")
    summarize_init_segments(name, nlat, nlon, wf, wr)
    print("fortran finite =", np.isfinite(w_f).all())
    print("rust finite    =", np.isfinite(w_r).all())


def summarize_init_segments(name, nlat, nlon, wf, wr):
    if name in ("shaeci", "shseci"):
        imid = (nlat + 1) // 2
        mmax = min(nlat, nlon // 2 + 1)
        lzz1 = 2 * nlat * imid
        labc = 3 * ((mmax - 2) * (2 * nlat - mmax - 1)) // 2
        summarize_diff(f"{name} base(2*nlat*imid)", wf[:lzz1], wr[:lzz1])
        summarize_diff(f"{name} abc", wf[lzz1:lzz1 + labc], wr[lzz1:lzz1 + labc])
        summarize_diff(f"{name} hrffti", wf[lzz1 + labc:], wr[lzz1 + labc:])
    elif name in ("vhaeci", "vhseci"):
        imid = (nlat + 1) // 2
        mmax = min(nlat, (nlon + 1) // 2)
        lzz1 = 2 * nlat * imid
        labc = 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2
        block = lzz1 + labc
        summarize_diff(f"{name} block1", wf[:block], wr[:block])
        summarize_diff(f"{name} block2", wf[block:2 * block], wr[block:2 * block])
        summarize_diff(f"{name} hrffti", wf[2 * block:], wr[2 * block:])


def check_regular_computed_init():
    cases = [
        (3, 4),
        (4, 4),
        (5, 8),
        (8, 8),
        (9, 16),
    ]

    for nlat, nlon in cases:
        run_case("shaeci", rust_sp.shaeci, fort_sp.shaeci, nlat, nlon, calc_shaec_sizes)
        run_case("shseci", rust_sp.shseci, fort_sp.shseci, nlat, nlon, calc_shsec_sizes)
        run_case("vhaeci", rust_sp.vhaeci, fort_sp.vhaeci, nlat, nlon, calc_vhaec_sizes)
        run_case("vhseci", rust_sp.vhseci, fort_sp.vhseci, nlat, nlon, calc_vhsec_sizes)


def check_error_paths():
    error_cases = {
        "shaeci": (
            rust_sp.shaeci,
            fort_sp.shaeci,
            [
                (2, 4, 10, 3),
                (3, 3, 10, 4),
                (4, 4, 1, 5),
                (4, 4, 100, 1),
            ],
        ),
        "shseci": (
            rust_sp.shseci,
            fort_sp.shseci,
            [
                (2, 4, 10, 3),
                (3, 3, 10, 4),
                (4, 4, 1, 5),
                (4, 4, 100, 1),
            ],
        ),
        "vhaeci": (
            rust_sp.vhaeci,
            fort_sp.vhaeci,
            [
                (2, 4, 10, 3),
                (3, 0, 10, 8),
                (4, 4, 1, 10),
                (4, 4, 100, 1),
            ],
        ),
        "vhseci": (
            rust_sp.vhseci,
            fort_sp.vhseci,
            [
                (2, 4, 10, 3),
                (3, 0, 10, 8),
                (4, 4, 1, 10),
                (4, 4, 100, 1),
            ],
        ),
    }

    for name, (rust_func, fort_func, cases) in error_cases.items():
        print(f"\n{'=' * 80}\n{name} ERROR PATHS\n{'=' * 80}")
        for nlat, nlon, out_len, ldwork in cases:
            w_f, ierr_f = fort_func(nlat, nlon, out_len, ldwork)
            w_r, ierr_r = rust_func(nlat, nlon, out_len, ldwork)
            print(f"case nlat={nlat}, nlon={nlon}, out_len={out_len}, ldwork={ldwork}")
            print("ierror fortran =", ierr_f)
            print("ierror rust    =", ierr_r)
            print("len(fortran)   =", len(np.asarray(w_f).reshape(-1)))
            print("len(rust)      =", len(np.asarray(w_r).reshape(-1)))


if __name__ == "__main__":
    check_regular_computed_init()
    check_error_paths()
