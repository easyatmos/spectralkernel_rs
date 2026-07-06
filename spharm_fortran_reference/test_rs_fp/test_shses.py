import numpy as np

import sys
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open("test_shses.log", "w")
sys.stdout = Tee(sys.stdout, log_file)

import os
os.environ["SHSES_TRACE"] = "1"
os.environ["SHSES_TRACE_ILAT"] = "34"
os.environ["SHSES_TRACE_JLON"] = "138"
os.environ["SHSES_TRACE_K"] = "0"

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


def calc_shsesi_sizes(nlat: int, nlon: int):
    n1 = min(nlat, (nlon + 2) // 2)
    n2 = (nlat + 1) // 2
    lshses = (n1 * n2 * (2 * nlat - n1 + 1)) // 2 + nlon + 15
    lwork = 5 * nlat * n2 + (3 * ((n1 - 2) * (2 * nlat - n1 - 1))) // 2
    ldwork = nlat + 1
    return lshses, lwork, ldwork


def make_coeffs(nlat: int, nt: int):
    a = np.zeros((nlat, nlat, nt), dtype=np.float32)
    b = np.zeros((nlat, nlat, nt), dtype=np.float32)
    for k in range(nt):
        for m in range(nlat):
            for n in range(m, nlat):
                a[m, n, k] = (m + 1) * 0.1 + (n + 1) * 0.03 + k * 0.07
                b[m, n, k] = (m + 1) * 0.05 - (n + 1) * 0.02 + k * 0.04
    return a, b


def make_single_mode_coeffs(nlat: int, nt: int, m: int, n: int, use_b: bool = False):
    a = np.zeros((nlat, nlat, nt), dtype=np.float32)
    b = np.zeros((nlat, nlat, nt), dtype=np.float32)
    target = b if use_b else a
    for k in range(nt):
        target[m, n, k] = 1.0
    return a, b


def summarize_row_diff(name, a, b, row: int, k: int = 0):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 2:
        diff_row = a[row, :] - b[row, :]
    else:
        diff_row = a[row, :, k] - b[row, :, k]
    print(f"[{name}] row={row} k={k}")
    print("row max |diff| :", np.max(np.abs(diff_row)))
    print("row mean|diff| :", np.mean(np.abs(diff_row)))
    print("row first 20   :", diff_row[:20])
    print("row last 20    :", diff_row[-20:])


def run_case(nlat: int, nlon: int, nt: int):
    lshses, init_lwork, ldwork = calc_shsesi_sizes(nlat, nlon)
    wshses, ierr0 = rust_sp.shsesi(nlat, nlon, lshses, init_lwork, ldwork)
    assert ierr0 == 0
    a, b = make_coeffs(nlat, nt)
    lwork = (nt + 1) * nlat * nlon
    g_f, ierr_f = fort_sp.shses(nlon, a, b, np.asarray(wshses, dtype=np.float32), lwork)
    g_r, ierr_r = rust_sp.shses(a, b, np.asarray(wshses, dtype=np.float32), lwork)
    print(f"\n{'=' * 80}\nshses: nlat={nlat}, nlon={nlon}, nt={nt}\n{'=' * 80}")
    print("ierror fortran =", ierr_f)
    print("ierror rust    =", ierr_r)
    summarize_diff("shses g", g_f, g_r)
    if nlat > 34:
        summarize_row_diff("shses row diff", g_f, g_r, row=34, k=0)


def run_single_mode_case(nlat: int, nlon: int, nt: int, m: int, n: int, use_b: bool = False):
    lshses, init_lwork, ldwork = calc_shsesi_sizes(nlat, nlon)
    wshses, ierr0 = rust_sp.shsesi(nlat, nlon, lshses, init_lwork, ldwork)
    assert ierr0 == 0
    a, b = make_single_mode_coeffs(nlat, nt, m, n, use_b=use_b)
    lwork = (nt + 1) * nlat * nlon
    g_f, ierr_f = fort_sp.shses(nlon, a, b, np.asarray(wshses, dtype=np.float32), lwork)
    g_r, ierr_r = rust_sp.shses(a, b, np.asarray(wshses, dtype=np.float32), lwork)
    channel = "b" if use_b else "a"
    print(
        f"\n{'=' * 80}\nshses single-mode: nlat={nlat}, nlon={nlon}, nt={nt}, channel={channel}, m={m}, n={n}\n{'=' * 80}"
    )
    print("ierror fortran =", ierr_f)
    print("ierror rust    =", ierr_r)
    summarize_diff(f"shses single-mode {channel}[{m},{n}]", g_f, g_r)
    if nlat > 34:
        summarize_row_diff(f"single-mode row diff {channel}[{m},{n}]", g_f, g_r, row=34, k=0)


def run_fourier_debug_case():
    row = np.array([[0.0, 0.5, 0.0, -0.5]], dtype=np.float32)
    out = np.asarray(rust_sp.fourier_analysis_real_debug(row), dtype=np.float32)
    print(f"\n{'=' * 80}\nfourier_analysis_real_debug: nlon=4 special sine row\n{'=' * 80}")
    print("input :", row[0])
    print("output:", out[0])
    print("expected pattern: DC≈0, cos(m=1)≈0, -sin(m=1)显著非零, Nyquist≈0")


if __name__ == "__main__":
    run_fourier_debug_case()
    for case in [(3, 4, 1), (4, 4, 1), (5, 8, 2), (73, 144, 1)]:
        run_case(*case)
    for mode_case in [
        (73, 144, 1, 0, 0, False),
        (73, 144, 1, 1, 1, False),
        (73, 144, 1, 10, 10, False),
        (73, 144, 1, 30, 40, False),
        (73, 144, 1, 50, 60, False),
        (73, 144, 1, 72, 72, False),
        (73, 144, 1, 1, 1, True),
        (73, 144, 1, 30, 40, True),
        (73, 144, 1, 72, 72, True),
    ]:
        run_single_mode_case(*mode_case)
