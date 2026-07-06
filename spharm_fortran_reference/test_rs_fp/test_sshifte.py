import numpy as np
import os
from pathlib import Path

import os
# os.environ["SSHIFTE_TRACE"] = "1"
# os.environ["SSHIFTE_TRACE_CASE"] = "9x6"
# os.environ["SSHIFTE_TRACE_PYROW"] = "1"

from spharm_fortran_reference.pyspharm import _spherepack as fort_sp
import spectralkernel_rs as rust_sp


TRACE_FILES = [
    "sshifte_trace_shftoff_rlon_in.txt",
    "sshifte_trace_shftoff_rlon_shifted.txt",
    "sshifte_trace_shftoff_rlat_before_lat_shift.txt",
    "sshifte_trace_shftoff_rlat_after_lat_shift.txt",
    "sshifte_trace_shftoff_poles.txt",
    "sshifte_trace_shftoff_final_rlon_before_lon_shift.txt",
    "sshifte_trace_shftoff_final_rlon_after_lon_shift.txt",
    "sshifte_trace_shftoff_greg_final.txt",
    "sshifte_trace_shftreg_rlon_in.txt",
    "sshifte_trace_shftreg_rlon_after_first_lon_shift.txt",
    "sshifte_trace_shftreg_rlat_before_lat_shift.txt",
    "sshifte_trace_shftreg_rlat_after_lat_shift.txt",
    "sshifte_trace_shftreg_final_rlon_before_lon_shift.txt",
    "sshifte_trace_shftreg_final_rlon_after_lon_shift.txt",
    "sshifte_trace_shftreg_goff_final.txt",
]


def cleanup_trace_files():
    for name in TRACE_FILES:
        path = Path(name)
        if path.exists():
            path.unlink()


def dump_matrix_txt(path: str, arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float32)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"shape={arr.shape}\n")
        for row in arr:
            f.write(" ".join(f"{float(v):.12e}" for v in row) + "\n")


def dump_debug_marker(path: str, **kwargs):
    with open(path, "w", encoding="utf-8") as f:
        for key, value in kwargs.items():
            f.write(f"{key}={value}\n")


def should_run_case(nlon: int, nlat: int) -> bool:
    spec = os.environ.get("SSHIFTE_TRACE_CASE", "").strip()
    if not spec:
        return True
    return spec == f"{nlon}x{nlat}"


# def summarize_diff(name, a, b, eps=1e-12):
#     a = np.asarray(a)
#     b = np.asarray(b)
#     diff = a - b
#     abs_diff = np.abs(diff)
#     abs_b = np.abs(b)

#     print(f"\n[{name}]")
#     print("shape:", a.shape, b.shape)
#     print("max |diff| :", np.max(abs_diff))
#     print("mean|diff| :", np.mean(abs_diff))
#     print("rms diff   :", np.sqrt(np.mean(abs_diff ** 2)))
#     idx = np.unravel_index(np.argmax(abs_diff), diff.shape)
#     print("worst index:", idx)
#     print("fortran    :", a[idx])
#     print("rust       :", b[idx])
#     print("diff       :", diff[idx])
#     print("max rel    :", np.max(abs_diff / np.maximum(abs_b, eps)))
def summarize_diff(name, a, b, eps=1e-12):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a - b

    abs_a = np.abs(a)
    abs_b = np.abs(b)
    abs_diff = np.abs(diff)

    print(f"\n[{name}]")
    print("shape:", a.shape, b.shape)
    print("dtype:", a.dtype, b.dtype)

    # --- 原有误差 ---
    print("max |diff| :", np.max(abs_diff))
    print("mean|diff| :", np.mean(abs_diff))
    print("rms diff   :", np.sqrt(np.mean(abs_diff ** 2)))

    idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    print("worst index:", idx)
    print("fortran    :", a[idx])
    print("rust       :", b[idx])
    print("diff       :", diff[idx])

    # --- 新增：量级 ---
    # print("\n-- magnitude of a --")
    # print("max |a| :", np.max(abs_a))
    # print("mean|a| :", np.mean(abs_a))
    # print("rms  |a|:", np.sqrt(np.mean(abs_a ** 2)))

    # print("\n-- magnitude of b --")
    # print("max |b| :", np.max(abs_b))
    # print("mean|b| :", np.mean(abs_b))
    # print("rms  |b|:", np.sqrt(np.mean(abs_b ** 2)))

    # --- 相对误差（整体）---
    denom = np.maximum(abs_b, eps)
    rel = abs_diff / denom

    print("\n-- relative error (vs b) --")
    print("max rel  :", np.max(rel))
    print("mean rel :", np.mean(rel))
    print("rms rel  :", np.sqrt(np.mean(rel ** 2)))

    # --- 一个快速判断 ---
    scale = np.max(abs_b)
    if scale > 0:
        ratio = np.max(abs_diff) / scale
        print("\n-- quick scale check --")
        print("max|diff| / max|b| =", ratio)
        if ratio < 1e-6:
            print("→ likely pure floating-point noise level")
        elif ratio < 1e-3:
            print("→ small but noticeable numerical deviation")
        else:
            print("→ significant difference (check algorithm)")


def calc_lsav(nlon: int, nlat: int):
    return 2 * (2 * nlat + nlon + 16)


def calc_lwork(nlon: int, nlat: int):
    if nlon % 2 == 0:
        return 2 * nlon * (nlat + 1)
    return nlon * (5 * nlat + 1)


def make_offset_field(nlon: int, nlat: int):
    pi = np.pi
    dlat = pi / nlat
    dlon = 2.0 * pi / nlon
    dlat2 = 0.5 * dlat
    dlon2 = 0.5 * dlon
    goff = np.zeros((nlon, nlat), dtype=np.float32)
    for j in range(nlon):
        lon = dlon2 + j * dlon
        for i in range(nlat):
            lat = -0.5 * pi + dlat2 + i * dlat
            x = np.cos(lat) * np.cos(lon)
            y = np.cos(lat) * np.sin(lon)
            z = np.sin(lat)
            goff[j, i] = np.exp(x + y + z)
    return goff


def make_regular_exact(nlon: int, nlat: int):
    pi = np.pi
    dlat = pi / nlat
    dlon = 2.0 * pi / nlon
    greg = np.zeros((nlon, nlat + 1), dtype=np.float32)
    for j in range(nlon):
        lon = j * dlon
        for i in range(nlat + 1):
            lat = -0.5 * pi + i * dlat
            x = np.cos(lat) * np.cos(lon)
            y = np.cos(lat) * np.sin(lon)
            z = np.sin(lat)
            greg[j, i] = np.exp(x + y + z)
    return greg


def run_case(nlon: int, nlat: int):
    cleanup_trace_files()
    # dump_debug_marker(
    #     "sshifte_trace_python_case_marker.txt",
    #     nlon=nlon,
    #     nlat=nlat,
    #     pid=os.getpid(),
    #     cwd=Path.cwd(),
    # )
    lsav = calc_lsav(nlon, nlat)
    lwork = calc_lwork(nlon, nlat)
    goff = make_offset_field(nlon, nlat)

    wsav_f0, ierr_f0 = fort_sp.sshifti(0, nlon, nlat, lsav)
    wsav_r0, ierr_r0 = rust_sp.sshifti(0, nlon, nlat, lsav)
    assert ierr_f0 == 0
    assert ierr_r0 == 0

    print(
        "[sshifte inputs ioff=0] "
        f"nlon={nlon} nlat={nlat} lsav={lsav} lwork={lwork} "
        f"goff_shape={goff.shape} "
        f"wsav_f0_shape={np.asarray(wsav_f0).shape}"
    )

    # 调一次 Fortran 版本，确保 [`sshifte.f`](fortran_code/pyspharm/src/sshifte.f:105)
    # 中新增的文件 trace 逻辑真正执行。
    greg_f = np.zeros((nlon, nlat + 1), dtype=np.float32)
    goff_f0, greg_f, ierr0_f = fort_sp.sshifte(
        0,
        nlon,
        nlat,
        goff.copy(),
        greg_f,
        np.asarray(wsav_f0, dtype=np.float32),
        lsav,
        lwork,
    )
    print(f"[fortran sshifte ioff=0] ierr={ierr0_f}")
    assert ierr0_f == 0
    # dump_matrix_txt("sshifte_trace_python_greg_f_after_shftoff.txt", greg_f)

    greg_r, ierr0 = rust_sp.sshifte(goff, np.asarray(wsav_r0, dtype=np.float32), lwork, ioff=0)
    assert ierr0 == 0

    wsav_f1, ierr_f1 = fort_sp.sshifti(1, nlon, nlat, lsav)
    wsav_r1, ierr_r1 = rust_sp.sshifti(1, nlon, nlat, lsav)
    assert ierr_f1 == 0
    assert ierr_r1 == 0

    goff_back_f = np.zeros((nlon, nlat), dtype=np.float32)
    goff_back_f, greg_back_f, ierr1_f = fort_sp.sshifte(
        1,
        nlon,
        nlat,
        goff_back_f,
        np.asarray(greg_f, dtype=np.float32).copy(),
        np.asarray(wsav_f1, dtype=np.float32),
        lsav,
        lwork,
    )
    print(f"[fortran sshifte ioff=1] ierr={ierr1_f}")
    assert ierr1_f == 0
    # dump_matrix_txt("sshifte_trace_python_goff_f_after_shftreg.txt", goff_back_f)

    goff_back, ierr1 = rust_sp.sshifte(greg_r, np.asarray(wsav_r1, dtype=np.float32), lwork, ioff=1)
    assert ierr1 == 0

    print(f"\n{'=' * 80}\nsshifte: nlon={nlon}, nlat={nlat}\n{'=' * 80}")
    summarize_diff("sshifti ioff=0", wsav_f0, wsav_r0)
    summarize_diff("sshifti ioff=1", wsav_f1, wsav_r1)
    summarize_diff("fortran vs rust offset->regular", greg_f, greg_r)
    summarize_diff("fortran vs rust regular->offset roundtrip", goff_back_f, goff_back)


if __name__ == "__main__":
    for case in [(4, 4), (5, 8), (10, 6), (73, 144)]:
    # for case in [(4, 4)]:

        if should_run_case(*case):
            run_case(*case)
