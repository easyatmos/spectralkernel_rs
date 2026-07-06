import numpy as np

from spharm_fortran_reference.pyspharm import _spherepack as fort_sp
import spectralkernel_rs as rust_sp


# def summarize_diff(name, a, b, eps=1e-12):
#     a = np.asarray(a)
#     b = np.asarray(b)
#     diff = a - b
#     abs_diff = np.abs(diff)
#     abs_b = np.abs(b)

#     print(f"\n[{name}]")
#     print("shape:", a.shape, b.shape)
#     print("dtype:", a.dtype, b.dtype)
#     print("max |diff| :", np.max(abs_diff))
#     print("mean|diff| :", np.mean(abs_diff))
#     print("rms diff   :", np.sqrt(np.mean(abs_diff**2)))
#     idx = np.unravel_index(np.argmax(abs_diff), diff.shape)
#     print("worst index:", idx)
#     print("fortran    :", a[idx])
#     print("rust       :", b[idx])
#     print("diff       :", diff[idx])
#     rel = abs_diff / np.maximum(abs_b, eps)
#     print("max rel    :", np.max(rel))
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


def make_offset_vector_field(nlon: int, nlat: int):
    pi = np.pi
    dlat = pi / nlat
    dlon = 2.0 * pi / nlon
    u = np.zeros((nlon, nlat), dtype=np.float32)
    v = np.zeros((nlon, nlat), dtype=np.float32)
    for j in range(nlon):
        lon = 0.5 * dlon + j * dlon
        for i in range(nlat):
            lat = -0.5 * pi + 0.5 * dlat + i * dlat
            u[j, i] = np.cos(lat) * np.sin(lon) + 0.25 * np.sin(2.0 * lat)
            v[j, i] = np.sin(lat) * np.cos(lon) - 0.5 * np.cos(3.0 * lon)
    return u, v


def run_case(nlon: int, nlat: int):
    lsav = calc_lsav(nlon, nlat)
    lwork = calc_lwork(nlon, nlat)
    uoff, voff = make_offset_vector_field(nlon, nlat)

    wsav_f0, ierr_f0 = fort_sp.vshifti(0, nlon, nlat, lsav)
    wsav_r0, ierr_r0 = rust_sp.vshifti(0, nlon, nlat, lsav)
    assert ierr_f0 == 0
    assert ierr_r0 == 0

    ureg_f = np.zeros((nlon, nlat + 1), dtype=np.float32)
    vreg_f = np.zeros((nlon, nlat + 1), dtype=np.float32)
    _, _, ureg_f, vreg_f, ierr0_f = fort_sp.vshifte(
        0,
        nlon,
        nlat,
        uoff.copy(),
        voff.copy(),
        ureg_f,
        vreg_f,
        np.asarray(wsav_f0, dtype=np.float32),
        lsav,
        lwork,
    )
    assert ierr0_f == 0

    ureg_r, vreg_r, ierr0_r = rust_sp.vshifte(
        uoff,
        voff,
        np.asarray(wsav_r0, dtype=np.float32),
        lwork,
        ioff=0,
    )
    assert ierr0_r == 0

    wsav_f1, ierr_f1 = fort_sp.vshifti(1, nlon, nlat, lsav)
    wsav_r1, ierr_r1 = rust_sp.vshifti(1, nlon, nlat, lsav)
    assert ierr_f1 == 0
    assert ierr_r1 == 0

    uoff_back_f = np.zeros((nlon, nlat), dtype=np.float32)
    voff_back_f = np.zeros((nlon, nlat), dtype=np.float32)
    uoff_back_f, voff_back_f, _, _, ierr1_f = fort_sp.vshifte(
        1,
        nlon,
        nlat,
        uoff_back_f,
        voff_back_f,
        np.asarray(ureg_f, dtype=np.float32).copy(),
        np.asarray(vreg_f, dtype=np.float32).copy(),
        np.asarray(wsav_f1, dtype=np.float32),
        lsav,
        lwork,
    )
    assert ierr1_f == 0

    uoff_back_r, voff_back_r, ierr1_r = rust_sp.vshifte(
        ureg_r,
        vreg_r,
        np.asarray(wsav_r1, dtype=np.float32),
        lwork,
        ioff=1,
    )
    assert ierr1_r == 0

    print(f"\n{'=' * 80}\nvshifte: nlon={nlon}, nlat={nlat}\n{'=' * 80}")
    summarize_diff("vshifti ioff=0", wsav_f0, wsav_r0)
    summarize_diff("vshifti ioff=1", wsav_f1, wsav_r1)
    summarize_diff("u offset->regular", ureg_f, ureg_r)
    summarize_diff("v offset->regular", vreg_f, vreg_r)
    summarize_diff("u regular->offset roundtrip", uoff_back_f, uoff_back_r)
    summarize_diff("v regular->offset roundtrip", voff_back_f, voff_back_r)


if __name__ == "__main__":
    for case in [(4, 4), (5, 8), (10, 6), (73, 144)]:
        run_case(*case)
