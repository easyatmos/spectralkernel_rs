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
#     print("max |diff| :", np.max(abs_diff))
#     print("mean|diff| :", np.mean(abs_diff))
#     print("rms diff   :", np.sqrt(np.mean(abs_diff ** 2)))

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


def make_scalar(nlon: int, nlat: int):
    return (0.1 + 0.01 * np.arange(nlon * nlat, dtype=np.float32)).reshape(nlon, nlat)


def make_vector(nlon: int, nlat: int):
    base = np.arange(nlon * nlat, dtype=np.float32).reshape(nlon, nlat)
    ug = 0.2 + 0.03 * base
    vg = -0.4 + 0.02 * base
    return ug.astype(np.float32), vg.astype(np.float32)


def run_scalar_case(nlon: int, nlat: int, ig: int):
    sg = make_scalar(nlon, nlat)
    sm_f = fort_sp.geo2maths(sg, ig=ig)
    sm_r = rust_sp.geo2maths(sg, ig=ig)
    sg_back_f = fort_sp.math2geos(sm_f, ig=ig)
    sg_back_r = rust_sp.math2geos(sm_r, ig=ig)

    print(f"\n{'=' * 80}\ngeo2math scalar: nlon={nlon}, nlat={nlat}, ig={ig}\n{'=' * 80}")
    summarize_diff("geo2maths", sm_f, sm_r)
    summarize_diff("math2geos", sg_back_f, sg_back_r)
    summarize_diff("scalar roundtrip (orig vs rust back)", sg, sg_back_r)


def run_vector_case(nlon: int, nlat: int, ig: int):
    ug, vg = make_vector(nlon, nlat)
    vm_f, wm_f = fort_sp.geo2mathv(ug, vg, ig=ig)
    vm_r, wm_r = rust_sp.geo2mathv(ug, vg, ig=ig)
    ug_back_f, vg_back_f = fort_sp.math2geov(vm_f, wm_f, ig=ig)
    ug_back_r, vg_back_r = rust_sp.math2geov(vm_r, wm_r, ig=ig)

    print(f"\n{'=' * 80}\ngeo2math vector: nlon={nlon}, nlat={nlat}, ig={ig}\n{'=' * 80}")
    summarize_diff("geo2mathv vm", vm_f, vm_r)
    summarize_diff("geo2mathv wm", wm_f, wm_r)
    summarize_diff("math2geov ug", ug_back_f, ug_back_r)
    summarize_diff("math2geov vg", vg_back_f, vg_back_r)
    summarize_diff("vector roundtrip ug", ug, ug_back_r)
    summarize_diff("vector roundtrip vg", vg, vg_back_r)


if __name__ == "__main__":
    for ig in (0, 1):
        for nlon, nlat in ((4, 3), (8, 5), (16, 10)):
            run_scalar_case(nlon, nlat, ig)
            run_vector_case(nlon, nlat, ig)
