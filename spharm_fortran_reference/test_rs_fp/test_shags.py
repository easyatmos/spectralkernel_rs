import numpy as np

import spectralkernel_rs as rust_sp
from spharm_fortran_reference.pyspharm import _spherepack as fort_sp


# def summarize_diff(name, a, b):
#     a = np.asarray(a)
#     b = np.asarray(b)
#     diff = a - b
#     print(f"\n[{name}]")
#     print("shape:", a.shape, b.shape)
#     print("dtype:", a.dtype, b.dtype)
#     print("max |diff| :", np.max(np.abs(diff)))
#     print("mean|diff| :", np.mean(np.abs(diff)))
#     print("rms diff   :", np.sqrt(np.mean(np.abs(diff) ** 2)))
#     idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
#     print("worst index:", idx)
#     print("fortran    :", a[idx])
#     print("rust       :", b[idx])
#     print("diff       :", diff[idx])
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


def calc_shags_sizes(nlat: int, nlon: int):
    n1 = min(nlat, (nlon + 2) // 2)
    n2 = (nlat + 1) // 2
    lshags = nlat * (3 * (n1 + n2) - 2) + (n1 - 1) * (n2 * (2 * nlat - n1) - 3 * n1) // 2 + nlon + 15
    lwork = 4 * nlat * (nlat + 2) + 2
    ldwork = nlat * (nlat + 4)
    return lshags, lwork, ldwork


def make_grid(nlat: int, nlon: int, nt: int):
    lat = np.linspace(-1.0, 1.0, nlat, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    out = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        out[:, :, k] = (
            np.cos((k + 1) * lon)[None, :]
            + np.sin((k + 2) * lon)[None, :] * lat[:, None]
            + (k + 1) * 0.1 * lat[:, None] ** 2
        )
    return out


def run_case(nlat: int, nlon: int, nt: int):
    lshags, init_lwork, ldwork = calc_shags_sizes(nlat, nlon)
    wshags, ierr0 = rust_sp.shagsi(nlat, nlon, lshags, init_lwork, ldwork)
    assert ierr0 == 0
    g = make_grid(nlat, nlon, nt)
    lwork = nlat * nlon * (nt + 1)
    a_f, b_f, ierr_f = fort_sp.shags(g, np.asarray(wshags, dtype=np.float32), lwork)
    a_r, b_r, ierr_r = rust_sp.shags(g, np.asarray(wshags, dtype=np.float32), lwork)
    print(f"\n{'=' * 80}\nshags: nlat={nlat}, nlon={nlon}, nt={nt}\n{'=' * 80}")
    print("ierror fortran =", ierr_f)
    print("ierror rust    =", ierr_r)
    summarize_diff("shags a", a_f, a_r)
    summarize_diff("shags b", b_f, b_r)


if __name__ == "__main__":
    for case in [(3, 4, 1), (4, 4, 1), (5, 8, 2), (73, 144, 1)]:
        run_case(*case)
