import numpy as np

from spharm_fortran_reference.pyspharm import _spherepack as fort_sp
import spectralkernel_rs as rust_sp


def build_fortran_reference(nlat: int, nlon: int, xlmbda: float):
    pi = 4.0 * np.arctan(1.0)
    llsave = nlat * (nlat + 1) + 3 * ((nlat - 2) * (nlat - 1) + (nlon + 15))
    llwork = nlat * (2 * nlon + 3 * (nlat + 1) + 2 * nlat + 1)
    lldwork = nlat + 1

    dlat = pi / (nlat - 1)
    dlon = 2.0 * pi / nlon
    theta = -0.5 * pi + np.arange(nlat, dtype=np.float32) * dlat
    phi = np.arange(nlon, dtype=np.float32) * dlon
    sint = np.sin(theta)
    cost = np.cos(theta)
    sinp = np.sin(phi)
    cosp = np.cos(phi)

    r = np.zeros((nlat, nlon), dtype=np.float32)
    for j in range(nlon):
        for i in range(nlat):
            x = cost[i] * cosp[j]
            y = cost[i] * sinp[j]
            z = sint[i]
            r[i, j] = -((x * y * (z * z + 6.0 * (z + 1.0))) + z * (z + 2.0)) * np.exp(z)

    wshaec, ierr0 = fort_sp.shaeci(nlat, nlon, llsave, lldwork)
    assert ierr0 == 0
    wshsec, ierr1 = fort_sp.shseci(nlat, nlon, llsave, lldwork)
    assert ierr1 == 0
    a, b, ierr2 = fort_sp.shaec(r, np.asarray(wshaec, dtype=np.float32), llwork)
    assert ierr2 == 0
    u, pertrb, ierr3 = fort_sp.islapec(nlon, np.asarray([xlmbda], dtype=np.float32), a, b, np.asarray(wshsec, dtype=np.float32), llwork)
    assert ierr3 == 0

    errm = 0.0
    for j in range(nlon):
        for i in range(nlat):
            x = cost[i] * cosp[j]
            y = cost[i] * sinp[j]
            z = sint[i]
            ue = (1.0 + x * y) * np.exp(z)
            errm = max(errm, abs(float(u[i, j].item()) - float(ue.item())))
    return u, float(pertrb[0]), float(errm)


# def summarize_diff(name, a, b):
#     a = np.asarray(a)
#     b = np.asarray(b)
#     diff = a - b
#     print(f"\n[{name}]")
#     print("shape:", a.shape, b.shape)
#     print("max |diff| :", np.max(np.abs(diff)))
#     print("mean|diff| :", np.mean(np.abs(diff)))
#     print("rms diff   :", np.sqrt(np.mean(np.abs(diff) ** 2)))
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

    # --- 新增：量级 ---
    print("\n-- magnitude of a --")
    print("max |a| :", np.max(abs_a))
    print("mean|a| :", np.mean(abs_a))
    print("rms  |a|:", np.sqrt(np.mean(abs_a ** 2)))

    print("\n-- magnitude of b --")
    print("max |b| :", np.max(abs_b))
    print("mean|b| :", np.mean(abs_b))
    print("rms  |b|:", np.sqrt(np.mean(abs_b ** 2)))

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


def run_case(nlat: int, nlon: int, xlmbda: float = 1.0):
    print(f"\n{'=' * 80}\nhelmsph: nlat={nlat}, nlon={nlon}, xlmbda={xlmbda}\n{'=' * 80}")
    u_f, pertrb_f, errm_f = build_fortran_reference(nlat, nlon, xlmbda)
    u_r, pertrb_r, errm_r, ierr_r = rust_sp.helmsph(nlat, nlon, np.float32(xlmbda))
    assert ierr_r == 0, ("rust helmsph failed", nlat, nlon, xlmbda, ierr_r)
    summarize_diff("helmsph solution", u_f, u_r)
    print("fortran pertrb:", pertrb_f)
    print("rust pertrb   :", pertrb_r)
    print("fortran errm  :", errm_f)
    print("rust errm     :", errm_r)


if __name__ == "__main__":
    for case in [(19, 36, 1.0), (10, 18, 1.0), (37, 72, 0.5)]:
        run_case(*case)
