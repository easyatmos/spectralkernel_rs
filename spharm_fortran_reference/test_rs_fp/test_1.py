import math
import numpy as np

import spectralkernel_rs as rust_sp
from spharm_fortran_reference.pyspharm import _spherepack as fort_sp


# =========================================================
# 从 spharm.py 单独抽出来的纯 Python 包装函数
# =========================================================
def getspecindx(ntrunc: int):
    """
    从 spharm.py 抽出的 getspecindx，不依赖 spharm 模块。
    返回每个谱系数对应的 (m, n)。
    """
    indexn = np.indices((ntrunc + 1, ntrunc + 1))[1, :, :]
    indexm = np.indices((ntrunc + 1, ntrunc + 1))[0, :, :]
    indices = np.nonzero(np.greater(indexn, indexm - 1).flatten())
    indxn = np.take(indexn.flatten(), indices)
    indxm = np.take(indexm.flatten(), indices)
    return np.squeeze(indxm), np.squeeze(indxn)


def legendre_from_backend(sp, lat: float, ntrunc: int):
    """
    对应 spharm.py:
        return _spherepack.getlegfunc(lat, ntrunc)
    """
    return sp.getlegfunc(lat, ntrunc)


def getgeodesicpts_from_backend(sp, m: int):
    """
    对应 spharm.py 里的 getgeodesicpts，
    直接从 backend 的 ihgeod(x,y,z) 转成 lat/lon。
    """
    x, y, z = sp.ihgeod(m)

    rad2dg = 180.0 / math.pi
    r1 = x * x + y * y
    r = np.sqrt(r1 + z * z)  # 保留，方便调试
    r1 = np.sqrt(r1)

    xtmp = np.where(np.logical_or(x, y), x, np.ones(x.shape, np.float32))
    ztmp = np.where(np.logical_or(r1, z), z, np.ones(z.shape, np.float32))

    lons = rad2dg * np.arctan2(y, xtmp) + 180.0
    lats = rad2dg * np.arctan2(r1, ztmp) - 90.0

    lat = np.zeros(10 * (m - 1) ** 2 + 2, np.float32)
    lon = np.zeros(10 * (m - 1) ** 2 + 2, np.float32)

    lat[0] = 90.0
    lat[1] = -90.0
    lon[0] = 0.0
    lon[1] = 0.0

    lat[2:] = lats[0 : 2 * (m - 1), 0 : m - 1, :].flatten()
    lon[2:] = lons[0 : 2 * (m - 1), 0 : m - 1, :].flatten()

    return lat, lon, x, y, z, r


def specintrp_from_backend(sp, lon_deg: float, dataspec, legfuncs):
    """
    对应 spharm.py 里的 specintrp 包装。
    注意：只有 backend 已经实现了 specintrp 才能调用。
    """
    ntrunc1 = int(-1.5 + 0.5 * math.sqrt(9.0 - 8.0 * (1.0 - dataspec.shape[0])))
    ntrunc2 = int(-1.5 + 0.5 * math.sqrt(9.0 - 8.0 * (1.0 - legfuncs.shape[0])))

    if ntrunc1 != ntrunc2:
        raise ValueError(
            "dataspec and legfuncs imply inconsistent truncations"
        )

    lon_rad = (math.pi / 180.0) * lon_deg
    return sp.specintrp(lon_rad, ntrunc1, dataspec, legfuncs)


# =========================================================
# 通用误差汇总
# =========================================================
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
    print("A value    :", a[idx])
    print("B value    :", b[idx])
    print("diff       :", diff[idx])


def summarize_lon_diff(name, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    raw_diff = a - b
    diff = ((raw_diff + 180.0) % 360.0) - 180.0

    print(f"\n[{name}]")
    print("shape:", a.shape, b.shape)
    print("dtype:", a.dtype, b.dtype)
    print("max |wrapped diff| :", np.max(np.abs(diff)))
    print("mean|wrapped diff| :", np.mean(np.abs(diff)))
    print("rms wrapped diff   :", np.sqrt(np.mean(np.abs(diff) ** 2)))

    idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    print("worst index:", idx)
    print("A value    :", a[idx])
    print("B value    :", b[idx])
    print("raw diff   :", raw_diff[idx])
    print("wrap diff  :", diff[idx])


# =========================================================
# 1) gaqd 比较
# =========================================================
def check_gaqd(nlat_list=(4, 5, 8, 16, 32)):
    print("\n" + "=" * 80)
    print("GAQD COMPARISON")
    print("=" * 80)

    for nlat in nlat_list:
        print(f"\n-------------------- nlat = {nlat} --------------------")

        # 假定 gaqd 返回 (lats, wts, ierror) 或 (points, weights, ierror)
        out_f = fort_sp.gaqd(nlat)
        out_r = rust_sp.gaqd(nlat)

        if len(out_f) == 3:
            lat_f, wt_f, ierr_f = out_f
            lat_r, wt_r, ierr_r = out_r
        else:
            raise RuntimeError("Unexpected gaqd return signature")

        print("ierror fortran =", ierr_f)
        print("ierror rust    =", ierr_r)

        summarize_diff("gaqd latitudes", lat_f, lat_r)
        summarize_diff("gaqd weights", wt_f, wt_r)

        print("sum(weights) fortran =", np.sum(wt_f))
        print("sum(weights) rust    =", np.sum(wt_r))


# =========================================================
# 2) getlegfunc 比较
# =========================================================
def check_getlegfunc(cases=None):
    print("\n" + "=" * 80)
    print("GETLEGFUNC COMPARISON")
    print("=" * 80)

    if cases is None:
        cases = [
            (-90.0, 0),
            (-60.0, 3),
            (-30.0, 5),
            (0.0, 5),
            (30.0, 8),
            (60.0, 10),
            (89.0, 12),
        ]

    for lat, ntrunc in cases:
        print(f"\n-------------------- lat = {lat}, ntrunc = {ntrunc} --------------------")

        p_f = legendre_from_backend(fort_sp, lat, ntrunc)
        p_r = legendre_from_backend(rust_sp, lat, ntrunc)

        summarize_diff("getlegfunc", p_f, p_r)

        # 额外打印两个最基础项
        idx00 = 0
        print("p00 fortran =", p_f[idx00])
        print("p00 rust    =", p_r[idx00])

        if ntrunc >= 1:
            indxm, indxn = getspecindx(ntrunc)
            idx01 = np.where((indxm == 0) & (indxn == 1))[0][0]
            print("p10 fortran =", p_f[idx01])
            print("p10 rust    =", p_r[idx01])


# =========================================================
# 3) ihgeod + getgeodesicpts 比较
# =========================================================
def check_ihgeod(m_list=(2, 3, 4, 6, 8)):
    print("\n" + "=" * 80)
    print("IHGEOD / GETGEODESICPTS COMPARISON")
    print("=" * 80)

    for m in m_list:
        print(f"\n==================== m = {m} ====================")

        lat_f, lon_f, x_f, y_f, z_f, r_f = getgeodesicpts_from_backend(fort_sp, m)
        lat_r, lon_r, x_r, y_r, z_r, r_r = getgeodesicpts_from_backend(rust_sp, m)

        print("\n[raw xyz shape]")
        print("fortran x.shape:", x_f.shape)
        print("rust    x.shape:", x_r.shape)

        summarize_diff("ihgeod x", x_f, x_r)
        summarize_diff("ihgeod y", y_f, y_r)
        summarize_diff("ihgeod z", z_f, z_r)

        print("\n[Unit sphere check]")
        print("fortran max |r-1| =", np.max(np.abs(r_f - 1.0)))
        print("rust    max |r-1| =", np.max(np.abs(r_r - 1.0)))

        print("\n[Lat/Lon comparison]")
        summarize_diff("geodesic lat", lat_f, lat_r)
        summarize_lon_diff("geodesic lon", lon_f, lon_r)

        print("\n[Pole check]")
        print("fortran poles:", (lat_f[0], lon_f[0]), (lat_f[1], lon_f[1]))
        print("rust    poles:", (lat_r[0], lon_r[0]), (lat_r[1], lon_r[1]))

        print("\n[Nearest-neighbor sanity]")
        pts_f = np.column_stack([
            np.cos(np.deg2rad(lat_f)) * np.cos(np.deg2rad(lon_f)),
            np.cos(np.deg2rad(lat_f)) * np.sin(np.deg2rad(lon_f)),
            np.sin(np.deg2rad(lat_f)),
        ])
        pts_r = np.column_stack([
            np.cos(np.deg2rad(lat_r)) * np.cos(np.deg2rad(lon_r)),
            np.cos(np.deg2rad(lat_r)) * np.sin(np.deg2rad(lon_r)),
            np.sin(np.deg2rad(lat_r)),
        ])

        def min_pairwise_dist(pts):
            mind = np.inf
            n = len(pts)
            for i in range(n):
                d = np.sqrt(np.sum((pts[i + 1 :] - pts[i]) ** 2, axis=1))
                if d.size > 0:
                    mind = min(mind, np.min(d))
            return mind

        print("fortran min pair distance =", min_pairwise_dist(pts_f))
        print("rust    min pair distance =", min_pairwise_dist(pts_r))


# =========================================================
# 4) specintrp 比较（等 Rust 和 Fortran 都实现 specintrp 后再用）
# =========================================================
def make_random_dataspec(ntrunc, seed=123, scale=1e-1):
    ncoeff = (ntrunc + 1) * (ntrunc + 2) // 2
    rng = np.random.default_rng(seed)
    real = rng.normal(scale=scale, size=ncoeff)
    imag = rng.normal(scale=scale, size=ncoeff)
    dataspec = (real + 1j * imag).astype(np.complex64)
    dataspec[0] = np.complex64(1.0 + 0.0j)
    return dataspec


def check_specintrp(latlon_cases=None, ntrunc_list=(3, 5, 8)):
    print("\n" + "=" * 80)
    print("SPECINTRP COMPARISON")
    print("=" * 80)

    if latlon_cases is None:
        latlon_cases = [
            (-80.0, 0.0),
            (-45.0, 30.0),
            (-10.0, 120.0),
            (0.0, 180.0),
            (25.0, 240.0),
            (60.0, 300.0),
        ]

    for ntrunc in ntrunc_list:
        print(f"\n-------------------- ntrunc = {ntrunc} --------------------")
        dataspec = make_random_dataspec(ntrunc=ntrunc, seed=100 + ntrunc)
        max_abs_diff = 0.0

        for lat, lon in latlon_cases:
            leg_f = legendre_from_backend(fort_sp, lat, ntrunc)
            leg_r = legendre_from_backend(rust_sp, lat, ntrunc)

            val_f = specintrp_from_backend(fort_sp, lon, dataspec, leg_f)
            val_r = specintrp_from_backend(rust_sp, lon, dataspec, leg_r)
            diff = val_r - val_f
            abs_diff = abs(diff)
            max_abs_diff = max(max_abs_diff, float(abs_diff))

            print(
                f"lat={lat:7.2f}, lon={lon:7.2f} | "
                f"fortran={val_f!r}, rust={val_r!r}, diff={diff!r}"
            )

            np.testing.assert_allclose(
                val_r,
                val_f,
                rtol=1e-6,
                atol=1e-6,
                err_msg=(
                    f"specintrp mismatch at ntrunc={ntrunc}, lat={lat}, lon={lon}: "
                    f"fortran={val_f!r}, rust={val_r!r}, diff={diff!r}"
                ),
            )

        print(f"max |specintrp diff| = {max_abs_diff}")


# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    check_gaqd(nlat_list=(4, 5, 8, 16, 32))
    check_getlegfunc(
        cases=[
            (-90.0, 0),
            (-60.0, 3),
            (-30.0, 5),
            (0.0, 5),
            (30.0, 8),
            (60.0, 10),
            (89.0, 12),
        ]
    )
    check_ihgeod(m_list=(2, 3, 4, 6, 8))

    # Rust 端实现 specintrp 后直接打开
    check_specintrp(
        latlon_cases=[
            (-80.0, 0.0),
            (-45.0, 30.0),
            (-10.0, 120.0),
            (0.0, 180.0),
            (25.0, 240.0),
            (60.0, 300.0),
        ],
        ntrunc_list=(3, 5, 8),
    )
