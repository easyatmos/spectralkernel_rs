import numpy as np
from spharm_fortran_reference.pyspharm._spherepack import shsesi
from spectralkernel_rs import shsesi as shsesi_rs

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

if __name__ == "__main__":
    nlat = 73
    nlon = 144
    lshses = 100096
    lwork = 21173
    nlat_plus1 = 74

    shsesi_result_for = shsesi(nlat, nlon, lshses, lwork, nlat_plus1)
    shsesi_result_rs = shsesi_rs(nlat, nlon, lshses, lwork, nlat_plus1)
    summarize_diff("shsesi", shsesi_result_for[0], shsesi_result_rs[0])
