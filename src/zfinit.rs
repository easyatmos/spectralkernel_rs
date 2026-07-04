use crate::sphcom_scalar::{dnlfk, rabcp1};
use numpy::PyArray1;
use pyo3::prelude::*;

/// Core Rust implementation of `zfinit`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
///
/// # Returns
/// A pair of double-precision work tables.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn zfinit_impl(nlat: usize, nlon: usize) -> (Vec<f64>, Vec<f64>) {
    let imid = (nlat + 1) / 2;
    let mut z = vec![0.0_f64; imid * nlat * 2];
    let pi = 4.0_f64 * (1.0_f64).atan();
    let dt = pi / ((nlat - 1) as f64);

    for mp1 in 1..=2.min(nlat) {
        let m = (mp1 - 1) as i32;
        for np1 in mp1..=nlat {
            let n = (np1 - 1) as i32;
            let work = dnzfk(nlat as i32, m, n);
            for i in 1..=imid {
                let th = ((i - 1) as f64) * dt;
                let zh = dnzft(nlat as i32, m, n, th, &work);
                let idx = (((np1 - 1) * imid) + (i - 1)) * 2 + (mp1 - 1);
                z[idx] = zh;
            }
            let pole_idx = (((np1 - 1) * imid) + 0) * 2 + (mp1 - 1);
            z[pole_idx] *= 0.5_f64;
        }
    }

    let (a, b, c) = rabcp1(nlat, nlon);
    let mut abc = vec![0.0_f64; a.len() + b.len() + c.len() - 3];
    let labc = a.len().saturating_sub(1);
    for i in 1..=labc {
        abc[i - 1] = a[i];
        abc[labc + i - 1] = b[i];
        abc[2 * labc + i - 1] = c[i];
    }

    let mut wzfin = vec![0.0_f64; 2 * imid * nlat + 3 * labc];
    for i in 0..(imid * nlat) {
        wzfin[i] = z[i * 2];
        wzfin[imid * nlat + i] = z[i * 2 + 1];
    }
    for i in 0..labc {
        wzfin[2 * imid * nlat + i] = abc[i];
        wzfin[2 * imid * nlat + labc + i] = abc[labc + i];
        wzfin[2 * imid * nlat + 2 * labc + i] = abc[2 * labc + i];
    }

    (wzfin, abc)
}

/// Rust entry point for `zfin_column`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `m`: Zonal wavenumber or refinement level, depending on the routine.
/// - `wzfin`: Computed scalar analysis workspace produced by `zfinit_impl`.
///
/// # Returns
/// A contiguous workspace or coefficient vector stored in double precision.
pub fn zfin_column(nlat: usize, nlon: usize, isym: i32, m: usize, wzfin: &[f64]) -> Vec<f64> {
    let imid = (nlat + 1) / 2;
    let lim = nlat * imid;
    let mmax = nlat.min(nlon / 2 + 1);
    let labc = ((mmax.saturating_sub(2)) * (nlat + nlat - mmax - 1)) / 2;
    let zz = &wzfin[0..lim];
    let z1 = &wzfin[lim..2 * lim];
    let a = &wzfin[2 * lim..2 * lim + labc];
    let b = &wzfin[2 * lim + labc..2 * lim + 2 * labc];
    let c = &wzfin[2 * lim + 2 * labc..2 * lim + 3 * labc];

    let mut z = vec![0.0_f64; imid * nlat * 3];
    let (mut i1, mut i2, mut i3) = (0usize, 1usize, 2usize);

    for mm in 0..=m {
        if mm == 0 {
            i1 = 0;
            i2 = 1;
            i3 = 2;
            for np1 in 1..=nlat {
                for i in 1..=imid {
                    let dst = (((i - 1) * nlat) + (np1 - 1)) * 3 + i3;
                    let src = (np1 - 1) * imid + (i - 1);
                    z[dst] = zz[src];
                }
            }
        } else {
            let ihold = i1;
            i1 = i2;
            i2 = i3;
            i3 = ihold;

            if mm == 1 {
                for np1 in 2..=nlat {
                    for i in 1..=imid {
                        let dst = (((i - 1) * nlat) + (np1 - 1)) * 3 + i3;
                        let src = (np1 - 1) * imid + (i - 1);
                        z[dst] = z1[src];
                    }
                }
            } else {
                let mut ns = ((mm - 2) * (nlat + nlat - mm - 1)) / 2;
                if isym != 1 {
                    for i in 1..=imid {
                        let dst = (((i - 1) * nlat) + mm) * 3 + i3;
                        let src_m_minus_2 = (((i - 1) * nlat) + (mm - 2)) * 3 + i1;
                        let src_m = (((i - 1) * nlat) + mm) * 3 + i1;
                        z[dst] = a[ns] * z[src_m_minus_2] - c[ns] * z[src_m];
                    }
                }
                if mm != nlat - 1 {
                    if isym != 2 {
                        ns += 1;
                        for i in 1..=imid {
                            let dst = (((i - 1) * nlat) + (mm + 1)) * 3 + i3;
                            let src_m_minus_1 = (((i - 1) * nlat) + (mm - 1)) * 3 + i1;
                            let src_m_plus_1 = (((i - 1) * nlat) + (mm + 1)) * 3 + i1;
                            z[dst] = a[ns] * z[src_m_minus_1] - c[ns] * z[src_m_plus_1];
                        }
                    }
                    let mut nstrt = mm + 3;
                    if isym == 1 {
                        nstrt = mm + 4;
                    }
                    if nstrt <= nlat {
                        let nstp = if isym == 0 { 1 } else { 2 };
                        let mut np1 = nstrt;
                        while np1 <= nlat {
                            ns += nstp;
                            for i in 1..=imid {
                                let dst = (((i - 1) * nlat) + (np1 - 1)) * 3 + i3;
                                let src_n_minus_2_i1 = (((i - 1) * nlat) + (np1 - 3)) * 3 + i1;
                                let src_n_minus_2_i3 = (((i - 1) * nlat) + (np1 - 3)) * 3 + i3;
                                let src_n_i1 = (((i - 1) * nlat) + (np1 - 1)) * 3 + i1;
                                z[dst] = a[ns] * z[src_n_minus_2_i1] + b[ns] * z[src_n_minus_2_i3]
                                    - c[ns] * z[src_n_i1];
                            }
                            np1 += nstp;
                        }
                    }
                }
            }
        }
    }

    let mut out = vec![0.0_f64; imid * nlat];
    for np1 in 1..=nlat {
        for i in 1..=imid {
            let src = (((i - 1) * nlat) + (np1 - 1)) * 3 + i3;
            let dst = (np1 - 1) * imid + (i - 1);
            out[dst] = z[src];
        }
    }
    out
}

#[pyfunction]
/// Python helper that exposes the scalar analysis workspace built by `zfinit_impl` for inspection.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn zfinit_debug<'py>(
    py: Python<'py>,
    nlat: usize,
    nlon: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (wzfin, abc) = zfinit_impl(nlat, nlon);
    Ok((
        PyArray1::from_vec(py, wzfin).to_owned(),
        PyArray1::from_vec(py, abc).to_owned(),
    ))
}

fn dnzfk(nlat: i32, m: i32, n: i32) -> Vec<f64> {
    let lc = usize::try_from((nlat + 1) / 2).unwrap_or(0);
    let mut cz = vec![0.0_f64; lc + 1];
    let work = dnlfk(m, n);
    let sc1 = 2.0_f64 / ((nlat - 1) as f64);
    let nmod = n % 2;
    let mmod = m % 2;

    if nmod == 0 {
        if mmod == 0 {
            let kdo = usize::try_from(n / 2 + 1).unwrap_or(0);
            for idx in 1..=lc {
                let i = (idx + idx - 2) as f64;
                let mut sum = work[1] / (1.0_f64 - i * i);
                if kdo >= 2 {
                    for kp1 in 2..=kdo {
                        let k = (kp1 - 1) as f64;
                        let t1 = 1.0_f64 - (k + k + i).powi(2);
                        let t2 = 1.0_f64 - (k + k - i).powi(2);
                        sum += work[kp1] * (t1 + t2) / (t1 * t2);
                    }
                }
                cz[idx] = sc1 * sum;
            }
        } else {
            let kdo = usize::try_from(n / 2).unwrap_or(0);
            for idx in 1..=lc {
                let i = (idx + idx - 2) as f64;
                let mut sum = 0.0_f64;
                for k in 1..=kdo {
                    let kf = k as f64;
                    let t1 = 1.0_f64 - (kf + kf + i).powi(2);
                    let t2 = 1.0_f64 - (kf + kf - i).powi(2);
                    sum += work[k] * (t1 - t2) / (t1 * t2);
                }
                cz[idx] = sc1 * sum;
            }
        }
    } else if mmod == 0 {
        let kdo = usize::try_from((n + 1) / 2).unwrap_or(0);
        for idx in 1..=lc {
            let i = (idx + idx - 1) as f64;
            let mut sum = 0.0_f64;
            for k in 1..=kdo {
                let kf = k as f64;
                let t1 = 1.0_f64 - (kf + kf - 1.0_f64 + i).powi(2);
                let t2 = 1.0_f64 - (kf + kf - 1.0_f64 - i).powi(2);
                sum += work[k] * (t1 + t2) / (t1 * t2);
            }
            cz[idx] = sc1 * sum;
        }
    } else {
        let kdo = usize::try_from((n + 1) / 2).unwrap_or(0);
        for idx in 1..=lc {
            let i = (2_i32 * idx as i32 - 3_i32) as f64;
            let mut sum = 0.0_f64;
            for k in 1..=kdo {
                let kf = k as f64;
                let t1 = 1.0_f64 - (kf + kf - 1.0_f64 + i).powi(2);
                let t2 = 1.0_f64 - (kf + kf - 1.0_f64 - i).powi(2);
                sum += work[k] * (t1 - t2) / (t1 * t2);
            }
            cz[idx] = sc1 * sum;
        }
    }

    cz
}

fn dnzft(nlat: i32, m: i32, n: i32, th: f64, cz: &[f64]) -> f64 {
    let mut zh = 0.0_f64;
    let cdt = (th + th).cos();
    let sdt = (th + th).sin();
    let lmod = nlat % 2;
    let mmod = m % 2;
    let nmod = n % 2;

    if lmod != 0 {
        let lc = usize::try_from((nlat + 1) / 2).unwrap_or(0);
        let lq = lc.saturating_sub(1);
        let ls = lc.saturating_sub(2);
        if nmod == 0 {
            if mmod == 0 {
                zh = 0.5_f64 * (cz[1] + cz[lc] * (((2 * lq) as f64) * th).cos());
                let mut cth = cdt;
                let mut sth = sdt;
                for k in 2..=lq {
                    zh += cz[k] * cth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            } else {
                let mut cth = cdt;
                let mut sth = sdt;
                for k in 1..=ls {
                    zh += cz[k + 1] * sth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            }
        } else if mmod == 0 {
            let mut cth = th.cos();
            let mut sth = th.sin();
            for k in 1..=lq {
                zh += cz[k] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            let mut cth = th.cos();
            let mut sth = th.sin();
            for k in 1..=lq {
                zh += cz[k + 1] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    } else {
        let lc = usize::try_from(nlat / 2).unwrap_or(0);
        let lq = lc.saturating_sub(1);
        if nmod == 0 {
            if mmod == 0 {
                zh = 0.5_f64 * cz[1];
                let mut cth = cdt;
                let mut sth = sdt;
                for k in 2..=lc {
                    zh += cz[k] * cth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            } else {
                let mut cth = cdt;
                let mut sth = sdt;
                for k in 1..=lq {
                    zh += cz[k + 1] * sth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            }
        } else if mmod == 0 {
            zh = 0.5_f64 * cz[lc] * (((nlat - 1) as f64) * th).cos();
            let mut cth = th.cos();
            let mut sth = th.sin();
            for k in 1..=lq {
                zh += cz[k] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            let mut cth = th.cos();
            let mut sth = th.sin();
            for k in 1..=lq {
                zh += cz[k + 1] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    }

    zh
}
