use crate::sphcom_scalar::{dnlfk, dnlft, rabcp1};

/// Core Rust implementation of `alinit`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
///
/// # Returns
/// A contiguous workspace or coefficient vector stored in double precision.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn alinit_impl(nlat: usize, nlon: usize) -> Vec<f64> {
    let imid = (nlat + 1) / 2;
    let pi = 4.0_f64 * (1.0_f64).atan();
    let dt = pi / ((nlat - 1) as f64);

    let mut p = vec![0.0_f64; imid * nlat * 2];
    for mp1 in 1..=2.min(nlat) {
        let m = (mp1 - 1) as i32;
        for np1 in mp1..=nlat {
            let n = (np1 - 1) as i32;
            let cp = dnlfk(m, n);
            for i in 1..=imid {
                let th = ((i - 1) as f64) * dt;
                let ph = dnlft(m, n, th, &cp);
                let idx = (((np1 - 1) * imid) + (i - 1)) * 2 + (mp1 - 1);
                p[idx] = ph;
            }
        }
    }

    let (a, b, c) = rabcp1(nlat, nlon);
    let labc = a.len().saturating_sub(1);
    let mut walin = vec![0.0_f64; 2 * imid * nlat + 3 * labc];

    for i in 0..(imid * nlat) {
        walin[i] = p[i * 2];
        walin[imid * nlat + i] = p[i * 2 + 1];
    }
    for i in 0..labc {
        walin[2 * imid * nlat + i] = a[i + 1];
        walin[2 * imid * nlat + labc + i] = b[i + 1];
        walin[2 * imid * nlat + 2 * labc + i] = c[i + 1];
    }

    walin
}

/// Core Rust implementation of `ses1`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
///
/// # Returns
/// A contiguous workspace or coefficient vector stored in double precision.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn ses1_impl(nlat: usize, nlon: usize) -> Vec<f64> {
    let imid = (nlat + 1) / 2;
    let mmax = nlat.min(nlon / 2 + 1);
    let idp = (mmax * (nlat + nlat - mmax + 1)) / 2;
    let walin = alinit_impl(nlat, nlon);

    let lim = nlat * imid;
    let labc = ((mmax.saturating_sub(2)) * (nlat + nlat - mmax - 1)) / 2;
    let pz = &walin[0..lim];
    let p1 = &walin[lim..2 * lim];
    let a = &walin[2 * lim..2 * lim + labc];
    let b = &walin[2 * lim + labc..2 * lim + 2 * labc];
    let c = &walin[2 * lim + 2 * labc..2 * lim + 3 * labc];

    let mut p = vec![0.0_f64; imid * idp];
    let mut pstate = vec![0.0_f64; imid * nlat * 3];
    let mut i1 = 0_usize;
    let mut i2 = 1_usize;
    let mut i3 = 2_usize;

    for mp1 in 1..=mmax {
        let m = mp1 - 1;
        if m == 0 {
            i1 = 0;
            i2 = 1;
            i3 = 2;
            for np1 in 1..=nlat {
                for i in 1..=imid {
                    let dst = (((i - 1) * nlat) + (np1 - 1)) * 3 + i3;
                    let src = (np1 - 1) * imid + (i - 1);
                    pstate[dst] = pz[src];
                }
            }
        } else {
            let ihold = i1;
            i1 = i2;
            i2 = i3;
            i3 = ihold;

            if m == 1 {
                for np1 in 2..=nlat {
                    for i in 1..=imid {
                        let dst = (((i - 1) * nlat) + (np1 - 1)) * 3 + i3;
                        let src = (np1 - 1) * imid + (i - 1);
                        pstate[dst] = p1[src];
                    }
                }
            } else {
                let mut ns = ((m - 2) * (nlat + nlat - m - 1)) / 2;
                for i in 1..=imid {
                    let dst = (((i - 1) * nlat) + m) * 3 + i3;
                    let src_m_minus_2 = (((i - 1) * nlat) + (m - 2)) * 3 + i1;
                    let src_m = (((i - 1) * nlat) + m) * 3 + i1;
                    pstate[dst] = a[ns] * pstate[src_m_minus_2] - c[ns] * pstate[src_m];
                }

                if m != nlat - 1 {
                    ns += 1;
                    for i in 1..=imid {
                        let dst = (((i - 1) * nlat) + (m + 1)) * 3 + i3;
                        let src_m_minus_1 = (((i - 1) * nlat) + (m - 1)) * 3 + i1;
                        let src_m_plus_1 = (((i - 1) * nlat) + (m + 1)) * 3 + i1;
                        pstate[dst] = a[ns] * pstate[src_m_minus_1] - c[ns] * pstate[src_m_plus_1];
                    }

                    let mut np1 = m + 3;
                    while np1 <= nlat {
                        ns += 1;
                        for i in 1..=imid {
                            let dst = (((i - 1) * nlat) + (np1 - 1)) * 3 + i3;
                            let src_n_minus_2_i1 = (((i - 1) * nlat) + (np1 - 3)) * 3 + i1;
                            let src_n_minus_2_i3 = (((i - 1) * nlat) + (np1 - 3)) * 3 + i3;
                            let src_n_i1 = (((i - 1) * nlat) + (np1 - 1)) * 3 + i1;
                            pstate[dst] = a[ns] * pstate[src_n_minus_2_i1]
                                + b[ns] * pstate[src_n_minus_2_i3]
                                - c[ns] * pstate[src_n_i1];
                        }
                        np1 += 1;
                    }
                }
            }
        }

        let m_i32 = m as i32;
        for np1 in mp1..=nlat {
            let mn = m_i32 * (nlat as i32 - 1) - (m_i32 * (m_i32 - 1)) / 2 + np1 as i32;
            for i in 1..=imid {
                let src = (((i - 1) * nlat) + (np1 - 1)) * 3 + i3;
                let dst = usize::try_from(mn - 1).unwrap_or(0) * imid + (i - 1);
                p[dst] = pstate[src];
            }
        }
    }

    p
}

/// Rust entry point for `alin_column`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `m`: Zonal wavenumber or refinement level, depending on the routine.
/// - `walin`: Stored scalar analysis workspace produced by `alinit_impl`.
///
/// # Returns
/// A contiguous workspace or coefficient vector stored in double precision.
pub fn alin_column(nlat: usize, nlon: usize, isym: i32, m: usize, walin: &[f64]) -> Vec<f64> {
    let imid = (nlat + 1) / 2;
    let lim = nlat * imid;
    let mmax = nlat.min(nlon / 2 + 1);
    let labc = ((mmax.saturating_sub(2)) * (nlat + nlat - mmax - 1)) / 2;
    let pz = &walin[0..lim];
    let p1 = &walin[lim..2 * lim];
    let a = &walin[2 * lim..2 * lim + labc];
    let b = &walin[2 * lim + labc..2 * lim + 2 * labc];
    let c = &walin[2 * lim + 2 * labc..2 * lim + 3 * labc];

    let mut p = vec![0.0_f64; imid * nlat * 3];
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
                    p[dst] = pz[src];
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
                        p[dst] = p1[src];
                    }
                }
            } else {
                let mut ns = ((mm - 2) * (nlat + nlat - mm - 1)) / 2;
                if isym != 1 {
                    for i in 1..=imid {
                        let dst = (((i - 1) * nlat) + mm) * 3 + i3;
                        let src_m_minus_2 = (((i - 1) * nlat) + (mm - 2)) * 3 + i1;
                        let src_m = (((i - 1) * nlat) + mm) * 3 + i1;
                        p[dst] = a[ns] * p[src_m_minus_2] - c[ns] * p[src_m];
                    }
                }
                if mm != nlat - 1 {
                    if isym != 2 {
                        ns += 1;
                        for i in 1..=imid {
                            let dst = (((i - 1) * nlat) + (mm + 1)) * 3 + i3;
                            let src_m_minus_1 = (((i - 1) * nlat) + (mm - 1)) * 3 + i1;
                            let src_m_plus_1 = (((i - 1) * nlat) + (mm + 1)) * 3 + i1;
                            p[dst] = a[ns] * p[src_m_minus_1] - c[ns] * p[src_m_plus_1];
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
                                p[dst] = a[ns] * p[src_n_minus_2_i1] + b[ns] * p[src_n_minus_2_i3]
                                    - c[ns] * p[src_n_i1];
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
            out[dst] = p[src];
        }
    }
    out
}
