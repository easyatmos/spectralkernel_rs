use crate::zfinit::zfinit_impl;
use numpy::PyArray1;
use pyo3::prelude::*;

/// Core Rust implementation of `sea1`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
///
/// # Returns
/// A contiguous workspace or coefficient vector stored in double precision.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn sea1_impl(nlat: usize, nlon: usize) -> Vec<f64> {
    let imid = (nlat + 1) / 2;
    let mmax = nlat.min(nlon / 2 + 1);
    let idz = (mmax * (nlat + nlat - mmax + 1)) / 2;
    let (wzfin, _abc) = zfinit_impl(nlat, nlon);
    let mut z = vec![0.0_f64; idz * imid];

    let lim = nlat * imid;
    let labc = ((mmax.saturating_sub(2)) * (nlat + nlat - mmax - 1)) / 2;
    let zz = &wzfin[0..lim];
    let z1 = &wzfin[lim..2 * lim];
    let a = &wzfin[2 * lim..2 * lim + labc];
    let b = &wzfin[2 * lim + labc..2 * lim + 2 * labc];
    let c = &wzfin[2 * lim + 2 * labc..2 * lim + 3 * labc];

    let mut zstate = vec![0.0_f64; imid * nlat * 3];
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
                    zstate[dst] = zz[src];
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
                        zstate[dst] = z1[src];
                    }
                }
            } else {
                let mut ns = ((m - 2) * (nlat + nlat - m - 1)) / 2;
                for i in 1..=imid {
                    let dst = (((i - 1) * nlat) + m) * 3 + i3;
                    let src_m_minus_2 = (((i - 1) * nlat) + (m - 2)) * 3 + i1;
                    let src_m = (((i - 1) * nlat) + m) * 3 + i1;
                    zstate[dst] = a[ns] * zstate[src_m_minus_2] - c[ns] * zstate[src_m];
                }

                if m != nlat - 1 {
                    ns += 1;
                    for i in 1..=imid {
                        let dst = (((i - 1) * nlat) + (m + 1)) * 3 + i3;
                        let src_m_minus_1 = (((i - 1) * nlat) + (m - 1)) * 3 + i1;
                        let src_m_plus_1 = (((i - 1) * nlat) + (m + 1)) * 3 + i1;
                        zstate[dst] = a[ns] * zstate[src_m_minus_1] - c[ns] * zstate[src_m_plus_1];
                    }

                    let mut np1 = m + 3;
                    while np1 <= nlat {
                        ns += 1;
                        for i in 1..=imid {
                            let dst = (((i - 1) * nlat) + (np1 - 1)) * 3 + i3;
                            let src_n_minus_2_i1 = (((i - 1) * nlat) + (np1 - 3)) * 3 + i1;
                            let src_n_minus_2_i3 = (((i - 1) * nlat) + (np1 - 3)) * 3 + i3;
                            let src_n_i1 = (((i - 1) * nlat) + (np1 - 1)) * 3 + i1;
                            zstate[dst] = a[ns] * zstate[src_n_minus_2_i1]
                                + b[ns] * zstate[src_n_minus_2_i3]
                                - c[ns] * zstate[src_n_i1];
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
                let dst = (i - 1) * idz + usize::try_from(mn - 1).unwrap_or(0);
                z[dst] = zstate[src];
            }
        }
    }

    z
}

#[pyfunction]
/// Python helper that exposes the scalar synthesis workspace built by `sea1_impl` for inspection.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
///
/// # Returns
/// A Python result containing a one-dimensional NumPy array.
pub fn sea1_debug<'py>(
    py: Python<'py>,
    nlat: usize,
    nlon: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(PyArray1::from_vec(py, sea1_impl(nlat, nlon)).to_owned())
}
