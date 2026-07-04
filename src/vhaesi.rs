use crate::hrffti::hrffti_impl;
use crate::sphcom_vector::{dzvk, dzvt, dzwk, dzwt};
use numpy::PyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;

fn direct_zv_column(nlat: usize, m: usize) -> Vec<f32> {
    let imid = (nlat + 1) / 2;
    let dt = std::f64::consts::PI / (nlat.saturating_sub(1) as f64);
    let mut out = vec![0.0_f32; imid * nlat];
    for np1 in (m + 1)..=nlat {
        let coeff = dzvk(nlat as i32, m as i32, (np1 - 1) as i32);
        for i in 1..=imid {
            let th = (i - 1) as f64 * dt;
            let mut val = dzvt(nlat as i32, m as i32, (np1 - 1) as i32, th, &coeff) as f32;
            if i == 1 {
                val *= 0.5;
            }
            out[(np1 - 1) * imid + (i - 1)] = val;
        }
    }
    out
}

fn direct_zw_column(nlat: usize, m: usize) -> Vec<f32> {
    let imid = (nlat + 1) / 2;
    let dt = std::f64::consts::PI / (nlat.saturating_sub(1) as f64);
    let mut out = vec![0.0_f32; imid * nlat];
    let mw = m.max(1);
    for np1 in (mw + 1)..=nlat {
        let coeff = dzwk(nlat as i32, mw as i32, (np1 - 1) as i32);
        for i in 1..=imid {
            let th = (i - 1) as f64 * dt;
            let mut val = dzwt(nlat as i32, mw as i32, (np1 - 1) as i32, th, &coeff) as f32;
            if i == 1 {
                val *= 0.5;
            }
            out[(np1 - 1) * imid + (i - 1)] = val;
        }
    }
    out
}

/// Initialize the workspace required by `vhaes_impl`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `lvhaes`: Declared length of the `wvhaes` workspace.
/// - `lwork`: Length of the caller-provided work array.
/// - `ldwork`: Length of the auxiliary workspace expected by the low-level interface.
///
/// # Returns
/// A tuple containing the workspace/result vector and the error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vhaesi_impl(nlat: i32, nlon: i32, lvhaes: i32, lwork: i32, ldwork: i32) -> (Vec<f32>, i32) {
    let mut ierror = 1;
    if nlat < 3 {
        return (Vec::new(), ierror);
    }
    ierror = 2;
    if nlon < 1 {
        return (Vec::new(), ierror);
    }
    ierror = 3;
    let mmax = nlat.min((nlon + 1) / 2);
    let imid = (nlat + 1) / 2;
    let lzimn = (imid * mmax * (nlat + nlat - mmax + 1)) / 2;
    if lvhaes < lzimn + lzimn + nlon + 15 {
        return (Vec::new(), ierror);
    }
    ierror = 4;
    let labc = 3 * (0.max(mmax - 2) * (nlat + nlat - mmax - 1)) / 2;
    if lwork < 5 * nlat * imid + labc {
        return (Vec::new(), ierror);
    }
    ierror = 5;
    if ldwork < 2 * (nlat + 1) {
        return (Vec::new(), ierror);
    }

    let nlat_usize = usize::try_from(nlat).unwrap_or(0);
    let lzimn_usize = usize::try_from(lzimn).unwrap_or(0);
    let idz = usize::try_from((mmax * (nlat + nlat - mmax + 1)) / 2).unwrap_or(0);
    let mut out = vec![0.0_f32; usize::try_from(lvhaes).unwrap_or(0)];

    for mp1 in 1..=usize::try_from(mmax).unwrap_or(0) {
        let m = mp1 - 1;
        let mb = if m == 0 {
            0
        } else {
            m * (nlat_usize - 1) - (m * (m - 1)) / 2
        };
        let col = direct_zv_column(nlat_usize, m);
        for np1 in mp1..=nlat_usize {
            let mn = mb + np1;
            for i in 1..=nlat_usize.div_ceil(2) {
                let src = (np1 - 1) * nlat_usize.div_ceil(2) + (i - 1);
                let dst = (mn - 1) + (i - 1) * idz;
                if dst < out.len() {
                    out[dst] = col[src] as f32;
                }
            }
        }
    }

    for mp1 in 1..=usize::try_from(mmax).unwrap_or(0) {
        let m = mp1 - 1;
        let mb = if m == 0 {
            0
        } else {
            m * (nlat_usize - 1) - (m * (m - 1)) / 2
        };
        let col = direct_zw_column(nlat_usize, m);
        for np1 in mp1..=nlat_usize {
            let mn = mb + np1;
            for i in 1..=nlat_usize.div_ceil(2) {
                let src = (np1 - 1) * nlat_usize.div_ceil(2) + (i - 1);
                let dst = lzimn_usize + (mn - 1) + (i - 1) * idz;
                if dst < out.len() {
                    out[dst] = col[src] as f32;
                }
            }
        }
    }
    let fft = hrffti_impl(nlon);
    for (i, v) in fft.iter().enumerate() {
        let dst = 2 * lzimn_usize + i;
        if dst < out.len() {
            out[dst] = *v;
        }
    }
    (out, 0)
}

/// Parallel initializer for the workspace required by `vhaes_impl`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `lvhaes`: Declared length of the `wvhaes` workspace.
/// - `lwork`: Length of the caller-provided work array.
/// - `ldwork`: Length of the auxiliary workspace expected by the low-level interface.
///
/// # Returns
/// A tuple containing the workspace/result vector and the error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vhaesi_impl_parallel(
    nlat: i32,
    nlon: i32,
    lvhaes: i32,
    lwork: i32,
    ldwork: i32,
) -> (Vec<f32>, i32) {
    let mut ierror = 1;
    if nlat < 3 {
        return (Vec::new(), ierror);
    }
    ierror = 2;
    if nlon < 1 {
        return (Vec::new(), ierror);
    }
    ierror = 3;
    let mmax = nlat.min((nlon + 1) / 2);
    let imid = (nlat + 1) / 2;
    let lzimn = (imid * mmax * (nlat + nlat - mmax + 1)) / 2;
    if lvhaes < lzimn + lzimn + nlon + 15 {
        return (Vec::new(), ierror);
    }
    ierror = 4;
    let labc = 3 * (0.max(mmax - 2) * (nlat + nlat - mmax - 1)) / 2;
    if lwork < 5 * nlat * imid + labc {
        return (Vec::new(), ierror);
    }
    ierror = 5;
    if ldwork < 2 * (nlat + 1) {
        return (Vec::new(), ierror);
    }

    let nlat_usize = usize::try_from(nlat).unwrap_or(0);
    let imid_usize = nlat_usize.div_ceil(2);
    let mmax_usize = usize::try_from(mmax).unwrap_or(0);
    let lzimn_usize = usize::try_from(lzimn).unwrap_or(0);
    let idz = usize::try_from((mmax * (nlat + nlat - mmax + 1)) / 2).unwrap_or(0);
    let out_len = usize::try_from(lvhaes).unwrap_or(0);
    let mut out = vec![0.0_f32; out_len];

    let zv_chunks: Vec<Vec<(usize, f32)>> = (1..=mmax_usize)
        .into_par_iter()
        .map(|mp1| {
            let m = mp1 - 1;
            let mb = if m == 0 {
                0
            } else {
                m * (nlat_usize - 1) - (m * (m - 1)) / 2
            };
            let col = direct_zv_column(nlat_usize, m);
            let mut writes = Vec::new();
            for np1 in mp1..=nlat_usize {
                let mn = mb + np1;
                for i in 1..=imid_usize {
                    let src = (np1 - 1) * imid_usize + (i - 1);
                    let dst = (mn - 1) + (i - 1) * idz;
                    if dst < out_len {
                        writes.push((dst, col[src]));
                    }
                }
            }
            writes
        })
        .collect();
    for chunk in zv_chunks {
        for (dst, value) in chunk {
            out[dst] = value;
        }
    }

    let zw_chunks: Vec<Vec<(usize, f32)>> = (1..=mmax_usize)
        .into_par_iter()
        .map(|mp1| {
            let m = mp1 - 1;
            let mb = if m == 0 {
                0
            } else {
                m * (nlat_usize - 1) - (m * (m - 1)) / 2
            };
            let col = direct_zw_column(nlat_usize, m);
            let mut writes = Vec::new();
            for np1 in mp1..=nlat_usize {
                let mn = mb + np1;
                for i in 1..=imid_usize {
                    let src = (np1 - 1) * imid_usize + (i - 1);
                    let dst = lzimn_usize + (mn - 1) + (i - 1) * idz;
                    if dst < out_len {
                        writes.push((dst, col[src]));
                    }
                }
            }
            writes
        })
        .collect();
    for chunk in zw_chunks {
        for (dst, value) in chunk {
            out[dst] = value;
        }
    }

    let fft = hrffti_impl(nlon);
    for (i, v) in fft.iter().enumerate() {
        let dst = 2 * lzimn_usize + i;
        if dst < out.len() {
            out[dst] = *v;
        }
    }
    (out, 0)
}

#[pyfunction]
/// Python wrapper for `vhaesi_impl` that returns the initialized workspace.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `lvhaes`: Declared length of the `wvhaes` workspace.
/// - `lwork`: Length of the caller-provided work array.
/// - `ldwork`: Length of the auxiliary workspace expected by the low-level interface.
///
/// # Returns
/// A one-dimensional NumPy workspace array together with a error code.
pub fn vhaesi<'py>(
    py: Python<'py>,
    nlat: i32,
    nlon: i32,
    lvhaes: i32,
    lwork: i32,
    ldwork: i32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, i32)> {
    let (wvhaes, ierror) = vhaesi_impl(nlat, nlon, lvhaes, lwork, ldwork);
    Ok((PyArray1::from_vec(py, wvhaes).to_owned(), ierror))
}
