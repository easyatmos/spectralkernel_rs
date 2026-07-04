use crate::hrfftf::fourier_analysis_real;
use crate::legin::legin_compute;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Analyze scalar fields on a Gaussian grid using computed Legendre tables.
///
/// # Parameters
/// - `g`: Input scalar grid values stored in `(nlat, nlon[, nt])` order.
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `wshagc`: Workspace initialized by `shagci_impl` for Gaussian-grid scalar analysis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the cosine coefficients, sine coefficients, and an error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn shagc_impl(
    g: &[f32],
    nlat: usize,
    nlon: usize,
    nt: usize,
    wshagc: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 2;
    if nlon < 4 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    let mode = 0usize;
    let l = ((nlon + 2) / 2).min(nlat);
    let late = (nlat + (nlat % 2)) / 2;
    let lat = nlat;
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    if g.len() != nlat * nlon * nt {
        return Err(PyValueError::new_err("g size mismatch"));
    }
    let need_i64 = (nlat as i64) * (2 * late as i64 + 3 * l as i64 - 2)
        + (3 * l as i64 * (1 - l as i64)) / 2
        + nlon as i64
        + 15;
    let need = usize::try_from(need_i64)
        .map_err(|_| PyValueError::new_err("invalid shagc size formula"))?;
    ierror = 9;
    if wshagc.len() < need {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 10;
    if lwork < nlat * nlon * (nt + 1) {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let ifft = nlat + 2 * nlat * late + 3 * (l * (l - 1) / 2 + (nlat - l) * (l - 1));
    let whrfft = &wshagc[ifft..ifft + nlon + 15];
    let mut grid = g.to_vec();
    for k in 0..nt {
        let mut plane = vec![0.0_f32; lat * nlon];
        for i in 0..lat {
            for j in 0..nlon {
                plane[i * nlon + j] = grid[(i * nlon + j) * nt + k];
            }
        }
        fourier_analysis_real(lat, nlon, &mut plane, whrfft)?;
        let sfn = 2.0_f32 / nlon as f32;
        for i in 0..lat {
            for j in 0..nlon {
                grid[(i * nlon + j) * nt + k] = sfn * plane[i * nlon + j];
            }
        }
    }

    let mut a = vec![0.0_f32; nlat * nlat * nt];
    let mut b = vec![0.0_f32; nlat * nlat * nt];
    let lm1 = if nlon == l + l - 2 { l - 1 } else { l };
    let nl2 = nlat / 2;
    let mut pmn = vec![0.0_f32; nlat * late * 3];
    let mut km_state = (0usize, 1usize, 2usize);

    for k in 0..nt {
        for j in 0..nlon {
            for i in 0..nl2 {
                let is = nlat - i - 1;
                let t1 = grid[(i * nlon + j) * nt + k];
                let t2 = grid[(is * nlon + j) * nt + k];
                grid[(i * nlon + j) * nt + k] = wshagc[i] * (t1 + t2);
                grid[(is * nlon + j) * nt + k] = wshagc[i] * (t1 - t2);
            }
            if nlat % 2 != 0 {
                let idx = ((late - 1) * nlon + j) * nt + k;
                grid[idx] = wshagc[late - 1] * grid[idx];
            }
        }
    }

    let km = legin_compute(mode, l, nlat, 0, wshagc, &mut pmn, &mut km_state);
    for k in 0..nt {
        for i in 1..=late {
            let is = nlat - i + 1;
            for np1 in (1..=nlat).step_by(2) {
                let aidx = (np1 - 1) * nt + k;
                let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                a[aidx] += grid[((i - 1) * nlon) * nt + k] * pmn[pidx];
            }
            for np1 in (2..=nlat).step_by(2) {
                let aidx = (np1 - 1) * nt + k;
                let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                a[aidx] += grid[((is - 1) * nlon) * nt + k] * pmn[pidx];
            }
        }
    }

    for mp1 in 2..=lm1 {
        let m = mp1 - 1;
        let mp2 = m + 2;
        let km = legin_compute(mode, l, nlat, m, wshagc, &mut pmn, &mut km_state);
        for k in 0..nt {
            for i in 1..=late {
                let is = nlat - i + 1;
                for np1 in (mp1..=nlat).step_by(2) {
                    let aidx = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                    let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                    a[aidx] += grid[((i - 1) * nlon + (2 * m - 1)) * nt + k] * pmn[pidx];
                    b[aidx] += grid[((i - 1) * nlon + (2 * m)) * nt + k] * pmn[pidx];
                }
                for np1 in (mp2..=nlat).step_by(2) {
                    let aidx = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                    let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                    a[aidx] += grid[((is - 1) * nlon + (2 * m - 1)) * nt + k] * pmn[pidx];
                    b[aidx] += grid[((is - 1) * nlon + (2 * m)) * nt + k] * pmn[pidx];
                }
            }
        }
    }

    if nlon == l + l - 2 {
        let km = legin_compute(mode, l, nlat, l - 1, wshagc, &mut pmn, &mut km_state);
        for k in 0..nt {
            for i in 1..=late {
                let is = nlat - i + 1;
                for np1 in (l..=nlat).step_by(2) {
                    let aidx = (((l - 1) * nlat) + (np1 - 1)) * nt + k;
                    let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                    a[aidx] += 0.5 * grid[((i - 1) * nlon + (nlon - 1)) * nt + k] * pmn[pidx];
                }
                for np1 in ((l + 1)..=nlat).step_by(2) {
                    let aidx = (((l - 1) * nlat) + (np1 - 1)) * nt + k;
                    let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                    a[aidx] += 0.5 * grid[((is - 1) * nlon + (nlon - 1)) * nt + k] * pmn[pidx];
                }
            }
        }
    }

    Ok((a, b, 0))
}

/// Parallel implementation of `shagc_impl`.
///
/// # Parameters
/// - `g`: Input scalar grid values stored in `(nlat, nlon[, nt])` order.
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `wshagc`: Workspace initialized by `shagci_impl` for Gaussian-grid scalar analysis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the cosine coefficients, sine coefficients, and an error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn shagc_impl_parallel(
    g: &[f32],
    nlat: usize,
    nlon: usize,
    nt: usize,
    wshagc: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 2;
    if nlon < 4 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    let mode = 0usize;
    let l = ((nlon + 2) / 2).min(nlat);
    let late = (nlat + (nlat % 2)) / 2;
    let lat = nlat;
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    if g.len() != nlat * nlon * nt {
        return Err(PyValueError::new_err("g size mismatch"));
    }
    let need_i64 = (nlat as i64) * (2 * late as i64 + 3 * l as i64 - 2)
        + (3 * l as i64 * (1 - l as i64)) / 2
        + nlon as i64
        + 15;
    let need = usize::try_from(need_i64)
        .map_err(|_| PyValueError::new_err("invalid shagc size formula"))?;
    ierror = 9;
    if wshagc.len() < need {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 10;
    if lwork < nlat * nlon * (nt + 1) {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let ifft = nlat + 2 * nlat * late + 3 * (l * (l - 1) / 2 + (nlat - l) * (l - 1));
    let whrfft = &wshagc[ifft..ifft + nlon + 15];
    let lm1 = if nlon == l + l - 2 { l - 1 } else { l };
    let nl2 = nlat / 2;
    let coeff_len = nlat * nlat;
    let coeffs = (0..nt)
        .into_par_iter()
        .map(|k| {
            let mut grid = vec![0.0_f32; lat * nlon];
            for i in 0..lat {
                for j in 0..nlon {
                    grid[i * nlon + j] = g[(i * nlon + j) * nt + k];
                }
            }
            fourier_analysis_real(lat, nlon, &mut grid, whrfft)?;
            let sfn = 2.0_f32 / nlon as f32;
            for value in &mut grid {
                *value *= sfn;
            }

            for j in 0..nlon {
                for i in 0..nl2 {
                    let is = nlat - i - 1;
                    let t1 = grid[i * nlon + j];
                    let t2 = grid[is * nlon + j];
                    grid[i * nlon + j] = wshagc[i] * (t1 + t2);
                    grid[is * nlon + j] = wshagc[i] * (t1 - t2);
                }
                if nlat % 2 != 0 {
                    let idx = (late - 1) * nlon + j;
                    grid[idx] = wshagc[late - 1] * grid[idx];
                }
            }

            let mut ak = vec![0.0_f32; coeff_len];
            let mut bk = vec![0.0_f32; coeff_len];
            let mut pmn = vec![0.0_f32; nlat * late * 3];
            let mut km_state = (0usize, 1usize, 2usize);

            let km = legin_compute(mode, l, nlat, 0, wshagc, &mut pmn, &mut km_state);
            for i in 1..=late {
                let is = nlat - i + 1;
                for np1 in (1..=nlat).step_by(2) {
                    let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                    ak[np1 - 1] += grid[(i - 1) * nlon] * pmn[pidx];
                }
                for np1 in (2..=nlat).step_by(2) {
                    let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                    ak[np1 - 1] += grid[(is - 1) * nlon] * pmn[pidx];
                }
            }

            for mp1 in 2..=lm1 {
                let m = mp1 - 1;
                let mp2 = m + 2;
                let km = legin_compute(mode, l, nlat, m, wshagc, &mut pmn, &mut km_state);
                for i in 1..=late {
                    let is = nlat - i + 1;
                    for np1 in (mp1..=nlat).step_by(2) {
                        let aidx = ((mp1 - 1) * nlat) + (np1 - 1);
                        let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                        ak[aidx] += grid[(i - 1) * nlon + (2 * m - 1)] * pmn[pidx];
                        bk[aidx] += grid[(i - 1) * nlon + (2 * m)] * pmn[pidx];
                    }
                    for np1 in (mp2..=nlat).step_by(2) {
                        let aidx = ((mp1 - 1) * nlat) + (np1 - 1);
                        let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                        ak[aidx] += grid[(is - 1) * nlon + (2 * m - 1)] * pmn[pidx];
                        bk[aidx] += grid[(is - 1) * nlon + (2 * m)] * pmn[pidx];
                    }
                }
            }

            if nlon == l + l - 2 {
                let km = legin_compute(mode, l, nlat, l - 1, wshagc, &mut pmn, &mut km_state);
                for i in 1..=late {
                    let is = nlat - i + 1;
                    for np1 in (l..=nlat).step_by(2) {
                        let aidx = ((l - 1) * nlat) + (np1 - 1);
                        let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                        ak[aidx] += 0.5 * grid[(i - 1) * nlon + (nlon - 1)] * pmn[pidx];
                    }
                    for np1 in ((l + 1)..=nlat).step_by(2) {
                        let aidx = ((l - 1) * nlat) + (np1 - 1);
                        let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                        ak[aidx] += 0.5 * grid[(is - 1) * nlon + (nlon - 1)] * pmn[pidx];
                    }
                }
            }
            Ok::<(Vec<f32>, Vec<f32>), PyErr>((ak, bk))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let mut a = vec![0.0_f32; nlat * nlat * nt];
    let mut b = vec![0.0_f32; nlat * nlat * nt];
    for (k, (ak, bk)) in coeffs.into_iter().enumerate() {
        for idx in 0..coeff_len {
            a[idx * nt + k] = ak[idx];
            b[idx * nt + k] = bk[idx];
        }
    }
    Ok((a, b, 0))
}

#[pyfunction]
/// Python wrapper for `shagc_impl` that accepts rank-2 or rank-3 NumPy arrays.
///
/// # Parameters
/// - `g`: Input scalar grid values stored in `(nlat, nlon[, nt])` order.
/// - `wshagc`: Workspace initialized by `shagci_impl` for Gaussian-grid scalar analysis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn shagc<'py>(
    py: Python<'py>,
    g: PyReadonlyArrayDyn<'py, f32>,
    wshagc: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    let shape = g.shape().to_vec();
    if shape.len() != 2 && shape.len() != 3 {
        return Err(PyValueError::new_err("shagc expects rank-2 or rank-3 g"));
    }
    let nlat = shape[0];
    let nlon = shape[1];
    let nt = if shape.len() == 2 { 1 } else { shape[2] };
    let gbuf = g.as_slice()?.to_vec();
    let wbuf = wshagc.as_slice()?.to_vec();
    let (a, b, ierror) = py
        .detach(|| {
            shagc_impl_parallel(&gbuf, nlat, nlon, nt, &wbuf, lwork).map_err(|err| err.to_string())
        })
        .map_err(PyValueError::new_err)?;
    let ashape = if shape.len() == 2 {
        vec![nlat, nlat]
    } else {
        vec![nlat, nlat, nt]
    };
    let aarr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&ashape), a)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let barr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&ashape), b)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        aarr.into_pyarray(py).into_any().unbind(),
        barr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}
