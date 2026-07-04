use crate::vhsec::vhsec_impl;
use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn collect_logical_coeffs(view: ndarray::ArrayViewD<'_, f32>) -> Vec<f32> {
    match view.ndim() {
        2 => {
            let nlat = view.shape()[0];
            let mut out = Vec::with_capacity(nlat * nlat);
            for m in 0..nlat {
                for n in 0..nlat {
                    out.push(view[[m, n]]);
                }
            }
            out
        }
        3 => {
            let nlat = view.shape()[0];
            let nt = view.shape()[2];
            let mut out = Vec::with_capacity(nlat * nlat * nt);
            for m in 0..nlat {
                for n in 0..nlat {
                    for k in 0..nt {
                        out.push(view[[m, n, k]]);
                    }
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

fn expand_ivlapec_output(
    data: Vec<f32>,
    nlat: usize,
    nlon: usize,
    nt: usize,
    ityp: usize,
) -> Vec<f32> {
    let idvw = if ityp <= 2 { nlat } else { nlat.div_ceil(2) };
    if idvw == nlat {
        return data;
    }
    let mut full = vec![0.0_f32; nlat * nlon * nt];
    for i in 0..idvw {
        for j in 0..nlon {
            for k in 0..nt {
                full[(i * nlon + j) * nt + k] = data[(i * nlon + j) * nt + k];
            }
        }
    }
    full
}

/// Core Rust implementation of `ivlapec`.
///
/// # Parameters
/// - `nlon`: Number of longitudes in the grid.
/// - `br`: First vector coefficient family in vector layout.
/// - `bi`: Second vector coefficient family in vector layout.
/// - `cr`: Third vector coefficient family in vector layout.
/// - `ci`: Fourth vector coefficient family in vector layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `ityp`: Vector storage selector controlling the coefficient families in use.
/// - `wvhsec`: Workspace initialized by `vhseci_impl` for regular-grid vector synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the cosine coefficients, sine coefficients, and an error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn ivlapec_impl(
    nlon: usize,
    br: &[f32],
    bi: &[f32],
    cr: &[f32],
    ci: &[f32],
    nlat: usize,
    nt: usize,
    ityp: usize,
    wvhsec: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 2;
    if nlon < 1 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 3;
    if ityp > 8 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let expected = nlat * nlat * nt;
    if br.len() != expected || bi.len() != expected || cr.len() != expected || ci.len() != expected
    {
        return Err(PyValueError::new_err("br/bi/cr/ci size mismatch"));
    }

    let imid = nlat.div_ceil(2);
    let mmax = nlat.min((nlon + 1) / 2);
    let lzz1 = 2 * nlat * imid;
    let labc = 3 * mmax.saturating_sub(2) * (2 * nlat - mmax - 1) / 2;
    let need_lvhsec = 2 * (lzz1 + labc) + nlon + 15;
    ierror = 9;
    if wvhsec.len() < need_lvhsec {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let mn = mmax * nlat * nt;
    let lwkmin = if ityp < 3 {
        if ityp == 0 {
            nlat * (2 * nt * nlon + (6 * imid).max(nlon) + 1) + 4 * mn
        } else {
            nlat * (2 * nt * nlon + (6 * imid).max(nlon) + 1) + 2 * mn
        }
    } else if matches!(ityp, 3 | 6) {
        imid * (2 * nt * nlon + (6 * nlat).max(nlon)) + 4 * mn + nlat
    } else {
        imid * (2 * nt * nlon + (6 * nlat).max(nlon)) + 2 * mn + nlat
    };
    ierror = 10;
    if lwork < lwkmin {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let mut brvw = vec![0.0_f32; expected];
    let mut bivw = vec![0.0_f32; expected];
    let mut crvw = vec![0.0_f32; expected];
    let mut civw = vec![0.0_f32; expected];
    let fnn: Vec<f32> = (0..nlat)
        .map(|idx| {
            if idx == 0 {
                0.0
            } else {
                let f = idx as f32;
                -1.0 / (f * (f + 1.0))
            }
        })
        .collect();

    match ityp {
        0 | 3 | 6 => {
            for k in 0..nt {
                for n in 2..=nlat {
                    let idx = (n - 1) * nt + k;
                    brvw[idx] = fnn[n - 1] * br[idx];
                    bivw[idx] = fnn[n - 1] * bi[idx];
                    crvw[idx] = fnn[n - 1] * cr[idx];
                    civw[idx] = fnn[n - 1] * ci[idx];
                }
                for m in 2..=mmax {
                    for n in m..=nlat {
                        let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                        brvw[idx] = fnn[n - 1] * br[idx];
                        bivw[idx] = fnn[n - 1] * bi[idx];
                        crvw[idx] = fnn[n - 1] * cr[idx];
                        civw[idx] = fnn[n - 1] * ci[idx];
                    }
                }
            }
        }
        1 | 4 | 7 => {
            for k in 0..nt {
                for n in 2..=nlat {
                    let idx = (n - 1) * nt + k;
                    brvw[idx] = fnn[n - 1] * br[idx];
                    bivw[idx] = fnn[n - 1] * bi[idx];
                }
                for m in 2..=mmax {
                    for n in m..=nlat {
                        let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                        brvw[idx] = fnn[n - 1] * br[idx];
                        bivw[idx] = fnn[n - 1] * bi[idx];
                    }
                }
            }
        }
        _ => {
            for k in 0..nt {
                for n in 2..=nlat {
                    let idx = (n - 1) * nt + k;
                    crvw[idx] = fnn[n - 1] * cr[idx];
                    civw[idx] = fnn[n - 1] * ci[idx];
                }
                for m in 2..=mmax {
                    for n in m..=nlat {
                        let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                        crvw[idx] = fnn[n - 1] * cr[idx];
                        civw[idx] = fnn[n - 1] * ci[idx];
                    }
                }
            }
        }
    }

    let lwork_vhsec = if matches!(ityp, 0 | 3 | 6) {
        lwork.saturating_sub(4 * mn + nlat)
    } else {
        lwork.saturating_sub(2 * mn + nlat)
    };
    vhsec_impl(
        &brvw,
        &bivw,
        &crvw,
        &civw,
        nlat,
        nt,
        ityp,
        &wvhsec[..need_lvhsec],
        lwork_vhsec,
    )
}

#[pyfunction]
/// Rust entry point for `ivlapec`.
///
/// # Parameters
/// - `nlon`: Number of longitudes in the grid.
/// - `br`: First vector coefficient family in vector layout.
/// - `bi`: Second vector coefficient family in vector layout.
/// - `cr`: Third vector coefficient family in vector layout.
/// - `ci`: Fourth vector coefficient family in vector layout.
/// - `wvhsec`: Workspace initialized by `vhseci_impl` for regular-grid vector synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn ivlapec<'py>(
    py: Python<'py>,
    nlon: usize,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    wvhsec: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    ivlapec_ityp(py, nlon, br, bi, cr, ci, 0, wvhsec, lwork)
}

#[pyfunction]
/// Rust entry point for `ivlapec_ityp`.
///
/// # Parameters
/// - `nlon`: Number of longitudes in the grid.
/// - `br`: First vector coefficient family in vector layout.
/// - `bi`: Second vector coefficient family in vector layout.
/// - `cr`: Third vector coefficient family in vector layout.
/// - `ci`: Fourth vector coefficient family in vector layout.
/// - `ityp`: Vector storage selector controlling the coefficient families in use.
/// - `wvhsec`: Workspace initialized by `vhseci_impl` for regular-grid vector synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn ivlapec_ityp<'py>(
    py: Python<'py>,
    nlon: usize,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    ityp: usize,
    wvhsec: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    let shape = br.shape().to_vec();
    if shape != bi.shape() || shape != cr.shape() || shape != ci.shape() {
        return Err(PyValueError::new_err(
            "br/bi/cr/ci must have identical shapes",
        ));
    }
    if shape.len() != 2 && shape.len() != 3 {
        return Err(PyValueError::new_err(
            "ivlapec expects rank-2 or rank-3 coefficient arrays",
        ));
    }
    let nlat = shape[0];
    let nt = if shape.len() == 2 { 1 } else { shape[2] };
    let (v, w, ierror) = ivlapec_impl(
        nlon,
        &collect_logical_coeffs(br.as_array()),
        &collect_logical_coeffs(bi.as_array()),
        &collect_logical_coeffs(cr.as_array()),
        &collect_logical_coeffs(ci.as_array()),
        nlat,
        nt,
        ityp,
        wvhsec.as_slice()?,
        lwork,
    )?;
    if ierror != 0 {
        let empty = ArrayD::<f32>::from_shape_vec(IxDyn(&[0, 0]), Vec::new())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok((
            empty.clone().into_pyarray(py).into_any().unbind(),
            empty.into_pyarray(py).into_any().unbind(),
            ierror,
        ));
    }
    let v = expand_ivlapec_output(v, nlat, nlon, nt, ityp);
    let w = expand_ivlapec_output(w, nlat, nlon, nt, ityp);
    let out_shape = if shape.len() == 2 {
        IxDyn(&[nlat, nlon])
    } else {
        IxDyn(&[nlat, nlon, nt])
    };
    let v_arr = ArrayD::from_shape_vec(out_shape.clone(), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let w_arr =
        ArrayD::from_shape_vec(out_shape, w).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        v_arr.into_pyarray(py).into_any().unbind(),
        w_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}
