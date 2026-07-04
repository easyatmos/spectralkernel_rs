use crate::shsgc::shsgc_impl;
use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn collect_logical_ab(view: ndarray::ArrayViewD<'_, f32>) -> Vec<f32> {
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

/// Core Rust implementation of `islapgc`.
///
/// # Parameters
/// - `nlon`: Number of longitudes in the grid.
/// - `xlmbda`: Parameter `xlmbda` passed through to the routine.
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `nt`: Number of stacked fields processed together.
/// - `wshsgc`: Workspace initialized by `shsgci_impl` for Gaussian-grid scalar synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the cosine coefficients, sine coefficients, and an error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn islapgc_impl(
    nlon: usize,
    xlmbda: &[f32],
    a: &[f32],
    b: &[f32],
    nlat: usize,
    isym: usize,
    nt: usize,
    wshsgc: &[f32],
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
    ierror = 3;
    if isym > 2 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    if xlmbda.len() != nt {
        return Err(PyValueError::new_err("xlmbda size mismatch"));
    }
    if a.len() != b.len() || a.len() != nlat * nlat * nt {
        return Err(PyValueError::new_err("a/b size mismatch"));
    }

    let imid = nlat.div_ceil(2);
    let l1 = nlat.min((nlon + 2) / 2);
    let l2 = imid;
    let lwmin_i64 = (nlat as i64) * (2 * l2 as i64 + 3 * l1 as i64 - 2)
        + (3 * l1 as i64 * (1 - l1 as i64)) / 2
        + nlon as i64
        + 15;
    let lwmin = usize::try_from(lwmin_i64)
        .map_err(|_| PyValueError::new_err("invalid lshsgc minimum length computation"))?;
    ierror = 9;
    if wshsgc.len() < lwmin {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let mmax = nlat.min(nlon / 2 + 1);
    let lwkmin = if isym == 0 {
        nlat * (2 * nt * nlon + (6 * l2).max(nlon) + 2 * l1 * nt + 1)
    } else {
        l2 * (2 * nt * nlon + (6 * nlat).max(nlon)) + nlat * (2 * l1 * nt + 1)
    };
    ierror = 10;
    if lwork < lwkmin {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    ierror = 0;
    let mut pertrb = vec![0.0_f32; nt];
    for (k, &lam) in xlmbda.iter().enumerate() {
        if lam < 0.0 {
            ierror = -1;
        }
        if lam == 0.0 {
            pertrb[k] = a[k] / (2.0 * 2.0_f32.sqrt());
        }
    }

    let mut as_ = vec![0.0_f32; nlat * nlat * nt];
    let mut bs_ = vec![0.0_f32; nlat * nlat * nt];
    let fnn: Vec<f32> = (0..nlat)
        .map(|idx| {
            let f = idx as f32;
            f * (f + 1.0)
        })
        .collect();

    for k in 0..nt {
        let lam = xlmbda[k];
        if lam == 0.0 {
            for n in 2..=nlat {
                let idx = (n - 1) * nt + k;
                as_[idx] = -a[idx] / fnn[n - 1];
                bs_[idx] = -b[idx] / fnn[n - 1];
            }
            for m in 2..=mmax {
                for n in m..=nlat {
                    let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                    as_[idx] = -a[idx] / fnn[n - 1];
                    bs_[idx] = -b[idx] / fnn[n - 1];
                }
            }
        } else {
            pertrb[k] = 0.0;
            for n in 1..=nlat {
                let idx = (n - 1) * nt + k;
                as_[idx] = -a[idx] / (fnn[n - 1] + lam);
                bs_[idx] = -b[idx] / (fnn[n - 1] + lam);
            }
            for m in 2..=mmax {
                for n in m..=nlat {
                    let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                    as_[idx] = -a[idx] / (fnn[n - 1] + lam);
                    bs_[idx] = -b[idx] / (fnn[n - 1] + lam);
                }
            }
        }
    }

    let ls = if isym == 0 { nlat } else { imid };
    let lwork_shsgc = if isym == 0 {
        nlat * nlon * (nt + 1)
    } else {
        ls * nlon * nt + nlat * nlon
    };
    let (sf, ierr) = shsgc_impl(&as_, &bs_, nlat, nt, wshsgc, lwork_shsgc)?;
    Ok((sf, pertrb, if ierror != 0 { ierror } else { ierr }))
}

#[pyfunction]
/// Rust entry point for `islapgc`.
///
/// # Parameters
/// - `nlon`: Number of longitudes in the grid.
/// - `xlmbda`: Parameter `xlmbda` passed through to the routine.
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `wshsgc`: Workspace initialized by `shsgci_impl` for Gaussian-grid scalar synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn islapgc<'py>(
    py: Python<'py>,
    nlon: usize,
    xlmbda: PyReadonlyArray1<'py, f32>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    wshsgc: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err(
            "islapgc expects rank-2 or rank-3 a/b",
        ));
    }

    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let abuf = collect_logical_ab(a.as_array());
    let bbuf = collect_logical_ab(b.as_array());
    let (sf, pertrb, ierror) = islapgc_impl(
        nlon,
        xlmbda.as_slice()?,
        &abuf,
        &bbuf,
        nlat,
        0,
        nt,
        wshsgc.as_slice()?,
        lwork,
    )?;
    if ierror != 0 {
        let empty2 = ArrayD::<f32>::from_shape_vec(IxDyn(&[0, 0]), Vec::new())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let empty1 = ArrayD::<f32>::from_shape_vec(IxDyn(&[0]), Vec::new())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok((
            empty2.into_pyarray(py).into_any().unbind(),
            empty1.into_pyarray(py).into_any().unbind(),
            ierror,
        ));
    }

    let ids = nlat;
    let shape = if ashape.len() == 2 {
        IxDyn(&[ids, nlon])
    } else {
        IxDyn(&[ids, nlon, nt])
    };
    let sf_arr =
        ArrayD::from_shape_vec(shape, sf).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let p_arr = ArrayD::from_shape_vec(IxDyn(&[nt]), pertrb)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        sf_arr.into_pyarray(py).into_any().unbind(),
        p_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}

#[pyfunction]
/// Rust entry point for `islapgc_isym`.
///
/// # Parameters
/// - `nlon`: Number of longitudes in the grid.
/// - `xlmbda`: Parameter `xlmbda` passed through to the routine.
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `wshsgc`: Workspace initialized by `shsgci_impl` for Gaussian-grid scalar synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn islapgc_isym<'py>(
    py: Python<'py>,
    nlon: usize,
    xlmbda: PyReadonlyArray1<'py, f32>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    isym: usize,
    wshsgc: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err(
            "islapgc_isym expects rank-2 or rank-3 a/b",
        ));
    }

    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let abuf = collect_logical_ab(a.as_array());
    let bbuf = collect_logical_ab(b.as_array());
    let (sf, pertrb, ierror) = islapgc_impl(
        nlon,
        xlmbda.as_slice()?,
        &abuf,
        &bbuf,
        nlat,
        isym,
        nt,
        wshsgc.as_slice()?,
        lwork,
    )?;
    if ierror != 0 {
        let empty2 = ArrayD::<f32>::from_shape_vec(IxDyn(&[0, 0]), Vec::new())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let empty1 = ArrayD::<f32>::from_shape_vec(IxDyn(&[0]), Vec::new())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok((
            empty2.into_pyarray(py).into_any().unbind(),
            empty1.into_pyarray(py).into_any().unbind(),
            ierror,
        ));
    }

    let ids = if isym == 0 { nlat } else { nlat.div_ceil(2) };
    let shape = if ashape.len() == 2 {
        IxDyn(&[ids, nlon])
    } else {
        IxDyn(&[ids, nlon, nt])
    };
    let sf_arr =
        ArrayD::from_shape_vec(shape, sf).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let p_arr = ArrayD::from_shape_vec(IxDyn(&[nt]), pertrb)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        sf_arr.into_pyarray(py).into_any().unbind(),
        p_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}
