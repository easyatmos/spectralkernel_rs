use crate::vhses::vhses_impl;
use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
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

fn infer_nlon_from_wvhses(nlat: usize, ltotal: usize) -> PyResult<usize> {
    let imid = nlat.div_ceil(2);
    for cand in 4..=4 * nlat.max(4) {
        let mmax = nlat.min((cand + 1) / 2);
        let lzimn = (imid * mmax * (2 * nlat - mmax + 1)) / 2;
        let need = 2 * lzimn + cand + 15;
        if need == ltotal {
            return Ok(cand);
        }
        let mmax_even = nlat.min((cand + 2) / 2);
        let lzimn_even = (imid * mmax_even * (2 * nlat - mmax_even + 1)) / 2;
        let need_even = 2 * lzimn_even + cand + 15;
        if need_even == ltotal {
            return Ok(cand);
        }
    }
    Err(PyValueError::new_err(
        "failed to infer nlon from wvhses length",
    ))
}

fn ityp_from_isym(isym: usize) -> PyResult<usize> {
    match isym {
        0 => Ok(2),
        1 => Ok(5),
        2 => Ok(8),
        _ => Err(PyValueError::new_err("isym must be 0, 1 or 2")),
    }
}

/// Core Rust implementation of `ivrtes`.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `wvhses`: Workspace initialized by `vhsesi_impl` for regular-grid vector synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn ivrtes_impl(
    a: &[f32],
    b: &[f32],
    nlat: usize,
    nt: usize,
    isym: usize,
    wvhses: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let nlon = infer_nlon_from_wvhses(nlat, wvhses.len())?;
    let ityp = ityp_from_isym(isym)?;

    ierror = 2;
    if nlon < 4 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    if a.len() != b.len() || a.len() != nlat * nlat * nt {
        return Err(PyValueError::new_err("a/b size mismatch"));
    }

    let imid = nlat.div_ceil(2);
    let mmax = nlat.min((nlon + 1) / 2);
    let lzimn = (imid * mmax * (2 * nlat - mmax + 1)) / 2;
    ierror = 9;
    if wvhses.len() < lzimn + nlon + 15 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let mn = mmax * nlat * nt;
    let lwkmin = if isym != 0 {
        nlat * (2 * nt * nlon + (6 * imid).max(nlon)) + 2 * mn + nlat
    } else {
        imid * (2 * nt * nlon + (6 * nlat).max(nlon)) + 2 * mn + nlat
    };
    ierror = 10;
    if lwork < lwkmin {
        return Ok((Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let br = vec![0.0_f32; nlat * nlat * nt];
    let bi = vec![0.0_f32; nlat * nlat * nt];
    let mut cr = vec![0.0_f32; nlat * nlat * nt];
    let mut ci = vec![0.0_f32; nlat * nlat * nt];
    let mut pertrb = vec![0.0_f32; nt];
    let sqnn: Vec<f32> = (0..nlat)
        .map(|idx| {
            if idx == 0 {
                0.0
            } else {
                ((idx as f32) * (idx as f32 + 1.0)).sqrt()
            }
        })
        .collect();

    for k in 0..nt {
        pertrb[k] = a[k] / (2.0 * 2.0_f32.sqrt());
        for n in 2..=nlat {
            let idx = (n - 1) * nt + k;
            cr[idx] = a[idx] / sqnn[n - 1];
            ci[idx] = b[idx] / sqnn[n - 1];
        }
        for m in 2..=mmax {
            for n in m..=nlat {
                let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                cr[idx] = a[idx] / sqnn[n - 1];
                ci[idx] = b[idx] / sqnn[n - 1];
            }
        }
    }

    let lwork_vhses = if ityp <= 2 {
        nlat * (2 * nt * nlon + (6 * imid).max(nlon))
    } else {
        imid * (2 * nt * nlon + (6 * nlat).max(nlon))
    };
    let (v, w, ierr) = vhses_impl(
        &br,
        &bi,
        &cr,
        &ci,
        nlat,
        nlon,
        nt,
        ityp,
        wvhses,
        lwork_vhses,
    )?;
    Ok((v, w, pertrb, ierr))
}

#[pyfunction]
/// Rust entry point for `ivrtes`.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `wvhses`: Workspace initialized by `vhsesi_impl` for regular-grid vector synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn ivrtes<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    wvhses: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err("ivrtes expects rank-2 or rank-3 a/b"));
    }
    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let nlon = infer_nlon_from_wvhses(nlat, wvhses.len())?;
    let abuf = collect_logical_ab(a.as_array());
    let bbuf = collect_logical_ab(b.as_array());
    let (v, w, pertrb, ierror) = ivrtes_impl(&abuf, &bbuf, nlat, nt, 0, wvhses.as_slice()?, lwork)?;
    if ierror != 0 {
        let empty2 = ArrayD::<f32>::from_shape_vec(IxDyn(&[0, 0]), Vec::new())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let empty1 = ArrayD::<f32>::from_shape_vec(IxDyn(&[0]), Vec::new())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok((
            empty2.clone().into_pyarray(py).into_any().unbind(),
            empty2.into_pyarray(py).into_any().unbind(),
            empty1.into_pyarray(py).into_any().unbind(),
            ierror,
        ));
    }
    let shape = if ashape.len() == 2 {
        IxDyn(&[nlat, nlon])
    } else {
        IxDyn(&[nlat, nlon, nt])
    };
    let v_arr = ArrayD::from_shape_vec(shape.clone(), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let w_arr =
        ArrayD::from_shape_vec(shape, w).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let p_arr = ArrayD::from_shape_vec(IxDyn(&[nt]), pertrb)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        v_arr.into_pyarray(py).into_any().unbind(),
        w_arr.into_pyarray(py).into_any().unbind(),
        p_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}

#[pyfunction]
/// Rust entry point for `ivrtes_isym`.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `wvhses`: Workspace initialized by `vhsesi_impl` for regular-grid vector synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn ivrtes_isym<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    isym: usize,
    wvhses: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err(
            "ivrtes_isym expects rank-2 or rank-3 a/b",
        ));
    }
    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let nlon = infer_nlon_from_wvhses(nlat, wvhses.len())?;
    let idvw = if isym == 0 { nlat } else { nlat.div_ceil(2) };
    let abuf = collect_logical_ab(a.as_array());
    let bbuf = collect_logical_ab(b.as_array());
    let (v, w, pertrb, ierror) =
        ivrtes_impl(&abuf, &bbuf, nlat, nt, isym, wvhses.as_slice()?, lwork)?;
    if ierror != 0 {
        let empty2 = ArrayD::<f32>::from_shape_vec(IxDyn(&[0, 0]), Vec::new())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let empty1 = ArrayD::<f32>::from_shape_vec(IxDyn(&[0]), Vec::new())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok((
            empty2.clone().into_pyarray(py).into_any().unbind(),
            empty2.into_pyarray(py).into_any().unbind(),
            empty1.into_pyarray(py).into_any().unbind(),
            ierror,
        ));
    }
    let shape = if ashape.len() == 2 {
        IxDyn(&[idvw, nlon])
    } else {
        IxDyn(&[idvw, nlon, nt])
    };
    let v_arr = ArrayD::from_shape_vec(shape.clone(), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let w_arr =
        ArrayD::from_shape_vec(shape, w).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let p_arr = ArrayD::from_shape_vec(IxDyn(&[nt]), pertrb)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        v_arr.into_pyarray(py).into_any().unbind(),
        w_arr.into_pyarray(py).into_any().unbind(),
        p_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}
