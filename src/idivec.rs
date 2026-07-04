use crate::vhsec::vhsec_impl;
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

fn ityp_from_isym(isym: usize) -> PyResult<usize> {
    match isym {
        0 => Ok(1),
        1 => Ok(4),
        2 => Ok(7),
        _ => Err(PyValueError::new_err("isym must be 0, 1 or 2")),
    }
}

/// Core Rust implementation of `idivec`.
///
/// # Parameters
/// - `nlon`: Number of longitudes in the grid.
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `wvhsec`: Workspace initialized by `vhseci_impl` for regular-grid vector synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn idivec_impl(
    nlon: usize,
    a: &[f32],
    b: &[f32],
    nlat: usize,
    nt: usize,
    isym: usize,
    wvhsec: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), ierror));
    }
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

    let mmax = nlat.min((nlon + 1) / 2);
    let imid = nlat.div_ceil(2);
    let l1 = nlat.min((nlon + 1) / 2);
    let l2 = imid;
    let lwmin = 4 * nlat * l2 + 3 * l1.saturating_sub(2) * (2 * nlat - l1 - 1) + nlon + 15;
    ierror = 9;
    if wvhsec.len() < lwmin {
        return Ok((Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let mn = mmax * nlat * nt;
    let lwkmin = if isym == 0 {
        imid * (2 * nt * nlon + (6 * nlat).max(nlon)) + 2 * mn + nlat
    } else {
        nlat * (2 * nt * nlon + (6 * imid).max(nlon)) + 2 * mn + nlat
    };
    ierror = 10;
    if lwork < lwkmin {
        return Ok((Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let mut br = vec![0.0_f32; nlat * nlat * nt];
    let mut bi = vec![0.0_f32; nlat * nlat * nt];
    let cr = vec![0.0_f32; nlat * nlat * nt];
    let ci = vec![0.0_f32; nlat * nlat * nt];
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
            br[idx] = -a[idx] / sqnn[n - 1];
            bi[idx] = -b[idx] / sqnn[n - 1];
        }
        for m in 2..=mmax {
            for n in m..=nlat {
                let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                br[idx] = -a[idx] / sqnn[n - 1];
                bi[idx] = -b[idx] / sqnn[n - 1];
            }
        }
    }

    let lwork_vhsec = if ityp <= 2 {
        nlat * (2 * nt * nlon + (6 * imid).max(nlon))
    } else {
        imid * (2 * nt * nlon + (6 * nlat).max(nlon))
    };
    let (v, w, ierr) = vhsec_impl(&br, &bi, &cr, &ci, nlat, nt, ityp, wvhsec, lwork_vhsec)?;
    Ok((v, w, pertrb, ierr))
}

#[pyfunction]
/// Rust entry point for `idivec`.
///
/// # Parameters
/// - `nlon`: Number of longitudes in the grid.
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `wvhsec`: Workspace initialized by `vhseci_impl` for regular-grid vector synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn idivec<'py>(
    py: Python<'py>,
    nlon: usize,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    wvhsec: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err("idivec expects rank-2 or rank-3 a/b"));
    }
    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let abuf = collect_logical_ab(a.as_array());
    let bbuf = collect_logical_ab(b.as_array());
    let (v, w, pertrb, ierror) =
        idivec_impl(nlon, &abuf, &bbuf, nlat, nt, 0, wvhsec.as_slice()?, lwork)?;
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
/// Rust entry point for `idivec_isym`.
///
/// # Parameters
/// - `nlon`: Number of longitudes in the grid.
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `wvhsec`: Workspace initialized by `vhseci_impl` for regular-grid vector synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn idivec_isym<'py>(
    py: Python<'py>,
    nlon: usize,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    isym: usize,
    wvhsec: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err(
            "idivec_isym expects rank-2 or rank-3 a/b",
        ));
    }
    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let idvw = if isym == 0 { nlat } else { nlat.div_ceil(2) };
    let abuf = collect_logical_ab(a.as_array());
    let bbuf = collect_logical_ab(b.as_array());
    let (v, w, pertrb, ierror) = idivec_impl(
        nlon,
        &abuf,
        &bbuf,
        nlat,
        nt,
        isym,
        wvhsec.as_slice()?,
        lwork,
    )?;
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
