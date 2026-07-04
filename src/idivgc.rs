use crate::vhsgc::vhsgc_impl;
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

fn infer_nlon_from_wvhsgc(nlat: usize, ltotal: usize) -> PyResult<usize> {
    let imid = nlat.div_ceil(2);
    let lzz1 = 2 * nlat * imid;
    for cand in 4..=4 * nlat.max(4) {
        let mmax = nlat.min((cand + 1) / 2);
        let labc = 3 * mmax.saturating_sub(2) * (2 * nlat - mmax - 1) / 2;
        let need = 2 * (lzz1 + labc) + cand + 15;
        if need == ltotal {
            return Ok(cand);
        }
    }
    Err(PyValueError::new_err(
        "failed to infer nlon from wvhsgc length",
    ))
}

fn ityp_from_isym(isym: usize) -> PyResult<usize> {
    match isym {
        0 => Ok(1),
        1 => Ok(4),
        2 => Ok(7),
        _ => Err(PyValueError::new_err("isym must be 0, 1 or 2")),
    }
}

/// Core Rust implementation of `idivgc`.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `wvhsgc`: Workspace initialized by `vhsgci_impl` for Gaussian-grid vector synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn idivgc_impl(
    a: &[f32],
    b: &[f32],
    nlat: usize,
    nt: usize,
    isym: usize,
    wvhsgc: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let nlon = infer_nlon_from_wvhsgc(nlat, wvhsgc.len())?;
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
    let lwkmin = if isym == 0 {
        imid * (2 * nt * nlon + (6 * nlat).max(nlon)) + 2 * (mmax * nlat * nt) + nlat
    } else {
        nlat * (2 * nt * nlon + (6 * imid).max(nlon)) + 2 * (mmax * nlat * nt) + nlat
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
                let fnn = idx as f32;
                (fnn * (fnn + 1.0)).sqrt()
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

    let lwork_vhsgc = if ityp <= 2 {
        nlat * (2 * nt * nlon + (6 * imid).max(nlon))
    } else {
        imid * (2 * nt * nlon + (6 * nlat).max(nlon))
    };
    let (v, w, _idv, _nlon, ierr) =
        vhsgc_impl(&br, &bi, &cr, &ci, nlat, nt, ityp, wvhsgc, lwork_vhsgc)?;
    Ok((v, w, pertrb, ierr))
}

#[pyfunction]
/// Rust entry point for `idivgc`.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `wvhsgc`: Workspace initialized by `vhsgci_impl` for Gaussian-grid vector synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn idivgc<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    wvhsgc: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err("idivgc expects rank-2 or rank-3 a/b"));
    }

    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let nlon = infer_nlon_from_wvhsgc(nlat, wvhsgc.len())?;
    let abuf = collect_logical_ab(a.as_array());
    let bbuf = collect_logical_ab(b.as_array());
    let (v, w, pertrb, ierror) = idivgc_impl(&abuf, &bbuf, nlat, nt, 0, wvhsgc.as_slice()?, lwork)?;
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
/// Rust entry point for `idivgc_isym`.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `wvhsgc`: Workspace initialized by `vhsgci_impl` for Gaussian-grid vector synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn idivgc_isym<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    isym: usize,
    wvhsgc: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err(
            "idivgc_isym expects rank-2 or rank-3 a/b",
        ));
    }

    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let nlon = infer_nlon_from_wvhsgc(nlat, wvhsgc.len())?;
    let idvw = if isym == 0 { nlat } else { nlat.div_ceil(2) };
    let abuf = collect_logical_ab(a.as_array());
    let bbuf = collect_logical_ab(b.as_array());
    let (v, w, pertrb, ierror) =
        idivgc_impl(&abuf, &bbuf, nlat, nt, isym, wvhsgc.as_slice()?, lwork)?;
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
