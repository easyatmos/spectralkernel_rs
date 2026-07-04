use crate::vhsgs::vhsgs_impl;
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

fn infer_nlon_from_wvhsgs(nlat: usize, ltotal: usize) -> PyResult<usize> {
    let imid = nlat.div_ceil(2);
    let lmn = nlat * (nlat + 1) / 2;
    for nlon in 4..=4 * nlat.max(4) {
        let mmax = nlat.min((nlon + 1) / 2);
        let idz = (mmax * (2 * nlat - mmax + 1)) / 2;
        let lzimn = imid * idz;
        let need = lzimn + lzimn + nlon + 15 + 2 * nlat;
        if need == ltotal {
            return Ok(nlon);
        }
        let alt = imid * lmn * 2 + nlon + 15;
        if alt == ltotal {
            return Ok(nlon);
        }
    }
    Err(PyValueError::new_err(
        "failed to infer nlon from wvhsgs length",
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

/// Core Rust implementation of `gradgs`.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `wvhsgs`: Workspace initialized by `vhsgsi_impl` for Gaussian-grid vector synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn gradgs_impl(
    a: &[f32],
    b: &[f32],
    nlat: usize,
    nt: usize,
    isym: usize,
    wvhsgs: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, usize, usize, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), 0, 0, ierror));
    }

    let nlon = infer_nlon_from_wvhsgs(nlat, wvhsgs.len())?;

    ierror = 2;
    if nlon < 4 {
        return Ok((Vec::new(), Vec::new(), 0, 0, ierror));
    }
    let ityp = ityp_from_isym(isym)?;
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), Vec::new(), 0, 0, ierror));
    }
    if a.len() != b.len() || a.len() != nlat * nlat * nt {
        return Err(PyValueError::new_err("a/b size mismatch"));
    }

    let mmax = nlat.min((nlon + 1) / 2);
    let imid = nlat.div_ceil(2);
    let idv = if isym == 0 { nlat } else { imid };
    let mn = mmax * nlat * nt;
    let lwkmin = (2 * nt + 1) * idv * nlon + 2 * mn + nlat;
    ierror = 10;
    if lwork < lwkmin {
        return Ok((Vec::new(), Vec::new(), 0, 0, ierror));
    }

    let mut br = vec![0.0_f32; nlat * nlat * nt];
    let mut bi = vec![0.0_f32; nlat * nlat * nt];
    let cr = vec![0.0_f32; nlat * nlat * nt];
    let ci = vec![0.0_f32; nlat * nlat * nt];
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
        for n in 2..=nlat {
            let idx = (n - 1) * nt + k;
            br[idx] = sqnn[n - 1] * a[idx];
            bi[idx] = sqnn[n - 1] * b[idx];
        }
        for m in 2..=mmax {
            for n in m..=nlat {
                let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                br[idx] = sqnn[n - 1] * a[idx];
                bi[idx] = sqnn[n - 1] * b[idx];
            }
        }
    }

    vhsgs_impl(&br, &bi, &cr, &ci, nlat, nt, ityp, wvhsgs, lwork)
}

#[pyfunction]
/// Rust entry point for `gradgs`.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `wvhsgs`: Workspace initialized by `vhsgsi_impl` for Gaussian-grid vector synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn gradgs<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    wvhsgs: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err("gradgs expects rank-2 or rank-3 a/b"));
    }

    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let abuf = collect_logical_ab(a.as_array());
    let bbuf = collect_logical_ab(b.as_array());
    let (v, w, _idv, nlon, ierror) =
        gradgs_impl(&abuf, &bbuf, nlat, nt, 0, wvhsgs.as_slice()?, lwork)?;

    let shape = if ashape.len() == 2 {
        IxDyn(&[nlat, nlon])
    } else {
        IxDyn(&[nlat, nlon, nt])
    };
    let v_arr = ArrayD::from_shape_vec(shape.clone(), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let w_arr =
        ArrayD::from_shape_vec(shape, w).map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((
        v_arr.into_pyarray(py).into_any().unbind(),
        w_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}

#[pyfunction]
/// Rust entry point for `gradgs_isym`.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `wvhsgs`: Workspace initialized by `vhsgsi_impl` for Gaussian-grid vector synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn gradgs_isym<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    isym: usize,
    wvhsgs: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err(
            "gradgs_isym expects rank-2 or rank-3 a/b",
        ));
    }

    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let abuf = collect_logical_ab(a.as_array());
    let bbuf = collect_logical_ab(b.as_array());
    let (v, w, idv, nlon, ierror) =
        gradgs_impl(&abuf, &bbuf, nlat, nt, isym, wvhsgs.as_slice()?, lwork)?;

    let shape = if ashape.len() == 2 {
        IxDyn(&[idv, nlon])
    } else {
        IxDyn(&[idv, nlon, nt])
    };
    let v_arr = ArrayD::from_shape_vec(shape.clone(), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let w_arr =
        ArrayD::from_shape_vec(shape, w).map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((
        v_arr.into_pyarray(py).into_any().unbind(),
        w_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}
