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
    let imid = (nlat + 1) / 2;
    for cand in 4..=4 * nlat.max(4) {
        let mmax = nlat.min((cand + 1) / 2);
        let idz = (mmax * (2 * nlat - mmax + 1)) / 2;
        let lzimn = idz * imid;
        let need = lzimn + lzimn + cand + 15;
        if need == ltotal {
            return Ok(cand);
        }
        let mmax_even = nlat.min((cand + 2) / 2);
        let idz_even = (mmax_even * (2 * nlat - mmax_even + 1)) / 2;
        let lzimn_even = idz_even * imid;
        let need_even = lzimn_even + lzimn_even + cand + 15;
        if need_even == ltotal {
            return Ok(cand);
        }
    }
    Err(PyValueError::new_err(
        "failed to infer nlon from wvhses length",
    ))
}

fn crop_half_sphere_output(
    data: Vec<f32>,
    nlat: usize,
    nlon: usize,
    nt: usize,
    isym: usize,
) -> Vec<f32> {
    if isym == 0 {
        return data;
    }
    let idvw = (nlat + 1) / 2;
    let mut out = vec![0.0_f32; idvw * nlon * nt];
    for i in 0..idvw {
        for j in 0..nlon {
            for k in 0..nt {
                out[(i * nlon + j) * nt + k] = data[(i * nlon + j) * nt + k];
            }
        }
    }
    out
}

fn ityp_from_isym(isym: usize) -> PyResult<usize> {
    match isym {
        0 => Ok(0),
        1 => Ok(4),
        2 => Ok(7),
        _ => Err(PyValueError::new_err("isym must be 0, 1 or 2")),
    }
}

/// Core Rust implementation of `grades`.
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
/// A Python result containing the cosine coefficients, sine coefficients, and an error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn grades_impl(
    a: &[f32],
    b: &[f32],
    nlat: usize,
    nt: usize,
    isym: usize,
    wvhses: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let nlon = infer_nlon_from_wvhses(nlat, wvhses.len())?;
    let ityp = ityp_from_isym(isym)?;

    ierror = 2;
    if nlon < 4 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    if a.len() != b.len() || a.len() != nlat * nlat * nt {
        return Err(PyValueError::new_err("a/b size mismatch"));
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

    let mmax = nlat.min((nlon + 1) / 2);
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

    let (v, w, ierror) = vhses_impl(&br, &bi, &cr, &ci, nlat, nlon, nt, ityp, wvhses, lwork)?;
    Ok((
        crop_half_sphere_output(v, nlat, nlon, nt, isym),
        crop_half_sphere_output(w, nlat, nlon, nt, isym),
        ierror,
    ))
}

#[pyfunction]
/// Rust entry point for `grades`.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `wvhses`: Workspace initialized by `vhsesi_impl` for regular-grid vector synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn grades<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    wvhses: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err("grades expects rank-2 or rank-3 a/b"));
    }

    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let abuf = collect_logical_ab(a.as_array());
    let bbuf = collect_logical_ab(b.as_array());
    let (v, w, ierror) = grades_impl(&abuf, &bbuf, nlat, nt, 0, wvhses.as_slice()?, lwork)?;
    let nlon = infer_nlon_from_wvhses(nlat, wvhses.len())?;
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
/// Rust entry point for `grades_isym`.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `wvhses`: Workspace initialized by `vhsesi_impl` for regular-grid vector synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn grades_isym<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    isym: usize,
    wvhses: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err(
            "grades_isym expects rank-2 or rank-3 a/b",
        ));
    }

    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let nlon = infer_nlon_from_wvhses(nlat, wvhses.len())?;
    let idvw = if isym == 0 { nlat } else { (nlat + 1) / 2 };
    let abuf = collect_logical_ab(a.as_array());
    let bbuf = collect_logical_ab(b.as_array());
    let (v, w, ierror) = grades_impl(&abuf, &bbuf, nlat, nt, isym, wvhses.as_slice()?, lwork)?;
    let shape = if ashape.len() == 2 {
        IxDyn(&[idvw, nlon])
    } else {
        IxDyn(&[idvw, nlon, nt])
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
