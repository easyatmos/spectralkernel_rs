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

fn trace_enabled() -> bool {
    std::env::var("GRADGC_TRACE")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
}

fn infer_nlon_from_wvhsgc(nlat: usize, ltotal: usize) -> PyResult<usize> {
    let imid = (nlat + 1) / 2;
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
        0 => Ok(0),
        1 => Ok(4),
        2 => Ok(7),
        _ => Err(PyValueError::new_err("isym must be 0, 1 or 2")),
    }
}

/// Core Rust implementation of `gradgc`.
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
pub fn gradgc_impl(
    a: &[f32],
    b: &[f32],
    nlat: usize,
    nt: usize,
    isym: usize,
    wvhsgc: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, usize, usize, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), 0, 0, ierror));
    }

    let nlon = infer_nlon_from_wvhsgc(nlat, wvhsgc.len())?;
    let ityp = ityp_from_isym(isym)?;

    ierror = 2;
    if nlon < 4 {
        return Ok((Vec::new(), Vec::new(), 0, 0, ierror));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), Vec::new(), 0, 0, ierror));
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
            let n0 = n - 1;
            let idx = n0 * nt + k;
            br[idx] = sqnn[n0] * a[idx];
            bi[idx] = sqnn[n0] * b[idx];
        }
        for m in 2..=mmax {
            for n in m..=nlat {
                let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                br[idx] = sqnn[n - 1] * a[idx];
                bi[idx] = sqnn[n - 1] * b[idx];
            }
        }
    }

    if trace_enabled() {
        let take = br.len().min(12);
        eprintln!(
            "[gradgc_impl] nlat={} nlon={} nt={} mmax={} | a0={:?} | br0={:?} | bi0={:?}",
            nlat,
            nlon,
            nt,
            mmax,
            &a[..take],
            &br[..take],
            &bi[..take]
        );
    }

    let (v, w, idv, nlon2, ierr) = vhsgc_impl(&br, &bi, &cr, &ci, nlat, nt, ityp, wvhsgc, lwork)?;
    Ok((v, w, idv, nlon2, ierr))
}

#[pyfunction]
/// Rust entry point for `gradgc`.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `wvhsgc`: Workspace initialized by `vhsgci_impl` for Gaussian-grid vector synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn gradgc<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    wvhsgc: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err("gradgc expects rank-2 or rank-3 a/b"));
    }

    if trace_enabled() {
        eprintln!(
            "[gradgc.py] ashape={:?} bshape={:?} astrides={:?} bstrides={:?} a_c={} a_f={} b_c={} b_f={}",
            ashape,
            bshape,
            a.as_array().strides(),
            b.as_array().strides(),
            a.is_c_contiguous(),
            a.is_fortran_contiguous(),
            b.is_c_contiguous(),
            b.is_fortran_contiguous(),
        );
    }

    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let abuf = collect_logical_ab(a.as_array());
    let bbuf = collect_logical_ab(b.as_array());
    let (v, w, _idv, nlon, ierror) =
        gradgc_impl(&abuf, &bbuf, nlat, nt, 0, wvhsgc.as_slice()?, lwork)?;

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
/// Rust entry point for `gradgc_isym`.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `wvhsgc`: Workspace initialized by `vhsgci_impl` for Gaussian-grid vector synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn gradgc_isym<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    isym: usize,
    wvhsgc: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err(
            "gradgc_isym expects rank-2 or rank-3 a/b",
        ));
    }

    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let abuf = collect_logical_ab(a.as_array());
    let bbuf = collect_logical_ab(b.as_array());
    let (v, w, idv, nlon, ierror) =
        gradgc_impl(&abuf, &bbuf, nlat, nt, isym, wvhsgc.as_slice()?, lwork)?;

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
