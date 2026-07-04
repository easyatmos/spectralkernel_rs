use crate::shsgs::shsgs_impl;
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

fn infer_nlon_from_wshsgs(nlat: usize, ltotal: usize) -> PyResult<usize> {
    for nlon in 4..=4 * nlat.max(4) {
        let l = ((nlon + 2) / 2).min(nlat);
        let late = (nlat + (nlat % 2)) / 2;
        let labc = l * (l - 1) / 2 + (nlat - l) * (l - 1);
        let prefix = nlat + 2 * nlat * late + 3 * labc + nlon + 15;
        let pmnf_count = l * (2 * nlat - l + 1) / 2;
        if prefix + late * pmnf_count == ltotal {
            return Ok(nlon);
        }
    }
    Err(PyValueError::new_err(
        "failed to infer nlon from wshsgs length",
    ))
}

/// Core Rust implementation of `igradgs`.
///
/// # Parameters
/// - `br`: First vector coefficient family in vector layout.
/// - `bi`: Second vector coefficient family in vector layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `wshsgs`: Workspace initialized by `shsgsi_impl` for Gaussian-grid scalar synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn igradgs_impl(
    br: &[f32],
    bi: &[f32],
    nlat: usize,
    nt: usize,
    wshsgs: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), ierror));
    }

    let nlon = infer_nlon_from_wshsgs(nlat, wshsgs.len())?;

    ierror = 2;
    if nlon < 4 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), ierror));
    }
    if br.len() != bi.len() || br.len() != nlat * nlat * nt {
        return Err(PyValueError::new_err("br/bi size mismatch"));
    }

    let mut a = vec![0.0_f32; nlat * nlat * nt];
    let mut b = vec![0.0_f32; nlat * nlat * nt];
    let sqnn: Vec<f32> = (0..nlat)
        .map(|idx| {
            if idx == 0 {
                0.0
            } else {
                let fnn = idx as f32;
                1.0 / (fnn * (fnn + 1.0)).sqrt()
            }
        })
        .collect();

    let mmax = nlat.min((nlon + 1) / 2);
    for k in 0..nt {
        for n in 2..=nlat {
            let idx = (n - 1) * nt + k;
            a[idx] = br[idx] * sqnn[n - 1];
            b[idx] = bi[idx] * sqnn[n - 1];
        }
        for m in 2..=mmax {
            for n in m..=nlat {
                let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                a[idx] = br[idx] * sqnn[n - 1];
                b[idx] = bi[idx] * sqnn[n - 1];
            }
        }
    }

    shsgs_impl(&a, &b, nlat, nt, wshsgs, lwork)
}

#[pyfunction]
/// Rust entry point for `igradgs`.
///
/// # Parameters
/// - `br`: First vector coefficient family in vector layout.
/// - `bi`: Second vector coefficient family in vector layout.
/// - `wshsgs`: Workspace initialized by `shsgsi_impl` for Gaussian-grid scalar synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn igradgs<'py>(
    py: Python<'py>,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    wshsgs: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, i32)> {
    let brshape = br.shape().to_vec();
    let bishape = bi.shape().to_vec();
    if brshape != bishape {
        return Err(PyValueError::new_err(
            "br and bi must have identical shapes",
        ));
    }
    if brshape.len() != 2 && brshape.len() != 3 {
        return Err(PyValueError::new_err(
            "igradgs expects rank-2 or rank-3 br/bi",
        ));
    }

    let nlat = brshape[0];
    let nt = if brshape.len() == 2 { 1 } else { brshape[2] };
    let brbuf = collect_logical_ab(br.as_array());
    let bibuf = collect_logical_ab(bi.as_array());
    let (sf, ierror) = igradgs_impl(&brbuf, &bibuf, nlat, nt, wshsgs.as_slice()?, lwork)?;

    let nlon = infer_nlon_from_wshsgs(nlat, wshsgs.len())?;
    let shape = if brshape.len() == 2 {
        IxDyn(&[nlat, nlon])
    } else {
        IxDyn(&[nlat, nlon, nt])
    };
    let sf_arr =
        ArrayD::from_shape_vec(shape, sf).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((sf_arr.into_pyarray(py).into_any().unbind(), ierror))
}
