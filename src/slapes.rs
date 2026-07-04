use crate::shses::shses_impl;
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

/// Core Rust implementation of `slapes`.
///
/// # Parameters
/// - `nlon`: Number of longitudes in the grid.
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `nt`: Number of stacked fields processed together.
/// - `wshses`: Workspace initialized by `shsesi_impl` for regular-grid scalar synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn slapes_impl(
    nlon: usize,
    a: &[f32],
    b: &[f32],
    nlat: usize,
    isym: usize,
    nt: usize,
    wshses: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 2;
    if nlon < 4 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 3;
    if isym > 2 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), ierror));
    }
    if a.len() != b.len() || a.len() != nlat * nlat * nt {
        return Err(PyValueError::new_err("a/b size mismatch"));
    }

    let imid = nlat.div_ceil(2);
    let mmax = nlat.min(nlon / 2 + 1);
    let lpimn = (imid * mmax * (2 * nlat - mmax + 1)) / 2;
    ierror = 9;
    if wshses.len() < lpimn + nlon + 15 {
        return Ok((Vec::new(), ierror));
    }

    let l1 = nlat.min(nlon / 2 + 1);
    let lwkmin = if isym == 0 {
        (nt + 1) * nlat * nlon + nlat * (2 * nt * l1 + 1)
    } else {
        (nt + 1) * imid * nlon + nlat * (2 * nt * l1 + 1)
    };
    ierror = 10;
    if lwork < lwkmin {
        return Ok((Vec::new(), ierror));
    }

    let mut alap = vec![0.0_f32; nlat * nlat * nt];
    let mut blap = vec![0.0_f32; nlat * nlat * nt];
    let fnn: Vec<f32> = (0..nlat)
        .map(|idx| {
            if idx == 0 {
                0.0
            } else {
                let f = idx as f32;
                f * (f + 1.0)
            }
        })
        .collect();

    for k in 0..nt {
        for n in 2..=nlat {
            let idx = (n - 1) * nt + k;
            alap[idx] = -fnn[n - 1] * a[idx];
            blap[idx] = -fnn[n - 1] * b[idx];
        }
        for m in 2..=mmax {
            for n in m..=nlat {
                let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                alap[idx] = -fnn[n - 1] * a[idx];
                blap[idx] = -fnn[n - 1] * b[idx];
            }
        }
    }

    let lwork_shses = lwork.saturating_sub(2 * (mmax * nlat * nt) + nlat);
    shses_impl(&alap, &blap, nlat, nt, wshses, lwork_shses)
}

#[pyfunction]
/// Rust entry point for `slapes`.
///
/// # Parameters
/// - `nlon`: Number of longitudes in the grid.
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `wshses`: Workspace initialized by `shsesi_impl` for regular-grid scalar synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
/// - `isym`: Symmetry selector used by Legendre tables.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn slapes<'py>(
    py: Python<'py>,
    nlon: usize,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    wshses: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
    isym: Option<usize>,
) -> PyResult<(Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err("slapes expects rank-2 or rank-3 a/b"));
    }

    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let isym = isym.unwrap_or(0);
    let abuf = collect_logical_ab(a.as_array());
    let bbuf = collect_logical_ab(b.as_array());
    let (slap, ierror) = slapes_impl(
        nlon,
        &abuf,
        &bbuf,
        nlat,
        isym,
        nt,
        wshses.as_slice()?,
        lwork,
    )?;
    if ierror != 0 {
        let empty2 = ArrayD::<f32>::from_shape_vec(IxDyn(&[0, 0]), Vec::new())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok((empty2.into_pyarray(py).into_any().unbind(), ierror));
    }
    let ids = if isym == 0 { nlat } else { nlat.div_ceil(2) };
    let shape = if ashape.len() == 2 {
        IxDyn(&[ids, nlon])
    } else {
        IxDyn(&[ids, nlon, nt])
    };
    let slap_arr =
        ArrayD::from_shape_vec(shape, slap).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((slap_arr.into_pyarray(py).into_any().unbind(), ierror))
}
