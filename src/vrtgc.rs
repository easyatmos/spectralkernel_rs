use crate::shsgc::shsgc_impl;
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

fn infer_nlon_from_wshsgc(nlat: usize, ltotal: usize) -> PyResult<usize> {
    for cand in 4..=4 * nlat.max(4) {
        let l = ((cand + 2) / 2).min(nlat);
        let late = (nlat + (nlat % 2)) / 2;
        let need_i64 = (nlat as i64) * (2 * late as i64 + 3 * l as i64 - 2)
            + (3 * l as i64 * (1 - l as i64)) / 2
            + cand as i64
            + 15;
        if usize::try_from(need_i64).ok() == Some(ltotal) {
            return Ok(cand);
        }
    }
    Err(PyValueError::new_err(
        "failed to infer nlon from wshsgc length",
    ))
}

/// Core Rust implementation of `vrtgc`.
///
/// # Parameters
/// - `cr`: Third vector coefficient family in vector layout.
/// - `ci`: Fourth vector coefficient family in vector layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `nt`: Number of stacked fields processed together.
/// - `wshsgc`: Workspace initialized by `shsgci_impl` for Gaussian-grid scalar synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vrtgc_impl(
    cr: &[f32],
    ci: &[f32],
    nlat: usize,
    isym: usize,
    nt: usize,
    wshsgc: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), ierror));
    }

    let nlon = infer_nlon_from_wshsgc(nlat, wshsgc.len())?;

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
    if cr.len() != ci.len() || cr.len() != nlat * nlat * nt {
        return Err(PyValueError::new_err("cr/ci size mismatch"));
    }

    let imid = nlat.div_ceil(2);
    let ls = if isym == 0 { nlat } else { imid };
    let l1 = nlat.min((nlon + 2) / 2);
    let l2 = imid;
    let lwkmin = if isym == 0 {
        nlat * (nt * nlon + (3 * l2).max(nlon) + 2 * nt * l1 + 1)
    } else {
        l2 * (nt * nlon + (3 * nlat).max(nlon)) + nlat * (2 * nt * l1 + 1)
    };
    ierror = 10;
    if lwork < lwkmin {
        return Ok((Vec::new(), ierror));
    }

    let mut a = vec![0.0_f32; nlat * nlat * nt];
    let mut b = vec![0.0_f32; nlat * nlat * nt];
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
            a[idx] = sqnn[n - 1] * cr[idx];
            b[idx] = sqnn[n - 1] * ci[idx];
        }
        for m in 2..=mmax {
            for n in m..=nlat {
                let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                a[idx] = sqnn[n - 1] * cr[idx];
                b[idx] = sqnn[n - 1] * ci[idx];
            }
        }
    }

    let lwork_shsgc = if isym == 0 {
        nlat * nlon * (nt + 1)
    } else {
        ls * nlon * nt + nlat * nlon
    };

    shsgc_impl(&a, &b, nlat, nt, wshsgc, lwork_shsgc)
}

#[pyfunction]
/// Rust entry point for `vrtgc`.
///
/// # Parameters
/// - `nlon`: Number of longitudes in the grid.
/// - `cr`: Third vector coefficient family in vector layout.
/// - `ci`: Fourth vector coefficient family in vector layout.
/// - `wshsgc`: Workspace initialized by `shsgci_impl` for Gaussian-grid scalar synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
/// - `isym`: Symmetry selector used by Legendre tables.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn vrtgc<'py>(
    py: Python<'py>,
    nlon: usize,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    wshsgc: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
    isym: Option<usize>,
) -> PyResult<(Py<PyAny>, i32)> {
    let crshape = cr.shape().to_vec();
    let cishape = ci.shape().to_vec();
    if crshape != cishape {
        return Err(PyValueError::new_err(
            "cr and ci must have identical shapes",
        ));
    }
    if crshape.len() != 2 && crshape.len() != 3 {
        return Err(PyValueError::new_err(
            "vrtgc expects rank-2 or rank-3 cr/ci",
        ));
    }

    let nlat = crshape[0];
    let inferred_nlon = infer_nlon_from_wshsgc(nlat, wshsgc.len())?;
    if nlon != inferred_nlon {
        return Err(PyValueError::new_err(format!(
            "nlon mismatch: got {nlon}, inferred {inferred_nlon} from wshsgc"
        )));
    }
    let isym = isym.unwrap_or(0);
    let nt = if crshape.len() == 2 { 1 } else { crshape[2] };
    let crbuf = collect_logical_ab(cr.as_array());
    let cibuf = collect_logical_ab(ci.as_array());
    let (vort, ierror) = vrtgc_impl(&crbuf, &cibuf, nlat, isym, nt, wshsgc.as_slice()?, lwork)?;
    let nlat_out = if isym == 0 { nlat } else { nlat.div_ceil(2) };
    let shape = if crshape.len() == 2 {
        IxDyn(&[nlat_out, nlon])
    } else {
        IxDyn(&[nlat_out, nlon, nt])
    };
    let vort_arr =
        ArrayD::from_shape_vec(shape, vort).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((vort_arr.into_pyarray(py).into_any().unbind(), ierror))
}
