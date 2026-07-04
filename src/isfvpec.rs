use crate::vhsec::vhsec_impl;
use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn collect_logical_coeffs(view: ndarray::ArrayViewD<'_, f32>) -> Vec<f32> {
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

fn isym_to_ityp(isym: usize) -> usize {
    match isym {
        0 => 0,
        1 => 3,
        2 => 6,
        _ => unreachable!(),
    }
}

fn expand_isfvpec_output(
    data: Vec<f32>,
    nlat: usize,
    nlon: usize,
    nt: usize,
    isym: usize,
) -> Vec<f32> {
    let idvw = if isym == 0 { nlat } else { nlat.div_ceil(2) };
    if idvw == nlat {
        return data;
    }
    let mut full = vec![0.0_f32; nlat * nlon * nt];
    for i in 0..idvw {
        for j in 0..nlon {
            for k in 0..nt {
                full[(i * nlon + j) * nt + k] = data[(i * nlon + j) * nt + k];
            }
        }
    }
    full
}

/// Core Rust implementation of `isfvpec`.
///
/// # Parameters
/// - `nlon`: Number of longitudes in the grid.
/// - `as_`: Parameter `as_` passed through to the routine.
/// - `bs`: Parameter `bs` passed through to the routine.
/// - `av`: Parameter `av` passed through to the routine.
/// - `bv`: Parameter `bv` passed through to the routine.
/// - `nlat`: Number of latitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `wvhsec`: Workspace initialized by `vhseci_impl` for regular-grid vector synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the cosine coefficients, sine coefficients, and an error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn isfvpec_impl(
    nlon: usize,
    as_: &[f32],
    bs: &[f32],
    av: &[f32],
    bv: &[f32],
    nlat: usize,
    nt: usize,
    isym: usize,
    wvhsec: &[f32],
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

    let expected = nlat * nlat * nt;
    if as_.len() != expected || bs.len() != expected || av.len() != expected || bv.len() != expected
    {
        return Err(PyValueError::new_err("as/bs/av/bv size mismatch"));
    }

    let l2 = nlat.div_ceil(2);
    let l1 = nlat.min((nlon + 1) / 2);
    let lzz1 = 2 * nlat * l2;
    let labc = 3 * l1.saturating_sub(2) * (2 * nlat - l1 - 1) / 2;
    let need_lvhsec = 2 * (lzz1 + labc) + nlon + 15;
    ierror = 9;
    if wvhsec.len() < need_lvhsec {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let mmax = nlat.min((nlon + 1) / 2);
    let mn = mmax * nlat * nt;
    let lwkmin = if isym == 0 {
        nlat * (2 * nt * nlon + (6 * l2).max(nlon) + 4 * l1 * nt + 1)
    } else {
        l2 * (2 * nt * nlon + (6 * nlat).max(nlon)) + nlat * (4 * l1 * nt + 1)
    };
    ierror = 10;
    if lwork < lwkmin {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let mut br = vec![0.0_f32; expected];
    let mut bi = vec![0.0_f32; expected];
    let mut cr = vec![0.0_f32; expected];
    let mut ci = vec![0.0_f32; expected];

    let fnn: Vec<f32> = (0..nlat)
        .map(|idx| {
            if idx == 0 {
                0.0
            } else {
                let n = idx as f32;
                -(n * (n + 1.0)).sqrt()
            }
        })
        .collect();

    for k in 0..nt {
        for n in 2..=nlat {
            let idx = (n - 1) * nt + k;
            br[idx] = -fnn[n - 1] * av[idx];
            bi[idx] = -fnn[n - 1] * bv[idx];
            cr[idx] = fnn[n - 1] * as_[idx];
            ci[idx] = fnn[n - 1] * bs[idx];
        }
        for m in 2..=mmax {
            for n in m..=nlat {
                let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                br[idx] = -fnn[n - 1] * av[idx];
                bi[idx] = -fnn[n - 1] * bv[idx];
                cr[idx] = fnn[n - 1] * as_[idx];
                ci[idx] = fnn[n - 1] * bs[idx];
            }
        }
    }

    let ityp = isym_to_ityp(isym);
    let lwork_vhsec = lwork.saturating_sub(4 * mn + nlat);
    let (v, w, ierr) = vhsec_impl(&br, &bi, &cr, &ci, nlat, nt, ityp, wvhsec, lwork_vhsec)?;
    Ok((v, w, ierr))
}

#[pyfunction]
/// Rust entry point for `isfvpec`.
///
/// # Parameters
/// - `nlon`: Number of longitudes in the grid.
/// - `as_`: Parameter `as_` passed through to the routine.
/// - `bs`: Parameter `bs` passed through to the routine.
/// - `av`: Parameter `av` passed through to the routine.
/// - `bv`: Parameter `bv` passed through to the routine.
/// - `wvhsec`: Workspace initialized by `vhseci_impl` for regular-grid vector synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn isfvpec<'py>(
    py: Python<'py>,
    nlon: usize,
    as_: PyReadonlyArrayDyn<'py, f32>,
    bs: PyReadonlyArrayDyn<'py, f32>,
    av: PyReadonlyArrayDyn<'py, f32>,
    bv: PyReadonlyArrayDyn<'py, f32>,
    wvhsec: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    isfvpec_isym(py, nlon, 0, as_, bs, av, bv, wvhsec, lwork)
}

#[pyfunction]
/// Rust entry point for `isfvpec_isym`.
///
/// # Parameters
/// - `nlon`: Number of longitudes in the grid.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `as_`: Parameter `as_` passed through to the routine.
/// - `bs`: Parameter `bs` passed through to the routine.
/// - `av`: Parameter `av` passed through to the routine.
/// - `bv`: Parameter `bv` passed through to the routine.
/// - `wvhsec`: Workspace initialized by `vhseci_impl` for regular-grid vector synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn isfvpec_isym<'py>(
    py: Python<'py>,
    nlon: usize,
    isym: usize,
    as_: PyReadonlyArrayDyn<'py, f32>,
    bs: PyReadonlyArrayDyn<'py, f32>,
    av: PyReadonlyArrayDyn<'py, f32>,
    bv: PyReadonlyArrayDyn<'py, f32>,
    wvhsec: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    let shape = as_.shape().to_vec();
    if shape != bs.shape() || shape != av.shape() || shape != bv.shape() {
        return Err(PyValueError::new_err(
            "as/bs/av/bv must have identical shapes",
        ));
    }
    if shape.len() != 2 && shape.len() != 3 {
        return Err(PyValueError::new_err(
            "isfvpec expects rank-2 or rank-3 coefficient arrays",
        ));
    }

    let nlat = shape[0];
    let nt = if shape.len() == 2 { 1 } else { shape[2] };
    let (v, w, ierror) = isfvpec_impl(
        nlon,
        &collect_logical_coeffs(as_.as_array()),
        &collect_logical_coeffs(bs.as_array()),
        &collect_logical_coeffs(av.as_array()),
        &collect_logical_coeffs(bv.as_array()),
        nlat,
        nt,
        isym,
        wvhsec.as_slice()?,
        lwork,
    )?;
    if ierror != 0 {
        let empty = ArrayD::<f32>::from_shape_vec(IxDyn(&[0, 0]), Vec::new())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok((
            empty.clone().into_pyarray(py).into_any().unbind(),
            empty.into_pyarray(py).into_any().unbind(),
            ierror,
        ));
    }

    let v = expand_isfvpec_output(v, nlat, nlon, nt, isym);
    let w = expand_isfvpec_output(w, nlat, nlon, nt, isym);
    let out_shape = if shape.len() == 2 {
        IxDyn(&[nlat, nlon])
    } else {
        IxDyn(&[nlat, nlon, nt])
    };
    let v_arr = ArrayD::from_shape_vec(out_shape.clone(), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let w_arr =
        ArrayD::from_shape_vec(out_shape, w).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        v_arr.into_pyarray(py).into_any().unbind(),
        w_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}
