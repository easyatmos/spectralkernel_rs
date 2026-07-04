use crate::vhses::vhses_impl;
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

fn expand_isfvpes_output(
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

/// Core Rust implementation of `isfvpes`.
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
/// - `wvhses`: Workspace initialized by `vhsesi_impl` for regular-grid vector synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the cosine coefficients, sine coefficients, and an error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn isfvpes_impl(
    nlon: usize,
    as_: &[f32],
    bs: &[f32],
    av: &[f32],
    bv: &[f32],
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
    let l1 = nlat.min((nlon + 2) / 2);
    let need_lvhses = l1 * l2 * (2 * nlat - l1 + 1) + nlon + 15;
    ierror = 9;
    if wvhses.len() < need_lvhses {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let mn = l1 * nlat * nt;
    ierror = 10;
    if isym == 0 {
        if lwork < nlat * ((2 * nt + 1) * nlon + 4 * l1 * nt + 1) {
            return Ok((Vec::new(), Vec::new(), ierror));
        }
    } else if lwork < (2 * nt + 1) * nlon + nlat * (4 * l1 * nt + 1) {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let mut br = vec![0.0_f32; expected];
    let mut bi = vec![0.0_f32; expected];
    let mut cr = vec![0.0_f32; expected];
    let mut ci = vec![0.0_f32; expected];
    let mut fnn = vec![0.0_f32; nlat + 1];
    for n in 2..=nlat {
        fnn[n] = -((n as f32) * ((n - 1) as f32)).sqrt();
    }

    let mmax = nlat.min((nlon + 1) / 2);
    for k in 0..nt {
        for n in 2..=nlat {
            let idx = (n - 1) * nt + k;
            br[idx] = -fnn[n] * av[idx];
            bi[idx] = -fnn[n] * bv[idx];
            cr[idx] = fnn[n] * as_[idx];
            ci[idx] = fnn[n] * bs[idx];
        }
        for m in 2..=mmax {
            for n in m..=nlat {
                let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                br[idx] = -fnn[n] * av[idx];
                bi[idx] = -fnn[n] * bv[idx];
                cr[idx] = fnn[n] * as_[idx];
                ci[idx] = fnn[n] * bs[idx];
            }
        }
    }

    let ityp = isym_to_ityp(isym);
    let lwork_vhses = lwork.saturating_sub(4 * mn + nlat);
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
    Ok((v, w, ierr))
}

#[pyfunction]
/// Rust entry point for `isfvpes`.
///
/// # Parameters
/// - `nlon`: Number of longitudes in the grid.
/// - `as_`: Parameter `as_` passed through to the routine.
/// - `bs`: Parameter `bs` passed through to the routine.
/// - `av`: Parameter `av` passed through to the routine.
/// - `bv`: Parameter `bv` passed through to the routine.
/// - `wvhses`: Workspace initialized by `vhsesi_impl` for regular-grid vector synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn isfvpes<'py>(
    py: Python<'py>,
    nlon: usize,
    as_: PyReadonlyArrayDyn<'py, f32>,
    bs: PyReadonlyArrayDyn<'py, f32>,
    av: PyReadonlyArrayDyn<'py, f32>,
    bv: PyReadonlyArrayDyn<'py, f32>,
    wvhses: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    isfvpes_isym(py, nlon, 0, as_, bs, av, bv, wvhses, lwork)
}

#[pyfunction]
/// Rust entry point for `isfvpes_isym`.
///
/// # Parameters
/// - `nlon`: Number of longitudes in the grid.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `as_`: Parameter `as_` passed through to the routine.
/// - `bs`: Parameter `bs` passed through to the routine.
/// - `av`: Parameter `av` passed through to the routine.
/// - `bv`: Parameter `bv` passed through to the routine.
/// - `wvhses`: Workspace initialized by `vhsesi_impl` for regular-grid vector synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn isfvpes_isym<'py>(
    py: Python<'py>,
    nlon: usize,
    isym: usize,
    as_: PyReadonlyArrayDyn<'py, f32>,
    bs: PyReadonlyArrayDyn<'py, f32>,
    av: PyReadonlyArrayDyn<'py, f32>,
    bv: PyReadonlyArrayDyn<'py, f32>,
    wvhses: PyReadonlyArrayDyn<'py, f32>,
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
            "isfvpes expects rank-2 or rank-3 coefficient arrays",
        ));
    }

    let nlat = shape[0];
    let nt = if shape.len() == 2 { 1 } else { shape[2] };
    let (v, w, ierror) = isfvpes_impl(
        nlon,
        &collect_logical_coeffs(as_.as_array()),
        &collect_logical_coeffs(bs.as_array()),
        &collect_logical_coeffs(av.as_array()),
        &collect_logical_coeffs(bv.as_array()),
        nlat,
        nt,
        isym,
        wvhses.as_slice()?,
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

    let v = expand_isfvpes_output(v, nlat, nlon, nt, isym);
    let w = expand_isfvpes_output(w, nlat, nlon, nt, isym);
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
