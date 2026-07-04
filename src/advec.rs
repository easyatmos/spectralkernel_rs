use crate::gaqd::gaqd_impl;
use crate::gradgc::gradgc_impl;
use crate::shagc::shagc_impl;
use crate::shagci::shagci_impl;
use crate::shsgc::shsgc_impl;
use crate::shsgci::shsgci_impl;
use crate::vhsgci::vhsgci_impl;
use ndarray::{ArrayD, IxDyn};
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn idx(nlon: usize, i: usize, j: usize) -> usize {
    i * nlon + j
}

fn atanxy(x: f32, y: f32) -> f32 {
    if x == 0.0 && y == 0.0 {
        0.0
    } else {
        y.atan2(x)
    }
}

fn stoc(r: f32, theta: f32, phi: f32) -> (f32, f32, f32) {
    let st = theta.sin();
    (r * st * phi.cos(), r * st * phi.sin(), r * theta.cos())
}

fn gpot_impl(
    t: f32,
    alpha: f32,
    beta: f32,
    omega: f32,
    hzero: f32,
    re: f32,
    nlat: usize,
    nlon: usize,
    colat: &[f32],
) -> PyResult<Vec<f32>> {
    if colat.len() != nlat {
        return Err(PyValueError::new_err("colat length must match nlat"));
    }

    let lambdc = omega * t;
    let (xc, yc, zc) = stoc(1.0, beta, lambdc);
    let ca = alpha.cos();
    let sa = alpha.sin();
    let pi = 4.0_f32 * 1.0_f32.atan();
    let tpi = pi + pi;
    let dlon = tpi / nlon as f32;
    let mut h = vec![0.0_f32; nlat * nlon];

    for j in 0..nlon {
        let lambda = j as f32 * dlon;
        let cl = lambda.cos();
        let sl = lambda.sin();
        for i in 0..nlat {
            let theta = colat[i];
            let st = theta.cos();
            let ct = theta.sin();
            let sth = ca * st + sa * ct * cl;
            let cthclh = ca * ct * cl - sa * st;
            let cthslh = ct * sl;
            let lhat = atanxy(cthclh, cthslh);
            let clh = lhat.cos();
            let slh = lhat.sin();
            let cth = clh * cthclh + slh * cthslh;
            let that = atanxy(sth, cth);
            let (x1, y1, z1) = stoc(1.0, that, lhat);
            let dist = ((x1 - xc).powi(2) + (y1 - yc).powi(2) + (z1 - zc).powi(2)).sqrt();
            if dist < re {
                let r = 2.0 * (dist / 2.0).asin();
                if r < re {
                    h[idx(nlon, i, j)] = hzero * 0.5 * ((r * pi / re).cos() + 1.0);
                }
            }
        }
    }

    Ok(h)
}

fn calc_shagc_sizes(nlat: usize, nlon: usize) -> (usize, usize) {
    let l1 = nlat.min((nlon + 2) / 2);
    let l2 = nlat.div_ceil(2);
    let lsave_i64 = nlat as i64 * (2 * l2 as i64 + 3 * l1 as i64 - 2)
        + 3 * l1 as i64 * (1 - l1 as i64) / 2
        + nlon as i64
        + 15;
    let lsave = usize::try_from(lsave_i64).unwrap_or(0);
    let ldwork = nlat * (nlat + 4);
    (lsave, ldwork)
}

fn calc_vhsgc_sizes(nlat: usize, nlon: usize) -> (usize, usize) {
    let l1 = nlat.min((nlon + 1) / 2);
    let l2 = nlat.div_ceil(2);
    let lsave = 4 * nlat * l2 + 3 * l1.saturating_sub(2) * (2 * nlat - l1 - 1) + nlon + 15;
    let ldwork = 2 * nlat * (nlat + 1) + 1;
    (lsave, ldwork)
}

fn smooth_scalar(
    field: &[f32],
    nlat: usize,
    nlon: usize,
    wshagc: &[f32],
    wshsgc: &[f32],
    lwork: usize,
) -> PyResult<Vec<f32>> {
    let (ar, br, ierr_a) = shagc_impl(field, nlat, nlon, 1, wshagc, lwork)?;
    if ierr_a != 0 {
        return Err(PyValueError::new_err(format!(
            "shagc failed with ierror={ierr_a}"
        )));
    }
    let (smoothed, ierr_s) = shsgc_impl(&ar, &br, nlat, 1, wshsgc, lwork)?;
    if ierr_s != 0 {
        return Err(PyValueError::new_err(format!(
            "shsgc failed with ierror={ierr_s}"
        )));
    }
    Ok(smoothed)
}

fn advec_impl(
    nlat: usize,
    nlon: usize,
    dt: f32,
    ntime: Option<usize>,
) -> PyResult<(Vec<f32>, Vec<f32>, f32, f32, f32, f32, i32)> {
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), 0.0, 0.0, 0.0, 0.0, 1));
    }
    if nlon < 4 {
        return Ok((Vec::new(), Vec::new(), 0.0, 0.0, 0.0, 0.0, 2));
    }
    if dt <= 0.0 {
        return Ok((Vec::new(), Vec::new(), 0.0, 0.0, 0.0, 0.0, 3));
    }

    let pi = 4.0_f32 * 1.0_f32.atan();
    let omega = (pi + pi) / (12.0 * 24.0 * 3600.0);
    let re = 1.0_f32 / 3.0;
    let hzero = 1000.0_f32;
    let alphad = 60.0_f32;
    let alpha = pi * alphad / 180.0;
    let beta = pi / 6.0;
    let tdt = dt + dt;

    let (dtheta, _dwts, ierr_g) = gaqd_impl(nlat as i32);
    if ierr_g != 0 {
        return Ok((Vec::new(), Vec::new(), 0.0, 0.0, 0.0, 0.0, ierr_g));
    }
    let colat: Vec<f32> = dtheta.into_iter().map(|v| v as f32).collect();

    let (lshagc, ldwork_shagc) = calc_shagc_sizes(nlat, nlon);
    let (lvhsgc, ldwork_vhsgc) = calc_vhsgc_sizes(nlat, nlon);
    let (wvhsgc, ierr_v) =
        vhsgci_impl(nlat as i32, nlon as i32, lvhsgc as i32, ldwork_vhsgc as i32);
    let (wshagc, ierr_a) =
        shagci_impl(nlat as i32, nlon as i32, lshagc as i32, ldwork_shagc as i32);
    let (wshsgc, ierr_s) =
        shsgci_impl(nlat as i32, nlon as i32, lshagc as i32, ldwork_shagc as i32);
    if ierr_v != 0 || ierr_a != 0 || ierr_s != 0 {
        return Err(PyValueError::new_err(format!(
            "init failed: vhsgci={ierr_v}, shagci={ierr_a}, shsgci={ierr_s}"
        )));
    }

    let lwork = 4 * nlat * nlon + 2 * nlat * (nlat + 1);
    let mut u = vec![0.0_f32; nlat * nlon];
    let mut v = vec![0.0_f32; nlat * nlon];
    let ca = alpha.cos();
    let sa = alpha.sin();
    let dlon = (pi + pi) / nlon as f32;
    for j in 0..nlon {
        let xlm = j as f32 * dlon;
        let sl = xlm.sin();
        let cl = xlm.cos();
        for i in 0..nlat {
            let st = colat[i].cos();
            let ct = colat[i].sin();
            let cthclh = ca * ct * cl - sa * st;
            let cthslh = ct * sl;
            let xlhat = atanxy(cthclh, cthslh);
            let clh = xlhat.cos();
            let slh = xlhat.sin();
            let cth = clh * cthclh + slh * cthslh;
            let uhat = omega * cth;
            u[idx(nlon, i, j)] = (ca * sl * slh + cl * clh) * uhat;
            v[idx(nlon, i, j)] = (ca * st * cl * slh - st * sl * clh + sa * ct * slh) * uhat;
        }
    }

    let mut phold = gpot_impl(-dt, alpha, beta, omega, hzero, re, nlat, nlon, &colat)?;
    let mut phi = gpot_impl(0.0, alpha, beta, omega, hzero, re, nlat, nlon, &colat)?;
    phold = smooth_scalar(&phold, nlat, nlon, &wshagc, &wshsgc, lwork)?;
    phi = smooth_scalar(&phi, nlat, nlon, &wshagc, &wshsgc, lwork)?;

    let mut p2 = 0.0_f32;
    let mut pmax = 0.0_f32;
    for &value in &phi {
        pmax = pmax.max(value.abs());
        p2 += value * value;
    }
    p2 = p2.sqrt();

    let steps = ntime.unwrap_or_else(|| ((12.0 * 24.0 * 3600.0) / dt + 0.5) as usize);
    let mut time = 0.0_f32;
    let mut pexact = gpot_impl(time, alpha, beta, omega, hzero, re, nlat, nlon, &colat)?;
    let mut phi_report = phi.clone();
    let mut errm = 0.0_f32;
    let mut err2 = 0.0_f32;

    for _ in 0..=steps {
        let (ar, br, ierr_a) = shagc_impl(&phi, nlat, nlon, 1, &wshagc, lwork)?;
        if ierr_a != 0 {
            return Err(PyValueError::new_err(format!(
                "shagc failed with ierror={ierr_a}"
            )));
        }
        let (gdpht, gdphl, _idv, _nlon2, ierr_gd) =
            gradgc_impl(&ar, &br, nlat, 1, 0, &wvhsgc, lwork)?;
        if ierr_gd != 0 {
            return Err(PyValueError::new_err(format!(
                "gradgc failed with ierror={ierr_gd}"
            )));
        }

        pexact = gpot_impl(time, alpha, beta, omega, hzero, re, nlat, nlon, &colat)?;
        phi_report = phi.clone();
        errm = 0.0;
        err2 = 0.0;
        for p in 0..phi.len() {
            let diff = pexact[p] - phi[p];
            errm = errm.max(diff.abs());
            err2 += diff * diff;
        }
        errm /= pmax;
        err2 = err2.sqrt() / p2;

        let mut phnew = vec![0.0_f32; nlat * nlon];
        for p in 0..phi.len() {
            let dpdt = -u[p] * gdphl[p] + v[p] * gdpht[p];
            phnew[p] = phold[p] + tdt * dpdt;
        }
        phold = phi;
        phi = phnew;
        time += dt;
    }

    Ok((phi_report, pexact, errm, err2, pmax, p2, 0))
}

#[pyfunction]
#[pyo3(signature = (nlat, nlon, t, alpha, beta, omega, hzero, re))]
/// Rust entry point for `advec_gpot`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `t`: Parameter `t` passed through to the routine.
/// - `alpha`: Parameter `alpha` passed through to the routine.
/// - `beta`: Parameter `beta` passed through to the routine.
/// - `omega`: Planetary rotation rate.
/// - `hzero`: Parameter `hzero` passed through to the routine.
/// - `re`: Parameter `re` passed through to the routine.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn advec_gpot<'py>(
    py: Python<'py>,
    nlat: usize,
    nlon: usize,
    t: f32,
    alpha: f32,
    beta: f32,
    omega: f32,
    hzero: f32,
    re: f32,
) -> PyResult<(Py<PyAny>, i32)> {
    let (dtheta, _dwts, ierr) = gaqd_impl(nlat as i32);
    if ierr != 0 {
        let arr = ArrayD::from_shape_vec(IxDyn(&[0usize, 0usize]), Vec::<f32>::new())
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        return Ok((arr.into_pyarray(py).into_any().unbind(), ierr));
    }
    let colat: Vec<f32> = dtheta.into_iter().map(|v| v as f32).collect();
    let h = gpot_impl(t, alpha, beta, omega, hzero, re, nlat, nlon, &colat)?;
    let arr = ArrayD::from_shape_vec(IxDyn(&[nlat, nlon]), h)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok((arr.into_pyarray(py).into_any().unbind(), 0))
}

#[pyfunction]
#[pyo3(signature = (nlat=23, nlon=45, dt=600.0, ntime=None))]
/// Rust entry point for `advec`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `dt`: Parameter `dt` passed through to the routine.
/// - `ntime`: Parameter `ntime` passed through to the routine.
///
/// # Returns
/// A Python result containing two NumPy arrays.
pub fn advec<'py>(
    py: Python<'py>,
    nlat: usize,
    nlon: usize,
    dt: f32,
    ntime: Option<usize>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, f32, f32, f32, f32, i32)> {
    let (phi, pexact, errm, err2, pmax, p2, ierr) = advec_impl(nlat, nlon, dt, ntime)?;
    let shape = if ierr == 0 {
        IxDyn(&[nlat, nlon])
    } else {
        IxDyn(&[0usize, 0usize])
    };
    let phi_arr = ArrayD::from_shape_vec(shape.clone(), phi)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let exact_arr = ArrayD::from_shape_vec(shape, pexact)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok((
        phi_arr.into_pyarray(py).into_any().unbind(),
        exact_arr.into_pyarray(py).into_any().unbind(),
        errm,
        err2,
        pmax,
        p2,
        ierr,
    ))
}
