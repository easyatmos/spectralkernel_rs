use crate::islapec::islapec_impl;
use crate::shaec::shaec_impl;
use crate::shaeci::shaeci_impl;
use crate::shseci::shseci_impl;
use ndarray::{ArrayD, IxDyn};
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Core Rust implementation of `helmsph`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `xlmbda`: Parameter `xlmbda` passed through to the routine.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn helmsph_impl(nlat: usize, nlon: usize, xlmbda: f32) -> PyResult<(Vec<f32>, f32, f32, i32)> {
    if nlat < 3 {
        return Ok((Vec::new(), 0.0, 0.0, 1));
    }
    if nlon < 4 {
        return Ok((Vec::new(), 0.0, 0.0, 2));
    }

    let pi = 4.0_f32 * 1.0_f32.atan();
    let imid = (nlat + 1) / 2;
    let mmax = nlat.min(nlon / 2 + 1);
    let llsave =
        2 * nlat * imid + 3 * ((mmax.saturating_sub(2)) * (2 * nlat - mmax - 1)) / 2 + nlon + 15;
    let llwork = nlat * (2 * nlon + 3 * (nlat + 1) + 2 * nlat + 1);
    let lldwork = nlat + 1;

    let dlat = pi / ((nlat - 1) as f32);
    let dlon = (2.0_f32 * pi) / (nlon as f32);

    let mut sint = vec![0.0_f32; nlat];
    let mut cost = vec![0.0_f32; nlat];
    for i in 0..nlat {
        let theta = -0.5_f32 * pi + (i as f32) * dlat;
        sint[i] = theta.sin();
        cost[i] = theta.cos();
    }

    let mut sinp = vec![0.0_f32; nlon];
    let mut cosp = vec![0.0_f32; nlon];
    for j in 0..nlon {
        let phi = (j as f32) * dlon;
        sinp[j] = phi.sin();
        cosp[j] = phi.cos();
    }

    let mut r = vec![0.0_f32; nlat * nlon];
    for j in 0..nlon {
        for i in 0..nlat {
            let x = cost[i] * cosp[j];
            let y = cost[i] * sinp[j];
            let z = sint[i];
            r[i * nlon + j] = -((x * y * (z * z + 6.0 * (z + 1.0))) + z * (z + 2.0)) * z.exp();
        }
    }

    let (wshaec, ierr_shaeci) =
        shaeci_impl(nlat as i32, nlon as i32, llsave as i32, lldwork as i32);
    if ierr_shaeci != 0 {
        return Ok((Vec::new(), 0.0, 0.0, ierr_shaeci));
    }
    let (wshsec, ierr_shseci) =
        shseci_impl(nlat as i32, nlon as i32, llsave as i32, lldwork as i32);
    if ierr_shseci != 0 {
        return Ok((Vec::new(), 0.0, 0.0, ierr_shseci));
    }

    let (a, b, ierr_shaec) = shaec_impl(&r, nlat, nlon, 1, &wshaec, llwork)?;
    if ierr_shaec != 0 {
        return Ok((Vec::new(), 0.0, 0.0, ierr_shaec));
    }

    let (u, pertrb, ierr_islapec) =
        islapec_impl(nlon, &[xlmbda], &a, &b, nlat, 0, 1, &wshsec, llwork)?;
    if ierr_islapec != 0 {
        return Ok((Vec::new(), 0.0, 0.0, ierr_islapec));
    }

    let mut errm = 0.0_f32;
    for j in 0..nlon {
        for i in 0..nlat {
            let x = cost[i] * cosp[j];
            let y = cost[i] * sinp[j];
            let z = sint[i];
            let ue = (1.0 + x * y) * z.exp();
            errm = errm.max((u[i * nlon + j] - ue).abs());
        }
    }

    Ok((u, pertrb[0], errm, 0))
}

#[pyfunction]
/// Solve the scalar Helmholtz equation on the sphere using the spectral plan.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `xlmbda`: Parameter `xlmbda` passed through to the routine.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn helmsph<'py>(
    py: Python<'py>,
    nlat: usize,
    nlon: usize,
    xlmbda: f32,
) -> PyResult<(Py<PyAny>, f32, f32, i32)> {
    let (u, pertrb, errm, ierror) = helmsph_impl(nlat, nlon, xlmbda)?;
    if ierror != 0 {
        let empty = ArrayD::<f32>::from_shape_vec(IxDyn(&[0, 0]), Vec::new())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok((
            empty.into_pyarray(py).into_any().unbind(),
            pertrb,
            errm,
            ierror,
        ));
    }
    let arr = ArrayD::from_shape_vec(IxDyn(&[nlat, nlon, 1]), u)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        arr.into_pyarray(py).into_any().unbind(),
        pertrb,
        errm,
        ierror,
    ))
}
