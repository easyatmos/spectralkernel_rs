use ndarray::Array3;
use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;

fn clean_zero(value: f32) -> f32 {
    if value.abs() < 1.0e-6_f32 {
        0.0_f32.copysign(value)
    } else {
        value
    }
}

fn stoc(r: f32, theta: f32, phi: f32) -> (f32, f32, f32) {
    let st = theta.sin();
    let x = r * st * phi.cos();
    let y = r * st * phi.sin();
    let z = r * theta.cos();
    (x, y, z)
}

fn ctos(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let mut r1 = x * x + y * y;
    if r1 != 0.0_f32 {
        let r = (r1 + z * z).sqrt();
        r1 = r1.sqrt();
        let phi = y.atan2(x);
        let theta = r1.atan2(z);
        return (r, theta, phi);
    }

    let mut theta = 0.0_f32;
    if z < 0.0_f32 {
        theta = 4.0_f32 * 1.0_f32.atan();
    }
    (0.0_f32, theta, 0.0_f32)
}

fn idx(i: usize, j: usize, k: usize, jdp: usize) -> usize {
    ((i - 1) * jdp + (j - 1)) * 5 + (k - 1)
}

/// Generate the geodesic grid coordinates on the sphere.
///
/// # Parameters
/// - `m`: Zonal wavenumber or refinement level, depending on the routine.
///
/// # Returns
/// Three coordinate arrays describing the generated grid.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn ihgeod_impl(m: i32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let m = usize::try_from(m.max(1)).unwrap_or(1);
    let idp = m + m - 1;
    let jdp = m;
    let size = idp * jdp * 5;
    let mut x = vec![0.0_f32; size];
    let mut y = vec![0.0_f32; size];
    let mut z = vec![0.0_f32; size];

    let pi = 4.0_f32 * 1.0_f32.atan();
    let dphi = 0.4_f32 * pi;
    let beta = dphi.cos();
    let theta1 = (beta / (1.0_f32 - beta)).acos();
    let theta2 = pi - theta1;
    let hdphi = dphi / 2.0_f32;
    let tdphi = 3.0_f32 * hdphi;

    for k in 1..=5 {
        let phi = (k as f32 - 1.0_f32) * dphi;
        let (x1, y1, z1) = stoc(1.0_f32, theta2, phi);
        let (x2, y2, z2) = stoc(1.0_f32, pi, phi + hdphi);
        let (x3, y3, z3) = stoc(1.0_f32, theta2, phi + dphi);
        let denom = (m - 1) as f32;
        let mut dxi = (x2 - x1) / denom;
        let mut dyi = (y2 - y1) / denom;
        let mut dzi = (z2 - z1) / denom;
        let mut dxj = (x3 - x2) / denom;
        let mut dyj = (y3 - y2) / denom;
        let mut dzj = (z3 - z2) / denom;

        for i in 1..=m {
            let xs = x1 + (i as f32 - 1.0_f32) * dxi;
            let ys = y1 + (i as f32 - 1.0_f32) * dyi;
            let zs = z1 + (i as f32 - 1.0_f32) * dzi;
            for j in 1..=i {
                let p = idx(j, i, k, jdp);
                x[p] = xs + (j as f32 - 1.0_f32) * dxj;
                y[p] = ys + (j as f32 - 1.0_f32) * dyj;
                z[p] = zs + (j as f32 - 1.0_f32) * dzj;
            }
        }

        let (x4, y4, z4) = stoc(1.0_f32, theta1, phi + hdphi);
        dxi = (x3 - x4) / denom;
        dyi = (y3 - y4) / denom;
        dzi = (z3 - z4) / denom;
        dxj = (x4 - x1) / denom;
        dyj = (y4 - y1) / denom;
        dzj = (z4 - z1) / denom;
        for j in 1..=m {
            let xs = x1 + (j as f32 - 1.0_f32) * dxj;
            let ys = y1 + (j as f32 - 1.0_f32) * dyj;
            let zs = z1 + (j as f32 - 1.0_f32) * dzj;
            for i in 1..=j {
                let p = idx(j, i, k, jdp);
                x[p] = xs + (i as f32 - 1.0_f32) * dxi;
                y[p] = ys + (i as f32 - 1.0_f32) * dyi;
                z[p] = zs + (i as f32 - 1.0_f32) * dzi;
            }
        }

        let (x5, y5, z5) = stoc(1.0_f32, theta1, phi + tdphi);
        dxj = (x5 - x3) / denom;
        dyj = (y5 - y3) / denom;
        dzj = (z5 - z3) / denom;
        for i in 1..=m {
            let xs = x4 + (i as f32 - 1.0_f32) * dxi;
            let ys = y4 + (i as f32 - 1.0_f32) * dyi;
            let zs = z4 + (i as f32 - 1.0_f32) * dzi;
            for j in 1..=i {
                let p = idx(j + m - 1, i, k, jdp);
                x[p] = xs + (j as f32 - 1.0_f32) * dxj;
                y[p] = ys + (j as f32 - 1.0_f32) * dyj;
                z[p] = zs + (j as f32 - 1.0_f32) * dzj;
            }
        }

        let (x6, y6, z6) = stoc(1.0_f32, 0.0_f32, phi + dphi);
        dxi = (x5 - x6) / denom;
        dyi = (y5 - y6) / denom;
        dzi = (z5 - z6) / denom;
        dxj = (x6 - x4) / denom;
        dyj = (y6 - y4) / denom;
        dzj = (z6 - z4) / denom;
        for j in 1..=m {
            let xs = x4 + (j as f32 - 1.0_f32) * dxj;
            let ys = y4 + (j as f32 - 1.0_f32) * dyj;
            let zs = z4 + (j as f32 - 1.0_f32) * dzj;
            for i in 1..=j {
                let p = idx(j + m - 1, i, k, jdp);
                x[p] = xs + (i as f32 - 1.0_f32) * dxi;
                y[p] = ys + (i as f32 - 1.0_f32) * dyi;
                z[p] = zs + (i as f32 - 1.0_f32) * dzi;
            }
        }
    }

    for k in 1..=5 {
        for j in 1..=idp {
            for i in 1..=jdp {
                let p = idx(j, i, k, jdp);
                let (_rad, theta, phi) = ctos(x[p], y[p], z[p]);
                let (xn, yn, zn) = stoc(1.0_f32, theta, phi);
                x[p] = clean_zero(xn);
                y[p] = clean_zero(yn);
                z[p] = clean_zero(zn);
            }
        }
    }

    (x, y, z)
}

#[pyfunction]
/// Python wrapper for `ihgeod_impl` that returns geodesic coordinates as NumPy arrays.
///
/// # Parameters
/// - `m`: Zonal wavenumber or refinement level, depending on the routine.
///
/// # Returns
/// A Python result containing three three-dimensional NumPy arrays.
pub fn ihgeod<'py>(
    py: Python<'py>,
    m: i32,
) -> PyResult<(
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray3<f32>>,
)> {
    let m_usize = usize::try_from(m.max(1)).unwrap_or(1);
    let idp = m_usize + m_usize - 1;
    let jdp = m_usize;
    let (x, y, z) = ihgeod_impl(m);
    let x = Array3::from_shape_vec((idp, jdp, 5_usize), x)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?
        .into_pyarray(py)
        .to_owned();
    let y = Array3::from_shape_vec((idp, jdp, 5_usize), y)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?
        .into_pyarray(py)
        .to_owned();
    let z = Array3::from_shape_vec((idp, jdp, 5_usize), z)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?
        .into_pyarray(py)
        .to_owned();
    Ok((x, y, z))
}
