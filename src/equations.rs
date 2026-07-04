use crate::invlap::invlap_impl;
use crate::ops::{SphereOps, kinetic_energy_vec, vec_to_py};
use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (ops, q, u, v, form="advective"))]
/// Compute scalar-advection tendency for transport equations on the sphere.
///
/// # Parameters
/// - `ops`: High-level SphereOps spectral operator set used for transforms and differential operators.
/// - `q`: Scalar tracer field on the operator grid.
/// - `u`: Zonal or first vector component on the operator grid.
/// - `v`: Input vector component stored in `(nlat, nlon[, nt])` order.
/// - `form`: Equation form selector for the high-level tendency routine.
///
/// # Returns
/// A Python object containing the returned NumPy array.
///
/// This is a high-level Rust/Python-facing convenience API built on top of this crate's lower-level spectral kernels.
pub fn scalar_advection_rhs<'py>(
    py: Python<'py>,
    ops: PyRef<'py, SphereOps>,
    q: PyReadonlyArrayDyn<'py, f32>,
    u: PyReadonlyArrayDyn<'py, f32>,
    v: PyReadonlyArrayDyn<'py, f32>,
    form: &str,
) -> PyResult<Py<PyAny>> {
    let (mut tendency, nt, was_2d) = match form {
        "advective" => ops.advect_scalar_vec(
            u.as_slice()?,
            &u.shape(),
            v.as_slice()?,
            &v.shape(),
            q.as_slice()?,
            &q.shape(),
        )?,
        "conservative" => ops.flux_div_scalar_vec(
            u.as_slice()?,
            &u.shape(),
            v.as_slice()?,
            &v.shape(),
            q.as_slice()?,
            &q.shape(),
        )?,
        "split" => ops.split_advect_scalar_vec(
            u.as_slice()?,
            &u.shape(),
            v.as_slice()?,
            &v.shape(),
            q.as_slice()?,
            &q.shape(),
        )?,
        _ => {
            return Err(PyValueError::new_err(
                "form must be 'advective', 'conservative', or 'split'",
            ));
        }
    };
    for value in &mut tendency {
        *value = -*value;
    }
    let shape = if was_2d {
        vec![ops.plan.nlat, ops.plan.nlon]
    } else {
        vec![ops.plan.nlat, ops.plan.nlon, nt]
    };
    vec_to_py(py, &shape, tendency)
}

#[pyfunction]
#[pyo3(signature = (ops, zeta, omega=7.292115e-5_f32, form="advective"))]
/// Compute the barotropic-vorticity tendency from relative vorticity.
///
/// # Parameters
/// - `ops`: High-level SphereOps spectral operator set used for transforms and differential operators.
/// - `zeta`: Parameter `zeta` passed through to the routine.
/// - `omega`: Planetary rotation rate.
/// - `form`: Equation form selector for the high-level tendency routine.
///
/// # Returns
/// A Python object containing the returned NumPy array.
///
/// This is a high-level Rust/Python-facing convenience API built on top of this crate's lower-level spectral kernels.
pub fn barotropic_vorticity_rhs<'py>(
    py: Python<'py>,
    ops: PyRef<'py, SphereOps>,
    zeta: PyReadonlyArrayDyn<'py, f32>,
    omega: f32,
    form: &str,
) -> PyResult<Py<PyAny>> {
    if form != "advective" {
        return Err(PyValueError::new_err(
            "barotropic_vorticity_rhs v1 only supports form='advective'",
        ));
    }
    let zshape = zeta.shape().to_vec();
    let (zeta_spec, nt, was_2d) = ops.scalar_to_spec_vec(zeta.as_slice()?, &zshape, None)?;
    let nmdim = ops.plan.nlat * (ops.plan.nlat + 1) / 2;
    let psi_spec = invlap_impl(&zeta_spec, nmdim, nt, ops.plan.radius)?;
    let spec_shape = if was_2d { vec![nmdim] } else { vec![nmdim, nt] };
    let (psi, _, _) = ops.spec_to_scalar_vec(&psi_spec, &spec_shape)?;
    let (u, v, _, _) = ops.streamfunction_to_wind_vec(&psi, &zshape)?;
    let mut eta = zeta.as_slice()?.to_vec();
    let f = ops.coriolis_vec(nt, omega);
    for (eta, f) in eta.iter_mut().zip(f.iter()) {
        *eta += *f;
    }
    let (mut rhs, _, _) = ops.advect_scalar_vec(&u, &zshape, &v, &zshape, &eta, &zshape)?;
    for value in &mut rhs {
        *value = -*value;
    }
    let shape = if was_2d {
        vec![ops.plan.nlat, ops.plan.nlon]
    } else {
        vec![ops.plan.nlat, ops.plan.nlon, nt]
    };
    vec_to_py(py, &shape, rhs)
}

#[pyfunction]
#[pyo3(signature = (ops, h, u, v, g=9.81_f32, omega=7.292115e-5_f32, form="vector_invariant"))]
/// Compute shallow-water tendencies using the vector-invariant form.
///
/// # Parameters
/// - `ops`: High-level SphereOps spectral operator set used for transforms and differential operators.
/// - `h`: Fluid layer depth or height field on the operator grid.
/// - `u`: Zonal or first vector component on the operator grid.
/// - `v`: Input vector component stored in `(nlat, nlon[, nt])` order.
/// - `g`: Input scalar grid values stored in `(nlat, nlon[, nt])` order.
/// - `omega`: Planetary rotation rate.
/// - `form`: Equation form selector for the high-level tendency routine.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This is a high-level Rust/Python-facing convenience API built on top of this crate's lower-level spectral kernels.
pub fn shallow_water_rhs<'py>(
    py: Python<'py>,
    ops: PyRef<'py, SphereOps>,
    h: PyReadonlyArrayDyn<'py, f32>,
    u: PyReadonlyArrayDyn<'py, f32>,
    v: PyReadonlyArrayDyn<'py, f32>,
    g: f32,
    omega: f32,
    form: &str,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    if form != "vector_invariant" {
        return Err(PyValueError::new_err(
            "shallow_water_rhs v1 only supports form='vector_invariant'",
        ));
    }
    let shape = h.shape().to_vec();
    let (mut dhdt, nt, was_2d) = ops.flux_div_scalar_vec(
        u.as_slice()?,
        &u.shape(),
        v.as_slice()?,
        &v.shape(),
        h.as_slice()?,
        &shape,
    )?;
    for value in &mut dhdt {
        *value = -*value;
    }

    let (zeta, _, _, _) =
        ops.wind_to_vrtdiv_vec(u.as_slice()?, &u.shape(), v.as_slice()?, &v.shape())?;
    let f = ops.coriolis_vec(nt, omega);
    let eta = zeta
        .iter()
        .zip(f.iter())
        .map(|(zeta, f)| zeta + f)
        .collect::<Vec<_>>();
    let ke = kinetic_energy_vec(u.as_slice()?, v.as_slice()?);
    let bernoulli = h
        .as_slice()?
        .iter()
        .zip(ke.iter())
        .map(|(h, ke)| g * h + ke)
        .collect::<Vec<_>>();
    let (grad_b_u, grad_b_v) = ops.gradient_grid_vec(&bernoulli, &shape)?;

    let mut dudt = vec![0.0_f32; eta.len()];
    let mut dvdt = vec![0.0_f32; eta.len()];
    for i in 0..eta.len() {
        // k x u = (-v, u), so -eta * (k x u) - grad(B).
        dudt[i] = eta[i] * v.as_slice()?[i] - grad_b_u[i];
        dvdt[i] = -eta[i] * u.as_slice()?[i] - grad_b_v[i];
    }

    let out_shape = if was_2d {
        vec![ops.plan.nlat, ops.plan.nlon]
    } else {
        vec![ops.plan.nlat, ops.plan.nlon, nt]
    };
    Ok((
        vec_to_py(py, &out_shape, dhdt)?,
        vec_to_py(py, &out_shape, dudt)?,
        vec_to_py(py, &out_shape, dvdt)?,
    ))
}
