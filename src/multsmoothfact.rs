use ndarray::{ArrayD, IxDyn};
use num_complex::Complex32;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn multsmoothfact_impl(
    dataspec: &[Complex32],
    smooth: &[f32],
    nmdim: usize,
    nt: usize,
) -> PyResult<Vec<Complex32>> {
    let expected_len = nmdim * nt;
    if dataspec.len() != expected_len {
        return Err(PyValueError::new_err(format!(
            "dataspec size mismatch: expected {expected_len}, got {}",
            dataspec.len()
        )));
    }

    let nmdim_f = nmdim as f32;
    let ntrunc = (-1.5_f32 + 0.5_f32 * (9.0_f32 - 8.0_f32 * (1.0_f32 - nmdim_f)).sqrt()) as i32;
    let ntrunc_usize = usize::try_from(ntrunc.max(0)).unwrap_or(0);

    if smooth.len() < ntrunc_usize + 1 {
        return Err(PyValueError::new_err(format!(
            "smooth length mismatch: need at least {}, got {}",
            ntrunc_usize + 1,
            smooth.len()
        )));
    }

    let mut dataspec_smooth = vec![Complex32::new(0.0_f32, 0.0_f32); expected_len];

    for i in 0..nt {
        let mut nmstrt = 0_usize;
        for m in 1..=ntrunc_usize + 1 {
            for n in m..=ntrunc_usize + 1 {
                let nm = nmstrt + n - m + 1;
                let idx = (nm - 1) * nt + i;
                dataspec_smooth[idx] = dataspec[idx] * smooth[n - 1];
            }
            nmstrt += (ntrunc_usize + 2) - m;
        }
    }

    Ok(dataspec_smooth)
}

#[pyfunction]
pub fn multsmoothfact<'py>(
    py: Python<'py>,
    dataspec: PyReadonlyArrayDyn<'py, Complex32>,
    smooth: PyReadonlyArray1<'py, f32>,
) -> PyResult<Py<PyAny>> {
    let shape = dataspec.shape().to_vec();
    if shape.len() != 1 && shape.len() != 2 {
        return Err(PyValueError::new_err(
            "multsmoothfact expects a rank-1 or rank-2 complex array",
        ));
    }

    let nmdim = shape[0];
    let nt = if shape.len() == 1 { 1 } else { shape[1] };
    let values = dataspec.as_slice()?;
    let smooth = smooth.as_slice()?;
    let result = multsmoothfact_impl(values, smooth, nmdim, nt)?;

    let array = if shape.len() == 1 {
        ArrayD::from_shape_vec(IxDyn(&[nmdim]), result)
    } else {
        ArrayD::from_shape_vec(IxDyn(&[nmdim, nt]), result)
    }
    .map_err(|err| PyValueError::new_err(err.to_string()))?;

    Ok(array.into_pyarray(py).into_any().unbind())
}
