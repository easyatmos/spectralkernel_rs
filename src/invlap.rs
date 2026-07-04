use ndarray::{ArrayD, IxDyn};
use num_complex::Complex32;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn invlap_impl(
    dataspec: &[Complex32],
    nmdim: usize,
    nt: usize,
    rsphere: f32,
) -> PyResult<Vec<Complex32>> {
    if rsphere == 0.0_f32 {
        return Err(PyValueError::new_err("rsphere must be non-zero"));
    }

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
    let radius_sq = rsphere * rsphere;

    let mut dataspec_ilap = vec![Complex32::new(0.0_f32, 0.0_f32); expected_len];

    for i in 0..nt {
        let mut nmstrt = 0_usize;
        for m in 1..=ntrunc_usize + 1 {
            let mut n1 = m;
            if m == 1 {
                n1 = 2;
            }
            for n in n1..=ntrunc_usize + 1 {
                let nm = nmstrt + n - m + 1;
                let factor = -(radius_sq / ((n as f32) * ((n - 1) as f32)));

                let idx = (nm - 1) * nt + i;
                dataspec_ilap[idx] = dataspec[idx] * factor;
            }
            nmstrt += (ntrunc_usize + 2) - m;
        }
        dataspec_ilap[i] = Complex32::new(0.0_f32, 0.0_f32);
    }

    Ok(dataspec_ilap)
}

#[pyfunction]
pub fn invlap<'py>(
    py: Python<'py>,
    dataspec: PyReadonlyArrayDyn<'py, Complex32>,
    rsphere: f32,
) -> PyResult<Py<PyAny>> {
    let shape = dataspec.shape().to_vec();
    if shape.len() != 1 && shape.len() != 2 {
        return Err(PyValueError::new_err(
            "invlap expects a rank-1 or rank-2 complex array",
        ));
    }

    let nmdim = shape[0];
    let nt = if shape.len() == 1 { 1 } else { shape[1] };
    let values = dataspec.as_slice()?;
    let result = invlap_impl(values, nmdim, nt, rsphere)?;

    let array = if shape.len() == 1 {
        ArrayD::from_shape_vec(IxDyn(&[nmdim]), result)
    } else {
        ArrayD::from_shape_vec(IxDyn(&[nmdim, nt]), result)
    }
    .map_err(|err| PyValueError::new_err(err.to_string()))?;

    Ok(array.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
pub fn invlap_nogil<'py>(
    py: Python<'py>,
    dataspec: PyReadonlyArrayDyn<'py, Complex32>,
    rsphere: f32,
) -> PyResult<Py<PyAny>> {
    let shape = dataspec.shape().to_vec();
    if shape.len() != 1 && shape.len() != 2 {
        return Err(PyValueError::new_err(
            "invlap_nogil expects a rank-1 or rank-2 complex array",
        ));
    }

    let nmdim = shape[0];
    let nt = if shape.len() == 1 { 1 } else { shape[1] };
    let values = dataspec.as_slice()?.to_vec();
    let result = py.detach(move || invlap_impl(&values, nmdim, nt, rsphere))?;

    let array = if shape.len() == 1 {
        ArrayD::from_shape_vec(IxDyn(&[nmdim]), result)
    } else {
        ArrayD::from_shape_vec(IxDyn(&[nmdim, nt]), result)
    }
    .map_err(|err| PyValueError::new_err(err.to_string()))?;

    Ok(array.into_pyarray(py).into_any().unbind())
}
