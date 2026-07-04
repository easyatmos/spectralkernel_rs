use ndarray::{ArrayD, IxDyn};
use num_complex::Complex32;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

pub fn twodtooned_impl(
    a: &[f32],
    b: &[f32],
    nlat: usize,
    ntrunc: i32,
    nt: usize,
) -> PyResult<Vec<Complex32>> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("a and b must have the same size"));
    }

    let expected_ab_len = nlat * nlat * nt;
    if a.len() != expected_ab_len {
        return Err(PyValueError::new_err(format!(
            "a/b size mismatch: expected {expected_ab_len}, got {}",
            a.len()
        )));
    }

    if ntrunc < 0 {
        return Err(PyValueError::new_err("ntrunc must be non-negative"));
    }

    let ntrunc_usize = usize::try_from(ntrunc).unwrap_or(0);
    let nmdim = (ntrunc_usize + 1) * (ntrunc_usize + 2) / 2;
    let mut dataspec = vec![Complex32::new(0.0_f32, 0.0_f32); nmdim * nt];
    let scale = 0.5_f32;

    dataspec
        .par_chunks_mut(nt)
        .enumerate()
        .for_each(|(nm_idx, chunk)| {
            let nm = nm_idx + 1;
            let mut remaining = nm;
            let mut m = 1_usize;
            let mut row_len = ntrunc_usize + 1;
            while remaining > row_len {
                remaining -= row_len;
                m += 1;
                row_len -= 1;
            }
            let n = m + remaining - 1;
            let base = ((m - 1) * nlat + (n - 1)) * nt;
            for i in 0..nt {
                chunk[i] = Complex32::new(scale * a[base + i], scale * b[base + i]);
            }
        });

    Ok(dataspec)
}

#[pyfunction]
pub fn twodtooned<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    ntrunc: i32,
) -> PyResult<Py<PyAny>> {
    let shape_a = a.shape().to_vec();
    let shape_b = b.shape().to_vec();
    if shape_a != shape_b {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if shape_a.len() != 2 && shape_a.len() != 3 {
        return Err(PyValueError::new_err(
            "twodtooned expects rank-2 or rank-3 arrays",
        ));
    }
    if shape_a[0] != shape_a[1] {
        return Err(PyValueError::new_err(
            "first two dimensions of a/b must be equal",
        ));
    }

    let nlat = shape_a[0];
    let nt = if shape_a.len() == 2 { 1 } else { shape_a[2] };
    let values_a = a.as_slice()?;
    let values_b = b.as_slice()?;
    let result = twodtooned_impl(values_a, values_b, nlat, ntrunc, nt)?;
    let nmdim =
        (usize::try_from(ntrunc).unwrap_or(0) + 1) * (usize::try_from(ntrunc).unwrap_or(0) + 2) / 2;

    let array = if shape_a.len() == 2 {
        ArrayD::from_shape_vec(IxDyn(&[nmdim]), result)
    } else {
        ArrayD::from_shape_vec(IxDyn(&[nmdim, nt]), result)
    }
    .map_err(|err| PyValueError::new_err(err.to_string()))?;

    Ok(array.into_pyarray(py).into_any().unbind())
}
