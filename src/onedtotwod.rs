use ndarray::{ArrayD, IxDyn};
use num_complex::Complex32;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Expand packed triangular spectral coefficients into separate cosine and sine coefficient planes.
///
/// # Parameters
/// - `dataspec`: Packed complex spectral coefficients.
/// - `nlat`: Number of latitudes in the grid.
/// - `nmdim`: Number of packed spectral coefficients per field.
/// - `nt`: Number of stacked fields processed together.
///
/// # Returns
/// A Python result containing the cosine and sine coefficient arrays.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn onedtotwod_impl(
    dataspec: &[Complex32],
    nlat: usize,
    nmdim: usize,
    nt: usize,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
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
    let scale = 0.5_f32;
    let ab_len = nlat * nlat * nt;
    let mut a = vec![0.0_f32; ab_len];
    let mut b = vec![0.0_f32; ab_len];

    a.par_chunks_mut(nt)
        .zip(b.par_chunks_mut(nt))
        .enumerate()
        .for_each(|(ab_flat_idx, (a_chunk, b_chunk))| {
            let m = ab_flat_idx / nlat + 1;
            let n = ab_flat_idx % nlat + 1;
            if m <= n && n <= ntrunc_usize + 1 {
                let m0 = m - 1;
                let nm_before = m0 * (ntrunc_usize + 1) - (m0 * m0.saturating_sub(1)) / 2;
                let nm = nm_before + (n - m + 1);
                let ds_base = (nm - 1) * nt;
                for i in 0..nt {
                    let value = dataspec[ds_base + i] / scale;
                    a_chunk[i] = value.re;
                    b_chunk[i] = value.im;
                }
            }
        });

    Ok((a, b))
}

#[pyfunction]
/// Python wrapper for `onedtotwod_impl` that reshapes packed coefficients into NumPy arrays.
///
/// # Parameters
/// - `dataspec`: Packed complex spectral coefficients.
/// - `nlat`: Number of latitudes in the grid.
///
/// # Returns
/// Two NumPy arrays containing the returned coefficient fields.
pub fn onedtotwod<'py>(
    py: Python<'py>,
    dataspec: PyReadonlyArrayDyn<'py, Complex32>,
    nlat: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let shape = dataspec.shape().to_vec();
    if shape.len() != 1 && shape.len() != 2 {
        return Err(PyValueError::new_err(
            "onedtotwod expects a rank-1 or rank-2 complex array",
        ));
    }

    let nmdim = shape[0];
    let nt = if shape.len() == 1 { 1 } else { shape[1] };
    let values = dataspec.as_slice()?;
    let (a, b) = onedtotwod_impl(values, nlat, nmdim, nt)?;

    let shape_out = if shape.len() == 1 {
        IxDyn(&[nlat, nlat])
    } else {
        IxDyn(&[nlat, nlat, nt])
    };

    let a = ArrayD::from_shape_vec(shape_out.clone(), a)
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .into_pyarray(py)
        .into_any()
        .unbind();
    let b = ArrayD::from_shape_vec(shape_out, b)
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .into_pyarray(py)
        .into_any()
        .unbind();

    Ok((a, b))
}
