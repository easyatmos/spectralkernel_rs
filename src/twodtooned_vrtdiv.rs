use ndarray::{ArrayD, IxDyn};
use num_complex::Complex32;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Pack two-dimensional vorticity and divergence coefficient arrays into one-dimensional spectra.
///
/// # Parameters
/// - `br`: First vector coefficient family in vector layout.
/// - `bi`: Second vector coefficient family in vector layout.
/// - `cr`: Third vector coefficient family in vector layout.
/// - `ci`: Fourth vector coefficient family in vector layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `ntrunc`: Triangular spectral truncation.
/// - `nt`: Number of stacked fields processed together.
/// - `rsphere`: Sphere radius used to scale Laplacian operators.
///
/// # Returns
/// A Python result containing the transformed complex coefficient arrays.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn twodtooned_vrtdiv_impl(
    br: &[f32],
    bi: &[f32],
    cr: &[f32],
    ci: &[f32],
    nlat: usize,
    ntrunc: i32,
    nt: usize,
    rsphere: f32,
) -> PyResult<(Vec<Complex32>, Vec<Complex32>)> {
    let expected_len = nlat * nlat * nt;
    if br.len() != expected_len
        || bi.len() != expected_len
        || cr.len() != expected_len
        || ci.len() != expected_len
    {
        return Err(PyValueError::new_err(
            "br/bi/cr/ci sizes must all match nlat*nlat*nt",
        ));
    }
    if ntrunc < 0 {
        return Err(PyValueError::new_err("ntrunc must be non-negative"));
    }

    let ntrunc_usize = usize::try_from(ntrunc).unwrap_or(0);
    let nmdim = (ntrunc_usize + 1) * (ntrunc_usize + 2) / 2;
    let scale = 0.5_f32;
    let mut vrtspec = vec![Complex32::new(0.0_f32, 0.0_f32); nmdim * nt];
    let mut divspec = vec![Complex32::new(0.0_f32, 0.0_f32); nmdim * nt];

    vrtspec
        .par_chunks_mut(nt)
        .zip(divspec.par_chunks_mut(nt))
        .enumerate()
        .for_each(|(nm_idx, (vrt_chunk, div_chunk))| {
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
            let factor = ((n as f32) * ((n - 1) as f32)).sqrt() / rsphere;
            let base = ((m - 1) * nlat + (n - 1)) * nt;
            for i in 0..nt {
                div_chunk[i] = Complex32::new(
                    -scale * factor * br[base + i],
                    -scale * factor * bi[base + i],
                );
                vrt_chunk[i] =
                    Complex32::new(scale * factor * cr[base + i], scale * factor * ci[base + i]);
            }
        });

    Ok((vrtspec, divspec))
}

#[pyfunction]
/// Python wrapper for `twodtooned_vrtdiv_impl` that returns packed NumPy arrays.
///
/// # Parameters
/// - `br`: First vector coefficient family in vector layout.
/// - `bi`: Second vector coefficient family in vector layout.
/// - `cr`: Third vector coefficient family in vector layout.
/// - `ci`: Fourth vector coefficient family in vector layout.
/// - `ntrunc`: Triangular spectral truncation.
/// - `rsphere`: Sphere radius used to scale Laplacian operators.
///
/// # Returns
/// Two NumPy arrays containing the returned coefficient fields.
pub fn twodtooned_vrtdiv<'py>(
    py: Python<'py>,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    ntrunc: i32,
    rsphere: f32,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let shape = br.shape().to_vec();
    if shape != bi.shape().to_vec() || shape != cr.shape().to_vec() || shape != ci.shape().to_vec()
    {
        return Err(PyValueError::new_err(
            "br/bi/cr/ci must have identical shapes",
        ));
    }
    if shape.len() != 2 && shape.len() != 3 {
        return Err(PyValueError::new_err(
            "twodtooned_vrtdiv expects rank-2 or rank-3 arrays",
        ));
    }
    if shape[0] != shape[1] {
        return Err(PyValueError::new_err(
            "first two dimensions of br/bi/cr/ci must be equal",
        ));
    }

    let nlat = shape[0];
    let nt = if shape.len() == 2 { 1 } else { shape[2] };
    let (vrtspec, divspec) = twodtooned_vrtdiv_impl(
        br.as_slice()?,
        bi.as_slice()?,
        cr.as_slice()?,
        ci.as_slice()?,
        nlat,
        ntrunc,
        nt,
        rsphere,
    )?;

    let nmdim =
        (usize::try_from(ntrunc).unwrap_or(0) + 1) * (usize::try_from(ntrunc).unwrap_or(0) + 2) / 2;
    let shape_out = if shape.len() == 2 {
        IxDyn(&[nmdim])
    } else {
        IxDyn(&[nmdim, nt])
    };

    let vrtspec = ArrayD::from_shape_vec(shape_out.clone(), vrtspec)
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .into_pyarray(py)
        .into_any()
        .unbind();
    let divspec = ArrayD::from_shape_vec(shape_out, divspec)
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .into_pyarray(py)
        .into_any()
        .unbind();

    Ok((vrtspec, divspec))
}
