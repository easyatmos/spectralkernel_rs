use ndarray::{ArrayD, IxDyn};
use num_complex::Complex32;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

pub fn onedtotwod_vrtdiv_impl(
    vrtspec: &[Complex32],
    divspec: &[Complex32],
    nlat: usize,
    nmdim: usize,
    nt: usize,
    rsphere: f32,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let expected_len = nmdim * nt;
    if vrtspec.len() != expected_len || divspec.len() != expected_len {
        return Err(PyValueError::new_err(
            "vrtspec/divspec sizes must both match nmdim*nt",
        ));
    }

    let nmdim_f = nmdim as f32;
    let ntrunc = (-1.5_f32 + 0.5_f32 * (9.0_f32 - 8.0_f32 * (1.0_f32 - nmdim_f)).sqrt()) as i32;
    let ntrunc_usize = usize::try_from(ntrunc.max(0)).unwrap_or(0);
    let scale = 0.5_f32;
    let out_len = nlat * nlat * nt;
    let mut br = vec![0.0_f32; out_len];
    let mut bi = vec![0.0_f32; out_len];
    let mut cr = vec![0.0_f32; out_len];
    let mut ci = vec![0.0_f32; out_len];

    br.par_chunks_mut(nt)
        .zip(bi.par_chunks_mut(nt))
        .zip(cr.par_chunks_mut(nt))
        .zip(ci.par_chunks_mut(nt))
        .enumerate()
        .for_each(
            |(ab_flat_idx, (((br_chunk, bi_chunk), cr_chunk), ci_chunk))| {
                let m = ab_flat_idx / nlat + 1;
                let n = ab_flat_idx % nlat + 1;
                if m <= n && n <= ntrunc_usize + 1 && n > 1 {
                    let m0 = m - 1;
                    let nm_before = m0 * (ntrunc_usize + 1) - (m0 * m0.saturating_sub(1)) / 2;
                    let nm = nm_before + (n - m + 1);
                    let ds_base = (nm - 1) * nt;
                    let factor = rsphere / ((n as f32) * ((n - 1) as f32)).sqrt();
                    for i in 0..nt {
                        let div_value = divspec[ds_base + i] / scale;
                        let vrt_value = vrtspec[ds_base + i] / scale;
                        br_chunk[i] = -factor * div_value.re;
                        bi_chunk[i] = -factor * div_value.im;
                        cr_chunk[i] = factor * vrt_value.re;
                        ci_chunk[i] = factor * vrt_value.im;
                    }
                }
            },
        );

    Ok((br, bi, cr, ci))
}

#[pyfunction]
pub fn onedtotwod_vrtdiv<'py>(
    py: Python<'py>,
    vrtspec: PyReadonlyArrayDyn<'py, Complex32>,
    divspec: PyReadonlyArrayDyn<'py, Complex32>,
    nlat: usize,
    rsphere: f32,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let shape = vrtspec.shape().to_vec();
    if shape != divspec.shape().to_vec() {
        return Err(PyValueError::new_err(
            "vrtspec and divspec must have identical shapes",
        ));
    }
    if shape.len() != 1 && shape.len() != 2 {
        return Err(PyValueError::new_err(
            "onedtotwod_vrtdiv expects rank-1 or rank-2 complex arrays",
        ));
    }

    let nmdim = shape[0];
    let nt = if shape.len() == 1 { 1 } else { shape[1] };
    let (br, bi, cr, ci) = onedtotwod_vrtdiv_impl(
        vrtspec.as_slice()?,
        divspec.as_slice()?,
        nlat,
        nmdim,
        nt,
        rsphere,
    )?;

    let shape_out = if shape.len() == 1 {
        IxDyn(&[nlat, nlat])
    } else {
        IxDyn(&[nlat, nlat, nt])
    };

    let br = ArrayD::from_shape_vec(shape_out.clone(), br)
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .into_pyarray(py)
        .into_any()
        .unbind();
    let bi = ArrayD::from_shape_vec(shape_out.clone(), bi)
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .into_pyarray(py)
        .into_any()
        .unbind();
    let cr = ArrayD::from_shape_vec(shape_out.clone(), cr)
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .into_pyarray(py)
        .into_any()
        .unbind();
    let ci = ArrayD::from_shape_vec(shape_out, ci)
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .into_pyarray(py)
        .into_any()
        .unbind();

    Ok((br, bi, cr, ci))
}
