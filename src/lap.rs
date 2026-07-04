use ndarray::{ArrayD, IxDyn};
use num_complex::Complex32;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

pub fn lap_impl(
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

    let mut dataspec_lap = vec![Complex32::new(0.0_f32, 0.0_f32); expected_len];

    for i in 0..nt {
        let mut nmstrt = 0_usize;
        for m in 1..=ntrunc_usize + 1 {
            for n in m..=ntrunc_usize + 1 {
                let nm = nmstrt + n - m + 1;
                let factor = -((n as f32) * ((n - 1) as f32) / radius_sq);
                let idx = (nm - 1) * nt + i;
                dataspec_lap[idx] = dataspec[idx] * factor;
            }
            nmstrt += (ntrunc_usize + 2) - m;
        }
    }

    Ok(dataspec_lap)
}

fn lap_factors(nmdim: usize, rsphere: f32) -> PyResult<Vec<f32>> {
    if rsphere == 0.0_f32 {
        return Err(PyValueError::new_err("rsphere must be non-zero"));
    }

    let nmdim_f = nmdim as f32;
    let ntrunc = (-1.5_f32 + 0.5_f32 * (9.0_f32 - 8.0_f32 * (1.0_f32 - nmdim_f)).sqrt()) as i32;
    let ntrunc_usize = usize::try_from(ntrunc.max(0)).unwrap_or(0);
    let radius_sq = rsphere * rsphere;

    let mut factors = vec![0.0_f32; nmdim];

    let mut nmstrt = 0_usize;
    for m in 1..=ntrunc_usize + 1 {
        for n in m..=ntrunc_usize + 1 {
            let nm = nmstrt + n - m + 1;
            factors[nm - 1] = -((n as f32) * ((n - 1) as f32) / radius_sq);
        }
        nmstrt += (ntrunc_usize + 2) - m;
    }

    Ok(factors)
}

pub fn lap_impl_par_nmdim(
    dataspec: &[Complex32],
    nmdim: usize,
    nt: usize,
    rsphere: f32,
) -> PyResult<Vec<Complex32>> {
    let expected_len = nmdim * nt;
    if dataspec.len() != expected_len {
        return Err(PyValueError::new_err(format!(
            "dataspec size mismatch: expected {expected_len}, got {}",
            dataspec.len()
        )));
    }

    let factors = lap_factors(nmdim, rsphere)?;
    let mut out = vec![Complex32::new(0.0_f32, 0.0_f32); expected_len];

    out.par_chunks_mut(nt)
        .enumerate()
        .for_each(|(k, out_chunk)| {
            let base = k * nt;
            let factor = factors[k];

            for i in 0..nt {
                out_chunk[i] = dataspec[base + i] * factor;
            }
        });

    Ok(out)
}

pub fn lap_impl_par_nt(
    dataspec: &[Complex32],
    nmdim: usize,
    nt: usize,
    rsphere: f32,
) -> PyResult<Vec<Complex32>> {
    let expected_len = nmdim * nt;
    if dataspec.len() != expected_len {
        return Err(PyValueError::new_err(format!(
            "dataspec size mismatch: expected {expected_len}, got {}",
            dataspec.len()
        )));
    }

    let factors = lap_factors(nmdim, rsphere)?;

    let per_t: Vec<Vec<Complex32>> = (0..nt)
        .into_par_iter()
        .map(|i| {
            let mut col = vec![Complex32::new(0.0_f32, 0.0_f32); nmdim];

            for k in 0..nmdim {
                let idx = k * nt + i;
                col[k] = dataspec[idx] * factors[k];
            }

            col
        })
        .collect();

    let mut out = vec![Complex32::new(0.0_f32, 0.0_f32); expected_len];

    for i in 0..nt {
        for k in 0..nmdim {
            out[k * nt + i] = per_t[i][k];
        }
    }

    Ok(out)
}

fn lap_wrap<'py>(
    py: Python<'py>,
    dataspec: PyReadonlyArrayDyn<'py, Complex32>,
    rsphere: f32,
    mode: &str,
) -> PyResult<Py<PyAny>> {
    let shape = dataspec.shape().to_vec();
    if shape.len() != 1 && shape.len() != 2 {
        return Err(PyValueError::new_err(
            "lap expects a rank-1 or rank-2 complex array",
        ));
    }

    let nmdim = shape[0];
    let nt = if shape.len() == 1 { 1 } else { shape[1] };
    let values = dataspec.as_slice()?;

    let result = match mode {
        "serial" => lap_impl(values, nmdim, nt, rsphere)?,
        "nmdim" => lap_impl_par_nmdim(values, nmdim, nt, rsphere)?,
        "nt" => lap_impl_par_nt(values, nmdim, nt, rsphere)?,
        _ => return Err(PyValueError::new_err("unknown lap mode")),
    };

    let array = if shape.len() == 1 {
        ArrayD::from_shape_vec(IxDyn(&[nmdim]), result)
    } else {
        ArrayD::from_shape_vec(IxDyn(&[nmdim, nt]), result)
    }
    .map_err(|err| PyValueError::new_err(err.to_string()))?;

    Ok(array.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
pub fn lap<'py>(
    py: Python<'py>,
    dataspec: PyReadonlyArrayDyn<'py, Complex32>,
    rsphere: f32,
) -> PyResult<Py<PyAny>> {
    lap_wrap(py, dataspec, rsphere, "serial")
}

#[pyfunction]
pub fn lap_nogil<'py>(
    py: Python<'py>,
    dataspec: PyReadonlyArrayDyn<'py, Complex32>,
    rsphere: f32,
) -> PyResult<Py<PyAny>> {
    lap_wrap(py, dataspec, rsphere, "nmdim")
}

#[pyfunction]
pub fn lap_latpar_nogil<'py>(
    py: Python<'py>,
    dataspec: PyReadonlyArrayDyn<'py, Complex32>,
    rsphere: f32,
) -> PyResult<Py<PyAny>> {
    lap_wrap(py, dataspec, rsphere, "nt")
}
