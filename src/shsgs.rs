use crate::hrfftb::hrfftb_impl;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

fn infer_nlon_from_wshsgs(nlat: usize, ltotal: usize) -> Option<(usize, usize, usize, usize)> {
    for nlon in 4..=4 * nlat.max(4) {
        let l = ((nlon + 2) / 2).min(nlat);
        let late = (nlat + (nlat % 2)) / 2;
        let labc = l * (l - 1) / 2 + (nlat - l) * (l - 1);
        let prefix = nlat + 2 * nlat * late + 3 * labc + nlon + 15;
        let pmnf_count = l * (2 * nlat - l + 1) / 2;
        if prefix + late * pmnf_count == ltotal {
            return Some((nlon, l, late, prefix));
        }
    }
    None
}

/// Synthesize scalar fields on a Gaussian grid using stored Legendre tables.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `wshsgs`: Workspace initialized by `shsgsi_impl` for Gaussian-grid scalar synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn shsgs_impl(
    a: &[f32],
    b: &[f32],
    nlat: usize,
    nt: usize,
    wshsgs: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), ierror));
    }
    let (nlon, l, late, ipmn) = infer_nlon_from_wshsgs(nlat, wshsgs.len())
        .ok_or_else(|| PyValueError::new_err("failed to infer nlon from wshsgs length"))?;
    let lqimn = ipmn - (nlon + 15);
    ierror = 2;
    if nlon < 4 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), ierror));
    }
    if a.len() != b.len() || a.len() != nlat * nlat * nt {
        return Err(PyValueError::new_err("a/b size mismatch"));
    }
    ierror = 10;
    if lwork < nlat * nlon * (nt + 1) {
        return Ok((Vec::new(), ierror));
    }

    let mode = 0usize;
    let lat = nlat;
    let mut g = vec![0.0_f32; lat * nlon * nt];
    let pmnf = &wshsgs[ipmn..];
    let lm1 = if nlon == l + l - 2 { l - 1 } else { l };
    let nl2 = nlat / 2;

    let synthesized_planes: Vec<Vec<f32>> = (0..nt)
        .into_par_iter()
        .map(|k| {
            let mut plane = vec![0.0_f64; lat * nlon];

            for np1 in (1..=nlat).step_by(2) {
                let mn = np1;
                for i in 1..=late {
                    plane[(i - 1) * nlon] +=
                        (a[(np1 - 1) * nt + k] as f64) * (pmnf[(mn - 1) * late + (i - 1)] as f64);
                }
            }
            for np1 in (2..=nlat).step_by(2) {
                let mn = np1;
                for i in 1..=nl2 {
                    let is = nlat - i + 1;
                    plane[(is - 1) * nlon] +=
                        (a[(np1 - 1) * nt + k] as f64) * (pmnf[(mn - 1) * late + (i - 1)] as f64);
                }
            }
            for i in 1..=nl2 {
                let is = nlat - i + 1;
                let t1 = plane[(i - 1) * nlon];
                let t3 = plane[(is - 1) * nlon];
                plane[(i - 1) * nlon] = t1 + t3;
                plane[(is - 1) * nlon] = t1 - t3;
            }

            for mp1 in 2..=lm1 {
                let m = mp1 - 1;
                let mml1 = m * (2 * nlat - m - 1) / 2;
                let mp2 = m + 2;
                for np1 in (mp1..=nlat).step_by(2) {
                    let mn = mml1 + np1;
                    for i in 1..=late {
                        plane[(i - 1) * nlon + (2 * m - 1)] +=
                            (a[(((mp1 - 1) * nlat) + (np1 - 1)) * nt + k] as f64)
                                * (pmnf[(mn - 1) * late + (i - 1)] as f64);
                        plane[(i - 1) * nlon + (2 * m)] +=
                            (b[(((mp1 - 1) * nlat) + (np1 - 1)) * nt + k] as f64)
                                * (pmnf[(mn - 1) * late + (i - 1)] as f64);
                    }
                }
                for np1 in (mp2..=nlat).step_by(2) {
                    let mn = mml1 + np1;
                    for i in 1..=nl2 {
                        let is = nlat - i + 1;
                        plane[(is - 1) * nlon + (2 * m - 1)] +=
                            (a[(((mp1 - 1) * nlat) + (np1 - 1)) * nt + k] as f64)
                                * (pmnf[(mn - 1) * late + (i - 1)] as f64);
                        plane[(is - 1) * nlon + (2 * m)] +=
                            (b[(((mp1 - 1) * nlat) + (np1 - 1)) * nt + k] as f64)
                                * (pmnf[(mn - 1) * late + (i - 1)] as f64);
                    }
                }
                for i in 1..=nl2 {
                    let is = nlat - i + 1;
                    let t1 = plane[(i - 1) * nlon + (2 * m - 1)];
                    let t2 = plane[(i - 1) * nlon + (2 * m)];
                    let t3 = plane[(is - 1) * nlon + (2 * m - 1)];
                    let t4 = plane[(is - 1) * nlon + (2 * m)];
                    plane[(i - 1) * nlon + (2 * m - 1)] = t1 + t3;
                    plane[(i - 1) * nlon + (2 * m)] = t2 + t4;
                    plane[(is - 1) * nlon + (2 * m - 1)] = t1 - t3;
                    plane[(is - 1) * nlon + (2 * m)] = t2 - t4;
                }
            }

            if nlon == l + l - 2 {
                let m = l - 1;
                let mml1 = m * (2 * nlat - m - 1) / 2;
                for np1 in (l..=nlat).step_by(2) {
                    let mn = mml1 + np1;
                    for i in 1..=late {
                        plane[(i - 1) * nlon + (nlon - 1)] += 2.0_f64
                            * (a[(((l - 1) * nlat) + (np1 - 1)) * nt + k] as f64)
                            * (pmnf[(mn - 1) * late + (i - 1)] as f64);
                    }
                }
                for np1 in ((l + 1)..=nlat).step_by(2) {
                    let mn = mml1 + np1;
                    for i in 1..=nl2 {
                        let is = nlat - i + 1;
                        plane[(is - 1) * nlon + (nlon - 1)] += 2.0_f64
                            * (a[(((l - 1) * nlat) + (np1 - 1)) * nt + k] as f64)
                            * (pmnf[(mn - 1) * late + (i - 1)] as f64);
                    }
                }
                for i in 1..=nl2 {
                    let is = nlat - i + 1;
                    let t1 = plane[(i - 1) * nlon + (nlon - 1)];
                    let t3 = plane[(is - 1) * nlon + (nlon - 1)];
                    plane[(i - 1) * nlon + (nlon - 1)] = t1 + t3;
                    plane[(is - 1) * nlon + (nlon - 1)] = t1 - t3;
                }
            }

            let mut packed = vec![0.0_f32; lat * nlon];
            for i in 0..lat {
                for j in 0..nlon {
                    packed[j * lat + i] = plane[i * nlon + j] as f32;
                }
            }
            hrfftb_impl(lat, nlon, &mut packed, &wshsgs[lqimn..lqimn + nlon + 15])?;
            let mut plane_out = vec![0.0_f32; lat * nlon];
            for i in 0..lat {
                for j in 0..nlon {
                    plane_out[i * nlon + j] = 0.5 * packed[j * lat + i];
                }
            }
            Ok::<Vec<f32>, PyErr>(plane_out)
        })
        .collect::<PyResult<Vec<_>>>()?;

    for (k, plane) in synthesized_planes.into_iter().enumerate() {
        for i in 0..lat {
            for j in 0..nlon {
                g[(i * nlon + j) * nt + k] = plane[i * nlon + j];
            }
        }
    }

    let _ = mode;
    Ok((g, 0))
}

#[pyfunction]
/// Python wrapper for `shsgs_impl` that returns NumPy arrays.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `wshsgs`: Workspace initialized by `shsgsi_impl` for Gaussian-grid scalar synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn shsgs<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    wshsgs: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err("shsgs expects rank-2 or rank-3 a/b"));
    }
    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let abuf = a.as_slice()?.to_vec();
    let bbuf = b.as_slice()?.to_vec();
    let wbuf = wshsgs.as_slice()?.to_vec();
    let wlen = wbuf.len();
    let (g, ierror) = py
        .detach(|| shsgs_impl(&abuf, &bbuf, nlat, nt, &wbuf, lwork).map_err(|err| err.to_string()))
        .map_err(PyValueError::new_err)?;
    let (nlon, _, _, _) = infer_nlon_from_wshsgs(nlat, wlen).unwrap_or((0, 0, 0, 0));
    let gshape = if ashape.len() == 2 {
        vec![nlat, nlon]
    } else {
        vec![nlat, nlon, nt]
    };
    let garr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&gshape), g)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((garr.into_pyarray(py).into_any().unbind(), ierror))
}
