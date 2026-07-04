use crate::hrfftb::hrfftb_impl;
use crate::legin::legin_compute;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

fn infer_nlon_from_wshsgc(nlat: usize, ltotal: usize) -> Option<(usize, usize, usize)> {
    for nlon in 4..=4 * nlat.max(4) {
        let l = ((nlon + 2) / 2).min(nlat);
        let late = (nlat + (nlat % 2)) / 2;
        let need_i64 = (nlat as i64) * (2 * late as i64 + 3 * l as i64 - 2)
            + (3 * l as i64 * (1 - l as i64)) / 2
            + nlon as i64
            + 15;
        if usize::try_from(need_i64).ok()? == ltotal {
            return Some((nlon, l, late));
        }
    }
    None
}

/// Synthesize scalar fields on a Gaussian grid using computed Legendre tables.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `wshsgc`: Workspace initialized by `shsgci_impl` for Gaussian-grid scalar synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn shsgc_impl(
    a: &[f32],
    b: &[f32],
    nlat: usize,
    nt: usize,
    wshsgc: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), ierror));
    }
    let (nlon, l, late) = infer_nlon_from_wshsgc(nlat, wshsgc.len())
        .ok_or_else(|| PyValueError::new_err("failed to infer nlon from wshsgc length"))?;
    let lpimn_i64 =
        (nlat as i64) * (2 * late as i64 + 3 * l as i64 - 2) + (3 * l as i64 * (1 - l as i64)) / 2;
    let lpimn = usize::try_from(lpimn_i64)
        .map_err(|_| PyValueError::new_err("invalid lpimn inferred from wshsgc length"))?;
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
    let mut pmn = vec![0.0_f32; nlat * late * 3];
    let mut km_state = (0usize, 1usize, 2usize);
    let lm1 = if nlon == l + l - 2 { l - 1 } else { l };
    let nl2 = nlat / 2;

    let km0 = legin_compute(mode, l, nlat, 0, wshsgc, &mut pmn, &mut km_state);
    for k in 0..nt {
        let mut acc = vec![0.0_f64; lat * nlon];

        for np1 in (1..=nlat).step_by(2) {
            for i in 1..=late {
                let pidx = ((km0 * late + (i - 1)) * nlat) + (np1 - 1);
                acc[(i - 1) * nlon] += (a[(np1 - 1) * nt + k] as f64) * (pmn[pidx] as f64);
            }
        }
        for np1 in (2..=nlat).step_by(2) {
            for i in 1..=nl2 {
                let is = nlat - i + 1;
                let pidx = ((km0 * late + (i - 1)) * nlat) + (np1 - 1);
                acc[(is - 1) * nlon] += (a[(np1 - 1) * nt + k] as f64) * (pmn[pidx] as f64);
            }
        }
        for i in 1..=nl2 {
            let is = nlat - i + 1;
            let t1 = acc[(i - 1) * nlon];
            let t3 = acc[(is - 1) * nlon];
            acc[(i - 1) * nlon] = t1 + t3;
            acc[(is - 1) * nlon] = t1 - t3;
        }

        for mp1 in 2..=lm1 {
            let m = mp1 - 1;
            let mp2 = m + 2;
            let km = legin_compute(mode, l, nlat, m, wshsgc, &mut pmn, &mut km_state);
            for np1 in (mp1..=nlat).step_by(2) {
                for i in 1..=late {
                    let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                    let abidx = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                    acc[(i - 1) * nlon + (2 * m - 1)] += (a[abidx] as f64) * (pmn[pidx] as f64);
                    acc[(i - 1) * nlon + (2 * m)] += (b[abidx] as f64) * (pmn[pidx] as f64);
                }
            }
            for np1 in (mp2..=nlat).step_by(2) {
                for i in 1..=nl2 {
                    let is = nlat - i + 1;
                    let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                    let abidx = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                    acc[(is - 1) * nlon + (2 * m - 1)] += (a[abidx] as f64) * (pmn[pidx] as f64);
                    acc[(is - 1) * nlon + (2 * m)] += (b[abidx] as f64) * (pmn[pidx] as f64);
                }
            }
            for i in 1..=nl2 {
                let is = nlat - i + 1;
                let t1 = acc[(i - 1) * nlon + (2 * m - 1)];
                let t2 = acc[(i - 1) * nlon + (2 * m)];
                let t3 = acc[(is - 1) * nlon + (2 * m - 1)];
                let t4 = acc[(is - 1) * nlon + (2 * m)];
                acc[(i - 1) * nlon + (2 * m - 1)] = t1 + t3;
                acc[(i - 1) * nlon + (2 * m)] = t2 + t4;
                acc[(is - 1) * nlon + (2 * m - 1)] = t1 - t3;
                acc[(is - 1) * nlon + (2 * m)] = t2 - t4;
            }
        }

        if nlon == l + l - 2 {
            let km = legin_compute(mode, l, nlat, l - 1, wshsgc, &mut pmn, &mut km_state);
            for np1 in (l..=nlat).step_by(2) {
                for i in 1..=late {
                    let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                    let aidx = (((l - 1) * nlat) + (np1 - 1)) * nt + k;
                    acc[(i - 1) * nlon + (nlon - 1)] +=
                        2.0_f64 * (a[aidx] as f64) * (pmn[pidx] as f64);
                }
            }
            for np1 in ((l + 1)..=nlat).step_by(2) {
                for i in 1..=nl2 {
                    let is = nlat - i + 1;
                    let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                    let aidx = (((l - 1) * nlat) + (np1 - 1)) * nt + k;
                    acc[(is - 1) * nlon + (nlon - 1)] +=
                        2.0_f64 * (a[aidx] as f64) * (pmn[pidx] as f64);
                }
            }
            for i in 1..=nl2 {
                let is = nlat - i + 1;
                let t1 = acc[(i - 1) * nlon + (nlon - 1)];
                let t3 = acc[(is - 1) * nlon + (nlon - 1)];
                acc[(i - 1) * nlon + (nlon - 1)] = t1 + t3;
                acc[(is - 1) * nlon + (nlon - 1)] = t1 - t3;
            }
        }

        for i in 0..lat {
            for j in 0..nlon {
                g[(i * nlon + j) * nt + k] = acc[i * nlon + j] as f32;
            }
        }
    }

    for k in 0..nt {
        let mut plane = vec![0.0_f32; lat * nlon];
        for i in 0..lat {
            for j in 0..nlon {
                plane[i * nlon + j] = g[(i * nlon + j) * nt + k];
            }
        }
        let mut packed = vec![0.0_f32; lat * nlon];
        for i in 0..lat {
            for j in 0..nlon {
                packed[j * lat + i] = plane[i * nlon + j];
            }
        }
        hrfftb_impl(lat, nlon, &mut packed, &wshsgc[lpimn..lpimn + nlon + 15])?;
        for i in 0..lat {
            for j in 0..nlon {
                g[(i * nlon + j) * nt + k] = 0.5 * packed[j * lat + i];
            }
        }
    }

    Ok((g, 0))
}

/// Parallel implementation of `shsgc_impl`.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `wshsgc`: Workspace initialized by `shsgci_impl` for Gaussian-grid scalar synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn shsgc_impl_parallel(
    a: &[f32],
    b: &[f32],
    nlat: usize,
    nt: usize,
    wshsgc: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), ierror));
    }
    let (nlon, l, late) = infer_nlon_from_wshsgc(nlat, wshsgc.len())
        .ok_or_else(|| PyValueError::new_err("failed to infer nlon from wshsgc length"))?;
    let lpimn_i64 =
        (nlat as i64) * (2 * late as i64 + 3 * l as i64 - 2) + (3 * l as i64 * (1 - l as i64)) / 2;
    let lpimn = usize::try_from(lpimn_i64)
        .map_err(|_| PyValueError::new_err("invalid lpimn inferred from wshsgc length"))?;
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
    let lm1 = if nlon == l + l - 2 { l - 1 } else { l };
    let nl2 = nlat / 2;
    let whrfft = &wshsgc[lpimn..lpimn + nlon + 15];
    let planes = (0..nt)
        .into_par_iter()
        .map(|k| {
            let mut acc = vec![0.0_f64; lat * nlon];
            let mut pmn = vec![0.0_f32; nlat * late * 3];
            let mut km_state = (0usize, 1usize, 2usize);

            let km0 = legin_compute(mode, l, nlat, 0, wshsgc, &mut pmn, &mut km_state);
            for np1 in (1..=nlat).step_by(2) {
                for i in 1..=late {
                    let pidx = ((km0 * late + (i - 1)) * nlat) + (np1 - 1);
                    acc[(i - 1) * nlon] += (a[(np1 - 1) * nt + k] as f64) * (pmn[pidx] as f64);
                }
            }
            for np1 in (2..=nlat).step_by(2) {
                for i in 1..=nl2 {
                    let is = nlat - i + 1;
                    let pidx = ((km0 * late + (i - 1)) * nlat) + (np1 - 1);
                    acc[(is - 1) * nlon] += (a[(np1 - 1) * nt + k] as f64) * (pmn[pidx] as f64);
                }
            }
            for i in 1..=nl2 {
                let is = nlat - i + 1;
                let t1 = acc[(i - 1) * nlon];
                let t3 = acc[(is - 1) * nlon];
                acc[(i - 1) * nlon] = t1 + t3;
                acc[(is - 1) * nlon] = t1 - t3;
            }

            for mp1 in 2..=lm1 {
                let m = mp1 - 1;
                let mp2 = m + 2;
                let km = legin_compute(mode, l, nlat, m, wshsgc, &mut pmn, &mut km_state);
                for np1 in (mp1..=nlat).step_by(2) {
                    for i in 1..=late {
                        let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                        let abidx = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                        acc[(i - 1) * nlon + (2 * m - 1)] += (a[abidx] as f64) * (pmn[pidx] as f64);
                        acc[(i - 1) * nlon + (2 * m)] += (b[abidx] as f64) * (pmn[pidx] as f64);
                    }
                }
                for np1 in (mp2..=nlat).step_by(2) {
                    for i in 1..=nl2 {
                        let is = nlat - i + 1;
                        let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                        let abidx = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                        acc[(is - 1) * nlon + (2 * m - 1)] +=
                            (a[abidx] as f64) * (pmn[pidx] as f64);
                        acc[(is - 1) * nlon + (2 * m)] += (b[abidx] as f64) * (pmn[pidx] as f64);
                    }
                }
                for i in 1..=nl2 {
                    let is = nlat - i + 1;
                    let t1 = acc[(i - 1) * nlon + (2 * m - 1)];
                    let t2 = acc[(i - 1) * nlon + (2 * m)];
                    let t3 = acc[(is - 1) * nlon + (2 * m - 1)];
                    let t4 = acc[(is - 1) * nlon + (2 * m)];
                    acc[(i - 1) * nlon + (2 * m - 1)] = t1 + t3;
                    acc[(i - 1) * nlon + (2 * m)] = t2 + t4;
                    acc[(is - 1) * nlon + (2 * m - 1)] = t1 - t3;
                    acc[(is - 1) * nlon + (2 * m)] = t2 - t4;
                }
            }

            if nlon == l + l - 2 {
                let km = legin_compute(mode, l, nlat, l - 1, wshsgc, &mut pmn, &mut km_state);
                for np1 in (l..=nlat).step_by(2) {
                    for i in 1..=late {
                        let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                        let aidx = (((l - 1) * nlat) + (np1 - 1)) * nt + k;
                        acc[(i - 1) * nlon + (nlon - 1)] +=
                            2.0_f64 * (a[aidx] as f64) * (pmn[pidx] as f64);
                    }
                }
                for np1 in ((l + 1)..=nlat).step_by(2) {
                    for i in 1..=nl2 {
                        let is = nlat - i + 1;
                        let pidx = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                        let aidx = (((l - 1) * nlat) + (np1 - 1)) * nt + k;
                        acc[(is - 1) * nlon + (nlon - 1)] +=
                            2.0_f64 * (a[aidx] as f64) * (pmn[pidx] as f64);
                    }
                }
                for i in 1..=nl2 {
                    let is = nlat - i + 1;
                    let t1 = acc[(i - 1) * nlon + (nlon - 1)];
                    let t3 = acc[(is - 1) * nlon + (nlon - 1)];
                    acc[(i - 1) * nlon + (nlon - 1)] = t1 + t3;
                    acc[(is - 1) * nlon + (nlon - 1)] = t1 - t3;
                }
            }

            let mut packed = vec![0.0_f32; lat * nlon];
            for i in 0..lat {
                for j in 0..nlon {
                    packed[j * lat + i] = acc[i * nlon + j] as f32;
                }
            }
            hrfftb_impl(lat, nlon, &mut packed, whrfft)?;
            let mut plane = vec![0.0_f32; lat * nlon];
            for i in 0..lat {
                for j in 0..nlon {
                    plane[i * nlon + j] = 0.5 * packed[j * lat + i];
                }
            }
            Ok::<Vec<f32>, PyErr>(plane)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let mut g = vec![0.0_f32; lat * nlon * nt];
    for (k, plane) in planes.into_iter().enumerate() {
        for i in 0..lat {
            for j in 0..nlon {
                g[(i * nlon + j) * nt + k] = plane[i * nlon + j];
            }
        }
    }
    Ok((g, 0))
}

#[pyfunction]
/// Python wrapper for `shsgc_impl` that returns NumPy arrays.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `wshsgc`: Workspace initialized by `shsgci_impl` for Gaussian-grid scalar synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn shsgc<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    wshsgc: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err("shsgc expects rank-2 or rank-3 a/b"));
    }
    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let abuf = a.as_slice()?.to_vec();
    let bbuf = b.as_slice()?.to_vec();
    let wbuf = wshsgc.as_slice()?.to_vec();
    let wlen = wbuf.len();
    let (g, ierror) = py
        .detach(|| {
            shsgc_impl_parallel(&abuf, &bbuf, nlat, nt, &wbuf, lwork).map_err(|err| err.to_string())
        })
        .map_err(PyValueError::new_err)?;
    let (nlon, _, _) = infer_nlon_from_wshsgc(nlat, wlen).unwrap_or((0, 0, 0));
    let gshape = if ashape.len() == 2 {
        vec![nlat, nlon]
    } else {
        vec![nlat, nlon, nt]
    };
    let garr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&gshape), g)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((garr.into_pyarray(py).into_any().unbind(), ierror))
}
