use crate::hrfftf::fourier_analysis_real;
use crate::zfinit::{zfin_column, zfinit_impl};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Analyze scalar fields on a regular grid using computed Legendre tables.
///
/// # Parameters
/// - `g`: Input scalar grid values stored in `(nlat, nlon[, nt])` order.
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `wshaec`: Workspace initialized by `shaeci_impl` for regular-grid scalar analysis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the cosine coefficients, sine coefficients, and an error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn shaec_impl(
    g: &[f32],
    nlat: usize,
    nlon: usize,
    nt: usize,
    wshaec: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 2;
    if nlon < 4 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 4;
    if nt == 0 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    if g.len() != nlat * nlon * nt {
        return Err(PyValueError::new_err("g size mismatch"));
    }
    let mmax = nlat.min(nlon / 2 + 1);
    let imid = (nlat + 1) / 2;
    let lzz1 = 2 * nlat * imid;
    let labc = 3 * ((mmax - 2) * (nlat + nlat - mmax - 1)) / 2;
    ierror = 9;
    if wshaec.len() < lzz1 + labc + nlon + 15 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 10;
    if lwork < (nt + 1) * nlat * nlon {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let (wzfin, _) = zfinit_impl(nlat, nlon);
    let mut a = vec![0.0_f32; nlat * nlat * nt];
    let mut b = vec![0.0_f32; nlat * nlat * nt];
    let mut ge = vec![0.0_f32; nlat * nlon * nt];
    let mut go = vec![0.0_f32; nlat * nlon * nt];
    let iw1 = lzz1 + labc;
    let whrfft = &wshaec[iw1..iw1 + nlon + 15];

    let mdo = if 2 * mmax - 1 > nlon { mmax - 1 } else { mmax };
    let nlp1 = nlat + 1;
    let tsn = 2.0_f32 / (nlon as f32);
    let modl = nlat % 2;
    let mut imm1 = imid;
    if modl != 0 {
        imm1 = imid - 1;
    }

    for k in 0..nt {
        for i in 1..=imm1 {
            for j in 1..=nlon {
                let top = ((i - 1) * nlon + (j - 1)) * nt + k;
                let bot = (((nlp1 - i) - 1) * nlon + (j - 1)) * nt + k;
                ge[top] = tsn * (g[top] + g[bot]);
                go[top] = tsn * (g[top] - g[bot]);
            }
        }
        if modl != 0 {
            for j in 1..=nlon {
                let idx = ((imid - 1) * nlon + (j - 1)) * nt + k;
                ge[idx] = tsn * g[idx];
            }
        }
    }

    for k in 0..nt {
        let mut plane_ge = vec![0.0_f32; imid * nlon];
        for i in 0..imid {
            for j in 0..nlon {
                plane_ge[i * nlon + j] = ge[(i * nlon + j) * nt + k];
            }
        }
        fourier_analysis_real(imid, nlon, &mut plane_ge, whrfft)?;
        if nlon % 2 == 0 {
            for i in 0..imid {
                plane_ge[i * nlon + (nlon - 1)] *= 0.5_f32;
            }
        }
        for i in 0..imid {
            for j in 0..nlon {
                ge[(i * nlon + j) * nt + k] = plane_ge[i * nlon + j];
            }
        }

        let mut plane_go = vec![0.0_f32; imm1 * nlon];
        for i in 0..imm1 {
            for j in 0..nlon {
                plane_go[i * nlon + j] = go[(i * nlon + j) * nt + k];
            }
        }
        fourier_analysis_real(imm1, nlon, &mut plane_go, whrfft)?;
        if nlon % 2 == 0 {
            for i in 0..imm1 {
                plane_go[i * nlon + (nlon - 1)] *= 0.5_f32;
            }
        }
        for i in 0..imm1 {
            for j in 0..nlon {
                go[(i * nlon + j) * nt + k] = plane_go[i * nlon + j];
            }
        }
    }

    let z0_even = zfin_column(nlat, nlon, 2, 0, &wzfin);
    for k in 0..nt {
        for i in 1..=imid {
            for np1 in (1..=nlat).step_by(2) {
                let zidx = (np1 - 1) * imid + (i - 1);
                let geidx = ((i - 1) * nlon) * nt + k;
                let aidx = (np1 - 1) * nt + k;
                a[aidx] += z0_even[zidx] as f32 * ge[geidx];
            }
        }
    }

    let ndo_even = if nlat % 2 == 0 { nlat - 1 } else { nlat };
    for mp1 in 2..=mdo {
        let m = mp1 - 1;
        let zcol = zfin_column(nlat, nlon, 2, m, &wzfin);
        for k in 0..nt {
            for i in 1..=imid {
                for np1 in (mp1..=ndo_even).step_by(2) {
                    let zidx = (np1 - 1) * imid + (i - 1);
                    let aidx = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                    let ge_cos = ((i - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k;
                    let ge_sin = ((i - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k;
                    a[aidx] += zcol[zidx] as f32 * ge[ge_cos];
                    b[aidx] += zcol[zidx] as f32 * ge[ge_sin];
                }
            }
        }
    }

    if mdo != mmax && mmax <= ndo_even {
        let zcol = zfin_column(nlat, nlon, 2, mdo, &wzfin);
        for k in 0..nt {
            for i in 1..=imid {
                for np1 in (mmax..=ndo_even).step_by(2) {
                    let zidx = (np1 - 1) * imid + (i - 1);
                    let aidx = (((mmax - 1) * nlat) + (np1 - 1)) * nt + k;
                    let ge_cos = ((i - 1) * nlon + (2 * mmax - 2 - 1)) * nt + k;
                    a[aidx] += zcol[zidx] as f32 * ge[ge_cos];
                }
            }
        }
    }

    let z0_odd = zfin_column(nlat, nlon, 1, 0, &wzfin);
    for k in 0..nt {
        for i in 1..=imm1 {
            for np1 in (2..=nlat).step_by(2) {
                let zidx = (np1 - 1) * imid + (i - 1);
                let goidx = ((i - 1) * nlon) * nt + k;
                let aidx = (np1 - 1) * nt + k;
                a[aidx] += z0_odd[zidx] as f32 * go[goidx];
            }
        }
    }

    let ndo_odd = if nlat % 2 != 0 { nlat - 1 } else { nlat };
    for mp1 in 2..=mdo {
        let m = mp1 - 1;
        let mp2 = mp1 + 1;
        let zcol = zfin_column(nlat, nlon, 1, m, &wzfin);
        for k in 0..nt {
            for i in 1..=imm1 {
                for np1 in (mp2..=ndo_odd).step_by(2) {
                    let zidx = (np1 - 1) * imid + (i - 1);
                    let aidx = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                    let go_cos = ((i - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k;
                    let go_sin = ((i - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k;
                    a[aidx] += zcol[zidx] as f32 * go[go_cos];
                    b[aidx] += zcol[zidx] as f32 * go[go_sin];
                }
            }
        }
    }

    if mdo != mmax {
        let mp2 = mmax + 1;
        if mp2 <= ndo_odd {
            let zcol = zfin_column(nlat, nlon, 1, mdo, &wzfin);
            for k in 0..nt {
                for i in 1..=imm1 {
                    for np1 in (mp2..=ndo_odd).step_by(2) {
                        let zidx = (np1 - 1) * imid + (i - 1);
                        let aidx = (((mmax - 1) * nlat) + (np1 - 1)) * nt + k;
                        let go_cos = ((i - 1) * nlon + (2 * mmax - 2 - 1)) * nt + k;
                        a[aidx] += zcol[zidx] as f32 * go[go_cos];
                    }
                }
            }
        }
    }

    Ok((a, b, 0))
}

/// Parallel implementation of `shaec_impl`.
///
/// # Parameters
/// - `g`: Input scalar grid values stored in `(nlat, nlon[, nt])` order.
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `wshaec`: Workspace initialized by `shaeci_impl` for regular-grid scalar analysis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the cosine coefficients, sine coefficients, and an error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn shaec_impl_parallel(
    g: &[f32],
    nlat: usize,
    nlon: usize,
    nt: usize,
    wshaec: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, i32)> {
    if nt <= 1 {
        return shaec_impl(g, nlat, nlon, nt, wshaec, lwork);
    }
    if g.len() != nlat * nlon * nt {
        return Err(PyValueError::new_err("g size mismatch"));
    }
    let coeff_len = nlat * nlat;
    let parts = (0..nt)
        .into_par_iter()
        .map(|k| {
            let mut gk = vec![0.0_f32; nlat * nlon];
            for i in 0..nlat {
                for j in 0..nlon {
                    gk[i * nlon + j] = g[(i * nlon + j) * nt + k];
                }
            }
            let (a, b, ierr) = shaec_impl(&gk, nlat, nlon, 1, wshaec, lwork)?;
            Ok::<_, PyErr>((a, b, ierr))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let mut a = vec![0.0_f32; coeff_len * nt];
    let mut b = vec![0.0_f32; coeff_len * nt];
    for (k, (ak, bk, ierr)) in parts.into_iter().enumerate() {
        if ierr != 0 {
            return Ok((Vec::new(), Vec::new(), ierr));
        }
        for idx in 0..coeff_len {
            a[idx * nt + k] = ak[idx];
            b[idx * nt + k] = bk[idx];
        }
    }
    Ok((a, b, 0))
}

#[pyfunction]
/// Python wrapper for `shaec_impl` that accepts rank-2 or rank-3 NumPy arrays.
///
/// # Parameters
/// - `g`: Input scalar grid values stored in `(nlat, nlon[, nt])` order.
/// - `wshaec`: Workspace initialized by `shaeci_impl` for regular-grid scalar analysis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn shaec<'py>(
    py: Python<'py>,
    g: PyReadonlyArrayDyn<'py, f32>,
    wshaec: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    let shape = g.shape().to_vec();
    if shape.len() != 2 && shape.len() != 3 {
        return Err(PyValueError::new_err("shaec expects rank-2 or rank-3 g"));
    }
    let nlat = shape[0];
    let nlon = shape[1];
    let nt = if shape.len() == 2 { 1 } else { shape[2] };
    let gbuf = g.as_slice()?.to_vec();
    let wbuf = wshaec.as_slice()?.to_vec();
    let (a, b, ierror) = py
        .detach(|| {
            shaec_impl_parallel(&gbuf, nlat, nlon, nt, &wbuf, lwork).map_err(|err| err.to_string())
        })
        .map_err(PyValueError::new_err)?;
    let ashape = if shape.len() == 2 {
        vec![nlat, nlat]
    } else {
        vec![nlat, nlat, nt]
    };
    let aarr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&ashape), a)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let barr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&ashape), b)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        aarr.into_pyarray(py).into_any().unbind(),
        barr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}
