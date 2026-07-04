use crate::hrfftb::hrfftb_impl;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

fn infer_nlon_from_wshses(nlat: usize, ltotal: usize) -> Option<usize> {
    let imid = (nlat + 1) / 2;
    for nlon in 4..=4 * nlat.max(4) {
        let mmax = nlat.min(nlon / 2 + 1);
        let lpimn = (imid * mmax * (nlat + nlat - mmax + 1)) / 2;
        if lpimn + nlon + 15 == ltotal {
            return Some(nlon);
        }
    }
    None
}

/// Synthesize scalar fields on a regular grid using stored Legendre tables.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `wshses`: Workspace initialized by `shsesi_impl` for regular-grid scalar synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn shses_impl(
    a: &[f32],
    b: &[f32],
    nlat: usize,
    nt: usize,
    wshses: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 2;
    let ltotal = wshses.len();
    if ltotal < 16 {
        return Ok((Vec::new(), ierror));
    }
    let nlon = match infer_nlon_from_wshses(nlat, ltotal) {
        Some(v) => v,
        None => {
            return Err(PyValueError::new_err(
                "failed to infer nlon from wshses length",
            ));
        }
    };
    if nlon < 4 {
        return Ok((Vec::new(), ierror));
    }
    let mmax = nlat.min(nlon / 2 + 1);
    let isym = 0usize;
    ierror = 4;
    if nt == 0 {
        return Ok((Vec::new(), ierror));
    }
    if a.len() != b.len() || a.len() != nlat * nlat * nt {
        return Err(PyValueError::new_err("a/b size mismatch"));
    }
    ierror = 9;
    let imid = (nlat + 1) / 2;
    let lpimn = (imid * mmax * (nlat + nlat - mmax + 1)) / 2;
    if wshses.len() < lpimn + nlon + 15 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 10;
    if lwork < (nt + 1) * nlat * nlon {
        return Ok((Vec::new(), ierror));
    }

    let ls = nlat;
    let mut ge = vec![0.0_f64; ls * nlon * nt];
    let mut go = vec![0.0_f64; imid * nlon * nt];
    let p = &wshses[..lpimn];
    let mdo = if 2 * mmax - 1 > nlon { mmax - 1 } else { mmax };
    let modl = nlat % 2;
    let mut imm1 = imid;
    if modl != 0 {
        imm1 = imid - 1;
    }

    for k in 0..nt {
        for np1 in (1..=nlat).step_by(2) {
            for i in 1..=imid {
                let pidx = (i - 1) + (np1 - 1) * imid;
                let aidx = (np1 - 1) * nt + k;
                let geidx = ((i - 1) * nlon) * nt + k;
                ge[geidx] += (a[aidx] as f64) * (p[pidx] as f64);
            }
        }
    }

    let mut ndo = if nlat % 2 == 0 { nlat - 1 } else { nlat };
    for mp1 in 2..=mdo {
        let m = mp1 - 1;
        let mb = m * (nlat - 1) - (m * (m - 1)) / 2;
        for np1 in (mp1..=ndo).step_by(2) {
            let mn = mb + np1;
            for k in 0..nt {
                for i in 1..=imid {
                    let pidx = (i - 1) + (mn - 1) * imid;
                    let aidx = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                    ge[((i - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k] +=
                        (a[aidx] as f64) * (p[pidx] as f64);
                    ge[((i - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k] +=
                        (b[aidx] as f64) * (p[pidx] as f64);
                }
            }
        }
    }

    if mdo != mmax && mmax <= ndo {
        let mb = mdo * (nlat - 1) - (mdo * (mdo - 1)) / 2;
        for np1 in (mmax..=ndo).step_by(2) {
            let mn = mb + np1;
            for k in 0..nt {
                for i in 1..=imid {
                    let pidx = (i - 1) + (mn - 1) * imid;
                    let aidx = (((mmax - 1) * nlat) + (np1 - 1)) * nt + k;
                    ge[((i - 1) * nlon + (2 * mmax - 2 - 1)) * nt + k] +=
                        (a[aidx] as f64) * (p[pidx] as f64);
                }
            }
        }
    }

    for k in 0..nt {
        for np1 in (2..=nlat).step_by(2) {
            for i in 1..=imm1 {
                let pidx = (i - 1) + (np1 - 1) * imid;
                let aidx = (np1 - 1) * nt + k;
                let goidx = ((i - 1) * nlon) * nt + k;
                go[goidx] += (a[aidx] as f64) * (p[pidx] as f64);
            }
        }
    }

    ndo = if nlat % 2 != 0 { nlat - 1 } else { nlat };
    for mp1 in 2..=mdo {
        let mp2 = mp1 + 1;
        let m = mp1 - 1;
        let mb = m * (nlat - 1) - (m * (m - 1)) / 2;
        for np1 in (mp2..=ndo).step_by(2) {
            let mn = mb + np1;
            for k in 0..nt {
                for i in 1..=imm1 {
                    let pidx = (i - 1) + (mn - 1) * imid;
                    let aidx = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                    go[((i - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k] +=
                        (a[aidx] as f64) * (p[pidx] as f64);
                    go[((i - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k] +=
                        (b[aidx] as f64) * (p[pidx] as f64);
                }
            }
        }
    }

    if mdo != mmax {
        let mp2 = mmax + 1;
        if mp2 <= ndo {
            let mb = mdo * (nlat - 1) - (mdo * (mdo - 1)) / 2;
            for np1 in (mp2..=ndo).step_by(2) {
                let mn = mb + np1;
                for k in 0..nt {
                    for i in 1..=imm1 {
                        let pidx = (i - 1) + (mn - 1) * imid;
                        let aidx = (((mmax - 1) * nlat) + (np1 - 1)) * nt + k;
                        go[((i - 1) * nlon + (2 * mmax - 2 - 1)) * nt + k] +=
                            (a[aidx] as f64) * (p[pidx] as f64);
                    }
                }
            }
        }
    }

    if isym == 0 {
        for k in 0..nt {
            for j in 0..nlon {
                for i in 0..imm1 {
                    let src = (i * nlon + j) * nt + k;
                    let dst = ((imid + i) * nlon + j) * nt + k;
                    ge[dst] = go[src];
                }
            }
        }
    }

    for k in 0..nt {
        let mut plane = vec![0.0_f32; ls * nlon];
        for i in 0..ls {
            for j in 0..nlon {
                plane[j * ls + i] = ge[(i * nlon + j) * nt + k] as f32;
            }
        }
        if nlon % 2 == 0 {
            for i in 0..ls {
                plane[(nlon - 1) * ls + i] *= 2.0_f32;
            }
        }

        hrfftb_impl(ls, nlon, &mut plane, &wshses[lpimn..lpimn + nlon + 15])?;

        for i in 0..ls {
            for j in 0..nlon {
                ge[(i * nlon + j) * nt + k] = plane[j * ls + i] as f64;
            }
        }
    }

    let mut g = vec![0.0_f32; nlat * nlon * nt];
    let nlp1 = nlat + 1;
    for k in 0..nt {
        for j in 1..=nlon {
            for i in 1..=imm1 {
                let top = ((i - 1) * nlon + (j - 1)) * nt + k;
                let bot = (((nlp1 - i) - 1) * nlon + (j - 1)) * nt + k;
                let goidx = ((imid + i - 1) * nlon + (j - 1)) * nt + k;
                g[top] = (0.5_f64 * (ge[top] + ge[goidx])) as f32;
                g[bot] = (0.5_f64 * (ge[top] - ge[goidx])) as f32;
            }
            if modl != 0 {
                let idx = ((imid - 1) * nlon + (j - 1)) * nt + k;
                g[idx] = (0.5_f64 * ge[idx]) as f32;
            }
        }
    }

    Ok((g, 0))
}

/// Parallel implementation of `shses_impl`.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `wshses`: Workspace initialized by `shsesi_impl` for regular-grid scalar synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn shses_impl_parallel(
    a: &[f32],
    b: &[f32],
    nlat: usize,
    nt: usize,
    wshses: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 2;
    let ltotal = wshses.len();
    if ltotal < 16 {
        return Ok((Vec::new(), ierror));
    }
    let nlon = infer_nlon_from_wshses(nlat, ltotal)
        .ok_or_else(|| PyValueError::new_err("failed to infer nlon from wshses length"))?;
    if nlon < 4 {
        return Ok((Vec::new(), ierror));
    }
    let mmax = nlat.min(nlon / 2 + 1);
    ierror = 4;
    if nt == 0 {
        return Ok((Vec::new(), ierror));
    }
    if a.len() != b.len() || a.len() != nlat * nlat * nt {
        return Err(PyValueError::new_err("a/b size mismatch"));
    }
    ierror = 9;
    let imid = (nlat + 1) / 2;
    let lpimn = (imid * mmax * (nlat + nlat - mmax + 1)) / 2;
    if wshses.len() < lpimn + nlon + 15 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 10;
    if lwork < (nt + 1) * nlat * nlon {
        return Ok((Vec::new(), ierror));
    }

    let p = &wshses[..lpimn];
    let whrfft = &wshses[lpimn..lpimn + nlon + 15];
    let mdo = if 2 * mmax - 1 > nlon { mmax - 1 } else { mmax };
    let modl = nlat % 2;
    let imm1 = if modl != 0 { imid - 1 } else { imid };
    let ls = nlat;
    let planes = (0..nt)
        .into_par_iter()
        .map(|k| {
            let mut ge = vec![0.0_f64; ls * nlon];
            let mut go = vec![0.0_f64; imid * nlon];

            for np1 in (1..=nlat).step_by(2) {
                for i in 1..=imid {
                    let pidx = (i - 1) + (np1 - 1) * imid;
                    let aidx = (np1 - 1) * nt + k;
                    ge[(i - 1) * nlon] += (a[aidx] as f64) * (p[pidx] as f64);
                }
            }

            let mut ndo = if nlat % 2 == 0 { nlat - 1 } else { nlat };
            for mp1 in 2..=mdo {
                let m = mp1 - 1;
                let mb = m * (nlat - 1) - (m * (m - 1)) / 2;
                for np1 in (mp1..=ndo).step_by(2) {
                    let mn = mb + np1;
                    for i in 1..=imid {
                        let pidx = (i - 1) + (mn - 1) * imid;
                        let aidx = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                        ge[(i - 1) * nlon + (2 * mp1 - 2 - 1)] +=
                            (a[aidx] as f64) * (p[pidx] as f64);
                        ge[(i - 1) * nlon + (2 * mp1 - 1 - 1)] +=
                            (b[aidx] as f64) * (p[pidx] as f64);
                    }
                }
            }

            if mdo != mmax && mmax <= ndo {
                let mb = mdo * (nlat - 1) - (mdo * (mdo - 1)) / 2;
                for np1 in (mmax..=ndo).step_by(2) {
                    let mn = mb + np1;
                    for i in 1..=imid {
                        let pidx = (i - 1) + (mn - 1) * imid;
                        let aidx = (((mmax - 1) * nlat) + (np1 - 1)) * nt + k;
                        ge[(i - 1) * nlon + (2 * mmax - 2 - 1)] +=
                            (a[aidx] as f64) * (p[pidx] as f64);
                    }
                }
            }

            for np1 in (2..=nlat).step_by(2) {
                for i in 1..=imm1 {
                    let pidx = (i - 1) + (np1 - 1) * imid;
                    let aidx = (np1 - 1) * nt + k;
                    go[(i - 1) * nlon] += (a[aidx] as f64) * (p[pidx] as f64);
                }
            }

            ndo = if nlat % 2 != 0 { nlat - 1 } else { nlat };
            for mp1 in 2..=mdo {
                let mp2 = mp1 + 1;
                let m = mp1 - 1;
                let mb = m * (nlat - 1) - (m * (m - 1)) / 2;
                for np1 in (mp2..=ndo).step_by(2) {
                    let mn = mb + np1;
                    for i in 1..=imm1 {
                        let pidx = (i - 1) + (mn - 1) * imid;
                        let aidx = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                        go[(i - 1) * nlon + (2 * mp1 - 2 - 1)] +=
                            (a[aidx] as f64) * (p[pidx] as f64);
                        go[(i - 1) * nlon + (2 * mp1 - 1 - 1)] +=
                            (b[aidx] as f64) * (p[pidx] as f64);
                    }
                }
            }

            if mdo != mmax {
                let mp2 = mmax + 1;
                if mp2 <= ndo {
                    let mb = mdo * (nlat - 1) - (mdo * (mdo - 1)) / 2;
                    for np1 in (mp2..=ndo).step_by(2) {
                        let mn = mb + np1;
                        for i in 1..=imm1 {
                            let pidx = (i - 1) + (mn - 1) * imid;
                            let aidx = (((mmax - 1) * nlat) + (np1 - 1)) * nt + k;
                            go[(i - 1) * nlon + (2 * mmax - 2 - 1)] +=
                                (a[aidx] as f64) * (p[pidx] as f64);
                        }
                    }
                }
            }

            for j in 0..nlon {
                for i in 0..imm1 {
                    ge[(imid + i) * nlon + j] = go[i * nlon + j];
                }
            }

            let mut plane = vec![0.0_f32; ls * nlon];
            for i in 0..ls {
                for j in 0..nlon {
                    plane[j * ls + i] = ge[i * nlon + j] as f32;
                }
            }
            if nlon % 2 == 0 {
                for i in 0..ls {
                    plane[(nlon - 1) * ls + i] *= 2.0_f32;
                }
            }
            hrfftb_impl(ls, nlon, &mut plane, whrfft)?;
            for i in 0..ls {
                for j in 0..nlon {
                    ge[i * nlon + j] = plane[j * ls + i] as f64;
                }
            }

            let mut gk = vec![0.0_f32; nlat * nlon];
            let nlp1 = nlat + 1;
            for j in 1..=nlon {
                for i in 1..=imm1 {
                    let top = (i - 1) * nlon + (j - 1);
                    let bot = ((nlp1 - i) - 1) * nlon + (j - 1);
                    let goidx = (imid + i - 1) * nlon + (j - 1);
                    gk[top] = (0.5_f64 * (ge[top] + ge[goidx])) as f32;
                    gk[bot] = (0.5_f64 * (ge[top] - ge[goidx])) as f32;
                }
                if modl != 0 {
                    let idx = (imid - 1) * nlon + (j - 1);
                    gk[idx] = (0.5_f64 * ge[idx]) as f32;
                }
            }
            Ok::<Vec<f32>, PyErr>(gk)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let mut g = vec![0.0_f32; nlat * nlon * nt];
    for (k, plane) in planes.into_iter().enumerate() {
        for i in 0..nlat {
            for j in 0..nlon {
                g[(i * nlon + j) * nt + k] = plane[i * nlon + j];
            }
        }
    }
    Ok((g, 0))
}

#[pyfunction]
/// Python wrapper for `shses_impl` that returns NumPy arrays.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `wshses`: Workspace initialized by `shsesi_impl` for regular-grid scalar synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn shses<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    wshses: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err("shses expects rank-2 or rank-3 a/b"));
    }
    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let abuf = a.as_slice()?.to_vec();
    let bbuf = b.as_slice()?.to_vec();
    let wbuf = wshses.as_slice()?.to_vec();
    let wlen = wbuf.len();
    let (g, ierror) = py
        .detach(|| {
            shses_impl_parallel(&abuf, &bbuf, nlat, nt, &wbuf, lwork).map_err(|err| err.to_string())
        })
        .map_err(PyValueError::new_err)?;
    let nlon = infer_nlon_from_wshses(nlat, wlen).unwrap_or(0);
    let gshape = if ashape.len() == 2 {
        vec![nlat, nlon]
    } else {
        vec![nlat, nlon, nt]
    };
    let garr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&gshape), g)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((garr.into_pyarray(py).into_any().unbind(), ierror))
}
