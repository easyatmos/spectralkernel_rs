use crate::alinit::alin_column;
use crate::hrfftb::hrfftb_impl;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

fn infer_nlon_from_wshsec(nlat: usize, ltotal: usize) -> Option<usize> {
    let imid = (nlat + 1) / 2;
    for nlon in 4..=4 * nlat.max(4) {
        let mmax = nlat.min(nlon / 2 + 1);
        let lzz1 = 2 * nlat * imid;
        let labc = 3 * ((mmax.saturating_sub(2)) * (nlat + nlat - mmax - 1)) / 2;
        if lzz1 + labc + nlon + 15 == ltotal {
            return Some(nlon);
        }
    }
    None
}

fn add_assign(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

fn mp1_ranges(start: usize, end: usize) -> Vec<(usize, usize)> {
    if start > end {
        return Vec::new();
    }
    let count = end - start + 1;
    let threads = rayon::current_num_threads().max(1);
    let chunk = count.div_ceil(threads).max(1);
    let mut ranges = Vec::new();
    let mut s = start;
    while s <= end {
        let e = (s + chunk - 1).min(end);
        ranges.push((s, e));
        s = e + 1;
    }
    ranges
}

fn build_shsec_output(
    py: Python<'_>,
    gshape: &[usize],
    g: Vec<f32>,
    ierror: i32,
) -> PyResult<(Py<PyAny>, i32)> {
    let garr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(gshape), g)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((garr.into_pyarray(py).into_any().unbind(), ierror))
}

/// Synthesize scalar fields on a regular grid using computed Legendre tables.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `wshsec`: Workspace initialized by `shseci_impl` for regular-grid scalar synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn shsec_impl(
    a: &[f32],
    b: &[f32],
    nlat: usize,
    nt: usize,
    isym: usize,
    wshsec: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 2;
    let nlon = infer_nlon_from_wshsec(nlat, wshsec.len())
        .ok_or_else(|| PyValueError::new_err("failed to infer nlon from wshsec length"))?;
    if nlon < 4 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 4;
    if nt == 0 {
        return Ok((Vec::new(), ierror));
    }
    if a.len() != b.len() || a.len() != nlat * nlat * nt {
        return Err(PyValueError::new_err("a/b size mismatch"));
    }

    let mmax = nlat.min(nlon / 2 + 1);
    let imid = (nlat + 1) / 2;
    let lzz1 = 2 * nlat * imid;
    let labc = 3 * ((mmax - 2) * (nlat + nlat - mmax - 1)) / 2;
    ierror = 9;
    if wshsec.len() < lzz1 + labc + nlon + 15 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 10;
    let ls = if isym == 0 { nlat } else { imid };
    let nln = nt * ls * nlon;
    if lwork < nln + (ls * nlon).max(3 * nlat * imid) {
        return Ok((Vec::new(), ierror));
    }

    let mut ge = vec![0.0_f32; ls * nlon * nt];
    let mut go = vec![0.0_f32; imid * nlon * nt];
    let mdo = if 2 * mmax - 1 > nlon { mmax - 1 } else { mmax };
    let modl = nlat % 2;
    let mut imm1 = imid;
    if modl != 0 {
        imm1 = imid - 1;
    }

    let walin_f64: Vec<f64> = wshsec[..(lzz1 + labc)].iter().map(|&v| v as f64).collect();
    let walin = walin_f64.as_slice();

    let p0_even = alin_column(nlat, nlon, 2, 0, walin);
    for k in 0..nt {
        for np1 in (1..=nlat).step_by(2) {
            for i in 1..=imid {
                let pidx = (np1 - 1) * imid + (i - 1);
                let aidx = (np1 - 1) * nt + k;
                let geidx = ((i - 1) * nlon) * nt + k;
                ge[geidx] += a[aidx] * p0_even[pidx] as f32;
            }
        }
    }

    let ndo_even = if nlat % 2 == 0 { nlat - 1 } else { nlat };
    for mp1 in 2..=mdo {
        let m = mp1 - 1;
        let pcol = alin_column(nlat, nlon, 2, m, walin);
        for np1 in (mp1..=ndo_even).step_by(2) {
            for k in 0..nt {
                for i in 1..=imid {
                    let pidx = (np1 - 1) * imid + (i - 1);
                    let aidx = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                    ge[((i - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k] +=
                        a[aidx] * pcol[pidx] as f32;
                    ge[((i - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k] +=
                        b[aidx] * pcol[pidx] as f32;
                }
            }
        }
    }

    if mdo != mmax && mmax <= ndo_even {
        let pcol = alin_column(nlat, nlon, 2, mdo, walin);
        for np1 in (mmax..=ndo_even).step_by(2) {
            for k in 0..nt {
                for i in 1..=imid {
                    let pidx = (np1 - 1) * imid + (i - 1);
                    let aidx = (((mmax - 1) * nlat) + (np1 - 1)) * nt + k;
                    ge[((i - 1) * nlon + (2 * mmax - 2 - 1)) * nt + k] +=
                        a[aidx] * pcol[pidx] as f32;
                }
            }
        }
    }

    let p0_odd = alin_column(nlat, nlon, 1, 0, walin);
    for k in 0..nt {
        for np1 in (2..=nlat).step_by(2) {
            for i in 1..=imm1 {
                let pidx = (np1 - 1) * imid + (i - 1);
                let aidx = (np1 - 1) * nt + k;
                let goidx = ((i - 1) * nlon) * nt + k;
                go[goidx] += a[aidx] * p0_odd[pidx] as f32;
            }
        }
    }

    let ndo_odd = if nlat % 2 != 0 { nlat - 1 } else { nlat };
    for mp1 in 2..=mdo {
        let mp2 = mp1 + 1;
        let m = mp1 - 1;
        let pcol = alin_column(nlat, nlon, 1, m, walin);
        for np1 in (mp2..=ndo_odd).step_by(2) {
            for k in 0..nt {
                for i in 1..=imm1 {
                    let pidx = (np1 - 1) * imid + (i - 1);
                    let aidx = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                    go[((i - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k] +=
                        a[aidx] * pcol[pidx] as f32;
                    go[((i - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k] +=
                        b[aidx] * pcol[pidx] as f32;
                }
            }
        }
    }

    if mdo != mmax {
        let mp2 = mmax + 1;
        if mp2 <= ndo_odd {
            let pcol = alin_column(nlat, nlon, 1, mdo, walin);
            for np1 in (mp2..=ndo_odd).step_by(2) {
                for k in 0..nt {
                    for i in 1..=imm1 {
                        let pidx = (np1 - 1) * imid + (i - 1);
                        let aidx = (((mmax - 1) * nlat) + (np1 - 1)) * nt + k;
                        go[((i - 1) * nlon + (2 * mmax - 2 - 1)) * nt + k] +=
                            a[aidx] * pcol[pidx] as f32;
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
                plane[i * nlon + j] = ge[(i * nlon + j) * nt + k];
            }
        }
        if nlon % 2 == 0 {
            for i in 0..ls {
                plane[i * nlon + (nlon - 1)] *= 2.0_f32;
            }
        }
        let mut packed = vec![0.0_f32; ls * nlon];
        for i in 0..ls {
            for j in 0..nlon {
                packed[j * ls + i] = plane[i * nlon + j];
            }
        }
        hrfftb_impl(
            ls,
            nlon,
            &mut packed,
            &wshsec[lzz1 + labc..lzz1 + labc + nlon + 15],
        )?;
        for i in 0..ls {
            for j in 0..nlon {
                ge[(i * nlon + j) * nt + k] = packed[j * ls + i];
            }
        }
    }

    let mut g = vec![0.0_f32; if isym == 0 { nlat } else { imid } * nlon * nt];
    let nlp1 = nlat;
    if isym == 0 {
        for k in 0..nt {
            for j in 0..nlon {
                for i in 0..imm1 {
                    let top = (i * nlon + j) * nt + k;
                    let bot = ((nlp1 - 1 - i) * nlon + j) * nt + k;
                    let goidx = ((imid + i) * nlon + j) * nt + k;
                    g[top] = 0.5_f32 * (ge[top] + ge[goidx]);
                    g[bot] = 0.5_f32 * (ge[top] - ge[goidx]);
                }
                if modl != 0 {
                    let idx = ((imid - 1) * nlon + j) * nt + k;
                    g[idx] = 0.5_f32 * ge[idx];
                }
            }
        }
    } else {
        for k in 0..nt {
            for j in 0..nlon {
                for i in 0..imid {
                    let idx = (i * nlon + j) * nt + k;
                    g[idx] = 0.5_f32 * ge[idx];
                }
            }
        }
    }

    Ok((g, 0))
}

/// Parallel synthesis of scalar fields on a regular grid using computed Legendre tables.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `isym`: Symmetry selector used by Legendre tables.
/// - `wshsec`: Workspace initialized by `shseci_impl` for regular-grid scalar synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn shsec_impl_parallel(
    a: &[f32],
    b: &[f32],
    nlat: usize,
    nt: usize,
    isym: usize,
    wshsec: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 2;
    let nlon = infer_nlon_from_wshsec(nlat, wshsec.len())
        .ok_or_else(|| PyValueError::new_err("failed to infer nlon from wshsec length"))?;
    if nlon < 4 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 4;
    if nt == 0 {
        return Ok((Vec::new(), ierror));
    }
    if a.len() != b.len() || a.len() != nlat * nlat * nt {
        return Err(PyValueError::new_err("a/b size mismatch"));
    }

    let mmax = nlat.min(nlon / 2 + 1);
    let imid = (nlat + 1) / 2;
    let lzz1 = 2 * nlat * imid;
    let labc = 3 * ((mmax - 2) * (nlat + nlat - mmax - 1)) / 2;
    ierror = 9;
    if wshsec.len() < lzz1 + labc + nlon + 15 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 10;
    let ls = if isym == 0 { nlat } else { imid };
    let nln = nt * ls * nlon;
    if lwork < nln + (ls * nlon).max(3 * nlat * imid) {
        return Ok((Vec::new(), ierror));
    }

    let mut ge = vec![0.0_f32; ls * nlon * nt];
    let mut go = vec![0.0_f32; imid * nlon * nt];
    let mdo = if 2 * mmax - 1 > nlon { mmax - 1 } else { mmax };
    let modl = nlat % 2;
    let mut imm1 = imid;
    if modl != 0 {
        imm1 = imid - 1;
    }

    let walin_f64: Vec<f64> = wshsec[..(lzz1 + labc)].iter().map(|&v| v as f64).collect();
    let walin = walin_f64.as_slice();

    let p0_even = alin_column(nlat, nlon, 2, 0, walin);
    for k in 0..nt {
        for np1 in (1..=nlat).step_by(2) {
            for i in 1..=imid {
                let pidx = (np1 - 1) * imid + (i - 1);
                let aidx = (np1 - 1) * nt + k;
                let geidx = ((i - 1) * nlon) * nt + k;
                ge[geidx] += a[aidx] * p0_even[pidx] as f32;
            }
        }
    }

    let ndo_even = if nlat % 2 == 0 { nlat - 1 } else { nlat };
    if mdo >= 2 {
        let ranges = mp1_ranges(2, mdo);
        let ge_add = ranges
            .par_iter()
            .map(|&(start, end)| {
                let mut ge_local = vec![0.0_f32; ls * nlon * nt];
                for mp1 in start..=end {
                    let m = mp1 - 1;
                    let pcol = alin_column(nlat, nlon, 2, m, walin);
                    for np1 in (mp1..=ndo_even).step_by(2) {
                        for k in 0..nt {
                            for i in 1..=imid {
                                let pidx = (np1 - 1) * imid + (i - 1);
                                let aidx = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                                ge_local[((i - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k] +=
                                    a[aidx] * pcol[pidx] as f32;
                                ge_local[((i - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k] +=
                                    b[aidx] * pcol[pidx] as f32;
                            }
                        }
                    }
                }
                ge_local
            })
            .reduce(
                || vec![0.0_f32; ls * nlon * nt],
                |mut acc, part| {
                    add_assign(&mut acc, &part);
                    acc
                },
            );
        add_assign(&mut ge, &ge_add);
    }

    if mdo != mmax && mmax <= ndo_even {
        let pcol = alin_column(nlat, nlon, 2, mdo, walin);
        for np1 in (mmax..=ndo_even).step_by(2) {
            for k in 0..nt {
                for i in 1..=imid {
                    let pidx = (np1 - 1) * imid + (i - 1);
                    let aidx = (((mmax - 1) * nlat) + (np1 - 1)) * nt + k;
                    ge[((i - 1) * nlon + (2 * mmax - 2 - 1)) * nt + k] +=
                        a[aidx] * pcol[pidx] as f32;
                }
            }
        }
    }

    let p0_odd = alin_column(nlat, nlon, 1, 0, walin);
    for k in 0..nt {
        for np1 in (2..=nlat).step_by(2) {
            for i in 1..=imm1 {
                let pidx = (np1 - 1) * imid + (i - 1);
                let aidx = (np1 - 1) * nt + k;
                let goidx = ((i - 1) * nlon) * nt + k;
                go[goidx] += a[aidx] * p0_odd[pidx] as f32;
            }
        }
    }

    let ndo_odd = if nlat % 2 != 0 { nlat - 1 } else { nlat };
    if mdo >= 2 {
        let ranges = mp1_ranges(2, mdo);
        let go_add = ranges
            .par_iter()
            .map(|&(start, end)| {
                let mut go_local = vec![0.0_f32; imid * nlon * nt];
                for mp1 in start..=end {
                    let mp2 = mp1 + 1;
                    let m = mp1 - 1;
                    let pcol = alin_column(nlat, nlon, 1, m, walin);
                    for np1 in (mp2..=ndo_odd).step_by(2) {
                        for k in 0..nt {
                            for i in 1..=imm1 {
                                let pidx = (np1 - 1) * imid + (i - 1);
                                let aidx = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                                go_local[((i - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k] +=
                                    a[aidx] * pcol[pidx] as f32;
                                go_local[((i - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k] +=
                                    b[aidx] * pcol[pidx] as f32;
                            }
                        }
                    }
                }
                go_local
            })
            .reduce(
                || vec![0.0_f32; imid * nlon * nt],
                |mut acc, part| {
                    add_assign(&mut acc, &part);
                    acc
                },
            );
        add_assign(&mut go, &go_add);
    }

    if mdo != mmax {
        let mp2 = mmax + 1;
        if mp2 <= ndo_odd {
            let pcol = alin_column(nlat, nlon, 1, mdo, walin);
            for np1 in (mp2..=ndo_odd).step_by(2) {
                for k in 0..nt {
                    for i in 1..=imm1 {
                        let pidx = (np1 - 1) * imid + (i - 1);
                        let aidx = (((mmax - 1) * nlat) + (np1 - 1)) * nt + k;
                        go[((i - 1) * nlon + (2 * mmax - 2 - 1)) * nt + k] +=
                            a[aidx] * pcol[pidx] as f32;
                    }
                }
            }
        }
    }

    for k in 0..nt {
        for j in 0..nlon {
            for i in 0..imm1 {
                let src = (i * nlon + j) * nt + k;
                let dst = ((imid + i) * nlon + j) * nt + k;
                ge[dst] = go[src];
            }
        }
    }

    let transformed_planes: Vec<Vec<f32>> = (0..nt)
        .into_par_iter()
        .map(|k| {
            let mut plane = vec![0.0_f32; ls * nlon];
            for i in 0..ls {
                for j in 0..nlon {
                    plane[i * nlon + j] = ge[(i * nlon + j) * nt + k];
                }
            }
            if nlon % 2 == 0 {
                for i in 0..ls {
                    plane[i * nlon + (nlon - 1)] *= 2.0_f32;
                }
            }
            let mut packed = vec![0.0_f32; ls * nlon];
            for i in 0..ls {
                for j in 0..nlon {
                    packed[j * ls + i] = plane[i * nlon + j];
                }
            }
            hrfftb_impl(
                ls,
                nlon,
                &mut packed,
                &wshsec[lzz1 + labc..lzz1 + labc + nlon + 15],
            )?;
            let mut plane_out = vec![0.0_f32; ls * nlon];
            for i in 0..ls {
                for j in 0..nlon {
                    plane_out[i * nlon + j] = packed[j * ls + i];
                }
            }
            Ok::<Vec<f32>, PyErr>(plane_out)
        })
        .collect::<PyResult<Vec<_>>>()?;

    for (k, plane) in transformed_planes.into_iter().enumerate() {
        for i in 0..ls {
            for j in 0..nlon {
                ge[(i * nlon + j) * nt + k] = plane[i * nlon + j];
            }
        }
    }

    let mut g = vec![0.0_f32; if isym == 0 { nlat } else { imid } * nlon * nt];
    let nlp1 = nlat;
    if isym == 0 {
        for k in 0..nt {
            for j in 0..nlon {
                for i in 0..imm1 {
                    let top = (i * nlon + j) * nt + k;
                    let bot = ((nlp1 - 1 - i) * nlon + j) * nt + k;
                    let goidx = ((imid + i) * nlon + j) * nt + k;
                    g[top] = 0.5_f32 * (ge[top] + ge[goidx]);
                    g[bot] = 0.5_f32 * (ge[top] - ge[goidx]);
                }
                if modl != 0 {
                    let idx = ((imid - 1) * nlon + j) * nt + k;
                    g[idx] = 0.5_f32 * ge[idx];
                }
            }
        }
    } else {
        for k in 0..nt {
            for j in 0..nlon {
                for i in 0..imid {
                    let idx = (i * nlon + j) * nt + k;
                    g[idx] = 0.5_f32 * ge[idx];
                }
            }
        }
    }

    Ok((g, 0))
}

#[pyfunction]
/// Python wrapper for `shsec_impl` that returns NumPy arrays.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `wshsec`: Workspace initialized by `shseci_impl` for regular-grid scalar synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn shsec<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    wshsec: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err("shsec expects rank-2 or rank-3 a/b"));
    }
    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let abuf = a.as_slice()?.to_vec();
    let bbuf = b.as_slice()?.to_vec();
    let wbuf = wshsec.as_slice()?.to_vec();
    let wlen = wbuf.len();
    let (g, ierror) = py
        .detach(|| {
            shsec_impl_parallel(&abuf, &bbuf, nlat, nt, 0, &wbuf, lwork)
                .map_err(|err| err.to_string())
        })
        .map_err(PyValueError::new_err)?;
    let nlon = infer_nlon_from_wshsec(nlat, wlen).unwrap_or(0);
    let gshape = if ashape.len() == 2 {
        vec![nlat, nlon]
    } else {
        vec![nlat, nlon, nt]
    };
    build_shsec_output(py, &gshape, g, ierror)
}

#[pyfunction]
/// Python wrapper for `shsec_impl_parallel` that releases the GIL during synthesis.
///
/// # Parameters
/// - `a`: Cosine spectral coefficients in scalar layout.
/// - `b`: Sine spectral coefficients in scalar layout.
/// - `wshsec`: Workspace initialized by `shseci_impl` for regular-grid scalar synthesis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn shsec_nogil<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    wshsec: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err(
            "shsec_nogil expects rank-2 or rank-3 a/b",
        ));
    }
    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let abuf = a.as_slice()?.to_vec();
    let bbuf = b.as_slice()?.to_vec();
    let wbuf = wshsec.as_slice()?.to_vec();
    let wbuf_len = wbuf.len();
    let result = py.detach(move || shsec_impl_parallel(&abuf, &bbuf, nlat, nt, 0, &wbuf, lwork));
    let (g, ierror) = result?;
    let nlon = infer_nlon_from_wshsec(nlat, wbuf_len).unwrap_or(0);
    let gshape = if ashape.len() == 2 {
        vec![nlat, nlon]
    } else {
        vec![nlat, nlon, nt]
    };
    build_shsec_output(py, &gshape, g, ierror)
}
