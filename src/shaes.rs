use crate::hrfftf::fourier_analysis_real;
use crate::hrffti::hrffti_impl;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn fourier_analysis_real_debug<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Py<PyAny>> {
    let shape = data.shape().to_vec();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(
            "fourier_analysis_real_debug expects rank-2 input",
        ));
    }
    let rows = shape[0];
    let nlon = shape[1];
    let mut buf: Vec<f32> = data.as_array().iter().copied().collect();
    let whrfft = hrffti_impl(nlon as i32);
    fourier_analysis_real(rows, nlon, &mut buf, &whrfft)?;
    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), buf)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py).into_any().unbind())
}

pub fn shaes_impl(
    g: &[f32],
    nlat: usize,
    nlon: usize,
    nt: usize,
    wshaes: &[f32],
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
    let isym = 0usize;
    ierror = 4;
    if nt == 0 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    if g.len() != nlat * nlon * nt {
        return Err(PyValueError::new_err("g size mismatch"));
    }
    let mmax = nlat.min(nlon / 2 + 1);
    let imid = (nlat + 1) / 2;
    let idz = (mmax * (nlat + nlat - mmax + 1)) / 2;
    let lzimn = idz * imid;
    ierror = 9;
    if wshaes.len() < lzimn + nlon + 15 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 10;
    if lwork < (nt + 1) * nlat * nlon {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let mut a = vec![0.0_f32; nlat * nlat * nt];
    let mut b = vec![0.0_f32; nlat * nlat * nt];
    let mut ge = vec![0.0_f32; nlat * nlon * nt];
    let mut go = vec![0.0_f32; nlat * nlon * nt];
    let z = &wshaes[..lzimn];
    let whrfft = &wshaes[lzimn..lzimn + nlon + 15];

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
                let ge_idx = top;
                let go_idx = top;
                ge[ge_idx] = tsn * (g[top] + g[bot]);
                go[go_idx] = tsn * (g[top] - g[bot]);
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
        let ge_rows = if isym == 0 { imid } else { nlat };
        let mut plane_ge = vec![0.0_f32; ge_rows * nlon];
        for i in 0..ge_rows {
            for j in 0..nlon {
                plane_ge[i * nlon + j] = ge[(i * nlon + j) * nt + k];
            }
        }
        fourier_analysis_real(ge_rows, nlon, &mut plane_ge, whrfft)?;
        if nlon % 2 == 0 {
            for i in 0..ge_rows {
                plane_ge[i * nlon + (nlon - 1)] *= 0.5_f32;
            }
        }
        for i in 0..ge_rows {
            for j in 0..nlon {
                ge[(i * nlon + j) * nt + k] = plane_ge[i * nlon + j];
            }
        }

        let go_rows = if isym == 0 { imm1 } else { 0 };
        if go_rows > 0 {
            let mut plane_go = vec![0.0_f32; go_rows * nlon];
            for i in 0..go_rows {
                for j in 0..nlon {
                    plane_go[i * nlon + j] = go[(i * nlon + j) * nt + k];
                }
            }
            fourier_analysis_real(go_rows, nlon, &mut plane_go, whrfft)?;
            if nlon % 2 == 0 {
                for i in 0..go_rows {
                    plane_go[i * nlon + (nlon - 1)] *= 0.5_f32;
                }
            }
            for i in 0..go_rows {
                for j in 0..nlon {
                    go[(i * nlon + j) * nt + k] = plane_go[i * nlon + j];
                }
            }
        }
    }

    for k in 0..nt {
        for np1 in (1..=nlat).step_by(2) {
            for i in 1..=imid {
                let zidx = (np1 - 1) + (i - 1) * idz;
                let geidx = ((i - 1) * nlon) * nt + k;
                let aidx = (np1 - 1) * nt + k;
                a[aidx] += z[zidx] * ge[geidx];
            }
        }
    }

    let mut ndo = if nlat % 2 == 0 { nlat - 1 } else { nlat };
    for mp1 in 2..=mdo {
        let m = mp1 - 1;
        let mb = m * (nlat - 1) - (m * (m - 1)) / 2;
        for k in 0..nt {
            for i in 1..=imid {
                for np1 in (mp1..=ndo).step_by(2) {
                    let zidx = (np1 + mb - 1) + (i - 1) * idz;
                    let aidx = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                    let ge_cos = ((i - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k;
                    let ge_sin = ((i - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k;
                    a[aidx] += z[zidx] * ge[ge_cos];
                    b[aidx] += z[zidx] * ge[ge_sin];
                }
            }
        }
    }

    if mdo != mmax && mmax <= ndo {
        let mb = mdo * (nlat - 1) - (mdo * (mdo - 1)) / 2;
        for k in 0..nt {
            for i in 1..=imid {
                for np1 in (mmax..=ndo).step_by(2) {
                    let zidx = (np1 + mb - 1) + (i - 1) * idz;
                    let aidx = (((mmax - 1) * nlat) + (np1 - 1)) * nt + k;
                    let ge_cos = ((i - 1) * nlon + (2 * mmax - 2 - 1)) * nt + k;
                    a[aidx] += z[zidx] * ge[ge_cos];
                }
            }
        }
    }

    for k in 0..nt {
        for np1 in (2..=nlat).step_by(2) {
            for i in 1..=imm1 {
                let zidx = (np1 - 1) + (i - 1) * idz;
                let goidx = ((i - 1) * nlon) * nt + k;
                let aidx = (np1 - 1) * nt + k;
                a[aidx] += z[zidx] * go[goidx];
            }
        }
    }

    ndo = if nlat % 2 != 0 { nlat - 1 } else { nlat };
    for mp1 in 2..=mdo {
        let m = mp1 - 1;
        let mp2 = mp1 + 1;
        let mb = m * (nlat - 1) - (m * (m - 1)) / 2;
        for k in 0..nt {
            for i in 1..=imm1 {
                for np1 in (mp2..=ndo).step_by(2) {
                    let zidx = (np1 + mb - 1) + (i - 1) * idz;
                    let aidx = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                    let go_cos = ((i - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k;
                    let go_sin = ((i - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k;
                    a[aidx] += z[zidx] * go[go_cos];
                    b[aidx] += z[zidx] * go[go_sin];
                }
            }
        }
    }

    if mdo != mmax {
        let mp2 = mmax + 1;
        if mp2 <= ndo {
            let mb = mdo * (nlat - 1) - (mdo * (mdo - 1)) / 2;
            for k in 0..nt {
                for i in 1..=imm1 {
                    for np1 in (mp2..=ndo).step_by(2) {
                        let zidx = (np1 + mb - 1) + (i - 1) * idz;
                        let aidx = (((mmax - 1) * nlat) + (np1 - 1)) * nt + k;
                        let go_cos = ((i - 1) * nlon + (2 * mmax - 2 - 1)) * nt + k;
                        a[aidx] += z[zidx] * go[go_cos];
                    }
                }
            }
        }
    }

    Ok((a, b, 0))
}

pub fn shaes_impl_parallel(
    g: &[f32],
    nlat: usize,
    nlon: usize,
    nt: usize,
    wshaes: &[f32],
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
    let idz = (mmax * (nlat + nlat - mmax + 1)) / 2;
    let lzimn = idz * imid;
    ierror = 9;
    if wshaes.len() < lzimn + nlon + 15 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 10;
    if lwork < (nt + 1) * nlat * nlon {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let z = &wshaes[..lzimn];
    let whrfft = &wshaes[lzimn..lzimn + nlon + 15];
    let mdo = if 2 * mmax - 1 > nlon { mmax - 1 } else { mmax };
    let nlp1 = nlat + 1;
    let tsn = 2.0_f32 / (nlon as f32);
    let modl = nlat % 2;
    let imm1 = if modl != 0 { imid - 1 } else { imid };
    let coeff_len = nlat * nlat;

    let coeffs = (0..nt)
        .into_par_iter()
        .map(|k| {
            let mut ge = vec![0.0_f32; nlat * nlon];
            let mut go = vec![0.0_f32; nlat * nlon];
            for i in 1..=imm1 {
                for j in 1..=nlon {
                    let top = ((i - 1) * nlon + (j - 1)) * nt + k;
                    let bot = (((nlp1 - i) - 1) * nlon + (j - 1)) * nt + k;
                    let local = (i - 1) * nlon + (j - 1);
                    ge[local] = tsn * (g[top] + g[bot]);
                    go[local] = tsn * (g[top] - g[bot]);
                }
            }
            if modl != 0 {
                for j in 1..=nlon {
                    let src = ((imid - 1) * nlon + (j - 1)) * nt + k;
                    let dst = (imid - 1) * nlon + (j - 1);
                    ge[dst] = tsn * g[src];
                }
            }

            let ge_rows = imid;
            let mut plane_ge = vec![0.0_f32; ge_rows * nlon];
            for i in 0..ge_rows {
                for j in 0..nlon {
                    plane_ge[i * nlon + j] = ge[i * nlon + j];
                }
            }
            fourier_analysis_real(ge_rows, nlon, &mut plane_ge, whrfft)?;
            if nlon % 2 == 0 {
                for i in 0..ge_rows {
                    plane_ge[i * nlon + (nlon - 1)] *= 0.5_f32;
                }
            }
            for i in 0..ge_rows {
                for j in 0..nlon {
                    ge[i * nlon + j] = plane_ge[i * nlon + j];
                }
            }

            if imm1 > 0 {
                let mut plane_go = vec![0.0_f32; imm1 * nlon];
                for i in 0..imm1 {
                    for j in 0..nlon {
                        plane_go[i * nlon + j] = go[i * nlon + j];
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
                        go[i * nlon + j] = plane_go[i * nlon + j];
                    }
                }
            }

            let mut ak = vec![0.0_f32; coeff_len];
            let mut bk = vec![0.0_f32; coeff_len];

            for np1 in (1..=nlat).step_by(2) {
                for i in 1..=imid {
                    let zidx = (np1 - 1) + (i - 1) * idz;
                    let geidx = (i - 1) * nlon;
                    ak[np1 - 1] += z[zidx] * ge[geidx];
                }
            }

            let mut ndo = if nlat % 2 == 0 { nlat - 1 } else { nlat };
            for mp1 in 2..=mdo {
                let m = mp1 - 1;
                let mb = m * (nlat - 1) - (m * (m - 1)) / 2;
                for i in 1..=imid {
                    for np1 in (mp1..=ndo).step_by(2) {
                        let zidx = (np1 + mb - 1) + (i - 1) * idz;
                        let aidx = ((mp1 - 1) * nlat) + (np1 - 1);
                        let ge_cos = (i - 1) * nlon + (2 * mp1 - 2 - 1);
                        let ge_sin = (i - 1) * nlon + (2 * mp1 - 1 - 1);
                        ak[aidx] += z[zidx] * ge[ge_cos];
                        bk[aidx] += z[zidx] * ge[ge_sin];
                    }
                }
            }

            if mdo != mmax && mmax <= ndo {
                let mb = mdo * (nlat - 1) - (mdo * (mdo - 1)) / 2;
                for i in 1..=imid {
                    for np1 in (mmax..=ndo).step_by(2) {
                        let zidx = (np1 + mb - 1) + (i - 1) * idz;
                        let aidx = ((mmax - 1) * nlat) + (np1 - 1);
                        let ge_cos = (i - 1) * nlon + (2 * mmax - 2 - 1);
                        ak[aidx] += z[zidx] * ge[ge_cos];
                    }
                }
            }

            for np1 in (2..=nlat).step_by(2) {
                for i in 1..=imm1 {
                    let zidx = (np1 - 1) + (i - 1) * idz;
                    let goidx = (i - 1) * nlon;
                    ak[np1 - 1] += z[zidx] * go[goidx];
                }
            }

            ndo = if nlat % 2 != 0 { nlat - 1 } else { nlat };
            for mp1 in 2..=mdo {
                let m = mp1 - 1;
                let mp2 = mp1 + 1;
                let mb = m * (nlat - 1) - (m * (m - 1)) / 2;
                for i in 1..=imm1 {
                    for np1 in (mp2..=ndo).step_by(2) {
                        let zidx = (np1 + mb - 1) + (i - 1) * idz;
                        let aidx = ((mp1 - 1) * nlat) + (np1 - 1);
                        let go_cos = (i - 1) * nlon + (2 * mp1 - 2 - 1);
                        let go_sin = (i - 1) * nlon + (2 * mp1 - 1 - 1);
                        ak[aidx] += z[zidx] * go[go_cos];
                        bk[aidx] += z[zidx] * go[go_sin];
                    }
                }
            }

            if mdo != mmax {
                let mp2 = mmax + 1;
                if mp2 <= ndo {
                    let mb = mdo * (nlat - 1) - (mdo * (mdo - 1)) / 2;
                    for i in 1..=imm1 {
                        for np1 in (mp2..=ndo).step_by(2) {
                            let zidx = (np1 + mb - 1) + (i - 1) * idz;
                            let aidx = ((mmax - 1) * nlat) + (np1 - 1);
                            let go_cos = (i - 1) * nlon + (2 * mmax - 2 - 1);
                            ak[aidx] += z[zidx] * go[go_cos];
                        }
                    }
                }
            }
            Ok::<(Vec<f32>, Vec<f32>), PyErr>((ak, bk))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let mut a = vec![0.0_f32; nlat * nlat * nt];
    let mut b = vec![0.0_f32; nlat * nlat * nt];
    for (k, (ak, bk)) in coeffs.into_iter().enumerate() {
        for idx in 0..coeff_len {
            a[idx * nt + k] = ak[idx];
            b[idx * nt + k] = bk[idx];
        }
    }
    Ok((a, b, 0))
}

#[pyfunction]
pub fn shaes<'py>(
    py: Python<'py>,
    g: PyReadonlyArrayDyn<'py, f32>,
    wshaes: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    let shape = g.shape().to_vec();
    if shape.len() != 2 && shape.len() != 3 {
        return Err(PyValueError::new_err("shaes expects rank-2 or rank-3 g"));
    }
    let nlat = shape[0];
    let nlon = shape[1];
    let nt = if shape.len() == 2 { 1 } else { shape[2] };
    let gbuf = g.as_slice()?.to_vec();
    let wbuf = wshaes.as_slice()?.to_vec();
    let (a, b, ierror) = py
        .detach(|| {
            shaes_impl_parallel(&gbuf, nlat, nlon, nt, &wbuf, lwork).map_err(|err| err.to_string())
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
