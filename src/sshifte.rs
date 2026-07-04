use crate::hrfftb::hrfftb_kernel_only_impl;
use crate::hrfftf::hrfftf_impl;
use crate::hrffti::hrffti_impl;
use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn required_lsav(nlon: usize, nlat: usize) -> usize {
    2 * (2 * nlat + nlon + 16)
}

fn required_lwork(nlon: usize, nlat: usize) -> usize {
    if nlon.is_multiple_of(2) {
        2 * nlon * (nlat + 1)
    } else {
        nlon * (5 * nlat + 1)
    }
}

fn validate_ioff(ioff: i32) -> bool {
    ioff == 0 || ioff == 1
}

fn shifti_block(n: usize, dp: f64) -> Vec<f32> {
    let mut out = vec![0.0_f32; 2 * n + 16];
    let n2 = (n + 1) / 2;
    for k in 2..=n2 {
        out[k - 1] = (((k - 1) as f64) * dp).sin() as f32;
        out[k + n2 - 1] = (((k - 1) as f64) * dp).cos() as f32;
    }
    let fft = hrffti_impl(n as i32);
    let start = n + 1;
    let end = (start + fft.len()).min(out.len());
    out[start..end].copy_from_slice(&fft[..(end - start)]);
    out
}

fn shift_rows_real(data: &[f32], m: usize, n: usize, wsav: &[f32]) -> PyResult<Vec<f32>> {
    if data.len() != m * n {
        return Err(PyValueError::new_err(
            "shift_rows_real: data length mismatch",
        ));
    }
    if wsav.len() < 2 * n + 16 {
        return Err(PyValueError::new_err(
            "shift_rows_real: wsav length too small",
        ));
    }

    let n2 = (n + 1) / 2;
    let mut packed = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            packed[col * m + row] = data[row * n + col];
        }
    }

    hrfftf_impl(m, n, &mut packed, &wsav[(n + 1)..])?;

    for row in 0..m {
        for k in 2..=n2 {
            let jc = (2 * k - 3) * m + row;
            let js = (2 * k - 2) * m + row;
            let r2km2 = packed[jc];
            let r2km1 = packed[js];
            let sinv = wsav[k - 1];
            let cosv = wsav[n2 + k - 1];
            packed[jc] = r2km2 * cosv - r2km1 * sinv;
            packed[js] = r2km2 * sinv + r2km1 * cosv;
        }
    }

    hrfftb_kernel_only_impl(m, n, &mut packed, &wsav[(n + 1)..])?;

    let scale = 1.0_f32 / (n as f32);
    let mut out = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            out[row * n + col] = packed[col * m + row] * scale;
        }
    }

    Ok(out)
}

fn idx_geo(nlat: usize, j: usize, i: usize) -> usize {
    j * nlat + i
}

fn idx_reg(nlatp1: usize, j: usize, i: usize) -> usize {
    j * nlatp1 + i
}

fn idx_row(cols: usize, row: usize, col: usize) -> usize {
    row * cols + col
}

fn shftoff_impl(nlon: usize, nlat: usize, goff: &[f32], wsav: &[f32]) -> PyResult<Vec<f32>> {
    let n2 = (nlon + 1) / 2;
    let nlat2 = nlat + nlat;
    let nlatp1 = nlat + 1;
    let mut greg = vec![0.0_f32; nlon * nlatp1];
    let lat_wsav = &wsav[..(4 * nlat + 16)];
    let lon_offset = 4 * nlat + 16;
    let lon_wsav = &wsav[lon_offset..];

    if !nlon.is_multiple_of(2) {
        let mut rlon = vec![0.0_f32; nlat * nlon];
        for i in 0..nlat {
            for j in 0..nlon {
                rlon[idx_row(nlon, i, j)] = goff[idx_geo(nlat, j, i)];
            }
        }
        let rlon = shift_rows_real(&rlon, nlat, nlon, lon_wsav)?;

        let mut rlat = vec![0.0_f32; nlon * nlat2];
        for j in 0..(n2 - 1) {
            let js = j + n2;
            for i in 0..nlat {
                rlat[idx_row(nlat2, j, i)] = goff[idx_geo(nlat, j, i)];
                rlat[idx_row(nlat2, j, nlat + i)] = rlon[idx_row(nlon, nlat - 1 - i, js)];
            }
        }
        for j in (n2 - 1)..nlon {
            let js = j - (n2 - 1);
            for i in 0..nlat {
                rlat[idx_row(nlat2, j, i)] = goff[idx_geo(nlat, j, i)];
                rlat[idx_row(nlat2, j, nlat + i)] = rlon[idx_row(nlon, nlat - 1 - i, js)];
            }
        }

        let rlat = shift_rows_real(&rlat, nlon, nlat2, lat_wsav)?;

        let mut gnorth = 0.0_f32;
        let mut gsouth = 0.0_f32;
        for j in 0..nlon {
            gnorth += rlat[idx_row(nlat2, j, 0)];
            gsouth += rlat[idx_row(nlat2, j, nlat)];
            for i in 1..nlat {
                greg[idx_reg(nlatp1, j, i)] = rlat[idx_row(nlat2, j, i)];
            }
        }
        gnorth /= nlon as f32;
        gsouth /= nlon as f32;

        for j in 0..nlon {
            greg[idx_reg(nlatp1, j, 0)] = gnorth;
            greg[idx_reg(nlatp1, j, nlat)] = gsouth;
        }
    } else {
        let mut rlat = vec![0.0_f32; n2 * nlat2];
        for j in 0..n2 {
            let js = n2 + j;
            for i in 0..nlat {
                rlat[idx_row(nlat2, j, i)] = goff[idx_geo(nlat, j, i)];
                rlat[idx_row(nlat2, j, nlat + i)] = goff[idx_geo(nlat, js, nlat - 1 - i)];
            }
        }

        let rlat = shift_rows_real(&rlat, n2, nlat2, lat_wsav)?;

        let mut gnorth = 0.0_f32;
        let mut gsouth = 0.0_f32;
        for j in 0..n2 {
            let js = n2 + j;
            gnorth += rlat[idx_row(nlat2, j, 0)];
            gsouth += rlat[idx_row(nlat2, j, nlat)];
            for i in 1..nlat {
                greg[idx_reg(nlatp1, j, i)] = rlat[idx_row(nlat2, j, i)];
                greg[idx_reg(nlatp1, js, i)] = rlat[idx_row(nlat2, j, nlat2 - i)];
            }
        }
        gnorth /= n2 as f32;
        gsouth /= n2 as f32;

        for j in 0..nlon {
            greg[idx_reg(nlatp1, j, 0)] = gnorth;
            greg[idx_reg(nlatp1, j, nlat)] = gsouth;
        }
    }

    let mut rlon = vec![0.0_f32; nlat * nlon];
    for j in 0..nlon {
        for i in 0..nlat {
            rlon[idx_row(nlon, i, j)] = greg[idx_reg(nlatp1, j, i)];
        }
    }

    let rlon = shift_rows_real(&rlon, nlat, nlon, lon_wsav)?;

    for j in 0..nlon {
        for i in 1..nlat {
            greg[idx_reg(nlatp1, j, i)] = rlon[idx_row(nlon, i, j)];
        }
    }

    Ok(greg)
}

fn shftreg_impl(nlon: usize, nlat: usize, greg: &[f32], wsav: &[f32]) -> PyResult<Vec<f32>> {
    let n2 = (nlon + 1) / 2;
    let nlat2 = nlat + nlat;
    let nlatp1 = nlat + 1;
    let mut goff = vec![0.0_f32; nlon * nlat];
    let mut rlon_full = vec![0.0_f32; nlatp1 * nlon];
    let mut even_final_workspace: Option<Vec<f32>> = None;
    let lat_wsav = &wsav[..(4 * nlat + 16)];
    let lon_offset = 4 * nlat + 16;
    let lon_wsav = &wsav[lon_offset..];

    if !nlon.is_multiple_of(2) {
        for i in 0..nlatp1 {
            for j in 0..nlon {
                rlon_full[idx_row(nlon, i, j)] = greg[idx_reg(nlatp1, j, i)];
            }
        }

        rlon_full = shift_rows_real(&rlon_full, nlatp1, nlon, lon_wsav)?;

        let mut rlat = vec![0.0_f32; nlon * nlat2];

        for j in 0..n2 {
            let js = j + n2 - 1;
            rlat[idx_row(nlat2, j, 0)] = greg[idx_reg(nlatp1, j, 0)];
            for i in 1..nlat {
                rlat[idx_row(nlat2, j, i)] = greg[idx_reg(nlatp1, j, i)];
                rlat[idx_row(nlat2, j, nlat + i)] = rlon_full[idx_row(nlon, nlat - i, js)];
            }
            rlat[idx_row(nlat2, j, nlat)] = greg[idx_reg(nlatp1, j, nlat)];
        }
        for j in n2..nlon {
            let js = j - n2;
            rlat[idx_row(nlat2, j, 0)] = greg[idx_reg(nlatp1, j, 0)];
            for i in 1..nlat {
                rlat[idx_row(nlat2, j, i)] = greg[idx_reg(nlatp1, j, i)];
                rlat[idx_row(nlat2, j, nlat + i)] = rlon_full[idx_row(nlon, nlat - i, js)];
            }
            rlat[idx_row(nlat2, j, nlat)] = greg[idx_reg(nlatp1, j, nlat)];
        }

        let rlat = shift_rows_real(&rlat, nlon, nlat2, lat_wsav)?;
        for j in 0..nlon {
            for i in 0..nlat {
                goff[idx_geo(nlat, j, i)] = rlat[idx_row(nlat2, j, i)];
            }
        }
    } else {
        let mut rlat = vec![0.0_f32; n2 * nlat2];
        for j in 0..n2 {
            let js = n2 + j;
            rlat[idx_row(nlat2, j, 0)] = greg[idx_reg(nlatp1, j, 0)];
            for i in 1..nlat {
                rlat[idx_row(nlat2, j, i)] = greg[idx_reg(nlatp1, j, i)];
                rlat[idx_row(nlat2, j, nlat + i)] = greg[idx_reg(nlatp1, js, nlat - i)];
            }
            rlat[idx_row(nlat2, j, nlat)] = greg[idx_reg(nlatp1, j, nlat)];
        }

        let rlat = shift_rows_real(&rlat, n2, nlat2, lat_wsav)?;
        let mut workspace = vec![0.0_f32; nlatp1 * nlon];
        for i in 0..nlat2 {
            for j in 0..n2 {
                workspace[j + n2 * i] = rlat[idx_row(nlat2, j, i)];
            }
        }

        let mut dbg_points: Vec<(&str, f32)> = vec![
            ("rlat_0_0", rlat[idx_row(nlat2, 0, 0)]),
            ("rlat_1_0", rlat[idx_row(nlat2, 1.min(n2 - 1), 0)]),
            ("rlat_0_1", rlat[idx_row(nlat2, 0, 1.min(nlat2 - 1))]),
            ("workspace_0_0", workspace[0]),
        ];

        if workspace.len() > 1 {
            dbg_points.push(("workspace_0_1", workspace[1]));
        }
        if n2 < workspace.len() {
            dbg_points.push(("workspace_1_0", workspace[n2]));
        }
        if workspace.len() > 72 {
            dbg_points.push(("workspace_72", workspace[72]));
        }
        if workspace.len() > 73 {
            dbg_points.push(("workspace_73", workspace[73]));
        }
        if workspace.len() > 144 {
            dbg_points.push(("workspace_144", workspace[144]));
        }
        if workspace.len() > 145 {
            dbg_points.push(("workspace_145", workspace[145]));
        }

        even_final_workspace = Some(workspace);
        for j in 0..n2 {
            let js = n2 + j;
            for i in 0..nlat {
                goff[idx_geo(nlat, j, i)] = rlat[idx_row(nlat2, j, i)];
                goff[idx_geo(nlat, js, i)] = rlat[idx_row(nlat2, j, nlat2 - 1 - i)];
            }
        }

        let mut goff_as_rlon = vec![0.0_f32; nlat * nlon];
        for j in 0..nlon {
            for i in 0..nlat {
                goff_as_rlon[idx_row(nlon, i, j)] = goff[idx_geo(nlat, j, i)];
            }
        }
    }

    if let Some(workspace) = even_final_workspace {
        for j in 0..nlon {
            for i in 0..nlatp1 {
                rlon_full[idx_row(nlon, i, j)] = workspace[i + nlatp1 * j];
            }
        }
    }
    for j in 0..nlon {
        for i in 0..nlat {
            rlon_full[idx_row(nlon, i, j)] = goff[idx_geo(nlat, j, i)];
        }
    }

    rlon_full = shift_rows_real(&rlon_full, nlatp1, nlon, lon_wsav)?;

    for j in 0..nlon {
        for i in 0..nlat {
            goff[idx_geo(nlat, j, i)] = rlon_full[idx_row(nlon, i, j)];
        }
    }

    Ok(goff)
}

/// Core Rust implementation of `sshifti`.
///
/// # Parameters
/// - `ioff`: Parameter `ioff` passed through to the routine.
/// - `nlon`: Number of longitudes in the grid.
/// - `nlat`: Number of latitudes in the grid.
/// - `lsav`: Parameter `lsav` passed through to the routine.
///
/// # Returns
/// A tuple containing the workspace/result vector and the error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn sshifti_impl(ioff: i32, nlon: i32, nlat: i32, lsav: i32) -> (Vec<f32>, i32) {
    let mut ier = 1;
    if !validate_ioff(ioff) {
        return (Vec::new(), ier);
    }
    ier = 2;
    if nlon < 4 {
        return (Vec::new(), ier);
    }
    ier = 3;
    if nlat < 3 {
        return (Vec::new(), ier);
    }
    ier = 4;
    let nlon_u = usize::try_from(nlon).unwrap_or(0);
    let nlat_u = usize::try_from(nlat).unwrap_or(0);
    if lsav < required_lsav(nlon_u, nlat_u) as i32 {
        return (Vec::new(), ier);
    }

    let mut wsav = vec![0.0_f32; usize::try_from(lsav).unwrap_or(0)];
    let dlat = std::f64::consts::PI / (nlat as f64);
    let dlon = 2.0_f64 * std::f64::consts::PI / (nlon as f64);
    let dp_lat = if ioff == 0 {
        -0.5_f64 * dlat
    } else {
        0.5_f64 * dlat
    };
    let dp_lon = if ioff == 0 {
        -0.5_f64 * dlon
    } else {
        0.5_f64 * dlon
    };
    let lat_block = shifti_block(2 * nlat_u, dp_lat);
    let lon_block = shifti_block(nlon_u, dp_lon);
    let first_len = lat_block.len().min(wsav.len());
    wsav[..first_len].copy_from_slice(&lat_block[..first_len]);
    let isav = 4 * nlat_u + 16;
    if isav < wsav.len() {
        let second_len = lon_block.len().min(wsav.len() - isav);
        wsav[isav..isav + second_len].copy_from_slice(&lon_block[..second_len]);
    }
    (wsav, 0)
}

/// Core Rust implementation of `sshifte`.
///
/// # Parameters
/// - `ioff`: Parameter `ioff` passed through to the routine.
/// - `input`: Parameter `input` passed through to the routine.
/// - `nlon`: Number of longitudes in the grid.
/// - `nlat`: Number of latitudes in the grid.
/// - `wsav`: Parameter `wsav` passed through to the routine.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn sshifte_impl(
    ioff: i32,
    input: &[f32],
    nlon: usize,
    nlat: usize,
    wsav: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, i32)> {
    let mut ier = 1;
    if !validate_ioff(ioff) {
        return Ok((Vec::new(), ier));
    }
    ier = 2;
    if nlon < 4 {
        return Ok((Vec::new(), ier));
    }
    ier = 3;
    if nlat < 3 {
        return Ok((Vec::new(), ier));
    }
    ier = 4;
    if wsav.len() < required_lsav(nlon, nlat) {
        return Ok((Vec::new(), ier));
    }
    ier = 5;
    if lwork < required_lwork(nlon, nlat) {
        return Ok((Vec::new(), ier));
    }

    if ioff == 0 {
        if input.len() != nlon * nlat {
            return Err(PyValueError::new_err(
                "sshifte ioff=0 expects input shape (nlon, nlat)",
            ));
        }
        Ok((shftoff_impl(nlon, nlat, input, wsav)?, 0))
    } else {
        if input.len() != nlon * (nlat + 1) {
            return Err(PyValueError::new_err(
                "sshifte ioff=1 expects input shape (nlon, nlat+1)",
            ));
        }
        Ok((shftreg_impl(nlon, nlat, input, wsav)?, 0))
    }
}

#[pyfunction]
/// Rust entry point for `sshifti`.
///
/// # Parameters
/// - `ioff`: Parameter `ioff` passed through to the routine.
/// - `nlon`: Number of longitudes in the grid.
/// - `nlat`: Number of latitudes in the grid.
/// - `lsav`: Parameter `lsav` passed through to the routine.
///
/// # Returns
/// A one-dimensional NumPy workspace array together with a error code.
pub fn sshifti<'py>(
    py: Python<'py>,
    ioff: i32,
    nlon: i32,
    nlat: i32,
    lsav: i32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, i32)> {
    let (wsav, ier) = sshifti_impl(ioff, nlon, nlat, lsav);
    Ok((PyArray1::from_vec(py, wsav).to_owned(), ier))
}

#[pyfunction]
#[pyo3(signature = (data, wsav, lwork, ioff=0))]
/// Rust entry point for `sshifte`.
///
/// # Parameters
/// - `data`: Rank-2 real array analyzed by the internal Fourier kernel.
/// - `wsav`: Parameter `wsav` passed through to the routine.
/// - `lwork`: Length of the caller-provided work array.
/// - `ioff`: Parameter `ioff` passed through to the routine.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn sshifte<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f32>,
    wsav: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
    ioff: i32,
) -> PyResult<(Py<PyAny>, i32)> {
    let shape = data.shape().to_vec();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("sshifte expects a rank-2 array"));
    }
    let nlon = shape[0];
    let nlat = if ioff == 0 {
        shape[1]
    } else {
        shape[1].checked_sub(1).ok_or_else(|| {
            PyValueError::new_err("sshifte ioff=1 expects second dimension at least 2")
        })?
    };
    let input = data.as_array().as_standard_layout().to_owned();
    let wsav_arr = wsav.as_array().as_standard_layout().to_owned();
    let (shifted, ier) = sshifte_impl(
        ioff,
        input
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("data is not standard-layout after copy"))?,
        nlon,
        nlat,
        wsav_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("wsav is not standard-layout after copy"))?,
        lwork,
    )?;
    let out_shape = if ioff == 0 {
        vec![nlon, nlat + 1]
    } else {
        vec![nlon, nlat]
    };
    let out = ArrayD::from_shape_vec(IxDyn(&out_shape), shifted)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok((out.into_pyarray(py).into_any().unbind(), ier))
}
