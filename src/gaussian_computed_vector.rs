use crate::gaqd::gaqd_impl;
use crate::hrffti::hrffti_impl;
use crate::sphcom_vector::{dvbk, dvbt, dvtk, dvtt, dwbk, dwbt, dwtk, dwtt, rabcv1, rabcw1};

fn pack_tables(
    base0: &[f32],
    base1: &[f32],
    a: &[f64],
    b: &[f64],
    c: &[f64],
    lim: usize,
) -> Vec<f32> {
    let labc = a.len().saturating_sub(1);
    let mut out = vec![0.0_f32; 2 * lim + 3 * labc];
    out[..lim].copy_from_slice(base0);
    out[lim..2 * lim].copy_from_slice(base1);
    for i in 0..labc {
        out[2 * lim + i] = a[i + 1] as f32;
        out[2 * lim + labc + i] = b[i + 1] as f32;
        out[2 * lim + 2 * labc + i] = c[i + 1] as f32;
    }
    out
}

/// Core Rust implementation of `vbgint`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `theta`: Colatitude in radians.
///
/// # Returns
/// A contiguous workspace or coefficient vector in storage.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vbgint_impl(nlat: usize, nlon: usize, theta: &[f64]) -> Vec<f32> {
    let imid = (nlat + 1) / 2;
    let lim = imid * nlat;
    let mdo = 2.min(nlat).min((nlon + 1) / 2);
    let mut base0 = vec![0.0_f32; lim];
    let mut base1 = vec![0.0_f32; lim];

    for mp1 in 1..=mdo {
        let m = (mp1 - 1) as i32;
        for np1 in mp1..=nlat {
            let n = (np1 - 1) as i32;
            let cv = dvbk(m, n);
            for i in 1..=imid {
                let vbh = dvbt(m, n, theta[i - 1], &cv) as f32;
                let idx = (np1 - 1) * imid + (i - 1);
                if mp1 == 1 {
                    base0[idx] = vbh;
                } else {
                    base1[idx] = vbh;
                }
            }
        }
    }

    let (a, b, c) = rabcv1(nlat, nlon);
    pack_tables(&base0, &base1, &a, &b, &c, lim)
}

/// Core Rust implementation of `wbgint`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `theta`: Colatitude in radians.
///
/// # Returns
/// A contiguous workspace or coefficient vector in storage.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn wbgint_impl(nlat: usize, nlon: usize, theta: &[f64]) -> Vec<f32> {
    let imid = (nlat + 1) / 2;
    let lim = imid * nlat;
    let mdo = 3.min(nlat).min((nlon + 1) / 2);
    let mut base0 = vec![0.0_f32; lim];
    let mut base1 = vec![0.0_f32; lim];

    if mdo >= 2 {
        for mp1 in 2..=mdo {
            let m = (mp1 - 1) as i32;
            for np1 in mp1..=nlat {
                let n = (np1 - 1) as i32;
                let cw = dwbk(m, n);
                for i in 1..=imid {
                    let wbh = dwbt(m, n, theta[i - 1], &cw) as f32;
                    let idx = (np1 - 1) * imid + (i - 1);
                    if m == 1 {
                        base0[idx] = wbh;
                    } else {
                        base1[idx] = wbh;
                    }
                }
            }
        }
    }

    let (a, b, c) = rabcw1(nlat, nlon);
    pack_tables(&base0, &base1, &a, &b, &c, lim)
}

/// Core Rust implementation of `vtgint`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `theta`: Colatitude in radians.
///
/// # Returns
/// A contiguous workspace or coefficient vector in storage.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vtgint_impl(nlat: usize, nlon: usize, theta: &[f64]) -> Vec<f32> {
    let imid = (nlat + 1) / 2;
    let lim = imid * nlat;
    let mdo = 2.min(nlat).min((nlon + 1) / 2);
    let mut base0 = vec![0.0_f32; lim];
    let mut base1 = vec![0.0_f32; lim];

    for mp1 in 1..=mdo {
        let m = (mp1 - 1) as i32;
        for np1 in mp1..=nlat {
            let n = (np1 - 1) as i32;
            let cv = dvtk(m, n);
            for i in 1..=imid {
                let vbh = dvtt(m, n, theta[i - 1], &cv) as f32;
                let idx = (np1 - 1) * imid + (i - 1);
                if mp1 == 1 {
                    base0[idx] = vbh;
                } else {
                    base1[idx] = vbh;
                }
            }
        }
    }

    let (a, b, c) = rabcv1(nlat, nlon);
    pack_tables(&base0, &base1, &a, &b, &c, lim)
}

/// Core Rust implementation of `wtgint`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `theta`: Colatitude in radians.
///
/// # Returns
/// A contiguous workspace or coefficient vector in storage.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn wtgint_impl(nlat: usize, nlon: usize, theta: &[f64]) -> Vec<f32> {
    let imid = (nlat + 1) / 2;
    let lim = imid * nlat;
    let mdo = 3.min(nlat).min((nlon + 1) / 2);
    let mut base0 = vec![0.0_f32; lim];
    let mut base1 = vec![0.0_f32; lim];

    if mdo >= 2 {
        for mp1 in 2..=mdo {
            let m = (mp1 - 1) as i32;
            for np1 in mp1..=nlat {
                let n = (np1 - 1) as i32;
                let cw = dwtk(m, n);
                for i in 1..=imid {
                    let wbh = dwtt(m, n, theta[i - 1], &cw) as f32;
                    let idx = (np1 - 1) * imid + (i - 1);
                    if m == 1 {
                        base0[idx] = wbh;
                    } else {
                        base1[idx] = wbh;
                    }
                }
            }
        }
    }

    let (a, b, c) = rabcw1(nlat, nlon);
    pack_tables(&base0, &base1, &a, &b, &c, lim)
}

/// Build the core workspace used by `vhagci`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
///
/// # Returns
/// `Ok` with the workspace vector, or `Err` with a error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vhagci_core(nlat: usize, nlon: usize) -> Result<Vec<f32>, i32> {
    let (theta, wts, ierr) = gaqd_impl(nlat as i32);
    if ierr != 0 {
        return Err(5);
    }
    let imid = (nlat + 1) / 2;
    let vbg = vbgint_impl(nlat, nlon, &theta);
    let wbg = wbgint_impl(nlat, nlon, &theta);
    let mut out = vec![0.0_f32; imid];
    for i in 0..imid {
        out[i] = wts[i] as f32;
    }
    out.extend_from_slice(&vbg);
    out.extend_from_slice(&wbg);
    out.extend_from_slice(&hrffti_impl(nlon as i32));
    Ok(out)
}

/// Build the core workspace used by `vhsgci`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
///
/// # Returns
/// `Ok` with the workspace vector, or `Err` with a error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vhsgci_core(nlat: usize, nlon: usize) -> Result<Vec<f32>, i32> {
    let (theta, _wts, ierr) = gaqd_impl(nlat as i32);
    if ierr != 0 {
        return Err(5);
    }
    let vbg = vbgint_impl(nlat, nlon, &theta);
    let wbg = wbgint_impl(nlat, nlon, &theta);
    let mut out = vbg;
    out.extend_from_slice(&wbg);
    out.extend_from_slice(&hrffti_impl(nlon as i32));
    Ok(out)
}

/// Build the core workspace used by `vtsgci`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
///
/// # Returns
/// `Ok` with the workspace vector, or `Err` with a error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vtsgci_core(nlat: usize, nlon: usize) -> Result<Vec<f32>, i32> {
    let (theta, _wts, ierr) = gaqd_impl(nlat as i32);
    if ierr != 0 {
        return Err(5);
    }
    let vtg = vtgint_impl(nlat, nlon, &theta);
    let wtg = wtgint_impl(nlat, nlon, &theta);
    let mut out = vtg;
    out.extend_from_slice(&wtg);
    out.extend_from_slice(&hrffti_impl(nlon as i32));
    Ok(out)
}
