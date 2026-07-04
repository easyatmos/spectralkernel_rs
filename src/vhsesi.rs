use crate::hrffti::hrffti_impl;
use crate::sphcom_vector::{dvbk, dvbt, dwbk, dwbt};
use numpy::PyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;

fn direct_vb_column(nlat: usize, m: usize) -> Vec<f32> {
    let imid = (nlat + 1) / 2;
    let dt = std::f64::consts::PI / (nlat.saturating_sub(1) as f64);
    let mut out = vec![0.0_f32; imid * nlat];
    for np1 in (m + 1)..=nlat {
        let coeff = dvbk(m as i32, (np1 - 1) as i32);
        for i in 1..=imid {
            let th = (i - 1) as f64 * dt;
            out[(np1 - 1) * imid + (i - 1)] = dvbt(m as i32, (np1 - 1) as i32, th, &coeff) as f32;
        }
    }
    out
}

fn direct_wb_column(nlat: usize, m: usize) -> Vec<f32> {
    let imid = (nlat + 1) / 2;
    let dt = std::f64::consts::PI / (nlat.saturating_sub(1) as f64);
    let mut out = vec![0.0_f32; imid * nlat];
    let mw = m.max(1);
    for np1 in (mw + 1)..=nlat {
        let coeff = dwbk(mw as i32, (np1 - 1) as i32);
        for i in 1..=imid {
            let th = (i - 1) as f64 * dt;
            out[(np1 - 1) * imid + (i - 1)] = dwbt(mw as i32, (np1 - 1) as i32, th, &coeff) as f32;
        }
    }
    out
}

pub fn vhsesi_impl(nlat: i32, nlon: i32, lvhses: i32, lwork: i32, ldwork: i32) -> (Vec<f32>, i32) {
    let mut ierror = 1;
    if nlat < 3 {
        return (Vec::new(), ierror);
    }
    ierror = 2;
    if nlon < 1 {
        return (Vec::new(), ierror);
    }
    ierror = 3;
    let mmax = nlat.min((nlon + 1) / 2);
    let imid = (nlat + 1) / 2;
    let lzimn = (imid * mmax * (nlat + nlat - mmax + 1)) / 2;
    if lvhses < lzimn + lzimn + nlon + 15 {
        return (Vec::new(), ierror);
    }
    ierror = 4;
    let labc = 3 * (0.max(mmax - 2) * (nlat + nlat - mmax - 1)) / 2;
    if lwork < 5 * nlat * imid + labc {
        return (Vec::new(), ierror);
    }
    ierror = 5;
    if ldwork < 2 * (nlat + 1) {
        return (Vec::new(), ierror);
    }

    let nlat_usize = usize::try_from(nlat).unwrap_or(0);
    let lzimn_usize = usize::try_from(lzimn).unwrap_or(0);
    let mut out = vec![0.0_f32; usize::try_from(lvhses).unwrap_or(0)];
    for mp1 in 1..=usize::try_from(mmax).unwrap_or(0) {
        let m = mp1 - 1;
        let mb = if m == 0 {
            0
        } else {
            m * (nlat_usize - 1) - (m * (m - 1)) / 2
        };
        let col = direct_vb_column(nlat_usize, m);
        for np1 in mp1..=nlat_usize {
            let mn = mb + np1;
            for i in 1..=imid as usize {
                let src = (np1 - 1) * imid as usize + (i - 1);
                let dst = (i - 1) + (mn - 1) * imid as usize;
                if dst < out.len() {
                    out[dst] = col[src];
                }
            }
        }
    }

    for mp1 in 1..=usize::try_from(mmax).unwrap_or(0) {
        let m = mp1 - 1;
        let mb = if m == 0 {
            0
        } else {
            m * (nlat_usize - 1) - (m * (m - 1)) / 2
        };
        let col = direct_wb_column(nlat_usize, m);
        for np1 in mp1..=nlat_usize {
            let mn = mb + np1;
            for i in 1..=imid as usize {
                let src = (np1 - 1) * imid as usize + (i - 1);
                let dst = lzimn_usize + (i - 1) + (mn - 1) * imid as usize;
                if dst < out.len() {
                    out[dst] = col[src];
                }
            }
        }
    }
    let fft = hrffti_impl(nlon);
    for (i, v) in fft.iter().enumerate() {
        let dst = 2 * lzimn_usize + i;
        if dst < out.len() {
            out[dst] = *v;
        }
    }
    (out, 0)
}

pub fn vhsesi_impl_parallel(
    nlat: i32,
    nlon: i32,
    lvhses: i32,
    lwork: i32,
    ldwork: i32,
) -> (Vec<f32>, i32) {
    let mut ierror = 1;
    if nlat < 3 {
        return (Vec::new(), ierror);
    }
    ierror = 2;
    if nlon < 1 {
        return (Vec::new(), ierror);
    }
    ierror = 3;
    let mmax = nlat.min((nlon + 1) / 2);
    let imid = (nlat + 1) / 2;
    let lzimn = (imid * mmax * (nlat + nlat - mmax + 1)) / 2;
    if lvhses < lzimn + lzimn + nlon + 15 {
        return (Vec::new(), ierror);
    }
    ierror = 4;
    let labc = 3 * (0.max(mmax - 2) * (nlat + nlat - mmax - 1)) / 2;
    if lwork < 5 * nlat * imid + labc {
        return (Vec::new(), ierror);
    }
    ierror = 5;
    if ldwork < 2 * (nlat + 1) {
        return (Vec::new(), ierror);
    }

    let nlat_usize = usize::try_from(nlat).unwrap_or(0);
    let imid_usize = usize::try_from(imid).unwrap_or(0);
    let mmax_usize = usize::try_from(mmax).unwrap_or(0);
    let lzimn_usize = usize::try_from(lzimn).unwrap_or(0);
    let out_len = usize::try_from(lvhses).unwrap_or(0);
    let mut out = vec![0.0_f32; out_len];

    let vb_chunks: Vec<Vec<(usize, f32)>> = (1..=mmax_usize)
        .into_par_iter()
        .map(|mp1| {
            let m = mp1 - 1;
            let mb = if m == 0 {
                0
            } else {
                m * (nlat_usize - 1) - (m * (m - 1)) / 2
            };
            let col = direct_vb_column(nlat_usize, m);
            let mut writes = Vec::new();
            for np1 in mp1..=nlat_usize {
                let mn = mb + np1;
                for i in 1..=imid_usize {
                    let src = (np1 - 1) * imid_usize + (i - 1);
                    let dst = (i - 1) + (mn - 1) * imid_usize;
                    if dst < out_len {
                        writes.push((dst, col[src]));
                    }
                }
            }
            writes
        })
        .collect();
    for chunk in vb_chunks {
        for (dst, value) in chunk {
            out[dst] = value;
        }
    }

    let wb_chunks: Vec<Vec<(usize, f32)>> = (1..=mmax_usize)
        .into_par_iter()
        .map(|mp1| {
            let m = mp1 - 1;
            let mb = if m == 0 {
                0
            } else {
                m * (nlat_usize - 1) - (m * (m - 1)) / 2
            };
            let col = direct_wb_column(nlat_usize, m);
            let mut writes = Vec::new();
            for np1 in mp1..=nlat_usize {
                let mn = mb + np1;
                for i in 1..=imid_usize {
                    let src = (np1 - 1) * imid_usize + (i - 1);
                    let dst = lzimn_usize + (i - 1) + (mn - 1) * imid_usize;
                    if dst < out_len {
                        writes.push((dst, col[src]));
                    }
                }
            }
            writes
        })
        .collect();
    for chunk in wb_chunks {
        for (dst, value) in chunk {
            out[dst] = value;
        }
    }

    let fft = hrffti_impl(nlon);
    for (i, v) in fft.iter().enumerate() {
        let dst = 2 * lzimn_usize + i;
        if dst < out.len() {
            out[dst] = *v;
        }
    }
    (out, 0)
}

#[pyfunction]
pub fn vhsesi<'py>(
    py: Python<'py>,
    nlat: i32,
    nlon: i32,
    lvhses: i32,
    lwork: i32,
    ldwork: i32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, i32)> {
    let (wvhses, ierror) = vhsesi_impl(nlat, nlon, lvhses, lwork, ldwork);
    Ok((PyArray1::from_vec(py, wvhses).to_owned(), ierror))
}
