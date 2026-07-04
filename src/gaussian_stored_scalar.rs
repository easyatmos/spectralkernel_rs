use crate::gaqd::gaqd_impl;
use crate::hrffti::hrffti_impl;
use crate::sphcom_scalar::{dnlfk, dnlft};

fn scalar_l(nlat: usize, nlon: usize) -> usize {
    ((nlon + 2) / 2).min(nlat)
}

fn scalar_late(nlat: usize) -> usize {
    (nlat + 1) / 2
}

fn scalar_labc(nlat: usize, l: usize) -> usize {
    l.saturating_sub(1) * l / 2 + (nlat.saturating_sub(l)) * l.saturating_sub(1)
}

fn scalar_pmn_count(nlat: usize, l: usize) -> usize {
    l * (2 * nlat - l + 1) / 2
}

fn idx_tri(m: usize, n: usize) -> usize {
    (n - 2) * (n - 1) / 2 + (m - 2)
}

fn idx_rect(l: usize, m: usize, n: usize) -> usize {
    (l - 1) * (l - 2) / 2 + (n - l) * (l - 1) + (m - 2)
}

#[allow(dead_code)]
fn idx_pmn(nlat: usize, m: usize, np1: usize) -> usize {
    m * (2 * nlat - m - 1) / 2 + np1
}

pub fn shagsp_impl(nlat: usize, nlon: usize) -> Result<Vec<f32>, i32> {
    if nlat < 3 {
        return Err(1);
    }
    if nlon < 4 {
        return Err(2);
    }

    let l = scalar_l(nlat, nlon);
    let late = (nlat + (nlat % 2)) / 2;
    let labc = scalar_labc(nlat, l);
    let prefix_len = nlat + 2 * nlat * late + 3 * labc + nlon + 15;

    let (theta, wts, ierr) = gaqd_impl(nlat as i32);
    if ierr != 0 {
        return Err(6);
    }

    let mut out = vec![0.0_f32; prefix_len];
    let i1 = 0;
    let i2 = i1 + nlat;
    let i3 = i2 + nlat * late;
    let i4 = i3 + nlat * late;
    let i5 = i4 + labc;
    let i6 = i5 + labc;
    let i7 = i6 + labc;

    for i in 0..nlat {
        out[i1 + i] = wts[i] as f32;
    }

    for np1 in 1..=nlat {
        let n = (np1 - 1) as i32;
        let cp0 = dnlfk(0, n);
        for i in 1..=late {
            let pb = dnlft(0, n, theta[i - 1], &cp0);
            let dst = i2 + (i - 1) * nlat + (np1 - 1);
            out[dst] = pb as f32;
        }

        if np1 >= 2 {
            let cp1 = dnlfk(1, n);
            for i in 1..=late {
                let pb = dnlft(1, n, theta[i - 1], &cp1);
                let dst = i3 + (i - 1) * nlat + (np1 - 1);
                out[dst] = pb as f32;
            }
        }
    }

    for n in 2..=nlat {
        let mlim = n.min(l);
        for m in 2..=mlim {
            let imn = if n >= l {
                idx_rect(l, m, n)
            } else {
                idx_tri(m, n)
            };
            let nf = n as f32;
            let mf = m as f32;
            out[i4 + imn] = (((2.0 * nf + 1.0) * (mf + nf - 2.0) * (mf + nf - 3.0))
                / ((2.0 * nf - 3.0) * (mf + nf - 1.0) * (mf + nf)))
                .sqrt();
            out[i5 + imn] = (((2.0 * nf + 1.0) * (nf - mf - 1.0) * (nf - mf))
                / ((2.0 * nf - 3.0) * (mf + nf - 1.0) * (mf + nf)))
                .sqrt();
            out[i6 + imn] =
                (((nf - mf + 1.0) * (nf - mf + 2.0)) / ((nf + mf - 1.0) * (nf + mf))).sqrt();
        }
    }

    let fft = hrffti_impl(nlon as i32);
    for (idx, value) in fft.iter().enumerate() {
        out[i7 + idx] = *value;
    }

    Ok(out)
}

pub fn shagss1_impl(nlat: usize, nlon: usize, w: &[f32]) -> Vec<f32> {
    let l = scalar_l(nlat, nlon);
    let late = scalar_late(nlat);
    let labc = scalar_labc(nlat, l);
    let pmnf_count = scalar_pmn_count(nlat, l);

    let i1 = nlat;
    let i2 = i1 + nlat * late;
    let i3 = i2 + nlat * late;
    let i4 = i3 + labc;
    let i5 = i4 + labc;

    let p0n = &w[i1..i2];
    let p1n = &w[i2..i3];
    let abel = &w[i3..i4];
    let bbel = &w[i4..i5];
    let cbel = &w[i5..i5 + labc];

    let mut pmnf = vec![0.0_f32; late * pmnf_count];
    let mut pmn = vec![0.0_f32; nlat * late * 3];
    let (mut km0, mut km1, mut km2) = (0_usize, 1_usize, 2_usize);

    for mp1 in 1..=l {
        let m = mp1 - 1;
        let ms = m + 1;
        let ninc = 1;

        if m > 1 {
            let mut np1 = ms;
            while np1 <= nlat {
                let n = np1 - 1;
                let imn = if n >= l {
                    idx_rect(l, m, n)
                } else {
                    idx_tri(m, n)
                };
                for i in 1..=late {
                    let dst = ((km0 * late + (i - 1)) * nlat) + (np1 - 1);
                    let src1 = ((km2 * late + (i - 1)) * nlat) + (n - 2);
                    let src2 = ((km0 * late + (i - 1)) * nlat) + (n - 2);
                    let src3 = ((km2 * late + (i - 1)) * nlat) + (np1 - 1);
                    pmn[dst] =
                        abel[imn] * pmn[src1] + bbel[imn] * pmn[src2] - cbel[imn] * pmn[src3];
                }
                np1 += ninc;
            }
        } else if m == 0 {
            let mut np1 = ms;
            while np1 <= nlat {
                for i in 1..=late {
                    let dst = ((km0 * late + (i - 1)) * nlat) + (np1 - 1);
                    let src = (i - 1) * nlat + (np1 - 1);
                    pmn[dst] = p0n[src];
                }
                np1 += ninc;
            }
        } else {
            let mut np1 = ms;
            while np1 <= nlat {
                for i in 1..=late {
                    let dst = ((km0 * late + (i - 1)) * nlat) + (np1 - 1);
                    let src = (i - 1) * nlat + (np1 - 1);
                    pmn[dst] = p1n[src];
                }
                np1 += ninc;
            }
        }

        let kmt = km0;
        km0 = km2;
        km2 = km1;
        km1 = kmt;
        let km = kmt;

        let mml1 = m * (2 * nlat - m - 1) / 2;
        for np1 in mp1..=nlat {
            let mn = mml1 + np1;
            for i in 1..=late {
                let src = ((km * late + (i - 1)) * nlat) + (np1 - 1);
                let dst = (mn - 1) * late + (i - 1);
                pmnf[dst] = pmn[src];
            }
        }
    }

    pmnf
}

pub fn shagsi_core(nlat: usize, nlon: usize) -> Result<Vec<f32>, i32> {
    let mut out = shagsp_impl(nlat, nlon)?;
    let pmnf = shagss1_impl(nlat, nlon, &out);
    out.extend_from_slice(&pmnf);
    Ok(out)
}
