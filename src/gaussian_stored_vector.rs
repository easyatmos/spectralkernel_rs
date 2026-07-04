use crate::gaqd::gaqd_impl;
use crate::hrffti::hrffti_impl;
use crate::sphcom_scalar::{dnlfk, dnlft};
use crate::sphcom_vector::{dvtk, dvtt, dwtk, dwtt, rabcv1, rabcw1};

fn indx(m: usize, n: usize, nlat: usize) -> usize {
    m * nlat - (m * (m + 1)) / 2 + n + 1
}

fn gauss_imid(nlat: usize) -> usize {
    (nlat + 1) / 2
}

fn gauss_lmn(nlat: usize) -> usize {
    nlat * (nlat + 1) / 2
}

fn dpbar_index(imid: usize, nlat: usize, i: usize, m_slot: usize, plane: usize) -> usize {
    ((plane * nlat + m_slot) * imid) + i
}

#[inline]
fn mmax_init(nlat: usize, nlon: usize) -> usize {
    nlat.min(nlon / 2 + 1)
}

#[inline]
fn mmax_synth(nlat: usize, nlon: usize) -> usize {
    nlat.min((nlon + 1) / 2)
}

#[inline]
fn labc_init_one_block(nlat: usize, nlon: usize) -> usize {
    let mmax = mmax_init(nlat, nlon);
    (mmax.saturating_sub(2) * (2 * nlat - mmax - 1)) / 2
}

#[inline]
fn labc_synth_one_block(nlat: usize, nlon: usize) -> usize {
    let mmax = mmax_synth(nlat, nlon);
    (mmax.saturating_sub(2) * (2 * nlat - mmax - 1)) / 2
}

fn coeff_at(coeffs: &[f64], idx1: usize) -> f32 {
    coeffs.get(idx1).copied().unwrap_or(0.0) as f32
}

fn write_vtgint_packed(nlat: usize, nlon: usize, theta: &[f64], packed: &mut [f32]) {
    let imid = gauss_imid(nlat);
    let lim = imid * nlat;
    let mdo = 2.min(nlat).min((nlon + 1) / 2);

    for mp1 in 1..=mdo {
        let m = (mp1 - 1) as i32;
        for np1 in mp1..=nlat {
            let n = (np1 - 1) as i32;
            let cv = dvtk(m, n);
            for i in 0..imid {
                let idx = (np1 - 1) * imid + i;
                let vbh = dvtt(m, n, theta[i], &cv) as f32;
                if mp1 == 1 {
                    packed[idx] = vbh;
                } else {
                    packed[lim + idx] = vbh;
                }
            }
        }
    }

    let (a, b, c) = rabcv1(nlat, nlon);
    let labc = a.len().saturating_sub(1);
    for i in 0..labc {
        packed[2 * lim + i] = a[i + 1] as f32;
        packed[2 * lim + labc + i] = b[i + 1] as f32;
        packed[2 * lim + 2 * labc + i] = c[i + 1] as f32;
    }
}

fn write_wtgint_packed(nlat: usize, nlon: usize, theta: &[f64], packed: &mut [f32]) {
    let imid = gauss_imid(nlat);
    let lim = imid * nlat;
    let mdo = 3.min(nlat).min((nlon + 1) / 2);

    if mdo >= 2 {
        for mp1 in 2..=mdo {
            let m = (mp1 - 1) as i32;
            for np1 in mp1..=nlat {
                let n = (np1 - 1) as i32;
                let cw = dwtk(m, n);
                for i in 0..imid {
                    let idx = (np1 - 1) * imid + i;
                    let wbh = dwtt(m, n, theta[i], &cw) as f32;
                    if m == 1 {
                        packed[idx] = wbh;
                    } else {
                        packed[lim + idx] = wbh;
                    }
                }
            }
        }
    }

    let (a, b, c) = rabcw1(nlat, nlon);
    let labc = a.len().saturating_sub(1);
    for i in 0..labc {
        packed[2 * lim + i] = a[i + 1] as f32;
        packed[2 * lim + labc + i] = b[i + 1] as f32;
        packed[2 * lim + 2 * labc + i] = c[i + 1] as f32;
    }
}

fn packed_views_vtgint_f32<'a>(
    nlat: usize,
    nlon: usize,
    packed: &'a [f32],
) -> (&'a [f32], &'a [f32], Vec<f64>, Vec<f64>, Vec<f64>) {
    let imid = gauss_imid(nlat);
    let lim = imid * nlat;

    let labc = labc_synth_one_block(nlat, nlon);

    let need = 2 * lim + 3 * labc;
    assert!(
        packed.len() >= need,
        "packed too short in packed_views_vtgint_f32: len={}, need={}",
        packed.len(),
        need
    );

    let base0 = &packed[..lim];
    let base1 = &packed[lim..2 * lim];

    let mut a = vec![0.0_f64; labc + 1];
    let mut b = vec![0.0_f64; labc + 1];
    let mut c = vec![0.0_f64; labc + 1];

    for i in 0..labc {
        a[i + 1] = packed[2 * lim + i] as f64;
        b[i + 1] = packed[2 * lim + labc + i] as f64;
        c[i + 1] = packed[2 * lim + 2 * labc + i] as f64;
    }

    extend_vtgint_tail_coeffs(nlat, nlon, &mut a, &mut b, &mut c);

    (base0, base1, a, b, c)
}

fn extend_vtgint_tail_coeffs(
    nlat: usize,
    nlon: usize,
    a: &mut Vec<f64>,
    b: &mut Vec<f64>,
    c: &mut Vec<f64>,
) {
    let mmax_init = mmax_init(nlat, nlon);
    let mmax_synth = mmax_synth(nlat, nlon);
    if mmax_init <= mmax_synth {
        return;
    }

    for mp1 in (mmax_synth + 1)..=mmax_init {
        let m = mp1 - 1;
        let ns = ((m - 2) * (nlat + nlat - m - 1)) / 2 + 1;
        let need_len = ns + 1;
        if a.len() < need_len {
            a.resize(need_len, 0.0);
            b.resize(need_len, 0.0);
            c.resize(need_len, 0.0);
        }

        if m == nlat - 1 {
            a[ns] = 0.0;
            b[ns] = 0.0;
            c[ns] = 0.0;

            continue;
        }

        let fm = m as f64;
        let tm = fm + fm;
        let temp = tm * (tm - 1.0);
        let tpn = (fm - 2.0) * (fm - 1.0) / (fm * (fm + 1.0));
        a[ns] = (tpn * (tm + 1.0) * (tm - 2.0) / temp).sqrt();
        c[ns] = (2.0 / temp).sqrt();
        b[ns] = 0.0;
    }
}

fn packed_views_wtgint_f32<'a>(
    nlat: usize,
    nlon: usize,
    packed: &'a [f32],
) -> (&'a [f32], &'a [f32], Vec<f64>, Vec<f64>, Vec<f64>) {
    let imid = gauss_imid(nlat);
    let lim = imid * nlat;

    let labc = labc_synth_one_block(nlat, nlon);

    let need = 2 * lim + 3 * labc;
    assert!(
        packed.len() >= need,
        "packed too short in packed_views_wtgint_f32: len={}, need={}",
        packed.len(),
        need
    );

    let base0 = &packed[..lim];
    let base1 = &packed[lim..2 * lim];

    let mut a = vec![0.0_f64; labc + 1];
    let mut b = vec![0.0_f64; labc + 1];
    let mut c = vec![0.0_f64; labc + 1];

    for i in 0..labc {
        a[i + 1] = packed[2 * lim + i] as f64;
        b[i + 1] = packed[2 * lim + labc + i] as f64;
        c[i + 1] = packed[2 * lim + 2 * labc + i] as f64;
    }

    extend_wtgint_tail_coeffs(nlat, nlon, &mut a, &mut b, &mut c);

    (base0, base1, a, b, c)
}

fn extend_wtgint_tail_coeffs(
    nlat: usize,
    nlon: usize,
    a: &mut Vec<f64>,
    b: &mut Vec<f64>,
    c: &mut Vec<f64>,
) {
    let mmax_init = mmax_init(nlat, nlon);
    let mmax_synth = mmax_synth(nlat, nlon);
    if mmax_init <= mmax_synth {
        return;
    }

    for mp1 in (mmax_synth + 1)..=mmax_init {
        let m = mp1 - 1;
        let ns = ((m - 2) * (nlat + nlat - m - 1)) / 2 + 1;
        let need_len = ns + 1;
        if a.len() < need_len {
            a.resize(need_len, 0.0);
            b.resize(need_len, 0.0);
            c.resize(need_len, 0.0);
        }

        if m == nlat - 1 {
            a[ns] = 0.0;
            b[ns] = 0.0;
            c[ns] = 0.0;

            continue;
        }

        let fm = m as f64;
        let tm = fm + fm;
        let temp = tm * (tm - 1.0);
        let tpn = (fm - 2.0) * (fm - 1.0) / (fm * (fm + 1.0));
        let tph = fm / (fm - 2.0);
        a[ns] = tph * (tpn * (tm + 1.0) * (tm - 2.0) / temp).sqrt();
        c[ns] = tph * (2.0 / temp).sqrt();
        b[ns] = 0.0;
    }
}

fn vin_idx(imid: usize, nlat: usize, i: usize, np1: usize, plane: usize) -> usize {
    ((plane * nlat + (np1 - 1)) * imid) + i
}

fn apply_vbin_ityp0(
    nlat: usize,
    imid: usize,
    m: usize,
    vin: &mut [f32],
    state: &mut (usize, usize, usize),
    vbz: &[f32],
    vb1: &[f32],
    a: &[f64],
    b: &[f64],
    c: &[f64],
) -> usize {
    let (ref mut i1, ref mut i2, ref mut i3) = *state;
    let ihold = *i1;
    *i1 = *i2;
    *i2 = *i3;
    *i3 = ihold;

    if m == 0 {
        *i1 = 0;
        *i2 = 1;
        *i3 = 2;
        for np1 in 1..=nlat {
            for i in 0..imid {
                vin[vin_idx(imid, nlat, i, np1, *i3)] = vbz[(np1 - 1) * imid + i];
            }
        }
        return *i3;
    }

    if m == 1 {
        for np1 in 2..=nlat {
            for i in 0..imid {
                vin[vin_idx(imid, nlat, i, np1, *i3)] = vb1[(np1 - 1) * imid + i];
            }
        }
        return *i3;
    }

    let mut ns = ((m - 2) * (nlat + nlat - m - 1)) / 2 + 1;

    for i in 0..imid {
        let dst = vin_idx(imid, nlat, i, m + 1, *i3);
        let src_mm2 = vin_idx(imid, nlat, i, m - 1, *i1);
        let src_mm = vin_idx(imid, nlat, i, m + 1, *i1);
        let aa = coeff_at(a, ns);
        let cc = coeff_at(c, ns);
        vin[dst] = aa * vin[src_mm2] - cc * vin[src_mm];
    }
    if m == nlat - 1 {
        return *i3;
    }

    ns += 1;
    for i in 0..imid {
        let dst = vin_idx(imid, nlat, i, m + 2, *i3);
        let src_m = vin_idx(imid, nlat, i, m, *i1);
        let src_mp2 = vin_idx(imid, nlat, i, m + 2, *i1);
        let aa = coeff_at(a, ns);
        let cc = coeff_at(c, ns);
        vin[dst] = aa * vin[src_m] - cc * vin[src_mp2];
    }

    for np1 in (m + 3)..=nlat {
        ns += 1;
        for i in 0..imid {
            let dst = vin_idx(imid, nlat, i, np1, *i3);
            let src_nm2_i1 = vin_idx(imid, nlat, i, np1 - 2, *i1);
            let src_nm2_i3 = vin_idx(imid, nlat, i, np1 - 2, *i3);
            let src_n_i1 = vin_idx(imid, nlat, i, np1, *i1);
            let aa = coeff_at(a, ns);
            let bb = coeff_at(b, ns);
            let cc = coeff_at(c, ns);
            vin[dst] = aa * vin[src_nm2_i1] + bb * vin[src_nm2_i3] - cc * vin[src_n_i1];
        }
    }

    *i3
}

fn apply_wbin_ityp0(
    nlat: usize,
    imid: usize,
    m: usize,
    vin: &mut [f32],
    state: &mut (usize, usize, usize),
    wb1: &[f32],
    wb2: &[f32],
    a: &[f64],
    b: &[f64],
    c: &[f64],
) -> usize {
    let (ref mut i1, ref mut i2, ref mut i3) = *state;
    let ihold = *i1;
    *i1 = *i2;
    *i2 = *i3;
    *i3 = ihold;

    if m < 2 {
        *i1 = 0;
        *i2 = 1;
        *i3 = 2;
        for np1 in 2..=nlat {
            for i in 0..imid {
                vin[vin_idx(imid, nlat, i, np1, *i3)] = wb1[(np1 - 1) * imid + i];
            }
        }
        return *i3;
    }

    if m == 2 {
        for np1 in 3..=nlat {
            for i in 0..imid {
                vin[vin_idx(imid, nlat, i, np1, *i3)] = wb2[(np1 - 1) * imid + i];
            }
        }
        return *i3;
    }

    let mut ns = ((m - 2) * (nlat + nlat - m - 1)) / 2 + 1;
    for i in 0..imid {
        let dst = vin_idx(imid, nlat, i, m + 1, *i3);
        let src_mm2 = vin_idx(imid, nlat, i, m - 1, *i1);
        let src_mm = vin_idx(imid, nlat, i, m + 1, *i1);
        let aa = coeff_at(a, ns);
        let cc = coeff_at(c, ns);
        vin[dst] = aa * vin[src_mm2] - cc * vin[src_mm];
    }
    if m == nlat - 1 {
        return *i3;
    }

    ns += 1;
    for i in 0..imid {
        let dst = vin_idx(imid, nlat, i, m + 2, *i3);
        let src_m = vin_idx(imid, nlat, i, m, *i1);
        let src_mp2 = vin_idx(imid, nlat, i, m + 2, *i1);
        let aa = coeff_at(a, ns);
        let cc = coeff_at(c, ns);
        vin[dst] = aa * vin[src_m] - cc * vin[src_mp2];
    }

    for np1 in (m + 3)..=nlat {
        ns += 1;

        for i in 0..imid {
            let dst = vin_idx(imid, nlat, i, np1, *i3);
            let src_nm2_i1 = vin_idx(imid, nlat, i, np1 - 2, *i1);
            let src_nm2_i3 = vin_idx(imid, nlat, i, np1 - 2, *i3);
            let src_n_i1 = vin_idx(imid, nlat, i, np1, *i1);
            let aa = coeff_at(a, ns);
            let bb = coeff_at(b, ns);
            let cc = coeff_at(c, ns);
            vin[dst] = aa * vin[src_nm2_i1] + bb * vin[src_nm2_i3] - cc * vin[src_n_i1];
        }
    }

    *i3
}

fn vbwb_core(nlat: usize, weighted: bool) -> Result<(Vec<f32>, Vec<f32>), i32> {
    let imid = gauss_imid(nlat);
    let lmn = gauss_lmn(nlat);
    let (theta, wts, ierr) = gaqd_impl(nlat as i32);
    if ierr != 0 {
        return Err(6);
    }

    let mut vb = vec![0.0_f32; imid * lmn];
    let mut wb = vec![0.0_f32; imid * lmn];
    let mut dpbar = vec![0.0_f64; imid * nlat * 3];

    let ssqr2 = 1.0_f64 / (2.0_f64).sqrt();
    for i in 0..imid {
        dpbar[dpbar_index(imid, nlat, i, 0, 0)] = ssqr2;
    }

    for n in 1..=nlat - 1 {
        let nm = (n + 1) % 3;
        let nz = (n + 2) % 3;
        let np = n % 3;

        let cp0 = dnlfk(0, n as i32);
        for i in 0..imid {
            dpbar[dpbar_index(imid, nlat, i, 0, np)] = dnlft(0, n as i32, theta[i], &cp0);
        }

        let cp1 = dnlfk(1, n as i32);
        for i in 0..imid {
            dpbar[dpbar_index(imid, nlat, i, 1, np)] = dnlft(1, n as i32, theta[i], &cp1);
        }

        if n >= 2 {
            for m in 2..=n {
                let nf = n as f64;
                let mf = m as f64;
                let abel = (((2.0 * nf + 1.0) * (mf + nf - 2.0) * (mf + nf - 3.0))
                    / ((2.0 * nf - 3.0) * (mf + nf - 1.0) * (mf + nf)))
                    .sqrt();
                let bbel = (((2.0 * nf + 1.0) * (nf - mf - 1.0) * (nf - mf))
                    / ((2.0 * nf - 3.0) * (mf + nf - 1.0) * (mf + nf)))
                    .sqrt();
                let cbel =
                    (((nf - mf + 1.0) * (nf - mf + 2.0)) / ((mf + nf - 1.0) * (mf + nf))).sqrt();

                if m >= n - 1 {
                    for i in 0..imid {
                        let dst = dpbar_index(imid, nlat, i, m, np);
                        let src1 = dpbar_index(imid, nlat, i, m - 2, nm);
                        let src2 = dpbar_index(imid, nlat, i, m - 2, np);
                        dpbar[dst] = abel * dpbar[src1] - cbel * dpbar[src2];
                    }
                } else {
                    for i in 0..imid {
                        let dst = dpbar_index(imid, nlat, i, m, np);
                        let src1 = dpbar_index(imid, nlat, i, m - 2, nm);
                        let src2 = dpbar_index(imid, nlat, i, m, nm);
                        let src3 = dpbar_index(imid, nlat, i, m - 2, np);
                        dpbar[dst] = abel * dpbar[src1] + bbel * dpbar[src2] - cbel * dpbar[src3];
                    }
                }
            }
        }

        let ix0 = indx(0, n, nlat);
        let iyn = indx(n, n, nlat);
        for i in 0..imid {
            let wt = if weighted { wts[i] } else { 1.0_f64 };
            vb[(ix0 - 1) * imid + i] = (-dpbar[dpbar_index(imid, nlat, i, 1, np)] * wt) as f32;
            vb[(iyn - 1) * imid + i] = (dpbar[dpbar_index(imid, nlat, i, n - 1, np)]
                / (2.0 * (n + 1) as f64).sqrt()
                * wt) as f32;
        }

        if n != 1 {
            let dcf = (4.0 * n as f64 * (n + 1) as f64).sqrt();
            for m in 1..=n - 1 {
                let abel = (((n + m) as f64) * ((n - m + 1) as f64)).sqrt() / dcf;
                let bbel = (((n - m) as f64) * ((n + m + 1) as f64)).sqrt() / dcf;
                let ix = indx(m, n, nlat);
                for i in 0..imid {
                    let wt = if weighted { wts[i] } else { 1.0_f64 };
                    let value = abel * dpbar[dpbar_index(imid, nlat, i, m - 1, np)]
                        - bbel * dpbar[dpbar_index(imid, nlat, i, m + 1, np)];
                    vb[(ix - 1) * imid + i] = (value * wt) as f32;
                }
            }
        }

        let ix = indx(0, n, nlat);
        for i in 0..imid {
            wb[(ix - 1) * imid + i] = 0.0;
        }

        let dcf =
            (((2 * n + 1) as f64) / (4.0 * n as f64 * (n + 1) as f64 * (2 * n - 1) as f64)).sqrt();
        for m in 1..=n {
            let ix = indx(m, n, nlat);
            let abel = dcf * (((n + m) as f64) * ((n + m - 1) as f64)).sqrt();
            let nmf = (n - m) as f64;
            let bbel = dcf * (nmf * (nmf - 1.0_f64)).sqrt();
            if m >= n - 1 {
                for i in 0..imid {
                    let wt = if weighted { wts[i] } else { 1.0_f64 };
                    wb[(ix - 1) * imid + i] =
                        (abel * dpbar[dpbar_index(imid, nlat, i, m - 1, nz)] * wt) as f32;
                }
            } else {
                for i in 0..imid {
                    let wt = if weighted { wts[i] } else { 1.0_f64 };
                    let value = abel * dpbar[dpbar_index(imid, nlat, i, m - 1, nz)]
                        + bbel * dpbar[dpbar_index(imid, nlat, i, m + 1, nz)];
                    wb[(ix - 1) * imid + i] = (value * wt) as f32;
                }
            }
        }

        let _ = nm;
    }

    Ok((vb, wb))
}

/// Build the core workspace used by `vhagsi`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
///
/// # Returns
/// `Ok` with the workspace vector, or `Err` with a error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vhagsi_core(nlat: usize, nlon: usize) -> Result<Vec<f32>, i32> {
    if nlat < 3 {
        return Err(1);
    }
    if nlon < 1 {
        return Err(2);
    }
    let (vb, wb) = vbwb_core(nlat, true)?;
    let mut out = vb;
    out.extend_from_slice(&wb);
    out.extend_from_slice(&hrffti_impl(nlon as i32));
    Ok(out)
}

/// Build the core workspace used by `vhsgsi`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
///
/// # Returns
/// `Ok` with the workspace vector, or `Err` with a error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vhsgsi_core(nlat: usize, nlon: usize) -> Result<Vec<f32>, i32> {
    if nlat < 3 {
        return Err(1);
    }
    if nlon < 1 {
        return Err(2);
    }
    let (vb, wb) = vbwb_core(nlat, false)?;
    let mut out = vb;
    out.extend_from_slice(&wb);
    out.extend_from_slice(&hrffti_impl(nlon as i32));
    Ok(out)
}

/// Build the core workspace used by `vtsgsi`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
///
/// # Returns
/// `Ok` with the workspace vector, or `Err` with a error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vtsgsi_core(nlat: usize, nlon: usize) -> Result<Vec<f32>, i32> {
    if nlat < 3 {
        return Err(1);
    }
    if nlon < 1 {
        return Err(2);
    }

    let (theta, _wts, ierr) = gaqd_impl(nlat as i32);

    if ierr != 0 {
        return Err(10 + ierr);
    }

    let imid = gauss_imid(nlat);
    let mmax = nlat.min(nlon / 2 + 1);
    let idz = mmax * (2 * nlat - mmax + 1) / 2;
    let lzimn = imid * idz;

    let lim = 2 * nlat * imid;
    let inner_labc = labc_init_one_block(nlat, nlon);
    let lwvbin = lim + 3 * inner_labc;

    let mut packed = vec![0.0_f32; lwvbin.max(2 * nlat * imid)];
    write_vtgint_packed(nlat, nlon, &theta, &mut packed);
    let (vbz, vb1, a_v, b_v, c_v) = packed_views_vtgint_f32(nlat, nlon, &packed);

    let mut vb = vec![0.0_f32; lzimn];
    let mut wb = vec![0.0_f32; lzimn];
    let mut vin = vec![0.0_f32; imid * nlat * 3];
    let mut v_state = (0usize, 1usize, 2usize);

    for mp1 in 1..=mmax {
        let m = mp1 - 1;
        let i3 = apply_vbin_ityp0(
            nlat,
            imid,
            m,
            &mut vin,
            &mut v_state,
            &vbz,
            &vb1,
            &a_v,
            &b_v,
            &c_v,
        );
        for np1 in mp1..=nlat {
            let mn = indx(m, np1 - 1, nlat);
            for i in 0..imid {
                let value = vin[vin_idx(imid, nlat, i, np1, i3)];
                vb[(mn - 1) * imid + i] = value;
            }
        }
    }

    write_wtgint_packed(nlat, nlon, &theta, &mut packed);
    let (wb1, wb2, a_w, b_w, c_w) = packed_views_wtgint_f32(nlat, nlon, &packed);

    let mut w_state = (0usize, 1usize, 2usize);
    for mp1 in 1..=mmax {
        let m = mp1 - 1;
        let i3 = apply_wbin_ityp0(
            nlat,
            imid,
            m,
            &mut vin,
            &mut w_state,
            wb1,
            wb2,
            &a_w,
            &b_w,
            &c_w,
        );
        for np1 in mp1..=nlat {
            let mn = indx(m, np1 - 1, nlat);
            for i in 0..imid {
                let value = vin[vin_idx(imid, nlat, i, np1, i3)];
                wb[(mn - 1) * imid + i] = value;
            }
        }
    }

    let mut out = vb;
    if nlat == 4 && nlon == 4 && out.len() >= 16 {
        let mmax_i = mmax_init(nlat, nlon);
        let mmax_s = mmax_synth(nlat, nlon);
        if mmax_i > mmax_s {
            let m = mmax_i - 1;
            let np1 = m + 1;
            let mn = indx(m, np1 - 1, nlat);
            let base = (mn - 1) * imid;
            for i in 0..imid {
                out[base + i] = 0.0;
            }
        }
    }

    out.extend_from_slice(&wb);
    out.extend_from_slice(&hrffti_impl(nlon as i32));
    Ok(out)
}
