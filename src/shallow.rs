use crate::dives::dives_impl;
use crate::grades::grades_impl;
use crate::shaes::shaes_impl;
use crate::shaesi::shaesi_impl;
use crate::shses::shses_impl;
use crate::shsesi::shsesi_impl;
use crate::vhaes::vhaes_impl;
use crate::vhaesi::vhaesi_impl;
use crate::vhses::vhses_impl;
use crate::vhsesi::vhsesi_impl;
use crate::vrtes::vrtes_impl;
use crate::vtses::vtsesi_impl;
use crate::vtsgs::vtsgs_impl;
use ndarray::{ArrayD, IxDyn};
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn idx(nlon: usize, i: usize, j: usize) -> usize {
    i * nlon + j
}

fn atanxy(x: f32, y: f32) -> f32 {
    if x == 0.0 && y == 0.0 {
        0.0
    } else {
        y.atan2(x)
    }
}

fn ui(amp: f32, thetad: f32) -> f32 {
    let pi = 4.0_f32 * 1.0_f32.atan();
    let thetab = -pi / 6.0;
    let thetae = pi / 2.0;
    let xe = 3.0e-1_f32;
    let x = xe * (thetad - thetab) / (thetae - thetab);
    if x <= 0.0 || x >= xe {
        0.0
    } else {
        amp * (-1.0 / x - 1.0 / (xe - x) + 4.0 / xe).exp()
    }
}

fn sine_transform(x: &mut [f32]) {
    let n = x.len();
    let arg = 4.0_f32 * 1.0_f32.atan() / (n as f32 + 1.0);
    let mut w = vec![0.0_f32; n];
    for j in 1..=n {
        for i in 1..=n {
            w[j - 1] += x[i - 1] * ((i * j) as f32 * arg).sin();
        }
    }
    for i in 0..n {
        x[i] = 2.0 * w[i];
    }
}

fn cosine(theta: f32, cf: &[f32]) -> f32 {
    cf.iter()
        .enumerate()
        .map(|(i, value)| *value * ((i + 1) as f32 * theta).cos())
        .sum()
}

fn scalar_saved_len(nlat: usize, nlon: usize) -> usize {
    let mmax = nlat.min(nlon / 2 + 1);
    let imid = nlat.div_ceil(2);
    (imid * mmax * (2 * nlat - mmax + 1)) / 2 + nlon + 15
}

fn vector_saved_len(nlat: usize, nlon: usize) -> usize {
    let mmax = nlat.min((nlon + 1) / 2);
    let imid = nlat.div_ceil(2);
    imid * mmax * (2 * nlat - mmax + 1) + nlon + 15
}

fn init_work_len(nlat: usize, nlon: usize) -> usize {
    let mmax = nlat.min(nlon / 2 + 1);
    let imid = nlat.div_ceil(2);
    let labc = 3 * mmax.saturating_sub(2) * (2 * nlat - mmax - 1) / 2;
    5 * nlat * imid + labc
}

fn low_level_lwork(nlat: usize, nlon: usize) -> usize {
    4 * nlat * nlon + 2 * nlat * (nlat + 1)
}

fn build_initial_fields(
    nlat: usize,
    nlon: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32, f32, f32, f32)> {
    if nlat < 3 {
        return Err(PyValueError::new_err("nlat must be at least 3"));
    }
    if nlon < 4 {
        return Err(PyValueError::new_err("nlon must be at least 4"));
    }

    let pi = 4.0_f32 * 1.0_f32.atan();
    let hpi = pi / 2.0;
    let aa = 6.37122e6_f32;
    let omega = 7.292e-5_f32;
    let fzero = omega + omega;
    let uzero = 40.0_f32;
    let alphad = 60.0_f32;
    let alpha = pi * alphad / 180.0;
    let ca = alpha.cos();
    let sa = alpha.sin();

    let nl = 91usize;
    let nlm1 = nl - 1;
    let nlm2 = nl - 2;
    let cfn = 1.0 / nlm1 as f32;
    let dlath = pi / nlm1 as f32;
    let mut phlt = vec![0.0_f32; nlm2];
    for i in 1..=nlm2 {
        let theta = i as f32 * dlath;
        let sth = theta.sin();
        let cth = theta.cos();
        let uhat = ui(uzero, hpi - theta);
        phlt[i - 1] = cfn * cth * uhat * (uhat / sth + aa * fzero);
    }
    sine_transform(&mut phlt);
    for i in 1..=nlm2 {
        phlt[i - 1] = -phlt[i - 1] / i as f32;
    }

    let dtheta = pi / (nlat - 1) as f32;
    let dlam = (pi + pi) / nlon as f32;
    let mut u = vec![0.0_f32; nlat * nlon];
    let mut v = vec![0.0_f32; nlat * nlon];
    let mut p = vec![0.0_f32; nlat * nlon];
    let mut f = vec![0.0_f32; nlat * nlon];
    for j in 0..nlon {
        let lambda = j as f32 * dlam;
        let cl = lambda.cos();
        let sl = lambda.sin();
        for i in 0..nlat {
            let theta = i as f32 * dtheta;
            let st = theta.cos();
            let ct = theta.sin();
            let sth = ca * st + sa * ct * cl;
            let cthclh = ca * ct * cl - sa * st;
            let cthslh = ct * sl;
            let lhat = atanxy(cthclh, cthslh);
            let clh = lhat.cos();
            let slh = lhat.sin();
            let cth = clh * cthclh + slh * cthslh;
            let that = atanxy(sth, cth);
            let uhat = ui(uzero, hpi - that);
            let k = idx(nlon, i, j);
            p[k] = cosine(that, &phlt);
            u[k] = uhat * (ca * sl * slh + cl * clh);
            v[k] = uhat * (ca * cl * slh * st - clh * sl * st + sa * slh * ct);
            f[k] = fzero * sth;
        }
    }

    let mut vmax = 0.0_f32;
    let mut pmax = 0.0_f32;
    let mut v2max = 0.0_f32;
    let mut p2max = 0.0_f32;
    for k in 0..u.len() {
        v2max += u[k] * u[k] + v[k] * v[k];
        p2max += p[k] * p[k];
        vmax = vmax.max(u[k].abs()).max(v[k].abs());
        pmax = pmax.max(p[k].abs());
    }

    Ok((u, v, p, f, vmax, pmax, v2max, p2max))
}

fn truncate_coeffs(nlat: usize, mmode: usize, a: &mut [f32], b: &mut [f32]) {
    let mp = mmode + 2;
    if mp > nlat {
        return;
    }
    for n in mp..=nlat {
        for m in 1..=n {
            let pos = (m - 1) * nlat + (n - 1);
            a[pos] = 0.0;
            b[pos] = 0.0;
        }
    }
}

fn geo_vhaes(
    u: &[f32],
    v: &[f32],
    nlat: usize,
    nlon: usize,
    wvha: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, i32)> {
    let v_math: Vec<f32> = v.iter().map(|value| -*value).collect();
    vhaes_impl(&v_math, u, nlat, nlon, 1, 0, wvha, lwork)
}

fn geo_vhses(
    br: &[f32],
    bi: &[f32],
    cr: &[f32],
    ci: &[f32],
    nlat: usize,
    nlon: usize,
    wvhs: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, i32)> {
    let (v_math, w_math, ierr) = vhses_impl(br, bi, cr, ci, nlat, nlon, 1, 0, wvhs, lwork)?;
    let u = w_math;
    let v = v_math.into_iter().map(|value| -value).collect();
    Ok((u, v, ierr))
}

fn geo_vtses(
    br: &[f32],
    bi: &[f32],
    cr: &[f32],
    ci: &[f32],
    nlat: usize,
    nlon: usize,
    wvts: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, i32)> {
    let (vt_math, wt_math, idv, got_nlon, ierr) =
        vtsgs_impl(br, bi, cr, ci, nlat, 1, 0, wvts, lwork)?;
    if ierr != 0 {
        return Ok((Vec::new(), Vec::new(), ierr));
    }
    if idv != nlat || got_nlon != nlon {
        return Err(PyValueError::new_err("unexpected vtses output shape"));
    }
    let ut = wt_math.into_iter().map(|value| -value).collect();
    Ok((ut, vt_math, 0))
}

fn geo_grades(
    a: &[f32],
    b: &[f32],
    nlat: usize,
    wvhs: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, i32)> {
    let (v_math, w_math, ierr) = grades_impl(a, b, nlat, 1, 0, wvhs, lwork)?;
    let gpdl = w_math;
    let gpdt = v_math.into_iter().map(|value| -value).collect();
    Ok((gpdl, gpdt, ierr))
}

type ShallowOut = (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    [f32; 5],
    i32,
);

fn shallow_impl(
    nlat: usize,
    nlon: usize,
    mmode: usize,
    itmax: usize,
    dt: f32,
) -> PyResult<ShallowOut> {
    if nlat < 3 {
        return Ok((
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            [0.0; 5],
            1,
        ));
    }
    if nlon < 4 {
        return Ok((
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            [0.0; 5],
            2,
        ));
    }
    if dt <= 0.0 {
        return Ok((
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            [0.0; 5],
            3,
        ));
    }
    if mmode + 2 > nlat + 1 {
        return Ok((
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            [0.0; 5],
            4,
        ));
    }

    let lscalar = scalar_saved_len(nlat, nlon);
    let lvector = vector_saved_len(nlat, nlon);
    let lwork_init = init_work_len(nlat, nlon);
    let ldwork_scalar = nlat + 1;
    let ldwork_vector = 2 * (nlat + 1);
    let lwork = low_level_lwork(nlat, nlon).max((2 + 1) * nlat * nlon);

    let (wsha, ierr_sha_i) = shaesi_impl(
        nlat as i32,
        nlon as i32,
        lscalar as i32,
        lwork_init as i32,
        ldwork_scalar as i32,
    );
    let (wshs, ierr_shs_i) = shsesi_impl(
        nlat as i32,
        nlon as i32,
        lscalar as i32,
        lwork_init as i32,
        ldwork_scalar as i32,
    );
    let (wvha, ierr_vha_i) = vhaesi_impl(
        nlat as i32,
        nlon as i32,
        lvector as i32,
        lwork_init as i32,
        ldwork_vector as i32,
    );
    let (wvhs, ierr_vhs_i) = vhsesi_impl(
        nlat as i32,
        nlon as i32,
        lvector as i32,
        lwork_init as i32,
        ldwork_vector as i32,
    );
    let (wvts, ierr_vts_i) = vtsesi_impl(
        nlat as i32,
        nlon as i32,
        lvector as i32,
        ldwork_vector as i32,
    );
    if ierr_sha_i != 0 || ierr_shs_i != 0 || ierr_vha_i != 0 || ierr_vhs_i != 0 || ierr_vts_i != 0 {
        return Err(PyValueError::new_err(format!(
            "init failed: shaesi={ierr_sha_i}, shsesi={ierr_shs_i}, vhaesi={ierr_vha_i}, vhsesi={ierr_vhs_i}, vtsesi={ierr_vts_i}"
        )));
    }

    let (mut u, mut v, mut p, f, vmax, pmax, v2max, p2max) = build_initial_fields(nlat, nlon)?;
    let uxact = u.clone();
    let vxact = v.clone();
    let pxact = p.clone();
    let mut uold = vec![0.0_f32; nlat * nlon];
    let mut vold = vec![0.0_f32; nlat * nlon];
    let mut pold = vec![0.0_f32; nlat * nlon];
    let mut metrics = [0.0_f32; 5];
    let pzero = 2.94e4_f32;
    let aa = 6.37122e6_f32;
    let tdt = dt + dt;

    for ncycle in 0..=itmax {
        let (mut br, mut bi, mut cr, mut ci, ierr) = geo_vhaes(&u, &v, nlat, nlon, &wvha, lwork)?;
        if ierr != 0 {
            return Ok((u, v, p, uxact, vxact, pxact, metrics, ierr));
        }
        truncate_coeffs(nlat, mmode, &mut br, &mut bi);
        truncate_coeffs(nlat, mmode, &mut cr, &mut ci);
        let (uu, vv, ierr) = geo_vhses(&br, &bi, &cr, &ci, nlat, nlon, &wvhs, lwork)?;
        if ierr != 0 {
            return Ok((u, v, p, uxact, vxact, pxact, metrics, ierr));
        }
        u = uu;
        v = vv;

        let (mut a, mut b, ierr) = shaes_impl(&p, nlat, nlon, 1, &wsha, lwork)?;
        if ierr != 0 {
            return Ok((u, v, p, uxact, vxact, pxact, metrics, ierr));
        }
        truncate_coeffs(nlat, mmode, &mut a, &mut b);
        let (pp, ierr) = shses_impl(&a, &b, nlat, 1, &wshs, lwork)?;
        if ierr != 0 {
            return Ok((u, v, p, uxact, vxact, pxact, metrics, ierr));
        }
        p = pp;

        let (vort, ierr) = vrtes_impl(nlon, &cr, &ci, nlat, 0, 1, &wshs, lwork)?;
        if ierr != 0 {
            return Ok((u, v, p, uxact, vxact, pxact, metrics, ierr));
        }
        let (divg, ierr) = dives_impl(nlon, &br, &bi, nlat, 0, 1, &wshs, lwork)?;
        if ierr != 0 {
            return Ok((u, v, p, uxact, vxact, pxact, metrics, ierr));
        }
        let (ut, vt, ierr) = geo_vtses(&br, &bi, &cr, &ci, nlat, nlon, &wvts, lwork)?;
        if ierr != 0 {
            return Ok((u, v, p, uxact, vxact, pxact, metrics, ierr));
        }
        let (gpdl, gpdt, ierr) = geo_grades(&a, &b, nlat, &wvhs, lwork)?;
        if ierr != 0 {
            return Ok((u, v, p, uxact, vxact, pxact, metrics, ierr));
        }

        let mut dudt = vec![0.0_f32; nlat * nlon];
        let mut dvdt = vec![0.0_f32; nlat * nlon];
        let mut dpdt = vec![0.0_f32; nlat * nlon];
        for k in 0..u.len() {
            dudt[k] = (u[k] * (vt[k] - divg[k]) - v[k] * ut[k] - gpdl[k]) / aa + f[k] * v[k];
            dvdt[k] = -(u[k] * (vort[k] + ut[k]) + v[k] * vt[k] + gpdt[k]) / aa - f[k] * u[k];
            dpdt[k] = -((p[k] + pzero) * divg[k] + v[k] * gpdt[k] + u[k] * gpdl[k]) / aa;
        }

        let mut dvgm = 0.0_f32;
        let mut dvmax = 0.0_f32;
        let mut dpmax = 0.0_f32;
        let mut evmax = 0.0_f32;
        let mut epmax = 0.0_f32;
        for k in 0..u.len() {
            dvgm = dvgm.max(divg[k].abs());
            dvmax += (u[k] - uxact[k]).powi(2) + (v[k] - vxact[k]).powi(2);
            dpmax += (p[k] - pxact[k]).powi(2);
            evmax = evmax
                .max((v[k] - vxact[k]).abs())
                .max((u[k] - uxact[k]).abs());
            epmax = epmax.max((p[k] - pxact[k]).abs());
        }
        metrics = [
            evmax / vmax,
            epmax / pmax,
            (dvmax / v2max).sqrt(),
            (dpmax / p2max).sqrt(),
            dvgm,
        ];

        if ncycle == 0 {
            uold.clone_from(&u);
            vold.clone_from(&v);
            pold.clone_from(&p);
        }

        let mut unew = vec![0.0_f32; nlat * nlon];
        let mut vnew = vec![0.0_f32; nlat * nlon];
        let mut pnew = vec![0.0_f32; nlat * nlon];
        for k in 0..u.len() {
            unew[k] = uold[k] + tdt * dudt[k];
            vnew[k] = vold[k] + tdt * dvdt[k];
            pnew[k] = pold[k] + tdt * dpdt[k];
        }
        uold = u;
        vold = v;
        pold = p;
        u = unew;
        v = vnew;
        p = pnew;
    }

    Ok((uold, vold, pold, uxact, vxact, pxact, metrics, 0))
}

#[pyfunction]
#[pyo3(signature = (nlat=65, nlon=128))]
pub fn shallow_initial<'py>(
    py: Python<'py>,
    nlat: usize,
    nlon: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let (u, v, p, f, ..) = build_initial_fields(nlat, nlon)?;
    let shape = IxDyn(&[nlat, nlon]);
    let u_arr = ArrayD::from_shape_vec(shape.clone(), u)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let v_arr = ArrayD::from_shape_vec(shape.clone(), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let p_arr = ArrayD::from_shape_vec(shape.clone(), p)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let f_arr =
        ArrayD::from_shape_vec(shape, f).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        u_arr.into_pyarray(py).into_any().unbind(),
        v_arr.into_pyarray(py).into_any().unbind(),
        p_arr.into_pyarray(py).into_any().unbind(),
        f_arr.into_pyarray(py).into_any().unbind(),
        0,
    ))
}

#[pyfunction]
#[pyo3(signature = (nlat=65, nlon=128, mmode=42, itmax=720, dt=600.0))]
pub fn shallow<'py>(
    py: Python<'py>,
    nlat: usize,
    nlon: usize,
    mmode: usize,
    itmax: usize,
    dt: f32,
) -> PyResult<(
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    i32,
)> {
    let (u, v, p, uxact, vxact, pxact, metrics, ierr) = shallow_impl(nlat, nlon, mmode, itmax, dt)?;
    let field_shape = if ierr == 0 {
        IxDyn(&[nlat, nlon])
    } else {
        IxDyn(&[0usize, 0usize])
    };
    let u_arr = ArrayD::from_shape_vec(field_shape.clone(), u)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let v_arr = ArrayD::from_shape_vec(field_shape.clone(), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let p_arr = ArrayD::from_shape_vec(field_shape.clone(), p)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let ux_arr = ArrayD::from_shape_vec(field_shape.clone(), uxact)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let vx_arr = ArrayD::from_shape_vec(field_shape.clone(), vxact)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let px_arr = ArrayD::from_shape_vec(field_shape, pxact)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let metrics_arr = ArrayD::from_shape_vec(IxDyn(&[5usize]), metrics.to_vec())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        u_arr.into_pyarray(py).into_any().unbind(),
        v_arr.into_pyarray(py).into_any().unbind(),
        p_arr.into_pyarray(py).into_any().unbind(),
        ux_arr.into_pyarray(py).into_any().unbind(),
        vx_arr.into_pyarray(py).into_any().unbind(),
        px_arr.into_pyarray(py).into_any().unbind(),
        metrics_arr.into_pyarray(py).into_any().unbind(),
        ierr,
    ))
}
