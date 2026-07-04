use crate::grid::{gaussian_latitudes_weights, longitudes, regular_latitudes, GridType};
use crate::shaeci::shaeci_impl;
use crate::shaesi::shaesi_impl;
use crate::shagci::shagci_impl;
use crate::shagsi::shagsi_impl;
use crate::shseci::shseci_impl;
use crate::shsesi::shsesi_impl;
use crate::shsgci::shsgci_impl;
use crate::shsgsi::shsgsi_impl;
use crate::vhaeci::vhaeci_impl;
use crate::vhaesi::vhaesi_impl_parallel;
use crate::vhagci::vhagci_impl;
use crate::vhagsi::vhagsi_impl;
use crate::vhseci::vhseci_impl;
use crate::vhsesi::vhsesi_impl_parallel;
use crate::vhsgci::vhsgci_impl;
use crate::vhsgsi::vhsgsi_impl;
use crate::vtsec::vtseci_impl;
use crate::vtses::vtsesi_impl;
use crate::vtsgc::vtsgci_impl;
use crate::vtsgs::vtsgsi_impl;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LegFunc {
    Stored,
    Computed,
}

impl LegFunc {
    pub fn parse(value: &str) -> PyResult<Self> {
        match value {
            "stored" => Ok(Self::Stored),
            "computed" => Ok(Self::Computed),
            _ => Err(PyValueError::new_err(
                "legfunc must be either 'stored' or 'computed'",
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stored => "stored",
            Self::Computed => "computed",
        }
    }
}

#[derive(Clone)]
pub struct SpectralPlan {
    pub nlat: usize,
    pub nlon: usize,
    pub radius: f32,
    pub grid_type: GridType,
    pub legfunc: LegFunc,
    pub lat: Vec<f32>,
    pub lon: Vec<f32>,
    pub weights: Option<Vec<f32>>,
    pub scalar_analysis_work: Vec<f32>,
    pub scalar_synthesis_work: Vec<f32>,
    pub vector_analysis_work: Vec<f32>,
    pub vector_synthesis_work: Vec<f32>,
    pub vector_vts_work: Vec<f32>,
}

impl SpectralPlan {
    pub fn regular(nlat: usize, nlon: usize, radius: f32, legfunc: LegFunc) -> PyResult<Self> {
        validate_grid(nlat, nlon, radius)?;
        let n1 = if nlon % 2 == 1 {
            nlat.min((nlon + 1) / 2)
        } else {
            nlat.min((nlon + 2) / 2)
        };
        let n2 = if nlat % 2 == 1 {
            (nlat + 1) / 2
        } else {
            nlat / 2
        };
        let lscalar = n1 * n2 * (2 * nlat - n1 + 1) / 2 + nlon + 15;
        let labc = 3 * n1.saturating_sub(2) * (2 * nlat - n1 - 1) / 2;
        let scalar_init_work = 5 * nlat * n2 + labc;
        let lvector = n1 * n2 * (2 * nlat - n1 + 1) + nlon + 15;
        let vector_init_work = 3 * n1.saturating_sub(2) * (2 * nlat - n1 - 1) / 2 + 5 * n2 * nlat;

        let (wsha, wshs, wvha, wvhs, wvts) = match legfunc {
            LegFunc::Stored => {
                let (wsha, ierr_sha) = shaesi_impl(
                    nlat as i32,
                    nlon as i32,
                    lscalar as i32,
                    scalar_init_work as i32,
                    (nlat + 1) as i32,
                );
                check_init("shaesi", ierr_sha)?;
                let (wshs, ierr_shs) = shsesi_impl(
                    nlat as i32,
                    nlon as i32,
                    lscalar as i32,
                    scalar_init_work as i32,
                    (nlat + 1) as i32,
                );
                check_init("shsesi", ierr_shs)?;
                let (wvha, ierr_vha) = vhaesi_impl_parallel(
                    nlat as i32,
                    nlon as i32,
                    lvector as i32,
                    vector_init_work as i32,
                    (2 * (nlat + 1)) as i32,
                );
                check_init("vhaesi", ierr_vha)?;
                let (wvhs, ierr_vhs) = vhsesi_impl_parallel(
                    nlat as i32,
                    nlon as i32,
                    lvector as i32,
                    vector_init_work as i32,
                    (2 * (nlat + 1)) as i32,
                );
                check_init("vhsesi", ierr_vhs)?;
                let (wvts, ierr_vts) = vtsesi_impl(
                    nlat as i32,
                    nlon as i32,
                    lvector as i32,
                    (2 * (nlat + 1)) as i32,
                );
                check_init("vtsesi", ierr_vts)?;
                (wsha, wshs, wvha, wvhs, wvts)
            }
            LegFunc::Computed => {
                let n1_vector = if nlon % 2 == 0 {
                    nlat.min(nlon / 2)
                } else {
                    nlat.min((nlon + 1) / 2)
                };
                let lscalar_c =
                    2 * nlat * n2 + 3 * n1.saturating_sub(2) * (2 * nlat - n1 - 1) / 2 + nlon + 15;
                let lvector_c = 4 * nlat * n2
                    + 3 * n1_vector.saturating_sub(2) * (2 * nlat - n1_vector - 1)
                    + nlon
                    + 15;
                let (wsha, ierr_sha) = shaeci_impl(
                    nlat as i32,
                    nlon as i32,
                    lscalar_c as i32,
                    (2 * (nlat + 1)) as i32,
                );
                check_init("shaeci", ierr_sha)?;
                let (wshs, ierr_shs) = shseci_impl(
                    nlat as i32,
                    nlon as i32,
                    lscalar_c as i32,
                    (2 * (nlat + 1)) as i32,
                );
                check_init("shseci", ierr_shs)?;
                let (wvha, ierr_vha) = vhaeci_impl(
                    nlat as i32,
                    nlon as i32,
                    lvector_c as i32,
                    (2 * (nlat + 1)) as i32,
                );
                check_init("vhaeci", ierr_vha)?;
                let (wvhs, ierr_vhs) = vhseci_impl(
                    nlat as i32,
                    nlon as i32,
                    lvector_c as i32,
                    (2 * (nlat + 1)) as i32,
                );
                check_init("vhseci", ierr_vhs)?;
                let (wvts, ierr_vts) = vtseci_impl(
                    nlat as i32,
                    nlon as i32,
                    lvector_c as i32,
                    (2 * nlat + 2) as i32,
                );
                check_init("vtseci", ierr_vts)?;
                (wsha, wshs, wvha, wvhs, wvts)
            }
        };

        Ok(Self {
            nlat,
            nlon,
            radius,
            grid_type: GridType::Regular,
            legfunc,
            lat: regular_latitudes(nlat),
            lon: longitudes(nlon),
            weights: None,
            scalar_analysis_work: wsha,
            scalar_synthesis_work: wshs,
            vector_analysis_work: wvha,
            vector_synthesis_work: wvhs,
            vector_vts_work: wvts,
        })
    }

    pub fn gaussian(nlat: usize, nlon: usize, radius: f32, legfunc: LegFunc) -> PyResult<Self> {
        validate_grid(nlat, nlon, radius)?;
        let n1 = nlat.min((nlon + 2) / 2);
        let n2 = (nlat + 1) / 2;
        let lscalar = (nlat as isize) * (3 * (n1 + n2) - 2) as isize
            + (n1 - 1) as isize * ((n2 * (2 * nlat - n1)) as isize - (3 * n1) as isize) / 2
            + nlon as isize
            + 15;
        let lscalar = usize::try_from(lscalar)
            .map_err(|_| PyValueError::new_err("gaussian scalar workspace length is negative"))?;
        let scalar_init_work = 4 * nlat * (nlat + 2) + 2;
        let scalar_dwork = nlat * (nlat + 4);
        let lvhags = (nlat + 1) * (nlat + 1) * nlat / 2 + nlon + 15;
        let vector_dwork = (3 * nlat * (nlat + 3) + 2) / 2;
        let lmn = nlat * (nlat + 1) / 2;
        let lvhsgs = 2 * n2 * lmn + nlon + 15;

        let (wsha, wshs, wvha, wvhs, wvts) = match legfunc {
            LegFunc::Stored => {
                let (wsha, ierr_sha) = shagsi_impl(
                    nlat as i32,
                    nlon as i32,
                    lscalar as i32,
                    scalar_init_work as i32,
                    scalar_dwork as i32,
                );
                check_init("shagsi", ierr_sha)?;
                let (wshs, ierr_shs) = shsgsi_impl(
                    nlat as i32,
                    nlon as i32,
                    lscalar as i32,
                    scalar_init_work as i32,
                    scalar_dwork as i32,
                );
                check_init("shsgsi", ierr_shs)?;
                let (wvha, ierr_vha) =
                    vhagsi_impl(nlat as i32, nlon as i32, lvhags as i32, vector_dwork as i32);
                check_init("vhagsi", ierr_vha)?;
                let (wvhs, ierr_vhs) =
                    vhsgsi_impl(nlat as i32, nlon as i32, lvhsgs as i32, vector_dwork as i32);
                check_init("vhsgsi", ierr_vhs)?;
                let (wvts, ierr_vts) =
                    vtsgsi_impl(nlat as i32, nlon as i32, lvhsgs as i32, vector_dwork as i32);
                check_init("vtsgsi", ierr_vts)?;
                (wsha, wshs, wvha, wvhs, wvts)
            }
            LegFunc::Computed => {
                let lscalar_c = nlat * (2 * n2 + 3 * n1 - 2) - 3 * n1 * (n1 - 1) / 2 + nlon + 15;
                let lvhagc =
                    4 * nlat * n2 + 3 * n1.saturating_sub(2) * (2 * nlat - n1 - 1) + nlon + n2 + 15;
                let lvhsgc =
                    4 * nlat * n2 + 3 * n1.saturating_sub(2) * (2 * nlat - n1 - 1) + nlon + 15;
                let vector_c_dwork = 2 * nlat * (nlat + 1) + 1;
                let (wsha, ierr_sha) = shagci_impl(
                    nlat as i32,
                    nlon as i32,
                    lscalar_c as i32,
                    scalar_dwork as i32,
                );
                check_init("shagci", ierr_sha)?;
                let (wshs, ierr_shs) = shsgci_impl(
                    nlat as i32,
                    nlon as i32,
                    lscalar_c as i32,
                    scalar_dwork as i32,
                );
                check_init("shsgci", ierr_shs)?;
                let (wvha, ierr_vha) = vhagci_impl(
                    nlat as i32,
                    nlon as i32,
                    lvhagc as i32,
                    vector_c_dwork as i32,
                );
                check_init("vhagci", ierr_vha)?;
                let (wvhs, ierr_vhs) = vhsgci_impl(
                    nlat as i32,
                    nlon as i32,
                    lvhsgc as i32,
                    vector_c_dwork as i32,
                );
                check_init("vhsgci", ierr_vhs)?;
                let (wvts, ierr_vts) = vtsgci_impl(
                    nlat as i32,
                    nlon as i32,
                    lvhsgc as i32,
                    vector_c_dwork as i32,
                );
                check_init("vtsgci", ierr_vts)?;
                (wsha, wshs, wvha, wvhs, wvts)
            }
        };
        let (lat, weights, ierr_gaqd) = gaussian_latitudes_weights(nlat);
        check_init("gaqd", ierr_gaqd)?;

        Ok(Self {
            nlat,
            nlon,
            radius,
            grid_type: GridType::Gaussian,
            legfunc,
            lat,
            lon: longitudes(nlon),
            weights: Some(weights),
            scalar_analysis_work: wsha,
            scalar_synthesis_work: wshs,
            vector_analysis_work: wvha,
            vector_synthesis_work: wvhs,
            vector_vts_work: wvts,
        })
    }
}

fn validate_grid(nlat: usize, nlon: usize, radius: f32) -> PyResult<()> {
    if nlat < 3 {
        return Err(PyValueError::new_err("nlat must be at least 3"));
    }
    if nlon < 4 {
        return Err(PyValueError::new_err("nlon must be at least 4"));
    }
    if radius <= 0.0 {
        return Err(PyValueError::new_err("radius must be positive"));
    }
    Ok(())
}

fn check_init(name: &str, ierr: i32) -> PyResult<()> {
    if ierr == 0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} initialization failed with ierror={ierr}"
        )))
    }
}
