use crate::gaqd::gaqd_impl;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridType {
    Regular,
    Gaussian,
}

impl GridType {
    pub fn as_str(self) -> &'static str {
        match self {
            GridType::Regular => "regular",
            GridType::Gaussian => "gaussian",
        }
    }
}

pub fn regular_latitudes(nlat: usize) -> Vec<f32> {
    if nlat % 2 == 1 {
        (0..nlat)
            .map(|i| 90.0_f32 - 180.0_f32 * i as f32 / (nlat as f32 - 1.0_f32))
            .collect()
    } else {
        let dlat = 180.0_f32 / nlat as f32;
        (0..nlat).map(|i| 90.0_f32 - 0.5_f32 * dlat - dlat * i as f32).collect()
    }
}

pub fn gaussian_latitudes_weights(nlat: usize) -> (Vec<f32>, Vec<f32>, i32) {
    let (theta, weights, ierr) = gaqd_impl(nlat as i32);
    let lat = theta
        .into_iter()
        .map(|theta| 90.0_f32 - (theta as f32).to_degrees())
        .collect();
    let weights = weights.into_iter().map(|w| w as f32).collect();
    (lat, weights, ierr)
}

pub fn longitudes(nlon: usize) -> Vec<f32> {
    (0..nlon)
        .map(|i| 360.0_f32 * i as f32 / nlon as f32)
        .collect()
}
