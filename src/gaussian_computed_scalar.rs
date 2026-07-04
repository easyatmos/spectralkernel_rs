use crate::gaussian_stored_scalar::shagsp_impl;

pub fn shagci_core(nlat: usize, nlon: usize) -> Result<Vec<f32>, i32> {
    shagsp_impl(nlat, nlon)
}

pub fn shsgci_core(nlat: usize, nlon: usize) -> Result<Vec<f32>, i32> {
    shagsp_impl(nlat, nlon)
}
