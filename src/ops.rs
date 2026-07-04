use crate::divec::divec_impl;
use crate::dives::dives_impl;
use crate::divgc::divgc_impl;
use crate::divgs::divgs_impl;
use crate::geo2math::{geo2maths_impl, geo2mathv_impl, math2geos_impl, math2geov_impl};
use crate::gradec::gradec_impl;
use crate::grades::grades_impl;
use crate::gradgc::gradgc_impl;
use crate::gradgs::gradgs_impl;
use crate::grid::GridType;
use crate::helmsph::helmsph_impl;
use crate::idivec::idivec_impl;
use crate::idives::idives_impl;
use crate::idivgc::idivgc_impl;
use crate::idivgs::idivgs_impl;
use crate::idvtec::idvtec_impl;
use crate::idvtes::idvtes_impl;
use crate::idvtgc::idvtgc_impl;
use crate::idvtgs::idvtgs_impl;
use crate::igradec::igradec_impl;
use crate::igrades::igrades_impl;
use crate::igradgc::igradgc_impl;
use crate::igradgs::igradgs_impl;
use crate::isfvpec::isfvpec_impl;
use crate::isfvpes::isfvpes_impl;
use crate::isfvpgc::isfvpgc_impl;
use crate::isfvpgs::isfvpgs_impl;
use crate::islapec::islapec_impl;
use crate::islapes::islapes_impl;
use crate::islapgc::islapgc_impl;
use crate::islapgs::islapgs_impl;
use crate::ivlapec::ivlapec_impl;
use crate::ivlapes::ivlapes_impl;
use crate::ivlapgc::ivlapgc_impl;
use crate::ivlapgs::ivlapgs_impl;
use crate::ivrtec::ivrtec_impl;
use crate::ivrtes::ivrtes_impl;
use crate::ivrtgc::ivrtgc_impl;
use crate::ivrtgs::ivrtgs_impl;
use crate::onedtotwod::onedtotwod_impl;
use crate::sfvpec::sfvpec_impl;
use crate::sfvpes::sfvpes_impl;
use crate::sfvpgc::sfvpgc_impl;
use crate::sfvpgs::sfvpgs_impl;
use crate::shaec::shaec_impl_parallel;
use crate::shaes::shaes_impl_parallel;
use crate::shagc::shagc_impl_parallel;
use crate::shags::shags_impl;
use crate::shsec::shsec_impl_parallel;
use crate::shses::shses_impl_parallel;
use crate::shsgc::shsgc_impl_parallel;
use crate::shsgs::shsgs_impl;
use crate::slapec::slapec_impl;
use crate::slapes::slapes_impl;
use crate::slapgc::slapgc_impl;
use crate::slapgs::slapgs_impl;
use crate::spectral_plan::{LegFunc, SpectralPlan};
use crate::sshifte::{sshifte_impl, sshifti_impl};
use crate::twodtooned::twodtooned_impl;
use crate::twodtooned_vrtdiv::twodtooned_vrtdiv_impl;
use crate::vhaec::vhaec_impl_parallel;
use crate::vhaes::vhaes_impl;
use crate::vhagc::{vhagc_impl, vhagc_impl_latpar};
use crate::vhags::{vhags_impl_latpar, vhags_impl_parallel};
use crate::vlapec::vlapec_impl;
use crate::vlapes::vlapes_impl;
use crate::vlapgc::vlapgc_impl;
use crate::vlapgs::vlapgs_impl;
use crate::vrtec::vrtec_impl;
use crate::vrtes::vrtes_impl;
use crate::vrtgc::vrtgc_impl;
use crate::vrtgs::vrtgs_impl;
use crate::vshifte::{vshifte_impl, vshifti_impl};
use crate::vtsgc::vtsgc_impl;
use crate::vtsgs::vtsgs_impl;
use ndarray::{ArrayD, IxDyn};
use num_complex::Complex32;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TransformBackend {
    Rayon,
    LatPar,
}

impl TransformBackend {
    fn parse(value: &str) -> PyResult<Self> {
        let value = value.to_ascii_lowercase();
        match value.as_str() {
            "parallel" | "rayon" | "threaded" | "serial" | "series" | "single"
            | "single_thread" | "nogil" | "no_gil" | "python_free" => Ok(Self::Rayon),
            "latpar" | "lat" | "latitude" | "latitude_parallel" => Ok(Self::LatPar),
            _ => Err(PyValueError::new_err(
                "backend must be 'rayon', 'serial', 'nogil', or 'latpar'",
            )),
        }
    }
}

#[pyclass]
pub struct SphereOps {
    pub(crate) plan: SpectralPlan,
}

#[pymethods]
impl SphereOps {
    #[staticmethod]
    #[pyo3(signature = (nlat, nlon, radius=6.3712e6_f32, legfunc="stored"))]
    pub fn regular(nlat: usize, nlon: usize, radius: f32, legfunc: &str) -> PyResult<Self> {
        let legfunc = LegFunc::parse(legfunc)?;
        Ok(Self {
            plan: SpectralPlan::regular(nlat, nlon, radius, legfunc)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (nlat, nlon, radius=6.3712e6_f32, legfunc="stored"))]
    pub fn gaussian(nlat: usize, nlon: usize, radius: f32, legfunc: &str) -> PyResult<Self> {
        let legfunc = LegFunc::parse(legfunc)?;
        Ok(Self {
            plan: SpectralPlan::gaussian(nlat, nlon, radius, legfunc)?,
        })
    }

    #[getter]
    pub fn nlat(&self) -> usize {
        self.plan.nlat
    }

    #[getter]
    pub fn nlon(&self) -> usize {
        self.plan.nlon
    }

    #[getter]
    pub fn radius(&self) -> f32 {
        self.plan.radius
    }

    #[getter]
    pub fn grid_type(&self) -> &'static str {
        self.plan.grid_type.as_str()
    }

    #[getter]
    pub fn legfunc(&self) -> &'static str {
        self.plan.legfunc.as_str()
    }

    pub fn lat<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        vec_to_py(py, &[self.plan.nlat], self.plan.lat.clone())
    }

    pub fn lon<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        vec_to_py(py, &[self.plan.nlon], self.plan.lon.clone())
    }

    pub fn weights<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let weights = self
            .plan
            .weights
            .clone()
            .unwrap_or_else(|| vec![1.0_f32; self.plan.nlat]);
        vec_to_py(py, &[self.plan.nlat], weights)
    }

    #[pyo3(signature = (sg, ig=0))]
    pub fn geo_to_math_scalar<'py>(
        &self,
        py: Python<'py>,
        sg: PyReadonlyArrayDyn<'py, f32>,
        ig: i32,
    ) -> PyResult<Py<PyAny>> {
        validate_rank2_shape(&sg.shape(), self.plan.nlon, self.plan.nlat, "geo scalar")?;
        let data = sg.as_slice()?.to_vec();
        let sm = detach_pyresult(py, || {
            geo2maths_impl(ig, &data, self.plan.nlon, self.plan.nlat)
        })?;
        vec_to_py(py, &[self.plan.nlat, self.plan.nlon], sm)
    }

    #[pyo3(signature = (sm, ig=0))]
    pub fn math_to_geo_scalar<'py>(
        &self,
        py: Python<'py>,
        sm: PyReadonlyArrayDyn<'py, f32>,
        ig: i32,
    ) -> PyResult<Py<PyAny>> {
        validate_rank2_shape(&sm.shape(), self.plan.nlat, self.plan.nlon, "math scalar")?;
        let data = sm.as_slice()?.to_vec();
        let sg = detach_pyresult(py, || {
            math2geos_impl(ig, &data, self.plan.nlat, self.plan.nlon)
        })?;
        vec_to_py(py, &[self.plan.nlon, self.plan.nlat], sg)
    }

    #[pyo3(signature = (ug, vg, ig=0))]
    pub fn geo_to_math_vector<'py>(
        &self,
        py: Python<'py>,
        ug: PyReadonlyArrayDyn<'py, f32>,
        vg: PyReadonlyArrayDyn<'py, f32>,
        ig: i32,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        validate_rank2_shape(&ug.shape(), self.plan.nlon, self.plan.nlat, "geo u")?;
        validate_rank2_shape(&vg.shape(), self.plan.nlon, self.plan.nlat, "geo v")?;
        let udata = ug.as_slice()?.to_vec();
        let vdata = vg.as_slice()?.to_vec();
        let (vm, wm) = detach_pyresult(py, || {
            geo2mathv_impl(ig, &udata, &vdata, self.plan.nlon, self.plan.nlat)
        })?;
        Ok((
            vec_to_py(py, &[self.plan.nlat, self.plan.nlon], vm)?,
            vec_to_py(py, &[self.plan.nlat, self.plan.nlon], wm)?,
        ))
    }

    #[pyo3(signature = (vm, wm, ig=0))]
    pub fn math_to_geo_vector<'py>(
        &self,
        py: Python<'py>,
        vm: PyReadonlyArrayDyn<'py, f32>,
        wm: PyReadonlyArrayDyn<'py, f32>,
        ig: i32,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        validate_rank2_shape(&vm.shape(), self.plan.nlat, self.plan.nlon, "math v")?;
        validate_rank2_shape(&wm.shape(), self.plan.nlat, self.plan.nlon, "math w")?;
        let vdata = vm.as_slice()?.to_vec();
        let wdata = wm.as_slice()?.to_vec();
        let (ug, vg) = detach_pyresult(py, || {
            math2geov_impl(ig, &vdata, &wdata, self.plan.nlat, self.plan.nlon)
        })?;
        Ok((
            vec_to_py(py, &[self.plan.nlon, self.plan.nlat], ug)?,
            vec_to_py(py, &[self.plan.nlon, self.plan.nlat], vg)?,
        ))
    }

    #[pyo3(signature = (data, ioff=0))]
    pub fn scalar_shift<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArrayDyn<'py, f32>,
        ioff: i32,
    ) -> PyResult<Py<PyAny>> {
        let nlat_in = if ioff == 0 {
            self.plan.nlat
        } else {
            self.plan.nlat + 1
        };
        validate_rank2_shape(&data.shape(), self.plan.nlon, nlat_in, "scalar shift input")?;
        let lsav = shift_lsav(self.plan.nlon, self.plan.nlat);
        let (wsav, ierr_init) = sshifti_impl(
            ioff,
            self.plan.nlon as i32,
            self.plan.nlat as i32,
            lsav as i32,
        );
        check_ierror("sshifti", ierr_init)?;
        let input = data.as_slice()?.to_vec();
        let (shifted, ierr) = detach_pyresult(py, || {
            sshifte_impl(
                ioff,
                &input,
                self.plan.nlon,
                self.plan.nlat,
                &wsav,
                shift_lwork(self.plan.nlon, self.plan.nlat),
            )
        })?;
        check_ierror("sshifte", ierr)?;
        let out_nlat = if ioff == 0 {
            self.plan.nlat + 1
        } else {
            self.plan.nlat
        };
        vec_to_py(py, &[self.plan.nlon, out_nlat], shifted)
    }

    #[pyo3(signature = (u, v, ioff=0))]
    pub fn vector_shift<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
        ioff: i32,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let nlat_in = if ioff == 0 {
            self.plan.nlat
        } else {
            self.plan.nlat + 1
        };
        validate_rank2_shape(&u.shape(), self.plan.nlon, nlat_in, "vector shift u")?;
        validate_rank2_shape(&v.shape(), self.plan.nlon, nlat_in, "vector shift v")?;
        let lsav = shift_lsav(self.plan.nlon, self.plan.nlat);
        let (wsav, ierr_init) = vshifti_impl(
            ioff,
            self.plan.nlon as i32,
            self.plan.nlat as i32,
            lsav as i32,
        );
        check_ierror("vshifti", ierr_init)?;
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let (u_shifted, v_shifted, ierr) = detach_pyresult(py, || {
            vshifte_impl(
                ioff,
                &udata,
                &vdata,
                self.plan.nlon,
                self.plan.nlat,
                &wsav,
                shift_lwork(self.plan.nlon, self.plan.nlat),
            )
        })?;
        check_ierror("vshifte", ierr)?;
        let out_nlat = if ioff == 0 {
            self.plan.nlat + 1
        } else {
            self.plan.nlat
        };
        Ok((
            vec_to_py(py, &[self.plan.nlon, out_nlat], u_shifted)?,
            vec_to_py(py, &[self.plan.nlon, out_nlat], v_shifted)?,
        ))
    }

    #[pyo3(signature = (f, truncation=None, backend="rayon"))]
    pub fn scalar_to_spec<'py>(
        &self,
        py: Python<'py>,
        f: PyReadonlyArrayDyn<'py, f32>,
        truncation: Option<usize>,
        backend: &str,
    ) -> PyResult<Py<PyAny>> {
        TransformBackend::parse(backend)?;
        let shape = f.shape().to_vec();
        let data = f.as_slice()?.to_vec();
        let (spec, nt, was_2d) = py
            .detach(|| {
                self.scalar_to_spec_vec(&data, &shape, truncation)
                    .map_err(|err| err.to_string())
            })
            .map_err(PyValueError::new_err)?;
        let ntrunc = truncation.unwrap_or(self.plan.nlat - 1);
        let nmdim = (ntrunc + 1) * (ntrunc + 2) / 2;
        if was_2d {
            complex_to_py(py, &[nmdim], spec)
        } else {
            complex_to_py(py, &[nmdim, nt], spec)
        }
    }

    #[pyo3(signature = (spec, backend="rayon"))]
    pub fn spec_to_scalar<'py>(
        &self,
        py: Python<'py>,
        spec: PyReadonlyArrayDyn<'py, Complex32>,
        backend: &str,
    ) -> PyResult<Py<PyAny>> {
        TransformBackend::parse(backend)?;
        let shape = spec.shape().to_vec();
        let data = spec.as_slice()?.to_vec();
        let (grid, nt, was_1d) = py
            .detach(|| {
                self.spec_to_scalar_vec(&data, &shape)
                    .map_err(|err| err.to_string())
            })
            .map_err(PyValueError::new_err)?;
        if was_1d {
            vec_to_py(py, &[self.plan.nlat, self.plan.nlon], grid)
        } else {
            vec_to_py(py, &[self.plan.nlat, self.plan.nlon, nt], grid)
        }
    }

    #[pyo3(signature = (u, v, truncation=None, backend="rayon"))]
    pub fn vector_to_spec<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
        truncation: Option<usize>,
        backend: &str,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let backend = TransformBackend::parse(backend)?;
        let ushape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let (vrt, div, nt, was_2d) = py
            .detach(|| {
                self.vector_to_spec_vec_backend(
                    &udata, &ushape, &vdata, &vshape, truncation, backend,
                )
                .map_err(|err| err.to_string())
            })
            .map_err(PyValueError::new_err)?;
        let ntrunc = truncation.unwrap_or(self.plan.nlat - 1);
        let nmdim = (ntrunc + 1) * (ntrunc + 2) / 2;
        let shape = if was_2d { vec![nmdim] } else { vec![nmdim, nt] };
        Ok((
            complex_to_py(py, &shape, vrt)?,
            complex_to_py(py, &shape, div)?,
        ))
    }

    #[pyo3(signature = (vort_spec, div_spec, backend="rayon"))]
    pub fn spec_to_vector<'py>(
        &self,
        py: Python<'py>,
        vort_spec: PyReadonlyArrayDyn<'py, Complex32>,
        div_spec: PyReadonlyArrayDyn<'py, Complex32>,
        backend: &str,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        TransformBackend::parse(backend)?;
        let vort_shape = vort_spec.shape().to_vec();
        let div_shape = div_spec.shape().to_vec();
        let vort_data = vort_spec.as_slice()?.to_vec();
        let div_data = div_spec.as_slice()?.to_vec();
        let (u, v, nt, was_1d) = py
            .detach(|| {
                self.spec_to_vector_vec(&vort_data, &vort_shape, &div_data, &div_shape)
                    .map_err(|err| err.to_string())
            })
            .map_err(PyValueError::new_err)?;
        let shape = if was_1d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((vec_to_py(py, &shape, u)?, vec_to_py(py, &shape, v)?))
    }

    #[pyo3(signature = (f, truncation=None))]
    pub fn truncate_scalar<'py>(
        &self,
        py: Python<'py>,
        f: PyReadonlyArrayDyn<'py, f32>,
        truncation: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let shape = f.shape().to_vec();
        let data = f.as_slice()?.to_vec();
        let (spec, nt, was_2d) =
            detach_pyresult(py, || self.scalar_to_spec_vec(&data, &shape, truncation))?;
        let ntrunc = truncation.unwrap_or(self.plan.nlat - 1);
        let shape = if was_2d {
            vec![(ntrunc + 1) * (ntrunc + 2) / 2]
        } else {
            vec![(ntrunc + 1) * (ntrunc + 2) / 2, nt]
        };
        let (grid, _, _) = detach_pyresult(py, || self.spec_to_scalar_vec(&spec, &shape))?;
        if was_2d {
            vec_to_py(py, &[self.plan.nlat, self.plan.nlon], grid)
        } else {
            vec_to_py(py, &[self.plan.nlat, self.plan.nlon, nt], grid)
        }
    }

    pub fn grad<'py>(
        &self,
        py: Python<'py>,
        chispec: PyReadonlyArrayDyn<'py, Complex32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        self.gradient_from_spec(py, chispec)
    }

    pub fn gradient<'py>(
        &self,
        py: Python<'py>,
        chispec: PyReadonlyArrayDyn<'py, Complex32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        self.gradient_from_spec(py, chispec)
    }

    pub fn gradient_grid<'py>(
        &self,
        py: Python<'py>,
        f: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let shape = f.shape().to_vec();
        let (_, nt, was_2d) = validate_scalar_shape(&shape, self.plan.nlat, self.plan.nlon)?;
        let data = f.as_slice()?.to_vec();
        let (u, v) = detach_pyresult(py, || self.gradient_grid_vec(&data, &shape))?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((vec_to_py(py, &out_shape, u)?, vec_to_py(py, &out_shape, v)?))
    }

    pub fn gradient_from_spec<'py>(
        &self,
        py: Python<'py>,
        chispec: PyReadonlyArrayDyn<'py, Complex32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let shape = chispec.shape().to_vec();
        let (_, nt, was_1d) = validate_spec_shape(&shape)?;
        let data = chispec.as_slice()?.to_vec();
        let (u, v) = detach_pyresult(py, || self.gradient_from_spec_vec(&data, &shape))?;

        let out_shape = if was_1d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };

        Ok((vec_to_py(py, &out_shape, u)?, vec_to_py(py, &out_shape, v)?))
    }

    pub fn inverse_grad<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Py<PyAny>> {
        self.inverse_gradient(py, u, v)
    }

    pub fn inverse_gradient<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Py<PyAny>> {
        let shape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let (_, nt, was_2d) =
            validate_vector_shapes(&shape, &vshape, self.plan.nlat, self.plan.nlon)?;
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let (sf, _, _) = detach_pyresult(py, || {
            self.inverse_gradient_vec(&udata, &shape, &vdata, &vshape)
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        vec_to_py(py, &out_shape, sf)
    }

    pub fn div<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Py<PyAny>> {
        self.divergence(py, u, v)
    }

    pub fn divergence<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Py<PyAny>> {
        let ushape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let (grid, nt, was_2d) =
            detach_pyresult(py, || self.divergence_vec(&udata, &ushape, &vdata, &vshape))?;
        if was_2d {
            vec_to_py(py, &[self.plan.nlat, self.plan.nlon], grid)
        } else {
            vec_to_py(py, &[self.plan.nlat, self.plan.nlon, nt], grid)
        }
    }

    pub fn inverse_div<'py>(
        &self,
        py: Python<'py>,
        div: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        self.inverse_divergence(py, div)
    }

    pub fn inverse_divergence<'py>(
        &self,
        py: Python<'py>,
        div: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let shape = div.shape().to_vec();
        let data = div.as_slice()?.to_vec();
        let (u, v, nt, was_2d) =
            detach_pyresult(py, || self.inverse_divergence_vec(&data, &shape))?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((vec_to_py(py, &out_shape, u)?, vec_to_py(py, &out_shape, v)?))
    }

    pub fn vort<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Py<PyAny>> {
        self.vorticity(py, u, v)
    }

    pub fn vorticity<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Py<PyAny>> {
        let ushape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let (grid, nt, was_2d) =
            detach_pyresult(py, || self.vorticity_vec(&udata, &ushape, &vdata, &vshape))?;
        if was_2d {
            vec_to_py(py, &[self.plan.nlat, self.plan.nlon], grid)
        } else {
            vec_to_py(py, &[self.plan.nlat, self.plan.nlon, nt], grid)
        }
    }

    pub fn inverse_vort<'py>(
        &self,
        py: Python<'py>,
        vort: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        self.inverse_vorticity(py, vort)
    }

    pub fn inverse_vorticity<'py>(
        &self,
        py: Python<'py>,
        vort: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let shape = vort.shape().to_vec();
        let data = vort.as_slice()?.to_vec();
        let (u, v, nt, was_2d) = detach_pyresult(py, || self.inverse_vorticity_vec(&data, &shape))?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((vec_to_py(py, &out_shape, u)?, vec_to_py(py, &out_shape, v)?))
    }

    pub fn laplacian<'py>(
        &self,
        py: Python<'py>,
        f: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Py<PyAny>> {
        let shape = f.shape().to_vec();
        let data = f.as_slice()?.to_vec();
        let (grid, nt, was_2d) = detach_pyresult(py, || self.laplacian_vec(&data, &shape))?;
        if was_2d {
            vec_to_py(py, &[self.plan.nlat, self.plan.nlon], grid)
        } else {
            vec_to_py(py, &[self.plan.nlat, self.plan.nlon, nt], grid)
        }
    }

    #[pyo3(signature = (f, zero_mean=true))]
    pub fn inverse_laplacian<'py>(
        &self,
        py: Python<'py>,
        f: PyReadonlyArrayDyn<'py, f32>,
        zero_mean: bool,
    ) -> PyResult<Py<PyAny>> {
        let shape = f.shape().to_vec();
        let data = f.as_slice()?.to_vec();
        let (grid, nt, was_2d) =
            detach_pyresult(py, || self.inverse_laplacian_vec(&data, &shape, zero_mean))?;
        if was_2d {
            vec_to_py(py, &[self.plan.nlat, self.plan.nlon], grid)
        } else {
            vec_to_py(py, &[self.plan.nlat, self.plan.nlon, nt], grid)
        }
    }

    pub fn wind_to_vrtdiv<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let ushape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let (vrt_grid, div_grid, nt, was_2d) = detach_pyresult(py, || {
            let (vrt, div, nt, was_2d) =
                self.vector_to_spec_vec(&udata, &ushape, &vdata, &vshape, None)?;
            let nmdim = self.plan.nlat * (self.plan.nlat + 1) / 2;
            let spec_shape = if was_2d { vec![nmdim] } else { vec![nmdim, nt] };
            let (vrt_grid, _, _) = self.spec_to_scalar_vec(&vrt, &spec_shape)?;
            let (div_grid, _, _) = self.spec_to_scalar_vec(&div, &spec_shape)?;
            Ok((vrt_grid, div_grid, nt, was_2d))
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((
            vec_to_py(py, &out_shape, vrt_grid)?,
            vec_to_py(py, &out_shape, div_grid)?,
        ))
    }

    pub fn vrtdiv_to_wind<'py>(
        &self,
        py: Python<'py>,
        vort: PyReadonlyArrayDyn<'py, f32>,
        div: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let vort_shape = vort.shape().to_vec();
        let div_shape = div.shape().to_vec();
        let vort_data = vort.as_slice()?.to_vec();
        let div_data = div.as_slice()?.to_vec();
        let (u, v, nt, was_2d) = detach_pyresult(py, || {
            self.vrtdiv_to_wind_grid_vec(&vort_data, &vort_shape, &div_data, &div_shape)
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((vec_to_py(py, &out_shape, u)?, vec_to_py(py, &out_shape, v)?))
    }

    pub fn reconstruct_vector_vts<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let ushape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let (out_u, out_v, nt, was_2d) = detach_pyresult(py, || {
            self.vector_reconstruct_vts_vec(&udata, &ushape, &vdata, &vshape)
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((
            vec_to_py(py, &out_shape, out_u)?,
            vec_to_py(py, &out_shape, out_v)?,
        ))
    }

    pub fn streamfunction_to_wind<'py>(
        &self,
        py: Python<'py>,
        psi: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let shape = psi.shape().to_vec();
        let data = psi.as_slice()?.to_vec();
        let (u, v, nt, was_2d) =
            detach_pyresult(py, || self.streamfunction_to_wind_vec(&data, &shape))?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((vec_to_py(py, &out_shape, u)?, vec_to_py(py, &out_shape, v)?))
    }

    pub fn velocity_potential_to_wind<'py>(
        &self,
        py: Python<'py>,
        chi: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let shape = chi.shape().to_vec();
        let data = chi.as_slice()?.to_vec();
        let (u, v, nt, was_2d) =
            detach_pyresult(py, || self.velocity_potential_to_wind_vec(&data, &shape))?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((vec_to_py(py, &out_shape, u)?, vec_to_py(py, &out_shape, v)?))
    }

    pub fn streamfunction_velocity_potential<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let ushape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let (sf, vp, nt, was_2d) = detach_pyresult(py, || {
            self.streamfunction_velocity_potential_vec(&udata, &ushape, &vdata, &vshape)
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((
            vec_to_py(py, &out_shape, sf)?,
            vec_to_py(py, &out_shape, vp)?,
        ))
    }

    pub fn helmholtz_decompose<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let ushape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let (uchi, vchi, upsi, vpsi, nt, was_2d) = detach_pyresult(py, || {
            let (sf, vp, nt, was_2d) =
                self.streamfunction_velocity_potential_vec(&udata, &ushape, &vdata, &vshape)?;
            let shape = if was_2d {
                vec![self.plan.nlat, self.plan.nlon]
            } else {
                vec![self.plan.nlat, self.plan.nlon, nt]
            };
            let zeros = vec![0.0_f32; sf.len()];
            let (upsi, vpsi, _, _) =
                self.streamfunction_velocity_potential_to_wind_vec(&sf, &shape, &zeros, &shape)?;
            let (uchi, vchi, _, _) =
                self.streamfunction_velocity_potential_to_wind_vec(&zeros, &shape, &vp, &shape)?;
            Ok((uchi, vchi, upsi, vpsi, nt, was_2d))
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((
            vec_to_py(py, &out_shape, uchi)?,
            vec_to_py(py, &out_shape, vchi)?,
            vec_to_py(py, &out_shape, upsi)?,
            vec_to_py(py, &out_shape, vpsi)?,
        ))
    }

    #[pyo3(signature = (xlmbda=0.0_f32))]
    pub fn helmsph<'py>(&self, py: Python<'py>, xlmbda: f32) -> PyResult<(Py<PyAny>, f32, f32)> {
        let (grid, pertrb, errm, ierr) =
            detach_pyresult(py, || helmsph_impl(self.plan.nlat, self.plan.nlon, xlmbda))?;
        check_ierror("helmsph", ierr)?;
        Ok((
            vec_to_py(py, &[self.plan.nlat, self.plan.nlon], grid)?,
            pertrb,
            errm,
        ))
    }

    #[pyo3(signature = (omega=7.292115e-5_f32))]
    pub fn coriolis<'py>(&self, py: Python<'py>, omega: f32) -> PyResult<Py<PyAny>> {
        let out = py.detach(|| self.coriolis_vec(1, omega));
        vec_to_py(py, &[self.plan.nlat, self.plan.nlon], out)
    }

    #[pyo3(signature = (u, v, omega=7.292115e-5_f32))]
    pub fn absolute_vorticity<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
        omega: f32,
    ) -> PyResult<Py<PyAny>> {
        let shape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let (_, nt, was_2d) =
            validate_vector_shapes(&shape, &vshape, self.plan.nlat, self.plan.nlon)?;
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let f = detach_pyresult(py, || {
            let (vrt, _, _, _) = self.wind_to_vrtdiv_vec(&udata, &shape, &vdata, &vshape)?;
            let mut f = self.coriolis_vec(nt, omega);
            f.par_iter_mut().zip(vrt.par_iter()).for_each(|(dst, z)| {
                *dst += *z;
            });
            Ok(f)
        })?;
        let _ = omega;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        vec_to_py(py, &out_shape, f)
    }

    pub fn kinetic_energy<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Py<PyAny>> {
        let shape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let (_, nt, was_2d) =
            validate_vector_shapes(&shape, &vshape, self.plan.nlat, self.plan.nlon)?;
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let out = py.detach(|| kinetic_energy_vec(&udata, &vdata));
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        vec_to_py(py, &out_shape, out)
    }

    pub fn k_cross<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let shape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let (_, nt, was_2d) =
            validate_vector_shapes(&shape, &vshape, self.plan.nlat, self.plan.nlon)?;
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let (ku, kv) = py.detach(|| {
            let ku = vdata.par_iter().map(|x| -*x).collect::<Vec<_>>();
            (ku, udata)
        });
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((
            vec_to_py(py, &out_shape, ku)?,
            vec_to_py(py, &out_shape, kv)?,
        ))
    }

    pub fn advect_scalar<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
        q: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Py<PyAny>> {
        let ushape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let qshape = q.shape().to_vec();
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let qdata = q.as_slice()?.to_vec();
        let (out, nt, was_2d) = detach_pyresult(py, || {
            self.advect_scalar_vec(&udata, &ushape, &vdata, &vshape, &qdata, &qshape)
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        vec_to_py(py, &out_shape, out)
    }

    /// Return the zonal contribution to scalar advection: u * dq/dx.
    ///
    /// Input arrays must be rank-2 `(nlat, nlon)` or rank-3 `(nlat, nlon, nt)`.
    /// This avoids materializing `gradient(q)` and multiplying in Python/xarray.
    pub fn advect_scalar_x<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        q: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Py<PyAny>> {
        let ushape = u.shape().to_vec();
        let qshape = q.shape().to_vec();
        let udata = u.as_slice()?.to_vec();
        let qdata = q.as_slice()?.to_vec();
        let (out, nt, was_2d) = detach_pyresult(py, || {
            self.advect_scalar_x_vec(&udata, &ushape, &qdata, &qshape)
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        vec_to_py(py, &out_shape, out)
    }

    /// Return the meridional contribution to scalar advection: v * dq/dy.
    pub fn advect_scalar_y<'py>(
        &self,
        py: Python<'py>,
        v: PyReadonlyArrayDyn<'py, f32>,
        q: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Py<PyAny>> {
        let vshape = v.shape().to_vec();
        let qshape = q.shape().to_vec();
        let vdata = v.as_slice()?.to_vec();
        let qdata = q.as_slice()?.to_vec();
        let (out, nt, was_2d) = detach_pyresult(py, || {
            self.advect_scalar_y_vec(&vdata, &vshape, &qdata, &qshape)
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        vec_to_py(py, &out_shape, out)
    }

    /// Return `(u*dq/dx, v*dq/dy)` in one Rust call.
    pub fn advect_scalar_components<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
        q: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let ushape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let qshape = q.shape().to_vec();
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let qdata = q.as_slice()?.to_vec();
        let (out_x, out_y, nt, was_2d) = detach_pyresult(py, || {
            self.advect_scalar_components_vec(&udata, &ushape, &vdata, &vshape, &qdata, &qshape)
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((
            vec_to_py(py, &out_shape, out_x)?,
            vec_to_py(py, &out_shape, out_y)?,
        ))
    }

    pub fn flux_div_scalar<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
        q: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Py<PyAny>> {
        let ushape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let qshape = q.shape().to_vec();
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let qdata = q.as_slice()?.to_vec();
        let (out, nt, was_2d) = detach_pyresult(py, || {
            self.flux_div_scalar_vec(&udata, &ushape, &vdata, &vshape, &qdata, &qshape)
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        vec_to_py(py, &out_shape, out)
    }

    pub fn split_advect_scalar<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
        q: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Py<PyAny>> {
        let ushape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let qshape = q.shape().to_vec();
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let qdata = q.as_slice()?.to_vec();
        let (out, nt, was_2d) = detach_pyresult(py, || {
            self.split_advect_scalar_vec(&udata, &ushape, &vdata, &vshape, &qdata, &qshape)
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        vec_to_py(py, &out_shape, out)
    }

    #[pyo3(signature = (f, kind="exponential", strength=16.0_f32, order=8))]
    pub fn spectral_filter<'py>(
        &self,
        py: Python<'py>,
        f: PyReadonlyArrayDyn<'py, f32>,
        kind: &str,
        strength: f32,
        order: usize,
    ) -> PyResult<Py<PyAny>> {
        let shape = f.shape().to_vec();
        let (_, nt, was_2d) = validate_scalar_shape(&shape, self.plan.nlat, self.plan.nlon)?;
        let nmdim = self.plan.nlat * (self.plan.nlat + 1) / 2;
        let spec_shape = if was_2d { vec![nmdim] } else { vec![nmdim, nt] };
        let data = f.as_slice()?.to_vec();
        let kind = kind.to_string();
        let grid = detach_pyresult(py, || {
            let (spec, _, _) = self.scalar_to_spec_vec(&data, &shape, None)?;
            let filtered = filter_spec(&spec, nmdim, nt, &kind, strength, order)?;
            let (grid, _, _) = self.spec_to_scalar_vec(&filtered, &spec_shape)?;
            Ok(grid)
        })?;
        if was_2d {
            vec_to_py(py, &[self.plan.nlat, self.plan.nlon], grid)
        } else {
            vec_to_py(py, &[self.plan.nlat, self.plan.nlon, nt], grid)
        }
    }

    #[pyo3(signature = (f, order=4, tau=None, nu=None))]
    pub fn hyperdiffusion<'py>(
        &self,
        py: Python<'py>,
        f: PyReadonlyArrayDyn<'py, f32>,
        order: usize,
        tau: Option<f32>,
        nu: Option<f32>,
    ) -> PyResult<Py<PyAny>> {
        let shape = f.shape().to_vec();
        let data = f.as_slice()?.to_vec();
        let (out, nt, was_2d) = detach_pyresult(py, || {
            self.hyperdiffusion_vec(&data, &shape, order, tau, nu)
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        vec_to_py(py, &out_shape, out)
    }

    #[pyo3(signature = (u, v, order=4, tau=None, nu=None))]
    pub fn vector_hyperdiffusion<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
        order: usize,
        tau: Option<f32>,
        nu: Option<f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let ushape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let (out_u, out_v, nt, was_2d) = detach_pyresult(py, || {
            self.vector_hyperdiffusion_vec(&udata, &ushape, &vdata, &vshape, order, tau, nu)
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((
            vec_to_py(py, &out_shape, out_u)?,
            vec_to_py(py, &out_shape, out_v)?,
        ))
    }

    pub fn vector_laplacian<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let ushape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let (out_u, out_v, nt, was_2d) = detach_pyresult(py, || {
            self.vector_laplacian_vec(&udata, &ushape, &vdata, &vshape)
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((
            vec_to_py(py, &out_shape, out_u)?,
            vec_to_py(py, &out_shape, out_v)?,
        ))
    }

    pub fn inverse_vector_laplacian<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let ushape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let (out_u, out_v, nt, was_2d) = detach_pyresult(py, || {
            self.inverse_vector_laplacian_vec(&udata, &ushape, &vdata, &vshape)
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((
            vec_to_py(py, &out_shape, out_u)?,
            vec_to_py(py, &out_shape, out_v)?,
        ))
    }

    pub fn project_nondivergent<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let ushape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let (out_u, out_v, nt, was_2d) = detach_pyresult(py, || {
            let (vrt, _div, nt, was_2d) =
                self.vector_to_spec_vec(&udata, &ushape, &vdata, &vshape, None)?;
            let zeros = vec![Complex32::new(0.0, 0.0); vrt.len()];
            let nmdim = self.plan.nlat * (self.plan.nlat + 1) / 2;
            let spec_shape = if was_2d { vec![nmdim] } else { vec![nmdim, nt] };
            let (out_u, out_v, _, _) =
                self.spec_to_vector_vec(&vrt, &spec_shape, &zeros, &spec_shape)?;
            Ok((out_u, out_v, nt, was_2d))
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((
            vec_to_py(py, &out_shape, out_u)?,
            vec_to_py(py, &out_shape, out_v)?,
        ))
    }

    pub fn project_irrotational<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let ushape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let (out_u, out_v, nt, was_2d) = detach_pyresult(py, || {
            let (_vrt, div, nt, was_2d) =
                self.vector_to_spec_vec(&udata, &ushape, &vdata, &vshape, None)?;
            let zeros = vec![Complex32::new(0.0, 0.0); div.len()];
            let nmdim = self.plan.nlat * (self.plan.nlat + 1) / 2;
            let spec_shape = if was_2d { vec![nmdim] } else { vec![nmdim, nt] };
            let (out_u, out_v, _, _) =
                self.spec_to_vector_vec(&zeros, &spec_shape, &div, &spec_shape)?;
            Ok((out_u, out_v, nt, was_2d))
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((
            vec_to_py(py, &out_shape, out_u)?,
            vec_to_py(py, &out_shape, out_v)?,
        ))
    }

    pub fn rotated_grad<'py>(
        &self,
        py: Python<'py>,
        f: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let shape = f.shape().to_vec();
        let (_, nt, was_2d) = validate_scalar_shape(&shape, self.plan.nlat, self.plan.nlon)?;
        let data = f.as_slice()?.to_vec();
        let (out_u, out_v) = detach_pyresult(py, || {
            let (gx, gy) = self.gradient_grid_vec(&data, &shape)?;
            let out_u = gy.par_iter().map(|x| -*x).collect::<Vec<_>>();
            Ok((out_u, gx))
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((
            vec_to_py(py, &out_shape, out_u)?,
            vec_to_py(py, &out_shape, out_v)?,
        ))
    }

    #[pyo3(signature = (u, v, absolute=true, omega=7.292115e-5_f32))]
    pub fn vorticity_flux<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
        absolute: bool,
        omega: f32,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let ushape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let (out_u, out_v, nt, was_2d) = detach_pyresult(py, || {
            self.vorticity_flux_vec(&udata, &ushape, &vdata, &vshape, absolute, omega)
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((
            vec_to_py(py, &out_shape, out_u)?,
            vec_to_py(py, &out_shape, out_v)?,
        ))
    }

    pub fn kinetic_energy_grad<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let shape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let (_, nt, was_2d) =
            validate_vector_shapes(&shape, &vshape, self.plan.nlat, self.plan.nlon)?;
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let (gx, gy) = detach_pyresult(py, || {
            let ke = kinetic_energy_vec(&udata, &vdata);
            self.gradient_grid_vec(&ke, &shape)
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((
            vec_to_py(py, &out_shape, gx)?,
            vec_to_py(py, &out_shape, gy)?,
        ))
    }

    #[pyo3(signature = (u, v, omega=7.292115e-5_f32))]
    pub fn momentum_vector_invariant<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
        omega: f32,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let shape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let (_, nt, was_2d) =
            validate_vector_shapes(&shape, &vshape, self.plan.nlat, self.plan.nlon)?;
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let (out_u, out_v) = detach_pyresult(py, || {
            let (vfu, vfv, _, _) =
                self.vorticity_flux_vec(&udata, &shape, &vdata, &vshape, true, omega)?;
            let ke = kinetic_energy_vec(&udata, &vdata);
            let (gku, gkv) = self.gradient_grid_vec(&ke, &shape)?;
            let out_u = vfu
                .par_iter()
                .zip(gku.par_iter())
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>();
            let out_v = vfv
                .par_iter()
                .zip(gkv.par_iter())
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>();
            Ok((out_u, out_v))
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((
            vec_to_py(py, &out_shape, out_u)?,
            vec_to_py(py, &out_shape, out_v)?,
        ))
    }

    pub fn advect_vector<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'py, f32>,
        v: PyReadonlyArrayDyn<'py, f32>,
        a: PyReadonlyArrayDyn<'py, f32>,
        b: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let ushape = u.shape().to_vec();
        let vshape = v.shape().to_vec();
        let ashape = a.shape().to_vec();
        let bshape = b.shape().to_vec();
        let udata = u.as_slice()?.to_vec();
        let vdata = v.as_slice()?.to_vec();
        let adata = a.as_slice()?.to_vec();
        let bdata = b.as_slice()?.to_vec();
        let (out_a, out_b, nt, was_2d) = detach_pyresult(py, || {
            self.advect_vector_vec(
                &udata, &ushape, &vdata, &vshape, &adata, &ashape, &bdata, &bshape,
            )
        })?;
        let out_shape = if was_2d {
            vec![self.plan.nlat, self.plan.nlon]
        } else {
            vec![self.plan.nlat, self.plan.nlon, nt]
        };
        Ok((
            vec_to_py(py, &out_shape, out_a)?,
            vec_to_py(py, &out_shape, out_b)?,
        ))
    }
}

impl SphereOps {
    pub(crate) fn scalar_to_spec_vec(
        &self,
        data: &[f32],
        shape: &[usize],
        truncation: Option<usize>,
    ) -> PyResult<(Vec<Complex32>, usize, bool)> {
        let (_, nt, was_2d) = validate_scalar_shape(shape, self.plan.nlat, self.plan.nlon)?;
        let lwork = (nt + 1) * self.plan.nlat * self.plan.nlon;
        let (a, b, ierr) = match (self.plan.grid_type, self.plan.legfunc) {
            (GridType::Regular, LegFunc::Stored) => shaes_impl_parallel(
                data,
                self.plan.nlat,
                self.plan.nlon,
                nt,
                &self.plan.scalar_analysis_work,
                lwork,
            )?,
            (GridType::Regular, LegFunc::Computed) => shaec_impl_parallel(
                data,
                self.plan.nlat,
                self.plan.nlon,
                nt,
                &self.plan.scalar_analysis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Stored) => shags_impl(
                data,
                self.plan.nlat,
                self.plan.nlon,
                nt,
                &self.plan.scalar_analysis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Computed) => shagc_impl_parallel(
                data,
                self.plan.nlat,
                self.plan.nlon,
                nt,
                &self.plan.scalar_analysis_work,
                lwork,
            )?,
        };
        check_ierror("scalar_to_spec", ierr)?;
        let ntrunc = truncation.unwrap_or(self.plan.nlat - 1);
        if ntrunc >= self.plan.nlat {
            return Err(PyValueError::new_err("truncation must be less than nlat"));
        }
        Ok((
            twodtooned_impl(&a, &b, self.plan.nlat, ntrunc as i32, nt)?,
            nt,
            was_2d,
        ))
    }

    pub(crate) fn spec_to_scalar_vec(
        &self,
        spec: &[Complex32],
        shape: &[usize],
    ) -> PyResult<(Vec<f32>, usize, bool)> {
        let (nmdim, nt, was_1d) = validate_spec_shape(shape)?;
        let (a, b) = onedtotwod_impl(spec, self.plan.nlat, nmdim, nt)?;
        let lwork = (nt + 1) * self.plan.nlat * self.plan.nlon;
        let (grid, ierr) = match (self.plan.grid_type, self.plan.legfunc) {
            (GridType::Regular, LegFunc::Stored) => shses_impl_parallel(
                &a,
                &b,
                self.plan.nlat,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Regular, LegFunc::Computed) => shsec_impl_parallel(
                &a,
                &b,
                self.plan.nlat,
                nt,
                0,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Stored) => shsgs_impl(
                &a,
                &b,
                self.plan.nlat,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Computed) => shsgc_impl_parallel(
                &a,
                &b,
                self.plan.nlat,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
        };
        check_ierror("spec_to_scalar", ierr)?;
        Ok((grid, nt, was_1d))
    }

    pub(crate) fn laplacian_vec(
        &self,
        data: &[f32],
        shape: &[usize],
    ) -> PyResult<(Vec<f32>, usize, bool)> {
        let (_, nt, was_2d) = validate_scalar_shape(shape, self.plan.nlat, self.plan.nlon)?;
        let (spec, _, _) = self.scalar_to_spec_vec(data, shape, None)?;
        let nmdim = self.plan.nlat * (self.plan.nlat + 1) / 2;
        let (a, b) = onedtotwod_impl(&spec, self.plan.nlat, nmdim, nt)?;
        let lwork = gradient_lwork(self.plan.nlat, self.plan.nlon, nt);
        let (mut grid, ierr) = match (self.plan.grid_type, self.plan.legfunc) {
            (GridType::Regular, LegFunc::Stored) => slapes_impl(
                self.plan.nlon,
                &a,
                &b,
                self.plan.nlat,
                0,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Regular, LegFunc::Computed) => slapec_impl(
                self.plan.nlon,
                &a,
                &b,
                self.plan.nlat,
                0,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Stored) => slapgs_impl(
                self.plan.nlon,
                &a,
                &b,
                self.plan.nlat,
                0,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Computed) => slapgc_impl(
                self.plan.nlon,
                &a,
                &b,
                self.plan.nlat,
                0,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
        };
        check_ierror("laplacian", ierr)?;
        let scale = 1.0_f32 / (self.plan.radius * self.plan.radius);
        grid.par_iter_mut().for_each(|value| *value *= scale);
        Ok((grid, nt, was_2d))
    }

    pub(crate) fn inverse_laplacian_vec(
        &self,
        data: &[f32],
        shape: &[usize],
        _zero_mean: bool,
    ) -> PyResult<(Vec<f32>, usize, bool)> {
        let (_, nt, was_2d) = validate_scalar_shape(shape, self.plan.nlat, self.plan.nlon)?;
        let (spec, _, _) = self.scalar_to_spec_vec(data, shape, None)?;
        let nmdim = self.plan.nlat * (self.plan.nlat + 1) / 2;
        let (a, b) = onedtotwod_impl(&spec, self.plan.nlat, nmdim, nt)?;
        let xlmbda = vec![0.0_f32; nt];
        let lwork = gradient_lwork(self.plan.nlat, self.plan.nlon, nt);
        let (mut grid, _pertrb, ierr) = match (self.plan.grid_type, self.plan.legfunc) {
            (GridType::Regular, LegFunc::Stored) => islapes_impl(
                self.plan.nlon,
                &xlmbda,
                &a,
                &b,
                self.plan.nlat,
                0,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Regular, LegFunc::Computed) => islapec_impl(
                self.plan.nlon,
                &xlmbda,
                &a,
                &b,
                self.plan.nlat,
                0,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Stored) => islapgs_impl(
                self.plan.nlon,
                &xlmbda,
                &a,
                &b,
                self.plan.nlat,
                0,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Computed) => islapgc_impl(
                self.plan.nlon,
                &xlmbda,
                &a,
                &b,
                self.plan.nlat,
                0,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
        };
        check_ierror("inverse_laplacian", ierr)?;
        let scale = self.plan.radius * self.plan.radius;
        grid.par_iter_mut().for_each(|value| *value *= scale);
        Ok((grid, nt, was_2d))
    }

    pub(crate) fn vector_to_spec_vec(
        &self,
        u: &[f32],
        ushape: &[usize],
        v: &[f32],
        vshape: &[usize],
        truncation: Option<usize>,
    ) -> PyResult<(Vec<Complex32>, Vec<Complex32>, usize, bool)> {
        self.vector_to_spec_vec_backend(u, ushape, v, vshape, truncation, TransformBackend::Rayon)
    }

    pub(crate) fn vector_to_spec_vec_backend(
        &self,
        u: &[f32],
        ushape: &[usize],
        v: &[f32],
        vshape: &[usize],
        truncation: Option<usize>,
        backend: TransformBackend,
    ) -> PyResult<(Vec<Complex32>, Vec<Complex32>, usize, bool)> {
        let (_, nt, was_2d) =
            validate_vector_shapes(ushape, vshape, self.plan.nlat, self.plan.nlon)?;
        let v_math = v.par_iter().map(|x| -*x).collect::<Vec<_>>();
        let w_math = u.to_vec();
        let lwork = gradient_lwork(self.plan.nlat, self.plan.nlon, nt);
        let (br, bi, cr, ci, ierr) = match (self.plan.grid_type, self.plan.legfunc) {
            (GridType::Regular, LegFunc::Stored) => vhaes_impl(
                &v_math,
                &w_math,
                self.plan.nlat,
                self.plan.nlon,
                nt,
                0,
                &self.plan.vector_analysis_work,
                lwork,
            )?,
            (GridType::Regular, LegFunc::Computed) => vhaec_impl_parallel(
                &v_math,
                &w_math,
                self.plan.nlat,
                self.plan.nlon,
                nt,
                0,
                &self.plan.vector_analysis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Stored) => {
                if backend == TransformBackend::LatPar {
                    vhags_impl_latpar(
                        &v_math,
                        &w_math,
                        self.plan.nlat,
                        self.plan.nlon,
                        nt,
                        0,
                        &self.plan.vector_analysis_work,
                        lwork,
                    )?
                } else {
                    vhags_impl_parallel(
                        &v_math,
                        &w_math,
                        self.plan.nlat,
                        self.plan.nlon,
                        nt,
                        0,
                        &self.plan.vector_analysis_work,
                        lwork,
                    )?
                }
            }
            (GridType::Gaussian, LegFunc::Computed) => {
                if backend == TransformBackend::LatPar {
                    vhagc_impl_latpar(
                        &v_math,
                        &w_math,
                        self.plan.nlat,
                        self.plan.nlon,
                        nt,
                        0,
                        &self.plan.vector_analysis_work,
                        lwork,
                    )?
                } else {
                    vhagc_impl(
                        &v_math,
                        &w_math,
                        self.plan.nlat,
                        self.plan.nlon,
                        nt,
                        0,
                        &self.plan.vector_analysis_work,
                        lwork,
                    )?
                }
            }
        };
        check_ierror("vector_to_spec", ierr)?;
        let ntrunc = truncation.unwrap_or(self.plan.nlat - 1);
        if ntrunc >= self.plan.nlat {
            return Err(PyValueError::new_err("truncation must be less than nlat"));
        }
        let (vrt, div) = twodtooned_vrtdiv_impl(
            &br,
            &bi,
            &cr,
            &ci,
            self.plan.nlat,
            ntrunc as i32,
            nt,
            self.plan.radius,
        )?;
        Ok((vrt, div, nt, was_2d))
    }

    pub(crate) fn spec_to_vector_vec(
        &self,
        vort_spec: &[Complex32],
        vort_shape: &[usize],
        div_spec: &[Complex32],
        div_shape: &[usize],
    ) -> PyResult<(Vec<f32>, Vec<f32>, usize, bool)> {
        if vort_shape != div_shape {
            return Err(PyValueError::new_err(
                "vort_spec and div_spec must have the same shape",
            ));
        }
        let (nmdim, nt, was_1d) = validate_spec_shape(vort_shape)?;
        let (av, bv) = onedtotwod_impl(vort_spec, self.plan.nlat, nmdim, nt)?;
        let (ad, bd) = onedtotwod_impl(div_spec, self.plan.nlat, nmdim, nt)?;
        let ad = ad
            .into_iter()
            .map(|value| value * self.plan.radius)
            .collect::<Vec<_>>();
        let bd = bd
            .into_iter()
            .map(|value| value * self.plan.radius)
            .collect::<Vec<_>>();
        let av = av
            .into_iter()
            .map(|value| value * self.plan.radius)
            .collect::<Vec<_>>();
        let bv = bv
            .into_iter()
            .map(|value| value * self.plan.radius)
            .collect::<Vec<_>>();
        let lwork = vector_reconstruction_lwork(self.plan.nlat, self.plan.nlon, nt);
        let (v_math, w_math, ierr) = match (self.plan.grid_type, self.plan.legfunc) {
            (GridType::Regular, LegFunc::Stored) => {
                let (v, w, _pertbd, _pertbv, ierr) = idvtes_impl(
                    self.plan.nlon,
                    &ad,
                    &bd,
                    &av,
                    &bv,
                    self.plan.nlat,
                    nt,
                    0,
                    &self.plan.vector_synthesis_work,
                    lwork,
                )?;
                (v, w, ierr)
            }
            (GridType::Regular, LegFunc::Computed) => {
                let (v, w, _pertbd, _pertbv, ierr) = idvtec_impl(
                    self.plan.nlon,
                    &ad,
                    &bd,
                    &av,
                    &bv,
                    self.plan.nlat,
                    nt,
                    0,
                    &self.plan.vector_synthesis_work,
                    lwork,
                )?;
                (v, w, ierr)
            }
            (GridType::Gaussian, LegFunc::Stored) => {
                let (v, w, _pertbd, _pertbv, ierr) = idvtgs_impl(
                    &ad,
                    &bd,
                    &av,
                    &bv,
                    self.plan.nlat,
                    nt,
                    0,
                    &self.plan.vector_synthesis_work,
                    lwork,
                )?;
                (v, w, ierr)
            }
            (GridType::Gaussian, LegFunc::Computed) => {
                let (v, w, _pertbd, _pertbv, ierr) = idvtgc_impl(
                    &ad,
                    &bd,
                    &av,
                    &bv,
                    self.plan.nlat,
                    nt,
                    0,
                    &self.plan.vector_synthesis_work,
                    lwork,
                )?;
                (v, w, ierr)
            }
        };
        check_ierror("spec_to_vector", ierr)?;
        let u = w_math;
        let v = v_math.into_par_iter().map(|x| -x).collect::<Vec<_>>();
        Ok((u, v, nt, was_1d))
    }

    pub(crate) fn vector_reconstruct_vts_vec(
        &self,
        u: &[f32],
        ushape: &[usize],
        v: &[f32],
        vshape: &[usize],
    ) -> PyResult<(Vec<f32>, Vec<f32>, usize, bool)> {
        let (br, bi, cr, ci, nt, was_2d) = self.vector_analysis_coeffs(u, ushape, v, vshape)?;
        let lwork = vector_reconstruction_lwork(self.plan.nlat, self.plan.nlon, nt);
        let (v_math, w_math, _idv, _nlon, ierr) = match (self.plan.grid_type, self.plan.legfunc) {
            (GridType::Regular, LegFunc::Stored) | (GridType::Gaussian, LegFunc::Stored) => {
                vtsgs_impl(
                    &br,
                    &bi,
                    &cr,
                    &ci,
                    self.plan.nlat,
                    nt,
                    0,
                    &self.plan.vector_vts_work,
                    lwork,
                )?
            }
            (GridType::Regular, LegFunc::Computed) | (GridType::Gaussian, LegFunc::Computed) => {
                vtsgc_impl(
                    &br,
                    &bi,
                    &cr,
                    &ci,
                    self.plan.nlat,
                    nt,
                    0,
                    &self.plan.vector_vts_work,
                    lwork,
                )?
            }
        };
        check_ierror("reconstruct_vector_vts", ierr)?;
        let u = w_math;
        let v = v_math.into_par_iter().map(|x| -x).collect::<Vec<_>>();
        Ok((u, v, nt, was_2d))
    }

    fn vector_analysis_coeffs(
        &self,
        u: &[f32],
        ushape: &[usize],
        v: &[f32],
        vshape: &[usize],
    ) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, usize, bool)> {
        let (_, nt, was_2d) =
            validate_vector_shapes(ushape, vshape, self.plan.nlat, self.plan.nlon)?;
        let v_math = v.par_iter().map(|x| -*x).collect::<Vec<_>>();
        let w_math = u.to_vec();
        let lwork = gradient_lwork(self.plan.nlat, self.plan.nlon, nt);
        let (br, bi, cr, ci, ierr) = match (self.plan.grid_type, self.plan.legfunc) {
            (GridType::Regular, LegFunc::Stored) => vhaes_impl(
                &v_math,
                &w_math,
                self.plan.nlat,
                self.plan.nlon,
                nt,
                0,
                &self.plan.vector_analysis_work,
                lwork,
            )?,
            (GridType::Regular, LegFunc::Computed) => vhaec_impl_parallel(
                &v_math,
                &w_math,
                self.plan.nlat,
                self.plan.nlon,
                nt,
                0,
                &self.plan.vector_analysis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Stored) => vhags_impl_parallel(
                &v_math,
                &w_math,
                self.plan.nlat,
                self.plan.nlon,
                nt,
                0,
                &self.plan.vector_analysis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Computed) => vhagc_impl(
                &v_math,
                &w_math,
                self.plan.nlat,
                self.plan.nlon,
                nt,
                0,
                &self.plan.vector_analysis_work,
                lwork,
            )?,
        };
        check_ierror("vector_analysis", ierr)?;
        Ok((br, bi, cr, ci, nt, was_2d))
    }

    pub(crate) fn divergence_vec(
        &self,
        u: &[f32],
        ushape: &[usize],
        v: &[f32],
        vshape: &[usize],
    ) -> PyResult<(Vec<f32>, usize, bool)> {
        let (br, bi, _cr, _ci, nt, was_2d) = self.vector_analysis_coeffs(u, ushape, v, vshape)?;
        let lwork = gradient_lwork(self.plan.nlat, self.plan.nlon, nt);
        let (mut div, ierr) = match (self.plan.grid_type, self.plan.legfunc) {
            (GridType::Regular, LegFunc::Stored) => dives_impl(
                self.plan.nlon,
                &br,
                &bi,
                self.plan.nlat,
                0,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Regular, LegFunc::Computed) => divec_impl(
                self.plan.nlon,
                &br,
                &bi,
                self.plan.nlat,
                0,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Stored) => divgs_impl(
                self.plan.nlon,
                &br,
                &bi,
                self.plan.nlat,
                0,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Computed) => divgc_impl(
                self.plan.nlon,
                &br,
                &bi,
                self.plan.nlat,
                0,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
        };
        check_ierror("divergence", ierr)?;
        let scale = 1.0_f32 / self.plan.radius;
        for value in &mut div {
            *value *= scale;
        }
        Ok((div, nt, was_2d))
    }

    pub(crate) fn inverse_divergence_vec(
        &self,
        div: &[f32],
        shape: &[usize],
    ) -> PyResult<(Vec<f32>, Vec<f32>, usize, bool)> {
        let (_, nt, was_2d) = validate_scalar_shape(shape, self.plan.nlat, self.plan.nlon)?;
        let (spec, _, _) = self.scalar_to_spec_vec(div, shape, None)?;
        let nmdim = self.plan.nlat * (self.plan.nlat + 1) / 2;
        let (a, b) = onedtotwod_impl(&spec, self.plan.nlat, nmdim, nt)?;
        let lwork = gradient_lwork(self.plan.nlat, self.plan.nlon, nt);
        let (v_math, w_math, _pertrb, ierr) = match (self.plan.grid_type, self.plan.legfunc) {
            (GridType::Regular, LegFunc::Stored) => idives_impl(
                &a,
                &b,
                self.plan.nlat,
                nt,
                0,
                &self.plan.vector_synthesis_work,
                lwork,
            )?,
            (GridType::Regular, LegFunc::Computed) => idivec_impl(
                self.plan.nlon,
                &a,
                &b,
                self.plan.nlat,
                nt,
                0,
                &self.plan.vector_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Stored) => {
                let (v, w, pertrb, _idvw, _nlon, ierr) = idivgs_impl(
                    &a,
                    &b,
                    self.plan.nlat,
                    nt,
                    0,
                    &self.plan.vector_synthesis_work,
                    lwork,
                )?;
                (v, w, pertrb, ierr)
            }
            (GridType::Gaussian, LegFunc::Computed) => idivgc_impl(
                &a,
                &b,
                self.plan.nlat,
                nt,
                0,
                &self.plan.vector_synthesis_work,
                lwork,
            )?,
        };
        check_ierror("inverse_divergence", ierr)?;
        let scale = self.plan.radius;
        let u = w_math.into_iter().map(|x| x * scale).collect::<Vec<_>>();
        let v = v_math
            .into_par_iter()
            .map(|x| -x * scale)
            .collect::<Vec<_>>();
        Ok((u, v, nt, was_2d))
    }

    pub(crate) fn vorticity_vec(
        &self,
        u: &[f32],
        ushape: &[usize],
        v: &[f32],
        vshape: &[usize],
    ) -> PyResult<(Vec<f32>, usize, bool)> {
        let (_br, _bi, cr, ci, nt, was_2d) = self.vector_analysis_coeffs(u, ushape, v, vshape)?;
        let lwork = gradient_lwork(self.plan.nlat, self.plan.nlon, nt);
        let (mut vort, ierr) = match (self.plan.grid_type, self.plan.legfunc) {
            (GridType::Regular, LegFunc::Stored) => vrtes_impl(
                self.plan.nlon,
                &cr,
                &ci,
                self.plan.nlat,
                0,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Regular, LegFunc::Computed) => vrtec_impl(
                &cr,
                &ci,
                self.plan.nlat,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Stored) => vrtgs_impl(
                self.plan.nlon,
                &cr,
                &ci,
                self.plan.nlat,
                0,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Computed) => vrtgc_impl(
                &cr,
                &ci,
                self.plan.nlat,
                0,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
        };
        check_ierror("vorticity", ierr)?;
        let scale = 1.0_f32 / self.plan.radius;
        for value in &mut vort {
            *value *= scale;
        }
        Ok((vort, nt, was_2d))
    }

    pub(crate) fn inverse_vorticity_vec(
        &self,
        vort: &[f32],
        shape: &[usize],
    ) -> PyResult<(Vec<f32>, Vec<f32>, usize, bool)> {
        let (_, nt, was_2d) = validate_scalar_shape(shape, self.plan.nlat, self.plan.nlon)?;
        let (spec, _, _) = self.scalar_to_spec_vec(vort, shape, None)?;
        let nmdim = self.plan.nlat * (self.plan.nlat + 1) / 2;
        let (a, b) = onedtotwod_impl(&spec, self.plan.nlat, nmdim, nt)?;
        let lwork = gradient_lwork(self.plan.nlat, self.plan.nlon, nt);
        let (v_math, w_math, _pertrb, ierr) = match (self.plan.grid_type, self.plan.legfunc) {
            (GridType::Regular, LegFunc::Stored) => ivrtes_impl(
                &a,
                &b,
                self.plan.nlat,
                nt,
                0,
                &self.plan.vector_synthesis_work,
                lwork,
            )?,
            (GridType::Regular, LegFunc::Computed) => ivrtec_impl(
                self.plan.nlon,
                &a,
                &b,
                self.plan.nlat,
                nt,
                0,
                &self.plan.vector_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Stored) => {
                let (v, w, pertrb, _idvw, _nlon, ierr) = ivrtgs_impl(
                    &a,
                    &b,
                    self.plan.nlat,
                    nt,
                    0,
                    &self.plan.vector_synthesis_work,
                    lwork,
                )?;
                (v, w, pertrb, ierr)
            }
            (GridType::Gaussian, LegFunc::Computed) => ivrtgc_impl(
                &a,
                &b,
                self.plan.nlat,
                nt,
                0,
                &self.plan.vector_synthesis_work,
                lwork,
            )?,
        };
        check_ierror("inverse_vorticity", ierr)?;
        let scale = self.plan.radius;
        let u = w_math.into_iter().map(|x| x * scale).collect::<Vec<_>>();
        let v = v_math
            .into_par_iter()
            .map(|x| -x * scale)
            .collect::<Vec<_>>();
        Ok((u, v, nt, was_2d))
    }

    pub(crate) fn wind_to_vrtdiv_vec(
        &self,
        u: &[f32],
        ushape: &[usize],
        v: &[f32],
        vshape: &[usize],
    ) -> PyResult<(Vec<f32>, Vec<f32>, usize, bool)> {
        let (vrt, div, nt, was_2d) = self.vector_to_spec_vec(u, ushape, v, vshape, None)?;
        let nmdim = self.plan.nlat * (self.plan.nlat + 1) / 2;
        let spec_shape = if was_2d { vec![nmdim] } else { vec![nmdim, nt] };
        let (vrt_grid, _, _) = self.spec_to_scalar_vec(&vrt, &spec_shape)?;
        let (div_grid, _, _) = self.spec_to_scalar_vec(&div, &spec_shape)?;
        Ok((vrt_grid, div_grid, nt, was_2d))
    }

    pub(crate) fn vrtdiv_to_wind_grid_vec(
        &self,
        vort: &[f32],
        vort_shape: &[usize],
        div: &[f32],
        div_shape: &[usize],
    ) -> PyResult<(Vec<f32>, Vec<f32>, usize, bool)> {
        let (_, nt, was_2d) =
            validate_vector_shapes(vort_shape, div_shape, self.plan.nlat, self.plan.nlon)?;
        let (vort_spec, _, _) = self.scalar_to_spec_vec(vort, vort_shape, None)?;
        let (div_spec, _, _) = self.scalar_to_spec_vec(div, div_shape, None)?;
        let nmdim = self.plan.nlat * (self.plan.nlat + 1) / 2;
        let spec_shape = if was_2d { vec![nmdim] } else { vec![nmdim, nt] };
        let (u, v, _, _) =
            self.spec_to_vector_vec(&vort_spec, &spec_shape, &div_spec, &spec_shape)?;
        Ok((u, v, nt, was_2d))
    }

    pub(crate) fn streamfunction_to_wind_vec(
        &self,
        psi: &[f32],
        shape: &[usize],
    ) -> PyResult<(Vec<f32>, Vec<f32>, usize, bool)> {
        let (_, _, _) = validate_scalar_shape(shape, self.plan.nlat, self.plan.nlon)?;
        let zeros = vec![0.0_f32; psi.len()];
        self.streamfunction_velocity_potential_to_wind_vec(psi, shape, &zeros, shape)
    }

    pub(crate) fn velocity_potential_to_wind_vec(
        &self,
        chi: &[f32],
        shape: &[usize],
    ) -> PyResult<(Vec<f32>, Vec<f32>, usize, bool)> {
        let (_, _, _) = validate_scalar_shape(shape, self.plan.nlat, self.plan.nlon)?;
        let zeros = vec![0.0_f32; chi.len()];
        self.streamfunction_velocity_potential_to_wind_vec(&zeros, shape, chi, shape)
    }

    pub(crate) fn streamfunction_velocity_potential_vec(
        &self,
        u: &[f32],
        ushape: &[usize],
        v: &[f32],
        vshape: &[usize],
    ) -> PyResult<(Vec<f32>, Vec<f32>, usize, bool)> {
        let (br, bi, cr, ci, nt, was_2d) = self.vector_analysis_coeffs(u, ushape, v, vshape)?;
        let lwork = sfvp_lwork(self.plan.nlat, self.plan.nlon, nt);
        let (mut sf, mut vp, ierr) = match (self.plan.grid_type, self.plan.legfunc) {
            (GridType::Regular, LegFunc::Stored) => sfvpes_impl(
                self.plan.nlon,
                &br,
                &bi,
                &cr,
                &ci,
                self.plan.nlat,
                nt,
                0,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Regular, LegFunc::Computed) => sfvpec_impl(
                self.plan.nlon,
                &br,
                &bi,
                &cr,
                &ci,
                self.plan.nlat,
                nt,
                0,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Stored) => sfvpgs_impl(
                self.plan.nlon,
                &br,
                &bi,
                &cr,
                &ci,
                self.plan.nlat,
                nt,
                0,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Computed) => sfvpgc_impl(
                self.plan.nlon,
                &br,
                &bi,
                &cr,
                &ci,
                self.plan.nlat,
                nt,
                0,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
        };
        check_ierror("streamfunction_velocity_potential", ierr)?;
        for value in &mut sf {
            *value *= self.plan.radius;
        }
        for value in &mut vp {
            *value *= self.plan.radius;
        }
        Ok((sf, vp, nt, was_2d))
    }

    pub(crate) fn streamfunction_velocity_potential_to_wind_vec(
        &self,
        sf: &[f32],
        sf_shape: &[usize],
        vp: &[f32],
        vp_shape: &[usize],
    ) -> PyResult<(Vec<f32>, Vec<f32>, usize, bool)> {
        let (_, nt, was_2d) =
            validate_vector_shapes(sf_shape, vp_shape, self.plan.nlat, self.plan.nlon)?;
        let (sf_spec, _, _) = self.scalar_to_spec_vec(sf, sf_shape, None)?;
        let (vp_spec, _, _) = self.scalar_to_spec_vec(vp, vp_shape, None)?;
        let nmdim = self.plan.nlat * (self.plan.nlat + 1) / 2;
        let (mut as_, mut bs) = onedtotwod_impl(&sf_spec, self.plan.nlat, nmdim, nt)?;
        let (mut av, mut bv) = onedtotwod_impl(&vp_spec, self.plan.nlat, nmdim, nt)?;
        let inv_radius = 1.0_f32 / self.plan.radius;
        for value in &mut as_ {
            *value *= inv_radius;
        }
        for value in &mut bs {
            *value *= inv_radius;
        }
        for value in &mut av {
            *value *= inv_radius;
        }
        for value in &mut bv {
            *value *= inv_radius;
        }
        let lwork = isfvp_lwork(self.plan.nlat, self.plan.nlon, nt);
        let (v_math, w_math, ierr) = match (self.plan.grid_type, self.plan.legfunc) {
            (GridType::Regular, LegFunc::Stored) => isfvpes_impl(
                self.plan.nlon,
                &as_,
                &bs,
                &av,
                &bv,
                self.plan.nlat,
                nt,
                0,
                &self.plan.vector_synthesis_work,
                lwork,
            )?,
            (GridType::Regular, LegFunc::Computed) => isfvpec_impl(
                self.plan.nlon,
                &as_,
                &bs,
                &av,
                &bv,
                self.plan.nlat,
                nt,
                0,
                &self.plan.vector_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Stored) => isfvpgs_impl(
                self.plan.nlon,
                &as_,
                &bs,
                &av,
                &bv,
                self.plan.nlat,
                nt,
                0,
                &self.plan.vector_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Computed) => isfvpgc_impl(
                self.plan.nlon,
                &as_,
                &bs,
                &av,
                &bv,
                self.plan.nlat,
                nt,
                0,
                &self.plan.vector_synthesis_work,
                lwork,
            )?,
        };
        check_ierror("streamfunction_velocity_potential_to_wind", ierr)?;
        let u = w_math;
        let v = v_math
            .into_par_iter()
            .map(|value| -value)
            .collect::<Vec<_>>();
        Ok((u, v, nt, was_2d))
    }

    pub(crate) fn gradient_from_spec_vec(
        &self,
        spec: &[Complex32],
        shape: &[usize],
    ) -> PyResult<(Vec<f32>, Vec<f32>)> {
        let (nmdim, nt, _) = validate_spec_shape(shape)?;
        let (a, b) = onedtotwod_impl(spec, self.plan.nlat, nmdim, nt)?;
        let lwork = gradient_lwork(self.plan.nlat, self.plan.nlon, nt);
        let (v_math, w_math, ierr) = match (self.plan.grid_type, self.plan.legfunc) {
            (GridType::Regular, LegFunc::Stored) => {
                let (v, w, ierr) = grades_impl(
                    &a,
                    &b,
                    self.plan.nlat,
                    nt,
                    0,
                    &self.plan.vector_synthesis_work,
                    lwork,
                )?;
                (v, w, ierr)
            }
            (GridType::Regular, LegFunc::Computed) => {
                let (v, w, ierr) = gradec_impl(
                    &a,
                    &b,
                    self.plan.nlat,
                    self.plan.nlon,
                    nt,
                    0,
                    &self.plan.vector_synthesis_work,
                    lwork,
                )?;
                (v, w, ierr)
            }
            (GridType::Gaussian, LegFunc::Stored) => {
                let (v, w, _, _, ierr) = gradgs_impl(
                    &a,
                    &b,
                    self.plan.nlat,
                    nt,
                    0,
                    &self.plan.vector_synthesis_work,
                    lwork,
                )?;
                (v, w, ierr)
            }
            (GridType::Gaussian, LegFunc::Computed) => {
                let (v, w, _, _, ierr) = gradgc_impl(
                    &a,
                    &b,
                    self.plan.nlat,
                    nt,
                    0,
                    &self.plan.vector_synthesis_work,
                    lwork,
                )?;
                (v, w, ierr)
            }
        };
        check_ierror("gradient", ierr)?;
        let scale = 1.0_f32 / self.plan.radius;
        let u = w_math.into_iter().map(|x| x * scale).collect::<Vec<_>>();
        let v = v_math
            .into_par_iter()
            .map(|x| -x * scale)
            .collect::<Vec<_>>();
        Ok((u, v))
    }

    pub(crate) fn inverse_gradient_vec(
        &self,
        u: &[f32],
        ushape: &[usize],
        v: &[f32],
        vshape: &[usize],
    ) -> PyResult<(Vec<f32>, usize, bool)> {
        let (_, nt, was_2d) =
            validate_vector_shapes(ushape, vshape, self.plan.nlat, self.plan.nlon)?;
        let radius = self.plan.radius;
        let v_math = v.par_iter().map(|x| -*x * radius).collect::<Vec<_>>();
        let w_math = u.par_iter().map(|x| *x * radius).collect::<Vec<_>>();
        let lwork = gradient_lwork(self.plan.nlat, self.plan.nlon, nt);
        let (br, bi, _cr, _ci, ierr) = match (self.plan.grid_type, self.plan.legfunc) {
            (GridType::Regular, LegFunc::Stored) => vhaes_impl(
                &v_math,
                &w_math,
                self.plan.nlat,
                self.plan.nlon,
                nt,
                0,
                &self.plan.vector_analysis_work,
                lwork,
            )?,
            (GridType::Regular, LegFunc::Computed) => vhaec_impl_parallel(
                &v_math,
                &w_math,
                self.plan.nlat,
                self.plan.nlon,
                nt,
                0,
                &self.plan.vector_analysis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Stored) => vhags_impl_parallel(
                &v_math,
                &w_math,
                self.plan.nlat,
                self.plan.nlon,
                nt,
                0,
                &self.plan.vector_analysis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Computed) => vhagc_impl(
                &v_math,
                &w_math,
                self.plan.nlat,
                self.plan.nlon,
                nt,
                0,
                &self.plan.vector_analysis_work,
                lwork,
            )?,
        };
        check_ierror("inverse_gradient vector_analysis", ierr)?;
        let (sf, ierr) = match (self.plan.grid_type, self.plan.legfunc) {
            (GridType::Regular, LegFunc::Stored) => igrades_impl(
                &br,
                &bi,
                self.plan.nlat,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Regular, LegFunc::Computed) => igradec_impl(
                &br,
                &bi,
                self.plan.nlat,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Stored) => igradgs_impl(
                &br,
                &bi,
                self.plan.nlat,
                nt,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Computed) => igradgc_impl(
                &br,
                &bi,
                self.plan.nlat,
                nt,
                0,
                &self.plan.scalar_synthesis_work,
                lwork,
            )?,
        };
        check_ierror("inverse_gradient", ierr)?;
        Ok((sf, nt, was_2d))
    }

    pub(crate) fn advect_scalar_vec(
        &self,
        u: &[f32],
        ushape: &[usize],
        v: &[f32],
        vshape: &[usize],
        q: &[f32],
        qshape: &[usize],
    ) -> PyResult<(Vec<f32>, usize, bool)> {
        let (_, nt, was_2d) =
            validate_vector_shapes(ushape, vshape, self.plan.nlat, self.plan.nlon)?;
        let (_, qnt, qwas_2d) = validate_scalar_shape(qshape, self.plan.nlat, self.plan.nlon)?;
        if nt != qnt || was_2d != qwas_2d {
            return Err(PyValueError::new_err(
                "u/v/q must have matching ranks and nt",
            ));
        }
        let (dqdx, dqdy) = self.gradient_grid_vec(q, qshape)?;
        let out = u
            .par_iter()
            .zip(v.par_iter())
            .zip(dqdx.par_iter().zip(dqdy.par_iter()))
            .map(|((u, v), (gx, gy))| u * gx + v * gy)
            .collect::<Vec<_>>();
        Ok((out, nt, was_2d))
    }

    pub(crate) fn advect_scalar_x_vec(
        &self,
        u: &[f32],
        ushape: &[usize],
        q: &[f32],
        qshape: &[usize],
    ) -> PyResult<(Vec<f32>, usize, bool)> {
        let (_, nt, was_2d) = validate_scalar_shape(ushape, self.plan.nlat, self.plan.nlon)?;
        let (_, qnt, qwas_2d) = validate_scalar_shape(qshape, self.plan.nlat, self.plan.nlon)?;
        if nt != qnt || was_2d != qwas_2d {
            return Err(PyValueError::new_err(
                "u and q must have matching ranks and nt",
            ));
        }
        let (dqdx, _) = self.gradient_grid_vec(q, qshape)?;
        let out = u
            .par_iter()
            .zip(dqdx.par_iter())
            .map(|(u, gx)| u * gx)
            .collect::<Vec<_>>();
        Ok((out, nt, was_2d))
    }

    pub(crate) fn advect_scalar_y_vec(
        &self,
        v: &[f32],
        vshape: &[usize],
        q: &[f32],
        qshape: &[usize],
    ) -> PyResult<(Vec<f32>, usize, bool)> {
        let (_, nt, was_2d) = validate_scalar_shape(vshape, self.plan.nlat, self.plan.nlon)?;
        let (_, qnt, qwas_2d) = validate_scalar_shape(qshape, self.plan.nlat, self.plan.nlon)?;
        if nt != qnt || was_2d != qwas_2d {
            return Err(PyValueError::new_err(
                "v and q must have matching ranks and nt",
            ));
        }
        let (_, dqdy) = self.gradient_grid_vec(q, qshape)?;
        let out = v
            .par_iter()
            .zip(dqdy.par_iter())
            .map(|(v, gy)| v * gy)
            .collect::<Vec<_>>();
        Ok((out, nt, was_2d))
    }

    pub(crate) fn advect_scalar_components_vec(
        &self,
        u: &[f32],
        ushape: &[usize],
        v: &[f32],
        vshape: &[usize],
        q: &[f32],
        qshape: &[usize],
    ) -> PyResult<(Vec<f32>, Vec<f32>, usize, bool)> {
        let (_, nt, was_2d) =
            validate_vector_shapes(ushape, vshape, self.plan.nlat, self.plan.nlon)?;
        let (_, qnt, qwas_2d) = validate_scalar_shape(qshape, self.plan.nlat, self.plan.nlon)?;
        if nt != qnt || was_2d != qwas_2d {
            return Err(PyValueError::new_err(
                "u/v/q must have matching ranks and nt",
            ));
        }
        let (dqdx, dqdy) = self.gradient_grid_vec(q, qshape)?;
        let out_x = u
            .par_iter()
            .zip(dqdx.par_iter())
            .map(|(u, gx)| u * gx)
            .collect::<Vec<_>>();
        let out_y = v
            .par_iter()
            .zip(dqdy.par_iter())
            .map(|(v, gy)| v * gy)
            .collect::<Vec<_>>();
        Ok((out_x, out_y, nt, was_2d))
    }

    pub(crate) fn flux_div_scalar_vec(
        &self,
        u: &[f32],
        ushape: &[usize],
        v: &[f32],
        vshape: &[usize],
        q: &[f32],
        qshape: &[usize],
    ) -> PyResult<(Vec<f32>, usize, bool)> {
        let (_, nt, was_2d) =
            validate_vector_shapes(ushape, vshape, self.plan.nlat, self.plan.nlon)?;
        let (_, qnt, qwas_2d) = validate_scalar_shape(qshape, self.plan.nlat, self.plan.nlon)?;
        if nt != qnt || was_2d != qwas_2d {
            return Err(PyValueError::new_err(
                "u/v/q must have matching ranks and nt",
            ));
        }
        let qu = u
            .par_iter()
            .zip(q.par_iter())
            .map(|(u, q)| u * q)
            .collect::<Vec<_>>();
        let qv = v
            .par_iter()
            .zip(q.par_iter())
            .map(|(v, q)| v * q)
            .collect::<Vec<_>>();
        let (vrt, div, _, _) = self.vector_to_spec_vec(&qu, ushape, &qv, vshape, None)?;
        let _ = vrt;
        let nmdim = self.plan.nlat * (self.plan.nlat + 1) / 2;
        let spec_shape = if was_2d { vec![nmdim] } else { vec![nmdim, nt] };
        let (grid, _, _) = self.spec_to_scalar_vec(&div, &spec_shape)?;
        Ok((grid, nt, was_2d))
    }

    pub(crate) fn split_advect_scalar_vec(
        &self,
        u: &[f32],
        ushape: &[usize],
        v: &[f32],
        vshape: &[usize],
        q: &[f32],
        qshape: &[usize],
    ) -> PyResult<(Vec<f32>, usize, bool)> {
        let (adv, nt, was_2d) = self.advect_scalar_vec(u, ushape, v, vshape, q, qshape)?;
        let (flux, _, _) = self.flux_div_scalar_vec(u, ushape, v, vshape, q, qshape)?;
        let (_, div, _, _) = self.wind_to_vrtdiv_vec(u, ushape, v, vshape)?;
        let out = adv
            .par_iter()
            .zip(flux.par_iter())
            .zip(q.par_iter().zip(div.par_iter()))
            .map(|((a, f), (q, d))| 0.5_f32 * (a + f + q * d))
            .collect::<Vec<_>>();
        Ok((out, nt, was_2d))
    }

    pub(crate) fn gradient_grid_vec(
        &self,
        q: &[f32],
        qshape: &[usize],
    ) -> PyResult<(Vec<f32>, Vec<f32>)> {
        let (_, nt, _) = validate_scalar_shape(qshape, self.plan.nlat, self.plan.nlon)?;
        let (spec, _, _) = self.scalar_to_spec_vec(q, qshape, None)?;
        let nmdim = self.plan.nlat * (self.plan.nlat + 1) / 2;
        let spec_shape = if qshape.len() == 2 {
            vec![nmdim]
        } else {
            vec![nmdim, nt]
        };
        self.gradient_from_spec_vec(&spec, &spec_shape)
    }

    pub(crate) fn coriolis_vec(&self, nt: usize, omega: f32) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.plan.nlat * self.plan.nlon * nt];
        out.par_chunks_mut(self.plan.nlon * nt)
            .enumerate()
            .for_each(|(i, row)| {
                let value = 2.0_f32 * omega * self.plan.lat[i].to_radians().sin();
                row.fill(value);
            });
        out
    }

    pub(crate) fn hyperdiffusion_vec(
        &self,
        f: &[f32],
        shape: &[usize],
        order: usize,
        tau: Option<f32>,
        nu: Option<f32>,
    ) -> PyResult<(Vec<f32>, usize, bool)> {
        let (_, nt, was_2d) = validate_scalar_shape(shape, self.plan.nlat, self.plan.nlon)?;
        let (spec, _, _) = self.scalar_to_spec_vec(f, shape, None)?;
        let nmdim = self.plan.nlat * (self.plan.nlat + 1) / 2;
        let diff_spec = hyperdiffuse_spec(
            &spec,
            nmdim,
            nt,
            self.plan.nlat - 1,
            self.plan.radius,
            order,
            tau,
            nu,
        )?;
        let spec_shape = if was_2d { vec![nmdim] } else { vec![nmdim, nt] };
        let (grid, _, _) = self.spec_to_scalar_vec(&diff_spec, &spec_shape)?;
        Ok((grid, nt, was_2d))
    }

    pub(crate) fn vector_laplacian_vec(
        &self,
        u: &[f32],
        ushape: &[usize],
        v: &[f32],
        vshape: &[usize],
    ) -> PyResult<(Vec<f32>, Vec<f32>, usize, bool)> {
        let (br, bi, cr, ci, nt, was_2d) = self.vector_analysis_coeffs(u, ushape, v, vshape)?;
        let lwork = vector_laplacian_lwork(self.plan.nlat, self.plan.nlon, nt);
        let (v_math, mut out_u, ierr) = match (self.plan.grid_type, self.plan.legfunc) {
            (GridType::Regular, LegFunc::Stored) => vlapes_impl(
                self.plan.nlon,
                &br,
                &bi,
                &cr,
                &ci,
                self.plan.nlat,
                nt,
                0,
                &self.plan.vector_synthesis_work,
                lwork,
            )?,
            (GridType::Regular, LegFunc::Computed) => vlapec_impl(
                self.plan.nlon,
                &br,
                &bi,
                &cr,
                &ci,
                self.plan.nlat,
                nt,
                0,
                &self.plan.vector_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Stored) => vlapgs_impl(
                self.plan.nlon,
                &br,
                &bi,
                &cr,
                &ci,
                self.plan.nlat,
                nt,
                0,
                &self.plan.vector_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Computed) => vlapgc_impl(
                self.plan.nlon,
                &br,
                &bi,
                &cr,
                &ci,
                self.plan.nlat,
                nt,
                0,
                &self.plan.vector_synthesis_work,
                lwork,
            )?,
        };
        check_ierror("vector_laplacian", ierr)?;
        let scale = 1.0_f32 / (self.plan.radius * self.plan.radius);
        out_u.par_iter_mut().for_each(|value| *value *= scale);
        let out_v = v_math
            .into_par_iter()
            .map(|value| -value * scale)
            .collect::<Vec<_>>();
        Ok((out_u, out_v, nt, was_2d))
    }

    pub(crate) fn inverse_vector_laplacian_vec(
        &self,
        u: &[f32],
        ushape: &[usize],
        v: &[f32],
        vshape: &[usize],
    ) -> PyResult<(Vec<f32>, Vec<f32>, usize, bool)> {
        let (br, bi, cr, ci, nt, was_2d) = self.vector_analysis_coeffs(u, ushape, v, vshape)?;
        let lwork = vector_laplacian_lwork(self.plan.nlat, self.plan.nlon, nt);
        let (v_math, mut out_u, ierr) = match (self.plan.grid_type, self.plan.legfunc) {
            (GridType::Regular, LegFunc::Stored) => ivlapes_impl(
                self.plan.nlon,
                &br,
                &bi,
                &cr,
                &ci,
                self.plan.nlat,
                nt,
                0,
                &self.plan.vector_synthesis_work,
                lwork,
            )?,
            (GridType::Regular, LegFunc::Computed) => ivlapec_impl(
                self.plan.nlon,
                &br,
                &bi,
                &cr,
                &ci,
                self.plan.nlat,
                nt,
                0,
                &self.plan.vector_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Stored) => ivlapgs_impl(
                self.plan.nlon,
                &br,
                &bi,
                &cr,
                &ci,
                self.plan.nlat,
                nt,
                0,
                &self.plan.vector_synthesis_work,
                lwork,
            )?,
            (GridType::Gaussian, LegFunc::Computed) => ivlapgc_impl(
                self.plan.nlon,
                &br,
                &bi,
                &cr,
                &ci,
                self.plan.nlat,
                nt,
                0,
                &self.plan.vector_synthesis_work,
                lwork,
            )?,
        };
        check_ierror("inverse_vector_laplacian", ierr)?;
        let scale = self.plan.radius * self.plan.radius;
        out_u.par_iter_mut().for_each(|value| *value *= scale);
        let out_v = v_math
            .into_par_iter()
            .map(|value| -value * scale)
            .collect::<Vec<_>>();
        Ok((out_u, out_v, nt, was_2d))
    }

    pub(crate) fn vector_hyperdiffusion_vec(
        &self,
        u: &[f32],
        ushape: &[usize],
        v: &[f32],
        vshape: &[usize],
        order: usize,
        tau: Option<f32>,
        nu: Option<f32>,
    ) -> PyResult<(Vec<f32>, Vec<f32>, usize, bool)> {
        let (vrt, div, nt, was_2d) = self.vector_to_spec_vec(u, ushape, v, vshape, None)?;
        let nmdim = self.plan.nlat * (self.plan.nlat + 1) / 2;
        let dvrt = hyperdiffuse_spec(
            &vrt,
            nmdim,
            nt,
            self.plan.nlat - 1,
            self.plan.radius,
            order,
            tau,
            nu,
        )?;
        let ddiv = hyperdiffuse_spec(
            &div,
            nmdim,
            nt,
            self.plan.nlat - 1,
            self.plan.radius,
            order,
            tau,
            nu,
        )?;
        let spec_shape = if was_2d { vec![nmdim] } else { vec![nmdim, nt] };
        let (out_u, out_v, _, _) =
            self.spec_to_vector_vec(&dvrt, &spec_shape, &ddiv, &spec_shape)?;
        Ok((out_u, out_v, nt, was_2d))
    }

    pub(crate) fn vorticity_flux_vec(
        &self,
        u: &[f32],
        ushape: &[usize],
        v: &[f32],
        vshape: &[usize],
        absolute: bool,
        omega: f32,
    ) -> PyResult<(Vec<f32>, Vec<f32>, usize, bool)> {
        let (_, nt, was_2d) =
            validate_vector_shapes(ushape, vshape, self.plan.nlat, self.plan.nlon)?;
        let (mut zeta, _, _, _) = self.wind_to_vrtdiv_vec(u, ushape, v, vshape)?;
        if absolute {
            let f = self.coriolis_vec(nt, omega);
            zeta.par_iter_mut().zip(f.par_iter()).for_each(|(z, f)| {
                *z += *f;
            });
        }
        let out_u = zeta
            .par_iter()
            .zip(v.par_iter())
            .map(|(eta, v)| -eta * v)
            .collect::<Vec<_>>();
        let out_v = zeta
            .par_iter()
            .zip(u.par_iter())
            .map(|(eta, u)| eta * u)
            .collect::<Vec<_>>();
        Ok((out_u, out_v, nt, was_2d))
    }

    pub(crate) fn advect_vector_vec(
        &self,
        u: &[f32],
        ushape: &[usize],
        v: &[f32],
        vshape: &[usize],
        a: &[f32],
        ashape: &[usize],
        b: &[f32],
        bshape: &[usize],
    ) -> PyResult<(Vec<f32>, Vec<f32>, usize, bool)> {
        let (_, nt, was_2d) =
            validate_vector_shapes(ushape, vshape, self.plan.nlat, self.plan.nlon)?;
        let (_, ant, awas_2d) =
            validate_vector_shapes(ashape, bshape, self.plan.nlat, self.plan.nlon)?;
        if nt != ant || was_2d != awas_2d {
            return Err(PyValueError::new_err(
                "advect_vector inputs must have matching ranks and nt",
            ));
        }
        let (agx, agy) = self.gradient_grid_vec(a, ashape)?;
        let (bgx, bgy) = self.gradient_grid_vec(b, bshape)?;
        let out_a = u
            .par_iter()
            .zip(v.par_iter())
            .zip(agx.par_iter().zip(agy.par_iter()))
            .map(|((u, v), (gx, gy))| u * gx + v * gy)
            .collect::<Vec<_>>();
        let out_b = u
            .par_iter()
            .zip(v.par_iter())
            .zip(bgx.par_iter().zip(bgy.par_iter()))
            .map(|((u, v), (gx, gy))| u * gx + v * gy)
            .collect::<Vec<_>>();
        Ok((out_a, out_b, nt, was_2d))
    }
}

fn validate_scalar_shape(
    shape: &[usize],
    nlat: usize,
    nlon: usize,
) -> PyResult<(usize, usize, bool)> {
    if shape.len() != 2 && shape.len() != 3 {
        return Err(PyValueError::new_err("scalar field must be rank 2 or 3"));
    }
    if shape[0] != nlat || shape[1] != nlon {
        return Err(PyValueError::new_err(format!(
            "expected scalar shape ({nlat}, {nlon}) or ({nlat}, {nlon}, nt)"
        )));
    }
    let nt = if shape.len() == 2 { 1 } else { shape[2] };
    Ok((nlat * nlon * nt, nt, shape.len() == 2))
}

fn validate_vector_shapes(
    ushape: &[usize],
    vshape: &[usize],
    nlat: usize,
    nlon: usize,
) -> PyResult<(usize, usize, bool)> {
    if ushape != vshape {
        return Err(PyValueError::new_err(
            "vector components must have the same shape",
        ));
    }
    validate_scalar_shape(ushape, nlat, nlon)
}

fn validate_spec_shape(shape: &[usize]) -> PyResult<(usize, usize, bool)> {
    if shape.len() != 1 && shape.len() != 2 {
        return Err(PyValueError::new_err(
            "spectral coefficients must be rank 1 or 2",
        ));
    }
    let nmdim = shape[0];
    let nt = if shape.len() == 1 { 1 } else { shape[1] };
    Ok((nmdim, nt, shape.len() == 1))
}

fn validate_rank2_shape(shape: &[usize], dim0: usize, dim1: usize, context: &str) -> PyResult<()> {
    if shape != [dim0, dim1] {
        return Err(PyValueError::new_err(format!(
            "{context} must have shape ({dim0}, {dim1})"
        )));
    }
    Ok(())
}

fn check_ierror(context: &str, ierror: i32) -> PyResult<()> {
    if ierror == 0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{context} failed with ierror={ierror}"
        )))
    }
}

fn detach_pyresult<T, F>(py: Python<'_>, f: F) -> PyResult<T>
where
    T: Send,
    F: FnOnce() -> PyResult<T> + Send,
{
    py.detach(|| f().map_err(|err| err.to_string()))
        .map_err(PyValueError::new_err)
}

fn shift_lsav(nlon: usize, nlat: usize) -> usize {
    2 * (2 * nlat + nlon + 16)
}

fn shift_lwork(nlon: usize, nlat: usize) -> usize {
    if nlon.is_multiple_of(2) {
        2 * nlon * (nlat + 1)
    } else {
        nlon * (5 * nlat + 1)
    }
}

fn gradient_lwork(nlat: usize, nlon: usize, nt: usize) -> usize {
    let l1 = if nlon % 2 == 0 {
        nlat.min(nlon / 2)
    } else {
        nlat.min((nlon + 1) / 2)
    };
    let l2 = nlat.div_ceil(2);
    let stored = nlat * ((2 * nt + 1) * nlon + 2 * l1 * nt + 1);
    let computed_synthesis = nlat * (2 * nt * nlon + (6 * l2).max(nlon)) + nlat * (2 * l1 * nt + 1);
    let computed_analysis = nlat * (4 * nlon * nt + 6 * l2);
    stored.max(computed_synthesis).max(computed_analysis)
}

fn vector_laplacian_lwork(nlat: usize, nlon: usize, nt: usize) -> usize {
    let imid = nlat.div_ceil(2);
    let mmax = nlat.min((nlon + 1) / 2);
    let mn = mmax * nlat * nt;
    let stored_full = (2 * nt + 1) * nlat * nlon + nlat * (4 * nt * mmax + 1);
    let stored_half = (2 * nt + 1) * imid * nlon + nlat * (4 * nt * mmax + 1);
    let computed_full = nlat * (2 * nt * nlon + (6 * imid).max(nlon) + 1) + 4 * mn;
    let computed_half = imid * (2 * nt * nlon + (6 * nlat).max(nlon)) + 4 * mn + nlat;
    stored_full
        .max(stored_half)
        .max(computed_full)
        .max(computed_half)
}

fn vector_reconstruction_lwork(nlat: usize, nlon: usize, nt: usize) -> usize {
    let imid = nlat.div_ceil(2);
    let mmax = nlat.min((nlon + 1) / 2);
    let mn = mmax * nlat * nt;
    let full = nlat * (2 * nt * nlon + (6 * imid).max(nlon)) + 4 * mn + nlat;
    let half = imid * (2 * nt * nlon + (6 * nlat).max(nlon)) + 4 * mn + nlat;
    let stored_vts = (2 * nt + 1) * nlat * nlon;
    full.max(half).max(stored_vts)
}

fn sfvp_lwork(nlat: usize, nlon: usize, nt: usize) -> usize {
    let imid = nlat.div_ceil(2);
    let ls_full = nlat;
    let ls_half = imid;
    let mab = nlat.min(nlon / 2 + 1);
    let mn = mab * nlat * nt;
    let stored_full = ls_full * (nt + 1) * nlon + 2 * mab * nlat * nt + nlat;
    let stored_half = ls_half * (nt + 1) * nlon + 2 * mab * nlat * nt + nlat;
    let computed_full = ls_full * nt * nlon + (ls_full * nlon).max(3 * nlat * imid) + 2 * mn + nlat;
    let computed_half = ls_half * nt * nlon + (ls_half * nlon).max(3 * nlat * imid) + 2 * mn + nlat;
    stored_full
        .max(stored_half)
        .max(computed_full)
        .max(computed_half)
}

fn isfvp_lwork(nlat: usize, nlon: usize, nt: usize) -> usize {
    let imid = nlat.div_ceil(2);
    let l1 = nlat.min((nlon + 2) / 2);
    let mn = l1 * nlat * nt;
    let stored_full = nlat * ((2 * nt + 1) * nlon + 4 * l1 * nt + 1);
    let stored_half = (2 * nt + 1) * nlon + nlat * (4 * l1 * nt + 1);
    let computed_full = nlat * (2 * nt * nlon + (6 * imid).max(nlon) + 1) + 4 * mn;
    let computed_half = imid * (2 * nt * nlon + (6 * nlat).max(nlon)) + 4 * mn + nlat;
    stored_full
        .max(stored_half)
        .max(computed_full)
        .max(computed_half)
}

fn filter_spec(
    spec: &[Complex32],
    nmdim: usize,
    nt: usize,
    kind: &str,
    strength: f32,
    order: usize,
) -> PyResult<Vec<Complex32>> {
    if spec.len() != nmdim * nt {
        return Err(PyValueError::new_err("spectral coefficient size mismatch"));
    }
    let ntrunc = infer_ntrunc(nmdim)?;
    let mut out = vec![Complex32::new(0.0, 0.0); spec.len()];
    let denom = ntrunc.max(1) as f32;
    for k in 0..nt {
        let mut nmstrt = 0_usize;
        for m in 1..=ntrunc + 1 {
            for n in m..=ntrunc + 1 {
                let degree = n - 1;
                let sigma = match kind {
                    "none" => 1.0_f32,
                    "exponential" => (-strength * (degree as f32 / denom).powi(order as i32)).exp(),
                    _ => {
                        return Err(PyValueError::new_err(
                            "filter kind must be 'exponential' or 'none'",
                        ));
                    }
                };
                let nm = nmstrt + n - m + 1;
                let idx = (nm - 1) * nt + k;
                out[idx] = spec[idx] * sigma;
            }
            nmstrt += (ntrunc + 2) - m;
        }
    }
    Ok(out)
}

fn hyperdiffuse_spec(
    spec: &[Complex32],
    nmdim: usize,
    nt: usize,
    ntrunc: usize,
    radius: f32,
    order: usize,
    tau: Option<f32>,
    nu: Option<f32>,
) -> PyResult<Vec<Complex32>> {
    if order == 0 || order % 2 != 0 {
        return Err(PyValueError::new_err(
            "hyperdiffusion order must be a positive even integer",
        ));
    }
    if spec.len() != nmdim * nt {
        return Err(PyValueError::new_err("spectral coefficient size mismatch"));
    }
    let p = order / 2;
    let lambda_max = (ntrunc as f32 * (ntrunc as f32 + 1.0_f32)) / (radius * radius);
    let coeff = match (nu, tau) {
        (Some(nu), _) => nu,
        (None, Some(tau)) if tau > 0.0 && lambda_max > 0.0 => {
            1.0_f32 / (tau * lambda_max.powi(p as i32))
        }
        (None, Some(_)) => return Err(PyValueError::new_err("tau must be positive")),
        (None, None) => return Err(PyValueError::new_err("either tau or nu must be provided")),
    };
    let inferred = infer_ntrunc(nmdim)?;
    let mut out = vec![Complex32::new(0.0, 0.0); spec.len()];
    for k in 0..nt {
        let mut nmstrt = 0_usize;
        for m in 1..=inferred + 1 {
            for n in m..=inferred + 1 {
                let degree = n - 1;
                let lambda = (degree as f32 * (degree as f32 + 1.0_f32)) / (radius * radius);
                let factor = -coeff * lambda.powi(p as i32);
                let nm = nmstrt + n - m + 1;
                let idx = (nm - 1) * nt + k;
                out[idx] = spec[idx] * factor;
            }
            nmstrt += (inferred + 2) - m;
        }
    }
    Ok(out)
}

fn infer_ntrunc(nmdim: usize) -> PyResult<usize> {
    let nmdim_f = nmdim as f32;
    let ntrunc = (-1.5_f32 + 0.5_f32 * (9.0_f32 - 8.0_f32 * (1.0_f32 - nmdim_f)).sqrt()) as isize;
    if ntrunc < 0 {
        Err(PyValueError::new_err(
            "invalid spectral coefficient dimension",
        ))
    } else {
        Ok(ntrunc as usize)
    }
}

pub(crate) fn vec_to_py<'py>(
    py: Python<'py>,
    shape: &[usize],
    data: Vec<f32>,
) -> PyResult<Py<PyAny>> {
    let arr = ArrayD::from_shape_vec(IxDyn(shape), data)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(arr.into_pyarray(py).into_any().unbind())
}

fn complex_to_py<'py>(
    py: Python<'py>,
    shape: &[usize],
    data: Vec<Complex32>,
) -> PyResult<Py<PyAny>> {
    let arr = ArrayD::from_shape_vec(IxDyn(shape), data)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(arr.into_pyarray(py).into_any().unbind())
}

pub(crate) fn kinetic_energy_vec(u: &[f32], v: &[f32]) -> Vec<f32> {
    u.par_iter()
        .zip(v.par_iter())
        .map(|(u, v)| 0.5_f32 * (u * u + v * v))
        .collect()
}
