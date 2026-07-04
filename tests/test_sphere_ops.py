import numpy as np

import spectralkernel_rs as sk

def sample_scalar(ops):
    lat = np.deg2rad(ops.lat())
    lon = np.deg2rad(ops.lon())
    return (
        np.cos(lat)[:, None] * np.cos(lon)[None, :]
        + 0.5 * np.sin(lat)[:, None]
        + 0.25 * (np.cos(lat)[:, None] ** 2) * np.sin(2.0 * lon)[None, :]
    )


def remove_mean(field):
    return field - np.mean(field)


def check_grid(grid_factory):
    ops = grid_factory(9, 16, radius=1.0)
    field = sample_scalar(ops)

    grad_u, grad_v = ops.grad(field)
    assert grad_u.shape == field.shape
    assert grad_v.shape == field.shape
    assert grad_u.dtype == np.float32

    lap = ops.laplacian(field)
    inv_lap = ops.inverse_laplacian(lap)
    np.testing.assert_allclose(remove_mean(inv_lap), remove_mean(field), atol=2e-4, rtol=2e-4)

    u_psi, v_psi = ops.streamfunction_to_wind(field)
    div_psi = ops.div(u_psi, v_psi)
    vort_psi = ops.vort(u_psi, v_psi)
    np.testing.assert_allclose(div_psi, 0.0, atol=2e-5)
    np.testing.assert_allclose(vort_psi, lap, atol=2e-4, rtol=2e-4)

    u_chi, v_chi = ops.velocity_potential_to_wind(field)
    vort_chi = ops.vort(u_chi, v_chi)
    div_chi = ops.div(u_chi, v_chi)
    np.testing.assert_allclose(vort_chi, 0.0, atol=2e-5)
    np.testing.assert_allclose(div_chi, lap, atol=2e-4, rtol=2e-4)

    ones = np.ones_like(field)
    flux = ops.flux_div_scalar(u_psi, v_psi, ones)
    np.testing.assert_allclose(flux, div_psi, atol=2e-5)

    dhdt, dudt, dvdt = sk.shallow_water_rhs(ops, ones, np.zeros_like(field), np.zeros_like(field))
    np.testing.assert_allclose(dhdt, 0.0, atol=2e-6)
    np.testing.assert_allclose(dudt, 0.0, atol=2e-6)
    np.testing.assert_allclose(dvdt, 0.0, atol=1e-5)


def test_sphere_ops_regular_identities():
    check_grid(sk.SphereOps.regular)


def test_sphere_ops_gaussian_identities():
    check_grid(sk.SphereOps.gaussian)


def test_sphere_ops_legfunc_modes_gradient_roundtrip():
    for factory in (sk.SphereOps.regular, sk.SphereOps.gaussian):
        for legfunc in ("stored", "computed"):
            ops = factory(9, 16, radius=1.0, legfunc=legfunc)
            field = sample_scalar(ops)
            gx, gy = ops.gradient_grid(field)
            recovered = ops.inverse_gradient(gx, gy)
            np.testing.assert_allclose(
                remove_mean(recovered),
                remove_mean(field),
                atol=3e-4,
                rtol=3e-4,
            )


def test_sphere_ops_legfunc_modes_divergence_inverse():
    for factory in (sk.SphereOps.regular, sk.SphereOps.gaussian):
        for legfunc in ("stored", "computed"):
            ops = factory(9, 16, radius=1.0, legfunc=legfunc)
            field = sample_scalar(ops)
            u, v = ops.velocity_potential_to_wind(field)
            div = ops.divergence(u, v)
            rec_u, rec_v = ops.inverse_divergence(div)
            np.testing.assert_allclose(rec_u, u, atol=3e-4, rtol=3e-4)
            np.testing.assert_allclose(rec_v, v, atol=3e-4, rtol=3e-4)


def test_sphere_ops_legfunc_modes_vorticity_inverse():
    for factory in (sk.SphereOps.regular, sk.SphereOps.gaussian):
        for legfunc in ("stored", "computed"):
            ops = factory(9, 16, radius=1.0, legfunc=legfunc)
            field = sample_scalar(ops)
            u, v = ops.streamfunction_to_wind(field)
            vort = ops.vorticity(u, v)
            rec_u, rec_v = ops.inverse_vorticity(vort)
            np.testing.assert_allclose(rec_u, u, atol=3e-4, rtol=3e-4)
            np.testing.assert_allclose(rec_v, v, atol=3e-4, rtol=3e-4)


def test_sphere_ops_legfunc_modes_scalar_laplacian_inverse():
    for factory in (sk.SphereOps.regular, sk.SphereOps.gaussian):
        for legfunc in ("stored", "computed"):
            ops = factory(9, 16, radius=1.0, legfunc=legfunc)
            field = sample_scalar(ops)
            lap = ops.laplacian(field)
            recovered = ops.inverse_laplacian(lap)
            np.testing.assert_allclose(
                remove_mean(recovered),
                remove_mean(field),
                atol=3e-4,
                rtol=3e-4,
            )


def test_sphere_ops_legfunc_modes_vector_laplacian_inverse():
    for factory in (sk.SphereOps.regular, sk.SphereOps.gaussian):
        for legfunc in ("stored", "computed"):
            ops = factory(9, 16, radius=1.0, legfunc=legfunc)
            psi = sample_scalar(ops)
            chi = np.cos(np.deg2rad(ops.lat()))[:, None] ** 2 * np.cos(2.0 * np.deg2rad(ops.lon()))[None, :]
            upsi, vpsi = ops.streamfunction_to_wind(psi)
            uchi, vchi = ops.velocity_potential_to_wind(chi)
            u = upsi + uchi
            v = vpsi + vchi
            lu, lv = ops.vector_laplacian(u, v)
            rec_u, rec_v = ops.inverse_vector_laplacian(lu, lv)
            np.testing.assert_allclose(rec_u, u, atol=4e-4, rtol=4e-4)
            np.testing.assert_allclose(rec_v, v, atol=4e-4, rtol=4e-4)


def test_sphere_ops_legfunc_modes_sfvp_helmholtz_roundtrip():
    for factory in (sk.SphereOps.regular, sk.SphereOps.gaussian):
        for legfunc in ("stored", "computed"):
            ops = factory(9, 16, radius=1.0, legfunc=legfunc)
            psi = sample_scalar(ops)
            chi = np.cos(np.deg2rad(ops.lat()))[:, None] ** 2 * np.cos(2.0 * np.deg2rad(ops.lon()))[None, :]
            upsi, vpsi = ops.streamfunction_to_wind(psi)
            uchi, vchi = ops.velocity_potential_to_wind(chi)
            u = upsi + uchi
            v = vpsi + vchi
            sf, vp = ops.streamfunction_velocity_potential(u, v)
            np.testing.assert_allclose(remove_mean(sf), remove_mean(psi), atol=4e-4, rtol=4e-4)
            np.testing.assert_allclose(remove_mean(vp), remove_mean(chi), atol=4e-4, rtol=4e-4)
            rec_uchi, rec_vchi, rec_upsi, rec_vpsi = ops.helmholtz_decompose(u, v)
            np.testing.assert_allclose(rec_upsi, upsi, atol=4e-4, rtol=4e-4)
            np.testing.assert_allclose(rec_vpsi, vpsi, atol=4e-4, rtol=4e-4)
            np.testing.assert_allclose(rec_uchi, uchi, atol=4e-4, rtol=4e-4)
            np.testing.assert_allclose(rec_vchi, vchi, atol=4e-4, rtol=4e-4)


def test_sphere_ops_legfunc_modes_vts_reconstruction():
    for factory in (sk.SphereOps.regular, sk.SphereOps.gaussian):
        for legfunc in ("stored", "computed"):
            ops = factory(9, 16, radius=1.0, legfunc=legfunc)
            psi = sample_scalar(ops)
            chi = np.cos(np.deg2rad(ops.lat()))[:, None] ** 2 * np.cos(2.0 * np.deg2rad(ops.lon()))[None, :]
            upsi, vpsi = ops.streamfunction_to_wind(psi)
            uchi, vchi = ops.velocity_potential_to_wind(chi)
            u = upsi + uchi
            v = vpsi + vchi
            vort_spec, div_spec = ops.vector_to_spec(u, v)
            rec_u, rec_v = ops.spec_to_vector(vort_spec, div_spec)
            vts_u, vts_v = ops.reconstruct_vector_vts(u, v)
            np.testing.assert_allclose(rec_u, u, atol=4e-4, rtol=4e-4)
            np.testing.assert_allclose(rec_v, v, atol=4e-4, rtol=4e-4)
            assert vts_u.shape == u.shape
            assert vts_v.shape == v.shape
            assert np.all(np.isfinite(vts_u))
            assert np.all(np.isfinite(vts_v))


def test_sphere_ops_preprocessing_helpers():
    ops = sk.SphereOps.regular(9, 16, radius=1.0)
    sg = np.arange(ops.nlon * ops.nlat, dtype=np.float32).reshape(ops.nlon, ops.nlat)
    sm = ops.geo_to_math_scalar(sg)
    sg_back = ops.math_to_geo_scalar(sm)
    np.testing.assert_allclose(sg_back, sg, atol=1e-6)

    ug = np.sin(np.linspace(0.0, 1.0, ops.nlon * ops.nlat, dtype=np.float32)).reshape(ops.nlon, ops.nlat)
    vg = np.cos(np.linspace(0.0, 1.0, ops.nlon * ops.nlat, dtype=np.float32)).reshape(ops.nlon, ops.nlat)
    vm, wm = ops.geo_to_math_vector(ug, vg)
    ug_back, vg_back = ops.math_to_geo_vector(vm, wm)
    np.testing.assert_allclose(ug_back, ug, atol=1e-6)
    np.testing.assert_allclose(vg_back, vg, atol=1e-6)

    shifted = ops.scalar_shift(sg, ioff=0)
    assert shifted.shape == (ops.nlon, ops.nlat + 1)
    unshifted = ops.scalar_shift(shifted, ioff=1)
    assert unshifted.shape == sg.shape
    assert np.all(np.isfinite(shifted))
    assert np.all(np.isfinite(unshifted))

    su, sv = ops.vector_shift(ug, vg, ioff=0)
    assert su.shape == (ops.nlon, ops.nlat + 1)
    assert sv.shape == (ops.nlon, ops.nlat + 1)
    bu, bv = ops.vector_shift(su, sv, ioff=1)
    assert bu.shape == ug.shape
    assert bv.shape == vg.shape
    assert np.all(np.isfinite(su))
    assert np.all(np.isfinite(sv))
    assert np.all(np.isfinite(bu))
    assert np.all(np.isfinite(bv))


def test_scalar_advection_zero_wind():
    ops = sk.SphereOps.gaussian(9, 16, radius=1.0)
    q = sample_scalar(ops).astype(np.float64)
    zeros = np.zeros_like(q)
    rhs = sk.scalar_advection_rhs(ops, q, zeros, zeros)
    assert rhs.dtype == np.float32
    np.testing.assert_allclose(rhs, 0.0, atol=1e-6)


def test_v2_projection_and_vector_terms():
    ops = sk.SphereOps.gaussian(9, 16, radius=1.0)
    psi = sample_scalar(ops)
    chi = np.cos(np.deg2rad(ops.lat()))[:, None] ** 2 * np.cos(2.0 * np.deg2rad(ops.lon()))[None, :]
    upsi, vpsi = ops.streamfunction_to_wind(psi)
    uchi, vchi = ops.velocity_potential_to_wind(chi)
    u = upsi + uchi
    v = vpsi + vchi

    und, vnd = ops.project_nondivergent(u, v)
    uir, vir = ops.project_irrotational(u, v)
    np.testing.assert_allclose(ops.div(und, vnd), 0.0, atol=3e-5)
    np.testing.assert_allclose(ops.vort(uir, vir), 0.0, atol=3e-5)
    np.testing.assert_allclose(und + uir, u, atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(vnd + vir, v, atol=3e-4, rtol=3e-4)

    ru, rv = ops.rotated_grad(psi)
    np.testing.assert_allclose(ru, -ops.grad(psi)[1], atol=1e-6)
    np.testing.assert_allclose(rv, ops.grad(psi)[0], atol=1e-6)

    vi_u, vi_v = ops.momentum_vector_invariant(u, v, omega=0.0)
    vf_u, vf_v = ops.vorticity_flux(u, v, absolute=False)
    kg_u, kg_v = ops.kinetic_energy_grad(u, v)
    np.testing.assert_allclose(vi_u, vf_u + kg_u, atol=1e-6)
    np.testing.assert_allclose(vi_v, vf_v + kg_v, atol=1e-6)


def test_v2_filter_and_diffusion_smoke():
    ops = sk.SphereOps.gaussian(9, 16, radius=1.0)
    f = sample_scalar(ops)
    filtered = ops.spectral_filter(f, strength=8.0, order=8)
    assert filtered.shape == f.shape
    assert np.linalg.norm(filtered) <= np.linalg.norm(f) * 1.01

    const = np.ones_like(f)
    np.testing.assert_allclose(ops.hyperdiffusion(const, order=4, tau=10.0), 0.0, atol=1e-6)
    zeros = np.zeros_like(f)
    du, dv = ops.vector_hyperdiffusion(zeros, zeros, order=4, tau=10.0)
    np.testing.assert_allclose(du, 0.0, atol=1e-6)
    np.testing.assert_allclose(dv, 0.0, atol=1e-6)

    dhdt, dudt, dvdt = sk.shallow_water_rhs(
        ops,
        const,
        zeros,
        zeros,
        diffusion={"order": 4, "tau": 10.0},
        mass_correction=True,
    )
    np.testing.assert_allclose(dhdt, 0.0, atol=1e-6)
    np.testing.assert_allclose(dudt, 0.0, atol=1e-6)
    np.testing.assert_allclose(dvdt, 0.0, atol=1e-5)
