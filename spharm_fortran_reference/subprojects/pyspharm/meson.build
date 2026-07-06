project('_spherepack', 'c',
  version : '1.0.0',
  meson_version: '>=1.1.0',
  default_options : ['warning_level=2'],
)

add_languages('fortran', native: false)

py_mod = import('python')

py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

#  Intel Short Vector Math Library (SVML)
is_windows = build_machine.system() == 'windows'

if is_windows
  svml_lib = meson.get_compiler('c').find_library('svml_dispmd')
else
  svml_path = '/opt/intel/oneapi/compiler/latest/lib'
  svml_lib = meson.get_compiler('c').find_library('svml', dirs: svml_path)
endif

incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

# Fortran source files
srcs_spherepack =  ['src/gaqd.f','src/geo2math.f','src/shses.f','src/shaes.f','src/sshifte.f','src/vshifte.f',
                    'src/advec.f','src/shallow_api.f',
                    'src/vhaes.f','src/vhses.f','src/shsgs.f',
                    'src/shags.f','src/vhags.f','src/vhsgs.f',
                    'src/sphcom.f','src/hrfft.f','src/shaec.f',
                    'src/shagc.f','src/shsec.f','src/shsgc.f',
                     'src/vhaec.f','src/vhagc.f','src/vhsec.f','src/vlapec.f','src/vlapes.f',
                     'src/vhsgc.f','src/divec.f','src/dives.f','src/divgc.f','src/divgs.f','src/gradec.f','src/gradgc.f','src/grades.f','src/gradgs.f','src/idivec.f','src/idives.f','src/idivgc.f','src/idivgs.f','src/idvtec.f','src/idvtes.f','src/idvtgc.f','src/idvtgs.f','src/igradec.f','src/igrades.f','src/igradgc.f','src/igradgs.f','src/isfvpec.f','src/isfvpes.f','src/isfvpgc.f','src/isfvpgs.f','src/islapec.f','src/islapes.f','src/islapgc.f','src/islapgs.f','src/ivlapec.f','src/ivlapgc.f','src/ivlapgs.f','src/ivlapes.f','src/ivrtec.f','src/ivrtes.f','src/ivrtgc.f','src/ivrtgs.f','src/sfvpec.f','src/sfvpes.f','src/sfvpgc.f','src/sfvpgs.f','src/slapec.f','src/slapgc.f','src/slapgs.f','src/slapes.f','src/vlapgc.f','src/vlapgs.f','src/vrtec.f','src/vrtes.f','src/vrtgc.f','src/vrtgs.f','src/vtsec.f','src/vtses.f','src/vtsgc.f','src/vtsgs.f',
                     'src/ihgeod.f','src/alf.f']

srcs_local = ['src/getlegfunc.f', 'src/specintrp.f','src/onedtotwod.f',
              'src/onedtotwod_vrtdiv.f','src/twodtooned.f','src/twodtooned_vrtdiv.f',
              'src/multsmoothfact.f','src/lap.f','src/invlap.f']

# https://github.com/numpy/numpy/issues/25199
spherepack_source = custom_target('_spherepackmodule.c',
  input : ['_spherepack.pyf'],
  output : ['_spherepackmodule.c', '_spherepack-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '--lower', '--build-dir', 'subprojects/pyspharm']
)

py.extension_module('_spherepack',
  srcs_local + srcs_spherepack + spherepack_source,
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : [py_dep, svml_lib],
  install : true,
  install_dir: py.get_install_dir() / 'spharm_fortran_reference' / 'pyspharm'
)
