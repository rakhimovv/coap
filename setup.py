try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension
import numpy

# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# efficient mesh extraction (Occupancy networks: Learning 3d reconstruction in function space, CVPR 2019)
mise_module = Extension(
    'coap.tools.libmise.mise',
    sources=[
        'coap/tools/libmise/mise.pyx'
    ],
)

# occupancy checks needed for training
libmesh_module = Extension(
    'coap.tools.libmesh.triangle_hash',
    sources=[
        'coap/tools/libmesh/triangle_hash.pyx'
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[numpy_include_dir]
)

ext_modules = [
    libmesh_module,
    mise_module,
]

setup(
    name='coap',
    version='0.0.1',
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    },
    url='',
    license='',
    author='',
    author_email='',
    description=''
)