from distutils.core import setup
from Cython.Build import cythonize

setup(name='jiebafenci',
     ext_modules=cythonize('jiebafenci.py'))