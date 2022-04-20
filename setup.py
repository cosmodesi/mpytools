import os
import sys
from setuptools import setup

package_basename = 'mpytools'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), package_basename))
import _version
version = _version.__version__


setup(name=package_basename,
      version=version,
      author='cosmodesi',
      author_email='',
      description='package with MPI utilities',
      license='BSD3',
      url='http://github.com/cosmodesi/mpytools',
      install_requires=['numpy', 'scipy'],
      packages=[package_basename])
