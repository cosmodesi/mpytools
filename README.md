# mpytools

**mpytools** is a Python toolkit to manage/read/write catalog and array-type data over multiple processes.

Example notebooks are provided in directory nb/.

## Requirements

Strict requirements are:

  - numpy
  - mpi4py

Reading/writing fits, hdf5, bigfile, or asdf formats requires respectively:

  - fitsio
  - h5py
  - bigfile
  - asdf

## Installation

### pip

Simply run:
```
python -m pip install git+https://github.com/cosmodesi/mpytools
```

### git

First:
```
git clone https://github.com/cosmodesi/mpytools.git
```
To install the code::
```
python setup.py install --user
```
Or in development mode (any change to Python code will take place immediately)::
```
python setup.py develop --user
```

## License

**mpytools** is free software distributed under a BSD3 license. For details see the [LICENSE](https://github.com/cosmodesi/mpytools/blob/main/LICENSE).

## Credits

[nbodykit](https://github.com/bccp/nbodykit) for recipe for handling of various file formats and MPI array utilities.
Edmond Chaussidon for creating this package name.
