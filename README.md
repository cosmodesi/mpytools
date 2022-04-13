# mpytools

**mpytools** is a Python toolkit to

Example notebooks are provided in directory nb/.

## Requirements

Strict requirements are:

  - numpy
  - mpi4py

## Installation

### pip

Simply run:
```
python -m pip install git+https://github.com/adematti/mpytools
```

### git

First:
```
git clone https://github.com/adematti/mpytools.git
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

**mockfactory** is free software distributed under a GPLv3 license. For details see the [LICENSE](https://github.com/adematti/mpytools/blob/main/LICENSE).

## Credits

[nbodykit](https://github.com/bccp/nbodykit) for recipe for handling of various file formats and MPI array utilities.
Edmond Chaussidon for making up this package name.
