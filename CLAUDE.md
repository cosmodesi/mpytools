# mpytools — Codebase Guide

A Python toolkit for managing, reading, and writing **catalog and array-type data across multiple MPI processes**. Inspired by [nbodykit](https://github.com/bccp/nbodykit), targeted at cosmology/DESI pipelines.

**Strict requirements:** `numpy`, `mpi4py`
**Optional (for file formats):** `fitsio` (FITS), `h5py` (HDF5), `bigfile`, `asdf`

---

## Repository layout

```
mpytools/
├── mpytools/
│   ├── __init__.py       # re-exports: Catalog, array, random, CurrentMPIComm, setup_logging
│   ├── utils.py          # base infrastructure (MPI comm stack, logging, helpers)
│   ├── core.py           # MPI-scattered array + collective operations
│   ├── catalog.py        # BaseCatalog / Catalog — dict-of-columns with MPI-aware I/O
│   ├── io.py             # FileStack + per-format drivers (FITS, HDF5, BigFile, ASDF, NPY)
│   ├── random.py         # MPIRandomState and seed utilities
│   └── tests/
│       ├── test_catalog.py
│       ├── test_core.py
│       └── test_random.py
├── nb/                   # Example notebooks
├── pyproject.toml
└── README.md
```

---

## Module-by-module reference

### `utils.py`

#### `CurrentMPIComm`
Thread-local stack of MPI communicators. The default is `MPI.COMM_WORLD`.

```python
# Inject current communicator into a function
@CurrentMPIComm.enable
def my_func(..., mpicomm=None): ...

# Temporarily switch communicator
with CurrentMPIComm.enter(sub_comm):
    ...
```

- `CurrentMPIComm.get()` — return current communicator
- `CurrentMPIComm.push(comm)` / `.pop()` — manual stack management

`@CurrentMPIComm.enable` also inspects positional arguments for a `.mpicomm` attribute before falling back to the stack, so passing an `array` object as the first argument is enough.

#### `BaseClass`
Base class (via `BaseMetaClass`) that adds:
- `copy(**kwargs)` — shallow copy, optionally overriding attributes
- `save(filename)` / `load(filename)` — numpy `.npy` serialisation via `__getstate__`/`__setstate__`
- `log_debug`, `log_info`, `log_warning`, `log_error`, `log_critical` — rank-aware logging; pass `rank=0` to restrict output

#### Utilities
| Function | Description |
|---|---|
| `setup_logging(level, ...)` | Configure root logger with rank/time prefix |
| `match1d(id1, id2)` | Mutual index match between two unique ID arrays |
| `match1d_to(id1, id2)` | Indices where `id1` matches every element of `id2` |
| `weighted_quantile(x, q, weights, ...)` | Weighted quantiles (methods: linear, lower, higher, nearest, midpoint) |
| `MemoryMonitor()` | Context manager tracking RSS memory and elapsed time |
| `is_sequence(item)` | `True` for `list` or `tuple` |
| `list_concatenate(li)` | Flatten a list of sequences one level |
| `mkdir(dirname)` | `os.makedirs` ignoring `OSError` |

---

### `core.py` — MPI-scattered arrays

Public API exported via `__all__` and `from .core import *`.

#### `Slice`
Smart wrapper around a Python `slice` or integer/boolean index array. The central primitive for all distributed indexing.

```python
Slice(10)               # slice(0, 10, 1)
Slice(2, 20, 3)
Slice([1, 4, 7])        # array-based
Slice(None, size=10)    # slice(0, 10, 1)
```

Key methods:
| Method | Description |
|---|---|
| `.is_array` | `True` if backed by an index array |
| `.size` / `len()` | Number of elements |
| `.min` / `.max` | Min/max index |
| `.to_array()` | Convert to numpy array |
| `.to_slices()` | Decompose into python `slice` objects |
| `.split(nsplits)` | Split into `nsplits` sub-slices |
| `.find(*args)` | Indices of another slice within this one |
| `.slice(*args)` | Compose slices: `arr[sl1.slice(sl2)] == arr[sl1][sl2]` |
| `.shift(offset, stop)` | Translate indices |
| `.snap(*others)` | (classmethod) Merge adjacent compatible slices |
| `.send/.recv/.sendrecv` | Point-to-point MPI communication |

#### `MPIScatteredSource`
Manages globally-indexed slices split across ranks; performs cross-rank data exchange on `.get()`.

```python
source = MPIScatteredSource(slice(start, stop, 1))
data = source.get(local_array, global_slice)
```

Used internally by `cslice`, `creshape`, `cconcatenate`.

#### `array`
Subclass of `np.ndarray` with an `mpicomm` attribute. All numpy ufuncs work transparently (local computation). Collective operations communicate across ranks.

```python
arr = mpy.array(local_data)                    # already scattered
arr = mpy.array(global_data, mpiroot=0)        # scatter from rank 0
```

Collective methods (results broadcast to all ranks):
| Method | Description |
|---|---|
| `csize` / `cshape` | Global size / shape |
| `creshape(*shape)` | Reshape globally (may exchange data) |
| `cslice(*args)` | Global slice |
| `gather(mpiroot=0)` | Gather to one rank (or all if `None`) |
| `all_to_all(counts)` | Redistribute rows |
| `reduce(mpiroot, op)` | MPI reduce |
| `csum/cprod/cmean/cvar/cstd` | Reductions along axes |
| `cmin/cmax/cargmin/cargmax` | Extrema and their indices+rank |
| `csort(axis)` | Global sort (gather→sort→scatter) |

Constructors (`cshape` kwarg sets the collective first dimension):
```python
mpy.zeros(cshape=(1000,))
mpy.ones(cshape=(1000, 3))
mpy.empty(cshape=100)
mpy.full(cshape=100, fill_value=np.nan)
```

#### Low-level MPI primitives
All accept `mpicomm=None` (resolved via `@CurrentMPIComm.enable`).
Use `mpiroot=Ellipsis` (or `None`) to broadcast result to all ranks.

| Function | Description |
|---|---|
| `gather(data, mpiroot=0)` | Gatherv; handles structured arrays, >2 GB via custom MPI dtype |
| `scatter(data, size, mpiroot=0)` | Scatterv; balanced by default |
| `bcast(data, mpiroot=0)` | Broadcast array from root |
| `reduce(data, op, mpiroot=0)` | MPI reduce (op: `'sum'`, `'prod'`, `'min'`, `'max'`) |
| `all_to_all(data, counts)` | Alltoallv; balances load if `counts=None` |
| `send(data, dest, tag)` | Point-to-point send (shape+dtype first, then data) |
| `recv(source, tag)` | Point-to-point recv |
| `sendrecv(data, dest, source, ...)` | Combined send+recv |
| `local_size(size)` | Divide global size into local chunk for current rank |
| `csize(data)` / `cshape(data)` | Global size / shape of a distributed array |
| `creshape(arr, cshape)` | Globally reshape (may exchange data) |
| `cslice(arr, *args)` | Globally slice |
| `cconcatenate(*arrays, axis)` | Globally concatenate (preserving order) |
| `cappend(array, other)` | Globally append |
| `csort(data, axis, kind)` | Global sort (naive: gather→sort→scatter) |
| `cquantile(data, q, weights, ...)` | Global weighted quantile (naive gather) |
| `caverage(a, axis, weights, ...)` | Global weighted average |
| `cvar/cstd/ccov/ccorrcoef` | Global variance / std / covariance / correlation |

---

### `catalog.py` — Catalog

#### `BaseCatalog`
Dictionary of column arrays, MPI-scattered along the first axis. Columns are loaded lazily from a `FileStack` source on first access.

```python
cat = Catalog.read('data.fits')           # lazy — no data read yet
ra  = cat['RA']                           # triggers read of 'RA'
cat['WEIGHT'] = np.ones(cat.size)         # set column
```

**Construction:**
```python
Catalog(data={'RA': ra, 'DEC': dec}, attrs={'survey': 'DESI'})
Catalog.from_array(structured_array)
Catalog.from_array(array, mpiroot=0)      # scatter from rank 0
```

**Column access:**
| Method | Description |
|---|---|
| `cat[col]` | Return column as `mpy.array` |
| `cat[[col1, col2]]` | Return new catalog with subset of columns |
| `cat[col] = arr` | Set column |
| `del cat[col]` | Delete column |
| `cat.get(col, default, return_type=)` | Get with optional default; `return_type='nparray'` or `'mpyarray'` |
| `cat.cget(col, mpiroot=0)` | Gather column to `mpiroot` (or all if `None`) |
| `cat.set(col, arr)` | Set with size check |
| `cat.update(other)` | Update columns from dict or catalog |
| `cat.pop(col)` | Get and delete |
| `cat.columns(include=, exclude=)` | List columns; patterns are regex or `*`-glob |

**Shape / indexing:**
| Method | Description |
|---|---|
| `cat.size` / `len(cat)` | Local row count |
| `cat.csize` | Global row count |
| `cat.cindex()` | Global row indices (0-based) |
| `cat.slice(*args)` | Local slice → new catalog |
| `cat.cslice(*args)` | Global slice → new catalog |
| `cat[int_or_slice]` | Calls `.slice()` |

**Concatenation:**
```python
Catalog.concatenate(cat1, cat2)           # local (no cross-rank reorder)
Catalog.cconcatenate(cat1, cat2)          # global (preserves order)
cat1.append(cat2)
cat1.cappend(cat2)
```
Both accept `intersection=True` to keep only common columns.

**MPI redistribution:**
```python
cat.gather(mpiroot=0)                     # gather to one rank
Catalog.scatter(cat, mpiroot=0)           # scatter from one rank
cat.all_to_all(counts)                    # arbitrary redistribution
cat.csort(orderby='HPIX')                 # sort by column (requires mpsort)
```

**I/O:**
```python
Catalog.read('data.fits')                 # lazy read
Catalog.read(['a.fits', 'b.fits'])        # multiple files
cat.write('out.fits')
cat.save('out.npy')                       # gathers to rank 0 first
Catalog.load('out.npy')                   # scatters back
```

**Convenience array constructors** (return `mpy.array` of local size):
`cat.empty()`, `cat.zeros()`, `cat.ones()`, `cat.full(val)`, `cat.falses()`, `cat.trues()`, `cat.nans()`

---

### `io.py` — File I/O

#### `FileStack`
Wraps one or more `BaseFile` instances and load-balances rows across MPI ranks automatically. `BaseCatalog.read` / `.write` delegate here.

```python
fs = FileStack('a.fits', 'b.fits')
fs = FileStack('data.hdf5', filetype='hdf5', group='catalog')
arrays = fs.read(['RA', 'DEC'])
```

Lazy slicing (deferred until `read()`):
```python
fs.slice(0, 100)         # local rows 0–99
fs.cslice(0, 1000)       # global rows 0–999
```

Concatenation:
```python
FileStack.concatenate(fs1, fs2)    # local (no reorder)
FileStack.cconcatenate(fs1, fs2)   # global (preserves order)
```

#### File format drivers (all subclass `BaseFile`)
| Class | Extension(s) | Backend |
|---|---|---|
| `FitsFile` | `.fits`, `.fits.gz`, `.fits.bz2` | `fitsio` |
| `HDF5File` | `.hdf5`, `.h5`, `.hdf` | `h5py` |
| `BigFile` | `.bigfile` (dir) | `bigfile` |
| `AsdfFile` | `.asdf` | `asdf` |
| `NumpyFile` | `.npy` | `numpy` |

File type is auto-detected from extension; override with `filetype='fits'` etc.

Headers are read in parallel (round-robin, one file per rank) and broadcast.

---

### `random.py`

```python
from mpytools import random

# Seed utilities
random.set_common_seed(42)          # same seed on all ranks
random.set_independent_seed(42)     # different seed per rank

seeds = random.bcast_seed(42, size=8)   # array of 8 seeds, broadcast

# Per-rank RNG matched to a catalog size
rng = random.MPIRandomState(size=cat.size, seed=42)
u   = rng.uniform(0., 1.)
n   = rng.normal(loc=0., scale=1.)
p   = rng.poisson(lam=5.)
```

`MPIRandomState` ensures that the draws are **globally consistent** regardless of the number of MPI ranks — i.e., element `i` always gets the same value.

---

## Design conventions

- **`mpiroot=Ellipsis`** (or `None`) means "broadcast to all ranks" in `gather`, `reduce`, `cget`, etc.
- **`@CurrentMPIComm.enable`** on any function that needs `mpicomm`. The decorator resolves it from (in order): kwarg, first positional arg's `.mpicomm`, stack top.
- **Structured arrays** are handled natively throughout (column-by-column recursion in gather/scatter/reduce).
- **>2 GB arrays** are handled via `MPI.BYTE.Create_contiguous` custom dtypes to bypass mpi4py's 2 GB pickling limit.
- **`object` dtypes** are not supported in any MPI primitive — raise `ValueError` early.
- **Lazy I/O**: `Catalog.read()` stores `None` per column; first `cat[col]` triggers `FileStack.read([col])`.
- **`cast_array_wrapper`**: catalog methods default to `return_type='mpyarray'`; pass `return_type='nparray'` or `return_type=None` to skip the cast.
