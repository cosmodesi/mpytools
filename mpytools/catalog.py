"""Base classes to handle catalog of objects."""

import os
import functools

import numpy as np

from . import mpi, utils
from .mpi import CurrentMPIComm
from .utils import BaseClass, is_sequence
from .array import Slice, MPIScatteredSource, MPIScatteredArray
from .io import FileStack, _select_columns


def vectorize_columns(func):

    @functools.wraps(func)
    def wrapper(self, column, **kwargs):
        if not is_sequence(column):
            return func(self, column, **kwargs)
        toret = [func(self, col, **kwargs) for col in column]
        if all(t is None for t in toret):  # in case not broadcast to all ranks
            return None
        return np.asarray(toret)

    return wrapper


def _get_shape(size, itemshape):
    # join size and itemshape to get total shape
    if np.ndim(itemshape) == 0:
        return (size, itemshape)
    return (size,) + tuple(itemshape)


def _dict_to_array(data, struct=True):
    """
    Return dict as numpy array.

    Parameters
    ----------
    data : dict
        Data dictionary of name: array.

    struct : bool, default=True
        Whether to return structured array, with columns accessible through e.g. ``array['Position']``.
        If ``False``, numpy will attempt to cast types of different columns.

    Returns
    -------
    array : array
    """
    array = [(name, data[name]) for name in data]
    if struct:
        array = np.empty(array[0][1].shape[0], dtype=[(name, col.dtype, col.shape[1:]) for name, col in array])
        for name in data: array[name] = data[name]
    else:
        array = np.array([col for _, col in array])
    return array


class BaseCatalog(BaseClass):

    _attrs = ['attrs']

    """Base class that represents a catalog, as a dictionary of columns stored as arrays."""

    @CurrentMPIComm.enable
    def __init__(self, data=None, columns=None, attrs=None, mpicomm=None):
        """
        Initialize :class:`BaseCatalog`.

        Parameters
        ----------
        data : dict, BaseCatalog
            Dictionary of {name: array}.

        columns : list, default=None
            List of column names.
            Defaults to ``data.keys()``.

        attrs : dict, default=None
            Dictionary of other attributes.

        mpicomm : MPI communicator, default=None
            The current MPI communicator.
        """
        self.data = {}
        if columns is None:
            columns = list((data or {}).keys())
        if data is not None:
            for name in columns:
                self[name] = data[name]
        self.attrs = attrs or {}
        self.mpicomm = mpicomm
        self.mpiroot = 0

    def is_mpi_root(self):
        """Whether current rank is root."""
        return self.mpicomm.rank == self.mpiroot

    @classmethod
    def from_nbodykit(cls, catalog, columns=None):
        """
        Build new catalog from **nbodykit**.

        Parameters
        ----------
        catalog : nbodykit.base.catalog.CatalogSource
            **nbodykit** catalog.

        columns : list, default=None
            Columns to import. Defaults to all columns.

        Returns
        -------
        catalog : BaseCatalog
        """
        if columns is None: columns = catalog.columns
        data = {col: catalog[col].compute() for col in columns}
        return cls(data, mpicomm=catalog.comm, attrs=catalog.attrs)

    def to_nbodykit(self, columns=None):
        """
        Return catalog in **nbodykit** format.

        Parameters
        ----------
        columns : list, default=None
            Columns to export. Defaults to all columns.

        Returns
        -------
        catalog : nbodykit.source.catalog.ArrayCatalog
        """
        if columns is None: columns = self.columns()
        source = {col: self[col] for col in columns}
        from nbodykit.lab import ArrayCatalog
        attrs = {key: value for key, value in self.attrs.items() if key != 'fitshdr'}
        return ArrayCatalog(source, **attrs)

    def __len__(self):
        """Return catalog (local) length (``0`` if no column)."""
        keys = list(self.data.keys())
        if not keys:
            if self.has_source is not None:
                return self._source.size
            return 0
        return len(self[keys[0]])

    @property
    def size(self):
        """Equivalent for :meth:`__len__`."""
        return len(self)

    @property
    def csize(self):
        """Return catalog global size, i.e. sum of size in each process."""
        return self.mpicomm.allreduce(len(self))

    def columns(self, include=None, exclude=None):
        """
        Return catalog column names, after optional selections.

        Parameters
        ----------
        include : list, string, default=None
            Single or list of *regex* patterns to select column names to include.
            Defaults to all columns.

        exclude : list, string, default=None
            Single or list of *regex* patterns to select column names to exclude.
            Defaults to no columns.

        Returns
        -------
        columns : list
            Return catalog column names, after optional selections.
        """
        toret = None

        if self.is_mpi_root():
            columns = list(self.data.keys())
            source = getattr(self, '_source', None)
            if source is not None:
                columns += [column for column in source.columns if column not in columns]
            toret = _select_columns(columns, include=include, exclude=exclude)

        return self.mpicomm.bcast(toret, root=self.mpiroot)

    def __contains__(self, column):
        """Whether catalog contains column name ``column``."""
        return column in self.data or (self.has_source and column in self._source.columns)

    def __iter__(self):
        """Iterate on catalog columns."""
        return iter(self.data)

    def cindices(self):
        """Row numbers in the global catalog."""
        sizes = self.mpicomm.allgather(len(self))
        sizes = np.cumsum([0] + sizes)
        return sizes[self.mpicomm.rank] + np.arange(len(self))

    def zeros(self, itemshape=(), dtype='f8'):
        """Return array of size :attr:`size` filled with zero."""
        return np.zeros(_get_shape(len(self), itemshape), dtype=dtype)

    def ones(self, itemshape=(), dtype='f8'):
        """Return array of size :attr:`size` filled with one."""
        return np.ones(_get_shape(len(self), itemshape), dtype=dtype)

    def full(self, fill_value, itemshape=(), dtype='f8'):
        """Return array of size :attr:`size` filled with ``fill_value``."""
        return np.full(_get_shape(len(self), itemshape), fill_value, dtype=dtype)

    def falses(self, itemshape=()):
        """Return array of size :attr:`size` filled with ``False``."""
        return self.zeros(itemshape=itemshape, dtype=np.bool_)

    def trues(self, itemshape=()):
        """Return array of size :attr:`size` filled with ``True``."""
        return self.ones(itemshape=itemshape, dtype=np.bool_)

    def nans(self, itemshape=()):
        """Return array of size :attr:`size` filled with :attr:`numpy.nan`."""
        return self.ones(itemshape=itemshape) * np.nan

    @property
    def has_source(self):
        return getattr(self, '_source', None) is not None

    def get(self, column, *args, return_type='scattered', **kwargs):
        """Return catalog (local) column ``column`` if exists, else return provided default."""

        def cast(array):
            if return_type is None:
                return array
            return_type_lower = return_type.lower()
            if return_type_lower == 'scattered':
                return MPIScatteredArray(array, mpicomm=self.mpicomm)
            if return_type_lower in ['array', 'ndarray']:
                return np.array(array, copy=False)
            raise ValueError('return_type must be in ["scattered", "array", "ndarray"]')

        has_default = False
        if args:
            if len(args) > 1:
                raise SyntaxError('Too many arguments!')
            has_default = True
            default = args[0]
        if kwargs:
            if len(kwargs) > 1:
                raise SyntaxError('Too many arguments!')
            has_default = True
            default = kwargs['default']
        if column in self.data:
            return cast(self.data[column])
        # if not in data, try in _source
        if self.has_source and column in self._source.columns:
            self.data[column] = self._source.read(column)
            return cast(self.data[column])
        if has_default:
            return default
        raise KeyError('Column {} does not exist'.format(column))

    def set(self, column, item):
        """Set column of name ``column``."""
        self.data[column] = item

    def cget(self, column, mpiroot=None):
        """
        Return on process rank ``root`` catalog global column ``column`` if exists, else return provided default.
        If ``mpiroot`` is ``None`` or ``Ellipsis`` return result on all processes.
        """
        if mpiroot is None: mpiroot = Ellipsis
        return self.get(column, return_type='scattered').gathered(mpiroot=mpiroot)

    def cslice(self, *args):
        """
        Perform global slicing of catalog,
        e.g. ``catalog.cslice(0, 100, 1)`` will return a new catalog of global size ``100``.
        Same reference to :attr:`attrs`.
        """
        new = self.copy()
        cumsizes = np.cumsum([0] + self.mpicomm.allgather(self.size))
        global_slice = Slice(*args, size=cumsizes[-1])
        local_slice = global_slice.split(self.mpicomm.size)[self.mpicomm.rank]
        source = MPIScatteredSource(slice(cumsizes[self.mpicomm.rank], cumsizes[self.mpicomm.rank + 1], 1))
        for column in self.columns():
            if column in self.data:
                new[column] = source.get(self.get(column, return_type=None), local_slice)
        if self.has_source:
            new._source = self._source.cslice(global_slice)
        return new

    @classmethod
    def concatenate(cls, *others, keep_order=False):
        """
        Concatenate catalogs together.

        Parameters
        ----------
        others : list
            List of :class:`BaseCatalog` instances.

        keep_order : bool, default=False
            Whether to keep row order, which requires costly MPI-gather/scatter operations.
            If ``False``, rows on each MPI process will be added to those of the same MPI process.

        Returns
        -------
        new : BaseCatalog

        Warning
        -------
        :attr:`attrs` of returned catalog contains, for each key, the last value found in ``others`` :attr:`attrs` dictionaries.
        """
        if not others:
            raise ValueError('Provide at least one {} instance.'.format(cls.__name__))
        if len(others) == 1 and is_sequence(others[0]):
            others = others[0]
        attrs = {}
        for other in others: attrs.update(other.attrs)
        others = [other for other in others if other.columns()]

        new = others[0].copy()
        new.attrs = attrs
        new_columns = new.columns()

        for other in others:
            other_columns = other.columns()
            if other.mpicomm is not new.mpicomm:
                raise ValueError('Input catalogs with different mpicomm')
            if new_columns and other_columns and set(other_columns) != set(new_columns):
                raise ValueError('Cannot extend samples as columns do not match: {} != {}.'.format(other_columns, new_columns))

        in_data = {column: any(column in other.data for other in others) for column in new_columns}
        if keep_order and any(in_data.values()):
            source = []
            for other in others:
                cumsizes = np.cumsum([0] + other.mpicomm.allgather(other.size))
                source.append(MPIScatteredSource(slice(cumsizes[other.mpicomm.rank], cumsizes[other.mpicomm.rank + 1], 1)))
            source = MPIScatteredSource.concatenate(*source)

        for column in new_columns:
            if in_data[column]:
                if keep_order:
                    new[column] = source.get([other.get(column, return_type=None) for other in others])
                else:
                    new[column] = np.concatenate([other.get(column, return_type=None) for other in others])

        source = [other._source for other in others if other.has_source]
        if source:
            source = FileStack.concatenate(*source)
            new._source = source

        return new

    def append(self, other, **kwargs):
        """Extend catalog with ``other``."""
        return self.concatenate(self, other, **kwargs)

    def extend(self, other, **kwargs):
        """Extend catalog with ``other``."""
        new = self.append(self, other, **kwargs)
        self.__dict__.update(new.__dict__)

    def to_array(self, columns=None, struct=True):
        """
        Return catalog as *numpy* array.

        Parameters
        ----------
        columns : list, default=None
            Columns to use. Defaults to all catalog columns.

        struct : bool, default=True
            Whether to return structured array, with columns accessible through e.g. ``array['Position']``.
            If ``False``, *numpy* will attempt to cast types of different columns.

        Returns
        -------
        array : array
        """
        if columns is None:
            columns = self.columns()
        data = {col: self[col] for col in columns}
        return _dict_to_array(data, struct=struct)

    @classmethod
    @CurrentMPIComm.enable
    def from_array(cls, array, columns=None, mpicomm=None, mpiroot=None, **kwargs):
        """
        Build :class:`BaseCatalog` from input ``array``.

        Parameters
        ----------
        array : array, dict
            Input array to turn into catalog.

        columns : list, default=None
            List of columns to read from array.
            If ``None``, inferred from ``array``.

        mpicomm : MPI communicator, default=None
            MPI communicator.

        mpiroot : int, default=None
            If ``None``, input array is assumed to be scattered across all ranks.
            Else the MPI rank where input array is gathered.

        kwargs : dict
            Other arguments for :meth:`__init__`.

        Returns
        -------
        catalog : BaseCatalog
        """
        isstruct = None
        if mpicomm.rank == mpiroot or mpiroot is None:
            isstruct = isdict = not hasattr(array, 'dtype')
            if isdict:
                if columns is None: columns = list(array.keys())
            else:
                isstruct = array.dtype.names is not None
                if isstruct and columns is None: columns = array.dtype.names
        if mpiroot is not None:
            isstruct = mpicomm.bcast(isstruct, root=mpiroot)
            columns = mpicomm.bcast(columns, root=mpiroot)
        columns = list(columns)
        new = cls(data=dict.fromkeys(columns), mpicomm=mpicomm, **kwargs)

        def get(column):
            value = None
            if mpicomm.rank == mpiroot or mpiroot is None:
                if isstruct:
                    value = array.get(column, return_type=None)
                else:
                    value = columns.index(column)
            if mpiroot is not None:
                return mpi.scatter_array(value, mpicomm=mpicomm, root=mpiroot)
            return value

        new.data = {column: get(column) for column in columns}
        return new

    def copy(self, columns=None):
        """Return copy, including column names ``columns`` (defaults to all columns)."""
        new = super(BaseCatalog, self).__copy__()
        if columns is None: columns = list(self.data.keys())
        new.data = {col: self[col] if col in self else None for col in columns}
        if new.has_source: new._source = self._source.copy()
        import copy
        for name in new._attrs:
            if hasattr(self, name):
                tmp = copy.copy(getattr(self, name))
                setattr(new, name, tmp)
        return new

    def deepcopy(self, columns=None):
        """Return copy, including column names ``columns`` (defaults to all columns)."""
        import copy
        new = self.copy(columns=columns)
        for name in self._attrs:
            if hasattr(self, name):
                setattr(new, name, copy.deepcopy(getattr(self, name)))
        new.data = {col: self[col].copy() for col in new}
        return new

    def __getstate__(self):
        """Return this class state dictionary."""
        data = {str(name): col for name, col in self.data.items()}
        state = {'data': data}
        for name in self._attrs:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        """Set the class state dictionary."""
        self.__dict__.update(state)

    @classmethod
    @CurrentMPIComm.enable
    def from_state(cls, state, mpicomm=None):
        """Create class from state."""
        new = cls.__new__(cls)
        new.__setstate__(state)
        new.mpicomm = mpicomm
        new.mpiroot = 0
        return new

    def __getitem__(self, name):
        """Get catalog column ``name`` if string, else return copy with local slice."""
        if isinstance(name, str):
            return self.get(name)
        new = self.copy()
        if isinstance(name, slice):
            new.data = {col: self[col][name] for col in self.data}
            if self.has_source:
                new._source = self._source.slice(name)
        else:
            new.data = {col: self[col][name] for col in self.columns()}
            if self.has_source: del new._source
        return new

    def __setitem__(self, name, item):
        """Set catalog column ``name`` if string, else set slice ``name`` of all columns to ``item``."""
        if isinstance(name, str):
            return self.set(name, item)
        for col in self.columns():
            self[col][name] = item

    def __delitem__(self, name):
        """Delete column ``name``."""
        try:
            del self.data[name]
        except KeyError as exc:
            if self.has_source is not None:
                self._source.columns.remove(name)
            else:
                raise KeyError('Column {} not found'.format(name)) from exc

    def __repr__(self):
        """Return string representation of catalog, including global size and columns."""
        return '{}(size={:d}, columns={})'.format(self.__class__.__name__, self.csize, self.columns())

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and columns? (ignoring :attr:`attrs`)"""
        if not isinstance(other, self.__class__):
            return False
        self_columns = self.columns()
        other_columns = other.columns()
        if set(other_columns) != set(self_columns):
            return False
        assert self.mpicomm == other.mpicomm
        self, other = self.cslice(0, None), other.cslice(0, None)
        for col in self_columns:
            self_value = self.get(col)
            other_value = other.get(col)
            if not all(self.mpicomm.allgather(np.all(self_value == other_value))):
                return False

    @classmethod
    def read(cls, *args, **kwargs):
        source = FileStack(*args, **kwargs)
        new = cls(attrs={'header': source.header}, mpicomm=source.mpicomm)
        new._source = source
        return new

    def write(self, *args, **kwargs):
        """Save catalog to ``filename``."""
        source = FileStack(*args, **kwargs)
        source.write({name: self[name] for name in self.columns()})

    @classmethod
    @CurrentMPIComm.enable
    def load(cls, filename, mpicomm=None):
        """
        Load catalog in *npy* binary format from disk.

        Parameters
        ----------
        mpicomm : MPI communicator, default=None
            The MPI communicator.

        Returns
        -------
        catalog : BaseCatalog
        """
        mpiroot = 0
        if mpicomm.rank == mpiroot:
            cls.log_info('Loading {}.'.format(filename))
            state = np.load(filename, allow_pickle=True)[()]
            data = state.pop('data')
            columns = list(data.keys())
        else:
            state = None
            columns = None
        state = mpicomm.bcast(state, root=mpiroot)
        columns = mpicomm.bcast(columns, root=mpiroot)
        state['data'] = {}
        for name in columns:
            state['data'][name] = mpi.scatter_array(data[name] if mpicomm.rank == mpiroot else None, mpicomm=mpicomm, root=mpiroot)
        return cls.from_state(state, mpicomm=mpicomm)

    def save(self, filename):
        """Save catalog to ``filename`` as *npy* file."""
        if self.is_mpi_root():
            self.log_info('Saving to {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
        state = self.__getstate__()
        state['data'] = {name: self.cget(name, mpiroot=self.mpiroot) for name in self.columns()}
        if self.is_mpi_root():
            np.save(filename, state, allow_pickle=True)


class Catalog(BaseCatalog):

    """A simple catalog."""
