"""Base classes to handle catalog of objects."""

import os
import functools

import numpy as np

from . import utils
from .utils import BaseClass, CurrentMPIComm, is_sequence
from . import core as mpy
from .core import Slice, MPIScatteredSource
from .io import FileStack, select_columns


def _get_shape(size, itemshape):
    # Join size and itemshape to get total shape
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


@CurrentMPIComm.enable
def cast_array(array, return_type=None, mpicomm=None):
    """
    Cast input numpy array.

    Parameters
    ----------
    array : array
        Array to be cast.

    return_type : str, default=None
        If ``None``, directly return ``array``.
        If "nparray", return :class:`np.ndarray` instance.
        If "mpyarray", return :class:`mpyarray` instance.

    mpicomm : MPI communicator, default=None
        The current MPI communicator.

    Returns
    -------
    array : array
    """
    if return_type is None:
        return array
    return_type = return_type.lower()
    if return_type == 'mpyarray':
        return mpy.array(array, mpicomm=mpicomm)
    if return_type in ['nparray']:
        return np.array(array, copy=False)
    raise ValueError('return_type must be in ["mpyarray", "nparray"]')


def cast_array_wrapper(func):
    """Method wrapper applying :func:`cast_array` on result."""
    @functools.wraps(func)
    def wrapper(self, *args, return_type='mpyarray', **kwargs):
        toret = func(self, *args, **kwargs)
        if is_sequence(toret):
            return [cast_array(tt, return_type=return_type, mpicomm=self.mpicomm) for tt in toret]
        if toret is None:
            return toret
        return cast_array(toret, return_type=return_type, mpicomm=self.mpicomm)

    return wrapper


class BaseCatalog(BaseClass):

    _attrs = ['attrs']

    """Base class that represents a catalog, as a dictionary of columns stored as arrays."""

    @CurrentMPIComm.enable
    def __init__(self, data=None, columns=None, attrs=None, mpicomm=None, **kwargs):
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
        self.__dict__.update(self.from_dict(data=data, columns=columns, attrs=attrs, mpicomm=mpicomm, **kwargs).__dict__)

    def rng(self, **kwargs):
        if not kwargs and hasattr(self, '_rng'):
            if self._rng.mpicomm is self.mpicomm and all(self.mpicomm.allgather(self._rng.size == self.size)):
                return self._rng
        from .random import MPIRandomState
        self._rng = MPIRandomState(self.size, **kwargs)
        return self._rng

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
        return cls._new(data, mpicomm=catalog.comm, attrs=catalog.attrs)

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
        return ArrayCatalog(source, **self.attrs)

    def __len__(self):
        """Return catalog (local) length (``0`` if no column)."""
        if self.has_source:
            return self._source.size
        keys = list(self.data.keys())
        if keys:
            return len(self.get(keys[0], return_type=None))
        return 0

    @property
    def size(self):
        """Equivalent for :meth:`__len__`."""
        return len(self)

    @property
    def csize(self):
        """Return catalog collective size, i.e. sum of size within each process."""
        return self.mpicomm.allreduce(len(self))

    def columns(self, include=None, exclude=None):
        """
        Return catalog column names, after optional selections.

        Parameters
        ----------
        include : list, string, default=None
            Single or list of *regex* or Unix-style patterns to select column names to include.
            Defaults to all columns.

        exclude : list, string, default=None
            Single or list of *regex* or Unix-style patterns to select column names to exclude.
            Defaults to no columns.

        Returns
        -------
        columns : list
            Catalog column names, after optional selections.
        """
        columns = list(self.data.keys())
        return select_columns(columns, include=include, exclude=exclude)

    def __contains__(self, column):
        """Whether catalog contains column name ``column``."""
        return column in self.data

    def __iter__(self):
        """Iterate on catalog columns."""
        return iter(self.data)

    @cast_array_wrapper
    def cindex(self):
        """Row numbers in the global catalog."""
        cumsize = sum(self.mpicomm.allgather(len(self))[:self.mpicomm.rank])
        return cumsize + np.arange(len(self))

    @cast_array_wrapper
    def empty(self, itemshape=(), **kwargs):
        """Empty array of size :attr:`size`."""
        return np.empty(_get_shape(len(self), itemshape), **kwargs)

    @cast_array_wrapper
    def zeros(self, itemshape=(), **kwargs):
        """Array of size :attr:`size` filled with zero."""
        return np.zeros(_get_shape(len(self), itemshape), **kwargs)

    @cast_array_wrapper
    def ones(self, itemshape=(), **kwargs):
        """Array of size :attr:`size` filled with one."""
        return np.ones(_get_shape(len(self), itemshape), **kwargs)

    @cast_array_wrapper
    def full(self, fill_value, itemshape=(), **kwargs):
        """Array of size :attr:`size` filled with ``fill_value``."""
        return np.full(_get_shape(len(self), itemshape), fill_value, **kwargs)

    def falses(self, itemshape=()):
        """Array of size :attr:`size` filled with ``False``."""
        return self.zeros(itemshape=itemshape, dtype='?')

    def trues(self, itemshape=()):
        """Array of size :attr:`size` filled with ``True``."""
        return self.ones(itemshape=itemshape, dtype='?')

    def nans(self, itemshape=(), **kwargs):
        """Array of size :attr:`size` filled with :attr:`np.nan`."""
        return self.ones(itemshape=itemshape, **kwargs) * np.nan

    @property
    def header(self):
        return self._header

    @property
    def source(self):
        if self.has_source:
            return self._source
        raise AttributeError(f'{self.__class__.__name__} has no source, i.e. no file has been read')

    @property
    def has_source(self):
        """Whether a "source" (typically :class:`FileStack` instance) is attached to current catalog."""
        toret = getattr(self, '_source', None) is not None
        if toret and all(value is not None for value in self.data.values()): # or {key for key, value in self.data.items() if value is not None} == set(self._source.columns):
            # We have read everything
            toret = False
            self._source = None
        return toret

    @cast_array_wrapper
    def get(self, column, *args, **kwargs):
        """
        Return catalog (local) column(s) ``column`` if exists, else return provided default.
        Pass ``return_type`` to specify output type, see :func:`cast_array`.
        """
        isscalar = isinstance(column, str)
        if isscalar: column = [column]
        default = [None] * len(column)
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
        if not is_sequence(default):
            default = [default] * len(column)
        elif len(default) != len(column):
            raise ValueError('Provide as many default values as requested columns')
        read = []
        for c in column:
            if c in self.data and self.data[c] is None:
                read.append(c)
        if read:
            arrays = self.source.read(read)  # may be faster, for e.g. fits
            for c, array in zip(read, arrays):
                self.data[c] = array
        toret = []
        for c, d in zip(column, default):
            if c in self.data:
                toret.append(self.data[c])
            elif has_default:
                toret.append(d)
            else:
                raise KeyError(f'Column {c} does not exist')
        if isscalar:
            toret = toret[0]
        return toret

    def set(self, column, item):
        """Set column of name(s) ``column``."""
        isscalar = isinstance(column, str)
        if isscalar: column = [column]
        if not is_sequence(item):
            item = [item] * len(column)
        elif len(item) != len(column):
            raise ValueError('Provide as many values as requested columns')
        for c, i in zip(column, item):
            value = self.data[c] = np.atleast_1d(i)
            size = self.size
            if len(value) != size:
                raise ValueError('Catalog size is {:d}, but input column is of length {:d}'.format(size, len(value)))

    def cget(self, *args, mpiroot=0, **kwargs):
        """
        Return on rank ``mpiroot`` catalog global column ``column`` if exists, else provided default.
        If ``mpiroot`` is ``None`` or ``Ellipsis`` result is broadcast on all processes.
        """
        if mpiroot is None: mpiroot = Ellipsis
        toret = self.get(*args, return_type='mpyarray', **kwargs)
        if is_sequence(toret):
            return [tt.gather(mpiroot=mpiroot) for tt in toret]
        return toret.gather(mpiroot=mpiroot)

    def gather(self, mpiroot=0):
        """
        Return new catalog, gathered on rank ``mpiroot``.
        If ``mpiroot`` is ``None`` or ``Ellipsis`` result is broadcast on all processes.
        """
        columns = self.columns()
        data = dict(zip(columns, self.cget(columns, mpiroot=mpiroot)))
        new = self.copy()
        if mpiroot is None:
            new.data = data
            return new
        from mpi4py import MPI
        if self.mpicomm.rank == mpiroot:
            new.mpicomm = MPI.COMM_SELF
            new.data = data
        else:
            new = None
        return new

    @classmethod
    @CurrentMPIComm.enable
    def scatter(cls, catalog, mpiroot=0, mpicomm=None):
        """Return new catalog, scattered from rank ``mpiroot``."""
        columns = mpicomm.bcast(catalog.columns() if mpicomm.rank == mpiroot else None, root=mpiroot)
        attrs = mpicomm.bcast(catalog.attrs if mpicomm.rank == mpiroot else None, root=mpiroot)
        data = {}
        for name in columns:
            data[name] = mpy.scatter(catalog[name] if mpicomm.rank == mpiroot else None, mpicomm=mpicomm, mpiroot=mpiroot)
        return cls.from_dict(data=data, columns=columns, attrs=attrs, mpicomm=mpicomm)

    def slice(self, *args):
        """
        Slice catalog (locally), e.g.:
        >>> catalog.slice(0, 100, 1)  # catalog of local size :attr:`size` <= 100
        Same reference to :attr:`attrs`.
        """
        new = self.copy()
        local_slice = Slice(*args, size=new.size).idx
        if new.has_source:
            try:
                new._source = new._source.slice(local_slice)
            except AssertionError:  # general index, load all columns
                new.get(new.columns())
        new.data = {column: array[local_slice] if array is not None else None for column, array in new.data.items()}
        return new

    def cslice(self, *args):
        """
        Slice catalog (collectively), e.g.:
        >>> catalog.cslice(0, 100, 1)  # catalog of collective size :attr:`csize`  <= 100
        Same reference to :attr:`attrs`.
        """
        new = self.copy()
        cumsizes = np.cumsum([0] + new.mpicomm.allgather(new.size))
        global_slice = Slice(*args, size=cumsizes[-1])
        local_slice = global_slice.split(new.mpicomm.size)[new.mpicomm.rank]
        if new.has_source:
            try:
                new._source = new._source.cslice(global_slice)
            except AssertionError:  # general index, load all columns
                new.get(new.columns())
        in_data = [column for column in new.data if new.data[column] is not None]
        if in_data:
            source = MPIScatteredSource(slice(cumsizes[new.mpicomm.rank], cumsizes[new.mpicomm.rank + 1], 1))
            for column, array in zip(in_data, new.get(in_data, return_type=None)):
                new.data[column] = source.get(array, local_slice)
        return new

    @classmethod
    def concatenate(cls, *others, intersection=False):
        """
        Concatenate catalogs together, locally:
        no data is exchanged between processes, but order is not preserved,
        e.g. the first rank will receive the beginning of all input catalogs.

        Parameters
        ----------
        others : list
            List of :class:`BaseCatalog` instances.

        intersection : bool, default=False
            If ``True``, restrict to columns that are in all input catalogs.
            Else, if input catalogs have different columns, a :class:`ValueError` will be raised.

        Returns
        -------
        new : BaseCatalog

        Warning
        -------
        :attr:`attrs` of returned catalog contains, for each key, the last value found in ``others`` :attr:`attrs` dictionaries.
        """
        if not others:
            raise ValueError(f'Provide at least one {cls.__name__} instance.')
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        attrs = {}
        for other in others: attrs.update(other.attrs)
        others = [other for other in others if other.columns()]

        new = others[0].copy()
        new.attrs = attrs

        for other in others:
            if other.mpicomm is not new.mpicomm:
                raise ValueError('Input catalogs with different mpicomm')

        new_columns = new.columns()

        if intersection:
            for other in others:
                new_columns = [column for column in new_columns if column in other.columns()]
        else:
            for other in others:
                other_columns = other.columns()
                if new_columns and other_columns and set(other_columns) != set(new_columns):
                    raise ValueError(f'Cannot concatenate catalogs as columns do not match: {other_columns} != {new_columns}.')

        in_data = [column for column in new_columns if any(other.data[column] is not None for other in others)]
        if in_data:
            arrays = zip(*[other.get(in_data, return_type=None) for other in others])
            for column, arrays in zip(in_data, arrays):
                new.data[column] = np.concatenate(arrays, axis=0)

        source = [other._source for other in others if other.has_source]
        if source:
            source = FileStack.concatenate(*source)
            new._source = source

        return new

    def append(self, other, **kwargs):
        """(Locally) append ``other`` to current catalog."""
        return self.concatenate(self, other, **kwargs)

    @classmethod
    def cconcatenate(cls, *others, intersection=False):
        """
        Concatenate catalogs together, preserving global order.

        Parameters
        ----------
        others : list
            List of :class:`BaseCatalog` instances.

        intersection : bool, default=False
            If ``True``, restrict to columns that are in all input catalogs.
            Else, if input catalogs have different columns, a :class:`ValueError` will be raised.

        Returns
        -------
        new : BaseCatalog

        Warning
        -------
        :attr:`attrs` of returned catalog contains, for each key, the last value found in ``others`` :attr:`attrs` dictionaries.
        """
        if not others:
            raise ValueError(f'Provide at least one {cls.__name__} instance.')
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        attrs = {}
        for other in others: attrs.update(other.attrs)
        others = [other for other in others if other.columns()]

        new = others[0].copy()
        new.attrs = attrs

        for other in others:
            if other.mpicomm is not new.mpicomm:
                raise ValueError('Input catalogs with different mpicomm')

        new_columns = new.columns()

        if intersection:
            for other in others:
                new_columns = [column for column in new_columns if column in other.columns()]
        else:
            for other in others:
                other_columns = other.columns()
                if new_columns and other_columns and set(other_columns) != set(new_columns):
                    raise ValueError(f'Cannot concatenate catalogs as columns do not match: {other_columns} != {new_columns}.')

        in_data = [column for column in new_columns if any(other.data[column] is not None for other in others)]
        if in_data:
            source = []
            for other in others:
                cumsizes = np.cumsum([0] + other.mpicomm.allgather(other.size))
                source.append(MPIScatteredSource(slice(cumsizes[other.mpicomm.rank], cumsizes[other.mpicomm.rank + 1], 1)))
            source = MPIScatteredSource.cconcatenate(*source)
            arrays = zip(*[other.get(in_data, return_type=None) for other in others])
            for column, arrays in zip(in_data, arrays):
                new.data[column] = source.get(arrays)

        source = [other._source for other in others if other.has_source]
        if source:
            source = FileStack.cconcatenate(*source)
            new._source = source

        return new

    def cappend(self, other, **kwargs):
        """(Collectively) append ``other`` to current catalog."""
        return self.cconcatenate(self, other, **kwargs)

    def to_dict(self, columns=None, return_type=None):
        """
        Return catalog as dictionary of column name: array.

        Parameters
        ----------
        columns : list, default=None
            Columns to use. Defaults to all catalog columns.

        return_type : str, default=None
            If ``None`` or "nparray", return dictionary of :class:`np.ndarray` instances.
            If "mpyarray", return dictionary of :class:`mpy.array` instances.

        Returns
        -------
        array : array
        """
        if columns is None:
            columns = self.columns()
        return dict(zip(columns, self.get(columns, return_type=return_type)))

    @classmethod
    @CurrentMPIComm.enable
    def from_dict(cls, data=None, columns=None, attrs=None, mpicomm=None):
        """
        Construct :class:`BaseCatalog` from dictionary.
        This is an internal method; :meth:`from_array` has more options.

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
        self = cls.__new__(cls)
        self.data = {}
        if isinstance(data, BaseCatalog):
            self.__dict__.update(data.copy().__dict__)
            return self
        if columns is None:
            columns = list((data or {}).keys())
        self.mpicomm = mpicomm
        self.attrs = dict(attrs or {})
        if data is not None:
            for name in columns:
                self[name] = data[name]
        return self

    @cast_array_wrapper
    def to_array(self, columns=None, struct=True):
        """
        Return catalog as numpy array.

        Parameters
        ----------
        columns : list, default=None
            Columns to use. Defaults to all catalog columns.

        struct : bool, default=True
            Whether to return structured array, with columns accessible through e.g. ``array['Position']``.
            If ``False``, numpy will attempt to cast types of different columns.

        return_type : str, default=None
            If ``None`` or "nparray", return :class:`np.ndarray` instance.
            If "mpyarray", return :class:`mpy.array` instance.

        Returns
        -------
        array : array
        """
        return _dict_to_array(self.to_dict(columns=columns, return_type=None), struct=struct)

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
        new = cls.from_dict(mpicomm=mpicomm, **kwargs)
        if mpiroot is not None:
            new.mpiroot = int(mpiroot)

        def get(column):
            value = None
            if mpicomm.rank == mpiroot or mpiroot is None:
                if isstruct:
                    value = np.asarray(array[column])
                else:
                    value = np.asarray(array[columns.index(column)])
            if mpiroot is not None:
                return mpy.scatter(value, mpicomm=mpicomm, mpiroot=mpiroot)
            return value

        new.data = {column: get(column) for column in columns}
        return new

    def copy(self):
        """Shallow copy."""
        new = super(BaseCatalog, self).__copy__()
        new.data = self.data.copy()
        if new.has_source: new._source = self._source.copy()
        import copy
        for name in self._attrs:
            if hasattr(self, name): setattr(new, name, copy.deepcopy(getattr(self, name)))
        return new

    def deepcopy(self):
        """Deep copy."""
        new = self.copy()
        new.data = {column: array.copy() if array is not None else None for column, array in self.data.items()}
        import copy
        for name in self._attrs:
            if hasattr(self, name): setattr(new, name, copy.deepcopy(getattr(self, name)))
        return new

    def items(self, **kwargs):
        return [(col, self[col]) for col in self.columns(**kwargs)]

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
        """Get catalog column ``name`` if string, new catalog instance if list of strings, else call :meth:`slice`."""
        if isinstance(name, str):
            return self.get(name)
        elif is_sequence(name) and all(isinstance(n, str) for n in name):
            new = self.copy()
            new.data = {n: self.data[n] for n in name}
            return new
        return self.slice(name)

    def __setitem__(self, name, item):
        """Set catalog column(s) ``name`` if string, else set local slice ``name`` of all columns to ``item``."""
        isscalar = isinstance(name, str)
        if isscalar:
            name = [name]
            item = [item]
        name_is_columns = is_sequence(name) and all(isinstance(n, str) for n in name)
        if isinstance(item, BaseCatalog):
            if name_is_columns:
                for n in name:
                    self[n] = item[n]
            else:
                for col, value in item.items():
                    self[col][name] = value
        elif name_is_columns:
            if not is_sequence(item):
                item = [item] * len(name)
            elif len(item) != len(name):
                raise ValueError('Provide as many values as columns')
            self.set(name, item)
        else:
            for col in self.columns():
                self[col][name] = item

    def __delitem__(self, name):
        """Delete column(s) ``name``."""
        if isinstance(name, str): name = [name]
        for n in name:
            try:
                del self.data[n]
                if self.has_source:
                    try:
                        del self._source[n]
                    except KeyError:
                        pass
            except KeyError as exc:
                raise KeyError(f'Column {n} not found') from exc

    def __repr__(self):
        """Return string representation of catalog, including global size and columns."""
        return f'{self.__class__.__name__}(csize={self.csize:d}, size={self.size:d}, columns={self.columns()})'

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and columns? (ignoring :attr:`attrs`)"""
        if not isinstance(other, self.__class__):
            return False
        self_columns = self.columns()
        other_columns = other.columns()
        if set(other_columns) != set(self_columns):
            return False
        assert self.mpicomm == other.mpicomm
        self, other = self.cslice(0, None), other.cslice(0, None)  # make sure we have the same size on each rank
        for self_value, other_value in zip(self.get(self_columns), other.get(self_columns)):
            if not all(self.mpicomm.allgather(np.all(self_value == other_value))):
                return False
        return True

    @classmethod
    def read(cls, *args, attrs=None, **kwargs):
        """
        Read catalog from (list of) input file names.
        Specify ``filetype`` if file extension is not recognised.
        See specific :class:`io.BaseFile` subclass (e.g. :class:`io.FitsFile`) for optional arguments.
        """
        attrs = dict(attrs or {})
        init_kwargs = {name: kwargs.pop(name) for name in getattr(cls, '_init_kwargs', []) if name in kwargs}
        source = FileStack(*args, **kwargs)
        new = cls.from_dict(attrs=attrs, mpicomm=source.mpicomm, **init_kwargs)
        new._source = source
        new._header = source.header
        for column in source.columns: new.data[column] = None
        return new

    def write(self, *args, header=None, columns=None, **kwargs):
        """
        Save catalog to (list of) output file names.
        Specify ``filetype`` if file extension is not recognised.
        See specific :class:`io.BaseFile` subclass (e.g. :class:`io.FitsFile`) for optional arguments.

        Parameters
        ----------
        columns : list, default=None
            Columns to write. Defaults to all columns.
        """
        source = FileStack(*args, mpicomm=self.mpicomm, **kwargs)
        source.write(self.to_dict(columns=columns, return_type='nparray'), header=header)

    @classmethod
    @CurrentMPIComm.enable
    def load(cls, filename, mpicomm=None):
        """
        Load catalog in *npy* binary format from disk.

        Warning
        -------
        All data will be gathered on a single process, which may cause out-of-memory errors.

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
            cls.log_info(f'Loading {filename}.')
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
            state['data'][name] = mpy.scatter(data[name] if mpicomm.rank == mpiroot else None, mpicomm=mpicomm, mpiroot=mpiroot)
        return cls.from_state(state, mpicomm=mpicomm)

    def save(self, filename, columns=None):
        """
        Save catalog to ``filename`` as *npy* file.

        Parameters
        ----------
        columns : list, default=None
            Columns to save. Defaults to all columns.

        Warning
        -------
        All data will be gathered on a single process, which may cause out-of-memory errors.
        """
        if self.mpicomm.rank == 0:
            self.log_info(f'Saving to {filename}.')
            utils.mkdir(os.path.dirname(filename))
        state = self.__getstate__()
        if columns is None:
            columns = self.columns()
        state['data'] = dict(zip(columns, self.cget(columns, mpiroot=0)))
        if self.mpicomm.rank == 0:
            np.save(filename, state, allow_pickle=True)

    def csort(self, orderby, size=None):
        """
        Return new catalog, sorted by ``orderby``.
        One can provide the desired local ``size``, or pass 'orderby_counts',
        in which case each rank will get a similar number of unique ``orderby`` values.

        Warning
        -------
        ``orderby`` must be integer.
        """
        import mpsort
        self[orderby]  # to check column exists
        if size is None:
            size = mpy.local_size(self.csize)
        elif isinstance(size, str) and size == 'orderby_counts':
            # Let's group particles by orderby, with ~ similar number of orderby values on each rank
            # Caution: this may produce memory unbalance between different processes
            # hence potential memory error, which may be avoided using some criterion to rebalance load at the cost of less efficiency
            unique, counts = np.unique(self[orderby], return_counts=True)
            # Proceed rank-by-rank to save memory
            for irank in range(1, self.mpicomm.size):
                unique_irank = mpy.sendrecv(unique, source=irank, dest=0, tag=0, mpicomm=self.mpicomm)
                counts_irank = mpy.sendrecv(counts, source=irank, dest=0, tag=0, mpicomm=self.mpicomm)
                if self.mpicomm.rank == 0:
                    unique, counts = np.concatenate([unique, unique_irank]), np.concatenate([counts, counts_irank])
                    unique, inverse = np.unique(unique, return_inverse=True)
                    counts = np.bincount(inverse, weights=counts).astype(int)
            # Compute catalog size that each rank must have after sorting
            sizes = None
            if self.mpicomm.rank == 0:
                norderby = [(irank * unique.size // self.mpicomm.size, (irank + 1) * unique.size // self.mpicomm.size) for irank in range(self.mpicomm.size)]
                sizes = [np.sum(counts[low:high], dtype='i8') for low, high in norderby]

            # Send the number particles that each rank must contain after sorting
            size = self.mpicomm.scatter(sizes, root=0)
            csize = self.mpicomm.allreduce(size)
            assert csize == self.csize, 'float in bincount messes up total counts, {:d} != {:d}'.format(csize, self.csize)
        array = self.to_array(struct=True)
        out = np.empty_like(array, shape=size)
        mpsort.sort(array, orderby=orderby, out=out)
        new = self.copy()
        new.data = BaseCatalog.from_array(out, mpicomm=self.mpicomm).data
        return new

    def all_to_all(self, counts=None):
        """
        All-to-all operation. If ``counts`` is ``None``, balance load.

        counts : array, default=None
            Size of catalog chunks to send to each rank.
            An array or list of size ``self.mpicomm.size``.
        """
        new = self.copy()
        new.data = BaseCatalog.from_array(mpy.all_to_all(self.to_array(struct=True), counts=counts, mpicomm=self.mpicomm), mpicomm=self.mpicomm).data
        return new

    def balance_across_rank(self):
        """
        Return new catalog with roughly similar size on each rank. mpicomm not expected to be None.
        Note: In specific case, can produce unbalanced catalog (typically, when size < mpicomm.size ..)
        """
        raise NotImplementedError('use all_to_all instead')
        self['index'] = np.arange(self.size) % self.mpicomm.size
        new = self.csort('index', size='orderby_counts')
        new.__delitem__('index')
        return new

class Catalog(BaseCatalog):

    """A simple catalog."""
