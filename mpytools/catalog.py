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
        return cast_array(func(self, *args, **kwargs), return_type=return_type, mpicomm=self.mpicomm)

    return wrapper


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
        if isinstance(data, BaseCatalog):
            self.__dict__.update(data.copy().__dict__)
            return
        if columns is None:
            columns = list((data or {}).keys())
        self.mpicomm = mpicomm
        self.attrs = dict(attrs or {})
        self.mpiroot = 0
        if data is not None:
            for name in columns:
                self[name] = data[name]

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
        return ArrayCatalog(source, **self.attrs)

    def __len__(self):
        """Return catalog (local) length (``0`` if no column)."""
        keys = list(self.data.keys())
        if not keys:
            if self.has_source is not None:
                return self._source.size
            return 0
        return len(self.get(keys[0], return_type=None))

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
        return self.source.header

    @property
    def source(self):
        if self.has_source:
            return self._source
        raise AttributeError('{} has no source, i.e. no file has been read'.format(self.__class__.__name__))

    @property
    def has_source(self):
        """Whether a "source" (typically :class:`FileStack` instance) is attached to current catalog."""
        return getattr(self, '_source', None) is not None

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
        toret = []
        for c, d in zip(column, default):
            if c in self.data:
                if self.data[c] is None:
                    self.data[c] = self.source.read(c)
                toret.append(self.data[c])
            elif has_default:
                toret.append(d)
            else:
                raise KeyError('Column {} does not exist'.format(c))
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

    def cget(self, *args, mpiroot=None, **kwargs):
        """
        Return on rank ``mpiroot`` catalog global column ``column`` if exists, else provided default.
        If ``mpiroot`` is ``None`` or ``Ellipsis`` result is broadcast on all processes.
        """
        if mpiroot is None: mpiroot = Ellipsis
        toret = self.get(*args, return_type='mpyarray', **kwargs)
        if is_sequence(toret):
            return [tt.gather(mpiroot=mpiroot) for tt in toret]
        return toret.gather(mpiroot=mpiroot)

    def slice(self, *args):
        """
        Slice catalog (locally), e.g.:
        >>> catalog.slice(0, 100, 1)  # catalog of local size :attr:`size` <= 100
        Same reference to :attr:`attrs`.
        """
        new = self.copy()
        name = Slice(*args, size=self.size).idx
        new.data = {column: self.get(column, return_type=None)[name] for column in self.data}
        if self.has_source:
            new._source = self._source.slice(name)
        return new

    def cslice(self, *args):
        """
        Slice catalog (collectively), e.g.:
        >>> catalog.cslice(0, 100, 1)  # catalog of collective size :attr:`csize`  <= 100
        Same reference to :attr:`attrs`.
        """
        new = self.copy()
        cumsizes = np.cumsum([0] + self.mpicomm.allgather(self.size))
        global_slice = Slice(*args, size=cumsizes[-1])
        local_slice = global_slice.split(self.mpicomm.size)[self.mpicomm.rank]
        source = MPIScatteredSource(slice(cumsizes[self.mpicomm.rank], cumsizes[self.mpicomm.rank + 1], 1))
        for column in self.columns():
            if self.data[column] is not None:
                new[column] = source.get(self.get(column, return_type=None), local_slice)
        if self.has_source:
            new._source = self._source.cslice(global_slice)
        return new

    @classmethod
    def concatenate(cls, *others):
        """
        Concatenate catalogs together, locally:
        no data is exchanged between processes, but order is not preserved,
        e.g. the first rank will receive the beginning of all input catalogs.

        Parameters
        ----------
        others : list
            List of :class:`BaseCatalog` instances.

        Returns
        -------
        new : BaseCatalog

        Warning
        -------
        :attr:`attrs` of returned catalog contains, for each key, the last value found in ``others`` :attr:`attrs` dictionaries.
        """
        if not others:
            raise ValueError('Provide at least one {} instance.'.format(cls.__name__))
        if len(others) == 1 and utils.is_sequence(others[0]):
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

        for column in new_columns:
            if in_data[column]:
                new[column] = np.concatenate([other.get(column, return_type=None) for other in others], axis=0)

        source = [other._source for other in others if other.has_source]
        if source:
            source = FileStack.concatenate(*source)
            new._source = source

        return new

    def append(self, other, **kwargs):
        """(Locally) append ``other`` to current catalog."""
        return self.concatenate(self, other, **kwargs)

    def extend(self, other, **kwargs):
        """(Locally) extend (in-place) current catalog with ``other``."""
        new = self.append(self, other, **kwargs)
        self.__dict__.update(new.__dict__)

    @classmethod
    def cconcatenate(cls, *others):
        """
        Concatenate catalogs together, preserving global order.

        Parameters
        ----------
        others : list
            List of :class:`BaseCatalog` instances.

        Returns
        -------
        new : BaseCatalog

        Warning
        -------
        :attr:`attrs` of returned catalog contains, for each key, the last value found in ``others`` :attr:`attrs` dictionaries.
        """
        if not others:
            raise ValueError('Provide at least one {} instance.'.format(cls.__name__))
        if len(others) == 1 and utils.is_sequence(others[0]):
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
                raise ValueError('Cannot concatenate catalogs as columns do not match: {} != {}.'.format(other_columns, new_columns))

        in_data = {column: any(column in other.data for other in others) for column in new_columns}
        if any(in_data.values()):
            source = []
            for other in others:
                cumsizes = np.cumsum([0] + other.mpicomm.allgather(other.size))
                source.append(MPIScatteredSource(slice(cumsizes[other.mpicomm.rank], cumsizes[other.mpicomm.rank + 1], 1)))
            source = MPIScatteredSource.cconcatenate(*source)

        for column in new_columns:
            if in_data[column]:
                new[column] = source.get([other.get(column, return_type=None) for other in others])

        source = [other._source for other in others if other.has_source]
        if source:
            source = FileStack.cconcatenate(*source)
            new._source = source

        return new

    def cappend(self, other, **kwargs):
        """(Collectively) append ``other`` to current catalog."""
        return self.cconcatenate(self, other, **kwargs)

    def cextend(self, other, **kwargs):
        """(Collectively) extend (in-place) current catalog with ``other``."""
        new = self.cappend(self, other, **kwargs)
        self.__dict__.update(new.__dict__)

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
        if columns is None:
            columns = self.columns()
        data = {column: self.get(column, return_type=None) for column in columns}
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
        new.data = {column: self.get(column, return_type='nparray').copy() for column in new.data}
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
        if name_is_columns:
            if not is_sequence(item):
                item = [item] * len(name)
            elif len(item) != len(name):
                raise ValueError('Provide as many values as columns')
        if isinstance(item, BaseCatalog):
            if name_is_columns:
                for n in name:
                    self[n] = item[n]
            else:
                for col, value in item.items():
                    self[col][name] = value
        elif name_is_columns:
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
                raise KeyError('Column {} not found'.format(n)) from exc

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
        self, other = self.cslice(0, None), other.cslice(0, None)  # make sure we have the same size on each rank
        for col in self_columns:
            self_value = self.get(col)
            other_value = other.get(col)
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
        new = cls(attrs=attrs, mpicomm=source.mpicomm, **init_kwargs)
        new._source = source
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
        source = FileStack(*args, **kwargs)
        if columns is None:
            columns = self.columns()
        source.write({column: self.get(column, return_type='nparray') for column in columns}, header=header)

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
        if self.is_mpi_root():
            self.log_info('Saving to {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
        state = self.__getstate__()
        if columns is None:
            columns = self.columns()
        state['data'] = {name: self.cget(name, mpiroot=self.mpiroot) for name in columns}
        if self.is_mpi_root():
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
        new.data = self.__class__.from_array(out, mpicomm=self.mpicomm).data
        return new


class Catalog(BaseCatalog):

    """A simple catalog."""
