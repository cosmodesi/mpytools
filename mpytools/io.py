"""Base classes to handle catalog of objects."""

import os
import re
import shutil

import numpy as np

from . import utils
from .utils import BaseClass, is_sequence, CurrentMPIComm
from . import core as mpy
from .core import Slice, MPIScatteredSource


def select_columns(columns, include=None, exclude=None):
    """
    Select input column names.

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
        Column names, after optional selections.
    """
    def toregex(name):
        return name.replace('.', r'\.').replace('*', '(.*)')

    if not is_sequence(columns):
        columns = [columns]

    toret = columns

    if include is not None:
        if not is_sequence(include):
            include = [include]
        toret = []
        for column in columns:
            if any(re.match(toregex(inc), column) for inc in include):
                toret.append(column)
        columns = toret

    if exclude is not None:
        if not is_sequence(exclude):
            exclude = [exclude]
        toret = []
        for column in columns:
            if not any(re.match(toregex(exc), column) for exc in exclude):
                toret.append(column)

    return toret


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


class FileStack(BaseClass):

    @CurrentMPIComm.enable
    def __init__(self, *files, filetype=None, mpicomm=None, **kwargs):
        """
        Initialize :class:`FileStack`.

        Parameters
        ----------
        files : string, list of strings
            File name(s), or :class:`BaseFile` instances.

        filetype : string, type, default=None
            :class:`BaseFile` to use to read input files.
            See :func:`get_filetype`.

        mpicomm : MPI communicator, default=None
            The current MPI communicator.

        kwargs : dict
            Optional arguments for :class:`BaseFile` instances.
        """
        self.files = []
        self.mpicomm = mpicomm
        for file in utils.list_concatenate(files):
            if isinstance(file, BaseFile):
                self.files.append(file)
            else:
                FT = get_filetype(filetype=filetype, filename=file)
                self.files.append(FT(file, mpicomm=self.mpicomm, **kwargs))
        for file in self.files:
            if file.mpicomm is not self.mpicomm:
                raise ValueError('Input files with different mpicomm')
        self.mpiroot = 0

    def __enter__(self):
        """
        To use :class:`FileStack` as a context manager,
        >>> with FileStack(filenames) as fs:
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def is_mpi_root(self):
        """Whether current rank is root."""
        return self.mpicomm.rank == self.mpiroot

    @property
    def filesizes(self):
        """List of file (total) sizes."""
        return [file.csize for file in self.files]

    @property
    def cfilesize(self):
        """Sum of file sizes."""
        return sum(self.filesizes)

    @property
    def slices(self):
        """Local slice (within rows 0 to :attr:`cfilesize`)."""
        if getattr(self, '_slices', None) is None:
            self._slices = [Slice(self.mpicomm.rank * self.cfilesize // self.mpicomm.size, (self.mpicomm.rank + 1) * self.cfilesize // self.mpicomm.size, 1)]
        return self._slices

    @property
    def columns(self):
        """Columns shared by all files."""
        if getattr(self, '_columns', None) is None:
            if not self.files:
                return []
            self._columns = [column for column in self.files[0].columns if all(column in file.columns for file in self.files[1:])]
        return self._columns

    def __delitem__(self, name):
        self.columns.remove(name)

    @property
    def header(self):
        """Full header, concatenating all file headers."""
        if getattr(self, '_header', None) is None:
            self._header = {}
            for file in self.files:
                self._header.update(file.header)
        return self._header

    def fileslices(self, return_index=False):
        """
        For each local slice, yield slices in each file.
        Pass ``return_index = True`` to also yield indices in current local slice corresponding to each file slice.
        """
        # catalog slices
        cumsizes = np.cumsum([0] + self.filesizes)
        for sli in self.slices:
            #for start, stop in zip(cumsizes[:-1], cumsizes[1:]):
            #     print(self.mpicomm.rank, sli, start, stop, Slice(start, stop, 1).find(sli, return_index=return_index))
            yield [Slice(start, stop, 1).find(sli, return_index=return_index) for start, stop in zip(cumsizes[:-1], cumsizes[1:])]

    @property
    def size(self):
        """Local size of columns to be read."""
        return sum(sl.size for sl in self.slices)

    @property
    def csize(self):
        """Collective size of columns to be read."""
        return self.mpicomm.allreduce(self.size)

    @property
    def _is_slice_array(self):
        # Is any of slices an index array?
        return any(utils.list_concatenate(self.mpicomm.allgather([sl.is_array for sl in self.slices])))

    def slice(self, *args):
        """Slice :class:`FileStack` (locally), i.e. select local rows to read."""
        new = self.copy()
        sl = Slice(*args)
        new._slices = [sli.slice(sl) for sli in self.slices]
        return new

    def cslice(self, *args):
        """Slice :class:`FileStack` (collectively), i.e. select global rows to read."""
        new = self.copy()
        global_slice = Slice(*args, size=self.csize)
        local_slice = global_slice.split(self.mpicomm.size)[self.mpicomm.rank]
        #print(self.mpicomm.rank, local_slice)
        if local_slice.is_array or self._is_slice_array:
            cumsizes = np.cumsum([sum(self.mpicomm.allgather(self.size)[:self.mpicomm.rank])] + [sl.size for sl in self.slices])
            slices = [slice(size1, size2, 1) for size1, size2 in zip(cumsizes[:-1], cumsizes[1:])]
            source = MPIScatteredSource(*slices)
            new._slices = [Slice(source.get([sl.to_array() for sl in self._slices], local_slice))]
        else:
            all_slices = utils.list_concatenate(self.mpicomm.allgather(self.slices))
            tmp = []
            cumsize = 0
            for sli in all_slices:
                #self_slice_in_irank = sli.slice(local_slice.shift(-cumsize), return_index=False)
                #print(self.mpicomm.rank, local_slice, local_slice.shift(-cumsize), sli, sli.size, self_slice_in_irank, '\n')
                self_slice_in_irank = sli.slice(Slice(cumsize, cumsize + sli.size).find(local_slice), return_index=False)
                #print(self.mpicomm.rank, local_slice, sli, sli.size, self_slice_in_irank, '\n')
                if self_slice_in_irank: tmp.append(self_slice_in_irank)
                cumsize += sli.size
            if local_slice.idx.step < 0:
                tmp = tmp[::-1]
            new._slices = Slice.snap(*tmp)
            #print(self.mpicomm.rank, new._slices)
        return new

    @classmethod
    def concatenate(cls, *others):
        """Concatenate :class:`FileStack` locally, e.g. the first rank will get the beginning of all files."""
        new = cls(*utils.list_concatenate([other.files for other in others]))
        if any(getattr(other, '_slices', None) is not None for other in others):
            slices, cumsize = [], 0
            for other in others:
                slices += [sl.shift(cumsize) for sl in other.slices]
                cumsize += other.cfilesize
            new._slices = slices
        return new

    def append(self, other, **kwargs):
        """(Locally) append ``other`` to current :class:`FileStack`."""
        return self.concatenate(self, other, **kwargs)

    def extend(self, other, **kwargs):
        """(Locally) extend (in-place) current :class:`FileStack` with ``other``."""
        new = self.append(self, other, **kwargs)
        self.__dict__.update(new.__dict__)

    @classmethod
    def cconcatenate(cls, *others):
        """Concatenate :class:`FileStack` collectively, such that global order is preserved."""
        new = cls(*utils.list_concatenate([other.files for other in others]))
        if any(getattr(other, '_slices', None) is not None for other in others):
            if any(other._is_slice_array for other in others):
                csize = sum(other.csize for other in others)
                new_slice = Slice(new.mpicomm.rank * csize // new.mpicomm.size, (new.mpicomm.rank + 1) * csize // new.mpicomm.size, 1)
                source = []
                for other in others:
                    cumsizes = np.cumsum([sum(new.mpicomm.allgather(other.size)[:new.mpicomm.rank])] + [sl.size for sl in other.slices])
                    slices = [slice(size1, size2, 1) for size1, size2 in zip(cumsizes[:-1], cumsizes[1:])]
                    source.append(MPIScatteredSource(*slices))
                source = MPIScatteredSource.cconcatenate(*source)
                new._slices = [Slice(source.get(utils.list_concatenate([[sl.to_array() for sl in other._slices] for other in others]), new_slice))]
            else:
                slices, cumsize = [], 0
                for other in others:
                    slices += utils.list_concatenate(new.mpicomm.allgather([sl.shift(cumsize) for sl in other.slices]))
                    cumsize += other.cfilesize
                new._slices = slices if new.mpicomm.rank == 0 else []
                new = new.cslice(0, cumsize, 1)  # to balance load
        return new

    def cappend(self, other, **kwargs):
        """(Collectively) append ``other`` to current :class:`FileStack`."""
        return self.cconcatenate(self, other, **kwargs)

    def cextend(self, other, **kwargs):
        """(Collectively) extend (in-place) current class:`FileStack` with ``other``."""
        new = self.cappend(self, other, **kwargs)
        self.__dict__.update(new.__dict__)

    def read(self, column):
        """Read column of name ``column``."""
        toret = []
        for islice, slices in enumerate(self.fileslices(return_index=True)):
            tmp, idx = [], []
            for ifile, (rows, iidx) in enumerate(slices):
                tmp.append(self.files[ifile].read(column, rows=rows))
                idx.append(iidx.idx)
            if self.slices[islice].is_array:
                toret.append(np.concatenate(tmp, axis=0, dtype=tmp[0].dtype)[np.argsort(np.concatenate(idx, axis=0))])
            else:
                toret += [tmp[ii] for ii in np.argsort([iidx.start for iidx in idx])]
        tmp = np.concatenate(toret, axis=0, dtype=toret[0].dtype)
        return tmp

    def write(self, data, mpiroot=None, **kwargs):
        """Write data (numpy structured array or dict), optionally gathered on ``mpiroot``."""
        isdict = None
        if self.mpicomm.rank == mpiroot or mpiroot is None:
            isdict = isinstance(data, dict)
        if mpiroot is not None:
            isdict = self.mpicomm.bcast(isdict, root=mpiroot)
            if isdict:
                columns = self.mpicomm.bcast(list(data.keys()) if self.mpicomm.rank == mpiroot else None, root=mpiroot)
                data = {name: mpy.scatter(data[name] if self.mpicomm.rank == mpiroot else None, mpicomm=self.mpicomm, mpiroot=self.mpiroot) for name in columns}
            else:
                data = mpy.scatter(data, mpicomm=self.mpicomm, mpiroot=self.mpiroot)
        if isdict:
            for name in data: size = len(data[name]); break
        else:
            size = len(data)

        csize = self.mpicomm.allreduce(size)
        nfiles = len(self.files)
        for ifile, file in enumerate(self.files):
            file._csize = (ifile + 1) * csize // nfiles - ifile * csize // nfiles
        fcumsizes = np.cumsum([0] + self.filesizes)
        cumsizes = np.cumsum([0] + self.mpicomm.allgather(size))
        self._slices = [Slice(cumsizes[self.mpicomm.rank], cumsizes[self.mpicomm.rank] + size, 1)]
        for islice, slices in enumerate(self.fileslices()):
            for ifile, rows in enumerate(slices):
                rows = rows.shift(fcumsizes[ifile] - cumsizes[self.mpicomm.rank])
                if isdict:
                    self.files[ifile].write({name: data[name][rows.idx] for name in data}, **kwargs)
                else:
                    self.files[ifile].write(data[rows.idx], **kwargs)


def get_filetype(filetype=None, filename=None):
    """
    Return :class:`BaseFile` type.

    Parameters
    ----------
    filetype : string, type, default=None
        File type name (e.g. "fits", "hdf5", "binary", "bigfile", "asdf") or :class:`BaseFile` type.
        If ``None``, guess file type from ``filename``.

    filename : string
        File name; guess file type from its extension.

    Returns
    -------
    filetype : :class:`BaseFile` type
    """
    if filetype is None:
        if filename is not None:
            ext = os.path.splitext(filename)[1][1:]
            for filetype in RegisteredFile._registry.values():
                if ext in filetype.extensions:
                    return filetype
            raise IOError('Extension {} is unknown'.format(ext))
    if isinstance(filetype, str):
        filetype = RegisteredFile._registry[filetype.lower()]

    return filetype


class RegisteredFile(type(BaseClass)):

    """Metaclass registering :class:`BaseFile`-derived classes."""

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry[cls.name] = cls
        return cls


class BaseFile(BaseClass, metaclass=RegisteredFile):
    """
    Base class to read/write a file from/to disk.
    File handlers should extend this class, by (at least) implementing :meth:`_read_header_root`, :meth:`_read_rows` and :meth:`_write_data`.
    """
    name = 'base'
    extensions = []
    _type_read_rows = ['slice', 'sslice', 'islice', 'index']
    _type_write_data = ['dict', 'array']
    _read_nslices_max = 10

    @CurrentMPIComm.enable
    def __init__(self, filename, mpicomm=None):
        """
        Initialize :class:`BaseFile`.

        Parameters
        ----------
        filename : string
            File name.

        mpicomm : MPI communicator, default=None
            The current MPI communicator.
        """
        self.filename = filename
        self.mpicomm = mpicomm
        self.mpiroot = 0

    def __enter__(self):
        """To use :class:`BaseFile` as a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def is_mpi_root(self):
        """Whether current rank is root."""
        return self.mpicomm.rank == self.mpiroot

    def _read_header(self):
        # Read file header, the end user does not need to call it
        state = None
        if self.is_mpi_root():
            self.log_info('Reading {}.'.format(self.filename))
            state = self._read_header_root()
            state['_csize'] = int(state.pop('csize'))
            state['_columns'] = list(state.pop('columns'))
            state['_header'] = dict(state.pop('header', {}))
        state = self.mpicomm.bcast(state, root=self.mpiroot)
        self.__dict__.update(state)
        return state['_header']

    @property
    def csize(self):
        # File size
        if getattr(self, '_csize', None) is None:
            self._read_header()
        return self._csize

    @property
    def columns(self):
        # File columns (list of column names)
        if getattr(self, '_columns', None) is None:
            self._read_header()
        return self._columns

    @property
    def header(self):
        # File header (dict)
        if getattr(self, '_header', None) is None:
            self._read_header()
        return self._header

    def read(self, column, rows=slice(None)):
        """Read input ``rows`` of column name ``column``."""
        sl = Slice(rows, size=self.csize)
        rows, idxs = [sl.idx], None
        if sl.is_array:
            if 'index' not in self._type_read_rows:
                nslices = sl.nslices()
                if nslices > self._read_nslices_max:  # too many slices, let's just read (min, max + 1, 1) and mask afterwards
                    rows = [slice(sl.min, sl.max + 1, 1) if sl else slice(0, 0, 1)]
                    idxs = [Slice(rows[0]).find(sl).idx]
                else:
                    rows = sl.to_slices()
        else:
            if 'slice' not in self._type_read_rows:
                rows = [sl.to_array()]
        toret = []
        for irow, row in enumerate(rows):
            idx = None
            if idxs is not None: idx = idxs[irow]
            if isinstance(row, slice):
                start, stop, step = row.start, row.stop, row.step
                if step < 0:
                    if stop is None: stop = -1
                    start, stop = stop + 1, start + 1
                if row.step != 1 and 'sslice' not in self._type_read_rows:
                    row = slice(start, stop, 1)
                    idx = slice(None, None, step)
                elif row.step < 0 and 'islice' not in self._type_read_rows:
                    row = slice(start, stop, abs(step))
                    idx = slice(None, None, -1)
            tmp = self._read_rows(column, rows=row)
            if idx is not None: tmp = tmp[idx]
            toret.append(tmp)
        return np.concatenate(toret, axis=0, dtype=toret[0].dtype)

    def write(self, data, header=None, **kwargs):
        """
        Write input data to :attr:`filename`.

        Parameters
        ----------
        data : array, dict
            Data to write.

        header : dict, default=None
            Optional header.
        """
        if self.is_mpi_root():
            self.log_info('Writing {}.'.format(self.filename))
        utils.mkdir(os.path.dirname(self.filename))
        isdict = isinstance(data, dict)
        if isdict:
            if 'dict' not in self._type_write_data:
                data = _dict_to_array(data)
        else:
            data = np.asarray(data)
            if 'array' not in self._type_write_data:
                data = {name: data[name] for name in data.dtype.names}
        self._write_data(data, header=header or {}, **kwargs)

    def _read_header_root(self):
        # Must return a dictionary with (at least)
        # csize: file size
        # columns: list of column names
        # (optionally) header : dictionary
        # (optionally) other attributes
        raise NotImplementedError

    def _read_rows(self, column, rows):
        raise NotImplementedError

    def _write_data(self, data, header):
        raise NotImplementedError


try: import fitsio
except ImportError: fitsio = None


class FitsFile(BaseFile):
    """
    Class to read/write a Fits file from/to disk.

    Warning
    -------
    Writing a FITS file requires all data to be gathered on a single rank, which may cause out-of-memory errors.

    Note
    ----
    In some circumstances (e.g. catalog has just been written), :meth:`read` fails with a file not found error.
    We have tried making sure processes read the file one after the other, but that does not solve the issue.
    A similar issue happens with nbodykit - though at a lower frequency.
    """
    name = 'fits'
    extensions = ['fits']
    _type_read_rows = ['slice']
    _type_write_data = ['array']

    def __init__(self, filename, ext=None, **kwargs):
        """
        Initialize :class:`FitsFile`.

        Parameters
        ----------
        filename : string
            File name.

        ext : int, default=None
            FITS extension. Defaults to first extension with data.

        kwargs : dict
            Arguments for :class:`BaseFile` (mpicomm).
        """
        if fitsio is None:
            raise ImportError('Install fitsio')
        self.ext = ext
        super(FitsFile, self).__init__(filename=filename, **kwargs)

    def _read_header_root(self):
        # Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/io/fits.py
        with fitsio.FITS(self.filename) as file:
            if getattr(self, 'ext') is None:
                for i, hdu in enumerate(file):
                    if hdu.has_data():
                        self.ext = i
                        break
                if self.ext is None:
                    raise IOError('{} has no binary table to read'.format(self.filename))
            else:
                if isinstance(self.ext, str):
                    if self.ext not in file:
                        raise IOError('{} does not contain extension with name {}'.format(self.filename, self.ext))
                elif self.ext >= len(file):
                    raise IOError('{} extension {} is not valid'.format(self.filename, self.ext))
            file = file[self.ext]
            # make sure we crash if data is wrong or missing
            if not file.has_data() or file.get_exttype() == 'IMAGE_HDU':
                raise IOError('{} extension {} is not a readable binary table'.format(self.filename, self.ext))
            return {'csize': file.get_nrows(), 'columns': file.get_rec_dtype()[0].names, 'header': dict(file.read_header()), 'ext': self.ext}

    def _read_rows(self, column, rows):
        return fitsio.read(self.filename, ext=self.ext, columns=column, rows=range(rows.start, rows.stop))

    def _write_data(self, data, header):
        data = mpy.gather(data, mpicomm=self.mpicomm, mpiroot=self.mpiroot)
        if self.is_mpi_root():
            fitsio.write(self.filename, data, header=header, clobber=True)


try: import h5py
except ImportError: h5py = None


class HDF5File(BaseFile):
    """
    Class to read/write a HDF5 file from/to disk.

    Warning
    -------
    If hdf5 is not built with MPI support (see https://docs.h5py.org/en/stable/mpi.html)
    writing a HDF5 file requires each column to be gathered on a single rank,
    which may cause out-of-memory errors.

    Note
    ----
    In some circumstances (e.g. catalog has just been written), :meth:`get` fails with a file not found error.
    We have tried making sure processes read the file one after the other, but that does not solve the issue.
    A similar issue happens with nbodykit - though at a lower frequency.
    """
    name = 'hdf5'
    extensions = ['hdf', 'h4', 'hdf4', 'he2', 'h5', 'hdf5', 'he5', 'h5py']
    _type_read_rows = ['slice', 'sslice', 'index']
    _type_write_data = ['dict']

    def __init__(self, filename, group='/', **kwargs):
        """
        Initialize :class:`HDF5File`.

        Parameters
        ----------
        filename : string
            File name.

        group : string, default='/'
            HDF5 group where columns are located.

        kwargs : dict
            Arguments for :class:`BaseFile` (mpicomm).
        """
        if h5py is None:
            raise ImportError('Install h5py')
        self.group = group
        if not group or group == '/' * len(group):
            self.group = '/'
        super(HDF5File, self).__init__(filename=filename, **kwargs)

    def _read_header_root(self):
        with h5py.File(self.filename, 'r') as file:
            grp = file[self.group]
            columns = list(grp.keys())
            size = grp[columns[0]].shape[0]
            for name in columns:
                if grp[name].shape[0] != size:
                    raise IOError('Column {} has different length (expected {:d}, found {:d})'.format(name, size, grp[name].shape[0]))
            return {'csize': size, 'columns': columns, 'header': dict(grp.attrs)}

    def _read_rows(self, column, rows):
        with h5py.File(self.filename, 'r') as file:
            grp = file[self.group]
            if isinstance(rows, slice):
                return grp[column][rows]
            rows, inverse = np.unique(rows, return_inverse=True)
            return grp[column][rows][inverse]

    def _write_data(self, data, header):
        driver = 'mpio'
        kwargs = {'comm': self.mpicomm}
        import h5py
        try:
            h5py.File(self.filename, 'w', driver=driver, **kwargs)
        except ValueError:
            driver = None
            kwargs = {}
        if driver == 'mpio':
            for name in data: size = len(data[name]); break
            with h5py.File(self.filename, 'w', driver=driver, **kwargs) as file:
                cumsizes = np.cumsum([0] + self.mpicomm.allgather(size))
                start, stop = cumsizes[self.mpicomm.rank], cumsizes[self.mpicomm.rank + 1]
                csize = cumsizes[-1]
                grp = file
                if self.group != '/':
                    grp = file.create_group(self.group)
                grp.attrs.update(self.attrs)
                for name in data:
                    dset = grp.create_dataset(name, shape=(csize,) + data[name].shape[1:], dtype=data[name].dtype)
                    dset[start:stop] = data[name]
        else:
            if self.is_mpi_root():
                h5py.File(self.filename, 'w', driver=driver, **kwargs)
            first = True
            for name in data:
                array = mpy.gather(data[name], mpicomm=self.mpicomm, mpiroot=self.mpiroot)
                if self.is_mpi_root():
                    with h5py.File(self.filename, 'a', driver=driver, **kwargs) as file:
                        grp = file
                        if first:
                            if self.group != '/':
                                grp = file.create_group(self.group)
                            grp.attrs.update(header)
                        dset = grp.create_dataset(name, data=array)
                first = False


from numpy.lib.format import open_memmap


class BinaryFile(BaseFile):

    """Class to read/write a binary file from/to disk."""

    name = 'bin'
    extensions = ['npy']
    _type_read_rows = ['slice']
    _type_write_data = ['array']

    def _read_header_root(self):
        array = open_memmap(self.filename, mode='r')
        return {'csize': len(array), 'columns': array.dtype.names, 'header': {}}

    def _read_rows(self, column, rows):
        return open_memmap(self.filename, mode='r')[rows][column]

    def _write_data(self, data, header):
        cumsizes = np.cumsum([0] + self.mpicomm.allgather(len(data)))
        if self.is_mpi_root():
            fp = open_memmap(self.filename, mode='w+', dtype=data.dtype, shape=(cumsizes[-1],))
        self.mpicomm.Barrier()
        start, stop = cumsizes[self.mpicomm.rank], cumsizes[self.mpicomm.rank + 1]
        fp = open_memmap(self.filename, mode='r+')
        fp[start:stop] = data
        fp.flush()


import json
try: import bigfile
except ImportError: bigfile = None


class BigFile(BaseFile):
    """Class to read/write a BigFile from/to disk."""
    name = 'bigfile'
    extensions = ['bigfile']
    _type_read_rows = ['slice']
    _type_write_data = ['dict']

    def __init__(self, filename, group='/', header=None, exclude=None, overwrite=False, **kwargs):
        """
        Initialize :class:`BigFile`.

        Parameters
        ----------
        filename : string
            File name.

        group : string, default='/'
            :class:`BigFile` group where columns are located.

        header : string, list, default=None
            Name of headers.

        exclude : string, list
            List of datasets to exclude (when reading file size).
            Can contain *regex* or Unix-style patterns.

        overwrite : bool, default=False
            By default, :meth:`write` adds columns to the file/group.
            One can pass ``overwrite = True`` to remove previous file/group directory
            (at one's own risk) and write a new file.

        kwargs : dict
            Arguments for :class:`BaseFile` (mpicomm).
        """
        if bigfile is None:
            raise ImportError('Install bigfile')
        self.group = group
        if not group or group == '/' * len(group):
            self.group = '/'
        if not self.group.endswith('/'): self.group = self.group + '/'
        self.header_blocks = header
        self.exclude = exclude
        self.overwrite = bool(overwrite)
        super(BigFile, self).__init__(filename=filename, **kwargs)

    def _read_header_root(self):
        with bigfile.File(filename=self.filename) as file:
            columns = [block for block in file[self.group].blocks]
            header_blocks = self.header_blocks
            if header_blocks is None: header_blocks = ['Header', 'header', '.']
            if not isinstance(header_blocks, (tuple, list)): header_blocks = [header_blocks]
            headers = []
            for header in header_blocks:
                if header in file and header not in headers: headers.append(header)
            # Append the dataset itself
            headers.append(self.group.strip('/') + '/.')

            exclude = self.exclude
            if exclude is None:
                # By default exclude header only.
                exclude = list(headers)

            columns = select_columns(columns, exclude=exclude)
            csize = bigfile.Dataset(file[self.group], columns).size

            attrs = {}
            for header in headers:
                # Copy over the attrs
                fattrs = file[header].attrs
                for key in fattrs:
                    value = fattrs[key]
                    # Load a JSON representation if str starts with json:://
                    if isinstance(value, str) and value.startswith('json://'):
                        attrs[key] = json.loads(value[7:])  # , cls=JSONDecoder)
                    # Copy over an array
                    else:
                        attrs[key] = np.array(value, copy=True)
            return {'csize': csize, 'columns': columns, 'header': attrs}

    def _read_rows(self, column, rows):
        with bigfile.File(filename=self.filename)[self.group] as file:
            return bigfile.Dataset(file, [column])[column][rows]

    def _write_data(self, data, header, overwrite=None):
        columns = list(data.keys())
        if overwrite is None:
            overwrite = self.overwrite

        # FIXME: merge this logic into bigfile
        # the slice writing support in bigfile 0.1.47 does not
        # support tuple indices.
        class _ColumnWrapper:

            def __init__(self, bb):
                self.bb = bb

            def __setitem__(self, sl, value):
                assert len(sl) <= 2  # no array shall be of higher dimension.
                # use regions argument to pick the offset.
                start, stop, step = sl[0].indices(self.bb.size)
                assert step == 1
                if len(sl) > 1:
                    start1, stop1, step1 = sl[1].indices(value.shape[1])
                    assert step1 == 1
                    assert start1 == 0
                    assert stop1 == value.shape[1]
                self.bb.write(start, value)

        if overwrite:
            self.mpicomm.Barrier()
            if self.mpicomm.rank == 0:
                try:
                    shutil.rmtree(os.path.join(self.filename, self.group.strip('/')), ignore_errors=True)  # otherwise previous columns are kept
                except FileNotFoundError:
                    pass
            self.mpicomm.Barrier()

        with bigfile.FileMPI(comm=self.mpicomm, filename=self.filename, create=True) as file:

            sources, targets, regions = [], [], []

            # save meta data and create blocks, prepare for the write.
            for column in columns:
                array = data[column]
                column = self.group + column
                # ensure data is only chunked in the first dimension
                size = self.mpicomm.allreduce(len(array))
                offset = sum(self.mpicomm.allgather(len(array))[:self.mpicomm.rank])

                # sane value -- 32 million items per physical file
                size_per_file = 32 * 1024 * 1024

                nfiles = (size + size_per_file - 1) // size_per_file

                dtype = np.dtype((array.dtype, array.shape[1:]))

                # save column attrs too
                # first create the block on disk
                with file.create(column, dtype, size, nfiles) as bb:
                    pass

                # first then open it for writing
                bb = file.open(column)

                targets.append(_ColumnWrapper(bb))
                sources.append(array)
                regions.append((slice(offset, offset + len(array)),))

            # writer header afterwards, such that header can be a block that saves data.
            if header:
                try:
                    bb = file.open('header')
                except:
                    bb = file.create('header')
                with bb:
                    for key in header:
                        try:
                            bb.attrs[key] = header[key]
                        except ValueError:
                            try:
                                json_str = 'json://' + json.dumps(header[key])
                                bb.attrs[key] = json_str
                            except:
                                raise ValueError('Cannot save {} key in attrs dictionary'.format(key))

            # lock=False to avoid dask from pickling the lock with the object.
            # write blocks one by one
            for column, source, target, region in zip(columns, sources, targets, regions):
                if self.is_mpi_root():
                    self.log_debug('Started writing column {}'.format(column))
                target[region] = source
                target.bb.close()
                if self.is_mpi_root():
                    self.log_debug('Finished writing column {}'.format(column))


try: import asdf
except ImportError: asdf = None


class AsdfFile(BaseFile):
    """
    Class to read/write an ASDF file from/to disk.

    Warning
    -------
    Writing an ASDF file requires all data to be gathered on a single rank, which may cause out-of-memory errors.
    """
    name = 'asdf'
    extensions = ['asdf']
    _type_read_rows = ['slice', 'sslice']
    _type_write_data = ['dict']

    def __init__(self, filename, group='', header=None, exclude=None, **kwargs):
        """
        Initialize :class:`AsdfFile`.

        Parameters
        ----------
        filename : string
            File name.

        group : string, default='/'
            Asdf group where columns are located.

        header : string, list, default=None
            Name of headers.

        exclude : string, list
            List of datasets to exclude (when reading file size).
            Can contain *regex* or Unix-style patterns.

        kwargs : dict
            Arguments for :class:`BaseFile` (mpicomm).
        """
        if asdf is None:
            raise ImportError('Install asdf')
        self.group = group
        self.header_blocks = header
        self.exclude = exclude
        super(AsdfFile, self).__init__(filename=filename, **kwargs)

    def _read_header_root(self):
        with asdf.open(self.filename) as file:
            file = file[self.group] if self.group else file
            columns = list(file.keys())
            header_blocks = self.header_blocks
            if header_blocks is None: header_blocks = ['Header', 'header']
            if not isinstance(header_blocks, (tuple, list)): header_blocks = [header_blocks]
            headers = []
            for header in header_blocks:
                if header in columns and header not in headers: headers.append(header)
            exclude = self.exclude
            if exclude is None:
                # By default exclude header
                exclude = headers
            exclude = exclude + ['asdf_library', 'history']  # to copy list

            columns = select_columns(columns, exclude=exclude)
            csize = len(file[columns[0]])

            attrs = {}
            for header in headers:
                # copy over the attrs
                fattrs = file[header]
                for key in fattrs:
                    value = fattrs[key]
                    # load a JSON representation if str starts with json:://
                    if isinstance(value, str) and value.startswith('json://'):
                        attrs[key] = json.loads(value[7:])  # , cls=JSONDecoder)
                    # copy over an array
                    else:
                        attrs[key] = np.array(value, copy=True)
            return {'csize': csize, 'columns': columns, 'header': attrs}

    def _read_rows(self, column, rows):
        with asdf.open(self.filename) as file:
            file = file[self.group] if self.group else file
            return np.array(file[column][rows], copy=True)  # otherwise, segfault due to memorymap

    def _write_data(self, data, header):
        data = {key: mpy.gather(data[key], mpicomm=self.mpicomm, mpiroot=self.mpiroot) for key in data}
        if self.is_mpi_root():
            af = asdf.AsdfFile(data)
            # Write the data to a new file
            af.write_to(self.filename)
