"""Implements MPI-scattered array."""

import numbers
import math

import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.mixins import NDArrayOperatorsMixin as NDArrayLike

from . import mpi, utils
from .mpi import MPI, CurrentMPIComm
from .utils import BaseClass, BaseMetaClass, is_sequence


class Slice(BaseClass):

    """Class that handles slices, either as python slice, or indices (numpy arrays)."""

    def __init__(self, *args, size=None, copy=False):
        """
        Initialize :class:`Slice`, e.g.:
        >>> Slice(10)  # slice(0, 10, 1)
        >>> Slice(10, None, -2)
        >>> Slice(None, size=10)  # slice(0, 10, 1)
        >>> Slice([1, 2, 2, 4])

        Parameters
        ----------
        args : tuple
            Arguments for python slice, or:
            - slice
            - Slice
            - list / array

        size : int, default=None
            In case one provides a slice (or arguments for a slice), the total size of the array to be sliced, e.g.:
            ``Slice(None, size=10)`` is ``Slice(0, 10, 1)``.

        copy : bool, default=False
            In case an array is provided, whether to copy input array.
        """
        if len(args) > 1 or args[0] is None or isinstance(args[0], numbers.Number):
            sl = slice(*args)
        else:
            sl = args[0]
        if isinstance(sl, self.__class__):
            self.__dict__.update(sl.__dict__)
            if self.is_array: self.idx = np.array(self.idx, copy=copy)
            return
        elif isinstance(sl, slice):
            if size is not None:
                start, stop, step = sl.indices(size)
            else:
                start, stop, step = sl.start, sl.stop, sl.step
            if step is None: step = 1
            if step == 0: raise ValueError('Zero step')
            if start is None:
                if step > 0: start = 0
                else: raise ValueError('Input slice must be bounded, or provide "size"')
            if stop is None:
                if step < 0: stop = -1
                else: raise ValueError('Input slice must be bounded, or provide "size"')
            if step < 0 and stop > start or step > 0 and stop < start:
                stop = start = 0
            else:
                stop = (stop - start + (-1) ** (step > 0)) // step * step + start + (-1) ** (step < 0)
                if stop < 0: stop = None
            sl = slice(start, stop, step)
        else:
            sl = np.array(sl, copy=copy)
            if not (np.issubdtype(sl.dtype, np.integer) or sl.dtype == '?'):
                raise ValueError('If array, must be of integer or boolean type')
            if sl.dtype == '?':
                sl = np.flatnonzero(sl)
        self.idx = sl

    @property
    def is_array(self):
        """Whether indexing is performed with an array (instead of python slice)."""
        return not isinstance(self.idx, slice)

    @classmethod
    def empty(cls):
        """Return empty slice."""
        return cls(slice(0, 0, 1))

    def __repr__(self):
        """String representation of current slice."""
        if self.is_array:
            return '{}({})'.format(self.__class__.__name__, self.idx)
        return '{}({}, {}, {})'.format(self.__class__.__name__, self.idx.start, self.idx.stop, self.idx.step)

    def to_array(self, copy=False):
        """Turn :class:`Slice` into a numpy array."""
        if self.is_array:
            return np.array(self.idx, copy=copy)
        return np.arange(self.idx.start, self.idx.stop, self.idx.step)

    def to_slices(self):
        """Turn :class:`Slice` into a list of python slices."""
        from itertools import groupby
        from operator import itemgetter
        from collections import deque
        if self.is_array:
            if self.idx.size <= 1:
                yield slice(self.idx.flat[0], self.idx.flat[0] + 1, 1)
            else:
                diff = np.diff(self.idx)
                diff = np.insert(diff, 0, diff[0])
                # This is a bit inefficient in the sense that for [0, 1, 2, 3, 5, 6, 8, 10, 12]
                # we will obtain (0, 4, 1), (5, 6, 1), (6, 7, 1), (8, 13, 2)
                for k, g in groupby(zip(self.idx, diff), lambda x: x[1]):
                    ind = map(itemgetter(0), g)
                    start = stop = next(ind)
                    try:
                        second = stop = next(ind)
                    except StopIteration:
                        yield slice(start, start + 1, 1)
                        continue
                    step = second - start
                    if step == 0:
                        yield slice(start, start + 1, 1)
                        yield slice(second, second + 1, 1)
                        for el in ind:
                            yield el
                        continue
                    try:
                        stop = deque(ind, maxlen=1).pop()
                    except IndexError:
                        pass
                    stop = stop + (-1) ** (step < 0)
                    if stop < 0: stop = None
                    yield slice(start, stop, step)
        else:
            yield self.idx

    def split(self, nsplits=1):
        """Split current :class:`Slice` into ``nsplits`` sub-slices."""
        if self.is_array:
            idxs = np.array_split(self.idx, nsplits)
        else:
            idxs = []
            for isplit in range(nsplits):
                istart = isplit * (self.size - 1) // nsplits
                istop = (isplit + 1) * (self.size - 1) // nsplits
                if isplit > 0: istart += 1
                step = self.idx.step
                start = step * istart + self.idx.start
                stop = step * istop + self.idx.start + (-1) ** (step < 0)
                if step < 0 and stop < 0: stop = None
                idxs.append(slice(start, stop, step))
        return [Slice(idx, copy=False) for idx in idxs]

    def find(self, *args, return_index=False):
        """
        Indices of input :class:`Slice` (built from ``args``) in current slice.
        With ``return_index = True``, also return indices in input slice.
        """
        sl2 = self.__class__(*args)
        if self.is_array or sl2.is_array:
            if return_index:
                idx, idx2 = utils.match1d_to(self.to_array(), sl2.to_array(), return_index=True)
            else:
                idx = utils.match1d_to(self.to_array(), sl2.to_array(), return_index=False)
        else:
            step1, step2, delta = self.idx.step, sl2.idx.step, sl2.idx.start - self.idx.start
            gcd = math.gcd(abs(step1), abs(step2))  # gcd always positive
            if delta % gcd != 0:
                idx = idx2 = slice(0, 0, 1)
            else:
                # Search solution
                a, b, c = abs(step1 // gcd), abs(step2 // gcd), delta // gcd
                if c == 0:
                    x0 = 0
                else:
                    for x0 in range(0, b):
                        if (a * x0) % b == 1: break
                    x0 *= c
                step = step2 // gcd
                if step1 < 0:
                    x0 *= -1
                    step *= -1
                # Positivity of ii1 & ii2
                stepa = step1 * step
                imin = (max(self.min, sl2.min) - step1 * x0 - self.idx.start) / stepa
                imax = (min(self.max, sl2.max) - step1 * x0 - self.idx.start) / stepa

                if stepa < 0:
                    imin, imax = imax, imin
                istart = math.ceil(imin)
                istop = math.floor(imax)
                if istop < istart:
                    idx = idx2 = slice(0, 0, 1)
                else:
                    start = step * istart + x0
                    stop = step * istop + x0 + (-1) ** (step < 0)
                    if step < 0 and stop < 0: stop = None
                    idx = slice(start, stop, step)
                    # indices in sl2
                    start = (step1 * (x0 + step * istart) - delta) // step2
                    stop = (step1 * (x0 + step * istop) - delta) // step2 + 1
                    step = step1 * step // step2  # always positive
                    idx2 = slice(start, stop, step)

        idx = self.__class__(idx, copy=False)
        if return_index:
            return idx, self.__class__(idx2, copy=False)
        return idx

    def slice(self, *args, return_index=False):
        """
        Slice current slice with input :class:`Slice` (built from ``args``), to get:
        ``all(array[sl1.slice(sl2)] == array[sl1][sl2])``.
        With ``return_index = True``, also return indices in input slice.
        """
        sl2 = self.__class__(*args)
        if self.is_array or sl2.is_array:
            idx2 = sl2.to_array()
            mask = (idx2 >= 0) & (idx2 < self.size)
            idx = self.to_array()[idx2[mask]]
            # indices in idx2
            if return_index: idx2 = np.flatnonzero(mask)
        else:
            # I = a i + b
            # I' = a' I + b' = a' a i + a' b + b' ' is self
            x0 = self.idx.step * sl2.idx.start + self.idx.start
            step = sl2.idx.step * self.idx.step
            min2 = self.idx.step * sl2.min + self.idx.start
            max2 = self.idx.step * sl2.max + self.idx.start
            if self.idx.step < 0: min2, max2 = max2, min2
            imin = (max(self.min, min2) - x0) / step
            imax = (min(self.max, max2) - x0) / step
            if step < 0:
                imin, imax = imax, imin
            istart = math.ceil(imin)
            istop = math.floor(imax)
            if istop < istart:
                idx = slice(0, 0, 1)
                idx2 = slice(0, 0, 1)
            else:
                start = step * istart + x0
                stop = step * istop + x0 + (-1) ** (step < 0)
                if step < 0 and stop < 0: stop = None
                idx = slice(start, stop, step)
                idx2 = self.find(sl2, return_index=True)[1]

        idx = self.__class__(idx, copy=False)
        if return_index:
            return idx, self.__class__(idx2, copy=False)
        return idx

    def shift(self, offset=0, stop=None):
        """
        Shift current slice by input ``offset``.
        Provide ``stop`` to limit the scope of returned :class:`Slice`.
        """
        if self.is_array:
            idx = self.idx + offset
            idx = idx[idx >= 0]
            if stop is not None:
                idx = idx[idx < stop]
        else:
            nstart = self.idx.start + offset
            nstop = (self.idx.stop if self.idx.stop is not None else -1) + offset
            nstep = self.idx.step
            if stop is not None: nstop = min(nstop, stop)
            if nstart < 0 and nstop <= 0:
                return self.empty()
            if nstep < 0 and nstop < 0: nstop = None
            idx = slice(nstart, nstop, nstep)
        return self.__class__(idx, copy=False)

    def __len__(self):
        """Slice length."""
        if self.is_array:
            return self.idx.size
        return ((-1 if self.idx.stop is None else self.idx.stop) - self.idx.start + (-1)**(self.idx.step > 0)) // self.idx.step + 1

    @property
    def size(self):
        """Equivalent for :meth:`__len__`."""
        return len(self)

    @property
    def min(self):
        """Minimum (i.e. minimum index) of the slice."""
        if self.is_array:
            return self.idx.min()
        if self.idx.step < 0:
            return self.idx.step * (self.size - 1) + self.idx.start
        return self.idx.start

    @property
    def max(self):
        """Maximum (i.e. maximum index) of the slice."""
        if self.is_array:
            return self.idx.max()
        if self.idx.step > 0:
            return self.idx.step * (self.size - 1) + self.idx.start
        return self.idx.start

    def __eq__(self, other):
        """Whether two slices are equal."""
        try:
            other = Slice(other)
        except ValueError:
            return False
        if self.is_array and other.is_array:
            return other.idx.size == self.idx.size and np.all(other.idx == self.idx)
        if (not self.is_array) and (not other.is_array):
            return other.idx == self.idx
        return False

    @classmethod
    def snap(cls, *others):
        """
        Snap input slices together, e.g.:
        >>> Slice.snap(slice(0, 10, 2), slice(10, 20, 2))  # Slice(0, 20, 2)
        """
        others = [Slice(other) for other in others]
        if any(other.is_array for other in others):
            return [Slice(np.concatenate([other.to_array() for other in others], axis=0))]
        if not others:
            return [Slice(0, 0, 1)]
        slices = [others[0].idx]
        for other in others[1:]:
            if other.idx.step == slices[-1].step and other.idx.start == slices[-1].stop + other.idx.step + (-1) ** (other.idx.step > 0):
                slices[-1] = slice(slices[-1].start, other.idx.stop, slices[-1].step)
            else:
                slices.append(other.idx)
        return [Slice(sl) for sl in slices]

    @CurrentMPIComm.enable
    def mpi_send(self, dest, tag=0, blocking=True, mpicomm=None):
        """Send slice to rank ``dest``."""
        if blocking: send = mpicomm.send
        else: send = mpicomm.isend
        send(self.is_array, dest=dest, tag=tag + 1)
        if self.is_array:
            mpi.send_array(self.idx, dest=dest, tag=tag, blocking=blocking, mpicomm=mpicomm)
        else:
            send(self.idx, dest=dest, tag=tag)

    @classmethod
    @CurrentMPIComm.enable
    def mpi_recv(cls, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, mpicomm=None):
        """Receive slice from ``source``."""
        if mpicomm.recv(source=source, tag=tag + 1):  # is_array
            idx = mpi.recv_array(source=source, tag=tag, mpicomm=mpicomm)
        else:
            idx = mpicomm.recv(source=source, tag=tag)
        return cls(idx)


class MPIScatteredSource(BaseClass):
    """
    Utility class to manage array slices on different processes,
    typically useful when applying collective slicing or concatenate operations.
    """
    @CurrentMPIComm.enable
    def __init__(self, *slices, csize=None, mpicomm=None):
        """
        Initialize :class:`MPIScatteredSource`.

        Parameters
        ----------
        slices : tuple
            Sequence of input :class:`Slice`, representing the indices of the global array on the local rank.

        csize : int, default=None
            The collective array slice, passed as ``size`` to :class:`Slice`.

        mpicomm : MPI communicator, default=None
            The current MPI communicator.
        """
        # let's restrict to disjoint slices...
        self.mpicomm = mpicomm
        self.slices = [Slice(sl, size=csize) for sl in slices]
        if any(sl.is_array for sl in self.slices):
            raise NotImplementedError('Only slices supported so far')
        if csize is not None and csize != self.csize:
            raise ValueError('Input slices do not have collective size "csize"')

    @property
    def size(self):
        """The (total) array length on this rank."""
        if getattr(self, '_size', None) is None:
            self._size = sum(sl.size for sl in self.slices)
        return self._size

    @property
    def csize(self):
        """The collective array length."""
        return self.mpicomm.allreduce(self.size)

    def get(self, arrays, *args):
        """
        Return a global slice of input arrays.

        Parameters
        ----------
        arrays : tuple, list
            List of input arrays, corresponding to indices in :attr:`slices`.

        args : tuple
            Sequence of global slices/indices to return array for.

        Returns
        -------
        array : array
            Array corresponding to input slices.
        """
        # Here, slice in global coordinates
        if not is_sequence(arrays):
            arrays = [arrays]
        if len(arrays) != len(self.slices):
            raise ValueError('Expected list of arrays of length {:d}, found {:d}'.format(len(self.slices), len(arrays)))
        size = sum(map(len, arrays))
        if size != self.size:
            raise ValueError('Expected list of arrays of total length {:d}, found {:d}'.format(self.size, size))
        if not args:
            args = (slice(self.mpicomm.rank * self.csize // self.mpicomm.size, (self.mpicomm.rank + 1) * self.csize // self.mpicomm.size, 1), )

        all_slices = self.mpicomm.allgather(self.slices)
        nslices = max(map(len, all_slices))
        toret = []

        for sli in args:
            sli = Slice(sli, size=self.csize)
            idx, tmp = [None] * self.mpicomm.size, [None] * self.mpicomm.size
            for irank in range(self.mpicomm.size):
                self_slice_in_irank = [sl.find(sli, return_index=True) for sl in all_slices[irank]]
                idx[irank] = [sl[1].idx for sl in self_slice_in_irank]
                if irank == self.mpicomm.rank:
                    tmp[irank] = [array[sl[0].idx] for iarray, (array, sl) in enumerate(zip(arrays, self_slice_in_irank))]
                else:
                    for isl, sl in enumerate(self_slice_in_irank): sl[0].mpi_send(dest=irank, tag=isl)
                    self_slice_in_irank = [Slice.mpi_recv(source=irank, tag=isl) for isl in range(len(self.slices))]
                    for iarray, (array, sl) in enumerate(zip(arrays, self_slice_in_irank)):
                        mpi.send_array(array[sl.idx], dest=irank, tag=nslices + iarray, mpicomm=self.mpicomm)
                    tmp[irank] = [mpi.recv_array(source=irank, tag=nslices + iarray, mpicomm=self.mpicomm) for iarray in range(len(self_slice_in_irank))]
            idx, tmp = utils.list_concatenate(idx), utils.list_concatenate(tmp)
            if sli.is_array:
                toret.append(np.concatenate(tmp, axis=0)[np.argsort(np.concatenate(idx, axis=0))])
            else:
                toret += [tmp[ii] for ii in np.argsort([iidx.start for iidx in idx])]
        return np.concatenate(toret)

    @classmethod
    def cconcatenate(cls, *others):
        """Concatenate input :class:`MPIScatteredArray`, collectively, i.e. preserving order accross all ranks."""
        if not others:
            raise ValueError('Provide at least one {} instance.'.format(cls.__name__))
        if len(others) == 1 and is_sequence(others[0]):
            others = others[0]
        slices, cumsize = [], 0
        for other in others:
            slices += [sl.shift(cumsize) for sl in other.slices]
            cumsize += other.csize
        return cls(*slices)

    def cappend(self, other, **kwargs):
        """Append ``other`` to current :class:`MPIScatteredSource`."""
        return self.cconcatenate(self, [other], **kwargs)

    def cextend(self, other, **kwargs):
        """Extend (in-place) current :class:`MPIScatteredSource` with ``other``."""
        new = self.cappend(self, other, **kwargs)
        self.__dict__.update(new.__dict__)


class MPIScatteredArray(NDArrayLike, BaseClass, metaclass=BaseMetaClass):

    """
    A class representing a numpy array scattered on several processes.
    It can be used transparently with any numpy function (in which case computation is local).
    """

    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    @CurrentMPIComm.enable
    def __init__(self, value=None, copy=False, dtype=None, mpiroot=None, mpicomm=None):
        """
        Initalize :class:`MPIScatteredArray`.

        Parameters
        ----------
        value : array
            Local array value.

        copy : bool, default=False
            Whether to copy input array.

        dtype : dtype, default=None
            If provided, enforce this dtype.

        mpiroot : int, default=None
            If ``None``, input array is assumed to be scattered across all ranks.
            Else the MPI rank where input array is gathered.

        mpicomm : MPI communicator, default=None
            The current MPI communicator.
        """
        self.mpicomm = mpicomm
        if mpiroot is None or self.mpicomm.rank == mpiroot:
            value = np.array(value, copy=copy, dtype=dtype, order='C')
        if mpiroot is not None:
            value = mpi.scatter_array(value, mpicomm=mpicomm, root=mpiroot)
        self.value = value

    @classmethod
    def falses(cls, *args, **kwargs):
        """Return array full of ``False``."""
        return cls.zeros(*args, **kwargs, dtype=np.bool_)

    @classmethod
    def trues(cls, *args, **kwargs):
        """Return array full of ``True``."""
        return cls.ones(*args, **kwargs, dtype=np.bool_)

    @classmethod
    def nans(cls, *args, **kwargs):
        """Return array full of ``NaN``."""
        return cls.ones(*args, **kwargs) * np.nan

    def __mul__(self, other):
        # Multiply array
        r = self.copy()
        r.value = r.value * other
        return r

    def __imul__(self, other):
        self.value *= other
        return self

    def __div__(self, other):
        # Divide array
        r = self.copy()
        r.value = r.value / other
        return r

    __truediv__ = __div__

    def __rdiv__(self, other):
        r = self.copy()
        r.value = other / r.value
        return r

    __rtruediv__ = __rdiv__

    def __idiv__(self, other):
        self.value /= other
        return self

    __itruediv__ = __idiv__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Taken from https://numpy.org/doc/stable/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html
        # See also https://github.com/rainwoodman/pmesh/blob/master/pmesh/pm.py
        out = kwargs.get('out', ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use BaseMesh instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle BaseMesh objects.
            if not isinstance(x, self._HANDLED_TYPES + (MPIScatteredArray,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.value if isinstance(x, MPIScatteredArray) else x for x in inputs)
        if out:
            kwargs['out'] = tuple(x.value if isinstance(x, MPIScatteredArray) else x for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        def cast(result):
            # really only cast when we are using simple +-* **, etc.
            if result.ndim == 0:
                return result
            new = self.copy()
            new.value = result
            return new

        if type(result) is tuple:
            # multiple return values
            return tuple(cast(x) for x in result)
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            return cast(result)

    def __getitem__(self, index):
        toret = self.value.__getitem__(index)
        if toret.ndim == 0:  # scalar
            return toret
        return self.__class__(toret, mpicomm=self.mpicomm)

    def __setitem__(self, index, value):
        return self.value.__setitem__(index, value)

    def __array__(self, dtype=None):
        return self.value.astype(dtype, copy=False)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.value)

    def __str__(self):
        return str(self.value)

    @property
    def csize(self):
        """Collective array size."""
        return self.mpicomm.allreduce(self.value.size)

    @property
    def cshape(self):
        """Collective array shape."""
        return (self.mpicomm.allreduce(self.value.shape[0]),) + self.value.shape[1:]

    def __len__(self):
        return len(self.value)

    @cshape.setter
    def cshape(self, cshape):
        """
        Set collective shape attr:`cshape`, e.g.:
        >>> array.cshape = (100, 2)

        This will induce data exchange between various processes if local size cannot be divided by ``prod(cshape[1:])``.
        """
        if np.ndim(cshape) == 0:
            cshape = (cshape,)
        cshape = tuple(cshape)
        unknown = [s < 0 for s in cshape]
        if sum(unknown) > 1:
            raise ValueError('can only specify one unknown dimension')
        if sum(unknown) == 1:
            size = self.csize // np.prod([s for s in cshape if s >= 0], dtype='i')
            cshape = tuple(s if s >= 0 else size for s in cshape)
        csize = np.prod(cshape, dtype='i')
        if csize != self.csize:
            raise ValueError('cannot reshape array of size {:d} into shape {}'.format(self.csize, cshape))
        itemsize = np.prod(cshape[1:], dtype='i')
        local_slice = Slice(self.mpicomm.rank * cshape[0] // self.mpicomm.size * itemsize, (self.mpicomm.rank + 1) * cshape[0] // self.mpicomm.size * itemsize, 1)
        if not all(self.mpicomm.allgather(local_slice.size == self.size)):  # need to reorder!
            cumsizes = np.cumsum([0] + self.mpicomm.allgather(self.size))
            source = MPIScatteredSource(slice(cumsizes[self.mpicomm.rank], cumsizes[self.mpicomm.rank + 1], 1))
            self.value = source.get(self.value.ravel(), local_slice)
        self.value.shape = (-1,) + cshape[1:]

    def slice(self, *args):
        """
        Perform local array slice, e.g.:
        >>> array.slice(0, 10, 2)
        >>> array.slice([1, 2, 2])
        """
        new = self.copy()
        new.value = self.value[Slice(*args, size=self.size).idx]
        return new

    def cslice(self, *args):
        """Perform collective array slice, e.g.:
        >>> array.cslice(0, 10, 2)
        >>> array.cslice([1, 2, 2])
        """
        new = self.copy()
        cumsizes = np.cumsum([0] + self.mpicomm.allgather(len(self)))
        global_slice = Slice(*args, size=cumsizes[-1])
        local_slice = global_slice.split(self.mpicomm.size)[self.mpicomm.rank]
        source = MPIScatteredSource(slice(cumsizes[self.mpicomm.rank], cumsizes[self.mpicomm.rank + 1], 1))
        new.value = source.get(self.value, local_slice)
        return new

    @classmethod
    def cconcatenate(cls, *others, axis=0):
        """Concatenate input :class:`MPIScatteredArray`, collectively, i.e. preserving order accross all ranks."""
        if not others:
            raise ValueError('Provide at least one {} instance.'.format(cls.__name__))
        if len(others) == 1 and is_sequence(others[0]):
            others = others[0]
        new = others[0].copy()
        axis = normalize_axis_tuple(axis, new.value.ndim)[0]
        if axis > 1:
            new.value = np.concatenate([other.value for other in others], axis=axis)
        else:
            source = []
            for other in others:
                cumsizes = np.cumsum([0] + other.mpicomm.allgather(len(other)))
                source.append(MPIScatteredSource(slice(cumsizes[other.mpicomm.rank], cumsizes[other.mpicomm.rank + 1], 1)))
            source = MPIScatteredSource.cconcatenate(*source)
            new.value = source.get([other.value for other in others])
        return new

    def cappend(self, other, **kwargs):
        """Append ``other`` to current :class:`MPIScatteredSource`."""
        return self.cconcatenate(self, [other], **kwargs)

    def cextend(self, other, **kwargs):
        """Extend (in-place) current :class:`MPIScatteredSource` with ``other``."""
        new = self.cappend(self, other, **kwargs)
        self.__dict__.update(new.__dict__)

    def gathered(self, mpiroot=0):
        """Return numpy array gathered on rank ``mpiroot`` (``None`` to gather on all ranks)."""
        return mpi.gather_array(self.value, mpicomm=self.mpicomm, root=mpiroot)

    def csort(self, axis=0, kind=None):
        """
        Sort input array ``data`` along ``axis``.
        Naive implementation: array is gathered, sorted, and scattered again.
        Faster than naive distributed sorts (bitonic, transposition)...

        Parameters
        ----------
        axis : int, default=-1
            Sorting axis.

        kind : string, default=None
            Sorting algorithm. The default is ‘quicksort’.
            See :func:`numpy.sort`.

        mpicomm : MPI communicator
            Communicator. Defaults to current communicator.

        Returns
        -------
        out : array
            Sorted array (scattered).
        """
        # import mpsort
        # self.value = mpsort.sort(self.value, orderby=None, comm=self.mpicomm, tuning=[])
        self.value = mpi.sort_array(self.value, axis=axis, kind=kind, mpicomm=self.mpicomm)  # most naive implementation

    def csum(self, axis=0):
        """Collective array sum along ``axis``."""
        return mpi.sum_array(self.value, axis=axis, mpicomm=self.mpicomm)

    def caverage(self, weights=None, axis=0):
        """Collective array average along ``axis``."""
        return mpi.average_array(self.value, weights=weights, axis=axis, mpicomm=self.mpicomm)

    def cmean(self, axis=0):
        """Collective array mean along ``axis``."""
        return self.caverage(axis=axis)

    def cvar(self, axis=0, fweights=None, aweights=None, ddof=1):
        """
        Estimate collective variance, given weights.
        See :func:`numpy.var`.
        TODO: allow several axes.

        Parameters
        ----------
        axis : int, default=-1
            Axis along which the variance is computed.

        fweights : array, int, default=None
            1D array of integer frequency weights; the number of times each
            observation vector should be repeated.

        aweights : array, default=None
            1D array of observation vector weights. These relative weights are
            typically large for observations considered "important" and smaller for
            observations considered less "important". If ``ddof=0`` the array of
            weights can be used to assign probabilities to observation vectors.

        ddof : int, default=1
            Note that ``ddof=1`` will return the unbiased estimate, even if both
            `fweights` and `aweights` are specified, and ``ddof=0`` will return
            the simple average.

        mpicomm : MPI communicator
            Current MPI communicator.

        Returns
        -------
        out : array
            The variance of the variables.
        """
        return mpi.var_array(self.value, axis=axis, fweights=fweights, aweights=aweights, ddof=ddof, mpicomm=self.mpicomm)

    def cstd(self, axis=0, fweights=None, aweights=None, ddof=1):
        """
        Collective weighted standard deviation along axis ``axis``.
        Simply take square root of :meth:`cvar` result.
        TODO: allow for several axes.
        """
        return np.sqrt(self.cvar(axis=axis, fweights=fweights, aweights=aweights, ddof=ddof))

    def cmin(self, axis=0):
        """Collective minimum along ``axis``."""
        return mpi.min_array(self.value, axis=axis, mpicomm=self.mpicomm)

    def cmax(self, axis=0):
        """Collective maximum along ``axis``."""
        return mpi.max_array(self.value, axis=axis, mpicomm=self.mpicomm)

    def cargmin(self, axis=0):
        """Local index and rank of collective minimum along ``axis``."""
        return mpi.argmin_array(self.value, axis=axis, mpicomm=self.mpicomm)

    def cargmax(self, axis=0):
        """Local index and rank of collective maximum along ``axis``."""
        return mpi.argmax_array(self.value, axis=axis, mpicomm=self.mpicomm)

    def cquantile(self, q, weights=None, axis=0, interpolation='linear'):
        """
        Return weighted array quantiles.
        Naive implementation: array is gathered before taking quantile.

        Parameters
        ----------
        a : array
            Input array or object that can be converted to an array.

        q : tuple, list, array
            Quantile or sequence of quantiles to compute, which must be between
            0 and 1 inclusive.

        weights : array, default=None
            An array of weights associated with the values in ``a``. Each value in
            ``a`` contributes to the cumulative distribution according to its associated weight.
            The weights array can either be 1D (in which case its length must be
            the size of ``a`` along the given axis) or of the same shape as ``a``.
            If ``weights=None``, then all data in ``a`` are assumed to have a
            weight equal to one.
            The only constraint on ``weights`` is that ``sum(weights)`` must not be 0.

        axis : int, tuple, default=None
            Axis or axes along which the quantiles are computed. The
            default is to compute the quantile(s) along a flattened
            version of the array.

        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}, default='linear'
            This optional parameter specifies the interpolation method to
            use when the desired quantile lies between two data points
            ``i < j``:

            * linear: ``i + (j - i) * fraction``, where ``fraction``
              is the fractional part of the index surrounded by ``i``
              and ``j``.
            * lower: ``i``.
            * higher: ``j``.
            * nearest: ``i`` or ``j``, whichever is nearest.
            * midpoint: ``(i + j) / 2``.

        Returns
        -------
        quantile : scalar, array
            If ``q`` is a single quantile and ``axis=None``, then the result
            is a scalar. If multiple quantiles are given, first axis of
            the result corresponds to the quantiles. The other axes are
            the axes that remain after the reduction of ``a``. If the input
            contains integers or floats smaller than ``float64``, the output
            data-type is ``float64``. Otherwise, the output data-type is the
            same as that of the input. If ``out`` is specified, that array is
            returned instead.

        Note
        ----
        Inspired from https://github.com/minaskar/cronus/blob/master/cronus/plot.py.
        """
        return mpi.weighted_quantile_array(self.value, q, weights=weights, axis=axis, interpolation=interpolation, mpicomm=self.mpicomm)


def _make_getter(name):

    def getter(self):
        return getattr(self.value, name)

    return property(getter)


def _make_getter_setter(name):

    def getter(self):
        return getattr(self.value, name)

    def setter(self, value):
        setattr(self.value, name, value)

    return property(getter, setter)


for name in ['size']:

    setattr(MPIScatteredArray, name, _make_getter(name))


for name in ['shape', 'dtype']:

    setattr(MPIScatteredArray, name, _make_getter_setter(name))


def _make_constructor(name):

    @classmethod
    @CurrentMPIComm.enable
    def constructor(cls, *args, mpicomm=None, **kwargs):
        cshape = kwargs.pop('cshape', None)
        if cshape is not None:
            if np.ndim(cshape) == 0:
                cshape = (cshape,)
            cshape = tuple(cshape)
            size = (mpicomm.rank + 1) * cshape[0] // mpicomm.size - mpicomm.rank * cshape[0] // mpicomm.size
            kwargs['shape'] = (size,) + cshape[1:]
        array = getattr(np, name)(*args, **kwargs)
        return cls(array, mpicomm=mpicomm)

    return constructor


for name in ['empty', 'zeros', 'ones', 'full']:

    setattr(MPIScatteredArray, name, _make_constructor(name))
