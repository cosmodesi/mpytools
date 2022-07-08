"""Implements MPI-scattered array."""

import warnings
import numbers
import math

import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from mpi4py import MPI

from . import utils
from .utils import BaseClass, is_sequence, CurrentMPIComm


__all__ = ['array', 'reduce', 'gather', 'bcast', 'scatter', 'send', 'recv', 'sendrecv',
           'csize', 'cshape', 'creshape', 'cslice', 'cconcatenate', 'cappend',
           'csum', 'cprod', 'cmean', 'caverage', 'cmin', 'cmax', 'cargmin', 'cargmax', 'csort', 'cquantile',
           'cvar', 'cstd', 'ccov', 'ccorrcoef']


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

    def nslices(self):
        if self.is_array:
            return 1 + np.sum(np.diff(np.diff(self.idx)) != 0)
        return 1

    def to_slices(self):
        """Turn :class:`Slice` into a list of python slices."""
        from itertools import groupby
        from operator import itemgetter
        from collections import deque
        if self.is_array:
            if self.idx.size == 0:
                yield slice(0, 0, 1)
            elif self.idx.size == 1:
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
    def send(self, dest, tag=0, mpicomm=None):
        """Send slice to rank ``dest``."""
        mpicomm.send(self.is_array, dest=dest, tag=tag + 1)
        if self.is_array:
            send(self.idx, dest=dest, tag=tag, mpicomm=mpicomm)
        else:
            mpicomm.send(self.idx, dest=dest, tag=tag)

    @classmethod
    @CurrentMPIComm.enable
    def recv(cls, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, mpicomm=None):
        """Receive slice from ``source``."""
        if mpicomm.recv(source=source, tag=tag + 1):  # is_array
            idx = recv(source=source, tag=tag, mpicomm=mpicomm)
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
                    for isl, sl in enumerate(self_slice_in_irank): sl[0].send(dest=irank, tag=isl)
                    self_slice_in_irank = [Slice.recv(source=irank, tag=isl) for isl in range(len(self.slices))]
                    for iarray, (array, sl) in enumerate(zip(arrays, self_slice_in_irank)):
                        send(array[sl.idx], dest=irank, tag=nslices + iarray, mpicomm=self.mpicomm)
                    tmp[irank] = [recv(source=irank, tag=nslices + iarray, mpicomm=self.mpicomm) for iarray in range(len(self_slice_in_irank))]
            idx, tmp = utils.list_concatenate(idx), utils.list_concatenate(tmp)
            if sli.is_array:
                toret.append(np.concatenate(tmp, axis=0)[np.argsort(np.concatenate(idx, axis=0))])
            else:
                toret += [tmp[ii] for ii in np.argsort([iidx.start for iidx in idx])]
        return np.concatenate(toret)

    @classmethod
    def cconcatenate(cls, *others):
        """Concatenate input :class:`array`, collectively, i.e. preserving order accross all ranks."""
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


class array(np.ndarray):
    """
    A class representing a numpy array scattered on several processes.
    It can be used transparently with any numpy function (in which case computation is local).
    """
    @CurrentMPIComm.enable
    def __new__(cls, value, copy=False, dtype=None, mpiroot=None, mpicomm=None):
        """
        Initalize :class:`array`.

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
        if mpiroot is None or mpicomm.rank == mpiroot:
            value = np.array(value, copy=copy, dtype=dtype)
        if mpiroot is not None:
            value = scatter(value, mpicomm=mpicomm, mpiroot=mpiroot)
        obj = value.view(cls)
        obj.mpicomm = mpicomm
        return obj

    def __array_finalize__(self, obj):
        self.mpicomm = getattr(obj, 'mpicomm', None)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        args = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, array):
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, array):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], array):
                inputs[0].mpicomm = self.mpicomm
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(array)
                         if output is None else output)
                        for result, output in zip(results, outputs))

        for result in results:
            if isinstance(result, array):
                result.mpicomm = self.mpicomm

        return results[0] if len(results) == 1 else results

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self)

    @property
    def csize(self):
        """Collective array size."""
        return csize(self)

    @property
    def cshape(self):
        """Collective array shape."""
        return cshape(self)

    def creshape(self, *cshape):
        """
        Set collective shape attr:`cshape`.
        This will induce data exchange between various processes if local size cannot be divided by ``prod(cshape[1:])``.
        """
        if len(cshape) == 0:
            cshape = cshape[0]
        if np.ndim(cshape) == 0:
            cshape = (cshape,)
        cshape = tuple(cshape)
        return creshape(self, cshape)

    def gather(self, mpiroot=0):
        """Return numpy array gathered on rank ``mpiroot`` (``None`` to gather on all ranks)."""
        return gather(self, mpicomm=self.mpicomm, mpiroot=mpiroot)

    def reduce(self, mpiroot=0, op='sum'):
        """Return numpy array reduced on rank ``mpiroot`` (``None`` to reduce on all ranks)."""
        return reduce(self, mpicomm=self.mpicomm, op=op)

    def cslice(self, *args):
        """Perform collective array slicing."""
        return cslice(self, *args)

    def csort(self, axis=0, kind=None):
        """(Collectively) sort input array ``data`` along ``axis``."""
        # import mpsort
        # self.value = mpsort.sort(self.value, orderby=None, comm=self.mpicomm, tuning=[])
        self.data = csort(self, axis=axis, kind=kind).data  # most naive implementation

    def csum(self, axis=0, **kwargs):
        """Collective array sum along ``axis``."""
        return csum(self, axis=axis, **kwargs)

    def cprod(self, axis=0, **kwargs):
        """Collective array product along ``axis``."""
        return cprod(self, axis=axis, **kwargs)

    def cmean(self, axis=0, **kwargs):
        """Collective array mean along ``axis``."""
        return cmean(self, axis=axis, **kwargs)

    def cvar(self, axis=0, **kwargs):
        """Collective array variance along ``axis``."""
        return cvar(self, axis=axis, **kwargs)

    def cstd(self, axis=0, **kwargs):
        """Collective weighted standard deviation along axis ``axis``."""
        return cstd(self, axis=axis, **kwargs)

    def cmin(self, axis=0, **kwargs):
        """Collective minimum along ``axis``."""
        return cmin(self, axis=axis, **kwargs)

    def cmax(self, axis=0, **kwargs):
        """Collective maximum along ``axis``."""
        return cmax(self, axis=axis, **kwargs)

    def cargmin(self, axis=0, **kwargs):
        """Local index and rank of collective minimum along ``axis``."""
        return cargmin(self, axis=axis, **kwargs)

    def cargmax(self, axis=0, **kwargs):
        """Local index and rank of collective maximum along ``axis``."""
        return cargmax(self, axis=axis, **kwargs)


def _make_constructor(name):

    @CurrentMPIComm.enable
    def constructor(*args, mpicomm=None, **kwargs):
        cshape = kwargs.pop('cshape', None)
        if cshape is not None:
            if np.ndim(cshape) == 0:
                cshape = (cshape,)
            cshape = tuple(cshape)
            size = (mpicomm.rank + 1) * cshape[0] // mpicomm.size - mpicomm.rank * cshape[0] // mpicomm.size
            kwargs['shape'] = (size,) + cshape[1:]
        arr = getattr(np, name)(*args, **kwargs)
        return array(arr, mpicomm=mpicomm)

    return constructor


for name in ['empty', 'zeros', 'ones', 'full']:

    globals()[name] = _make_constructor(name)
    __all__.append(name)


@CurrentMPIComm.enable
def reduce(data, op='sum', mpiroot=0, mpicomm=None):
    """
    Reduce the input data array from all ranks to the specified ``mpiroot``.

    Parameters
    ----------
    data : array_like
        The data on each rank to gather.

    op : string, MPI.Op
        MPI operation, or 'sum', 'prod', 'min', 'max'.

    mpiroot : int, Ellipsis, default=0
        The rank number to reduce the data to. If mpiroot is Ellipsis or None,
        broadcast the result to all ranks.

    mpicomm : MPI communicator, default=None
        The MPI communicator.

    Returns
    -------
    recvbuffer : array_like, None
        The reduced data on mpiroot, and `None` otherwise.
    """
    if mpiroot is None: mpiroot = Ellipsis

    if isinstance(op, str):
        op = {'sum': MPI.SUM, 'prod': MPI.PROD, 'min': MPI.MIN, 'max': MPI.MAX}[op]

    if np.isscalar(data):
        if mpiroot is Ellipsis:
            return mpicomm.allreduce(data, op=op)
        return mpicomm.reduce(data, op=op, root=mpiroot)

    data = np.asarray(data)
    isstruct = data.dtype.names is not None
    if isstruct:
        toret = None
        if mpiroot is Ellipsis or mpicomm.rank == mpiroot:
            toret = np.empty_like(data)
        for name in data.dtype.names:
            tmp = reduce(data[name], op=op, mpiroot=mpiroot, mpicomm=mpicomm)
            if mpiroot is Ellipsis or mpicomm.rank == mpiroot:
                toret[name] = tmp
        return toret

    shape, dtype = data.shape, data.dtype
    data = np.ascontiguousarray(data)

    if np.issubdtype(dtype, np.floating):
        # Otherwise, weird error   File "mpi4py/MPI/commimpl.pxi", line 142, in mpi4py.MPI.PyMPI_Lock KeyError: '<d'
        data = data.astype('f{:d}'.format(data.dtype.itemsize))

    if mpiroot is Ellipsis:
        total = np.empty_like(data)
        mpicomm.Allreduce(data, total, op=op)
        total = total.reshape(shape).astype(dtype, copy=False)
    else:
        total = None
        if mpicomm.rank == mpiroot:
            total = np.empty_like(data)
        mpicomm.Reduce(data, total, op=op, root=mpiroot)
        if mpicomm.rank == mpiroot:
            total = total.reshape(shape).astype(dtype, copy=False)
    return total


@CurrentMPIComm.enable
def gather(data, mpiroot=0, mpicomm=None):
    """
    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py.
    Gather the input data array from all ranks to the specified ``mpiroot``.
    This uses ``Gatherv``, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype.

    Parameters
    ----------
    data : array_like
        The data on each rank to gather.

    mpiroot : int, Ellipsis, default=0
        The rank number to gather the data to. If mpiroot is Ellipsis or None,
        broadcast the result to all ranks.

    mpicomm : MPI communicator, default=None
        The MPI communicator.

    Returns
    -------
    recvbuffer : array_like, None
        The gathered data on mpiroot, and `None` otherwise.
    """
    if mpiroot is None: mpiroot = Ellipsis

    if np.isscalar(data):
        if mpiroot is Ellipsis:
            return np.array(mpicomm.allgather(data))
        gathered = mpicomm.gather(data, root=mpiroot)
        if mpicomm.rank == mpiroot:
            return np.array(gathered)
        return None

    # Need C-contiguous order
    data = np.asarray(data)
    shape, dtype = data.shape, data.dtype
    data = np.ascontiguousarray(data)

    local_length = data.shape[0]

    # check dtypes and shapes
    shapes = mpicomm.allgather(data.shape)
    dtypes = mpicomm.allgather(data.dtype)

    # check for structured data
    if dtypes[0].char == 'V':

        # check for structured data mismatch
        names = set(dtypes[0].names)
        if any(set(dt.names) != names for dt in dtypes[1:]):
            raise ValueError('mismatch between data type fields in structured data')

        # check for 'O' data types
        if any(dtypes[0][name] == 'O' for name in dtypes[0].names):
            raise ValueError('object data types ("O") not allowed in structured data in gather')

        # compute the new shape for each rank
        newlength = mpicomm.allreduce(local_length)
        newshape = list(data.shape)
        newshape[0] = newlength

        # the return array
        if mpiroot is Ellipsis or mpicomm.rank == mpiroot:
            recvbuffer = np.empty(newshape, dtype=dtypes[0], order='C')
        else:
            recvbuffer = None

        for name in dtypes[0].names:
            d = gather(data[name], mpiroot=mpiroot, mpicomm=mpicomm)
            if mpiroot is Ellipsis or mpicomm.rank == mpiroot:
                recvbuffer[name] = d

        return recvbuffer

    # check for 'O' data types
    if dtypes[0] == 'O':
        raise ValueError('object data types ("O") not allowed in structured data in gather')

    # check for bad dtypes and bad shapes
    if mpiroot is Ellipsis or mpicomm.rank == mpiroot:
        bad_shape = any(s[1:] != shapes[0][1:] for s in shapes[1:])
        bad_dtype = any(dt != dtypes[0] for dt in dtypes[1:])
    else:
        bad_shape, bad_dtype = None, None

    if mpiroot is not Ellipsis:
        bad_shape, bad_dtype = mpicomm.bcast((bad_shape, bad_dtype), root=mpiroot)

    if bad_shape:
        raise ValueError('mismatch between shape[1:] across ranks in gather')
    if bad_dtype:
        raise ValueError('mismatch between dtypes across ranks in gather')

    shape = data.shape
    dtype = data.dtype

    # setup the custom dtype
    duplicity = np.prod(shape[1:], dtype='intp')
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    # compute the new shape for each rank
    newlength = mpicomm.allreduce(local_length)
    newshape = list(shape)
    newshape[0] = newlength

    # the return array
    if mpiroot is Ellipsis or mpicomm.rank == mpiroot:
        recvbuffer = np.empty(newshape, dtype=dtype, order='C')
    else:
        recvbuffer = None

    # the recv counts
    counts = mpicomm.allgather(local_length)
    counts = np.array(counts, order='C')

    # the recv offsets
    offsets = np.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    # gather to mpiroot
    if mpiroot is Ellipsis:
        mpicomm.Allgatherv([data, dt], [recvbuffer, (counts, offsets), dt])
    else:
        mpicomm.Gatherv([data, dt], [recvbuffer, (counts, offsets), dt], root=mpiroot)

    dt.Free()

    return recvbuffer


@CurrentMPIComm.enable
def bcast(data, mpiroot=0, mpicomm=None):
    """
    Broadcast the input data array across all ranks, assuming ``data`` is
    initially only on `mpiroot` (and `None` on other ranks).
    This uses ``Scatterv``, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype.

    Parameters
    ----------
    data : array_like or None
        On `mpiroot`, this gives the data to broadcast.

    mpiroot : int, default=0
        The rank number that initially has the data.

    mpicomm : MPI communicator, default=None
        The MPI communicator.

    Returns
    -------
    recvbuffer : array_like
        ``data`` on each rank.
    """
    if mpicomm.rank == mpiroot:
        recvbuffer = np.asarray(data)
        for rank in range(mpicomm.size):
            if rank != mpiroot: send(data, rank, tag=0, mpicomm=mpicomm)
    else:
        recvbuffer = recv(source=mpiroot, tag=0, mpicomm=mpicomm)
    return recvbuffer


@CurrentMPIComm.enable
def scatter(data, counts=None, mpiroot=0, mpicomm=None):
    """
    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py
    Scatter the input data array across all ranks, assuming ``data`` is
    initially only on `mpiroot` (and `None` on other ranks).
    This uses ``Scatterv``, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype

    Parameters
    ----------
    data : array_like or None
        On `mpiroot`, this gives the data to split and scatter.

    counts : list of int
        List of the lengths of data to send to each rank.

    mpiroot : int, default=0
        The rank number that initially has the data.

    mpicomm : MPI communicator, default=None
        The MPI communicator.

    Returns
    -------
    recvbuffer : array_like
        The chunk of ``data`` that each rank gets.
    """
    if counts is not None:
        counts = np.asarray(counts, order='C')
        if len(counts) != mpicomm.size:
            raise ValueError('counts array has wrong length!')

    if mpicomm.rank == mpiroot:
        # Need C-contiguous order
        data = np.ascontiguousarray(data)
        shape_and_dtype = (data.shape, data.dtype)
    else:
        shape_and_dtype = None

    # each rank needs shape/dtype of input data
    shape, dtype = mpicomm.bcast(shape_and_dtype, root=mpiroot)

    # object dtype is not supported
    fail = False
    if dtype.char == 'V':
        fail = any(dtype[name] == 'O' for name in dtype.names)
    else:
        fail = dtype == 'O'
    if fail:
        raise ValueError('"object" data type not supported in scatter; please specify specific data type')

    # initialize empty data on non-mpiroot ranks
    if mpicomm.rank != mpiroot:
        np_dtype = np.dtype((dtype, shape[1:]))
        data = np.empty(0, dtype=np_dtype)

    # setup the custom dtype
    duplicity = np.prod(shape[1:], dtype='intp')
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    # compute the new shape for each rank
    newshape = list(shape)

    if counts is None:
        newshape[0] = newlength = local_size(shape[0], mpicomm=mpicomm)
    else:
        if counts.sum() != shape[0]:
            raise ValueError('the sum of the `counts` array needs to be equal to data length')
        newshape[0] = counts[mpicomm.rank]

    # the return array
    recvbuffer = np.empty(newshape, dtype=dtype, order='C')

    # the send counts, if not provided
    if counts is None:
        counts = mpicomm.allgather(newlength)
        counts = np.array(counts, order='C')

    # the send offsets
    offsets = np.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    # do the scatter
    mpicomm.Barrier()
    mpicomm.Scatterv([data, (counts, offsets), dt], [recvbuffer, dt], root=mpiroot)
    dt.Free()
    return recvbuffer


@CurrentMPIComm.enable
def send(data, dest, tag=0, mpicomm=None):
    """
    Send input array ``data`` to process ``dest``.

    Parameters
    ----------
    data : array
        Array to send.

    dest : int
        Rank of process to send array to.

    tag : int, default=0
        Message identifier.

    mpicomm : MPI communicator, default=None
        Communicator. Defaults to current communicator.
    """
    data = np.asarray(data)
    shape, dtype = (data.shape, data.dtype)
    data = np.ascontiguousarray(data)

    fail = False
    if dtype.char == 'V':
        fail = any(dtype[name] == 'O' for name in dtype.names)
    else:
        fail = dtype == 'O'
    if fail:
        raise ValueError('"object" data type not supported in send; please specify specific data type')

    duplicity = np.prod(shape[1:], dtype='intp')
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    mpicomm.send((shape, dtype), dest=dest, tag=tag)
    mpicomm.Send([data, dt], dest=dest, tag=tag)


@CurrentMPIComm.enable
def recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, mpicomm=None):
    """
    Receive array from process ``source``.

    Parameters
    ----------
    source : int, default=MPI.ANY_SOURCE
        Rank of process to receive array from.

    tag : int, default=MPI.ANY_TAG
        Message identifier.

    mpicomm : MPI communicator, default=None
        Communicator. Defaults to current communicator.

    Returns
    -------
    data : array
    """
    shape, dtype = mpicomm.recv(source=source, tag=tag)
    data = np.zeros(shape, dtype=dtype)

    duplicity = np.prod(shape[1:], dtype='intp')
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    mpicomm.Recv([data, dt], source=source, tag=tag)
    return data


@CurrentMPIComm.enable
def sendrecv(data, source=0, dest=0, tag=0, mpicomm=None):
    """Send array from process ``source`` and receive on process ``dest``."""
    if dest == source:
        return np.asarray(data)
    if mpicomm.rank == source:
        send(data, dest=dest, tag=tag, mpicomm=mpicomm)
    toret = None
    if mpicomm.rank == dest:
        toret = recv(source=source, tag=tag, mpicomm=mpicomm)
    return toret


@CurrentMPIComm.enable
def local_size(size, mpicomm=None):
    """
    Divide global ``size`` into local (process) size.

    Parameters
    ----------
    size : int
        Global size.

    mpicomm : MPI communicator, default=None
        Communicator. Defaults to current communicator.

    Returns
    -------
    localsize : int
        Local size. Sum of local sizes over all processes equals global size.
    """
    start = mpicomm.rank * size // mpicomm.size
    stop = (mpicomm.rank + 1) * size // mpicomm.size
    # localsize = size // mpicomm.size
    # if mpicomm.rank < size % mpicomm.size: localsize += 1
    return stop - start


@CurrentMPIComm.enable
def creshape(arr, cshape, mpicomm=None):
    """
    Set collective shape attr:`cshape`, e.g.:
    >>> arr.creshape(100, 2)

    This will induce data exchange between various processes if local size cannot be divided by ``prod(cshape[1:])``.
    """
    if np.ndim(cshape) == 0:
        cshape = (cshape,)
    cshape = tuple(cshape)
    unknown = [s < 0 for s in cshape]
    if sum(unknown) > 1:
        raise ValueError('can only specify one unknown dimension')
    arr = np.asanyarray(arr)
    csize_arr = csize(arr, mpicomm=mpicomm)
    if sum(unknown) == 1:
        size = csize_arr // np.prod([s for s in cshape if s >= 0], dtype='intp')
        cshape = tuple(s if s >= 0 else size for s in cshape)
    csize_shape = np.prod(cshape, dtype='intp')
    if csize_shape != csize_arr:
        raise ValueError('cannot reshape arr of size {:d} into shape {}'.format(csize_arr, cshape))
    itemsize = np.prod(cshape[1:], dtype='intp')
    local_slice = Slice(mpicomm.rank * cshape[0] // mpicomm.size * itemsize, (mpicomm.rank + 1) * cshape[0] // mpicomm.size * itemsize, 1)
    new = arr
    if not all(mpicomm.allgather(local_slice.size == arr.size)):  # need to reorder!
        cumsizes = np.cumsum([0] + mpicomm.allgather(arr.size))
        source = MPIScatteredSource(slice(cumsizes[mpicomm.rank], cumsizes[mpicomm.rank + 1], 1))
        new = source.get(arr.ravel(), local_slice).view(new.__class__)
        if isinstance(new, array): new.mpicomm = mpicomm
    return new.reshape((-1,) + cshape[1:])


@CurrentMPIComm.enable
def cslice(arr, *args, mpicomm=None):
    """Perform collective arr slice, e.g.:
    >>> arr.cslice(0, 10, 2)
    >>> arr.cslice([1, 2, 2])
    """
    arr = np.asanyarray(arr)
    cumsizes = np.cumsum([0] + mpicomm.allgather(len(arr)))
    global_slice = Slice(*args, size=cumsizes[-1])
    local_slice = global_slice.split(mpicomm.size)[mpicomm.rank]
    source = MPIScatteredSource(slice(cumsizes[mpicomm.rank], cumsizes[mpicomm.rank + 1], 1))
    new = source.get(arr, local_slice).view(arr.__class__)
    if isinstance(new, array): new.mpicomm = mpicomm
    return new


@CurrentMPIComm.enable
def cconcatenate(*others, axis=0, mpicomm=None):
    """Concatenate input :class:`array`, collectively, i.e. preserving order accross all ranks."""
    if not others:
        raise ValueError('Provide at least one array.')
    if len(others) == 1 and is_sequence(others[0]):
        others = others[0]

    axis = normalize_axis_tuple(axis, others[0].ndim)[0]
    if axis > 1:
        value = np.concatenate([other for other in others], axis=axis)
    else:
        source = []
        for other in others:
            cumsizes = np.cumsum([0] + other.mpicomm.allgather(len(other)))
            source.append(MPIScatteredSource(slice(cumsizes[other.mpicomm.rank], cumsizes[other.mpicomm.rank + 1], 1)))
        source = MPIScatteredSource.cconcatenate(*source)
        value = source.get([other for other in others])

    mpicomm = None
    for other in others:
        if isinstance(other, array):
            mpicomm = other.mpicomm
            break
    if mpicomm is not None:
        value = value.view(array)
        value.mpicomm = mpicomm
    return value


def cappend(array, other, **kwargs):
    """Append ``other`` to ``array``."""
    return cconcatenate([array, other], **kwargs)


@CurrentMPIComm.enable
def csize(data, mpicomm=None):
    """Return global size of ``data`` array."""
    return mpicomm.allreduce(np.size(data), op=MPI.SUM)


@CurrentMPIComm.enable
def cshape(data, mpicomm=None):
    """Return global shape of ``data`` array (scattered along the first dimension)."""
    shape = np.shape(data)
    return (mpicomm.allreduce(shape[0], op=MPI.SUM),) + shape[1:]


def _reduce_op_array(data, npop, mpiop, *args, mpicomm=None, axis=None, **kwargs):
    """
    Apply operation ``npop`` on input array ``data`` and reduce the result
    with MPI operation ``mpiop``(e.g. sum).

    Parameters
    ----------
    data : array
        Input array to reduce with operations ``npop`` and ``mpiop``.

    npop : callable
        Function that takes ``data``, ``args``, ``axis`` and ``kwargs`` as argument,
        and keyword arguments and return (array) value.

    mpiop : MPI operation
        MPI operation to apply on ``npop`` result.

    mpicomm : MPI communicator
        Communicator. Defaults to current communicator.

    axis : int, list, default=None
        Array axis (axes) on which to apply operations.
        If ``0`` not in ``axis``, ``mpiop`` is not used.
        Defaults to all axes.

    Returns
    -------
    out : scalar, array
        Result of reduce operations ``npop`` and ``mpiop``.
        If ``0`` in ``axis``, result is broadcast on all ranks.
        Else, result is local.
    """
    toret = npop(data, *args, axis=axis, **kwargs)
    if axis is None: axis = tuple(range(data.ndim))
    else: axis = normalize_axis_tuple(axis, data.ndim)
    if 0 in axis:
        return reduce(toret, mpicomm=mpicomm, op=mpiop, mpiroot=Ellipsis)
    return toret


def _reduce_arg_array(data, npop, mpiargop, mpiop, *args, mpicomm=None, axis=None, **kwargs):
    """
    Apply operation ``npop`` on input array ``data`` and reduce the result
    with MPI operation ``mpiargop``.
    Contrary to :func:`_reduce_op_array`, ``npop`` is expected to return index in array.
    (e.g. index of minimum).

    Parameters
    ----------
    data : array
        Input array to reduce with operations ``npop`` and ``mpiop``.

    npop : callable
        Function that takes ``data``, ``args``, ``axis`` and ``kwargs`` as argument,
        and keyword arguments, and returns array index.

    mpiargop : MPI operation
        MPI operation to select index returned by ``npop`` among all processes
        (takes as input ``(value, rank)`` with ``value`` array value at index returned by ``npop``).

    mpiop : MPI operation
        MPI operation to apply on array value at index returned by ``npop``.

    mpicomm : MPI communicator
        Communicator. Defaults to current communicator.

    axis : int, list, default=None
        Array axis (axes) on which to apply operations.
        If ``0`` not in ``axis``, ``mpiop`` is not used.
        Defaults to all axes.

    Returns
    -------
    arg : scalar, array
        If ``0`` in ``axis``, index in global array; result is broadcast on all ranks.
        Else, result is local.

    rank : int, None
        If ``0`` in ``axis``, rank where index resides in.
        Else, ``None``.
    """
    arg = npop(data, *args, axis=axis, **kwargs)
    if axis is None:
        val = data[np.unravel_index(arg, data.shape)]
    else:
        val = np.take_along_axis(data, np.expand_dims(arg, axis=axis), axis=axis)[0]
    # could not find out how to do mpicomm.Allreduce([tmp,MPI.INT_INT],[total,MPI.INT_INT],op=MPI.MINLOC) for e.g. (double,int)...
    if axis is None: axis = tuple(range(data.ndim))
    else: axis = normalize_axis_tuple(axis, data.ndim)
    if 0 in axis:
        if np.isscalar(arg):
            rank = mpicomm.allreduce((val, mpicomm.rank), op=mpiargop)[1]
            arg = mpicomm.bcast(arg, root=rank)
            return arg, rank
        # raise NotImplementedError('MPI argmin/argmax with non-scalar output is not implemented.')
        total = np.empty_like(val)
        # first decide from which rank we get the solution
        mpicomm.Allreduce(val, total, op=mpiop)
        mask = val == total
        rank = np.ones_like(arg) + mpicomm.size
        rank[mask] = mpicomm.rank
        totalrank = np.empty_like(rank)
        mpicomm.Allreduce(rank, totalrank, op=MPI.MIN)
        # f.. then fill in argmin
        mask = totalrank == mpicomm.rank
        tmparg = np.zeros_like(arg)
        tmparg[mask] = arg[mask]
        # print(mpicomm.rank,arg,mask)
        totalarg = np.empty_like(tmparg)
        mpicomm.Allreduce(tmparg, totalarg, op=MPI.SUM)
        return totalarg, totalrank

    return arg, None


@CurrentMPIComm.enable
def csum(data, *args, mpicomm=None, axis=None, **kwargs):
    """Return sum of input array ``data`` along ``axis``."""
    return _reduce_op_array(data, np.sum, MPI.SUM, *args, mpicomm=mpicomm, axis=axis, **kwargs)


@CurrentMPIComm.enable
def cmean(data, *args, mpicomm=None, axis=-1, **kwargs):
    """Return mean of array ``data`` along ``axis``."""
    if axis is None: axis = tuple(range(data.ndim))
    else: axis = normalize_axis_tuple(axis, data.ndim)
    if 0 not in axis:
        toret = np.mean(data, *args, axis=axis, **kwargs)
    else:
        toret = csum(data, *args, mpicomm=mpicomm, axis=axis, **kwargs)
        N = csize(data, mpicomm=mpicomm)
        if toret.size: N = N / toret.size
        toret = toret / N
    return toret


@CurrentMPIComm.enable
def cprod(data, *args, mpicomm=None, axis=None, **kwargs):
    """Return product of input array ``data`` along ``axis``."""
    return _reduce_op_array(data, np.prod, MPI.PROD, *args, mpicomm=mpicomm, axis=axis, **kwargs)


@CurrentMPIComm.enable
def cmin(data, *args, mpicomm=None, axis=None, **kwargs):
    """Return minimum of input array ``data`` along ``axis``."""
    return _reduce_op_array(data, np.min, MPI.MIN, *args, mpicomm=mpicomm, axis=axis, **kwargs)


@CurrentMPIComm.enable
def cmax(data, *args, mpicomm=None, axis=None, **kwargs):
    """Return maximum of input array ``data`` along ``axis``."""
    return _reduce_op_array(data, np.max, MPI.MAX, *args, mpicomm=mpicomm, axis=axis, **kwargs)


@CurrentMPIComm.enable
def cargmin(data, *args, mpicomm=None, axis=None, **kwargs):
    """Return (local) index and rank of minimum in input array ``data`` along ``axis``."""
    return _reduce_arg_array(data, np.argmin, MPI.MINLOC, MPI.MIN, *args, mpicomm=mpicomm, axis=axis, **kwargs)


@CurrentMPIComm.enable
def cargmax(data, *args, return_rank=False, mpicomm=None, axis=None, **kwargs):
    """Return (local) index and rank of maximum in input array ``data`` along ``axis``."""
    return _reduce_arg_array(data, np.argmax, MPI.MAXLOC, MPI.MAX, *args, mpicomm=mpicomm, axis=axis, **kwargs)


@CurrentMPIComm.enable
def csort(data, axis=-1, kind=None, mpicomm=None):
    """
    Sort input array ``data`` along ``axis``.
    Naive implementation: array is gathered, sorted, and scattered again.
    Faster than naive distributed sorts (bitonic, transposition)...

    Parameters
    ----------
    data : array
        Array to be sorted.

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
    toret = np.sort(data, axis=axis, kind=kind)
    if mpicomm.size == 1:
        return toret
    if axis is None:
        data = data.flat
        axis = 0
    else:
        axis = normalize_axis_tuple(axis, data.ndim)[0]
    if axis != 0:
        return toret

    counts = mpicomm.allgather(len(toret))
    gathered = gather(toret, mpiroot=0, mpicomm=mpicomm)
    toret = None
    if mpicomm.rank == 0:
        toret = np.sort(gathered, axis=axis, kind=kind)
    return scatter(toret, mpiroot=0, counts=counts, mpicomm=mpicomm)


@CurrentMPIComm.enable
def cquantile(data, q, weights=None, axis=None, interpolation='linear', mpicomm=None):
    """
    Return weighted array quantiles. See :func:`utils.weighted_quantile`.
    Naive implementation: array is gathered before taking quantile.
    """
    from . import utils
    if axis is None or 0 in normalize_axis_tuple(axis, data.ndim):
        gathered = gather(data, mpiroot=0, mpicomm=mpicomm)
        isnoneweights = all(mpicomm.allgather(weights is None))
        if not isnoneweights: weights = gather(weights, mpiroot=0, mpicomm=mpicomm)
        toret = None
        if mpicomm.rank == 0:
            toret = utils.weighted_quantile(gathered, q, weights=weights, axis=axis, interpolation=interpolation)
        return bcast(toret, mpiroot=0, mpicomm=mpicomm)
    return utils.weighted_quantile(data, q, weights=weights, axis=axis, interpolation=interpolation)


@CurrentMPIComm.enable
def cdot(a, b, mpicomm=None):
    """
    Return dot product of input arrays ``a`` and ``b``.
    Currently accepts one-dimensional ``b`` or two-dimensional ``a`` and ``b``.
    ``b`` must be scattered along first axis, hence ``a`` scattered along last axis.
    """
    # scatter axis is b first axis
    if b.ndim == 1:
        return csum(a * b, mpicomm=mpicomm)
    if a.ndim == b.ndim == 2:
        return csum(np.dot(a, b)[None, ...], axis=0, mpicomm=mpicomm)
    raise NotImplementedError


@CurrentMPIComm.enable
def caverage(a, axis=None, weights=None, returned=False, mpicomm=None):
    """
    Return weighted average of input array ``a`` along axis ``axis``.
    See :func:`numpy.average`.
    TODO: allow several axes.
    """
    if axis is None: axis = tuple(range(a.ndim))
    else: axis = normalize_axis_tuple(axis, a.ndim)
    if 0 not in axis:
        return np.average(a, axis=axis, weights=weights, returned=returned)
    axis = axis[0]

    a = np.asanyarray(a)

    if weights is None:
        avg = cmean(a, axis=axis, mpicomm=mpicomm)
        scl = avg.dtype.type(csize(a) / avg.size)
    else:
        wgt = np.asanyarray(weights)

        if issubclass(a.dtype.type, (np.integer, np.bool_)):
            result_dtype = np.result_type(a.dtype, wgt.dtype, 'f8')
        else:
            result_dtype = np.result_type(a.dtype, wgt.dtype)

        # Sanity checks
        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights "
                    "differ.")
            if wgt.ndim != 1:
                raise TypeError(
                    "1D weights expected when shapes of a and weights differ.")
            if wgt.shape[0] != a.shape[axis]:
                raise ValueError(
                    "Length of weights not compatible with specified axis.")

            # setup wgt to broadcast along axis
            wgt = np.broadcast_to(wgt, (a.ndim - 1) * (1,) + wgt.shape)
            wgt = wgt.swapaxes(-1, axis)

        scl = csum(wgt, axis=axis, dtype=result_dtype)
        if np.any(scl == 0.0):
            raise ZeroDivisionError(
                "Weights sum to zero, can't be normalized")

        avg = csum(np.multiply(a, wgt, dtype=result_dtype), axis=axis) / scl

    if returned:
        if scl.shape != avg.shape:
            scl = np.broadcast_to(scl, avg.shape).copy()
        return avg, scl
    else:
        return avg


@CurrentMPIComm.enable
def cvar(a, axis=-1, fweights=None, aweights=None, ddof=0, mpicomm=None):
    """
    Estimate variance, given data and weights.
    See :func:`numpy.var`.
    TODO: allow several axes.

    Parameters
    ----------
    a : array
        Array containing numbers whose variance is desired.
        If a is not an array, a conversion is attempted.

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

    ddof : int, default=0
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
    X = np.array(a)
    w = None
    if fweights is not None:
        fweights = np.asarray(fweights, dtype=float)
        if not np.all(fweights == np.around(fweights)):
            raise TypeError(
                "fweights must be integer")
        if fweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional fweights")
        if fweights.shape[0] != X.shape[axis]:
            raise RuntimeError(
                "incompatible numbers of samples and fweights")
        if any(fweights < 0):
            raise ValueError(
                "fweights cannot be negative")
        w = fweights
    if aweights is not None:
        aweights = np.asarray(aweights, dtype=float)
        if aweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional aweights")
        if aweights.shape[0] != X.shape[axis]:
            raise RuntimeError(
                "incompatible numbers of samples and aweights")
        if any(aweights < 0):
            raise ValueError(
                "aweights cannot be negative")
        if w is None:
            w = aweights
        else:
            w *= aweights

    avg, w_sum = caverage(X, axis=axis, weights=w, returned=True, mpicomm=mpicomm)

    # Determine the normalization
    if w is None:
        fact = cshape(a)[axis] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * csum(w * aweights, axis=axis, mpicomm=mpicomm) / w_sum

    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=3)
        fact = 0.0

    X = np.apply_along_axis(lambda x: x - avg, axis, X)
    if w is None:
        X_T = X
    else:
        X_T = (X * w)
    c = csum(X * X_T.conj(), axis=axis, mpicomm=mpicomm)
    c *= np.true_divide(1, fact)
    return c.squeeze()


@CurrentMPIComm.enable
def cstd(*args, **kwargs):
    """
    Return weighted standard deviation of input array along axis ``axis``.
    Simply take square root of :func:`cvar` result.
    TODO: allow for several axes.
    """
    return np.sqrt(cvar(*args, **kwargs))


@CurrentMPIComm.enable
def ccov(m, y=None, ddof=1, rowvar=True, fweights=None, aweights=None, dtype=None, mpicomm=None):
    """
    Estimate a covariance matrix, given data and weights.
    See :func:`numpy.cov`.

    Parameters
    ----------
    m : array
        A 1D or 2D array containing multiple variables and observations.
        Each row of ``m`` represents a variable, and each column a single
        observation of all those variables. Also see ``rowvar`` below.

    y : array, default=None
        An additional set of variables and observations. ``y`` has the same form
        as that of ``m``.

    rowvar : bool, default=True
        If ``rowvar`` is ``True`` (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.

    fweights : array, int, default=None
        1D array of integer frequency weights; the number of times each
        observation vector should be repeated.

    aweights : array, default=None
        1D array of observation vector weights. These relative weights are
        typically large for observations considered "important" and smaller for
        observations considered less "important". If ``ddof=0`` the array of
        weights can be used to assign probabilities to observation vectors.

    ddof : int, default=1
        Number of degrees of freedom.
        Note that ``ddof=1`` will return the unbiased estimate, even if both
        ``fweights`` and `aweights` are specified, and ``ddof=0`` will return
        the simple average.

    dtype : data-type, default=None
        Data-type of the result. By default, the return data-type will have
        at least ``numpy.float64`` precision.

    mpicomm : MPI communicator
        Current MPI communicator.

    Returns
    -------
    out : array
        The covariance matrix of the variables.
    """
    # scatter axis is data second axis
    # data (nobs, ndim)
    # Check inputs
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be integer")

    # Handles complex arrays too
    m = np.asarray(m)
    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")

    if y is not None:
        y = np.asarray(y)
        if y.ndim > 2:
            raise ValueError("y has more than 2 dimensions")

    if dtype is None:
        if y is None:
            dtype = np.result_type(m, np.float64)
        else:
            dtype = np.result_type(m, y, np.float64)

    X = np.array(m, ndmin=2, dtype=dtype)
    if not rowvar and X.shape[0] != 1:
        X = X.T
    if X.shape[0] == 0:
        return np.array([]).reshape(0, 0)
    if y is not None:
        y = np.array(y, copy=False, ndmin=2, dtype=dtype)
        if not rowvar and y.shape[0] != 1:
            y = y.T
        X = np.concatenate((X, y), axis=0)

    # Get the product of frequencies and weights
    w = None
    if fweights is not None:
        fweights = np.asarray(fweights, dtype=float)
        if not np.all(fweights == np.around(fweights)):
            raise TypeError(
                "fweights must be integer")
        if fweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional fweights")
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and fweights")
        if any(fweights < 0):
            raise ValueError(
                "fweights cannot be negative")
        w = fweights
    if aweights is not None:
        aweights = np.asarray(aweights, dtype=float)
        if aweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional aweights")
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and aweights")
        if any(aweights < 0):
            raise ValueError(
                "aweights cannot be negative")
        if w is None:
            w = aweights
        else:
            w *= aweights

    avg, w_sum = caverage(X.T, axis=0, weights=w, returned=True, mpicomm=mpicomm)
    w_sum = w_sum[0]

    # Determine the normalization
    if w is None:
        fact = cshape(X.T)[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * csum(w * aweights, mpicomm=mpicomm) / w_sum

    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=3)
        fact = 0.0

    X -= avg[:, None]
    if w is None:
        X_T = X.T
    else:
        X_T = (X * w).T
    c = cdot(X, X_T.conj(), mpicomm=mpicomm)
    c *= np.true_divide(1, fact)
    return c.squeeze()


@CurrentMPIComm.enable
def ccorrcoef(x, y=None, rowvar=True, fweights=None, aweights=None, dtype=None, mpicomm=None):
    """
    Return weighted correlation matrix of input arrays ``m`` (``y``).
    See :func:`cov_array`.
    """
    c = ccov(x, y, rowvar, fweights=None, aweights=None, dtype=dtype, mpicomm=mpicomm)
    try:
        d = np.diag(c)
    except ValueError:
        # scalar covariance
        # nan if incorrect value (nan, inf, 0), 1 otherwise
        return c / c
    stddev = np.sqrt(d.real)
    c /= stddev[:, None]
    c /= stddev[None, :]

    # Clip real and imaginary parts to [-1, 1].  This does not guarantee
    # abs(a[i,j]) <= 1 for complex arrays, but is the best we can do without
    # excessive work.
    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):
        np.clip(c.imag, -1, 1, out=c.imag)

    return c
