import numbers
import math

import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.mixins import NDArrayOperatorsMixin as NDArrayLike

from . import mpi, utils
from .mpi import MPI, CurrentMPIComm
from .utils import BaseClass, BaseMetaClass, is_sequence


class Slice(BaseClass):

    def __init__(self, *args, size=None, copy=False):
        if len(args) > 1:
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
        return not isinstance(self.idx, slice)

    def to_array(self, copy=False):
        if self.is_array:
            return np.array(self.idx, copy=copy)
        return np.arange(self.idx.start, self.idx.stop, self.idx.step)

    def to_slices(self):
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

    @classmethod
    def empty(cls):
        return cls(slice(0, 0, 1))

    def __repr__(self):
        if self.is_array:
            return '{}({})'.format(self.__class__.__name__, self.idx)
        return '{}({}, {}, {})'.format(self.__class__.__name__, self.idx.start, self.idx.stop, self.idx.step)

    def slice(self, *args, return_index=False):
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
        if self.is_array:
            return self.idx.size
        return ((-1 if self.idx.stop is None else self.idx.stop) - self.idx.start + (-1)**(self.idx.step > 0)) // self.idx.step + 1

    @property
    def size(self):
        """Equivalent for :meth:`__len__`."""
        return len(self)

    @property
    def min(self):
        if self.is_array:
            return self.idx.min()
        if self.idx.step < 0:
            return self.idx.step * (self.size - 1) + self.idx.start
        return self.idx.start

    @property
    def max(self):
        if self.is_array:
            return self.idx.max()
        if self.idx.step > 0:
            return self.idx.step * (self.size - 1) + self.idx.start
        return self.idx.start

    def __eq__(self, other):
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
        if mpicomm.recv(source=source, tag=tag + 1):  # is_array
            idx = mpi.recv_array(source=source, tag=tag, mpicomm=mpicomm)
        else:
            idx = mpicomm.recv(source=source, tag=tag)
        return cls(idx)


class MPIScatteredSource(BaseClass):

    @CurrentMPIComm.enable
    def __init__(self, *slices, csize=None, mpicomm=None):
        # let's restrict to disjoint slices...
        self.mpicomm = mpicomm
        self.slices = [Slice(sl, size=csize) for sl in slices]
        if any(sl.is_array for sl in self.slices):
            raise NotImplementedError('Only slices supported so far')
        if csize is not None and csize != self.csize:
            raise ValueError('Input slices do not have collective size "csize"')

    @property
    def size(self):
        if getattr(self, '_size', None) is None:
            self._size = sum(sl.size for sl in self.slices)
        return self._size

    @property
    def csize(self):
        return self.mpicomm.allreduce(self.size)

    def get(self, arrays, *args):
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
    def concatenate(cls, *others):
        if not others:
            raise ValueError('Provide at least one {} instance.'.format(cls.__name__))
        if len(others) == 1 and is_sequence(others[0]):
            others = others[0]
        slices, cumsize = [], 0
        for other in others:
            slices += [sl.shift(cumsize) for sl in other.slices]
            cumsize += other.csize
        return cls(*slices)

    def extend(self, other, **kwargs):
        new = self.concatenate(self, other, **kwargs)
        self.__dict__.update(new.__dict__)


class MPIScatteredArray(NDArrayLike, BaseClass, metaclass=BaseMetaClass):

    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    @CurrentMPIComm.enable
    def __init__(self, value=None, copy=False, dtype=None, mpicomm=None, mpiroot=None):
        """Initalize :class:`MPIScatteredArray`."""
        self.mpicomm = mpicomm
        if mpiroot is None or self.mpicomm.rank == mpiroot:
            value = np.array(value, copy=copy, dtype=dtype)
        if mpiroot is not None:
            value = mpi.scatter_array(value, mpicomm=mpicomm, root=mpiroot)
        self.value = value

    def __mul__(self, other):
        r = self.copy()
        r.value = r.value * other
        return r

    def __imul__(self, other):
        self.value *= other
        return self

    def __div__(self, other):
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
        return self.value.__getitem__(index)

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
        return self.mpicomm.allreduce(self.value.size)

    @property
    def cshape(self):
        return (self.mpicomm.allreduce(self.value.shape[0]),) + self.value.shape[1:]

    def __len__(self):
        return len(self.value)

    @cshape.setter
    def cshape(self, cshape):
        if np.ndim(cshape) == 0:
            cshape = (cshape,)
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
        new = self.copy()
        new.value = self.value[Slice(*args, size=self.size).idx]
        return new

    def cslice(self, *args):
        new = self.copy()
        cumsizes = np.cumsum([0] + self.mpicomm.allgather(len(self)))
        global_slice = Slice(*args, size=cumsizes[-1])
        local_slice = global_slice.split(self.mpicomm.size)[self.mpicomm.rank]
        source = MPIScatteredSource(slice(cumsizes[self.mpicomm.rank], cumsizes[self.mpicomm.rank + 1], 1))
        new.value = source.get(self.value, local_slice)
        return new

    @classmethod
    def concatenate(cls, *others, axis=0):
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
            source = MPIScatteredSource.concatenate(*source)
            new.value = source.get([other.value for other in others])
        return new

    def extend(self, other, **kwargs):
        """Extend catalog with ``other``."""
        return self.concatenate(self, [other], **kwargs)

    def append(self, other, **kwargs):
        """Extend catalog with ``other``."""
        new = self.extend(self, other, **kwargs)
        self.__dict__.update(new.__dict__)

    def gathered(self, mpiroot=0):
        return mpi.gather_array(self.value, mpicomm=self.mpicomm, root=mpiroot)

    def csort(self, axis=0, kind=None):
        # import mpsort
        # self.value = mpsort.sort(self.value, orderby=None, comm=self.mpicomm, tuning=[])
        self.value = mpi.sort_array(self.value, axis=axis, kind=kind, mpicomm=self.mpicomm)  # most naive implementation

    def csum(self, axis=0):
        return mpi.sum_array(self.value, axis=axis, mpicomm=self.mpicomm)

    def caverage(self, weights=None, axis=0):
        return mpi.average_array(self.value, weights=weights, axis=axis, mpicomm=self.mpicomm)

    def cmean(self, axis=0):
        return self.caverage(axis=axis)

    def cvar(self, axis=0, fweights=None, aweights=None, ddof=1):
        return mpi.var_array(self.value, axis=axis, fweights=fweights, aweights=aweights, ddof=ddof, mpicomm=self.mpicomm)

    def cstd(self, axis=0, fweights=None, aweights=None, ddof=1):
        return mpi.std_array(self.value, axis=axis, fweights=fweights, aweights=aweights, ddof=ddof, mpicomm=self.mpicomm)

    def cmin(self, axis=0):
        return mpi.min_array(self.value, axis=axis, mpicomm=self.mpicomm)

    def cmax(self, axis=0):
        """Return global maximum of column(s) ``column``."""
        return mpi.max_array(self.value, axis=axis, mpicomm=self.mpicomm)

    def cargmin(self, axis=0):
        return mpi.argmin_array(self.value, axis=axis, mpicomm=self.mpicomm)

    def cargmax(self, axis=0):
        """Return global maximum of column(s) ``column``."""
        return mpi.argmax_array(self.value, axis=axis, mpicomm=self.mpicomm)

    def cquantile(self, q, weights=None, axis=0, interpolation='linear'):
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
