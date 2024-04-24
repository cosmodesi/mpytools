"""
MPI routines, many taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/__init__.py
and https://github.com/bccp/nbodykit/blob/master/nbodykit/batch.py.
"""
import random

import numpy as np
from mpi4py import MPI

from .utils import CurrentMPIComm


__all__ = ['bcast_seed', 'set_common_seed', 'set_independent_seed', 'MPIRandomState']


@CurrentMPIComm.enable
def bcast_seed(seed=None, mpicomm=None, size=None):
    """
    Generate array of seeds.

    Parameters
    ---------
    seed : int, default=None
        Random seed to use when generating seeds.

    mpicomm : MPI communicator, default=None
        Communicator to use for broadcasting. Defaults to current communicator.

    size : int, default=None
        Number of seeds to be generated.

    Returns
    -------
    seeds : array
        Array of seeds.
    """
    if mpicomm.rank == 0:
        seeds = np.random.RandomState(seed=seed).randint(0, high=0xffffffff, size=size)
    from . import core
    return core.bcast(seeds if mpicomm.rank == 0 else None, mpiroot=0, mpicomm=mpicomm)


@CurrentMPIComm.enable
def set_common_seed(seed=None, mpicomm=None):
    """
    Set same global :mod:`np.random` and :mod:`random` seed for all MPI processes.

    Parameters
    ----------
    seed : int, default=None
        Random seed to broadcast on all processes.
        If ``None``, draw random seed.

    mpicomm : MPI communicator, default=None
        Communicator to use for broadcasting. Defaults to current communicator.

    Returns
    -------
    seed : int
        Seed used to initialize :mod:`np.random` and :mod:`random` global states.
    """
    if seed is None:
        if mpicomm.rank == 0:
            seed = np.random.randint(0, high=0xffffffff)
    seed = mpicomm.bcast(seed, root=0)
    np.random.seed(seed)
    random.seed(seed)
    return seed


@CurrentMPIComm.enable
def set_independent_seed(seed=None, mpicomm=None, size=10000):
    """
    Set independent global :mod:`np.random` and :mod:`random` seed for all MPI processes.

    Parameters
    ---------
    seed : int, default=None
        Random seed to use when generating seeds.

    mpicomm : MPI communicator, default=None
        Communicator to use for broadcasting. Defaults to current communicator.

    size : int, default=10000
        Number of seeds to be generated.
        To ensure random draws are independent of the number of ranks,
        this should be larger than the total number of processes that will ever be used.

    Returns
    -------
    seed : int
        Seed used to initialize :mod:`np.random` and :mod:`random` global states.
    """
    seed = bcast_seed(seed=seed, mpicomm=mpicomm, size=size)[mpicomm.rank]
    np.random.seed(seed)
    random.seed(seed)
    return seed


@CurrentMPIComm.enable
def front_pad_array(array, front, mpicomm=None):
    """
    Pad an array in the front with items before this rank.

    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py
    """
    N = np.array(mpicomm.allgather(len(array)), dtype='intp')
    offsets = np.cumsum(np.concatenate([[0], N], axis=0))
    mystart = offsets[mpicomm.rank] - front
    torecv = (offsets[:-1] + N) - mystart

    torecv[torecv < 0] = 0  # before mystart
    torecv[torecv > front] = 0  # no more than needed
    torecv[torecv > N] = N[torecv > N]  # fully enclosed

    if mpicomm.allreduce(torecv.sum() != front, MPI.LOR):
        raise ValueError('Cannot work out a plan to padd items. Some front values are too large. {:d} {:d}'.format(torecv.sum(), front))

    tosend = mpicomm.alltoall(torecv)
    sendbuf = [array[-items:] if items > 0 else array[0:0] for i, items in enumerate(tosend)]
    recvbuf = mpicomm.alltoall(sendbuf)
    return np.concatenate(list(recvbuf) + [array], axis=0)


class MPIRandomState(object):
    """
    A random number generator that is invariant against number of ranks,
    when the total size of random number requested is kept the same.
    The algorithm here assumes the random number generator from numpy
    produces uncorrelated results when the seeds are sampled from a single
    random generator.
    The sampler methods are collective calls; multiple calls will return
    uncorrelated results.
    The result is only invariant under different ``mpicomm.size`` when ``mpicomm.allreduce(size)``
    and ``chunksize`` are kept invariant.

    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/mpirng.py.
    """
    @CurrentMPIComm.enable
    def __init__(self, size, seed=None, chunksize=100000, mpicomm=None):
        self.mpicomm = mpicomm
        self.chunksize = chunksize

        self.size = size
        self.csize = np.sum(mpicomm.allgather(size), dtype='intp')

        self._start = np.sum(mpicomm.allgather(size)[:mpicomm.rank], dtype='intp')
        self._end = self._start + self.size

        self._first_ichunk = self._start // chunksize

        self._skip = self._start - self._first_ichunk * chunksize

        nchunks = (mpicomm.allreduce(np.array(size, dtype='intp')) + chunksize - 1) // chunksize
        self.nchunks = nchunks

        self._serial_rng = np.random.RandomState(seed)

    def _prepare_args_and_result(self, args, itemshape, dtype):
        """
        Pad every item in args with values from previous ranks,
        and create an array for holding the result with the same length.

        Returns
        -------
        padded_r, padded_args
        """
        r = np.zeros((self.size,) + tuple(itemshape), dtype=dtype)

        r_and_args = (r,) + tuple(args)
        r_and_args_b = np.broadcast_arrays(*r_and_args)

        padded = []

        # we don't need to pad scalars,
        # loop over broadcasted and non broadcasted version to figure this out)
        for a, a_b in zip(r_and_args, r_and_args_b):
            if np.ndim(a) == 0:
                # use the scalar, no need to pad.
                padded.append(a)
            else:
                # not a scalar, pad
                padded.append(front_pad_array(a_b, self._skip, mpicomm=self.mpicomm))

        return padded[0], padded[1:]

    def poisson(self, lam, itemshape=(), dtype='f8'):
        """Produce :attr:`size` poissons, each of shape itemshape. This is a collective MPI call."""
        def sampler(rng, args, size):
            lam, = args
            return rng.poisson(lam=lam, size=size)
        return self._call_rngmethod(sampler, (lam,), itemshape, dtype)

    def normal(self, loc=0, scale=1, itemshape=(), dtype='f8'):
        """Produce :attr:`size` normals, each of shape itemshape. This is a collective MPI call."""
        def sampler(rng, args, size):
            loc, scale = args
            return rng.normal(loc=loc, scale=scale, size=size)
        return self._call_rngmethod(sampler, (loc, scale), itemshape, dtype)

    def uniform(self, low=0., high=1.0, itemshape=(), dtype='f8'):
        """Produce :attr:`size` uniforms, each of shape itemshape. This is a collective MPI call."""
        def sampler(rng, args, size):
            low, high = args
            return rng.uniform(low=low, high=high, size=size)
        return self._call_rngmethod(sampler, (low, high), itemshape, dtype)

    def randint(self, low=0, high=10, itemshape=(), dtype='f8'):
        """Produce :attr:`size` randint, each of shape itemshape. This is a collective MPI call."""
        def sampler(rng, args, size):
            low, high = args
            return rng.randint(low=low, high=high, size=size)
        return self._call_rngmethod(sampler, (low, high), itemshape, dtype)

    def choice(self, choices, itemshape=(), p=None):
        """Produce :attr:`size` choices, each of shape itemshape. This is a collective MPI call."""
        dtype = np.array(choices).dtype

        def sampler(rng, args, size):
            # Cannot cope with replace=False as this would be correlated
            return rng.choice(choices, size=size, replace=True, p=p)

        return self._call_rngmethod(sampler, (), itemshape, dtype)

    def _call_rngmethod(self, sampler, args, itemshape, dtype='f8'):
        """
        Loop over the seed table, and call ``sampler(rng, args, size)``
        on each rng, with matched input args and size.
        the args are padded in the front such that the rng is invariant
        no matter how :attr:`size` is distributed.
        Truncate the return value at the front to match the requested :attr:`size`.
        """
        seeds = self._serial_rng.randint(0, high=0xffffffff, size=self.nchunks)

        if np.ndim(itemshape) == 0: itemshape = (itemshape, )
        padded_r, running_args = self._prepare_args_and_result(args, itemshape, dtype)

        running_r = padded_r
        ichunk = self._first_ichunk

        while len(running_r) > 0:
            # at most get a full chunk, or the remaining items
            nreq = min(len(running_r), self.chunksize)

            seed = seeds[ichunk]
            rng = np.random.RandomState(seed)
            args = tuple([a if np.ndim(a) == 0 else a[:nreq] for a in running_args])

            # generate nreq random items from the sampler
            chunk = sampler(rng, args=args, size=(nreq,) + tuple(itemshape))

            running_r[:nreq] = chunk

            # update running arrays, since we have finished nreq items
            running_r = running_r[nreq:]
            running_args = tuple([a if np.ndim(a) == 0 else a[nreq:] for a in running_args])

            ichunk = ichunk + 1

        return padded_r[self._skip:]
