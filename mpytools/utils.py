"""A few utilities."""

import os
import sys
import time
import logging
import functools
from contextlib import contextmanager
import traceback

import numpy as np
from mpi4py import MPI


class CurrentMPIComm(object):
    """Class to facilitate getting and setting the current MPI communicator, taken from nbodykit."""
    logger = logging.getLogger('CurrentMPIComm')

    _stack = [MPI.COMM_WORLD]

    @staticmethod
    def enable(func):
        """
        Decorator to attach the current MPI communicator to the input
        keyword arguments of ``func``, via the ``mpicomm`` keyword.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mpicomm = kwargs.get('mpicomm', None)
            if mpicomm is None:
                for arg in args:
                    mpicomm = getattr(arg, 'mpicomm', None)
            if mpicomm is None:
                mpicomm = CurrentMPIComm.get()
            kwargs['mpicomm'] = mpicomm
            return func(*args, **kwargs)

        return wrapper

    @classmethod
    @contextmanager
    def enter(cls, mpicomm):
        """
        Enter a context where the current default MPI communicator is modified to the
        argument ``mpicomm``. After leaving the context manager the communicator is restored.
        """
        cls.push(mpicomm)

        yield

        cls.pop()

    @classmethod
    def push(cls, mpicomm):
        """Switch to a new current default MPI communicator."""
        cls._stack.append(mpicomm)
        if mpicomm.rank == 0:
            cls.logger.info('Entering a current communicator of size {:d}'.format(mpicomm.size))
        cls._stack[-1].barrier()

    @classmethod
    def pop(cls):
        """Restore to the previous current default MPI communicator."""
        mpicomm = cls._stack[-1]
        if mpicomm.rank == 0:
            cls.logger.info('Leaving current communicator of size {:d}'.format(mpicomm.size))
        cls._stack[-1].barrier()
        cls._stack.pop()
        mpicomm = cls._stack[-1]
        if mpicomm.rank == 0:
            cls.logger.info('Restored current communicator to size {:d}'.format(mpicomm.size))

    @classmethod
    def get(cls):
        """Get the default current MPI communicator. The initial value is ``MPI.COMM_WORLD``."""
        return cls._stack[-1]


@CurrentMPIComm.enable
def exception_handler(exc_type, exc_value, exc_traceback, mpicomm=None):
    """Print exception with a logger."""
    # Do not print traceback if the exception has been handled and logged
    _logger_name = 'Exception'
    log = logging.getLogger(_logger_name)
    line = '=' * 100
    # log.critical(line[len(_logger_name) + 5:] + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    log.critical('\n' + line + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    if exc_type is KeyboardInterrupt:
        log.critical('Interrupted by the user.')
    else:
        log.critical('An error occured.')
    if mpicomm.size > 1:
        mpicomm.Abort()


def mkdir(dirname):
    """Try to create ``dirname`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname)  # MPI...
    except OSError:
        return


def setup_logging(level=logging.INFO, stream=sys.stdout, filename=None, filemode='w', **kwargs):
    """
    Set up logging.

    Parameters
    ----------
    level : string, int, default=logging.INFO
        Logging level.

    stream : _io.TextIOWrapper, default=sys.stdout
        Where to stream.

    filename : string, default=None
        If not ``None`` stream to file name.

    filemode : string, default='w'
        Mode to open file, only used if filename is not ``None``.

    kwargs : dict
        Other arguments for :func:`logging.basicConfig`.
    """
    # Cannot provide stream and filename kwargs at the same time to logging.basicConfig, so handle different cases
    # Thanks to https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    if isinstance(level, str):
        level = {'info': logging.INFO, 'debug': logging.DEBUG, 'warning': logging.WARNING}[level.lower()]
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    t0 = time.time()

    class MyFormatter(logging.Formatter):

        @CurrentMPIComm.enable
        def format(self, record, mpicomm=None):
            ranksize = '[{:{dig}d}/{:d}]'.format(mpicomm.rank, mpicomm.size, dig=len(str(mpicomm.size)))
            self._style._fmt = '[%09.2f] ' % (time.time() - t0) + ranksize + ' %(asctime)s %(name)-25s %(levelname)-8s %(message)s'
            return super(MyFormatter, self).format(record)

    fmt = MyFormatter(datefmt='%m-%d %H:%M ')
    if filename is not None:
        mkdir(os.path.dirname(filename))
        handler = logging.FileHandler(filename, mode=filemode)
    else:
        handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(fmt)
    logging.basicConfig(level=level, handlers=[handler], **kwargs)
    sys.excepthook = exception_handler


class BaseMetaClass(type):

    """Metaclass to add logging attributes to :class:`BaseClass` derived classes."""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        cls.set_logger()
        return cls

    def set_logger(cls):
        """
        Add attributes for logging:

        - logger
        - methods log_debug, log_info, log_warning, log_error, log_critical
        """
        cls.logger = logging.getLogger(cls.__name__)

        def make_logger(level):

            @classmethod
            @CurrentMPIComm.enable
            def logger(cls, *args, rank=None, mpicomm=None, **kwargs):
                if rank is None or mpicomm.rank == rank:
                    getattr(cls.logger, level)(*args, **kwargs)

            return logger

        for level in ['debug', 'info', 'warning', 'error', 'critical']:
            setattr(cls, 'log_{}'.format(level), make_logger(level))


class BaseClass(object, metaclass=BaseMetaClass):
    """
    Base class that implements :meth:`copy`.
    To be used throughout this package.
    """
    def __copy__(self):
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new

    def copy(self, **kwargs):
        new = self.__copy__()
        new.__dict__.update(kwargs)
        return new

    def __setstate__(self, state):
        self.__dict__.update(state)

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def save(self, filename):
        self.log_info('Saving {}.'.format(filename))
        mkdir(os.path.dirname(filename))
        np.save(filename, self.__getstate__(), allow_pickle=True)

    @classmethod
    def load(cls, filename):
        cls.log_info('Loading {}.'.format(filename))
        state = np.load(filename, allow_pickle=True)[()]
        new = cls.from_state(state)
        return new


def match1d(id1, id2):
    """
    Match ``id2`` array to ``id1`` array, such that ``np.all(id1[index1] == id2[index2])``.

    Parameters
    ----------
    id1 : array
        IDs 1, should be unique.

    id2 : array
        IDs 2, should be unique.

    Returns
    -------
    index1 : array
        Indices of matching ``id1``.

    index2 : array
        Indices of matching ``id2``.

    Warning
    -------
    Makes sense only if ``id1`` and ``id2`` elements are unique.

    References
    ----------
    https://www.followthesheep.com/?p=1366
    """
    sort1 = np.argsort(id1)
    sort2 = np.argsort(id2)

    ind1 = id2[sort2].searchsorted(id1[sort1], side='right') > id2[sort2].searchsorted(id1[sort1], side='left')
    ind2 = id1[sort1].searchsorted(id2[sort2], side='right') > id1[sort1].searchsorted(id2[sort2], side='left')

    return sort1[ind1], sort2[ind2]


def match1d_to(id1, id2, return_index=False):
    """
    Return indexes where ``id1`` matches ``id2``, such that ``np.all(id1[index1] == id2)``.

    Parameters
    ----------
    id1 : array
        IDs 1, should be unique.

    id2 : array
        IDs 2.

    return_index : bool, default=False
        Return indices in ``id2`` corresponding to ``id1[index1]``.

    Returns
    -------
    index1 : array
        Indices of matching ``id1``.
    """
    sort1 = np.argsort(id1)
    ind2 = id1[sort1].searchsorted(id2, side='left')
    mask = id1[sort1].searchsorted(id2, side='right') > ind2
    ind2 = ind2[mask]
    ind1 = sort1[ind2]
    if return_index:
        return ind1, np.flatnonzero(mask)
    return ind1


def weighted_quantile(x, q, weights=None, axis=None, interpolation='lower'):
    """
    Compute the q-th quantile of the weighted data along the specified axis.

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
    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.quantile(x, q, axis=axis, interpolation=interpolation)

    # Initial check.
    x = np.atleast_1d(x)
    isscalar = np.ndim(q) == 0
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.) or np.any(q > 1.):
        raise ValueError('Quantiles must be between 0. and 1.')

    if axis is None:
        axis = range(x.ndim)

    if np.ndim(axis) == 0:
        axis = (axis,)

    if weights.ndim > 1:
        if x.shape != weights.shape:
            raise ValueError('Dimension mismatch: shape(weights) != shape(x).')

    x = np.moveaxis(x, axis, range(x.ndim - len(axis), x.ndim))
    x = x.reshape(x.shape[:-len(axis)] + (-1,))
    if weights.ndim > 1:
        weights = np.moveaxis(weights, axis, range(x.ndim - len(axis), x.ndim))
        weights = weights.reshape(weights.shape[:-len(axis)] + (-1,))
    else:
        reps = x.shape[:-1] + (1,)
        weights = np.tile(weights, reps)

    idx = np.argsort(x, axis=-1)  # sort samples
    x = np.take_along_axis(x, idx, axis=-1)
    sw = np.take_along_axis(weights, idx, axis=-1)  # sort weights
    cdf = np.cumsum(sw, axis=-1)  # compute CDF
    cdf = cdf[..., :-1]
    cdf = cdf / cdf[..., -1][..., None]  # normalize CDF
    zeros = np.zeros_like(cdf, shape=cdf.shape[:-1] + (1,))
    cdf = np.concatenate([zeros, cdf], axis=-1)  # ensure proper span
    idx0 = np.apply_along_axis(np.searchsorted, -1, cdf, q, side='right') - 1
    if interpolation != 'higher':
        q0 = np.take_along_axis(x, idx0, axis=-1)
    if interpolation != 'lower':
        idx1 = np.clip(idx0 + 1, None, x.shape[-1] - 1)
        q1 = np.take_along_axis(x, idx1, axis=-1)
    if interpolation in ['nearest', 'linear']:
        cdf0, cdf1 = np.take_along_axis(cdf, idx0, axis=-1), np.take_along_axis(cdf, idx1, axis=-1)
    if interpolation == 'nearest':
        mask_lower = q - cdf0 < cdf1 - q
        quantiles = q1
        # in place, q1 not used in the following
        quantiles[mask_lower] = q0[mask_lower]
    if interpolation == 'linear':
        step = cdf1 - cdf0
        diff = q - cdf0
        mask = idx1 == idx0
        step[mask] = diff[mask]
        fraction = diff / step
        quantiles = q0 + fraction * (q1 - q0)
    if interpolation == 'lower':
        quantiles = q0
    if interpolation == 'higher':
        quantiles = q1
    if interpolation == 'midpoint':
        quantiles = (q0 + q1) / 2.
    quantiles = quantiles.swapaxes(-1, 0)
    if isscalar:
        return quantiles[0]
    return quantiles


def is_sequence(item):
    """Whether input item is a tuple or list."""
    return isinstance(item, (list, tuple))


def list_concatenate(li):
    """Concatenate input list of sequences (tuples or lists)."""
    toret = []
    for el in li:
        if is_sequence(el):
            toret += list(el)
        else:
            toret.append(el)
    return toret


class MemoryMonitor(object):
    """
    Class that monitors memory usage and clock, useful to check for memory leaks.

    >>> with MemoryMonitor() as mem:
            '''do something'''
            mem()
            '''do something else'''
    """
    def __init__(self, pid=None):
        """
        Initalize :class:`MemoryMonitor` and register current memory usage.

        Parameters
        ----------
        pid : int, default=None
            Process identifier. If ``None``, use the identifier of the current process.
        """
        import psutil
        self.proc = psutil.Process(os.getpid() if pid is None else pid)
        self.mem = self.proc.memory_info().rss / 1e6
        self.time = time.time()
        msg = 'using {:.3f} [Mb]'.format(self.mem)
        print(msg, flush=True)

    def __enter__(self):
        """Enter context."""
        return self

    def __call__(self, log=None):
        """Update memory usage."""
        mem = self.proc.memory_info().rss / 1e6
        t = time.time()
        msg = 'using {:.3f} [Mb] (increase of {:.3f} [Mb]) after {:.3f} [s]'.format(mem, mem - self.mem, t - self.time)
        if log:
            msg = '[{}] {}'.format(log, msg)
        print(msg, flush=True)
        self.mem = mem
        self.time = t

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit context."""
        self()
