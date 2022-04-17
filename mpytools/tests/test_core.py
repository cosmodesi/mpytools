import numpy as np

import mpytools as mpy
from mpytools import setup_logging


def test_mpi():
    # array = np.arange(100)
    array = np.ones(100, dtype='f4, f8, i4')
    array[:] = np.arange(array.size)
    mpicomm = mpy.COMM_WORLD
    if mpicomm.rank == 0:
        mpy.send(array, dest=1, tag=43, mpicomm=mpicomm)
    if mpicomm.rank == 1:
        array2 = mpy.recv(source=0, tag=43, mpicomm=mpicomm)
        for name in array.dtype.names:
            assert np.allclose(array2[name], array[name])

    def assert_allclose(test, ref):
        for name in ref.dtype.names:
            assert np.allclose(test[name], ref[name])

    gathered = mpicomm.allgather(array)
    assert_allclose(mpy.gather(array, mpiroot=None), np.concatenate(gathered))
    assert_allclose(mpy.bcast(array, mpiroot=0), array)
    assert_allclose(mpy.gather(mpy.scatter(array, mpiroot=0), mpiroot=None), array)
    test = mpy.reduce(array, mpiroot=None)
    ref = np.empty_like(gathered[0])
    for name in ref.dtype.names: ref[name] = sum(tt[name] for tt in gathered)
    assert_allclose(test, ref)


def test_array():
    mpicomm = mpy.COMM_WORLD
    carray = np.arange(100)
    local_array = mpy.scatter(carray, mpicomm=mpicomm, mpiroot=0)
    mpi_array = mpy.array(local_array)
    assert np.allclose(mpy.array(carray, mpiroot=0), mpi_array)
    assert np.allclose(mpi_array.gather(mpiroot=None), carray)
    assert np.allclose(mpy.gather(mpi_array, mpiroot=None), carray)
    assert np.allclose(mpi_array + 1, local_array + 1)
    assert np.allclose(mpi_array + local_array, local_array * 2)
    assert mpy.csize(mpi_array) == 100
    assert mpy.cshape(mpi_array) == (100, )
    assert mpi_array.creshape(-1, 5).cshape == (mpicomm.allreduce(local_array.size) // 5, 5)
    assert np.allclose(mpy.cconcatenate(mpi_array, mpi_array).gather(mpiroot=None), np.concatenate([carray, carray]))
    assert np.allclose(mpy.cappend(mpi_array, 2. * mpi_array), mpy.cconcatenate([mpi_array, 2. * mpi_array]))
    assert np.allclose(mpi_array.cslice(10, None, -1).gather(mpiroot=None), carray[10:None:-1])

    mpi_array = mpi_array.cslice(None, None, -1)
    mpi_array.csort(); carray.sort()
    assert np.allclose(mpi_array.gather(mpiroot=None), carray)

    for name in ['csum', 'cprod', 'cmean', 'cmin', 'cmax', 'cvar', 'cstd']:
        assert np.allclose(getattr(mpy, name)(mpi_array), getattr(carray, name[1:])())
        assert np.allclose(getattr(mpi_array, name)(), getattr(carray, name[1:])())

    assert np.allclose(mpy.caverage(mpi_array, weights=mpi_array), np.average(carray, weights=carray))
    assert np.allclose(mpy.cquantile(mpi_array, q=(0.2, 0.4)), np.quantile(carray, q=(0.2, 0.4)))
    assert np.allclose(mpy.ccov(mpi_array), np.cov(carray))
    assert np.allclose(mpy.ccorrcoef(mpi_array), np.corrcoef(carray))

    for name in ['empty', 'zeros', 'ones']:
        assert getattr(mpy, name)(shape=10).shape == (10,)
        assert getattr(mpy, name)(cshape=10).cshape == (10,)
    assert mpy.full(fill_value=4., shape=10).shape == (10,)
    assert mpy.full(fill_value=4., cshape=10).cshape == (10,)
    assert np.array(mpi_array, copy=False, dtype='f8').dtype == np.float64
    assert np.asanyarray(mpi_array, dtype=None) is mpi_array


if __name__ == '__main__':

    setup_logging()

    test_mpi()
    test_array()
