import numpy as np

from mpytools import MPIScatteredArray, setup_logging
from mpytools import mpi


def test_mpi():
    # array = np.arange(100)
    array = np.ones(100, dtype='f4, f8, i4')
    array[:] = np.arange(array.size)
    mpicomm = mpi.COMM_WORLD
    if mpicomm.rank == 0:
        mpi.send_array(array, dest=1, tag=43, mpicomm=mpicomm)
    if mpicomm.rank == 1:
        array2 = mpi.recv_array(source=0, tag=43, mpicomm=mpicomm)
        for name in array.dtype.names:
            assert np.allclose(array2[name], array[name])
    test = mpi.reduce_array(array, root=None)
    tmp = mpicomm.allgather(array)
    ref = np.empty_like(tmp[0])
    for name in ref.dtype.names: ref[name] = sum(tt[name] for tt in tmp)
    for name in ref.dtype.names: assert np.allclose(test[name], ref[name])


def test_array():
    mpicomm = mpi.COMM_WORLD
    carray = np.arange(100)
    local_array = mpi.scatter_array(carray, mpicomm=mpicomm, root=0)
    assert np.all((local_array >= 0) & (local_array >= 0))
    mpi_array = MPIScatteredArray(local_array)
    assert np.allclose(MPIScatteredArray(carray, mpiroot=0), mpi_array)
    assert np.allclose(mpi_array + 1, local_array + 1)
    assert np.allclose(mpi_array + local_array, local_array * 2)
    assert np.allclose(np.std(mpi_array), np.std(local_array))
    assert np.allclose(mpi_array.mpi_gather(mpiroot=None), carray)
    assert np.allclose(mpi.gather_array(mpi_array, root=None), carray)
    assert np.allclose(MPIScatteredArray.cconcatenate(mpi_array, mpi_array).mpi_gather(mpiroot=None), np.concatenate([carray, carray]))
    mpi_array = mpi_array.cslice(None, None, -1)
    mpi_array.csort(); carray.sort()
    assert np.allclose(mpi_array.mpi_gather(mpiroot=None), carray)
    assert np.allclose(mpi_array.csum(), carray.sum())
    assert np.allclose(mpi_array.caverage(weights=mpi_array), np.average(carray, weights=carray))
    assert np.allclose(mpi_array.cmean(), np.mean(carray))
    assert np.allclose(mpi_array.cvar(ddof=1), np.var(carray, ddof=1))
    assert np.allclose(mpi_array.cstd(ddof=1), np.std(carray, ddof=1))
    assert np.allclose(mpi_array.cmin(), carray.min())
    assert np.allclose(mpi_array.cmax(), carray.max())
    # assert np.allclose(mpi_array.cargmin(), carray.argmin())
    # assert np.allclose(mpi_array.cargmax(), carray.argmax())
    assert np.allclose(mpi_array.cquantile(q=(0.2, 0.4)), np.quantile(carray, q=(0.2, 0.4)))
    mpi_array = mpi_array.creshape(-1, 5)
    assert mpi_array.cshape == (mpicomm.allreduce(local_array.size) // 5, 5)

    for name in ['empty', 'zeros', 'ones', 'falses', 'trues', 'nans']:
        assert getattr(MPIScatteredArray, name)(shape=10).shape == (10,)
        assert getattr(MPIScatteredArray, name)(cshape=10).cshape == (10,)
    assert MPIScatteredArray.full(fill_value=4., shape=10).shape == (10,)
    assert MPIScatteredArray.full(fill_value=4., cshape=10).cshape == (10,)
    assert np.array(mpi_array, copy=False, dtype='f8').dtype == np.float64
    assert np.asanyarray(mpi_array, dtype=None) is mpi_array


if __name__ == '__main__':

    setup_logging()

    test_mpi()
    test_array()
