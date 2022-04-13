import numpy as np

from mpytools import MPIScatteredArray, setup_logging
from mpytools import mpi


def test_array():
    mpicomm = mpi.COMM_WORLD
    carray = np.arange(100)
    local_array = mpi.scatter_array(carray, mpicomm=mpicomm, root=0)
    mpi_array = MPIScatteredArray(local_array)
    assert np.allclose(MPIScatteredArray(carray, mpiroot=0), mpi_array)
    assert np.allclose(mpi_array + 1, local_array + 1)
    assert np.allclose(mpi_array + local_array, local_array * 2)
    assert np.allclose(np.std(mpi_array), np.std(local_array))
    assert np.allclose(mpi_array.gathered(mpiroot=None), carray)
    mpi_array = mpi_array.cslice(None, None, -1)
    mpi_array.csort(); carray.sort()
    assert np.allclose(mpi_array.gathered(mpiroot=None), carray)
    assert np.allclose(mpi_array.csum(), carray.sum())
    assert np.allclose(mpi_array.caverage(weights=mpi_array), np.average(carray, weights=carray))
    assert np.allclose(mpi_array.cmean(), np.mean(carray))
    assert np.allclose(mpi_array.cvar(ddof=1), np.var(carray, ddof=1))
    assert np.allclose(mpi_array.cstd(ddof=1), np.std(carray, ddof=1))
    assert np.allclose(mpi_array.cmin(), carray.min())
    assert np.allclose(mpi_array.cmax(), carray.max())
    #assert np.allclose(mpi_array.cargmin(), carray.argmin())
    #assert np.allclose(mpi_array.cargmax(), carray.argmax())
    assert np.allclose(mpi_array.cquantile(q=(0.2, 0.4)), np.quantile(carray, q=(0.2, 0.4)))
    mpi_array.cshape = (-1, 5)
    assert mpi_array.cshape == (mpicomm.allreduce(local_array.size) // 5, 5)


if __name__ == '__main__':

    setup_logging()

    test_array()
