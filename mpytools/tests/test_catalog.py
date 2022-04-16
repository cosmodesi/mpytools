import os
import tempfile

import numpy as np

from mpytools import Catalog, mpi, setup_logging
from mpytools.array import Slice, MPIScatteredSource
from mpytools.utils import MemoryMonitor


def test_slice():
    sl = Slice(0, None, size=100)
    assert np.allclose(sl.to_array(), np.arange(100))
    sl_array = Slice(sl.to_array())
    assert sl and sl_array
    assert not sl.is_array
    assert sl_array.is_array
    # assert sl_array.to_slices() == sl.to_slices() == [slice(0, 100, 1)]
    sl1, sl2 = sl.split(2)
    assert sl1.idx == slice(0, 50, 1) and sl2.idx == slice(50, 100, 1)
    sl1, sl2 = sl_array.split(2)
    assert np.allclose(sl1.idx, np.arange(50)) and np.allclose(sl2.idx, 50 + np.arange(50))
    assert len(Slice(0, 4).split(10)) == 10

    assert sl.find(slice(10, 120, 2)).idx == slice(10, 99, 2)
    assert Slice(2, 81, 4).find(slice(10, 120, 2), return_index=True) == (slice(2, 20, 1), slice(0, 35, 2))
    assert np.allclose(sl_array.find(slice(10, 120, 2)).idx, np.arange(10, 100, 2))
    assert np.allclose(sl.find([0, 1, 1, 1, 2, 3]).idx, [0, 1, 1, 1, 2, 3])
    assert np.allclose(Slice(2, None, size=100).find([0, 1, 1, 1, 2, 3]).idx, [0, 1])

    assert sl.slice(slice(10, 120, 2)).idx == slice(10, 99, 2)
    assert sl.slice(slice(10, None, -2)).idx == slice(10, None, -2)
    assert sl.slice(slice(120, 2, -2)).idx == slice(98, 3, -2)
    assert sl.slice(slice(120, 2, -2), return_index=True) == (slice(98, 3, -2), slice(11, 59, 1))
    assert np.allclose(sl_array.slice(slice(10, 120, 2)).idx, np.arange(10, 100, 2))
    assert sl.shift(20).idx == slice(20, 120, 1)
    assert np.allclose(sl_array.shift(20).idx, 20 + np.arange(100))
    sl_array = Slice(np.array([0, 1, 2, 3, 5, 6, 8, 10, 12]))
    ref = [(0, 4, 1), (5, 6, 1), (6, 7, 1), (8, 13, 2)]
    for isl, sl in enumerate(sl_array.to_slices()):
        assert sl == slice(*ref[isl])
    for sl in Slice([0, 0]).to_slices():
        assert sl == slice(0, 1, 1)
    assert Slice.snap(slice(2, 20), slice(20, 40)) == [Slice(slice(2, 40))]
    assert Slice.snap(slice(2, 20), slice(20, 2, -1)) == [Slice(slice(2, 20)), Slice(slice(20, 2, -1))]


def test_scattered_source():

    mpicomm = mpi.COMM_WORLD

    carray = np.arange(100)
    sl = slice(mpicomm.rank * carray.size // mpicomm.size, (mpicomm.rank + 1) * carray.size // mpicomm.size, 1)
    local_array = carray[sl]
    source = MPIScatteredSource(sl, csize=carray.size, mpicomm=mpicomm)
    assert np.allclose(source.get(local_array), carray[sl])
    assert np.allclose(source.get(local_array, slice(20, 400)), carray[20:400])
    assert np.allclose(source.get(local_array, slice(400, 20, -1)), carray[400:20:-1])


def test_cslice():

    csize = 10 * mpi.COMM_WORLD.size
    size = mpi.local_size(csize)
    rng = mpi.MPIRandomState(size, seed=42)
    local_slice = slice(rng.mpicomm.rank * csize // rng.mpicomm.size, (rng.mpicomm.rank + 1) * csize // rng.mpicomm.size)
    ref = Catalog(data={'RA': np.arange(csize)[local_slice]})
    assert ref.csize == csize
    for sl in [slice(0, ref.size // 2), slice(ref.size // 2, 2, -2), np.arange(1, ref.size // 2, 2)]:
        test = ref[sl]
        assert test.csize
        for name in ['RA']:
            # print(test[name], ref[name][sl], type(np.all(test[name] == ref[name][sl])))
            assert np.all(test[name] == ref[name][sl])

    for sl in [slice(0, csize * 3 // 4), slice(csize * 3 // 4, 2, -2), np.arange(csize // 4, csize * 3 // 4), np.arange(csize * 3 // 4, csize // 4, -1)]:
        test = ref.cslice(sl)
        assert test.csize
        for name in ['RA']:
            assert np.all(test.cget(name) == ref.cget(name)[sl])

    assert np.all(ref.cindices().mpi_gather(mpiroot=None) == np.arange(ref.csize))
    for name in ['empty', 'zeros', 'ones', 'falses', 'trues', 'nans']:
        getattr(ref, name)(itemshape=3)
    ref.full(fill_value=4.)


def test_io():

    csize = 100 * mpi.COMM_WORLD.size
    size = mpi.local_size(csize)
    rng = mpi.MPIRandomState(size, seed=42)
    ref = Catalog(data={'RA': rng.uniform(0., 1.), 'DEC': rng.uniform(0., 1.), 'Z': rng.uniform(0., 1.), 'Position': rng.uniform(0., 1., itemshape=3)})
    mpicomm = ref.mpicomm
    assert ref.csize == csize

    for ext in ['fits', 'hdf5', 'npy', 'bigfile', 'asdf']:

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = '_tests'
            fn = mpicomm.bcast(os.path.join(tmp_dir, 'tmp.{}'.format(ext)), root=0)
            ref.write(fn)

            fns = [mpicomm.bcast(os.path.join(tmp_dir, 'tmp{:d}.{}'.format(i, ext)), root=0) for i in range(4)]
            ref.write(fns)
            for ii in range(25):
                test = Catalog.read(fn)
                assert set(test.columns()) == set(ref.columns())
                assert np.all(test['Position'] == ref['Position'])
                test['Position'] += 10
            fns = [mpicomm.bcast(os.path.join(tmp_dir, 'tmp{:d}.{}'.format(i, ext)), root=0) for i in range(4)]
            test.write(fns)
            assert np.allclose(test['Position'], ref['Position'] + 10)
            ref.write(fns)
            test = Catalog.read(fns)
            assert np.all(test['Position'] == ref['Position'])

            def apply_slices(tmp, sls, name=None):
                if not isinstance(sls, list): sls = [sls]
                if name is None:
                    for sl in sls: tmp = tmp[sl]
                else:
                    for sl in sls: tmp = getattr(tmp, name)(sl)
                return tmp

            for tfn in [fn, fns]:
                for sls in [slice(0, ref.size // 2), slice(ref.size // 2, 2, -2), np.arange(1, ref.size // 2, 2),
                            np.array([1, 2, 2, 1, 3]),
                            [slice(ref.size // 2, 2, -2), slice(2, ref.size // 4, 2)],
                            [range(1, ref.size // 2, 2), slice(ref.size, 2, -2)],
                            [np.arange(ref.size // 2, 1, -2), slice(ref.size, 2, -2)]]:
                    test = Catalog.read(tfn)
                    test['RA']
                    test = apply_slices(test, sls)
                    assert test.csize
                    for name in ['Position', 'RA']:
                        assert np.all(test[name] == apply_slices(ref[name], sls))
                        test[name].cmean()
                    assert test == apply_slices(ref, sls, 'slice')

                for sls in [slice(0, csize * 3 // 4), slice(csize * 3 // 4, 2, -1), np.arange(ref.csize // 2, 1, -2),
                            np.array([1, 2, 2, 1, 3]),
                            [slice(ref.csize // 2, 2, -2), slice(2, ref.csize // 4, 2)],
                            [range(1, ref.csize // 2, 2), slice(ref.csize, 2, -2)],
                            [np.arange(ref.csize // 2, 1, -2), slice(ref.csize, 2, -2)]]:
                    if not isinstance(sls, list): sls = [sls]
                    test = Catalog.read(tfn)
                    test['RA']
                    test = apply_slices(test, sls, 'cslice')
                    assert test.csize
                    for name in ['Position', 'RA']:
                        assert np.all(test.cget(name) == apply_slices(ref.cget(name), sls))

                    test = Catalog.cconcatenate(apply_slices(Catalog.read(tfn), sls, 'cslice'), apply_slices(Catalog.read(tfn), sls, 'cslice'))
                    for name in ['Position', 'RA']:
                        col = apply_slices(ref.cget(name), sls)
                        assert np.all(test.cget(name) == np.concatenate([col, col]))

            test = ref.from_array(ref.to_array())
            assert test == ref
            test = ref.from_array(ref.to_array().mpi_gather(mpiroot=0), mpiroot=0)
            assert test == ref
            fn = mpicomm.bcast(os.path.join(tmp_dir, 'tmp.npy'), root=0)
            test.save(fn)
            test = Catalog.load(fn)
            assert test.attrs == ref.attrs
            assert test == ref


def test_misc():
    csize = 100 * mpi.COMM_WORLD.size
    size = mpi.local_size(csize)
    rng = mpi.MPIRandomState(size, seed=42)
    ref = Catalog(data={'RA': rng.uniform(0., 1.)})
    for name in ['empty', 'zeros', 'ones', 'falses', 'trues', 'nans']:
        assert getattr(ref, name)(itemshape=2).shape == (size, 2)


def test_memory():

    with MemoryMonitor() as mem:

        size = mpi.local_size(int(1e7))
        rng = mpi.MPIRandomState(size, seed=42)
        catalog = Catalog(data={'Position': rng.uniform(0., 1., itemshape=3)})
        catalog['Position2'] = catalog['Position'].copy()
        mem('randoms')
        fn = os.path.join('_tests', 'tmp.fits')
        catalog.write(fn)
        catalog.mpicomm.Barrier()
        mem('save')
        del catalog
        mem('free')
        catalog = Catalog.read(fn)
        mem('load')
        catalog['Position']
        catalog['Position2']
        mem('load2')


if __name__ == '__main__':

    setup_logging()

    test_slice()
    test_scattered_source()
    test_cslice()
    test_io()
    test_misc()
    test_memory()
