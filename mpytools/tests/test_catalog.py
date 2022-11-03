import os
import tempfile

import numpy as np

import mpytools as mpy
from mpytools import Catalog, setup_logging
from mpytools.random import MPIRandomState
from mpytools.core import Slice, local_size, MPIScatteredSource
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

    assert sl.slice(slice(10, 120, 2)).idx == slice(10, 100, 2)
    assert sl.slice(slice(10, None, -2)).idx == slice(10, None, -2)
    assert sl.slice(slice(120, 2, -2)).idx == slice(99, 2, -2)
    assert sl.slice(slice(120, 2, -2), return_index=True) == (slice(99, 2, -2), slice(11, 59, 1))
    assert np.allclose(sl_array.slice(slice(10, 120, 2)).idx, np.arange(10, 100, 2))
    assert sl.shift(20).idx == slice(20, 120, 1)
    assert np.allclose(sl_array.shift(20).idx, 20 + np.arange(100))
    sl_array = Slice(np.array([0, 1, 2, 3, 5, 6, 8, 10, 12]))
    ref = [(0, 4, 1), (5, 6, 1), (6, 7, 1), (8, 13, 2)]
    nslices = sl_array.nslices()
    for isl, sl in enumerate(sl_array.to_slices()):
        assert sl == slice(*ref[isl])
    assert isl == nslices - 1
    assert Slice([0, 10]).nslices() == 1
    for sl in Slice([0, 0]).to_slices():
        assert sl == slice(0, 1, 1)
    assert Slice.snap(slice(2, 20), slice(20, 40)) == [Slice(slice(2, 40))]
    assert Slice.snap(slice(2, 20), slice(20, 2, -1)) == [Slice(slice(2, 20)), Slice(slice(20, 2, -1))]


def test_scattered_source():

    mpicomm = mpy.COMM_WORLD

    carray = np.arange(100)
    sl = slice(mpicomm.rank * carray.size // mpicomm.size, (mpicomm.rank + 1) * carray.size // mpicomm.size, 1)
    local_array = carray[sl]
    source = MPIScatteredSource(sl, csize=carray.size, mpicomm=mpicomm)
    assert np.allclose(source.get(local_array), carray[sl])
    assert np.allclose(source.get(local_array, slice(20, 400)), carray[20:400])
    assert np.allclose(source.get(local_array, slice(400, 20, -1)), carray[400:20:-1])
    assert np.allclose(source.get(local_array, slice(-60, 50)), carray[-60:50])


def test_cslice():

    csize = 10 * mpy.COMM_WORLD.size
    size = local_size(csize)
    rng = MPIRandomState(size, seed=42)
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

    assert np.all(ref.cindex().gather(mpiroot=None) == np.arange(ref.csize))
    for name in ['empty', 'zeros', 'ones', 'falses', 'trues', 'nans']:
        getattr(ref, name)(itemshape=3)
    ref.full(fill_value=4.)


class FakeCatalog(Catalog):

    _init_kwargs = ['boxcenter']

    @classmethod
    def from_dict(cls, *args, boxcenter=None, **kwargs):
        self = super(FakeCatalog, cls).from_dict(*args, **kwargs)
        self.boxcenter = boxcenter
        return self


def test_io():

    csize = 100 * mpy.COMM_WORLD.size
    size = local_size(csize)
    rng = MPIRandomState(size, seed=42)
    ref = Catalog(data={'RA': rng.uniform(0., 1.), 'DEC': rng.uniform(0., 1.), 'Z': rng.uniform(0., 1.), 'Position': rng.uniform(0., 1., itemshape=3)})
    mpicomm = ref.mpicomm
    assert ref.csize == csize
    ref = ref[ref['Z'] < 0.9]
    csize = ref.csize
    rsize = csize // mpy.COMM_WORLD.size #- 1

    for ext in ['fits', 'npy', 'bigfile', 'asdf', 'hdf5']:

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = '_tests'
            fn = mpicomm.bcast(os.path.join(tmp_dir, 'tmp.{}'.format(ext)), root=0)
            ref.write(fn)
            fns = [mpicomm.bcast(os.path.join(tmp_dir, 'tmp{:d}.{}'.format(i, ext)), root=0) for i in range(4)]
            ref.write(fns)

            for ii in range(15):
                test = Catalog.read(fn)
                assert set(test.columns()) == set(ref.columns())
                assert np.all(test.cget('Position') == ref.cget('Position'))
                test['Position'] += 10
            if ext == 'bigfile':
                test.write(fns, columns=['Position', 'RA'], overwrite=True)
            else:
                test.write(fns, columns=['Position', 'RA'])
            assert np.allclose(test.cget('Position'), ref.cget('Position') + 10)
            test2 = Catalog.read(fns)
            assert set(test2.columns()) == set(['Position', 'RA'])  # bigfile does not conserve column order
            ref.write(fns)
            test = Catalog.read(fns)
            assert np.all(test.cget('Position') == ref.cget('Position'))

            def apply_slices(tmp, sls, name=None):
                if not isinstance(sls, list): sls = [sls]
                if name is None:
                    for sl in sls: tmp = tmp[sl]
                else:
                    for sl in sls: tmp = getattr(tmp, name)(sl)
                return tmp

            from mpytools.io import BaseFile
            for nslices_max in [0, 10]:
                BaseFile._read_nslices_max = nslices_max
                for tfn in [fn, fns]:

                    for sls in [slice(0, rsize // 2), slice(rsize // 2, 2, -2), np.arange(1, rsize // 2, 2),
                                np.array([1, 2, 2, 1, 3]), [slice(rsize // 2, -1, -1), slice(0, rsize * 3 // 4)],
                                [slice(rsize // 2, 2, -2), slice(2, rsize // 4, 2)],
                                [range(1, rsize // 2, 2), slice(rsize, 2, -2)],
                                [np.arange(rsize // 2, 1, -2), slice(rsize, 2, -2)]]:

                        test = Catalog.read(tfn)
                        test['RA']
                        #print(test['RA'].size)
                        test2 = apply_slices(test, sls)
                        #assert test.csize
                        for name in ['Position', 'RA']:
                            #print(sls, test2[name].shape, apply_slices(test[name], sls).shape)
                            assert np.all(test2[name] == apply_slices(test[name], sls))
                            test2[name].cmean()
                        assert test2 == apply_slices(test, sls, 'slice')

                    for sls in [slice(0, csize * 3 // 4), slice(csize * 3 // 4, 2, -1), np.arange(csize // 2, 1, -2),
                                np.array([1, 2, 2, 1, 3]), [slice(csize // 2, -1, -1), slice(0, csize * 3 // 4)],
                                [slice(csize // 2, 2, -2), slice(2, csize // 4, 2)],
                                [range(1, csize // 2, 2), slice(csize, 2, -2)],
                                [np.arange(csize // 2, 1, -2), slice(csize, 2, -2)]]:

                        if not isinstance(sls, list): sls = [sls]
                        test = Catalog.read(tfn)
                        #test['RA']
                        test = apply_slices(test, sls, 'cslice')
                        #assert test.csize
                        for name in ['Position', 'RA']:
                            #print(sls, test.cget(name).shape, apply_slices(ref.cget(name), sls).shape)
                            assert np.all(test.cget(name) == apply_slices(ref.cget(name), sls))

                        test = Catalog.cconcatenate(apply_slices(Catalog.read(tfn), sls, 'cslice'), apply_slices(Catalog.read(tfn), sls, 'cslice'))
                        #test = Catalog.cconcatenate(test, test)
                        for name in ['Position', 'RA']:
                            col = apply_slices(ref.cget(name), sls)
                            assert np.all(test.cget(name) == np.concatenate([col, col]))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = '_tests'
        test = ref.from_array(ref.to_array())
        assert test == ref
        test = ref.from_array(ref.to_array().gather(mpiroot=0), mpiroot=0)
        assert test == ref
        fn = mpicomm.bcast(os.path.join(tmp_dir, 'tmp.npy'), root=0)
        test.save(fn)
        test = Catalog.load(fn)
        assert test.attrs == ref.attrs
        assert test == ref
        test.save(fn, columns=['RA', 'DEC'])
        test = Catalog.load(fn)
        assert test.columns() == ['RA', 'DEC']

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = '_tests'
        fn = os.path.join(tmp_dir, 'test.bigfile')
        header = {'boxsize': 1.}
        attrs = {'boxcenter': 0.}
        ref.write(fn, group='1/', header=header, overwrite=True)
        test = Catalog.read(fn, group='1/', attrs=attrs)
        assert test.header == header
        assert test.attrs == attrs
        test = FakeCatalog.read(fn, group='1/', attrs=attrs, boxcenter=1.)
        assert test.header == header
        assert test.attrs == attrs
        assert test.boxcenter == 1.


def test_misc():
    csize = 100 * mpy.COMM_WORLD.size
    size = local_size(csize)
    rng = MPIRandomState(size, seed=42)
    ref = Catalog(data={'RA': rng.uniform(0., 1.), 'DEC': rng.uniform(0., 1.), 'Z': rng.uniform(0., 1.)})
    for name in ['empty', 'zeros', 'ones', 'falses', 'trues', 'nans']:
        assert getattr(ref, name)(itemshape=2).shape == (size, 2)
    test = Catalog(data=ref, columns=ref.columns())
    assert test == ref
    test = ref['Z', 'RA']
    assert test.columns() == ['Z', 'RA']
    Z, DEC = ref.get(['Z', 'DEC'])
    assert np.allclose(DEC, ref['DEC'])
    test = ref.copy()
    test['Z', 'RA'] = ref.ones()
    assert np.allclose(test['Z'], 1.) and not np.allclose(ref['Z'], 1.)
    test['Z', 'RA'] = ref
    assert np.allclose(test['RA'], ref['RA'])
    test.set(['Z', 'RA'], [ref.zeros(), ref.ones()])
    assert np.allclose(test['Z'], 0.) and np.allclose(test['RA'], 1.)
    test['Z', 'RA'][...] = ref['RA', 'Z']
    assert np.allclose(test['Z'], ref['Z'])
    del test['Z', 'RA']
    assert test.columns() == ['DEC']
    test = ref.copy()
    test['index'] = test.cindex()
    test['ob'] = test['index'][::-1]
    test = test.csort('ob')
    assert np.all(np.diff(test['ob']) >= 0.)
    test = test.csort('index')
    assert np.allclose(test['index'], test.cindex())
    test = test[slice(10 if test.mpicomm.rank == 0 else None)]
    test['rank'] = test.full(test.mpicomm.size - test.mpicomm.rank)
    test = test.csort('rank', size='orderby_counts')
    sizes = test.mpicomm.allgather(test.size)
    assert sizes[-1] == 10


def test_memory():

    with MemoryMonitor() as mem:

        size = local_size(int(1e7))
        rng = MPIRandomState(size, seed=42)
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
