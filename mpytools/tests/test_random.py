import numpy as np

from mpytools import random, setup_logging


def test_random():

    random.set_common_seed(42)
    assert np.ndim(random.bcast_seed(42)) == 0
    assert random.bcast_seed(42, size=10).shape == (10,)
    random.set_independent_seed(42)
    rng = random.MPIRandomState(size=100)
    rng.uniform()


if __name__ == '__main__':

    setup_logging()
    test_random()
