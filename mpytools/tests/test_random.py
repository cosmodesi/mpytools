from mpytools import random, setup_logging


def test_random():

    random.set_common_seed(42)
    random.bcast_seed(42)
    random.set_independent_seed(42)
    random.MPIRandomState(size=100)


if __name__ == '__main__':

    setup_logging()

    test_random()
