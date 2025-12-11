"""Test the timings for all implemented evaluation methods."""

from time import process_time
import numpy as np

from amllib.layers import Conv2D

def test_evaluations(
    n: int, c: int, h: int, w: int,
    fh: int, fw: int) -> None:
    """
    Test the different evaluation methods for our CNN layer.

    Parameters
    ----------
    n : int
        Number of inputs to test with.
    c : int
        Number of input channels to test with.
    h : int
        Height of the inputs to test with.
    w : int
        Width of the inputs to test with.
    fh : int
        Filter height to test with.
    fw : int
        Filter width to test with.
    """

    print('----------------------------------------------------------')
    print(f'Testing evaluations for 32 filters of shape {fh} x {fw}')
    print(f'and inputs of shape {n} x {c} x {h} x {w}')
    print('----------------------------------------------------------')

    tensor = (c, h, w)
    fshape = (32, fh, fw)
    conv = Conv2D(fshape, input_shape=tensor)

    # generate random input
    x = np.random.randint(0, 10, (n, c, h, w))

    # test im2col
    start = process_time()
    for i in range(1):
        y = conv.evaluate_im2col(x)
    time_im2col = process_time() - start
    print(f'Time used by im2col: {time_im2col:.2f} seconds')

    # test fft
    start = process_time()
    for i in range(1):
        y = conv.evaluate_fft(x)
    time_fft = process_time() - start
    print(f'Time used by fft:    {time_fft:.2f} seconds')

    # test scipy
    start = process_time()
    for i in range(1):
        y = conv.evaluate_scipy(x)
    time_scipy = process_time() - start
    print(f'Time used by scipy:  {time_scipy:.2f} seconds')

if __name__ == '__main__':

    for case in range(8):
        if case % 2 == 0: # small filters
            fh, fw = 3, 3
        else:             # large filters
            fh, fw = 7, 7
        dcase = case // 2
        if dcase == 0:   # MNIST
            n, c, h, w = 100, 3, 28, 28
        elif dcase == 1: # larger images
            n, c, h, w = 100, 3, 100, 100
        elif dcase == 2: # MNIST-like w/ many channels
            n, c, h, w = 20, 50, 28, 28
        else:            # larger images w/ many channels
            n, c, h, w = 20, 50, 100, 100
            if case % 2 == 1: # not for larger filters!
                break

        test_evaluations(n, c, h, w, fh, fw)
