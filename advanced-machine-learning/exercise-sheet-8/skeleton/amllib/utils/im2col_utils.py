"""Some routines for the im2col method"""

__author__ = 'Jens-Peter M. Zemke'
__version__ = '1.1'

__name__ = "amllib.utils.im2col_utils"
__package__ = "amllib.utils"

import numpy as np

######################################################
#                   im2col approach                  #
######################################################
def get_col_indices(tensor, fh, fw, pad=0, stride=1):

    # compute indices for im2col

    # get input size
    c, h, w = tensor

    # compute output size
    zh = (h + 2 * pad - fh) // stride + 1
    zw = (w + 2 * pad - fw) // stride + 1

    # compute channel index ch, will be broadcasted
    ch = np.repeat(np.arange(c), fh * fw).reshape(-1, 1)

    # compute “row” index i
    i0 = np.tile(np.repeat(np.arange(fh), fw), c)
    i1 = stride * np.repeat(np.arange(zh), zw)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)

    # compute “column” index j
    j0 = np.tile(np.arange(fw), fh * c)
    j1 = stride * np.tile(np.arange(zw), zh)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    return ch, i, j

def im2col(a, fh, fw, pad=0, stride=1):

    # get input shape
    n, c, h, w = a.shape

    # pad
    ap = np.pad(a, ((0,0), (0,0),
                    (pad, pad), (pad, pad)),
                mode='constant')

    # get col indices, compute col
    ch, i, j = get_col_indices((c, h, w), fh, fw, pad, stride)
    return np.concatenate(ap[:, ch, i, j], axis=1)

def col2im(dcol, a_shape, fh, fw, pad=0, stride=1):

    # get input size
    n, c, h, w = a_shape

    # get index matrices
    ch, i, j = get_col_indices((c, h, w), fh, fw, pad, stride)

    # split dcol into n minibatches
    dcol_split = np.array(np.hsplit(dcol, n))

    # add padding and sum parts into dap
    hp, wp = h + 2 * pad, w + 2 * pad
    dap = np.zeros((n, c, hp, wp), dtype=dcol.dtype)
    np.add.at(dap, (slice(None), ch, i, j), dcol_split)

    # remove padding
    if pad == 0:
        return dap
    return dap[:, :, pad:-pad, pad:-pad]

if __name__ == "__main__":
    # minibatch input
    n, c, h, w = 2, 3, 3, 3
    # filter dimensions
    m, fh, fw = 5, 2, 2

    # example
    a = np.arange(n * c * h * w).reshape(n, c, h, w)
    col = im2col(a, fh, fw)
    print(col)

    # overlap of backprop in summation
    #delta = np.random.randint(0, 10, col.shape)
    delta = np.ones(col.shape)
    da = col2im(delta, a.shape, fh, fw)
    print(da)
