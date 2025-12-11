"""Implementation of a dense connected FNN layer."""

__author__ = 'Jens-Peter M. Zemke, Jonas Grams'
__version__ = '1.1'

__name__ = "amllib.layer.conv2d"
__package__ = "amllib.layers"

from typing import Optional

import numpy as np
from numpy.fft import rfft2, irfft2
from scipy.signal import convolve2d

from amllib.utils.im2col_utils import im2col, col2im
from amllib.initializers.initializer import KernelInitializer
from amllib.initializers import RandnAverage
from amllib.optimizers.optimizer import Optimizer
from amllib.optimizers import SGD
from amllib.activations.activation import Activation
from amllib.activations import ReLU
from amllib.layers.layer import Layer


class Conv2D(Layer):
    """
    Class representation of a 2D convolutional layer.

    This class contains the implementation of a 2D
    convolutional layer. It provides three different
    implementations of convolutions.

    Attributes
    ----------
    fshape: tuple[int, int, int]
        Filter shape. Is of the form (M, FH, FW).

            - M: Number of filters in the filter bank.
            - FH: Height of the filter.
            - FW: Width of the filter.
    input_shape: tuple[int, int, int]
        Shape of the inputs without the batch dimension.
        The inputs are assumed to be in the NCHW format,
        so `input_shape` should be of the form (C, H, W):

        - C: number of channels of the input,
        - H: height of the input,
        - W: width of the input.
    afun: Activation
        Activation function of the layer.
    optim: Optimizer
        Optimizer used for the updates while training.
    initializer: KernelInitializer
        Initializer used to initializer the weights. The initializer
        is called in the build method.
    eval_method : str
            Evaluation method for the convolutions:

            - scipy: Use the scipy convolution function.
            - fft: Use FFT to compute the convolutions.
            - im2col: Use the im2col method to compute the convolutions.
    __z : np.ndarray
        Last computed convolutions.
    __a : np.ndarray
        Last computed activated outputs.
    f : np.ndarray
        Filter bank of the layer.
    b : np.ndarray
        Bias of the layer.
    df : np.ndarray
        Storage for the update of the filter bank.
    db : np.ndarray
        Storage for the update of the bias.
    """

    def __init__(self,
                 fshape: tuple[int, int, int],
                 input_shape: Optional[tuple[int, int, int]] = None,
                 afun: Optional[Activation] = None,
                 optim: Optional[Optimizer] = None,
                 initializer: Optional[KernelInitializer] = None,
                 eval_method: str = 'scipy') -> None:
        """
        Initialize the 2D convolutional layer.

        Parameters
        ----------
        fshape : tuple[int, int, int]
            Filter shape. Is of the form (M, FH, FW).

            - M: Number of filters in the filter bank.
            - FH: Height of the filter.
            - FW: Width of the filter.

        input_shape : tuple[int, int, int], optional
            Shape of the inputs. Is of the form (C, H, W).

            - C: Number of channels of the input.
            - H: Height of the input.
            - W: Width of the input,

            by default None. If it is not None, the layer is
            built on initialization.

        afun : Activation, optional
            Activation function, by default ReLU
        optim : Optimizer, optional
            Optimizer for training, by default SGD(0.01)
        initializer : KernelInitializer, optional
            Filterbank initializer, by default RandnAverage
        eval_method : str, optional
            Evaluation method for the convolutions, by default 'scipy'

            - scipy: Use the scipy convolution function.
            - fft: Use FFT to compute the convolutions.
            - im2col: Use the im2col method to compute the convolutions.
        """
        super().__init__()

        if afun is None:
            self.afun = ReLU()
        else:
            self.afun = afun

        if optim is None:
            self.optim = SGD()
        else:
            self.optim = optim

        if initializer is None:
            self.initializer = RandnAverage()
        else:
            self.initializer = initializer

        self.fshape = fshape

        self.__a = None
        self.__z = None

        # Cache for convolutions with im2col
        self.cache = None

        if input_shape is not None:

            self.build(input_shape)

        if eval_method in ['scipy', 'fft', 'im2col']:
            self.eval_method = eval_method
        else:
            raise NotImplementedError(f"The evaluation method {eval_method}"
                                      "is not implemented. Choose \'scipy\',"
                                      "\'fft\' or \'im2col\'.")

    def build(self, input_shape: tuple[int, int, int]) -> None:
        """
        Build the convolutional layer.

        The input shape is set, the output shape is computed and
        the filter bank as well as the biases are initialized.

        Parameters
        ----------
        input_shape : tuple[int, int, int]
            Shape of the inputs (Whithout the batch dimension).
        """

        m, fh, fw = self.fshape
        c, h, w = input_shape

        self.tensor = input_shape
        self.output_shape = (m, h - fh + 1, w - fw + 1)

        self.f = self.initializer.ffun(m, c, fh, fw)
        self.f *= self.afun.factor
        self.b = np.zeros(m)

        self.df = np.zeros_like(self.f)
        self.db = np.zeros_like(self.b)

        self.built = True

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Evaluate the layer with the method defined on intialization.

        Depending on the evaluation method selected on initialization
        the corresponding method is called to evaluate the layer.

        Parameters
        ----------
        input : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Activated output of the layer.
        """

        if self.eval_method == 'scipy':
            return self.evaluate_scipy(input)
        if self.eval_method == 'fft':
            return self.evaluate_fft(input)
        if self.eval_method == 'im2col':
            return self.evaluate_im2col(input)

    def evaluate_scipy(self, a: np.ndarray) -> np.ndarray:
        """
        Evaluate the layer with scipy's convolution function.

        Parameters
        ----------
        a : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Output array.
        """

        n, c, h, w = a.shape
        m, fh, fw = self.fshape

        m, zh, zw = self.output_shape
        z = np.zeros((n, m, zh, zw))
        for j in range(m):
            for i in range(n):
                z[i, j, :, :] += self.b[j]
                for k in range(c):
                    z[i, j, :, :] += convolve2d(a[i, k, :, :],
                                                self.f[j, k, :, :],
                                                mode='valid')

        return self.afun.feedforward(z)

    def evaluate_fft(self, a: np.ndarray) -> np.ndarray:
        """
          Evaluate the layer, where the convolutions are
          computed with FFTs.

          Parameters
          ----------
          a : np.ndarray
              Input array.

          Returns
          -------
          np.ndarray
              Output array.
        """
        # TODO Implement the evaluation of the CNN layer with the FFT approach
        # This can be done by
        # - transforming a and self.f with a 2d DFT (numpy.fft.rfft2)
        # - computing the componentwise product of the transformed input and filter
        # - transforming the result back with an IDFT (this yields the full convolution)
        # - restricting the result of the previous step to the valid convolution

        pass

    def evaluate_im2col(self, a: np.ndarray) -> np.ndarray:
        """
          Evaluate the layer, where the convolutions are
          computed with the im2col method.

          Parameters
          ----------
          a : np.ndarray
              Input array.

          Returns
          -------
          np.ndarray
              Output array.
        """
        n, c, h, w = a.shape
        m, fh, fw = self.fshape
        zh, zw = h - fh + 1, w - fw + 1

        # TODO create im2col matrix with amllib.utils.im2col
        a_im2col = im2col(a, fh, fw)
        # TODO reshape filterbank to a m by c*fh*fw matrix
        f_row = self.f.reshape(m, -1)
        # TODO Compute Matrix product
        # BLAS Level 3: GEMM
        z_mat = f_row @ a_im2col + self.b[:, np.newaxis]
        # reshape
        z = z_mat.reshape(m, n, zh, zw).transpose(1, 0, 2, 3)
        return self.afun(z)

    def feedforward(self, input: np.ndarray) -> np.ndarray:
        """
        Evaluate the layer with the method defined on intialization.

        Depending on the evaluation method selected on initialization
        the corresponding method is called to evaluate the layer.
        Data is cached for backpropagation.

        Parameters
        ----------
        input : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Activated output of the layer.
        """
        # Build the layer with the shape of the inputs, if not done yet.
        if not self.built:
            self.build(input.shape[1:])

        if self.eval_method == 'scipy':
            return self.feedforward_scipy(input)
        if self.eval_method == 'fft':
            return self.feedforward_fft(input)
        if self.eval_method == 'im2col':
            return self.feedforward_im2col(input)

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Backpropagate through the layer depending on the set
        evaluation method.

        The backpropagation method corresponding to the evaluation
        method defined on initialization is called.

        Parameters
        ----------
        delta : np.ndarray
            Derivative of the cost function by the output.

        Returns
        -------
        np.ndarray
            The derivative of the cost function by the input.
        """

        if self.eval_method == 'scipy':
            return self.backprop_scipy(delta)
        if self.eval_method == 'fft':
            return self.backprop_fft(delta)
        if self.eval_method == 'im2col':
            return self.backprop_im2col(delta)

    def feedforward_scipy(self, a: np.ndarray) -> np.ndarray:
        """
        Evaluate the layer with scipy's convolution function.

        Parameters
        ----------
        a : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Output array.
        """

        n, c, h, w = a.shape
        m, fh, fw = self.fshape

        self.__a = a

        m, zh, zw = self.output_shape
        self.__z = np.zeros((n, m, zh, zw))
        for j in range(m):
            for i in range(n):
                self.__z[i, j, :, :] += self.b[j]
                for k in range(c):
                    self.__z[i, j, :, :] += convolve2d(self.__a[i, k, :, :],
                                                       self.f[j, k, :, :],
                                                       mode='valid')

        return self.afun.feedforward(self.__z)

    def feedforward_fft(self, a: np.ndarray) -> np.ndarray:
        """
          Evaluate the layer, where the convolutions are
          computed with FFTs.

          Parameters
          ----------
          a : np.ndarray
              Input array.

          Returns
          -------
          np.ndarray
              Output array.
        """

        # TODO Implement the evaluation of the CNN layer with the FFT approach.
        # This can be done by
        # - transforming a and self.f with a 2d DFT (numpy.fft.rfft2)
        # - computing the componentwise product of the transformed input and filter
        # - transforming the result back with an IDFT (this yields the full convolution)
        # - restricting the result of the previous step to the valid convolution
        # Cache the non-activated convolution for the backpropagation, and use
        # self.afun.feedforward to activate the output.
        pass

    def feedforward_im2col(self, a: np.ndarray) -> np.ndarray:
        """
          Evaluate the layer, where the convolutions are
          computed with the im2col method.

          Parameters
          ----------
          a : np.ndarray
              Input array.

          Returns
          -------
          np.ndarray
              Output array.
        """
        self.__a = a

        n, c, h, w = a.shape
        m, fh, fw = self.fshape
        zh, zw = h - fh + 1, w - fw + 1

        # TODO create im2col matrix with amllib.utils.im2col
        a_im2col = im2col(a, fh, fw)
        # TODO reshape filterbank to a m by c*fh*fw matrix
        f_row = self.f.reshape(m, -1)
        # TODO Cache both matrices
        self.cache = (a_im2col, f_row)

        # TODO Compute Matrix product
        # BLAS Level 3: GEMM
        z_mat = f_row @ a_im2col + self.b[:, np.newaxis]

        # reshape
        self.__z = z_mat.reshape(m, n, zh, zw).transpose(1, 0, 2, 3)
        return self.afun.feedforward(self.__z)

    def backprop_scipy(self, delta: np.ndarray) -> np.ndarray:
        """
        Backpropagation for the evaluation with scipy's
        convolution function.

        The derivatives of the cost function by the filter bank,
        the bias and the input are computed from the derivative of
        the cost function by the output.


        Parameters
        ----------
        delta : np.ndarray
            Derivative of the cost function by the output.

        Returns
        -------
        np.ndarray
            Derivative of the cost function by the input.
        """
        n, c, h, w = self.__a.shape
        m, fh, fw = self.fshape
        # Compute a'(z)*delta
        delta = self.afun.backprop(delta)

        # Compute bias change
        self.db = np.sum(delta, axis=(0, 2, 3))

        # Compute df = delta * a^~. df has shape (m, c, fh, fw)
        for k in range(c):
            for j in range(m):
                for i in range(n):
                    self.df[j, k, :, :] += convolve2d(self.__a[i, k, ::-1, ::-1],
                                                      delta[i, j, :, :],
                                                      mode='valid')

        # Update delta
        # The new delta has the same shape as __a: (n,c,h,w)
        delta2 = np.zeros_like(self.__a)
        for k in range(c):
            for i in range(n):
                for j in range(m):
                    delta2[i, k, :, :] += convolve2d(delta[i, j, :, :],
                                                     self.f[j, k, ::-1, ::-1])

        return delta2

    def backprop_fft(self, delta):
        """
        Backpropagation for the evaluation with FFTs.

        The derivatives of the cost function by the filter bank,
        the bias and the input are computed from the derivative of
        the cost function by the output.


        Parameters
        ----------
        delta : np.ndarray
            Derivative of the cost function by the output.

        Returns
        -------
        np.ndarray
            Derivative of the cost function by the input.
        """
        n, c, h, w = self.__a.shape
        m, fh, fw = self.fshape
        dh, dw = h - fh, w - fw

        delta = self.afun.backprop(delta)

        # Compute bias change
        self.db = np.sum(delta, axis=(0, 2, 3))

        # Compute filter change
        a_hat = rfft2(self.__a[:, :, ::-1, ::-1], (h + dh, w + dw))
        delta_hat = rfft2(delta, (h + dh, w + dw))

        for k in range(c):
            for j in range(m):

                df_hat = irfft2(a_hat[:, k, :, :] *
                                delta_hat[:, j, :, :],
                                (h + dh, w + dw))
                self.df[j, k, :, :] = df_hat[:, dh:dh+fh, dw:dw+fw].sum(axis=0)

        # Compute the next delta
        delta_hat = rfft2(delta, (h, w))
        f_hat = rfft2(self.f[:, :, ::-1, ::-1], (h, w))

        da = np.zeros_like(self.__a)
        for i in range(n):
            for k in range(c):

                da_hat = irfft2(delta_hat[i, :, :, :] *
                                f_hat[:, k, :, :],
                                (h, w))
                da[i, k, :, :] = da_hat.sum(axis=0)

        return da

    def backprop_im2col(self, delta):
        """
        Backpropagation for the evaluation with the im2col method.

        The derivatives of the cost function by the filter bank,
        the bias and the input are computed from the derivative of
        the cost function by the output.


        Parameters
        ----------
        delta : np.ndarray
            Derivative of the cost function by the output.

        Returns
        -------
        np.ndarray
            Derivative of the cost function by the input.
        """
        n, c, h, w = self.__a.shape
        m, fh, fw = self.fshape

        a_col, f_row = self.cache

        delta = self.afun.backprop(delta)

        self.db = np.sum(delta, axis=(0, 2, 3))

        # reshape delta
        delta_mat = delta.transpose(1, 0, 2, 3).reshape(m, -1)

        # BLAS Level 3: GEMM for filter change
        df_row = delta_mat @ a_col.T
        # reshape df_row into df
        self.df = df_row.reshape(m, c, fh, fw)[:, :, ::-1, ::-1]

        # BLAS Level 3: GEMM for da_col
        da_col = f_row.T @ delta_mat
        # reshape back to image (col2im)
        return col2im(da_col, self.__a.shape, fh, fw)

    def get_output_shape(self) -> tuple[int, int, int]:
        """
        Return the shape of the output of the layer.

        Returns
        -------
        tuple[int, int, int]
            Output shape.
        """
        return self.output_shape

    def update(self):
        """
        Update all parameters of the layer.

        The filter bank and the bias are updated with
        the derivatives computed by the backpropagation method.

        To update the parameters the optimizer of the layer is used.
        """
        self.optim.update([self.f, self.b],
                          [self.df, self.db])
