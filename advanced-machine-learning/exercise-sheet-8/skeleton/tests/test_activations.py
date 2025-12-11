import numpy as np
import matplotlib.pyplot as plt

# import heaviside-like activation functions
from amllib.activations import Heaviside, ModifiedHeaviside
from amllib.activations import Logistic, Exp
# import sign-like activation functions
from amllib.activations import Sign, SoftSign, TanH
# import relu-like activation functions
from amllib.activations import ReLU, LeakyReLU, ELU
from amllib.activations import SoftPlus, Swish, Linear
# import abs-like activation functions
from amllib.activations import Abs, AbsAlpha, LOCo, SoftAbs, Twist

if __name__ == '__main__':

    plt.figure(figsize=(8, 6))

    heaviside_like = [Heaviside, ModifiedHeaviside,
                      Logistic, Exp]
    sign_like = [Sign, SoftSign, TanH]
    relu_like = [ReLU, LeakyReLU, ELU]
    abs_like = [Abs, AbsAlpha, LOCo, SoftAbs, Twist]

    for afun in heaviside_like + sign_like + relu_like + abs_like:

        # clear plot
        plt.clf()

        # generate activation function
        f = afun()

        # evaluate activation function and derivative
        x = np.linspace(-5, 5, 1001)
        y = f(x)
        z = f.derive(x)

        # test feedfoward and backprop, should also give derivative
        a = f.feedforward(x)
        g = f.backprop(np.ones_like(x))

        # plot function and derivative
        plt.plot(x, y, '-', label=f"{f.name}")
        plt.plot(x, z, '--', label=f"{f.name}': derive")
        plt.plot(x, g, '-.', label=f"{f.name}': feedforward & backprop")
        plt.legend()
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.title(f'Activation function {f.name} w/ derivative')
        plt.pause(1)
