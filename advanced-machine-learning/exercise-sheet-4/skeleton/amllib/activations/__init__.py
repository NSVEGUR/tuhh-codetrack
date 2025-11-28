"""
Activation function package of the neural network library
for "Advanced Machine Learning".
"""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations"
__package__ = "amllib.activations"

from .activation import Activation

# Import Heaviside-like activation functions
from .heaviside_like.heaviside import Heaviside
from .heaviside_like.modified_heaviside import ModifiedHeaviside
from .heaviside_like.logistic import Logistic
from .heaviside_like.softmax import SoftMax
from .heaviside_like.exponential import Exp
# Import Abs-like activation functions
from .abs_like.abs import Abs
from .abs_like.absalpha import AbsAlpha
from .abs_like.loco import LOCo
from .abs_like.softabs import SoftAbs
from .abs_like.twist import Twist
# Import ReLU-like activation functions
from .relu_like.relu import ReLU
from .relu_like.leakyrelu import LeakyReLU
from .relu_like.elu import ELU
from .relu_like.softplus import SoftPlus
from .relu_like.swish import Swish
from .relu_like.linear import Linear
# Import Sign-like activation functions
from .sign_like.sign import Sign
from .sign_like.tanh import TanH
from .sign_like.softsign import SoftSign
