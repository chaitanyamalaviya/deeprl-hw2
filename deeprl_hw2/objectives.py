"""Loss functions."""
from keras import backend as K
import theano.tensor as T
import semver
import numpy as np

def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    pass


def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """

    y_pred = y_pred * T.extra_ops.to_one_hot(T.argmax(T.abs_(y_true), axis=1), y_pred.shape[1])

    diff = T.abs_(y_true - y_pred)
    leqval = 0.5 * T.square(diff)
    gval = max_grad * diff - 0.5 * T.square(max_grad)
    huber_loss = T.switch(diff <= max_grad, leqval, gval)

    return T.mean(huber_loss)
