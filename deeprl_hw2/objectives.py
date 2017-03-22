"""Loss functions."""
from keras import backend as K
import tensorflow as tf
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

    diff = tf.abs(y_true - y_pred)
    leqval = 0.5 * tf.square(diff)
    gval = 0.5 * diff - 0.5 * tf.square(max_grad)
    return tf.where(tf.less(diff, max_grad), leqval, gval)

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

    diff = tf.abs(y_true - y_pred)
    leqval = 0.5 * tf.square(diff)
    gval = 0.5 * diff - 0.5 * tf.square(max_grad)
    huber_loss = tf.where(tf.less(diff, max_grad), leqval, gval)

    return tf.reduce_mean(huber_loss)
