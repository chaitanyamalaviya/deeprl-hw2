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
    if np.abs(y_true - y_pred) <= max_grad:
      return 0.5 * np.square(y_true - y_pred)
    else:
      return 0.5 * np.abs(y_true - y_pred) - 0.5 * max_grad**2


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
    print(type(K.eval(y_true)))
    # seess = tf.Session()
    # y_true_val = y_true.eval(session=seess)
    # y_pred_val = y_pred.eval(session=seess)

    y_true_val = K.eval(y_true)
    y_pred_val = K.eval(y_pred)

    if np.abs(y_true_val - y_pred_val) <= max_grad:
      huber_loss = 0.5 * np.square(y_true_val - y_pred_val)
    else:
      huber_loss = 0.5 * np.abs(y_true_val - y_pred_val) - 0.5 * max_grad**2

    return tf.reduce_mean(huber_loss)
