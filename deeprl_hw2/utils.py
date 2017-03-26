"""Common functions you may find useful in your implementation."""
from keras.models import Model, Sequential
import semver
import pickle

def get_soft_target_model_updates(target, source, tau):
    """Return list of target model update ops.

    These are soft target updates. Meaning that the target values are
    slowly adjusted, rather than directly copied over from the source
    model.

    The update is of the form:

    $W' \gets (1- \tau) W' + \tau W$ where $W'$ is the target weight
    and $W$ is the source weight.

    Parameters
    ----------
    target: keras.models.Model
      The target model. Should have same architecture as source model.
    source: keras.models.Model
      The source model. Should have same architecture as target model.
    tau: float
      The weight of the source weights to the target weights used
      during update.

    Returns
    -------
    list(tf.Tensor)
      List of tensor update ops.
    """
    config_src = source.get_config()
    weights_src = source.get_weights()
    target = Model.from_config(config_src)
    weights_tgt = target.get_weights()
    target.set_weights((1 - tau) * weights_tgt + tau * weights_src)
    return target


def get_hard_target_model_updates(target, source):
    """Return list of target model update ops.

    These are hard target updates. The source weights are copied
    directly to the target network.

    Parameters
    ----------
    target: keras.models.Model
      The target model. Should have same architecture as source model.
    source: keras.models.Model
      The source model. Should have same architecture as target model.

    Returns
    -------
    list(tf.Tensor)
      List of tensor update ops.
    """
    config_src = source.get_config()
    weights_src = source.get_weights()
    target = Sequential.from_config(config_src)
    target.set_weights(weights_src)
    return target

