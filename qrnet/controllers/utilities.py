import os
import datetime
import dill
import numpy as np
import tensorflow as tf

def xavier_init(n_in, n_out):
    '''
    Xavier normal initialization for dense network weights.

    Parameters
    ----------
    n_in : int
        Number of input neurons
    n_out : int
        Number of output neurons

    Returns
    -------
    W : (n_out, n_in) tensor
        Tensorflow variable initialized from a normal distribution
    '''
    std = np.sqrt(2. / (n_in + n_out))
    init = std * np.random.randn(n_out, n_in)
    return tf.Variable(init, dtype=tf.float32)

def initialize_dense(n_in, n_out, n_hidden, n_neurons, weights=None):
    '''
    Initializes tensorflow variables corresponding to weights and biases for a
    standard dense feedforward neural network. Weights and biases are stored in
    a list arranged as
        [weights(layer1), biases(layer1), weights(layer2), biases(layer2), ...]

    Parameters
    ----------
    n_in : int
        Number of input variables to the NN
    n_out : int
        Number of output variables from the NN
    n_hidden : int
        Number of hidden layers
    n_neurons : int
        Number of neurons per hidden layer
    weights : list of numpy arrays, optional
        List of pre-trained weights and biases for each layer

    Returns
    -------
    tf_weights : list of tensors
        Initialized tensorflow variables for the NN weights and biases
    '''
    tf_weights = []

    if weights is None or len(weights) < 2:
        layers = [n_in] + n_hidden * [n_neurons] + [n_out]

        for l in range(n_hidden + 1):
            tf_weights.append(
                xavier_init(n_in=layers[l], n_out=layers[l+1])
            )
            tf_weights.append(
                tf.Variable(tf.zeros((layers[l+1], 1), dtype=tf.float32))
            )
    else:
        for l in range(len(weights)):
            tf_weights.append(tf.Variable(weights[l], dtype=tf.float32))

    return tf_weights

def make_dense_graph(X, X_scale, weights, activation='tanh'):
    '''
    Makes a tensorflow computational graph for a standard dense feedforward
    neural network.

    Parameters
    ----------
    X : (n_in, n_data) tensor
        Inputs to the network
    X_scale : (n_in, 1) tensor or array
        Scale vector to multiply the inputs by
    weights : list of tensors
        List of weights and biases for each layer, arranged as
        [weights(layer1), biases(layer1), weights(layer2), biases(layer2), ...]
    activation : str or list of strs, default='tanh'
        Activation function to use. Can also be a list, in which case the
        activation functions are specified per hidden layer.

    Returns
    -------
    Y : (n_out, n_data) tensor
        Neural network predictions for each X
    '''
    n_hidden = int(len(weights) / 2) - 1

    if not isinstance(activation, list):
        activation = n_hidden * [activation]

    Y = X_scale * X

    for l in range(n_hidden + 1):
        W = weights[2*l]
        b = weights[2*l+1]
        Y = tf.matmul(W, Y) + b

        if l < len(activation):
            # TODO: choice of activation functions
            Y = tf.tanh(Y)

    return Y

def tf_jacobian(Y, X, stop_gradients=None):
    '''
    Compute the Jacobian, dYdX, of a vector-valued tensorflow graph Y=Y(X).

    Parameters
    ----------
    Y : (n_out, ?) tensor
        Dependent variables
    X : (n_in, ?) tensor
        Independent variables. X.shape[1] must be compatible with Y.shape[1]
    stop_gradients : tensor or list of tensors, optional
        Variables to be held constant during differentiation

    Returns
    -------
    dYdX : (n_out, n_in, ?) tensor
        The Jacobian dYdX of Y=Y(X) at each input point
    '''
    dYdX = [
        tf.gradients(Y[d], X, stop_gradients=stop_gradients)
        for d in range(Y.shape[0])
    ]
    return tf.concat(dYdX, axis=0)

# ---------------------------------------------------------------------------- #

def create_NN(architecture, LQR, **kwargs):
    '''
    Convenience function to initialize an NN with the architecture specified in
    the config file. Can be a new NN or an existing one.

    Parameters
    ----------
    architecture : str
        NN architecture to instantiate. See QRnet.controllers for options.
    LQR : object
        Instance of controllers.linear_quadratic_regulator.LQR.
    kwargs : name-value pairs
        Keyword arguments to pass to the controller.

    Returns
    -------
    controller : object
        Instantiated BaseNN subclass.
    '''
    from .model_factory import get_model_class

    if architecture == 'LQR':
        return LQR

    NN = get_model_class(architecture)

    return NN(LQR, **kwargs)

def load_NN(model_dir, timestamp=-1, verbose=True):
    '''
    Load a previously trained NN model from <model_dir>/<timestamp>.pkl.

    Parameters
    ----------
    model_dir : path-like
        Which folder to load model parameters from
    timestamp : int or str, optional
        UTC time at which the model was saved, without milliseconds. If
        timestamp is not in the list of timestamps in model_info.csv, then sorts
        this chronologically and treats the timestamp argument as a python list
        index to select from the timestamps list. If this fails, picks the most
        recently trained model.

    Returns
    -------
    controller : instantiated BaseNN subclass
        Model ready to use for control
    timestamp : int
        UTC time at which the model was saved, without milliseconds
    '''
    if timestamp is None:
        timestamp = -1
    timestamp = int(timestamp)

    timestamps_list = [
        int(fn[:-4]) for fn in os.listdir(model_dir) if fn[-4:] == '.pkl'
    ]

    if timestamp not in timestamps_list:
        timestamps_list = np.sort(timestamps_list)
        try:
            timestamp = timestamps_list[timestamp]
        except IndexError:
            timestamp = timestamps_list[-1]

    model_path = os.path.join(model_dir, str(timestamp) + '.pkl')

    with open(model_path, 'rb') as model_file:
        model_dict = dill.load(model_file)

    if verbose:
        load_msg = 'Loading {a:s} (timestamp {ts:d})...'
        print(load_msg.format(a=model_dict['architecture'], ts=timestamp))

    controller = create_NN(**model_dict)

    return controller, timestamp
