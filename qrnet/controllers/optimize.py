import numpy as np
from tqdm import tqdm
from tensorflow import train, global_variables_initializer
from tensorflow.contrib.opt import ExternalOptimizerInterface

from qrnet.utilities import get_batches, shuffle_data

_default_bfgs_opts = {
    'iprint': 50, 'maxfun': 30000, 'maxiter': 30000, 'average_weights': False
}

def optimize(
        loss_fun, tf_session,
        data_keys, data_vals, parameter_keys=[], parameter_vals=[],
        batch_size=None, n_epochs=1, optimizer='L-BFGS-B', options={},
        var_to_bounds={}, callback=None, callback_epoch=1
    ):
    '''
    Optimizes a given loss function.

    Parameters
    ----------
    loss_fun : scalar tensor
        Loss function tensor.
    tf_session : tf.Session instance
        Session to operate the graph.
    data_keys : list of tf.placeholders
        tf.placeholders to use as keys in the tensorflow feed dict.
    data_vals : list of arrays
        Numpy arrays of data to use as values in the tensorflow feed dict. Must
        all have the same number of entries in their last axis.
    parameter_keys : list of tf.placeholders, optional
        Scalar tf.placeholders used as keys in the tensorflow feed dict.
    parameter_vals : list of floats, optional
        Python or numpy floats of parameter values to use as values in the
        tensorflow feed dict.
    batch_size : int, optional
        Number of data points to use in each minibatch. Defaults to the whole
        dataset.
    n_epochs : int, default=1
        Number of times to pass through the dataset.
    optimizer : str, default='AdamOptimizer'
        Which optimizer to use. See tensorflow.train for options.
    options : dict, optional
        Options to pass to to the optimizer. See tensorflow.train documentation.
    var_to_bounds : dict, optional
        Maps variables to lower and upper bounds: keys are tensors and values
        are tuples of arrays
    callback : callable, optional
        Function to call every callback_epoch epochs. Takes two arguments: a
        tensorflow feed dict and an instance of a tf.Session.
    callback_epoch : int, default=1
        Specifies after how many epochs to call callback.
    '''
    if hasattr(train, optimizer):
        _optimize_SGD(
            loss_fun=loss_fun,
            tf_session=tf_session,
            data_keys=list(data_keys),
            data_vals=list(data_vals),
            parameter_keys=list(parameter_keys),
            parameter_vals=list(parameter_vals),
            batch_size=batch_size,
            n_epochs=n_epochs,
            optimizer=optimizer,
            options=options,
            callback=callback,
            callback_epoch=callback_epoch
        )
    else:
        _optimize_L_BFGS_B(
            loss_fun=loss_fun,
            tf_session=tf_session,
            data_keys=list(data_keys),
            data_vals=list(data_vals),
            parameter_keys=list(parameter_keys),
            parameter_vals=list(parameter_vals),
            batch_size=batch_size,
            options=options,
            callback=callback
        )

def _optimize_SGD(
        loss_fun, tf_session,
        data_keys, data_vals, parameter_keys=[], parameter_vals=[],
        batch_size=None, n_epochs=1, optimizer='AdamOptimizer', options={},
        callback=None, callback_epoch=1
    ):
    '''
    Implements optimization of the loss function with a specified variant of
    stochastic gradient descent.

    Parameters
    ----------
    loss_fun : scalar tensor
        Loss function tensor.
    tf_session : tf.Session instance
        Session to operate the graph.
    data_keys : list of tf.placeholders
        tf.placeholders to use as keys in the tensorflow feed dict.
    data_vals : list of arrays
        Numpy arrays of data to use as values in the tensorflow feed dict. Must
        all have the same number of entries in their last axis.
    parameter_keys : list of tf.placeholders, optional
        Scalar tf.placeholders used as keys in the tensorflow feed dict.
    parameter_vals : list of floats, optional
        Python or numpy floats of parameter values to use as values in the
        tensorflow feed dict.
    batch_size : int, optional
        Number of data points to use in each minibatch. Defaults to the whole
        dataset.
    n_epochs : int, default=1
        Number of times to pass through the dataset.
    optimizer : str, default='AdamOptimizer'
        Which optimizer to use. See tensorflow.train for options.
    options : dict, optional
        Options to pass to to the optimizer. See tensorflow.train documentation.
    callback : callable, optional
        Function to call every callback_epoch epochs. Takes two arguments: a
        tensorflow feed dict and an instance of a tf.Session.
    callback_epoch : int, default=1
        Specifies after how many epochs to call callback.
    '''

    # Initialize the optimizer and training ops
    optimizer = getattr(train, optimizer)(**options)
    train_step = optimizer.minimize(loss_fun)

    tf_session.run(global_variables_initializer())

    n_data_available = data_vals[0].shape[-1]

    if not batch_size or batch_size > n_data_available:
        batch_size = n_data_available

    print('\nBatch size = %d\n' % batch_size)

    feed_dict = dict(zip(parameter_keys, parameter_vals))

    # Loop over epochs
    for epoch in tqdm(range(n_epochs)):
        # Pre-shuffle the data for the epoch
        shuffle_data(data_vals, n_data_available)

        # Loop over minibatches of shuffled data
        batches = get_batches(
            n_data_available, batch_size, force_batch_size=True
        )
        for batch_idx in batches:
            feed_dict.update(dict(zip(
                data_keys, [data[...,batch_idx] for data in data_vals]
            )))

            tf_session.run(train_step, feed_dict=feed_dict)

        # Output progress
        if callable(callback) and callback_epoch:
            if epoch % callback_epoch == 0:
                callback(feed_dict, tf_session)

def _optimize_L_BFGS_B(
        loss_fun, tf_session,
        data_keys, data_vals, parameter_keys=[], parameter_vals=[],
        batch_size=None, options={}, var_to_bounds={}, callback=None
    ):
    '''
    Implements optimization of the loss function with L-BFGS-B.

    Parameters
    ----------
    loss_fun : scalar tensor
        Loss function tensor.
    tf_session : tf.Session instance
        Session to operate the graph.
    data_keys : list of tf.placeholders
        tf.placeholders to use as keys in the tensorflow feed dict.
    data_vals : list of arrays
        Numpy arrays of data to use as values in the tensorflow feed dict. Must
        all have the same number of entries in their last axis.
    parameter_keys : list of tf.placeholders, optional
        Scalar tf.placeholders used as keys in the tensorflow feed dict.
    parameter_vals : list of floats, optional
        Python or numpy floats of parameter values to use as values in the
        tensorflow feed dict.
    batch_size : int, optional
        Maximum number of data points to use for computing the loss. Defaults to
        the whole dataset.
    options : dict, optional
        Options to pass to L-BFGS-B. See scipy.optimize.minimize documentation.
        We set the defaults to
            'iprint': 50
            'maxiter': 30000
            'maxfun': 30000
        One additional option is available:
            'average_weights' : bool, default=True
            If True then average the weights of the NN after each batch with the
            weights from the previous batch.
    var_to_bounds : dict, optional
        Maps variables to lower and upper bounds: keys are tensors and values
        are tuples of arrays
    callback : callable, optional
        Function to call after optimization. Takes two arguments: a
        tensorflow feed dict and an instance of a tf.Session.
    '''
    options = {**_default_bfgs_opts, **options}
    average_weights = options.pop('average_weights')

    optimizer = ScipyOptimizerInterface(
        loss_fun,
        var_to_bounds=var_to_bounds,
        options=options
    )

    n_data_available = data_vals[0].shape[-1]

    if not batch_size or batch_size > n_data_available:
        batch_size = n_data_available

    feed_dict = dict(zip(parameter_keys, parameter_vals))

    # Pre-shuffle the data
    shuffle_data(data_vals, n_data_available)

    # Loop over batches of shuffled data
    batches = get_batches(
        n_data_available, batch_size, force_batch_size=False
    )
    for k, batch_idx in enumerate(batches):
        print('\nBatch size = %d\n' % (batch_idx.stop - batch_idx.start))

        feed_dict.update(dict(zip(
            data_keys, [data[...,batch_idx] for data in data_vals]
        )))

        optimizer.minimize(tf_session, feed_dict=feed_dict)

        # Update weights with average of previous and new weights
        if average_weights:
            if k >= 1:
                new_weights = [
                    (saved + new)/2. for saved, new
                    in zip(saved_weights, tf_session.run(optimizer._vars))
                ]
                tf_session.run(
                    optimizer._var_updates,
                    feed_dict=dict(zip(optimizer._update_placeholders, new_weights))
                )
            saved_weights = tf_session.run(optimizer._vars)

        if callable(callback):
            callback(feed_dict, tf_session)

class ScipyOptimizerInterface(ExternalOptimizerInterface):
    _DEFAULT_METHOD = 'L-BFGS-B'

    def _minimize(
            self, initial_val, loss_grad_func, equality_funcs,
            equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
            packed_bounds, step_callback, optimizer_kwargs
        ):

        def loss_grad_func_wrapper(x):
            loss, gradient = loss_grad_func(x)
            return loss, gradient.astype('float64')

        optimizer_kwargs = dict(optimizer_kwargs.items())
        method = optimizer_kwargs.pop('method', self._DEFAULT_METHOD)

        constraints = []
        for func, grad_func in zip(equality_funcs, equality_grad_funcs):
            constraints.append({'type': 'eq', 'fun': func, 'jac': grad_func})
        for func, grad_func in zip(inequality_funcs, inequality_grad_funcs):
            constraints.append({'type': 'ineq', 'fun': func, 'jac': grad_func})

        minimize_args = [loss_grad_func_wrapper, initial_val]
        minimize_kwargs = {
            'jac': True,
            'callback': step_callback,
            'method': method,
            'constraints': constraints,
            'bounds': packed_bounds,
        }

        for kwarg in minimize_kwargs:
            if kwarg in optimizer_kwargs:
                if kwarg == 'bounds':
                    # Special handling for 'bounds' kwarg since ability to specify bounds
                    # was added after this module was already publicly released.
                    raise ValueError(
                        'Bounds must be set using the var_to_bounds argument')
                raise ValueError(
                    'Optimizer keyword arg \'{}\' is set '
                    'automatically and cannot be injected manually'.format(kwarg))

        minimize_kwargs.update(optimizer_kwargs)

        from scipy.optimize import minimize
        result = minimize(*minimize_args, **minimize_kwargs)

        return result['x']
