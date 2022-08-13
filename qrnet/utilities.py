import numpy as np

def get_batches(n_data, batch_size, force_batch_size=False):
    '''
    Generates slices for taking subsets of arrays.

    Parameters
    ----------
    n_data : int
        Number of data points in the arrays of interest.
    batch_size : int
        The size of each slice to take. The last slice generated may be smaller
        than this.
    force_batch_size : bool, default=False
        If True and batch_size doesn't evenly divide n_data, the last batch will
        be omitted so that all batches have the same batch size.

    Yields
    ------
    batch_slice : slice
        The ith iterate is a slice from batch_size*i to batch_size*(i+1). If
        batch_size doesn't evenly divide n_data and force_batch_size=False, then
        the last batch will be larger to include all data.
    '''
    n_batches = max(1, n_data // batch_size)

    if force_batch_size or n_batches*batch_size==n_data:
        combine_last = False
    else:
        combine_last = True

    for i in range(n_batches):
        # If reached the end make the last batch bigger to include all data
        if combine_last and batch_size*(i+2)>n_data:
            yield slice(batch_size*i, n_data)
        else:
            yield slice(batch_size*i, batch_size*(i+1))


def shuffle_data(dataset, n_data):
    '''
    Shuffles a list of arrays in place with a common reindexing. Note that the
    list is modified in place but the original arrays are copied. Shuffles only
    those arrays whose last index has n_data elements.

    Parameters
    ----------
    dataset : list of arrays and/or floats
        Data to shuffle, for example dataset=[X, Y, c], where X, Y are
        (n_x, n_data) and (n_y, n_data) arrays and c is a float.
    n_data : int
        The number of data points in the dataset. Each entry X of dataset will
        be shuffled X.shape[-1] == n_data.
    '''
    shuffle_idx = np.random.permutation(n_data)

    for i, data in enumerate(dataset):
        if np.ndim(data) >= 1 and np.shape(data)[-1] == n_data:
            dataset[i] = data[...,shuffle_idx]

# ---------------------------------------------------------------------------- #

def saturate_np(U, U_lb, U_ub):
    '''
    Hard saturation of control for numpy arrays.

    Parameters
    ----------
    U : (n_controls, n_data) or (n_controls,) array
        Control(s) to saturate.
    U_lb : (n_controls, 1) array
        Lower control bounds.
    U_ub : (n_controls, 1) array
        Upper control bounds.

    Returns
    -------
    U : array with same shape as input
        Control(s) saturated between U_lb and U_ub
    '''
    if U_lb is not None or U_ub is not None:
        if U.ndim < 2:
            U = np.clip(U, U_lb.flatten(), U_ub.flatten())
        else:
            U = np.clip(U, U_lb, U_ub)

    return U

def saturate_tf(U, U_lb, U_ub):
    '''
    Hard saturation of control for tensorflow variables.

    Parameters
    ----------
    U : (n_controls, None) tensor
        Control tensor to saturate.
    U_lb : (n_controls, 1) array
        Lower control bounds.
    U_ub : (n_controls, 1) array
        Upper control bounds.

    Returns
    -------
    U : (n_controls, None) tensor
        Controls saturated between U_lb and U_ub
    '''
    from tensorflow import clip_by_value

    if U_lb is not None and U_ub is not None:
        U = clip_by_value(U, U_lb, U_ub)
    elif U_lb is not None:
        U = clip_by_value(U, U_lb, np.inf)
    elif U_ub is not None:
        U = clip_by_value(U, -np.inf, U_ub)

    return U

def cross_product_matrix(w):
    zeros = np.zeros_like(w[0])
    wx = np.array([
        [zeros, -w[2], w[1]],
        [w[2], zeros, -w[0]],
        [-w[1], w[0], zeros]]
    )
    return wx
