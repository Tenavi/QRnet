import numpy as np
from scipy.optimize import root

from .utilities import get_batches

def eval_errors(NN, data, batch_size=4096, return_predictions=False):
    '''
    Evaluate a set of error metrics for a given test (or train) data set.

    Parameters
    ----------
    controller : instantiated BaseNN subclass
        Model to evaluate error metrics for
    data : dict containing
        X : (n_states, n_data) array
            Input state data
        U : (n_controls, n_data) array
            Optimal control data
        dVdX : (n_states, n_data) array
            Value function gradient data
        V : (1, n_data) array
            Value function data
    batch_size : int, default=4096
        Number of data points to pass to the controller at once. Set smaller or
        larger to adjust memory footprint.
    return_predictions : bool, default=False
        If return_predictions=True then output will also contain raw NN
        predictions.

    Returns
    -------
    error_dict : dict containing (a subset of)
        U_maxL2 : maximum L2 error in control over test data set
        U_ML2 : mean L2 error in control over test data set
        U_RML2 : mean L2 error in control over test data set, scaled
            by the maximum control L2 norm
        U_pred : NN predictions of the control
        dVdX_ML2 : mean L2 error in value gradient over test data set
        dVdX_RML2 : mean L2 error in value gradient over test data set,
            scaled by the maximum value gradient L2 norm
        dVdX_pred : NN predictions of the value gradient
        V_MAE : mean absolute error in the value function over test data
        V_RMAE : mean absolute error in the value function over test
            data set, scaled by the maximum value of the value function data
        V_pred : NN predictions of the value function
    '''
    error_dict = {}

    def batch_predict(pred_fun):
        batches = get_batches(data['X'].shape[-1], batch_size)
        pred = [pred_fun(data['X'][:,batch_idx]) for batch_idx in batches]
        return np.hstack(pred)

    try:
        U_pred = batch_predict(NN.eval_U)
        U_err = np.linalg.norm(U_pred - data['U'], axis=0)
        U_max = np.max(np.linalg.norm(data['U'], axis=0))
        error_dict['U_maxL2'] = np.max(U_err)
        error_dict['U_ML2'] = np.mean(U_err)
        error_dict['U_RML2'] = error_dict['U_ML2'] / U_max
        if return_predictions:
            error_dict['U_pred'] = U_pred
    except NotImplementedError:
        pass

    try:
        dVdX_pred = batch_predict(NN.eval_dVdX)
        dVdX_err = np.linalg.norm(dVdX_pred - data['dVdX'], axis=0)
        dVdX_max = np.max(np.linalg.norm(data['dVdX'], axis=0))
        error_dict['dVdX_ML2'] = np.mean(dVdX_err)
        error_dict['dVdX_RML2'] = error_dict['dVdX_ML2'] / dVdX_max
        if return_predictions:
            error_dict['dVdX_pred'] = dVdX_pred
    except (NotImplementedError, KeyError):
        pass

    try:
        V_pred = batch_predict(NN.eval_V)
        V_err = np.abs(V_pred - data['V'])
        V_max = np.max(np.abs(data['V']))
        error_dict['V_MAE'] = np.mean(V_err)
        error_dict['V_RMAE'] = error_dict['V_MAE'] / V_max
        if return_predictions:
            error_dict['V_pred'] = V_pred
    except NotImplementedError:
        pass

    return error_dict

def print_errors(train_errs, test_errs):
    '''
    Prints out a table of error metrics.

    Parameters
    ----------
    train_errs : dict
        Dictionary of error metrics produced by eval_errors for training data
    test_errs : dict
        Dictionary of error metrics produced by eval_errors for test data
    '''
    col_width = np.max([len(key) for key in test_errs] + [len('error metric')])
    col_width = col_width.astype(str)
    header = ' {metric:<' + col_width + 's} |  train   |  test    '
    header = header.format(metric='error metric')
    row = ' {metric:<' + col_width + 's} | {train_err:1.2e} | {test_err:1.2e}'

    print('-'*len(header))
    print(header)
    print('-'*len(header))

    for key in np.sort(list(test_errs.keys())):
        print(
            row.format(
                metric=key, train_err=train_errs[key], test_err=test_errs[key]
            )
        )

    print('-'*len(header) + '\n')

# ---------------------------------------------------------------------------- #

def find_fixed_point(OCP, controller, tol, X0=None, verbose=True):
    '''
    Use root-finding to find a fixed point (equilibrium) of the closed-loop
    dynamics near the desired goal state OCP.X_bar. ALso computes the
    closed-loop Jacobian and its eigenvalues.

    Parameters
    ----------
    OCP : instance of QRnet.problem_template.TemplateOCP
    config : instance of QRnet.problem_template.MakeConfig
    tol : float
        Maximum value of the vector field allowed for a trajectory to be
        considered as convergence to an equilibrium
    X0 : array, optional
        Initial guess for the fixed point. If X0=None, use OCP.X_bar
    verbose : bool, default=True
        Set to True to print out the deviation of the fixed point from OCP.X_bar
        and the Jacobian eigenvalue

    Returns
    -------
    X_star : (n_states, 1) array
        Closed-loop equilibrium
    X_star_err : float
        ||X_star - OCP.X_bar||
    F_star : (n_states, 1) array
        Vector field evaluated at X_star. If successful should have F_star ~ 0
    Jac : (n_states, n_states) array
        Close-loop Jacobian at X_star
    eigs : (n_states, 1) complex array
        Eigenvalues of the closed-loop Jacobian
    max_eig : complex scalar
        Largest eigenvalue of the closed-loop Jacobian
    '''
    if X0 is None:
        X0 = OCP.X_bar
    X0 = np.reshape(X0, (OCP.n_states,))

    def dynamics_wrapper(X):
        U = controller.eval_U(X)
        F = OCP.dynamics(X, U)
        C = OCP.constraint_fun(X)
        if C is not None:
            F = np.concatenate((F.flatten(), C.flatten()))
        return F

    def Jacobian_wrapper(X):
        J = OCP.closed_loop_jacobian(X, controller)
        JC = OCP.constraint_jacobian(X)
        if JC is not None:
            J = np.vstack((
                J.reshape(-1,X.shape[0]), JC.reshape(-1,X.shape[0])
            ))
        return J

    sol = root(dynamics_wrapper, X0, jac=Jacobian_wrapper, method='lm')

    sol.x = OCP.apply_state_constraints(sol.x)

    X_star = sol.x.reshape(-1,1)
    U_star = controller.eval_U(X_star)
    F_star = OCP.dynamics(X_star, U_star).reshape(-1,1)
    Jac = OCP.closed_loop_jacobian(sol.x, controller)

    X_star_err = OCP.norm(X_star)[0]

    eigs = np.linalg.eigvals(Jac)
    idx = np.argsort(eigs.real)
    eigs = eigs[idx].reshape(-1,1)
    max_eig = np.squeeze(eigs[-1])

    # Some linearized systems always have one or more zero eigenvalues.
    # Handle this situation by taking the next largest.
    if np.abs(max_eig.real) < tol**2:
        Jac0 = np.squeeze(OCP.closed_loop_jacobian(OCP.X_bar, OCP.LQR))
        eigs0 = np.linalg.eigvals(Jac0)
        idx = np.argsort(eigs0.real)
        eigs0 = eigs0[idx].reshape(-1,1)
        max_eig0 = np.squeeze(eigs0[-1])

        i = 2
        while all([
                i <= OCP.n_states,
                np.abs(max_eig.real) < tol**2,
                np.abs(max_eig0.real) < tol**2
            ]):
            max_eig = np.squeeze(eigs[OCP.n_states - i])
            max_eig0 = np.squeeze(eigs0[OCP.n_states - i])
            i += 1

    if verbose:
        s = '||actual - desired_equilibrium|| = {norm:1.2e}'
        print(s.format(norm=X_star_err))
        if np.max(np.abs(F_star)) > tol:
            print('Dynamics f(X_star):')
            print(F_star)
        s = 'Largest Jacobian eigenvalue = {real:1.2e} + j{imag:1.2e} \n'
        print(s.format(real=max_eig.real, imag=np.abs(max_eig.imag)))

    return X_star, X_star_err, F_star, Jac, eigs, max_eig
