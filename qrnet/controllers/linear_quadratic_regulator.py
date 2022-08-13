import numpy as np

from scipy.linalg import solve_continuous_are as care
from scipy.optimize._numdiff import approx_derivative

from .base import BaseController
from qrnet.utilities import saturate_np

class LQR(BaseController):
    '''
    Implements a linear quadratic regulator (LQR) control with saturation
    constraints.

    Parameters
    ----------
    X_bar : (n_states, 1) array
        Goal state, nominal linearization point.
    U_bar : (n_controls, 1) array
        Control values at nominal linearization point.
    A : (n_states, n_states) array
        State Jacobian matrix at nominal equilibrium.
    B : (n_states, n_controls) array
        Control Jacobian matrix at nominal equilibrium.
    Q : (n_states, n_states) array
        Hessian of running cost with respect to states. Must be positive
        semi-definite.
    R : (n_controls, n_controls) array
        Hessian of running cost with respect to controls. Must be positive
        definite.
    U_lb : (n_controls, 1) array, optional
        Lower control saturation bounds.
    U_ub : (n_controls, 1) array, optional
        Upper control saturation bounds.
    '''
    def __init__(self, X_bar, U_bar, A, B, Q, R, P=None, U_lb=None, U_ub=None):
        self.X_bar = np.reshape(X_bar, (-1,1))
        self.U_bar = np.reshape(U_bar, (-1,1))

        self.n_states = self.X_bar.shape[0]
        self.n_controls = self.U_bar.shape[0]

        self.U_lb, self.U_ub = U_lb, U_ub

        if self.U_lb is not None:
            self.U_lb = np.reshape(self.U_lb, (-1,1))
        if self.U_ub is not None:
            self.U_ub = np.reshape(self.U_ub, (-1,1))

        # Make Riccati matrix and LQR control gain matrix
        if P is not None:
            self.P = np.asarray(P)
        else:
            self.P = care(A, B, Q, R)
        self.RB = np.linalg.solve(R, np.transpose(B))
        self.K = np.matmul(self.RB, self.P)

    def eval_V(self, X):
        '''
        Predicts the value function, V(X), for each sample state in X.

        Parameters
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to predict the value function for.

        Returns
        -------
        dVdX : (1, n_data) or (1,) array
            Value function prediction for each column in X.
        '''
        X_err = X.reshape(X.shape[0], -1) - self.X_bar
        XPX = X_err * np.matmul(self.P, X_err)
        XPX = np.sum(XPX, axis=0, keepdims=True)

        if X.ndim < 2:
            XPX = XPX.flatten()

        return XPX

    def eval_dVdX(self, X):
        '''
        Predicts the value function gradient, dV/dX(X), for each sample state in
        X.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to predict the value gradient for.

        Returns
        ----------
        dVdX : (n_states, n_data) or (n_states,) array
            Value gradient prediction for each column in X.
        '''
        X_err = X.reshape(X.shape[0], -1) - self.X_bar
        PX = 2. * np.matmul(self.P, X_err)
        return PX.reshape(X.shape)

    def eval_U(self, X):
        '''
        Evaluates the NN feedback control, U(X), for each sample state in X.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to evaluate the control for.

        Returns
        ----------
        U : (n_controls, n_data) or (n_controls,) array
            NN feedback control for each column in X.
        '''
        X_err = X.reshape(X.shape[0], -1) - self.X_bar
        U = self.U_bar - np.matmul(self.K, X_err)
        U = saturate_np(U, self.U_lb, self.U_ub)

        if X.ndim < 2:
            U = U.flatten()

        return U

    def eval_dUdX(self, X):
        '''
        Evaluates the Jacobian of the NN feedback control, [dU/dX](X), for each
        sample state in X.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to evaluate the control for.

        Returns
        ----------
        dUdX : (n_controls, n_states, n_data) or (n_controls, n_states) array
            Jacobian of NN feedback control for each column in X.
        '''
        if X.ndim < 2:
            return - self.K

        dUdX = np.expand_dims(- self.K, -1)
        dUdX = np.tile(dUdX, (1,1,X.shape[1]))
        return dUdX

    def bvp_guess(self, X):
        '''
        Predicts the value function V(X), its gradient dVdX(X), and the optimal
        control U(X) for each sample state in X.

        Parameters
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to make predictions for.

        Returns
        -------
        V : (1, n_data) or (1,) array
            Value function prediction for each column in X.
        dVdX : (n_states, n_data) or (n_states,) array
            Value gradient prediction for each column in X.
        U : (n_controls, n_data) or (n_controls,) array
            NN feedback control for each column in X.
        '''
        X_err = X.reshape(X.shape[0], -1) - self.X_bar

        PX = np.matmul(self.P, X_err)

        XPX = np.sum(X_err * PX, axis=0, keepdims=True)
        U = self.U_bar - np.matmul(self.RB, PX)
        U = saturate_np(U, self.U_lb, self.U_ub)
        PX = 2. * PX.reshape(X.shape)

        if X.ndim < 2:
            XPX = XPX.flatten()
            U = U.flatten()

        return XPX, PX, U

    def train(self, data, **kwargs):
        '''
        Dummy train method for the LQR, which requires no training.

        Parameters
        ----------
        data : ignored
            For API consistency only.
        kwargs : ignored
            For API consistency only.
        '''
        pass
