import numpy as np

from qrnet.problem_template import TemplateOCP, MakeConfig
from qrnet.utilities import saturate_np, saturate_tf

config = MakeConfig(
    ode_solver='RK23',
    atol=1e-08,
    rtol=1e-04,
    t1_sim=20.,
    t1_max=120.,
    n_trajectories_train=50,
    n_trajectories_test=50,
    value_loss_weight=0.1,
    gradient_loss_weight=0.1,
)

class MakeOCP(TemplateOCP):
    def __init__(self):
        n_states = 2
        n_controls = 1

        # Dynamics parameters
        self.mu = 2.
        self.b = 1.5

        # Cost parameters
        self.Wx = 1/2
        self.Wy = 1.
        self.Wu = 4.

        # Control constraints
        U_max = 1.
        if U_max is None:
            U_lb, U_ub = None, None
        else:
            U_lb = np.full((n_controls, 1), -U_max)
            U_ub = np.full((n_controls, 1), U_max)

        # Initial condition bounds
        X0_ub = np.array([[3.],[4.]])
        X0_lb = - X0_ub

        # Linearization point
        x_bar = 0.
        X_bar = np.array(([[x_bar], [0.]]))
        U_bar = x_bar / self.b

        # Dynamics linearized around X_bar (dxdt ~= Ax + Bu)
        A = [[0., 1.], [-1., self.mu*(1. - x_bar**2)]]
        B = [[0.], [self.b]]

        # Cost matrices
        Q = np.diag([self.Wx / 2., self.Wy / 2.])
        R = [[self.Wu / 2.]]

        super().__init__(
            X_bar, U_bar, A, B, Q, R,
            U_lb=U_lb, U_ub=U_ub, X0_lb=X0_lb, X0_ub=X0_ub
        )

        self.B = self._B

    def get_params(self, **params):
        '''
        Function to return a dict of parameters which might be needed by matlab
        scripts.

        Parameters
        ----------
        params : keyword arguments
            Additional parameters to return.

        Returns
        -------
        params_dict : dict
            Dict of name-value pairs including
            'n_states' : int
            'n_controls' : int
            'X_bar' : (n_states, 1) array
            'U_bar' : (n_controls, 1) array
            'U_lb' : (n_controls, 1) array or None
            'U_ub' : (n_controls, 1) array or None
            'P' : (n_states, n_states) array
            'K' : (n_controls, n_states) array
            'mu' : float
            **params
        '''
        return super().get_params(mu=self.mu, **params)

    def running_cost(self, X, U):
        '''
        Evaluate the running cost L(X,U) at one or multiple state-control pairs.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).

        Returns
        -------
        L : (1,) or (n_points,) array
            Running cost(s) L(X,U) evaluated at pair(s) (X,U).
        '''
        if X.ndim == 1:
            X_err = X - self.X_bar.flatten()
        else:
            X_err = X - self.X_bar

        if U.ndim == 1:
            U = U - self.U_bar.flatten()
        else:
            U = U - self.U_bar

        x1 = X_err[:1]
        x2 = X_err[1:]

        L = (self.Wx/2.) * x1**2 + (self.Wy/2.) * x2**2 + (self.Wu/2.) * U**2
        return np.squeeze(L)

    def running_cost_gradient(self, X, U, return_dLdX=True, return_dLdU=True):
        '''
        Evaluate the gradients of the running cost, dL/dX (X,U) and dL/dU (X,U),
        at one or multiple state-control pairs.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        return_dLdX : bool, default=True
            Set to True to compute the gradient with respect to states, dL/dX.
        return_dLdU : bool, default=True
            Set to True to compute the gradient with respect to controls, dL/dU.

        Returns
        -------
        dLdX : (n_states,) or (n_states, n_points) array
            Gradient dL/dX (X,U) evaluated at pair(s) (X,U).
        dLdU : (n_states,) or (n_states, n_points) array
            Gradient dL/dU (X,U) evaluated at pair(s) (X,U).
        '''
        if X.ndim == 1:
            X_err = X - self.X_bar.flatten()
        else:
            X_err = X - self.X_bar

        if U.ndim == 1:
            U = U - self.U_bar.flatten()
        else:
            U = U - self.U_bar

        x1 = X_err[:1]
        x2 = X_err[1:]

        if return_dLdX:
            dLdX = np.concatenate((self.Wx * x1, self.Wy * x2))
            if not return_dLdU:
                return dLdX

        if return_dLdU:
            dLdU = self.Wu * U
            if not return_dLdX:
                return dLdU

        return dLdX, dLdU

    def dynamics(self, X, U):
        '''
        Evaluate the closed-loop dynamics at single or multiple time instances.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            Current state.
        U : (n_controls,) or (n_controls, n_points)  array
            Feedback control U=U(X).

        Returns
        -------
        dXdt : (n_states,) or (n_states, n_points) array
            Dynamics dXdt = F(X,U).
        '''
        if X.ndim == 1:
            U = U.flatten()

        x1 = X[:1]
        x2 = X[1:]

        dx1dt = x2
        dx2dt = self.mu * (1. - x1**2) * x2 - x1 + self.b * U

        return np.concatenate((dx1dt, dx2dt))

    def jacobians(self, X, U, F0=None):
        '''
        Evaluate the Jacobians of the dynamics with respect to states and
        controls at single or multiple time instances. Default implementation
        approximates the Jacobians with central differences.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            Current states.
        U : (n_controls,) or (n_controls, n_points)  array
            Control inputs.
        F0 : ignored
            For API consistency only.

        Returns
        -------
        dFdX : (n_states, n_states, n_points) array
            Jacobian with respect to states, dF/dX.
        dFdU : (n_states, n_controls, n_points) array
            Jacobian with respect to controls, dF/dX.
        '''
        x1 = np.atleast_1d(X[0])
        x2 = np.atleast_1d(X[1])

        dFdX = np.array([
            [np.zeros_like(x1), np.ones_like(x1)],
            [-1. - 2.*self.mu*x1*x2, self.mu*(1. - x1**2)]
        ])
        dFdU = np.expand_dims(self.B, -1)
        dFdU = np.tile(dFdU, (1,1,x1.shape[-1]))

        return dFdX, dFdU

    def U_star(self, X, dVdX):
        '''
        Evaluate the optimal control as a function of state and costate.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).

        Returns
        -------
        U : (n_controls,) or (n_controls, n_points) array
            Optimal control(s) arranged by (dimension, time).
        '''
        U = self.U_bar - self.b / self.Wu * dVdX[1:]

        return saturate_np(U, self.U_lb, self.U_ub)

    def make_U_NN(self, X, dVdX):
        '''Makes TensorFlow graph of optimal control with NN value gradient.'''
        U = self.U_bar - self.b / self.Wu * dVdX[1:]

        return saturate_tf(U, self.U_lb, self.U_ub)

    def jac_U_star(self, X, dVdX, U0=None):
        '''
        Evaluate the Jacobian of the optimal control with respect to the state,
        leaving the costate fixed.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).
        U0 : ignored
            For API consistency only.

        Returns
        -------
        U : (n_controls,) or (n_controls, n_points) array
            Optimal control(s) arranged by (dimension, time).
        '''
        dVdX = dVdX.reshape(self.n_states, -1)
        return np.zeros((self.n_controls, self.n_states, dVdX.shape[-1]))

    def bvp_dynamics(self, t, X_aug):
        '''
        Evaluate the augmented dynamics for Pontryagin's Minimum Principle.

        Parameters
        ----------
        X_aug : (2*n_states+1, n_points) array
            Current state, costate, and running cost.

        Returns
        -------
        dX_aug_dt : (2*n_states+1, n_points) array
            Concatenation of dynamics dXdt = F(X,U^*), costate dynamics,
            dAdt = -dH/dX(X,U^*,dVdX), and change in cost dVdt = -L(X,U*),
            where U^* is the optimal control.
        '''
        # Optimal control as a function of the costate
        U = self.U_star(X_aug[:2], X_aug[2:4])

        x1 = X_aug[:1]
        x2 = X_aug[1:2]

        x1_err = x1 - self.X_bar[:1]

        # Costate
        A1 = X_aug[2:3]
        A2 = X_aug[3:4]

        # State dynamics
        dx1dt = x2
        dx2dt = self.mu * (1. - x1**2) * x2 - x1 + self.b * U

        # Costate dynamics
        dA1dt = -self.Wx * x1_err + A2 * (2.*self.mu*x1*x2 + 1.)
        dA2dt = -self.Wy * x2 - A1 - A2 * self.mu * (1. - x1**2)

        L = np.atleast_2d(self.running_cost(X_aug[:2], U))

        return np.vstack((dx1dt, dx2dt, dA1dt, dA2dt, -L))
