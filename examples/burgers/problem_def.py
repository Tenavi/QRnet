import numpy as np

from qrnet.problem_template import TemplateOCP, MakeConfig
from qrnet.utilities import saturate_np, saturate_tf

N_STATES = 64

config = MakeConfig(
    ode_solver='LSODA',
    indirect_tol=1e-04,
    indirect_max_nodes=1500,
    t1_sim=30.,
    t1_scale=6/5,
    t1_max=150.,
    n_trajectories_train=128,
    n_trajectories_test=500,
    value_loss_weight=1/5,
    gradient_loss_weight=0.
)

class MakeOCP(TemplateOCP):
    def __init__(self):
        n_states = N_STATES
        n_controls = 2

        self.nu = 0.02
        self.gamma = 0.1
        self.R = 0.5
        kappa = 25.

        U_max = None
        if U_max is None:
            U_lb, U_ub = None, None
        else:
            U_lb = np.full((n_controls, 1), - U_max)
            U_ub = np.full((n_controls, 1), U_max)

        # Chebyshev nodes, differentiation matrices, and Clenshaw-Curtis weights
        self.xi, self.D, self.w_flat = cheb(n_states+1)
        self.D2 = np.matmul(self.D, self.D)

        # Truncates system to account for zero boundary conditions
        self.xi = self.xi[1:-1].reshape(-1,1)
        self.w_flat = self.w_flat[1:-1]
        self.w = self.w_flat.reshape(-1,1)
        self.D = self.D[1:-1, 1:-1]
        self.D2 = self.D2[1:-1, 1:-1]

        # Control multiplier
        B = np.hstack((
            (-4/5 <= self.xi) & (self.xi <= -2/5),
            (2/5 <= self.xi) & (self.xi <= 4/5)
        ))
        B = -kappa * B * np.hstack((
            (self.xi + 4/5)*(self.xi + 2/5),
            (self.xi - 2/5)*(self.xi - 4/5)
        ))
        B = np.abs(B)

        # Forcing term coefficient
        self.alpha = np.abs(self.xi) <= 1/5
        self.alpha = - kappa * self.alpha * (self.xi + 1/5)*(self.xi - 1/5)
        self.alpha = np.abs(self.alpha)
        self.alpha_flat = self.alpha.flatten()

        self.RBT = - B.T / (2.*self.R)

        # Default number of sin functions for initial conditions
        self.n_X0_terms = 10

        ##### Makes LQR controller #####

        # Linearization point
        X_bar = np.zeros((n_states,1))
        U_bar = np.zeros((n_controls,1))

        # Dynamics linearized around origin (dxdt ~= Ax + Bu)
        A = self.nu*self.D2 + np.diag(self.alpha_flat)

        # Cost matrices
        Q = np.diag(self.w_flat)
        R = np.diag([self.R]*n_controls)

        super().__init__(X_bar, U_bar, A, B, Q, R, U_lb=U_lb, U_ub=U_ub)

        self.B = self._B

    def get_params(self, **params):
        '''
        Function to return a dict of parameters which might be needed by matlab
        scripts.

        Arguments
        ----------
        params : keyword arguments
            Additional parameters to return.

        Returns
        ----------
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
            'xi' : (n_states, 1) array
            'w' : (n_states, 1) array
            **params
        '''
        return super().get_params(xi=self.xi, w=self.w, **params)

    def norm(self, X, center_X_bar=True):
        '''
        Calculate the distance of a batch of spatial points from X_bar or zero.
        Uses the Clenshaw Curtis quadrature weights to compute a weighted norm.

        Arguments
        ----------
        X : (n_states, n_data) array
            Points to compute distances for
        center_X_bar : not used
            For API consistency only

        Returns
        ----------
        X_norm : (n_data,) array
            Norm for each point in X
        '''
        X = X.reshape(self.n_states, -1)
        return np.sqrt(np.sum(X**2 * self.w, axis=0))

    def sample_X0(self, Ns, dist=None, K=None):
        '''Sampling from sum of sine functions.'''

        if K is None:
            K = self.n_X0_terms

        xi = np.pi * self.xi
        X0 = np.zeros((self.n_states, Ns))

        for k in range(1,K+1):
            ak = (2.*np.random.rand(1,Ns) - 1.)/k
            X0 += ak * np.sin(k * xi)

        if dist is not None:
            X0_norm = self.norm(X0).reshape(1, -1)
            X0 *= dist / X0_norm

        if Ns == 1:
            X0 = X0.flatten()
        return X0

    def running_cost(self, X, U, wX=None):
        '''
        Evaluate the running cost L(X,U) at one or multiple state-control pairs.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        wX : (n_states,) or (n_states, n_points) array, optional
            States(s) multiplied by the Chebyshev quadrature weights.

        Returns
        -------
        L : (1,) or (n_points,) array
            Running cost(s) L(X,U) evaluated at pair(s) (X,U).
        '''
        if wX is None:
            if X.ndim == 1:
                wX = self.w_flat * X
            else:
                wX = self.w * X

        return np.sum(wX * X, axis=0) + self.R * np.sum(U**2, axis=0)

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
        if return_dLdX:
            if X.ndim == 1:
                dLdX = 2. * self.w_flat * X
            else:
                dLdX = 2. * self.w * X
            if not return_dLdU:
                return dLdX

        if return_dLdU:
            dLdU = 2. * self.R * U
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
        flat_out = X.ndim < 2
        X = X.reshape(self.n_states, -1)
        U = U.reshape(self.n_controls, -1)

        dXdt = (
            - 0.5*np.matmul(self.D, X**2)
            + np.matmul(self.nu*self.D2, X)
            + X * self.alpha * np.exp(-self.gamma * X)
            + np.matmul(self.B, U)
        )

        if flat_out:
            dXdt = dXdt.flatten()

        return dXdt

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
        dFdX : (n_states, n_states) or (n_states, n_states, n_points) array
            Jacobian with respect to states, dF/dX.
        dFdU : (n_states, n_controls) or (n_states, n_controls, n_points) array
            Jacobian with respect to controls, dF/dX.
        '''
        X = X.reshape(self.n_states, -1)

        gamma_X = -self.gamma * X
        gamma_X = (1. + gamma_X) * self.alpha * np.exp(gamma_X)

        dFdX = (
            - X * np.expand_dims(self.D, -1)
            + np.expand_dims(self.nu * self.D2, -1)
        )

        diag_idx = np.diag_indices(self.n_states)
        for k in range(X.shape[1]):
            dFdX[diag_idx[0],diag_idx[1],k] += gamma_X[:,k]

        dFdU = np.expand_dims(self.B, -1)
        dFdU = np.tile(dFdU, (1,1,X.shape[-1]))

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
        U = np.matmul(self.RBT, dVdX)
        return saturate_np(U, self.U_lb, self.U_ub)

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

    def make_U_NN(self, X, dVdX):
        '''Makes TensorFlow graph of optimal control with NN value gradient.'''
        from tensorflow import matmul

        U = matmul(self.RBT.astype(np.float32), dVdX)

        return saturate_tf(U, self.U_lb, self.U_ub)

    def bvp_dynamics(self, t, X_aug):
        '''
        Evaluate the augmented dynamics for Pontryagin's Minimum Principle.
        Default implementation uses finite differences for the costate dynamics.

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
        X = X_aug[:self.n_states].reshape(self.n_states, -1)
        A = X_aug[self.n_states:2*self.n_states].reshape(self.n_states, -1)

        # Control as a function of the costate
        U = self.U_star(X, A)

        wX = self.w * X
        aeX = self.alpha * np.exp(-self.gamma * X)

        dXdt = (
            - 0.5*np.matmul(self.D, X**2)
            + np.matmul(self.nu*self.D2, X)
            + X * aeX
            + np.matmul(self.B, U)
        )

        dAdt = (
            - 2.*wX
            + X * np.matmul(self.D.T, A)
            - np.matmul(self.nu * self.D2.T, A)
            - aeX * (1. - self.gamma*X) * A
        )

        L = np.atleast_2d(self.running_cost(X, U, wX))

        return np.vstack((dXdt, dAdt, -L))

def cheb(N):
    '''
    Build Chebyshev differentiation matrix.
    Uses algorithm on page 54 of Spectral Methods in MATLAB by Trefethen.
    '''
    theta = np.pi / N * np.arange(0, N+1)
    X_nodes = np.cos(theta)

    X = np.tile(X_nodes, (N+1, 1))
    X = X.T - X

    C = np.concatenate(([2.], np.ones(N-1), [2.]))
    C[1::2] = -C[1::2]
    C = np.outer(C, 1./C)

    D = C / (X + np.identity(N+1))
    D = D - np.diag(D.sum(axis=1))

    # Clenshaw-Curtis weights
    # Uses algorithm on page 128 of Spectral Methods in MATLAB
    w = np.empty_like(X_nodes)
    v = np.ones(N-1)
    for k in range(2, N, 2):
        v -= 2.*np.cos(k * theta[1:-1]) / (k**2 - 1)

    if N % 2 == 0:
        w[0] = 1./(N**2 - 1)
        v -= np.cos(N*theta[1:-1]) / (N**2 - 1)
    else:
        w[0] = 1./N**2

    w[-1] = w[0]
    w[1:-1] = 2.*v/N

    return X_nodes, D, w
