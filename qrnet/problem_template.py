import numpy as np

try:
    from scipy.integrate import cumtrapz
except:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.optimize._numdiff import approx_derivative
from scipy import sparse

from qrnet.utilities import saturate_np
from qrnet.validate import find_fixed_point

class MakeConfig:
    def __init__(
            self,
            ode_solver='RK45',
            ocp_solver='indirect',
            atol=1e-06,
            rtol=1e-03,
            fp_tol=1e-05,
            indirect_tol=1e-05,
            direct_tol=1e-06,
            direct_tol_scale=0.1,
            indirect_max_nodes=5000,
            direct_n_init_nodes=16,
            direct_n_add_nodes=16,
            direct_max_nodes=64,
            direct_max_slsqp_iter=500,
            t1_sim=60.,
            t1_max=300.,
            t1_scale=3/2,
            n_trajectories_train=100,
            n_trajectories_test=100,
            n_trajectories_MC=100,
            n_hidden=5,
            n_neurons=32,
            activation='tanh',
            value_loss_weight=1.,
            gradient_loss_weight=1.,
            batch_size=None,
            n_epochs=1,
            callback_epoch=1,
            optimizer='L-BFGS-B',
            optimizer_opts={}
        ):
        '''
        Class defining (default) configuration options for setting up how ODES
        and BVPs are integrated, how many trajectories are generated and over
        what time horizons, NN architecture parameters, and training options.

        Parameters
        ----------
        ode_solver : string, default='RK45'
            ODE solver for closed loop integration. See
            scipy.integrate.solve_ivp for options.
        ocp_solver : {'indirect', 'direct'}, default='indirect'
            Whether to use an indirect method (Pontryagin's principle + boundary
            value problem solver) or direct method (Pseudospectral collocation)
            to solve the open loop OCP.
        atol : float, default=1e-06
            Absolute accuracy tolerance for the ODE solver
        rtol : float, default=1e-03
            Relative accuracy tolerance for the ODE solver
        fp_tol : float, default=1e-05
            Maximum value of the vector field allowed for a trajectory to be
            considered as convergence to an equilibrium
        indirect_tol : float, default=1e-05
            Accuracy tolerance for the indirect BVP solver.
        direct_tol : float, default=1e-06
            Accuracy tolerance for the direct OCP solver.
        direct_tol_scale : float, default=0.1
            Number to multiply the accuracy tolerance for the direct OCP solver
            at each solution iteration.
        indirect_max_nodes : int, default=5000
            Maximum number of collocation points used by the indirect BVP solver.
        direct_n_init_nodes : int, default=16
            Initial number of collocation points used by the direct OCP solver.
        direct_n_add_nodes : int, default=16
            How many additional nodes to add when refining the grid used by the
            direct OCP solver.
        direct_max_nodes : int, default=64
            Maximum number of collocation points used by the direct OCP solver.
        direct_max_slsqp_iter : int, default=500
            Maximum number of iterations for the SLSQP optimization routine used
            by the direct OCP solver.
        t1_sim : float, default=60.
            Default time to integrate the ODE over
        t1_max : float, default=300.
            Maximum time horizon to integrate for.
        t1_scale : float, default=3/2
            Amount to multiply the time horizon by if need to integrate the ODE
            or BVP for longer to achieve convergence.
        n_trajectories_train : int, default=100
            Number of trajectories used for the training data set
        n_trajectories_test : int, default=100
            Number of trajectories used for the test data set
        n_trajectories_MC : int, default=100
            Number of trajectories integrated for Monte Carlo tests
        n_hidden : int, default=5
            Number of hidden layers to use
        n_neurons : int, default=32
            Number of neurons per layer
        activation : str, default='tanh'
            Activation function to use. Current only 'tanh' is implemented
        value_loss_weight : float, default=1.
            How much to weight the value function MSE term in the loss function
        gradient_loss_weight : float, default=1.
            How much to weight the value gradient MSE term in the loss function
        batch_size : int, optional
            Maximum number of data points (not trajectories) to use for
            training. If set to None (default), use the entire data set. Useful
            for controlling the data set size to speed up optimization.
        n_epochs : int, default=1
            How many times to iterate through the dataset (for SGD optimizers).
        callback_epoch : int, default=1
            Specifies after how many epochs to print loss functions (for SGD
            optimizers).
        optimizer : str, default='L-BFGS-B'
            Which optimizer to use. Options are 'L-BFGS-B' and any optimizer
            implemented in tensorflow.train.
        optimizer_opts : dict, optional
            Options to pass to the NN optimizer.
        '''
        self.ode_solver = ode_solver
        self.ocp_solver = ocp_solver

        self.atol = atol
        self.rtol = rtol
        self.fp_tol = fp_tol
        self.indirect_tol = indirect_tol
        self.direct_tol = direct_tol
        self.direct_tol_scale = direct_tol_scale

        self.indirect_max_nodes = indirect_max_nodes
        self.direct_n_init_nodes = direct_n_init_nodes
        self.direct_n_add_nodes = direct_n_add_nodes
        self.direct_max_nodes = direct_max_nodes
        self.direct_max_slsqp_iter = direct_max_slsqp_iter

        self.t1_sim = t1_sim
        self.t1_scale = np.maximum(t1_scale, 1. + 1e-02)
        self.t1_max = np.maximum(t1_max, t1_sim * self.t1_scale)

        self.n_trajectories_train = n_trajectories_train
        self.n_trajectories_test = n_trajectories_test
        self.n_trajectories_MC = n_trajectories_MC

        self.n_hidden = n_hidden
        self.n_neurons = n_neurons
        self.activation = activation

        self.value_loss_weight = value_loss_weight
        self.gradient_loss_weight = gradient_loss_weight
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.callback_epoch = callback_epoch
        self.optimizer = optimizer
        self.optimizer_opts = optimizer_opts


class TemplateOCP:
    '''Defines an optimal control problem (OCP).

    Template super class defining an optimal control problem (OCP) including
    dynamics, running cost, and optimal control as a function of costate.

    Parameters
    ----------
    X_bar : (n_states, 1) array
        Goal state, nominal linearization point.
    U_bar : (n_controls, 1) array
        Control values at nominal linearization point.
    A : (n_states, n_states) array or None
        State Jacobian matrix at nominal equilibrium. If None, approximates
        this with central differences.
    B : (n_states, n_controls) array or None
        Control Jacobian matrix at nominal equilibrium. If None, approximates
        this with central differences.
    Q : (n_states, n_states) array
        Hessian of running cost with respect to states. Must be positive
        semi-definite.
    R : (n_controls, n_controls) array
        Hessian of running cost with respect to controls. Must be positive
        definite.
    P : (n_states, n_states) array, optional
        Pre-computed Riccati matrix, if available.
    U_lb : (n_controls, 1) array, optional
        Lower control saturation bounds.
    U_ub : (n_controls, 1) array, optional
        Upper control saturation bounds.
    X0_lb : (n_states, 1) array, optional
        Lower bounds for (uniform) initial condition samples.
    X0_ub : (n_states, 1) array, optional
        Upper bounds for (uniform) initial condition samples.
    '''
    def __init__(
            self, X_bar, U_bar,
            A=None, B=None, Q=None, R=None, P=None,
            U_lb=None, U_ub=None, X0_lb=None, X0_ub=None
        ):
        self.X_bar = np.reshape(X_bar, (-1,1))
        self.U_bar = np.reshape(U_bar, (-1,1))

        self.n_states = self.X_bar.shape[0]
        self.n_controls = self.U_bar.shape[0]

        # Approximate state matrices numerically if not given
        if A is None or B is None:
            _A, _B = self.jacobians(X_bar, U_bar, F0=np.zeros_like(self.X_bar))

        if A is None:
            A = _A
            A[np.abs(A) < 1e-10] = 0.

        if B is None:
            B = _B
            B[np.abs(B) < 1e-10] = 0.

        self._A = np.reshape(A, (self.n_states, self.n_states))
        self._B = np.reshape(B, (self.n_states, self.n_controls))

        self._Q = np.reshape(Q, (self.n_states, self.n_states))
        self._R = np.reshape(R, (self.n_controls, self.n_controls))

        from .controllers import LQR
        self.LQR = LQR(
            X_bar, U_bar, self._A, self._B, self._Q, self._R,
            P=P, U_lb=U_lb, U_ub=U_ub
        )

        self.U_lb, self.U_ub = self.LQR.U_lb, self.LQR.U_ub

        self.X0_lb, self.X0_ub = X0_lb, X0_ub

        if self.X0_lb is not None:
            self.X0_lb = np.reshape(self.X0_lb, (-1,1))
        if self.X0_ub is not None:
            self.X0_ub = np.reshape(self.X0_ub, (-1,1))

    def get_params(self, **params):
        '''
        Function to return a dict of parameters which might be needed by matlab
        scripts.

        Parameters
        ----------
        params : keyword arguments
            Additional parameters to return, usually called by subclass.

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
            **params
        '''
        params_dict = {
            'n_states': self.n_states,
            'n_controls': self.n_controls,
            'X_bar': self.X_bar,
            'U_bar': self.U_bar,
            'U_lb': self.U_lb if self.U_lb is not None else np.nan,
            'U_ub': self.U_ub if self.U_ub is not None else np.nan,
            'A': self._A,
            'B': self._B,
            'Q': self._Q,
            'R': self._R,
            'P': self.LQR.P,
            'K': self.LQR.K,
            **params
        }
        return params_dict

    def norm(self, X, center_X_bar=True):
        '''
        Calculate the distance of a batch of spatial points from X_bar or zero.
        By default uses L2 norm.

        Parameters
        ----------
        X : (n_states, n_data) or (n_states,) array
            Points to compute distances for
        center_X_bar : bool, default=True
            If True calculate ||X - X_bar||, if False calculate ||X||

        Returns
        ----------
        X_norm : (n_data,) array
            Norm for each point in X
        '''
        X = X.reshape(self.n_states, -1)
        if center_X_bar:
            X = X - self.X_bar
        return np.linalg.norm(X, axis=0)

    def sample_X0(self, Ns, dist=None):
        '''Uniform sampling from the initial condition domain.'''
        X0 = np.random.rand(self.n_states, Ns)
        X0 = (self.X0_ub - self.X0_lb) * X0 + self.X0_lb

        if dist is not None:
            X0 = X0 - self.X_bar
            X0_norm = dist / np.linalg.norm(X0, 1, axis=0)
            X0 = X0_norm * X0 + self.X_bar

        if Ns == 1:
            X0 = X0.flatten()
        return X0

    def make_bc(self, X0):
        '''
        Generates a function to evaluate the boundary conditions for a given
        initial condition. Terminal cost is zero so final condition on lambda is
        zero.

        Parameters
        ----------
        X0 : (n_states, 1) array
            Initial condition.

        Returns
        -------
        bc : callable
            Function of X_aug_0 (augmented states at initial time) and X_aug_T
            (augmented states at final time), returning a function which
            evaluates to zero if the boundary conditions are satisfied.
        '''
        X0 = X0.flatten()
        def bc(X_aug_0, X_aug_T):
            return np.concatenate((
                X_aug_0[:self.n_states] - X0, X_aug_T[self.n_states:]
            ))
        return bc

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
        raise NotImplementedError

    def running_cost_gradient(self, X, U, return_dLdX=True, return_dLdU=True):
        '''
        Evaluate the gradients of the running cost, dL/dX (X,U) and dL/dU (X,U),
        at one or multiple state-control pairs. Default implementation
        approximates this with central differences.

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
        L = self.running_cost(X, U)

        if return_dLdX:
            dLdX = approx_derivative(lambda X: self.running_cost(X, U), X, f0=L)
            if not return_dLdU:
                return dLdX

        if return_dLdU:
            dLdU = approx_derivative(lambda U: self.running_cost(X, U), U, f0=L)
            if not return_dLdX:
                return dLdU

        return dLdX, dLdU

    def Hamiltonian(self, X, U, dVdX):
        '''
        Evaluate the Pontryagin Hamiltonian,
        H(X,U,dVdX) = L(X,U) + <dVdX, F(X,U)>
        where L(X,U) is the running cost, dVdX is the costate or value gradient,
        and F(X,U) is the dynamics. A necessary condition for optimality is that
        H(X,U,dVdX) ~ 0 for the whole trajectory.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Value gradient dV/dX (X,U) evaluated at pair(s) (X,U).

        Returns
        -------
        H : (1,) or (n_points,) array
            Pontryagin Hamiltonian each each point in time.
        '''
        L = self.running_cost(X, U)
        F = self.dynamics(X, U)
        return L + np.sum(dVdX * F, axis=0)

    def compute_cost(self, t, X, U):
        '''Computes the accumulated cost J(t) of a state-control trajectory.'''
        L = self.running_cost(X, U)
        J = cumtrapz(L.flatten(), t)
        return np.concatenate((J, J[-1:]))

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
        raise NotImplementedError

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
        F0 : (n_states,) or (n_states, n_points) array, optional
            Dynamics evaluated at current state and control pair.

        Returns
        -------
        dFdX : (n_states, n_states, n_points) array
            Jacobian with respect to states, dF/dX.
        dFdU : (n_states, n_controls, n_points) array
            Jacobian with respect to controls, dF/dX.
        '''
        X = X.reshape(self.n_states, -1)
        U = U.reshape(self.n_controls, -1)

        if F0 is None:
            F0 = self.dynamics(X, U)

        # Jacobian with respect to states
        def F_wrapper(X_flat):
            X = X_flat.reshape(self.n_states, -1)
            return self.dynamics(X, U).flatten()

        # Make sparsity pattern
        sparsity = sparse.hstack([sparse.identity(X.shape[-1])]*self.n_states)
        sparsity = sparse.vstack([sparsity]*self.n_states)

        dFdX = approx_derivative(
            F_wrapper, X.flatten(), f0=F0.flatten(), sparsity=sparsity
        )
        dFdX = np.asarray(dFdX[sparsity.nonzero()])
        dFdX = dFdX.reshape(self.n_states, self.n_states, -1)

        # Jacobian with respect to controls
        def F_wrapper(U_flat):
            U = U_flat.reshape(self.n_controls, -1)
            return self.dynamics(X, U).flatten()

        # Make sparsity pattern
        sparsity = sparse.hstack([sparse.identity(X.shape[-1])]*self.n_controls)
        sparsity = sparse.vstack([sparsity]*self.n_states)

        dFdU = approx_derivative(
            F_wrapper, U.flatten(), f0=F0.flatten(), sparsity=sparsity
        )
        dFdU = np.asarray(dFdU[sparsity.nonzero()])
        dFdU = dFdU.reshape(self.n_states, self.n_controls, -1)

        return dFdX, dFdU

    def closed_loop_jacobian(self, X, controller):
        '''
        Evaluate the Jacobian of the closed-loop dynamics at single or multiple
        time instances.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            Current states.
        controller : object
            Controller instance implementing eval_U and eval_dUdX methods.

        Returns
        -------
        dFdX : (n_states, n_states) or (n_states, n_states, n_points) array
            Closed-loop Jacobian dF/dX + dF/dU * dU/dX.
        '''
        dFdX, dFdU = self.jacobians(X, controller.eval_U(X))
        dUdX = controller.eval_dUdX(X)

        while dFdU.ndim < 3:
            dFdU = dFdU[...,None]
        while dUdX.ndim < 3:
            dUdX = dUdX[...,None]

        dFdX += np.einsum('ijk,jhk->ihk', dFdU, dUdX)

        if X.ndim < 2:
            dFdX = np.squeeze(dFdX)

        return dFdX

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
        raise NotImplementedError

    def jac_U_star(self, X, dVdX, U0=None):
        '''
        Evaluate the Jacobian of the optimal control with respect to the state,
        leaving the costate fixed. Default implementation uses finite
        differences.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).
        U0 : (n_controls,) or (n_controls, n_points) array, optional
            U_star(X, dVdX), pre-evaluated at the inputs.

        Returns
        -------
        dUdX : (n_controls, n_states, n_points) or (n_controls, n_states) array
            Jacobian of the optimal control with respect to states leaving
            costates fixed, dU/dX (X; dVdX).
        '''
        if U0 is None:
            U0 = self.U_star(X, dVdX)

        dVdX = dVdX.reshape(self.n_states, -1)

        # Numerical derivative of optimal feedback policy
        def U_wrapper(X_flat):
            X = X_flat.reshape(self.n_states, -1)
            return self.U_star(X, dVdX).flatten()

        # Make sparsity pattern
        sparsity = sparse.identity(dVdX.shape[-1])
        sparsity = sparse.hstack([sparsity]*self.n_states)
        sparsity = sparse.vstack([sparsity]*self.n_controls)

        dUdX = approx_derivative(
            U_wrapper, X.flatten(), f0=U0.flatten(), sparsity=sparsity
        )
        dUdX = np.asarray(dUdX[sparsity.nonzero()])
        return dUdX.reshape(self.n_controls, self.n_states, -1)

    def make_U_NN(self, X, dVdX):
        '''
        Makes TensorFlow graph of the optimal control as a function of the state
        and NN value gradient.

        Parameters
        ----------
        X : (n_states, None) tensor
            States arranged by (dimension, time).
        dVdX : (n_states, None) tensor
            Costates arranged by (dimension, time).

        Returns
        -------
        U : (n_controls, None) tensor
            Optimal controls arranged by (dimension, time).
        '''
        raise NotImplementedError

    def bvp_dynamics(self, t, X_aug):
        '''
        Evaluate the augmented dynamics for Pontryagin's Minimum Principle.
        Default implementation uses finite differences for the costate dynamics.

        Parameters
        ----------
        t : (n_points,) array
            Time collocation points for each state.
        X_aug : (2*n_states+1, n_points) array
            Current state, costate, and running cost.

        Returns
        -------
        dX_aug_dt : (2*n_states+1, n_points) array
            Concatenation of dynamics dXdt = F(X,U^*), costate dynamics,
            dAdt = -dH/dX(X,U^*,dVdX), and change in cost dVdt = -L(X,U*),
            where U^* is the optimal control.
        '''
        X = X_aug[:self.n_states]
        dVdX = X_aug[self.n_states:2*self.n_states]
        U = self.U_star(X, dVdX)

        # State dynamics
        dXdt = self.dynamics(X, U)

        # Evaluate closed loop Jacobian using chain rule
        dFdX, dFdU = self.jacobians(X, U, F0=dXdt)
        dUdX = self.jac_U_star(X, dVdX, U0=U)

        dFdX += np.einsum('ijk,jhk->ihk', dFdU, dUdX)

        # Lagrangian and Lagrangian gradient
        L = np.atleast_2d(self.running_cost(X, U))
        dLdX, dLdU = self.running_cost_gradient(X, U)

        if dLdX.ndim < 2:
            dLdX = dLdX[:,None]
        if dLdU.ndim < 2:
            dLdU = dLdU[:,None]

        dLdX += np.einsum('ik,ijk->jk', dLdU, dUdX)

        # Costate dynamics (gradient of optimized Hamiltonian)
        dHdX = dLdX + np.einsum('ijk,ik->jk', dFdX, dVdX)

        return np.vstack((dXdt, -dHdX, -L))

    def apply_state_constraints(self, X):
        '''
        Manually update states to satisfy some state constraints. At present
        time, the OCP format only supports constraints which are intrinsic to
        the dynamics (such as quaternions or periodicity), not dynamic
        constraints which need to be satisfied by admissible controls.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            Current states.

        Returns
        ----------
        X : (n_states, n_data) or (n_states,) array
            Current states with constrained values.
        '''
        return X

    def constraint_fun(self, X):
        '''
        A (vector-valued) function which is zero when the state constraints are
        satisfied. At present time, the OCP format only supports constraints
        which are intrinsic to the dynamics (such as quaternions or
        periodicity), not dynamic constraints which need to be satisfied by
        admissible controls.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            Current states.

        Returns
        ----------
        C : (n_constraints,) or (n_constraints, n_data) array or None
            Algebraic equation such that C(X)=0 means that X satisfies the state
            constraints.
        '''
        return

    def constraint_jacobian(self, X):
        '''
        Constraint function Jacobian dC/dX of self.constraint_fun. Default
        implementation approximates this with central differences.

        Parameters
        ----------
        X : (n_states,) array
            Current state.

        Returns
        -------
        dCdX : (n_constraints, n_states) array or None
            dC/dX evaluated at the point X, where C(X)=self.constraint_fun(X).
        '''
        C0 = self.constraint_fun(X)
        if C0 is None:
            return

        return approx_derivative(self.constraint_fun, X, f0=C0)

    def make_integration_events(self):
        '''
        Construct a (list of) callables that are tracked during integration for
        times at which they cross zero. Such events can terminate integration
        early.

        Returns
        -------
        events : None, callable, or list of callables
            Each callable has a function signature e = event(t, X). If the ODE
            integrator finds a sign change in e then it searches for the time t
            at which this occurs. If event.terminal = True then integration
            stops.
        '''
        return
