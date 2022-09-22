import os
import numpy as np
import scipy.io

from qrnet.problem_template import TemplateOCP, MakeConfig

from .dynamics_model.containers import VehicleState, Controls, STATES_IDX
from .dynamics_model.rotations import euler_to_quat
from .dynamics_model.dynamics import dynamics as uav_dynamics
from .dynamics_model.compute_trim import compute_trim
from .dynamics_model import constants, jacobians, optimal_controls

config = {
    'ode_solver': 'LSODA',
    'ocp_solver': 'direct',
    'fp_tol': 1e-03,
    'indirect_tol': 1e-04,
    'direct_tol': 1e-07,
    'direct_n_init_nodes': 48,
    'direct_max_nodes': 64,
    'direct_n_add_nodes': 16,
    'direct_max_slsqp_iter': 300,
    't1_sim': 30.,
    't1_scale': 3/2,
    't1_max': 180.,
    'n_trajectories_train': 16,
    'n_trajectories_test': 100,
    'batch_size': 128,
    'n_epochs': 1000,
    'callback_epoch': 100,
    'optimizer': 'AdamOptimizer',
    'batch_size': 4096
}

config = MakeConfig(**config)

class MakeOCP(TemplateOCP):
    def __init__(self):
        refresh_params = False

        try:
            if refresh_params:
                raise

            PARAMS = scipy.io.loadmat(
                os.path.join('examples', 'uav', 'params.mat')
            )

            X_bar = VehicleState(PARAMS['X_bar'])
            U_bar = Controls(PARAMS['U_bar'])

            Va_star, _, _ = X_bar.airspeed()
            Va_star = float(Va_star)
        except:
            PARAMS = {}

            # Linearization point
            Va_star = 20.
            X_bar, U_bar, success = compute_trim(
                self.dynamics, jacobians=self.jacobians,
                Va_star=Va_star, R_star=np.inf, gamma_star=0.
            )
            assert success

        # Cost limiter on altitude command changes
        self.h_cost_ceil = 50.
        # Initial condition altitude limit
        self.h_init_ceil = 3. * self.h_cost_ceil
        # Absolute altitude limit to stop integration
        self.h_abs_ceil = 2. * self.h_init_ceil

        # Cost parameters
        self.Q_h = 1. / self.h_cost_ceil**2
        self.Q_u = 10. / Va_star**2
        self.Q_v = 1. / 1.
        self.Q_w = 1. / 1.
        self.Q_attitude = 5. / 1.
        self.Q_p = 1. / np.deg2rad(30.)**2
        self.Q_q = 1. / np.deg2rad(30.)**2
        self.Q_r = 1. / np.deg2rad(30.)**2

        self.R_throttle = 0.1 / constants.max_controls.throttle**2
        self.R_aileron = 0.1 / constants.max_controls.aileron**2
        self.R_elevator = 1. / constants.max_controls.elevator**2
        self.R_rudder = 1. / constants.max_controls.rudder**2

        U_lb = constants.min_controls.as_array()
        U_ub = constants.max_controls.as_array()

        # Cost matrices
        Q = VehicleState(
            pd=self.Q_h, u=self.Q_u, v=self.Q_v, w=self.Q_w,
            p=self.Q_p, q=self.Q_q, r=self.Q_r,
            attitude=[self.Q_attitude]*3 + [0.]
        )
        Q = np.diag(Q.as_array())

        R = Controls(
            throttle=self.R_throttle, aileron=self.R_aileron,
            elevator=self.R_elevator, rudder=self.R_rudder
        )
        self.R_inv = 1./R.as_array().reshape(-1,1)
        self.R = R.as_array().reshape(-1,1)
        R = np.diag(self.R.flatten())

        P = None
        if 'Q' in PARAMS and 'R' in PARAMS:
            if np.allclose(Q, PARAMS['Q']) and np.allclose(R, PARAMS['R']):
                P = PARAMS.get('P', None)

        if P is None:
            print('Creating new LQR controller...')

        super().__init__(
            X_bar.as_array(), U_bar.as_array(),
            Q=Q, R=R, P=P, U_lb=U_lb, U_ub=U_ub
        )

    def _array_to_container(self, X, U, from_equilibrium=False):
        '''
        Convert numpy arrays X and U into VehicleState and Controls container
        class instances. Also allows shifting X and U relative to their
        respective equilibrium values, X_bar and U_bar. If X and U are already
        containers, these get passed through, shifting by equilibrium if
        desired.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array, or VehicleState
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array, or Controls
            Control(s) arranged by (dimension, time).
        from_equilibrium : bool, default=False
            If True, returns VehicleState(X - X_bar) and Controls(U - U_bar).

        Returns
        -------
        states : VehicleState
            State(s), offset from X_bar if from_equilibrium=True.
        controls : Controls
            Control input(s), offset from U_bar if from_equilibrium=True.
        '''
        if from_equilibrium:
            if isinstance(X, VehicleState):
                X = X.as_array()
            if isinstance(U, Controls):
                U = U.as_array()

            X = X.reshape(self.n_states, -1) - self.X_bar
            U = U.reshape(self.n_controls, -1) - self.U_bar

        if isinstance(X, VehicleState):
            states = X
        else:
            states = VehicleState(X)

        if isinstance(U, Controls):
            controls = U
        else:
            controls = Controls(U)

        return states, controls

    def sample_X0(self, Ns, dist=None):
        X0 = VehicleState(self.X_bar)

        if dist is None:
            dist = 1.
        elif dist == 0.:
            X0 = np.tile(self.X_bar, (1,Ns))
            return np.squeeze(X0)
        elif dist < 0. or dist > 1.:
            raise ValueError('dist argument must be None or in (0,1]')

        # +/- 150 [m] in initial altitude
        h0 = self.h_init_ceil*(2.*np.random.rand(Ns) - 1.)
        # +/- 5 [m/s] in each body velocity
        u_err, v_err, w_err = 5.*(2.*np.random.rand(3,Ns) - 1.)
        # +/- 30 [deg/s] in each body rate
        p_err, q_err, r_err = np.deg2rad(30.*(2.*np.random.rand(3,Ns) - 1.))
        # +/- 180 [deg] in initial heading
        course = (dist*np.pi)*(2.*np.random.rand(Ns) - 1.)
        # +/- 90 [deg] in initial pitch angle
        pitch = (dist*np.pi/2.)*(2.*np.random.rand(Ns) - 1.)
        # +/- 180 [deg] in initial roll angle
        roll = (dist*np.pi)*(2.*np.random.rand(Ns) - 1.)

        X0.set_state(
            pd=dist * -h0,
            u=X0.u + dist*u_err,
            v=X0.v + dist*v_err,
            w=X0.w + dist*w_err,
            p=X0.p + dist*p_err,
            q=X0.q + dist*q_err,
            r=X0.r + dist*r_err,
            attitude=euler_to_quat(course, pitch, roll)
        )

        return X0.as_array()

    def running_cost(self, X, U):
        '''
        Evaluate the running cost L(X,U) at one or multiple state-control pairs.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array, or VehicleState
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array, or Controls
            Control(s) arranged by (dimension, time).

        Returns
        -------
        L : (1,) or (n_points,) array
            Running cost(s) L(X,U) evaluated at pair(s) (X,U).
        '''
        err_states, err_controls = self._array_to_container(
            X, U, from_equilibrium=True
        )

        h_err = self.h_cost_ceil * np.tanh(err_states.pd / self.h_cost_ceil)
        quat_err = np.sum(err_states.attitude[:-1]**2, axis=0, keepdims=True)

        L_states = np.squeeze(
            self.Q_h * h_err**2
            + self.Q_u * err_states.u**2
            + self.Q_v * err_states.v**2
            + self.Q_w * err_states.w**2
            + self.Q_attitude * quat_err
            + self.Q_p * err_states.p**2
            + self.Q_q * err_states.q**2
            + self.Q_r * err_states.r**2
        )

        L_controls = np.squeeze(
            self.R_throttle * err_controls.throttle**2
            + self.R_aileron * err_controls.aileron**2
            + self.R_elevator * err_controls.elevator**2
            + self.R_rudder * err_controls.rudder**2
        )

        return L_states + L_controls

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
        err_states, err_controls = self._array_to_container(
            X, U, from_equilibrium=True
        )

        if return_dLdX:
            dLdpd = (
                (2.*self.Q_h*self.h_cost_ceil)
                * np.sinh(err_states.pd / self.h_cost_ceil)
                / np.cosh(err_states.pd / self.h_cost_ceil)**3
            )
            dLdquat = (2.*self.Q_attitude) * err_states.attitude
            dLdquat[-1] = 0.

            dLdX = VehicleState(
                pd=dLdpd,
                u=(2.*self.Q_u)*err_states.u,
                v=(2.*self.Q_v)*err_states.v,
                w=(2.*self.Q_w)*err_states.w,
                p=(2.*self.Q_p)*err_states.p,
                q=(2.*self.Q_q)*err_states.q,
                r=(2.*self.Q_r)*err_states.r,
                attitude=dLdquat
            )
            if not return_dLdU:
                return dLdX.as_array()

        if return_dLdU:
            err_controls = err_controls.as_array()
            if err_controls.ndim < 2:
                err_controls = err_controls[:,None]
            dLdU = (2.*self.R) * err_controls
            if not return_dLdX:
                return dLdU

        return dLdX.as_array(), dLdU

    def dynamics(self, X, U):
        '''
        Evaluate the closed-loop dynamics at single or multiple time instances.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            Current state.
        U : (n_controls,) or (n_controls, n_points) array
            Feedback control U=U(X).

        Returns
        -------
        dXdt : (n_states,) or (n_states, n_points) array
            Dynamics dXdt = F(X,U).
        '''
        states, controls = self._array_to_container(X, U)

        dXdt = uav_dynamics(states, controls)

        if not isinstance(X, VehicleState):
            dXdt = dXdt.as_array().reshape(X.shape)

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
        states, controls = self._array_to_container(X, U)

        dFdX = jacobians.jac_states(states, controls)
        dFdU = jacobians.jac_controls(states, controls)

        return dFdX, dFdU

    def U_star(self, X, dVdX, jac=False):
        '''
        Evaluate the optimal control as a function of state and costate.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).
        jac : bool, default=False
            If True, also return the Jacobian of the controls with respect to
            states.

        Returns
        -------
        U : (n_controls,) or (n_controls, n_points) array
            Optimal control(s) arranged by (dimension, time).
        dUdX : (n_controls, n_states, n_points) or (n_controls, n_states) array
            Jacobian of the optimal control with respect to states leaving
            costates fixed, dU/dX (X; dVdX). Only returned if jac=True.
        '''
        if jac:
            U, dUdX = optimal_controls.controls_and_jac(
                X, dVdX, self.R_inv, self.U_bar
            )
        else:
            U = optimal_controls.control(X, dVdX, self.R_inv, self.U_bar)

        U = U.as_array()

        if U.ndim < 2 and X.ndim >= 2:
            U = U[...,None]

        if jac:
            return U, dUdX

        return U

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
        U0 : (n_controls,) or (n_controls, n_points) array, optional
            U_star(X, dVdX), pre-evaluated at the inputs.

        Returns
        -------
        dUdX : (n_controls, n_states, n_points) or (n_controls, n_states) array
            Jacobian of the optimal control with respect to states leaving
            costates fixed, dU/dX (X; dVdX).
        '''
        if U0 is None:
            U0 = optimal_controls.control(X, dVdX, self.R_inv, self.U_bar)
        return optimal_controls.jacobian(X, dVdX, self.R_inv, U0)

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

        U, dUdX = self.U_star(X, dVdX, jac=True)

        # State dynamics
        dXdt = self.dynamics(X, U)

        # Evaluate closed loop Jacobian using chain rule
        dFdX, dFdU = self.jacobians(X, U, F0=dXdt)

        dFdX += np.einsum('ijk,jhk->ihk', dFdU, dUdX)

        # Lagrangian and Lagrangian gradient
        L = np.atleast_2d(self.running_cost(X, U))
        dLdX, dLdU = self.running_cost_gradient(X, U)

        if dLdX.ndim < 2:
            dLdX = dLdX[:,None]
        if dLdU.ndim < 2:
            dLdU = dLdU[:,None]

        dLdX = dLdX + np.einsum('ik,ijk->jk', dLdU, dUdX)

        # Costate dynamics (gradient of optimized Hamiltonian)
        dHdX = dLdX + np.einsum('ijk,ik->jk', dFdX, dVdX)

        return np.vstack((dXdt, -dHdX, -L))

    def apply_state_constraints(self, X):
        '''
        Manually update states to enforce the quaternion norm constraint.

        Parameters
        ----------
        X : (n_states, n_data) or (n_states,) array
            Current states.

        Returns
        -------
        X : (n_states, n_data) or (n_states,) array
            Current states with constrained values.
        '''
        states = VehicleState(X)
        states.attitude[-1] = np.sqrt(
            1. - np.sum(states.attitude[:-1]**2, axis=0, keepdims=True)
        )
        return states.as_array().reshape(X.shape)

    def constraint_fun(self, X):
        '''
        A function which is zero when the quaternion norm state constraint is
        satisfied.

        Parameters
        ----------
        X : (n_states, n_data) or (n_states,) array
            Current states.

        Returns
        -------
        C : (n_constraints,) or (n_constraints, n_data) array or None
            Algebraic equation such that C(X)=0 means that X satisfies the state
            constraints.
        '''
        quaternion = X[STATES_IDX['attitude']]
        return 1. - np.sum(quaternion**2, axis=0)

    def constraint_jacobian(self, X):
        '''
        Constraint function Jacobian dC/dX of self.constraint_fun.

        Parameters
        ----------
        X : (n_states,) array
            Current state.

        Returns
        -------
        jac : (n_constraints, n_states) array or None
            dC/dX evaluated at the point X, where C(X)=self.constraint_fun(X).
        '''
        jac = np.zeros((1,self.n_states))
        idx = STATES_IDX['attitude']
        jac[0, idx] = -2. * X.flatten()[idx]
        return jac

    def make_integration_events(self):
        '''
        Construct a (list of) callables that are tracked during integration for
        times at which they cross zero. For the UAV problem, checks if altitude
        is outside some large bounds and stops integration early to save time.

        Returns
        -------
        altitude_event : callable
            Functions that check if the altitude crosses a large positive or
            negative threshold.
        '''
        def altitude_event(t, X):
            #print(t, X[STATES_IDX['pd']].flatten())
            return np.abs(X[STATES_IDX['pd']].flatten()) - self.h_abs_ceil
        altitude_event.terminal = True
        return altitude_event
