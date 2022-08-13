import numpy as np
from scipy import optimize, sparse

from . import containers, constants
from .rotations import euler_to_quat, quat_to_euler

FREE_STATES = ('u', 'w', 'p', 'q', 'r', 'attitude')
FREE_CONTROLS = ('throttle', 'aileron', 'elevator', 'rudder')

def _make_free_indices():
    # Get slices for subsets of states and controls
    states_idx = containers._make_indices(containers.VehicleState, FREE_STATES)
    controls_idx = containers._make_indices(containers.Controls, FREE_CONTROLS)
    # Increase the control slicing index by the largest state index, since
    # states are concatenated first
    n_x = int(np.max([idx.stop for idx in states_idx.values()]))
    for var, idx in controls_idx.items():
        controls_idx[var] = slice(idx.start+n_x, idx.stop+n_x)

    return states_idx, controls_idx

FREE_STATES_IDX, FREE_CONTROLS_IDX = _make_free_indices()
_n_x = int(np.max([idx.stop for idx in FREE_STATES_IDX.values()]))
_n_free = int(np.max([idx.stop for idx in FREE_CONTROLS_IDX.values()]))
_n_u = _n_free - _n_x

def _join_free_vars(states, controls):
    X = [getattr(states, var) for var in FREE_STATES]
    U = [getattr(controls, var) for var in FREE_CONTROLS]
    return np.squeeze(np.vstack(X + U))

def _split_free_vars(XU):
    '''
    Split a numpy vector representation of the free states and controls into the
    respective components and return these as class instances.

    Parameters
    ----------
    XU : (n_vars,) or (n_vars, n_points) array
        Numpy array concatenation of the free states and controls.

    Returns
    -------
    trim_state : VehicleState
        Trim state.
    trim_controls : Controls
        Trim controls.
    '''
    X = [XU[FREE_STATES_IDX[var]] for var in FREE_STATES]
    X = dict(zip(FREE_STATES, X))

    U = [XU[FREE_CONTROLS_IDX[var]] for var in FREE_CONTROLS]
    U = dict(zip(FREE_CONTROLS, U))

    return containers.VehicleState(**X), containers.Controls(**U)

def _obj_and_jac(
        states, controls, dynamics, Va_star, kappa_star=0., gamma_star=0.,
        jacobians=None
    ):
    '''
    Parameters
    ----------
    states : VehicleState
        Trim state.
    controls : Controls
        Trim controls.
    dynamics : callable
        Function taking states X, controls U, and returning the vector field
        dXdt = F(X,U) = dynamics(X,U).
    jacobians : callable, optional
        Function taking states X, controls U, and returning the Jacobian
        matrices dF/dX and dF/dU.
    Va_star : float
        Desired trim airspeed [m/s].
    kappa_star : float, default=np.inf
        Inverse turn radius of desired trim state [1/m]. Positive for clockwise,
        negative for counterclockwise.
    gamma_star : float, default=0.
        Desired trim flight path angle [rad].

    Returns
    -------
    obj : (1,) array
        Discrepancy between the vector field evaluated at the current trim state
        and controls, and the desired vector field.
    jac : (_n_free,) array
        Gradient of the objective function.
    '''
    obj = dynamics(states, controls)

    obj.pd -= - Va_star * np.sin(gamma_star)

    if ~np.isclose(kappa_star, 0.):
        # If turning, translate desired yaw, pitch, and roll derivatives into
        # quaternion derivatives
        yaw, pitch, roll = quat_to_euler(states.attitude)
        theta_phi = np.vstack([pitch, roll]) / 2.
        s_theta, s_phi = np.sin(theta_phi)
        c_theta, c_phi = np.cos(theta_phi)

        psi_dot_star = Va_star * kappa_star * np.vstack([
            -s_theta * c_phi,
            c_theta * s_phi,
            c_theta * c_phi,
            s_theta * s_phi
        ])

        obj.attitude -= psi_dot_star

    obj = obj.as_array()

    if callable(jacobians):
        dFdX, dFdU = jacobians(states, controls)

        dFdX = containers.VehicleState(X=np.squeeze(dFdX).T)
        dFdU = containers.Controls(U=np.squeeze(dFdU).T)

        jac = _join_free_vars(dFdX, dFdU)
        jac = 2. * np.matmul(jac, obj)

        obj = np.sum(obj**2, axis=0)

        return obj, jac

    return np.sum(obj**2, axis=0)

def _make_bounds():
    lb = np.full((_n_free,), -np.inf)
    ub = np.full((_n_free,), np.inf)

    # Quaternions
    lb[FREE_STATES_IDX['attitude']] = -1.
    ub[FREE_STATES_IDX['attitude']] = 1.

    # Control constraints
    for var, idx in FREE_CONTROLS_IDX.items():
        lb[idx] = getattr(constants.min_controls, var)
        ub[idx] = getattr(constants.max_controls, var)

    return optimize.Bounds(lb=lb, ub=ub)

def _make_quat_constraint():
    '''
    Makes a constraint to force the quaternion to have unit norm.
    '''
    def constr_fun(XU):
        attitude = XU[FREE_STATES_IDX['attitude']]
        return np.sum(attitude**2, axis=0)

    def constr_jac(XU):
        Jac = np.zeros_like(XU)
        Jac[FREE_STATES_IDX['attitude']] = 2.*XU[FREE_STATES_IDX['attitude']]
        return Jac

    return optimize.NonlinearConstraint(
        fun=constr_fun, jac=constr_jac, lb=1., ub=1.
    )

def _make_airspeed_constraint(Va_star):
    '''
    Makes a constraint to force Va_star = sqrt(u**2 + w**2).
    '''
    def constr_fun(XU):
        u = XU[FREE_STATES_IDX['u']]
        w = XU[FREE_STATES_IDX['w']]
        return u**2 + w**2

    def constr_jac(XU):
        Jac = np.zeros_like(XU)
        for var in ['u', 'w']:
            Jac[FREE_STATES_IDX[var]] = 2.*XU[FREE_STATES_IDX[var]]
        return Jac

    return optimize.NonlinearConstraint(
        fun=constr_fun, jac=constr_jac, lb=Va_star**2, ub=Va_star**2
    )

def _make_attitude_constraints(Va_star, kappa_star, gamma_star, phi_star):
    '''
    Makes constraints to force yaw to zero and roll to a specified angle
    computed analytically using Beard (5.16). Constrain p, q, and r following
    Beard appendix F.2.
    '''
    def constr_fun(XU):
        attitude = XU[FREE_STATES_IDX['attitude']]
        yaw, pitch, roll = quat_to_euler(attitude.T)
        p = XU[FREE_STATES_IDX['p']]
        q = XU[FREE_STATES_IDX['q']]
        r = XU[FREE_STATES_IDX['r']]

        if np.isclose(kappa_star, 0.):
            pqr_constr = [0., 0., 0.]
        else:
            psi_dot_star = Va_star * kappa_star * np.cos(gamma_star)
            c_pitch = np.cos(pitch)

            pqr_constr = psi_dot_star * np.array([
                - np.sin(pitch),
                np.sin(phi_star) * c_pitch,
                np.cos(phi_star) * c_pitch
            ])

        constr = np.vstack((
            yaw,
            roll - phi_star,
            p - pqr_constr[0],
            q - pqr_constr[1],
            r - pqr_constr[2]
        ))

        if XU.ndim == 1:
            return constr.flatten()
        return constr

    def constr_jac(XU):
        jac = np.zeros((5, _n_free))

        attitude = XU[FREE_STATES_IDX['attitude']]
        yaw, pitch, roll = quat_to_euler(attitude.flatten())

        d_euler_d_quat = optimize._numdiff.approx_derivative(
            lambda quat: quat_to_euler(quat, normalize=False),
            attitude.flatten(),
            f0=[yaw, pitch, roll]
        )
        jac[0,FREE_STATES_IDX['attitude']] = d_euler_d_quat[0]
        jac[1,FREE_STATES_IDX['attitude']] = d_euler_d_quat[2]
        jac[2,FREE_STATES_IDX['p']] = 1.
        jac[3,FREE_STATES_IDX['q']] = 1.
        jac[4,FREE_STATES_IDX['r']] = 1.

        if ~np.isclose(kappa_star, 0.):
            psi_dot_star = Va_star * kappa_star * np.cos(gamma_star)
            s_pitch = np.sin(pitch)
            # Account for chain rule
            d_pitch_d_quat = d_euler_d_quat[1]
            psi_dot_star *= d_pitch_d_quat

            jac[2,FREE_STATES_IDX['attitude']] = psi_dot_star * np.cos(pitch)
            jac[3,FREE_STATES_IDX['attitude']] = psi_dot_star * np.sin(phi_star) * s_pitch
            jac[4,FREE_STATES_IDX['attitude']] = psi_dot_star * np.cos(phi_star) * s_pitch

        return sparse.csr_matrix(jac)

    return optimize.NonlinearConstraint(
        fun=constr_fun, lb=0., ub=0., jac=constr_jac
    )

def compute_trim(
        dynamics, Va_star, R_star=np.inf, gamma_star=0., jacobians=None,
        solver_tol=1e-10, fun_tol=1e-03, verbose=0
    ):
    '''
    Compute the trim state given a desired airspeed, constant turn radius, and
    constant flight path angle. Uses constrained optimization.

    Parameters
    ----------
    dynamics : callable
        Function taking states X, controls U, and returning the vector field
        dXdt = dynamics(X,U).
    Va_star : float
        Desired trim airspeed [m/s].
    R_star : float, default=np.inf
        Turn radius of desired trim state [m]. Positive for clockwise, negative
        for counterclockwise. Set |R_star| > 10000 for straight flight.
    gamma_star : float, default=0.
        Desired trim flight path angle [rad].
    jacobians : callable, optional
        Function taking states X, controls U, and returning the Jacobian
        matrices dF/dX and dF/dU. If not provided, uses finite differences.

    Returns
    -------
    trim_states : VehicleState
        Trim state. pn, pe, pd are not set.
    trim_controls : Controls
        Trim controls.
    success : bool
        True if optimization completed successfully.
    '''

    if verbose:
        print('Computing trim state...')

    max_radius = 10000.

    if np.isclose(R_star, 0.):
        R_star = 1.

    if R_star > max_radius:
        kappa_star = 0.
        phi_star = 0.
    else:
        kappa_star = 1./R_star
        phi_star = np.arctan2(
            Va_star**2 * kappa_star * np.cos(gamma_star), constants.g0
        )

    guess = _join_free_vars(
        containers.VehicleState(
            u=Va_star,
            attitude=euler_to_quat(yaw=0., pitch=0., roll=phi_star),
            q=Va_star*kappa_star*np.sin(phi_star)*np.cos(gamma_star),
            r=Va_star*kappa_star*np.cos(phi_star)*np.cos(gamma_star)
        ),
        containers.Controls(throttle=0.5)
    )

    def cost_fun_wrapper(XU):
        states, controls = _split_free_vars(XU)
        return _obj_and_jac(
            states, controls, dynamics,
            Va_star, kappa_star=kappa_star, gamma_star=gamma_star,
            jacobians=jacobians
        )

    NLP_res = optimize.minimize(
        fun=cost_fun_wrapper,
        jac=callable(jacobians),
        x0=guess,
        bounds=_make_bounds(),
        constraints=[
            _make_quat_constraint(),
            _make_attitude_constraints(Va_star, kappa_star, gamma_star, phi_star),
            _make_airspeed_constraint(Va_star)
        ],
        method='SLSQP',
        tol=solver_tol,
        options={'disp': int(verbose), 'iprint': int(verbose)}
    )

    trim_states, trim_controls = _split_free_vars(NLP_res.x)

    success = all((
        NLP_res.success,
        NLP_res.fun <= fun_tol,
        np.all(trim_controls.as_array() >= constants.min_controls.as_array() + solver_tol),
        np.all(trim_controls.as_array() <= constants.max_controls.as_array() - solver_tol)
    ))

    return trim_states, trim_controls, success
