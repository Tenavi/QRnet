import numpy as np

from . import constants
from .containers import VehicleState, Controls, STATES_IDX, CONTROLS_IDX
from .dynamics import saturate
from .jacobians import aero_jac_controls, prop_jac_controls
from .jacobians import Vb_idx, omega_idx

Vb_omega_idx = Vb_idx + omega_idx
Vb_slice = slice(Vb_idx[0], Vb_idx[-1]+1)
omega_slice = slice(omega_idx[0], omega_idx[-1]+1)

ail_rud_idx = [CONTROLS_IDX['aileron'].start, CONTROLS_IDX['rudder'].start]
elv_idx = CONTROLS_IDX['elevator'].start
ail_rud_elv_idx = ail_rud_idx + [elv_idx]

def control(states, costates, R_inverse, trim_controls):
    '''
    Evaluate the optimal control as a function of state and costate.

    Parameters
    ----------
    states : VehicleState, or (n_states,) or (n_states,n_points) array
        Current state.
    costates : VehicleState, or (n_states,) or (n_states,n_points) array
        Costates.
    R_inverse : Controls or (n_controls,1) or (n_controls,n_controls) array
        Inverse of the control cost matrix R. If a vector then it is assumed the
        R matrix is diagonal.
    trim_controls : Controls or (n_controls,1) array
        Controls which give the aircraft trim.

    Returns
    -------
    controls : Controls
        Optimal control inputs.
    '''
    if isinstance(states, np.ndarray):
        states = VehicleState(states)
    if isinstance(costates, VehicleState):
        costates = costates.as_array()
    if isinstance(trim_controls, Controls):
        trim_controls = trim_controls.as_array()
    if isinstance(R_inverse, Controls):
        R_inverse = R_inverse.as_array()

    while costates.ndim < 2:
        costates = costates[...,None]
    while trim_controls.ndim < 2:
        trim_controls = trim_controls[...,None]
    while R_inverse.ndim < 2:
        R_inverse = R_inverse[...,None]

    Va, alpha, beta = states.airspeed()

    d_forces, d_moments = aero_jac_controls(states)
    d_F_prop, d_M_prop = prop_jac_controls(Va)

    d_forces[0,CONTROLS_IDX['throttle']] = d_F_prop[0]
    d_moments[0,CONTROLS_IDX['throttle']] = d_M_prop[0]

    d_forces /= constants.mass
    d_moments = np.einsum('ij,jhk->ihk', constants.J_inv_body, d_moments)

    controls = np.einsum(
        'ijk,ik->jk',
        np.vstack((d_forces, d_moments)),
        costates[Vb_omega_idx]
    )

    if R_inverse.shape[1] == 1:
        # Diagonal R
        controls = trim_controls - (R_inverse/2.) * controls
    else:
        # Square R
        controls = trim_controls - np.matmul(R_inverse/2., controls)

    return saturate(Controls(controls))

def jacobian(states, costates, R_inverse, controls):
    '''
    Evaluate the Jacobian of the optimal control with respect to states, leaving
    the costates fixed

    Parameters
    ----------
    states : VehicleState, or (n_states,) or (n_states,n_points) array
        Current state.
    costates : VehicleState, or (n_states,) or (n_states,n_points) array
        Costates.
    R_inverse : Controls or (n_controls,1) or (n_controls,n_controls) array
        Inverse of the control cost matrix R. If a vector then it is assumed the
        R matrix is diagonal.
    controls : Controls or (n_controls,1) array
        Optimal controls evaluated at state-costate pair.

    Returns
    -------
    controls_jac : (n_controls, n_states, n_points) array
        Jacobian matrices dU/dX (X, dVdX). Equivalent to but more accurate and
        faster than numerical gradients of optimal_control(states, costates...).
    '''
    if isinstance(states, np.ndarray):
        states = VehicleState(states)
    if isinstance(costates, VehicleState):
        costates = costates.as_array()
    if isinstance(controls, Controls):
        controls = controls.as_array()
    if isinstance(R_inverse, Controls):
        R_inverse = R_inverse.as_array()

    while costates.ndim < 2:
        costates = costates[...,None]
    while controls.ndim < 2:
        controls = controls[...,None]
    while R_inverse.ndim < 2:
        R_inverse = R_inverse[...,None]

    Va, alpha, beta = states.airspeed()
    Va2 = Va**2

    # Partial derivatives of Va^2, divided by Va^2 since will use this to cancel
    # out Va^2 elsewhere
    d_Va2 = (2. / Va2) * np.vstack((states.u, states.v, states.w))

    # Partial derivatives of alpha
    d_alpha = np.vstack([-states.w, states.u]) / (Va2 - states.v**2)

    d_forces, d_moments = aero_jac_controls(states)

    d_forces /= constants.mass
    d_moments = np.einsum('ij,jhk->ihk', constants.J_inv_body, d_moments)

    n = states.n_states
    m = controls.shape[0]
    n_t = np.size(Va)

    dFdUX = np.zeros((d_forces.shape[0] + d_moments.shape[0], m, n, n_t))

    # Moments affect body rates
    dFdUX[3:, ail_rud_elv_idx, Vb_slice] = np.einsum(
        'ijk,hk->ijhk', d_moments[:,ail_rud_elv_idx], d_Va2
    )

    # Aileron and rudder force derivatives
    dFdUX[1, ail_rud_idx, Vb_slice] = (
        np.einsum('ik,jk->ijk', d_forces[1, ail_rud_idx], d_Va2)[None,...]
    )

    # Elevator force derivatives from chain rule with Va
    dFdUX[[0,2], elv_idx, Vb_slice] = (
        np.einsum('ik,jk->ijk', d_forces[[0,2], elv_idx], d_Va2)
    )

    # Elevator force derivatives from chain rule with alpha
    sin_alpha = np.sin(alpha)[0]
    cos_alpha = np.cos(alpha)[0]
    alpha_mat = np.array([[sin_alpha, cos_alpha], [-cos_alpha, sin_alpha]])

    CDCL = [constants.CDdeltaE, constants.CLdeltaE]
    d_forces = np.einsum('ijk,j->ik', alpha_mat, CDCL)
    d_forces *= Va2 * (0.5 * constants.rho * constants.S / constants.mass)

    d_forces = np.einsum('ik,jk->ijk', d_forces, d_alpha)
    for i, j in zip([0,2], [0,1]):
        dFdUX[i, elv_idx, [Vb_idx[0], Vb_idx[2]]] += d_forces[j]

    controls_jac = np.einsum('ijhk,ik->jhk', dFdUX, costates[Vb_omega_idx])

    if R_inverse.shape[1] == 1:
        # Diagonal R
        controls_jac = - np.einsum('ij,ijk->ijk', R_inverse/2., controls_jac)
    else:
        # Square R
        controls_jac = - np.einsum('ij,jhk->ihk', R_inverse/2., controls_jac)

    sat_idx = [
        controls <= constants.min_controls.as_array()[:,None],
        controls >= constants.max_controls.as_array()[:,None]
    ]
    sat_idx = np.where(np.any(sat_idx, axis=0))
    controls_jac[sat_idx[0],:,sat_idx[1]] = 0.

    return controls_jac

def controls_and_jac(states, costates, R_inverse, trim_controls):
    '''
    Evaluate the optimal control as a function of state and costate. Also
    evaluate its Jacobian with respect to states.

    Parameters
    ----------
    states : VehicleState, or (n_states,) or (n_states,n_points) array
        Current state.
    costates : VehicleState, or (n_states,) or (n_states,n_points) array
        Costates.
    R_inverse : Controls or (n_controls,1) or (n_controls,n_controls) array
        Inverse of the control cost matrix R. If a vector then it is assumed the
        R matrix is diagonal.
    trim_controls : Controls or (n_controls,1) array
        Controls which give the aircraft trim.

    Returns
    -------
    controls : Controls
        Optimal control inputs.
    controls_jac : (n_controls, n_states, n_points) array
        Jacobian matrices dU/dX (X, dVdX).
    '''
    if isinstance(states, np.ndarray):
        states = VehicleState(states)
    if isinstance(costates, VehicleState):
        costates = costates.as_array()
    if isinstance(trim_controls, Controls):
        trim_controls = trim_controls.as_array()
    if isinstance(R_inverse, Controls):
        R_inverse = R_inverse.as_array()

    while costates.ndim < 2:
        costates = costates[...,None]
    while trim_controls.ndim < 2:
        trim_controls = trim_controls[...,None]
    while R_inverse.ndim < 2:
        R_inverse = R_inverse[...,None]

    Va, alpha, beta = states.airspeed()
    Va2 = Va**2

    # Partial derivatives of Va^2, divided by Va^2 since will use this to cancel
    # out Va^2 elsewhere
    d_Va2 = (2. / Va2) * np.vstack((states.u, states.v, states.w))

    # Partial derivatives of alpha
    d_alpha = np.vstack([-states.w, states.u]) / (Va2 - states.v**2)

    d_forces, d_moments = aero_jac_controls(states)
    d_F_prop, d_M_prop = prop_jac_controls(Va)

    d_forces[0,CONTROLS_IDX['throttle']] = d_F_prop[0]
    d_moments[0,CONTROLS_IDX['throttle']] = d_M_prop[0]

    d_forces /= constants.mass
    d_moments = np.einsum('ij,jhk->ihk', constants.J_inv_body, d_moments)

    controls = np.einsum(
        'ijk,ik->jk',
        np.vstack((d_forces, d_moments)),
        costates[Vb_omega_idx]
    )

    n = states.n_states
    m = controls.shape[0]
    n_t = np.size(Va)

    dFdUX = np.zeros((d_forces.shape[0] + d_moments.shape[0], m, n, n_t))

    # Moments affect body rates
    dFdUX[3:, ail_rud_elv_idx, Vb_slice] = np.einsum(
        'ijk,hk->ijhk', d_moments[:,ail_rud_elv_idx], d_Va2
    )

    # Aileron and rudder force derivatives
    dFdUX[1, ail_rud_idx, Vb_slice] = (
        np.einsum('ik,jk->ijk', d_forces[1, ail_rud_idx], d_Va2)[None,...]
    )

    # Elevator force derivatives from chain rule with Va
    dFdUX[[0,2], elv_idx, Vb_slice] = (
        np.einsum('ik,jk->ijk', d_forces[[0,2], elv_idx], d_Va2)
    )

    # Elevator force derivatives from chain rule with alpha
    sin_alpha = np.sin(alpha)[0]
    cos_alpha = np.cos(alpha)[0]
    alpha_mat = np.array([[sin_alpha, cos_alpha], [-cos_alpha, sin_alpha]])

    CDCL = [constants.CDdeltaE, constants.CLdeltaE]
    d_forces = np.einsum('ijk,j->ik', alpha_mat, CDCL)
    d_forces *= Va2 * (0.5 * constants.rho * constants.S / constants.mass)

    d_forces = np.einsum('ik,jk->ijk', d_forces, d_alpha)
    for i, j in zip([0,2], [0,1]):
        dFdUX[i, elv_idx, [Vb_idx[0], Vb_idx[2]]] += d_forces[j]

    controls_jac = np.einsum('ijhk,ik->jhk', dFdUX, costates[Vb_omega_idx])

    R_inverse = R_inverse / 2.
    if R_inverse.shape[1] == 1:
        # Diagonal R
        controls = trim_controls - R_inverse * controls
        controls_jac = - np.einsum('ij,ijk->ijk', R_inverse, controls_jac)
    else:
        # Square R
        controls = trim_controls - np.matmul(R_inverse, controls)
        controls_jac = - np.einsum('ij,jhk->ihk', R_inverse, controls_jac)

    # Saturate controls and set Jacobian elements to zero where saturated
    U_lb = constants.min_controls.as_array()[:,None]
    U_ub = constants.max_controls.as_array()[:,None]

    sat_idx = [controls < U_lb, U_ub < controls]

    for i in range(m):
        controls[i, sat_idx[0][i]] = U_lb[i]
        controls[i, sat_idx[1][i]] = U_ub[i]

    sat_idx = np.where(np.any(sat_idx, axis=0))
    controls_jac[sat_idx[0],:,sat_idx[1]] = 0.

    return Controls(controls), controls_jac
