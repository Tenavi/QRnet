import numpy as np
from scipy import sparse

from qrnet.utilities import cross_product_matrix

from . import constants, dynamics
from .containers import VehicleState, Controls, STATES_IDX, CONTROLS_IDX

# Get indices of vector parts of states
Vb_idx = [STATES_IDX[x].start for x in ['u','v','w']]
omega_idx = [STATES_IDX[x].start for x in ['p','q','r']]

q0_idx = STATES_IDX['attitude'].stop - 1
q_vec_idx = slice(STATES_IDX['attitude'].start, q0_idx, None)

def aero_jac_states(states, controls):
    '''
    Jacobian of the aerodynamics model with respect to states.

    Parameters
    ----------
    states : VehicleState
        Current states.
    controls : Controls
        Control inputs.

    Returns
    -------
    d_forces : (3, n_states, n_points) array
        Jacobian of body forces with respect to states.
    d_moments : (3, n_states, n_points) array
        Jacobian of moments with respect to states.
    '''
    Va, alpha, beta = states.airspeed()

    d_forces = np.zeros((3, 11, Va.shape[1]))
    d_moments = np.zeros((3, 11, Va.shape[1]))

    # If airspeed is zero, no aerodynamics. Get index where airspeed is non-zero
    idx = Va > 0

    pressure = dynamics.dynamic_pressure(Va[idx])

    p_over_Va = states.p[idx] / Va[idx]
    q_over_Va = states.q[idx] / Va[idx]
    r_over_Va = states.r[idx] / Va[idx]

    # For chain rule: alpha = arctan(w/u),
    u2_w2 = states.u[idx]**2 + states.w[idx]**2
    d_alpha = {'u': -states.w[idx] / u2_w2, 'w': states.u[idx] / u2_w2}

    ##### Longitudinal aerodynamics #####

    sin_alpha, cos_alpha = np.sin(alpha[idx]), np.cos(alpha[idx])

    # Derivatives of CL, CD, CM with respect to angle of attack
    coefs_alpha = dynamics.coefs_alpha(
        alpha[idx], sin_alpha, cos_alpha, jac=True
    )
    CLalpha, CDalpha, CMalpha, d_CLalpha, d_CDalpha, d_CMalpha = coefs_alpha

    # CL, CD, CM due to elevator deflection
    CLdeltaE, CDdeltaE, CMdeltaE = np.outer(
        [constants.CLdeltaE, constants.CDdeltaE, constants.CMdeltaE],
        controls.elevator[idx]
    )

    # Lift and drag forces
    CLq, CDq = np.outer(
        [constants.CLq, constants.CDq], (constants.c / 2.) * q_over_Va
    )
    F_lift = pressure * (CLalpha + CLq + CLdeltaE)
    F_drag = pressure * (CDalpha + CDq + CDdeltaE)

    # Derivatives of lift, drag, and pitching moment with respect to velocities
    # Chain rule will divide everything by Va
    rho_S = constants.rho * constants.S
    rho_S_c = rho_S * constants.c
    rho_S_b = rho_S * constants.b
    d_FL_d_q = (rho_S_c / 4.) * constants.CLq
    d_FD_d_q = (rho_S_c / 4.) * constants.CDq
    d_M_d_q = (rho_S_c * constants.c / 4.) * constants.CMq

    d_FL_d_Va = rho_S * (CLalpha + CLdeltaE) + d_FL_d_q * q_over_Va
    d_FD_d_Va = rho_S * (CDalpha + CDdeltaE) + d_FD_d_q * q_over_Va
    d_M_d_Va = rho_S_c * (CMalpha + CMdeltaE) + d_M_d_q * q_over_Va

    # Derivatives of lift, drag, and pitching moment with respect to alpha
    d_FL_d_alpha = pressure * d_CLalpha
    d_FD_d_alpha = pressure * d_CDalpha
    d_M_d_alpha = constants.c * pressure * d_CMalpha

    # Derivative of x-axis and z-axis force with respect to velocities
    d_Fx_d_Va = - cos_alpha * d_FD_d_Va + sin_alpha * d_FL_d_Va
    d_Fz_d_Va = - sin_alpha * d_FD_d_Va - cos_alpha * d_FL_d_Va

    # Derivatives of x-axis and z-axis force with respect to alpha
    d_Fx_d_alpha = (
        sin_alpha * (F_drag + d_FL_d_alpha)
        + cos_alpha * (F_lift - d_FD_d_alpha)
    )
    d_Fz_d_alpha = (
        sin_alpha * (F_lift - d_FD_d_alpha)
        - cos_alpha * (F_drag + d_FL_d_alpha)
    )

    # Derivatives with respect to pitch rate
    # TODO: np.einsum probably good here

    d_FL_d_q *= Va[idx]
    d_FD_d_q *= Va[idx]
    d_forces[0:1,STATES_IDX['q'],idx[0]] = - cos_alpha * d_FD_d_q + sin_alpha * d_FL_d_q
    d_forces[2:3,STATES_IDX['q'],idx[0]] = - sin_alpha * d_FD_d_q - cos_alpha * d_FL_d_q
    d_moments[1:2,STATES_IDX['q'],idx[0]] = d_M_d_q * Va[idx]

    # Chain rule for velocities
    for var in ['u','v','w']:
        d_forces[0:1,STATES_IDX[var],idx[0]] = getattr(states, var)[idx] * d_Fx_d_Va
        d_forces[2:3,STATES_IDX[var],idx[0]] = getattr(states, var)[idx] * d_Fz_d_Va
        d_moments[1:2,STATES_IDX[var],idx[0]] = getattr(states, var)[idx] * d_M_d_Va

    # Chain rule for alpha
    for var in ['u','w']:
        d_forces[0:1,STATES_IDX[var],idx[0]] += d_alpha[var] * d_Fx_d_alpha
        d_forces[2:3,STATES_IDX[var],idx[0]] += d_alpha[var] * d_Fz_d_alpha
        d_moments[1:2,STATES_IDX[var],idx[0]] += d_alpha[var] * d_M_d_alpha

    ##### Lateral aerodynamics #####

    # Chain rule for sideslip
    # Not dividing by Va**2 since will multiply by this later with pressure
    sqrt_u2_w2 = np.sqrt(u2_w2)
    v_over_sqrt_u2_w2 = states.v[idx] / sqrt_u2_w2
    d_beta = {
        'u': - states.u[idx] * v_over_sqrt_u2_w2,
        'v': sqrt_u2_w2,
        'w': - states.w[idx] * v_over_sqrt_u2_w2
    }

    # Derivatives of Fy, Ml, Mn with respect to velocities
    d_FMlat_d_Va = np.reshape(
        [constants.CY0, constants.Cl0, constants.Cn0], (3,1)
    )
    d_FMlat_d_Va = d_FMlat_d_Va + np.outer(
        [constants.CYbeta, constants.Clbeta, constants.Cnbeta], beta[idx]
    )

    # Control surface contributions
    d_FMlat_d_Va += np.outer(
        [constants.CYdeltaA, constants.CldeltaA, constants.CndeltaA],
        controls.aileron[idx]
    )
    d_FMlat_d_Va += np.outer(
        [constants.CYdeltaR, constants.CldeltaR, constants.CndeltaR],
        controls.rudder[idx]
    )

    d_FMlat_d_Va *= rho_S

    # Roll and yaw contributions
    d_FMlat_d_Va += np.outer(
        (rho_S_b/4.) * np.array([constants.CYp, constants.Clp, constants.Cnp]),
        p_over_Va
    )
    d_FMlat_d_Va += np.outer(
        (rho_S_b/4.) * np.array([constants.CYr, constants.Clr, constants.Cnr]),
        r_over_Va
    )

    # TODO: np.einsum probably good here
    for var in ['u','v','w']:
        d_forces[1:2,STATES_IDX[var],idx[0]] = (
            (0.5 * rho_S * constants.CYbeta) * d_beta[var]
            + getattr(states, var)[idx] * d_FMlat_d_Va[0:1]
        )
        d_moments[0:1,STATES_IDX[var],idx[0]] = (
            (0.5 * rho_S * constants.Clbeta) * d_beta[var]
            + getattr(states, var)[idx] * d_FMlat_d_Va[1:2]
        )
        d_moments[2:3,STATES_IDX[var],idx[0]] = (
            (0.5 * rho_S * constants.Cnbeta) * d_beta[var]
            + getattr(states, var)[idx] * d_FMlat_d_Va[2:3]
        )

    # Derivatives with respect to roll and yaw rates
    d_forces[1:2,STATES_IDX['p'],idx[0]] = (rho_S_b/4.*constants.CYp) * Va[idx]
    d_forces[1:2,STATES_IDX['r'],idx[0]] = (rho_S_b/4.*constants.CYr) * Va[idx]
    d_moments[0:1,STATES_IDX['p'],idx[0]] = (rho_S_b/4.*constants.Clp) * Va[idx]
    d_moments[0:1,STATES_IDX['r'],idx[0]] = (rho_S_b/4.*constants.Clr) * Va[idx]
    d_moments[2:3,STATES_IDX['p'],idx[0]] = (rho_S_b/4.*constants.Cnp) * Va[idx]
    d_moments[2:3,STATES_IDX['r'],idx[0]] = (rho_S_b/4.*constants.Cnr) * Va[idx]

    d_moments[[0,2]] *= constants.b

    return d_forces, d_moments

def aero_jac_controls(states):
    '''
    Jacobian of the aerodynamics model with respect to control inputs.

    Parameters
    ----------
    states : VehicleState
        Current states.

    Returns
    -------
    d_forces : (3, n_controls, n_points) array
        Jacobian of body forces with respect to controls.
    d_moments : (3, n_controls, n_points) array
        Jacobian of moments with respect to controls.
    '''
    Va, alpha, beta = states.airspeed()

    sin_alpha, cos_alpha = np.sin(alpha), np.cos(alpha)

    d_forces = np.zeros((3, 4, Va.shape[1]))
    d_moments = np.zeros((3, 4, Va.shape[1]))

    # F_x
    d_forces[0,CONTROLS_IDX['elevator']] = (
        -cos_alpha*constants.CDdeltaE + sin_alpha*constants.CLdeltaE
    )
    # F_y
    d_forces[1,CONTROLS_IDX['aileron']] = constants.CYdeltaA
    d_forces[1,CONTROLS_IDX['rudder']] = constants.CYdeltaR
    # F_z
    d_forces[2,CONTROLS_IDX['elevator']] = (
        -sin_alpha*constants.CDdeltaE - cos_alpha*constants.CLdeltaE
    )
    # M_l
    d_moments[0,CONTROLS_IDX['aileron']] = constants.b*constants.CldeltaA
    d_moments[0,CONTROLS_IDX['rudder']] = constants.b*constants.CldeltaR
    # M_m
    d_moments[1,CONTROLS_IDX['elevator']] = constants.c*constants.CMdeltaE
    # M_n
    d_moments[2,CONTROLS_IDX['aileron']] = constants.b*constants.CndeltaA
    d_moments[2,CONTROLS_IDX['rudder']] = constants.b*constants.CndeltaR

    pressure = dynamics.dynamic_pressure(Va)

    return d_forces * pressure, d_moments * pressure

def prop_jac_states(u, v, w):
    '''
    Jacobian of the simple propellor model with respect to states.

    Parameters
    ----------
    states : VehicleState
        Current state.

    Returns
    -------
    d_forces : (3, n_states, n_points) array
        Jacobian of body forces with respect to states.
    '''
    d_forces = np.zeros((3,11,np.size(u)))

    coef = - constants.rho * constants.Sprop * constants.Cprop
    d_forces[0,Vb_idx] = coef * np.reshape([u, v, w], (3,-1))

    return d_forces

def prop_jac_controls(Va):
    '''
    Jacobian of the simple propellor model with respect to throttle input.

    Parameters
    ----------
    Va : (1, n_points) array
        Airspeed [m/s].

    Returns
    -------
    d_forces : (3, n_points) array
        Jacobian of body forces with respect to throttle.
    d_moments : (3, n_points) array
        Jacobian of moments with respect to throttle.
    '''
    d_forces = np.zeros((3,np.size(Va)))
    d_moments = np.zeros((3,np.size(Va)))

    coef = 0.5 * constants.rho * constants.Sprop * constants.Cprop

    d_forces[0] = coef * constants.kmotor**2
    d_moments[0] = -constants.kTp * constants.kOmega**2

    return d_forces, d_moments

def jac_states(states, controls):
    '''
    Evaluate the Jacobian of the dynamics with respect to control variables.

    Parameters
    ----------
    states : VehicleState
        Current state.
    controls : Controls
        Control inputs.

    Returns
    -------
    dFdX : (n_states, n_states, n_points) array
        Jacobian of state dynamics with respect to states, dF/dX.
    '''
    q0 = states.attitude[-1]
    q1, q2, q3 = states.attitude[:-1]

    Vb = np.vstack([states.u, states.v, states.w])
    omega = np.vstack([states.p, states.q, states.r])

    n_t = Vb.shape[1]

    Jw = np.matmul(constants.J_body, omega)

    wx = cross_product_matrix(omega)
    Jwx = cross_product_matrix(Jw)
    Vbx = cross_product_matrix(Vb)
    qx = cross_product_matrix(states.attitude[:-1])

    # Jacobian of last row of rotation matrix; for gravity and altitude
    d_Rot = 2. * np.asarray([
        [q3, -q0, q1, -q2],
        [q0, q3, q2, q1],
        [-q1, -q2, q3, q0]
    ])

    d_F_aero, d_M_aero = aero_jac_states(states, controls)
    d_F_prop = prop_jac_states(states.u, states.v, states.w)

    d_forces = (d_F_aero + d_F_prop)/constants.mass
    # Gravity is only force that depends on attitude. It also is multiplied by
    # constants.mass, so these cancel out
    d_forces[:,STATES_IDX['attitude']] = constants.g0 * d_Rot

    d_pd = np.zeros((d_forces.shape[1:]))
    d_pd[Vb_idx] = [
        2.*(q1*q3 - q0*q2),
        2.*(q2*q3 + q0*q1),
        q0**2 + q3**2 - q1**2 - q2**2
    ]
    d_pd[STATES_IDX['attitude']] = np.einsum('ijk,ik->jk', d_Rot, Vb)

    # Jacobian of velocities
    d_Vb = d_forces
    d_Vb[:, Vb_idx] -= wx
    d_Vb[:, omega_idx] += Vbx

    # Jacobian of quaternions
    d_quat = np.zeros((4,*d_forces.shape[1:]))

    q0_diag = np.kron(np.eye(3), q0).reshape(3, 3, -1)

    d_quat[:-1, q0_idx] = 0.5 * omega
    d_quat[-1, q_vec_idx] = - 0.5 * omega
    d_quat[:-1, q_vec_idx] = - 0.5 * wx
    d_quat[:-1, omega_idx] = 0.5 * (qx + q0_diag)
    d_quat[-1, omega_idx] = -0.5*states.attitude[:-1]

    # Jacobian of angular rates
    d_omega = d_M_aero
    d_omega[:,omega_idx] += Jwx - np.einsum('ijk,jh->ihk',wx,constants.J_body)
    d_omega = np.einsum('ij,jhk->ihk',constants.J_inv_body,d_omega)

    dFdX = np.zeros((d_omega.shape[1], d_omega.shape[1], d_omega.shape[-1]))
    dFdX[STATES_IDX['pd']] = d_pd
    dFdX[Vb_idx] = d_Vb
    dFdX[STATES_IDX['attitude']] = d_quat
    dFdX[omega_idx] = d_omega

    return dFdX

def jac_controls(states, controls=None):
    '''
    Evaluate the Jacobian of the dynamics with respect to control variables.

    Parameters
    ----------
    states : VehicleState
        Current state.
    controls : Controls
        Control inputs.

    Returns
    -------
    dFdU : (n_states, n_controls, n_points) array
        Jacobian of state dynamics with respect to control, dF/dU.
    '''
    Va, alpha, beta = states.airspeed()

    d_forces, d_moments = aero_jac_controls(states)
    d_F_prop, d_M_prop = prop_jac_controls(Va)

    d_forces[0,CONTROLS_IDX['throttle']] = d_F_prop[0]
    d_moments[0,CONTROLS_IDX['throttle']] = d_M_prop[0]

    d_forces /= constants.mass
    d_moments = np.einsum(
        'ij,jhk->ihk', constants.J_inv_body, d_moments,
    )

    dFdU = np.zeros((states.n_states, d_forces.shape[1], Va.shape[1]))
    dFdU[Vb_idx] = d_forces
    dFdU[omega_idx] = d_moments

    return dFdU
