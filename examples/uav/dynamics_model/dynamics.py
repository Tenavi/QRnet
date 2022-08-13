import numpy as np

from . import constants
from .containers import VehicleState, Controls

def gravity_forces(states):
    '''
    Compute the force of gravity acting in the body frame.

    Parameters
    ----------
    states : VehicleState
        Current states.

    Returns
    -------
    forces : (3,) or (3, n_points) array
        Gravity expressed in body frome along body x, y, and z axes.
    '''
    return states.inertial_to_body([0., 0., constants.mass * constants.g0])

def dynamic_pressure(Va):
    '''
    Utility function for computing dynamic pressure as a function of airspeed.

    Parameters
    ----------
    Va : (n_points,) or (1, n_points) array
        Airspeed for each state [m/s].

    Returns
    -------
    pressure : (n_points,) or (1, n_points) array
        1/2 * rho * S * Va**2
    '''
    return (0.5 * constants.rho * constants.S) * Va**2

def blending_fun(alpha, jac=False):
    '''
    Evaluates the sigmoid-type blending function from Beard (4.10), and
    optionally its derivative.

    Parameters
    ----------
    alpha : (n_points,) or (1, n_points) array
        Angle of attack for each state [rad].
    jac : bool, default=False
        If jac=True, also compute the derivative with respect to alpha.

    Returns
    -------
    sigma : (n_points,) array
        Smooth blending function which is approximately 0 for
        -alpha0 < alpha < alpha0 and 1 outside of this region.
    d_sigma : (n_points,) array
        Derivative d_sigma / d_alpha. Only returned if jac=True.
    '''
    _sigma = np.exp(
        constants.blending_const * (
            constants.alpha0 + np.vstack([-alpha, alpha])
        )
    )
    denominator = (1. + _sigma[0]) * (1. + _sigma[1])
    sigma = (1. + _sigma[0] + _sigma[1]) / denominator

    if not jac:
        return sigma

    M = constants.blending_const * np.exp(
        2. * constants.blending_const * constants.alpha0
    )
    d_sigma = M * (_sigma[1] - _sigma[0]) / denominator**2

    return sigma, d_sigma

def coefs_alpha(alpha, sin_alpha=None, cos_alpha=None, jac=False):
    '''
    Compute the coefficients of lift (CL), drag (CD), and pitching moment (CM),
    based on the angle of attack (alpha). Uses the models in Beard Chapter 4,
    with a modified models for post-stall drag and pitching moment. Optionally
    also compute the derivatives of each coefficent with respect to alpha.

    Parameters
    ----------
    alpha : (n_points,) or (1, n_points) array
        Angle of attack for each state [rad].
    sin_alpha : (n_points,) or (1, n_points) array, optional
        Pre-computed sin(angle of attack), if available.
    cos_alpha : (n_points,) or (1, n_points) array, optional
        Pre-computed cos(angle of attack).
    jac : bool, default=False
        If jac=True, also compute the derivatives with respect to alpha.

    Returns
    -------
    CL : (n_points,) or (1, n_points) array
        Lift coefficient model.
    CD : (n_points,) or (1, n_points) array
        Drag coefficient model.
    CM : (n_points,) or (1, n_points) array
        Pitching moment coefficient model.
    d_CL : (n_points,) or (1, n_points) array
        Derivative of lift coefficient. Only returned if jac=True.
    d_CD : (n_points,) or (1, n_points) array
        Derivative of drag coefficient. Only returned if jac=True.
    d_CM : (n_points,) or (1, n_points) array
        Derivative of pitching moment coefficient. Only returned if jac=True.
    '''

    # Linear components
    CL_lin = constants.CL0 + constants.CLalpha * alpha
    CD_lin = CL_lin / (np.pi * constants.e * constants.AR)
    if jac:
        d_CD_lin = (2. * constants.CLalpha) * CD_lin
    CD_lin = constants.CD0 + CL_lin * CD_lin
    CM_lin = np.tanh(constants.CM0 + constants.CMalpha * alpha)

    if jac:
        sigma, d_sigma = blending_fun(alpha, jac=True)
    else:
        sigma = blending_fun(alpha)
    sigma_inv = 1. - sigma

    if sin_alpha is None:
        sin_alpha = np.sin(alpha)
    if cos_alpha is None:
        cos_alpha = np.cos(alpha)

    sin2_alpha = sin_alpha**2
    sin_cos_alpha = sin_alpha * cos_alpha
    sign_sin_alpha = 2. * sin_alpha * np.sign(alpha)
    sign_2sincos_alpha = sign_sin_alpha * sin_cos_alpha

    # Nonlinear adjustment for post-stall model
    CL = sigma_inv * CL_lin + sigma * sign_2sincos_alpha
    CD = sigma_inv * CD_lin + 2. * sigma * sin2_alpha
    CM = sigma_inv * CM_lin - constants.CMinf * sigma * sin_alpha

    if not jac:
        return CL, CD, CM

    d_CL = (
        constants.CLalpha
        + sigma * (
            sign_sin_alpha * (2.* cos_alpha**2 - sin2_alpha) - constants.CLalpha
        )
        + d_sigma * (sign_2sincos_alpha - CL_lin)
    )

    d_CD = (
        sigma_inv * d_CD_lin
        + d_sigma * (2. * sin2_alpha - CD_lin)
        + sigma * 4. * sin_cos_alpha
    )

    d_CM = (
        sigma_inv * constants.CMalpha * (1. - CM_lin**2)
        - d_sigma * (CM_lin + constants.CMinf * sin_alpha)
        - sigma * constants.CMinf * cos_alpha
    )

    return CL, CD, CM, d_CL, d_CD, d_CM

def aero_forces(states, controls):
    '''
    Compute the aerodynamic forces and moments based on airspeed, angle of
    attack, sideslip, angular rates, and control inputs.

    Parameters
    ----------
    states : VehicleState
        Current states.
    controls : Controls
        Control inputs.

    Returns
    -------
    forces : (3,) or (3, n_points) array
        Forces acting along body x, y, and z axes.
    moments : (3,) or (3, n_points) array
        Moments acting in body yaw, pitch, and roll directions.
    '''

    Va, alpha, beta = states.airspeed()

    forces = np.zeros((3, Va.shape[1]))
    moments = np.zeros((3, Va.shape[1]))

    # If airspeed is zero, no aerodynamics. Get index where airspeed is non-zero
    idx = Va > 0

    # Normalize angular rates
    p_bar = (constants.b / 2.) * (states.p[idx] / Va[idx])
    q_bar = (constants.c / 2.) * (states.q[idx] / Va[idx])
    r_bar = (constants.b / 2.) * (states.r[idx] / Va[idx])

    ##### Longitudinal aerodynamics #####

    sin_alpha, cos_alpha = np.sin(alpha[idx]), np.cos(alpha[idx])

    # CL, CD, CM due to angle of attack
    CLalpha, CDalpha, CMalpha = coefs_alpha(alpha[idx], sin_alpha, cos_alpha)

    # CL, CD, CM due to pitch rate
    CLq, CDq, CMq = np.outer(
        [constants.CLq, constants.CDq, constants.CMq], q_bar
    )

    # CL, CD, CM due to elevator deflection
    CLdeltaE, CDdeltaE, CMdeltaE = np.outer(
        [constants.CLdeltaE, constants.CDdeltaE, constants.CMdeltaE],
        controls.elevator[idx]
    )

    # Pitching moment (m)
    moments[1,idx[0]] = constants.c * (CMalpha + CMq + CMdeltaE)

    # Lift and drag, rotated into body frame (Fx, Fz)
    F_lift = CLalpha + CLq + CLdeltaE
    F_drag = CDalpha + CDq + CDdeltaE
    forces[0,idx[0]] = - cos_alpha * F_drag + sin_alpha * F_lift
    forces[2,idx[0]] = - sin_alpha * F_drag - cos_alpha * F_lift

    ##### Lateral aerodynamics #####

    # Sideslip contribution (linear models)
    FM_lat = np.reshape([constants.CY0, constants.Cl0, constants.Cn0], (3,1))
    FM_lat = FM_lat + np.outer(
        [constants.CYbeta, constants.Clbeta, constants.Cnbeta], beta[idx]
    )

    # Roll and yaw contributions
    FM_lat += np.outer(
        [constants.CYp, constants.Clp, constants.Cnp], p_bar
    )
    FM_lat += np.outer(
        [constants.CYr, constants.Clr, constants.Cnr], r_bar
    )

    # Control surface contributions
    FM_lat += np.outer(
        [constants.CYdeltaA, constants.CldeltaA, constants.CndeltaA],
        controls.aileron[idx]
    )
    FM_lat += np.outer(
        [constants.CYdeltaR, constants.CldeltaR, constants.CndeltaR],
        controls.rudder[idx]
    )

    forces[1,idx[0]] = FM_lat[0]
    moments[0,idx[0]] = constants.b * FM_lat[1]
    moments[2,idx[0]] = constants.b * FM_lat[2]

    pressure = dynamic_pressure(Va[idx])
    forces[:,idx[0]] *= pressure
    moments[:,idx[0]] *= pressure

    return np.squeeze(forces), np.squeeze(moments)

def prop_forces(Va, throttle):
    '''
    Simple propellor model, modified from Beard Chapter 4.3.

    Parameters
    ----------
    Va : (1, n_points) array
        Airspeed for each state [m/s].
    throttle : (1, n_points) array
        Throttle setting corresponding to each state.

    Returns
    -------
    forces : (3,) or (3, n_points) array
        Forces acting along body x, y, and z axes.
    moments : (3,) or (3, n_points) array
        Moments acting in body yaw, pitch, and roll directions.
    '''
    forces = np.zeros((3, Va.shape[1]))
    moments = np.zeros((3, Va.shape[1]))

    coef = 0.5 * constants.rho * constants.Sprop * constants.Cprop
    forces[0] = coef * (constants.kmotor**2 * throttle - Va**2)
    moments[0] = -constants.kTp * constants.kOmega**2 * throttle

    return np.squeeze(forces), np.squeeze(moments)

def forces_and_moments(states, controls):
    '''
    Compute the total forces and moments acting on the vehicle for given
    states and controls.

    Parameters
    ----------
    states : VehicleState
        Current states.
    controls : Controls
        Control inputs.

    Returns
    -------
    forces : (3,) or (3, n_points) array
        Forces acting along body x, y, and z axes.
    moments : (3,) or (3, n_points) array
        Moments acting in body yaw, pitch, and roll directions.
    '''
    controls = saturate(controls)
    Va, alpha, beta = states.airspeed()

    F_gravity = gravity_forces(states).reshape(3,-1)
    F_aero, M_aero = aero_forces(states, controls)
    F_prop, M_prop = prop_forces(Va, controls.throttle)

    forces = F_gravity + F_aero.reshape(3,-1) + F_prop.reshape(3,-1)
    moments = M_aero.reshape(3,-1) + M_prop.reshape(3,-1)

    return np.squeeze(forces), np.squeeze(moments)

def rigid_body_dynamics(states, forces, moments):
    '''
    Evaluate the state derivatives given states, forces, and moments.

    Parameters
    ----------
    states : VehicleState
        Current states.
    forces : (3,) or (3, n_points) array
        Forces acting in body frome along body x, y, and z axes.
    moments : (3,) or (3, n_points) array
        Moments acting in body yaw, pitch, and roll directions.

    Returns
    -------
    derivatives : VehicleState
        State dynamics dX/dt.
    '''
    # Make some vectors for convenience
    Vb = np.vstack([states.u, states.v, states.w])
    omega = np.vstack([states.p, states.q, states.r])
    quat = states.attitude

    # Inertial position (Beard (B.1))
    d_pos = states.body_to_inertial(Vb)

    # Inertial velocity (Beard (3.7))
    d_Vb = - np.cross(omega, Vb, axis=0) + forces.reshape(3,-1)/constants.mass

    # Quaternions (Beard B.3)
    d_quat = 0.5 * (-np.cross(omega, quat[:-1], axis=0) + quat[-1:] * omega)
    d_q0 = - 0.5 * np.sum(omega * quat[:-1], axis=0, keepdims=True)
    d_quat = np.vstack((d_quat, d_q0))

    # Angular rates (Beard (3.11))
    J_omega = np.matmul(constants.J_body, omega)
    d_omega = - np.cross(omega, J_omega, axis=0) + moments.reshape(3,-1)
    d_omega = np.matmul(constants.J_inv_body, d_omega)

    return VehicleState(
        pd=d_pos[2], u=d_Vb[0], v=d_Vb[1], w=d_Vb[2],
        p=d_omega[0], q=d_omega[1], r=d_omega[2], attitude=d_quat
    )

def dynamics(states, controls):
    '''
    Evaluate the state derivatives given states and controls.

    Parameters
    ----------
    states : VehicleState
        Current states.
    controls : Controls
        Control inputs.

    Returns
    -------
    derivatives : VehicleState
        State dynamics dX/dt.
    '''
    forces, moments = forces_and_moments(states, controls)
    return rigid_body_dynamics(states, forces, moments)

def saturate(controls):
    for var in ['throttle', 'aileron', 'elevator', 'rudder']:
        setattr(controls, var, np.maximum(
            getattr(controls,var), getattr(constants.min_controls,var).flatten()
        ))
        setattr(controls, var, np.minimum(
            getattr(controls,var), getattr(constants.max_controls,var).flatten()
        ))
    return controls
