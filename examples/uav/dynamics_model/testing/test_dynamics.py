import pytest

import numpy as np

from .. import containers, constants, dynamics
from ..rotations import euler_to_quat

no_forces = np.zeros(3)
no_moments = np.zeros(3)

@pytest.mark.parametrize('attitude_shape', [(),(1,),(2,),(3,)])
@pytest.mark.parametrize('vel_shape', [(),(1,),(2,),(3,)])
def test_state_shapes(attitude_shape, vel_shape):
    yaw, pitch, roll = 2.*np.pi*(np.random.rand(3,*attitude_shape) - .5)
    attitude = euler_to_quat(yaw, pitch/2., roll)
    u, v, w = np.random.randn(3,*vel_shape)

    if len(attitude_shape) and len(vel_shape) and all((
        attitude_shape[0] > 1,
        vel_shape[0] > 1,
        attitude_shape[0] != vel_shape[0]
    )):
        with pytest.raises(ValueError):
            state = containers.VehicleState(attitude=attitude, u=u, v=v, w=w)
    else:
        state = containers.VehicleState(attitude=attitude, u=u, v=v, w=w)
        state = state.as_array()

        u_idx = np.argwhere(np.array(containers.STATES_ORDER) == 'u')[0,0]
        v_idx = np.argwhere(np.array(containers.STATES_ORDER) == 'v')[0,0]
        w_idx = np.argwhere(np.array(containers.STATES_ORDER) == 'w')[0,0]

        q_idx = np.argwhere(np.array(containers.STATES_ORDER) == 'attitude')[0,0]
        q_idx = np.arange(q_idx, q_idx+4)

        zero_idx = [
            i for i in range(state.shape[0])
            if i not in [u_idx, v_idx, w_idx] + list(q_idx)
        ]

        assert np.allclose(state[u_idx], u)
        assert np.allclose(state[v_idx], v)
        assert np.allclose(state[w_idx], w)
        assert np.allclose(state[q_idx].reshape(4,-1), attitude.reshape(4,-1))
        assert np.allclose(state[zero_idx], 0.)

@pytest.mark.parametrize('throttle_shape', range(1,4))
@pytest.mark.parametrize('elevator_shape', range(1,4))
def test_control_shapes(throttle_shape, elevator_shape):
    throttle = np.random.rand(throttle_shape)
    elevator = np.random.randn(elevator_shape)/10.

    if all((
        throttle_shape > 1,
        elevator_shape > 1,
        throttle_shape != elevator_shape
    )):
        with pytest.raises(ValueError):
            ctrl = containers.Controls(throttle=throttle, elevator=elevator)
    else:
        ctrl = containers.Controls(throttle=throttle, elevator=elevator)
        ctrl = ctrl.as_array()

        t_idx = np.argwhere(np.array(containers.CONTROLS_ORDER) == 'throttle')[0,0]
        e_idx = np.argwhere(np.array(containers.CONTROLS_ORDER) == 'elevator')[0,0]

        zero_idx = [i for i in range(ctrl.shape[0]) if i not in [t_idx, e_idx]]

        assert np.allclose(ctrl[t_idx], throttle)
        assert np.allclose(ctrl[e_idx], elevator)
        assert np.allclose(ctrl[zero_idx], 0.)

def test_dynamics_position():
    u = 22.5
    yaw, pitch, roll = np.deg2rad([145., -50., 0.])
    attitude = euler_to_quat(yaw, pitch, roll)
    state = containers.VehicleState(attitude=attitude, u=u)

    dXdt = dynamics.rigid_body_dynamics(state, no_forces, no_moments)
    d_pd_expected = 17.23600

    assert np.isclose(dXdt.pd, d_pd_expected)

def test_dynamics_velocity():
    u, q = 12., 0.6
    yaw, pitch, roll = np.deg2rad([0., 40., 0.])
    quat = euler_to_quat(yaw, pitch, roll)
    state = containers.VehicleState(attitude=quat, u=u, q=q)

    q0 = state.attitude[-1]
    q1, q2, q3 = state.attitude[:-1]

    gravity = constants.mass * constants.g0 * np.array([
        2.*(q1*q3 - q0*q2),
        2.*(q2*q3 + q0*q1),
        q0**2 + q3**2 - q1**2 - q2**2
    ])
    drag, lift = 2.5, 4.8
    test_force = gravity.flatten() - [drag, 0., lift]

    dXdt = dynamics.rigid_body_dynamics(state, test_force, no_moments)

    d_Vb = np.vstack([dXdt.u, dXdt.v, dXdt.w])
    d_Vb_expected = [
        -drag / constants.mass - constants.g0 * np.sin(pitch),
        0.,
        q * u - lift / constants.mass + constants.g0 * np.cos(pitch)
    ]
    assert np.allclose(d_Vb.flatten(), d_Vb_expected)

def test_dynamics_quaternion():
    n_points = 10

    p, q, r = np.random.randn(3,n_points)
    yaw, pitch, roll = 2.*np.pi*(np.random.rand(3,n_points) - .5)
    attitude = euler_to_quat(yaw, pitch/2., roll)
    state = containers.VehicleState(attitude=attitude, p=p, q=q, r=r)

    dXdt = dynamics.rigid_body_dynamics(state, no_forces, no_moments)
    d_quat = dXdt.attitude

    for i in range(n_points):
        Q = [
            [0., r[i], -q[i], p[i]],
            [-r[i], 0., p[i], q[i]],
            [q[i], -p[i], 0., r[i]],
            [-p[i], -q[i], -r[i], 0.]
        ]
        d_quat_expected = 0.5 * np.matmul(Q, attitude[:,i])
        assert np.allclose(d_quat[:,i], d_quat_expected)

def test_dynamics_omega():
    p, q, r = np.random.randn(3)
    state = containers.VehicleState(p=p, q=q, r=r)

    moments = np.random.randn(3)

    dXdt = dynamics.rigid_body_dynamics(state, no_forces, moments)
    d_omega = np.vstack([dXdt.p, dXdt.q, dXdt.r])
    d_omega = np.matmul(constants.J_body, d_omega)

    d_omega_exp = [
        constants.Jxz*p*q + (constants.Jyy-constants.Jzz)*q*r,
        constants.Jxz*(r**2 - p**2) + (constants.Jzz-constants.Jxx)*p*r,
        (constants.Jxx-constants.Jyy)*p*q - constants.Jxz*q*r
    ]
    d_omega_exp += moments

    assert np.allclose(d_omega.flatten(), d_omega_exp)

@pytest.mark.parametrize('shape', [(),(1,),(2,),(3,)])
def test_forces_shape(shape):
    u, v, w, p, q, r = np.random.randn(6,*shape)
    yaw, pitch, roll = 2.*np.pi*(np.random.rand(3,*shape) - .5)
    attitude = euler_to_quat(yaw, pitch/2., roll)
    states = containers.VehicleState(
        attitude=attitude, u=u, v=v, w=w, p=p, q=q, r=r
    )

    U = np.random.rand(4,*shape)
    controls = containers.Controls(U)

    forces, moments = dynamics.forces_and_moments(states, controls)

    if len(shape) and shape[0] == 1:
        shape = ()
    expected_shape = (3,) + shape
    assert forces.shape == expected_shape
    assert moments.shape == expected_shape

@pytest.mark.parametrize('attitude_shape', [(),(1,),(2,),(3,)])
def test_gravity(attitude_shape):
    yaw, pitch, roll = 2.*np.pi*(np.random.rand(3,*attitude_shape) - .5)
    quat = euler_to_quat(yaw, pitch/2., roll)
    states = containers.VehicleState(attitude=quat)

    q0 = states.attitude[-1]
    q1, q2, q3 = states.attitude[:-1]

    expected_gravity = constants.mass * constants.g0 * np.squeeze([
        2.*(q1*q3 - q0*q2),
        2.*(q2*q3 + q0*q1),
        q0**2 + q3**2 - q1**2 - q2**2
    ])

    gravity = dynamics.gravity_forces(states)

    assert gravity.shape == expected_gravity.shape
    assert np.allclose(gravity, expected_gravity)

def test_aero_zero_airspeed():
    idx = [5,7]
    u, v, w, p, q, r = np.random.randn(6,10)
    u[idx], v[idx], w[idx] = 0., 0., 0.

    yaw, pitch, roll = 2.*np.pi*(np.random.rand(3,10) - .5)
    attitude = euler_to_quat(yaw, pitch/2., roll)
    states = containers.VehicleState(
        attitude=attitude, u=u, v=v, w=w, p=p, q=q, r=r
    )

    U = np.random.rand(4,10)
    controls = containers.Controls(U)

    forces, moments = dynamics.aero_forces(states, controls)

    assert np.allclose(forces[:,idx], 0.)
    assert np.allclose(moments[:,idx], 0.)
