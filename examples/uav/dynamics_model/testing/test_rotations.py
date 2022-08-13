import pytest

import numpy as np

from .. import rotations
from ..containers import VehicleState

@pytest.mark.parametrize(
    'yaw,pitch,roll',
    [
        np.zeros(3),
        np.pi*(np.random.rand(3) - 0.5),
        np.pi*(np.random.rand(3,1) - 0.5),
        np.pi*(np.random.rand(3,10000) - 0.5)
    ]
)
def test_euler_to_quat(yaw, pitch, roll):
    c_yaw = np.cos(yaw/2.)
    c_pitch = np.cos(pitch/2.)
    c_roll = np.cos(roll/2.)
    s_yaw = np.sin(yaw/2.)
    s_pitch = np.sin(pitch/2.)
    s_roll = np.sin(roll/2.)

    q_expected = (
        c_yaw * c_pitch * s_roll - s_yaw * s_pitch * c_roll,
        c_yaw * s_pitch * c_roll + s_yaw * c_pitch * s_roll,
        s_yaw * c_pitch * c_roll - c_yaw * s_pitch * s_roll,
        c_yaw * c_pitch * c_roll + s_yaw * s_pitch * s_roll
    )

    q = rotations.euler_to_quat(yaw, pitch, roll)

    for i in range(4):
        assert np.allclose(q[i], q_expected[i])

@pytest.mark.parametrize(
    'q', [
        'gimbal_lock',
        np.random.randn(4),
        np.random.randn(4,1),
        np.random.randn(4,2)
    ]
)
def test_quat_to_euler(q):
    gimbal_lock = isinstance(q, str)
    if gimbal_lock:
        yaw, roll = np.pi*(2.*np.random.rand(2,2) - 1.)
        pitch = [np.pi/2., -np.pi/2.]
        q = rotations.euler_to_quat(yaw, pitch, roll)

    yaw, pitch, roll = rotations.quat_to_euler(q)

    if gimbal_lock:
        # Don't get unique results when in gimbal lock, so just check that
        # recover the original quaternion

        q_from_euler = rotations.euler_to_quat(yaw, pitch, roll)
        assert np.allclose(q_from_euler, q)
    else:
        q = q / np.linalg.norm(q, axis=0, keepdims=True)

        yaw_expected = np.arctan2(
            2.*(q[3]*q[2] + q[0]*q[1]),
            q[3]**2 + q[0]**2 - q[1]**2 - q[2]**2
        )
        pitch_expected = np.arcsin(
            2.*(q[3]*q[1] - q[0]*q[2])
        )
        roll_expected = np.arctan2(
            2.*(q[3]*q[0] + q[1]*q[2]),
            q[3]**2 + q[2]**2 - q[0]**2 - q[1]**2
        )

        assert np.allclose(yaw, yaw_expected)
        assert np.allclose(pitch, pitch_expected)
        assert np.allclose(roll, roll_expected)

def test_identity_rotation():
    # Aircraft pointing due North
    yaw, pitch, roll = 0., 0., 0.
    attitude = rotations.euler_to_quat(yaw, pitch, roll)
    state = VehicleState(attitude=attitude)

    u, v, w = np.random.randn(3)
    u_in, v_in, w_in = state.body_to_inertial([u, v, w])
    assert np.isclose(u_in, u)
    assert np.isclose(v_in, v)
    assert np.isclose(w_in, w)

    u_in, v_in, w_in = state.inertial_to_body([u, v, w])
    assert np.isclose(u_in, u)
    assert np.isclose(v_in, v)
    assert np.isclose(w_in, w)

@pytest.mark.parametrize('shape', [(3,), (3,1), (3,2)])
def test_inverse_rotation(shape):
    yaw, pitch, roll = 2.*np.pi*(np.random.rand(3) - .5)
    attitude = rotations.euler_to_quat(yaw, pitch/2., roll)
    state = VehicleState(attitude=attitude)

    F = np.random.randn(*shape)
    F_rot = state.body_to_inertial(F)
    assert F_rot.shape == F.shape
    F_rot_rot = state.inertial_to_body(F_rot)
    assert np.allclose(F_rot_rot, F)

def test_yaw():
    # Aircraft pointing due east
    yaw, pitch, roll = np.pi/2., 0., 0.
    attitude = rotations.euler_to_quat(yaw, pitch, roll)

    u, v, w = np.random.randn(3)
    state = VehicleState(attitude=attitude)

    u_in, v_in, w_in = state.body_to_inertial([u, v, w])
    assert np.isclose(u_in, -v)
    assert np.isclose(v_in, u)
    assert np.isclose(w_in, w)

def test_pitch():
    # Aircraft pointing up 45 degrees
    yaw, pitch, roll = 0., np.pi/4., 0.
    attitude = rotations.euler_to_quat(yaw, pitch, roll)

    u, v, w = np.random.randn(3)
    state = VehicleState(attitude=attitude)

    u_in, v_in, w_in = state.body_to_inertial([u, v, w])
    assert np.isclose(u_in, np.sqrt(1/2)*(u + w))
    assert np.isclose(v_in, v)
    assert np.isclose(w_in, np.sqrt(1/2)*(-u + w))

def test_roll():
    # Aircraft rolling left 90 degrees
    yaw, pitch, roll = 0., 0., -np.pi/2.
    attitude = rotations.euler_to_quat(yaw, pitch, roll)

    u, v, w = np.random.randn(3)
    state = VehicleState(attitude=attitude)

    u_in, v_in, w_in = state.body_to_inertial([u, v, w])
    assert np.isclose(u_in, u)
    assert np.isclose(v_in, w)
    assert np.isclose(w_in, -v)

@pytest.mark.parametrize('shape', [(3,), (3,1), (3,2), (3,3), (3,4)])
def test_multiple_attitudes_known(shape):
    '''
    Test that state rotations handle different shaped inputs appropriately.
    If inputs are not broadcastable to the number of states, then throw an
    exception. Otherwise, if the number of inputs is the same as the number of
    states rotate each input separately; or if the there is only one input
    rotate it separately by each attitude.
    '''
    yaw = [np.pi/2., 0., 0., 2.*np.pi*(np.random.rand(1) - .5)]
    pitch = [0., np.pi/4., 0., np.pi*(np.random.rand(1) - .5)]
    roll = [0., 0., -np.pi/2., 2.*np.pi*(np.random.rand(1) - .5)]
    yaw = np.array(yaw, dtype=float)
    pitch = np.array(pitch, dtype=float)
    roll = np.array(roll, dtype=float)
    attitude = rotations.euler_to_quat(yaw, pitch, roll)
    state = VehicleState(attitude=attitude)

    u, v, w = np.random.randn(*shape)
    if u.ndim > 0 and u.shape[0] not in (1, len(yaw)):
        with pytest.raises(ValueError):
            _ = state.body_to_inertial([u, v, w])
    else:
        u_in, v_in, w_in = state.body_to_inertial([u, v, w])

        if u.ndim > 0 and u.shape[0] == len(yaw):
            assert np.allclose(
                u_in[:-1], [-v[0], np.sqrt(1/2)*(u[1] + w[1]), u[2]]
            )
            assert np.allclose(
                v_in[:-1], [u[0], v[1], w[2]]
            )
            assert np.allclose(
                w_in[:-1], [w[0], np.sqrt(1/2)*(w[1] - u[1]), -v[2]]
            )
            assert np.allclose(
                [u_in[-1], v_in[-1], w_in[-1]],
                state.rotation()[-1].apply([u[-1], v[-1], w[-1]])
            )

@pytest.mark.parametrize('shape', [(3,), (3,1), (3,2)])
@pytest.mark.parametrize('fun', ['body_to_inertial', 'inertial_to_body'])
def test_multiple_attitudes_rand(shape, fun):
    '''
    Test that state rotations handle multiple rotations appropriately.
    '''
    yaw, pitch, roll = 2.*np.pi*(np.random.rand(*shape) - .5)
    attitude = rotations.euler_to_quat(yaw, pitch/2., roll)
    state = VehicleState(attitude=attitude)

    u, v, w = np.random.randn(*shape)
    u_out, v_out, w_out = getattr(state, fun)([u, v, w])

    u = u.reshape(-1)
    v = v.reshape(-1)
    w = w.reshape(-1)
    u_out = u_out.reshape(-1)
    v_out = v_out.reshape(-1)
    w_out = w_out.reshape(-1)
    for i in range(u.shape[0]):
        u_exp, v_exp, w_exp = getattr(state, fun)(
            np.tile(np.atleast_2d([u[i], v[i], w[i]]), (u.shape[0],1)).T
        )
        assert np.isclose(u_out[i], u_exp[i])
        assert np.isclose(v_out[i], v_exp[i])
        assert np.isclose(w_out[i], w_exp[i])

def test_course_angle():
    # Aircraft pointed northwest with due west projected velocity
    yaw, pitch, roll = -np.pi/4., 0., 0.
    u, v, w = 1., -1., np.random.randn(1)
    attitude = rotations.euler_to_quat(yaw, pitch, roll)
    state = VehicleState(attitude=attitude, u=u, v=v, w=w)

    u_in, v_in, w_in = state.body_to_inertial([state.u, state.v, state.w])

    assert state.chi is None
    assert np.isclose(u_in, 0.)
    assert np.isclose(v_in, -np.sqrt(u**2 + v**2))
    assert np.isclose(w_in, w)
    assert np.isclose(state.course(), -np.pi/2.)
