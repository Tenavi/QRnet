import pytest

import numpy as np

from ..dynamics import dynamics
from .. import constants, containers, rotations, jacobians, compute_trim

def _array_to_container(X, U):
    if isinstance(X, containers.VehicleState):
        states = X
    else:
        states = containers.VehicleState(X)

    if isinstance(U, containers.Controls):
        controls = U
    else:
        controls = containers.Controls(U)

    return states, controls

def dynamics_wrapper(X, U):
    states, controls = _array_to_container(X, U)

    dXdt = dynamics(states, controls)

    if not isinstance(X, containers.VehicleState):
        dXdt = dXdt.as_array().reshape(X.shape)

    return dXdt

def jacobians_wrapper(X, U):
    states, controls = _array_to_container(X, U)

    dFdX = jacobians.jac_states(states, controls)
    dFdU = jacobians.jac_controls(states, controls)

    return dFdX, dFdU

def _make_test_states(n_points=1):
    test_states = dict(zip(
        ['pd', 'u', 'v', 'w', 'p', 'q', 'r'], np.random.randn(7,n_points)
    ))
    yaw, pitch, roll = 2.*np.pi*(np.random.rand(3,n_points) - .5)
    test_states['attitude'] = rotations.euler_to_quat(yaw, pitch/2., roll)

    test_controls = dict(zip(
        ['aileron', 'elevator', 'rudder'],
        np.deg2rad(np.random.randn(3,n_points))
    ))
    test_controls['throttle'] = np.random.rand(1,n_points)
    return test_states, test_controls

def test_make_indices():
    test_state_dict, _ = _make_test_states(n_points=8)
    test_states = containers.VehicleState(**test_state_dict).as_array()

    state_idx = containers.STATES_IDX
    for var_name, expected in test_state_dict.items():
        assert np.allclose(test_states[state_idx[var_name]], expected)

@pytest.mark.parametrize('n_points', [1,2])
def test_join_free_vars(n_points):
    test_states, test_controls = _make_test_states(n_points)
    test_states = containers.VehicleState(**test_states)
    test_controls = containers.Controls(**test_controls)

    free_vars = compute_trim._join_free_vars(test_states, test_controls)
    free_vars = free_vars.reshape(-1,n_points)

    for var, idx in compute_trim.FREE_STATES_IDX.items():
        assert np.allclose(free_vars[idx], getattr(test_states, var))
    for var, idx in compute_trim.FREE_CONTROLS_IDX.items():
        assert np.allclose(free_vars[idx], getattr(test_controls, var))

@pytest.mark.parametrize('n_points', [1,2])
def test_split_free_vars(n_points):
    test_free_vars = np.random.randn(compute_trim._n_free, n_points)
    states, controls = compute_trim._split_free_vars(test_free_vars)

    # Check each variable individually
    for var, idx in compute_trim.FREE_STATES_IDX.items():
        observed = getattr(states, var).flatten()
        expected = test_free_vars[idx].flatten()
        assert np.allclose(observed, expected)

    for var, idx in compute_trim.FREE_CONTROLS_IDX.items():
        observed = getattr(controls, var).flatten()
        expected = test_free_vars[idx].flatten()
        assert np.allclose(observed, expected)

    # Split and join free vars should be able to invert each other
    free_vars = compute_trim._join_free_vars(states, controls)
    assert np.allclose(free_vars.flatten(), test_free_vars.flatten())

def _assert_trim(
        state, controls, expected_state, expected_controls, tol, verbose
    ):
    if verbose:
        print('')
    for var in containers.STATES_ORDER:
        if verbose:
            if var == 'attitude':
                yaw, pitch, roll = np.squeeze(rotations.quat_to_euler(
                    state.attitude.T, degrees=True, ignore_warnings=False
                ))
                yaw_e, pitch_e, roll_e = np.squeeze(rotations.quat_to_euler(
                    expected_state.attitude.T, degrees=True, ignore_warnings=False
                ))
                print('yaw [deg]')
                print(yaw, yaw_e)
                print('pitch [deg]')
                print(pitch, pitch_e)
                print('roll [deg]')
                print(roll, roll_e)
            elif var in ['p', 'q', 'r']:
                print(var, '[deg/s]')
                print(
                    np.squeeze(np.rad2deg(getattr(state, var))),
                    np.squeeze(np.rad2deg(getattr(expected_state, var)))
                )
            else:
                print(var)
                print(
                    np.squeeze(getattr(state, var)),
                    np.squeeze(getattr(expected_state, var))
                )
        assert np.allclose(
            getattr(state, var), getattr(expected_state, var), atol=tol
        )
    for var in containers.CONTROLS_ORDER:
        if verbose:
            if var != 'throttle':
                print(var, '[deg]')
                print(
                    np.squeeze(np.rad2deg(getattr(controls, var))),
                    np.squeeze(np.rad2deg(getattr(expected_controls, var)))
                )
            else:
                print(var)
                print(
                    np.squeeze(getattr(controls, var)),
                    np.squeeze(getattr(expected_controls, var))
                )
        assert np.allclose(
            getattr(controls, var), getattr(expected_controls, var), atol=tol
        )

@pytest.mark.parametrize(
    'Va,R,gamma',
    [
        (10., np.inf, 0.),
        (30., np.inf, 0.),
        (20., np.inf, np.deg2rad(45.)),
        (20., 40., 0.)
    ]
)
def test_trim_infeasible(Va, R, gamma):
    states, controls, flag = compute_trim.compute_trim(
        dynamics_wrapper, jacobians=jacobians_wrapper,
        Va_star=Va, gamma_star=gamma, R_star=R
    )

    with pytest.raises(AssertionError):
        assert flag

def test_trim_cruise():
    Va = 20.
    verbose = 0

    expected_quaternion = rotations.euler_to_quat(
        yaw=0., pitch=5.842111667711932, roll=0., degrees=True
    )
    expected_state = containers.VehicleState(
        u=19.89612333711169,
        w=2.035749531362834,
        attitude=expected_quaternion
    )
    expected_controls = containers.Controls(
        throttle=0.5210364296160614,
        aileron=-7.133344814025444e-09,
        elevator=-0.26241179461882747,
        rudder=4.842376903477779e-09
    )

    state, controls, flag = compute_trim.compute_trim(
        dynamics_wrapper, jacobians=jacobians_wrapper,
        Va_star=Va, verbose=verbose
    )

    assert flag
    _assert_trim(
        state, controls, expected_state, expected_controls, 1e-06, verbose
    )

def test_trim_climb():
    Va = 20.
    gamma = np.deg2rad(10.)
    verbose = 0

    expected_quaternion = rotations.euler_to_quat(
        yaw=0., pitch=15.57560079881704, roll=0., degrees=True
    )
    expected_state = containers.VehicleState(
        u=19.90537733913478,
        w=1.9431811512723391,
        attitude=expected_quaternion
    )
    expected_controls = containers.Controls(
        throttle=0.8369830435080777,
        aileron=1.111820198009618e-08,
        elevator=-0.2503677906797069,
        rudder=6.891525224719433e-09
    )

    state, controls, flag = compute_trim.compute_trim(
        dynamics_wrapper, jacobians=jacobians_wrapper,
        Va_star=Va, gamma_star=gamma, verbose=verbose
    )

    assert flag
    _assert_trim(
        state, controls, expected_state, expected_controls, 1e-06, verbose
    )

def test_trim_descent():
    Va = 30.
    gamma = np.deg2rad(-30.)
    verbose = 0

    expected_quaternion = rotations.euler_to_quat(
        yaw=0., pitch=-29.262402485184953, roll=0., degrees=True
    )
    expected_state = containers.VehicleState(
        u=29.997514130615215,
        w=0.3861942303922869,
        attitude=expected_quaternion
    )
    expected_controls = containers.Controls(
        throttle=0.20912016452999105,
        aileron=-7.499258757329388e-09,
        elevator=-0.021989827955342843,
        rudder=-6.695065912044809e-09
    )

    state, controls, flag = compute_trim.compute_trim(
        dynamics_wrapper, jacobians=jacobians_wrapper,
        Va_star=Va, gamma_star=gamma, verbose=verbose
    )

    assert flag
    _assert_trim(
        state, controls, expected_state, expected_controls, 1e-06, verbose
    )

def test_trim_turn():
    Va = 20.
    R = -300.
    verbose = 0

    expected_quaternion = rotations.euler_to_quat(
        yaw=0., pitch=5.862939203356055, roll=-7.739969986842337, degrees=True
    )
    expected_state = containers.VehicleState(
        u=19.8934472824993,
        w=2.0617359720442727,
        p=np.deg2rad(0.3901808543345413),
        q=np.deg2rad(0.5117391106280813),
        r=np.deg2rad(-3.7651205064461077),
        attitude=expected_quaternion
    )
    expected_controls = containers.Controls(
        throttle=0.5215283113803285,
        aileron=0.008393752309953918,
        elevator=-0.2674128204393705,
        rudder=0.004470034172617283
    )

    state, controls, flag = compute_trim.compute_trim(
        dynamics_wrapper, jacobians=jacobians_wrapper,
        Va_star=Va, R_star=R, verbose=verbose, fun_tol=1e-02
    )

    #assert flag
    _assert_trim(
        state, controls, expected_state, expected_controls, 1e-06, verbose
    )

def test_trim_climbing_turn():
    Va = 18.
    R = 250.
    gamma = np.deg2rad(10.)
    verbose = 0

    expected_quaternion = rotations.euler_to_quat(
        yaw=0., pitch=17.414384768442392, roll=7.412717840253017, degrees=True
    )
    expected_state = containers.VehicleState(
        u=17.84698644318356,
        w=2.342023675546904,
        p=np.deg2rad(-1.2158634391067797),
        q=np.deg2rad(0.5001183350356314),
        r=np.deg2rad(3.8440172116042812),
        attitude=expected_quaternion
    )
    expected_controls = containers.Controls(
        throttle=0.7521308890638192,
        aileron=-0.012974796772894327,
        elevator=-0.336218759688996,
        rudder=-0.0029929422063816862
    )

    state, controls, flag = compute_trim.compute_trim(
        dynamics_wrapper, jacobians=jacobians_wrapper,
        Va_star=Va, R_star=R, gamma_star=gamma, verbose=verbose, fun_tol=1e-02
    )

    #assert flag
    _assert_trim(
        state, controls, expected_state, expected_controls, 1e-06, verbose
    )
