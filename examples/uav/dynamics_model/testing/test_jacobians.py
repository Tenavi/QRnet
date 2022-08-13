import pytest

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from .. import constants, dynamics, jacobians, optimal_controls
from ..containers import VehicleState, Controls
from ..rotations import euler_to_quat

def _array_to_container(X, U):
    if isinstance(X, VehicleState):
        states = X
    else:
        states = VehicleState(X)

    if isinstance(U, Controls):
        controls = U
    else:
        controls = Controls(U)

    return states, controls

def dynamics_wrapper(X, U):
    states, controls = _array_to_container(X, U)

    dXdt = dynamics.dynamics(states, controls)

    if not isinstance(X, VehicleState):
        dXdt = dXdt.as_array().reshape(X.shape)

    return dXdt

def _make_test_states(n_points=1, seed=None):
    np.random.seed(seed)

    test_states = dict(zip(
        ['pd', 'u', 'v', 'w', 'p', 'q', 'r'], np.random.randn(7,n_points)
    ))
    yaw, pitch, roll = 2.*np.pi*(np.random.rand(3,n_points) - .5)
    test_states['attitude'] = euler_to_quat(yaw, pitch/2., roll)

    test_controls = dict(zip(
        ['aileron', 'elevator', 'rudder'],
        np.deg2rad(np.random.randn(3,n_points))
    ))
    test_controls['throttle'] = np.random.rand(1,n_points)
    test_states = VehicleState(**test_states)
    test_controls = Controls(**test_controls)
    return test_states, test_controls

@pytest.mark.parametrize('shape', [(),(1,),(3,)])
def test_prop_jac_states(shape):
    Vb = 10.*np.random.rand(3,*shape).reshape(3,-1)
    throttle = np.random.rand(1,*shape).reshape(1,-1)

    d_forces = jacobians.prop_jac_states(*Vb)

    assert d_forces.shape == (3,11,Vb.shape[1])

    for k in range(d_forces.shape[-1]):
        def simple_prop(Vb):
            u, v, w = Vb
            Va = np.sqrt(u**2 + v**2 + w**2).reshape(1,-1)
            forces, moments = dynamics.prop_forces(Va, throttle[:,k:k+1])
            return forces.flatten()

        expected_jac = approx_derivative(simple_prop, Vb[:,k])
        assert np.allclose(d_forces[:,1:4,k], expected_jac)

@pytest.mark.parametrize('shape', [(),(1,),(3,)])
def test_prop_jac_controls(shape):
    Va = 20.*np.random.rand(1,*shape).reshape(1,-1)
    throttle = np.random.rand(1,*shape).reshape(1,-1)

    d_forces, d_moments = jacobians.prop_jac_controls(Va)

    assert d_forces.shape == d_moments.shape == (3,Va.shape[1])
    assert np.allclose(d_forces[1:], 0.)
    assert np.allclose(d_moments[1:], 0.)

    for k in range(d_forces.shape[-1]):
        def simple_prop(throttle):
            throttle = np.atleast_2d(throttle)
            return dynamics.prop_forces(Va[:,k:k+1], throttle)

        def prop_forces(throttle):
            forces, moments = simple_prop(throttle)
            return forces.flatten()

        def prop_moments(throttle):
            forces, moments = simple_prop(throttle)
            return moments.flatten()

        expected_jac = approx_derivative(prop_forces, throttle[0,k])
        assert np.allclose(d_forces[:,k:k+1], expected_jac)

        expected_jac = approx_derivative(prop_moments, throttle[0,k])
        assert np.allclose(d_moments[:,k:k+1], expected_jac)

@pytest.mark.parametrize('shape', [(),(1,),(1000,)])
def test_blending_jac(shape):
    if len(shape) == 0:
        n_points = 1
    else:
        n_points = shape[0]

    if n_points == 1:
        alpha = 3. * constants.alpha0 * (np.random.rand(1, *shape) - 0.5)
    else:
        alpha = constants.alpha0 * np.linspace(-1.5, 1.5, n_points)
        alpha = np.atleast_2d(alpha)

    expected_jac = np.diag(approx_derivative(
        dynamics.blending_fun, alpha.flatten()
    ))
    _, jac = dynamics.blending_fun(alpha, jac=True)

    assert np.allclose(jac, expected_jac)

@pytest.mark.parametrize('shape', [(),(1,),(100,),(1,1),(1,100)])
def test_coefs_alpha_jac(shape):
    if len(shape) == 0:
        n_points = 1
    else:
        n_points = shape[0]

    alpha = np.pi * 2. * (np.random.rand(*shape) - 0.5)

    alpha = np.reshape(alpha, shape)

    def CL(alpha):
        return dynamics.coefs_alpha(alpha)[0]
    def CD(alpha):
        return dynamics.coefs_alpha(alpha)[1]
    def CM(alpha):
        return dynamics.coefs_alpha(alpha)[2]

    _, _, _, d_CL, d_CD, d_CM = dynamics.coefs_alpha(alpha, jac=True)

    if shape == ():
        shape = (1,)

    d_CL_expected = np.diag(approx_derivative(CL, alpha.flatten())).reshape(shape)
    d_CD_expected = np.diag(approx_derivative(CD, alpha.flatten())).reshape(shape)
    d_CM_expected = np.diag(approx_derivative(CM, alpha.flatten())).reshape(shape)

    assert d_CL.shape == d_CD.shape == d_CM.shape == shape

    assert np.allclose(d_CL, d_CL_expected)
    assert np.allclose(d_CD, d_CD_expected)
    assert np.allclose(d_CM, d_CM_expected)

@pytest.mark.parametrize('shape', [(),(1,),(3,)])
def test_aero_jac_states(shape):
    if len(shape) == 0:
        n_points = 1
    else:
        n_points = shape[0]
    states, controls = _make_test_states(n_points)

    d_forces, d_moments = jacobians.aero_jac_states(states, controls)

    assert d_forces.shape == d_moments.shape == (3,11,n_points)

    X = states.as_array().reshape(-1,n_points)
    U = controls.as_array().reshape(-1,n_points)

    for k in range(n_points):
        controls = Controls(U[:,k])
        def aero_wrapper(Xk):
            states = VehicleState(Xk)
            return dynamics.aero_forces(states, controls)

        def aero_forces(Xk):
            forces, moments = aero_wrapper(Xk)
            return forces.flatten()

        def aero_moments(Xk):
            forces, moments = aero_wrapper(Xk)
            return moments.flatten()

        expected_jac = approx_derivative(aero_forces, X[:,k])
        assert np.allclose(d_forces[...,k], expected_jac)

        expected_jac = approx_derivative(aero_moments, X[:,k])
        assert np.allclose(d_moments[...,k], expected_jac)

@pytest.mark.parametrize('shape', [(),(1,),(3,)])
def test_aero_jac_controls(shape):
    if len(shape) == 0:
        n_points = 1
    else:
        n_points = shape[0]
    states, controls = _make_test_states(n_points)

    d_forces, d_moments = jacobians.aero_jac_controls(states)

    assert d_forces.shape == d_moments.shape == (3,4,n_points)

    X = states.as_array().reshape(-1,n_points)
    U = controls.as_array().reshape(-1,n_points)

    for k in range(n_points):
        states = VehicleState(X[:,k])
        def aero_wrapper(Uk):
            controls = Controls(Uk)
            return dynamics.aero_forces(states, controls)

        def aero_forces(Uk):
            forces, moments = aero_wrapper(Uk)
            return forces.flatten()

        def aero_moments(Uk):
            forces, moments = aero_wrapper(Uk)
            return moments.flatten()

        expected_jac = approx_derivative(aero_forces, U[:,k])
        assert np.allclose(d_forces[...,k], expected_jac)

        expected_jac = approx_derivative(aero_moments, U[:,k])
        assert np.allclose(d_moments[...,k], expected_jac)

@pytest.mark.parametrize('shape', [(),(1,),(3,)])
def test_jac_states(shape):
    if len(shape) == 0:
        n_points = 1
    else:
        n_points = shape[0]

    states, controls = _make_test_states(n_points, seed=123)
    X = states.as_array().reshape(-1,n_points)
    U = controls.as_array().reshape(-1,n_points)

    dFdX = jacobians.jac_states(states, controls)

    assert dFdX.shape == (X.shape[0], X.shape[0], n_points)

    for k in range(n_points):
        dXdt0 = dynamics_wrapper(X[:,k], U[:,k])

        expected_jac = approx_derivative(
            lambda X: dynamics_wrapper(X, U[:,k]), X[:,k], f0=dXdt0
        )

        assert np.allclose(dFdX[...,k], expected_jac)

@pytest.mark.parametrize('shape', [(),(1,),(3,)])
def test_jac_controls(shape):
    if len(shape) == 0:
        n_points = 1
    else:
        n_points = shape[0]

    states, controls = _make_test_states(n_points, seed=123)
    X = states.as_array().reshape(-1,n_points)
    U = controls.as_array().reshape(-1,n_points)

    dFdU = jacobians.jac_controls(states, controls)

    assert dFdU.shape == (X.shape[0], U.shape[0], n_points)

    for k in range(n_points):
        expected_jac = approx_derivative(
            lambda U: dynamics_wrapper(X[:,k], U), U[:,k]
        )
        assert np.allclose(dFdU[...,k], expected_jac)

# ---------------------------------------------------------------------------- #

@pytest.mark.parametrize('shape', [(),(1,),(3,)])
@pytest.mark.parametrize('saturated', [False, True])
def test_opt_control_jac(shape, saturated):
    if len(shape) == 0:
        n_points = 1
    else:
        n_points = shape[0]

    states, _ = _make_test_states(n_points, seed=123)
    X = states.as_array().reshape(-1,n_points)
    dVdX = np.random.randn(*X.shape)

    if saturated:
        dVdX *= 100.

    trim_controls = np.reshape([0.5,0.,0.,0.], (-1,1))
    R_inv = np.diag([1e-02, 1e-01, 1e-01, 1e-01])

    U = optimal_controls.control(X, dVdX, R_inv, trim_controls)
    dUdX = optimal_controls.jacobian(X, dVdX, R_inv, U)
    _, dUdX2 = optimal_controls.controls_and_jac(X, dVdX, R_inv, trim_controls)

    assert dUdX.shape == dUdX2.shape == (4,11,n_points)

    for k in range(n_points):
        def opt_ctrl(X):
            U = optimal_controls.control(X, dVdX[:,k], R_inv, trim_controls)
            return U.as_array().flatten()

        expected_jac = approx_derivative(opt_ctrl, X[:,k])

        assert np.allclose(dUdX[...,k], expected_jac)
        assert np.allclose(dUdX2[...,k], expected_jac)
