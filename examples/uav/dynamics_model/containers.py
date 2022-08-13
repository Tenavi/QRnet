import numpy as np
from scipy.spatial.transform import Rotation

STATES_ORDER = ('pd', 'u', 'v', 'w', 'p', 'q', 'r', 'attitude')
CONTROLS_ORDER = ('throttle', 'aileron', 'elevator', 'rudder')

class VehicleState:
    n_states = 11

    def __init__(
            self, X=None,
            pd=0., u=0., v=0., w=0., p=0., q=0., r=0., attitude=[0.,0.,0.,1.]
        ):
        '''
        Container holding the vehicle state(s). Can be initialized from an array
        or by setting each state individually by name.

        Parameters
        ----------
        X : {(11,) array, (11, n_points) array}, optional
            State(s) as a numpy array. If provided, other inputs are ignored.
        pd : {float, (n_points,) array, (1, n_points) array}, optional
            Inertial down position (negative altitude) [m].
        u : {float, (n_points,) array, (1, n_points) array}, optional
            Ground speed in body x-axis [m/s].
        v : {float, (n_points,) array, (1, n_points) array}, optional
            Ground speed in body y-axis [m/s].
        w : {float, (n_points,) array, (1, n_points) array}, optional
            Ground speed in body z-axis [m/s].
        p : {float, (n_points,) array, (1, n_points) array}, optional
            Body roll rate about body-x axis [rad/s].
        q : {float, (n_points,) array, (1, n_points) array}, optional
            Body pitch rate about body-y axis [rad/s].
        r : {float, (n_points,) array, (1, n_points) array}, optional
            Body yaw rate about body-z axis [rad/s].
        attitude : {(4,) array, (4, n_points) array}, optional
            Quaternion of body frame attitude relative to inertial frame. The
            vector components are indices 0:3 and the scalar quaternion is in
            index 3.
        '''
        self.set_state(X, pd, u, v, w, p, q, r, attitude)

    def set_state(
            self, X=None,
            pd=None, u=None, v=None, w=None,
            p=None, q=None, r=None, attitude=None
        ):
        '''
        Set (some parts of) the vehicle state(s) by an array or by name.

        Parameters
        ----------
        X : {(11,) array, (11, n_points) array}, optional
            State(s) as a numpy array. If provided, other inputs are ignored.
        pd : {float, (n_points,) array, (1, n_points) array}, optional
            Inertial down position (negative altitude) [m].
        u : {float, (n_points,) array, (1, n_points) array}, optional
            Ground speed in body x-axis [m/s].
        v : {float, (n_points,) array, (1, n_points) array}, optional
            Ground speed in body y-axis [m/s].
        w : {float, (n_points,) array, (1, n_points) array}, optional
            Ground speed in body z-axis [m/s].
        p : {float, (n_points,) array, (1, n_points) array}, optional
            Body roll rate about body-x axis [rad/s].
        q : {float, (n_points,) array, (1, n_points) array}, optional
            Body pitch rate about body-y axis [rad/s].
        r : {float, (n_points,) array, (1, n_points) array}, optional
            Body yaw rate about body-z axis [rad/s].
        attitude : {(4,) array, (4, n_points) array}, optional
            Quaternion of body frame attitude relative to inertial frame. The
            vector components are indices 0:3 and the scalar quaternion is in
            index 3.
        '''
        # Initialize from numpy array
        if X is not None:
            if not X.shape[0] == self.n_states:
                err_msg = 'Tried to set the state with an array X of shape '
                err_msg += str(X.shape) + ', but X.shape[0] must be '
                err_msg += self.n_states + '.'
                raise ValueError(err_msg)
            pd, u, v, w, p, q, r = X[:self.n_states-4]
            attitude = X[self.n_states-4:]

        # Altitude
        if pd is not None:
            self.pd = np.atleast_2d(pd)
        # Body frame velocities
        if u is not None:
            self.u = np.atleast_2d(u)
        if v is not None:
            self.v = np.atleast_2d(v)
        if w is not None:
            self.w = np.atleast_2d(w)
        # Body frame angular rates
        if p is not None:
            self.p = np.atleast_2d(p)
        if q is not None:
            self.q = np.atleast_2d(q)
        if r is not None:
            self.r = np.atleast_2d(r)
        # Quaternion attitude representation
        if attitude is not None:
            self.attitude = np.reshape(attitude, (4,-1))
            self._rotation = None

        self.Va, self.alpha, self.beta, self.chi = None, None, None, None

        _fix_shapes(self, STATES_ORDER, 'state')

    def as_array(self):
        X = np.vstack([getattr(self, var) for var in STATES_ORDER])
        return np.squeeze(X)

    def rotation(self):
        if not isinstance(self._rotation, Rotation):
            self._rotation = Rotation(
                np.squeeze(self.attitude.T), normalize=False, copy=False
            )
        return self._rotation

    def inertial_to_body(self, X_inertial):
        # The scipy Rotation class treats rotation as expressing a rotated
        # inertial vector in the inertial frame, rather than in the body frame.
        # Thus we implement the transformation using inverse=True.
        X_body = self.rotation().apply(np.squeeze(X_inertial).T, inverse=True).T
        # Make the output shape the same as the input, if it didn't increase in
        # size due to multiple rotations.
        if np.ndim(X_inertial) == 1:
            X_body = np.squeeze(X_body)
        else:
            X_body = X_body.reshape(3,-1)
        return X_body

    def body_to_inertial(self, X_body):
        # The scipy Rotation class treats rotation as expressing a rotated
        # inertial vector in the inertial frame, rather than in the body frame.
        # Thus we implement the inverse transformation using inverse=False.
        X_inertial = self.rotation().apply(np.squeeze(X_body).T).T
        # Make the output shape the same as the input, if it didn't increase in
        # size due to multiple rotations.
        if np.ndim(X_body) == 1:
            X_inertial = np.squeeze(X_inertial)
        else:
            X_inertial = X_inertial.reshape(3,-1)
        return X_inertial

    def airspeed(self):
        '''
        Get current airspeed magnitude, angle of attack, and sideslip.

        Returns
        -------
        Va : float or (n_points,) array
            Airspeed [m/s] for each state contained in the input.
        alpha : float or (n_points,) array
            Angle of attack [rad] for each state contained in the input.
        beta : float or (n_points,) array
            Sideslip [rad] for each state contained in the input.
        '''
        if any([self.Va is None, self.alpha is None, self.beta is None]):
            # Airspeed and flight angles (assume wind is zero)
            self.Va = np.sqrt(self.u**2 + self.v**2 + self.w**2)
            self.alpha = np.arctan2(self.w, self.u)
            self.beta = np.zeros_like(self.Va)
            idx = ~np.isclose(self.Va, 0.)
            self.beta[idx] = np.arcsin(self.v[idx] / self.Va[idx])

        return self.Va, self.alpha, self.beta

    def course(self):
        '''
        Get the current course angle.

        Returns
        -------
        chi : float or (n_points,) array
            Course angle [rad] for each state.
        '''
        if self.chi is None:
            # Course angle computed by rotating body velocity into NED
            vel = self.body_to_inertial([self.u, self.v, self.w])
            self.chi = np.arctan2(vel[1], vel[0])

        return self.chi

class Controls:
    n_controls = 4

    def __init__(self, U=None, throttle=0., aileron=0., elevator=0., rudder=0.):
        '''
        Container holding the vehicle controls(s). Can be initialized from an
        array or by setting each input individually by name.

        Parameters
        ----------
        U : {(4,) array, (4, n_points) array}, optional
            Control(s) as a numpy array. If provided, other inputs are ignored.
        throttle : {float, (n_points,) array, (1, n_points) array}, optional
            Throttle setting (increases motor speed).
        aileron : {float, (n_points,) array, (1, n_points) array}, optional
            Aileron position [rad].
        elevator : {float, (n_points,) array, (1, n_points) array}, optional
            Elevator position [rad].
        rudder : {float, (n_points,) array, (1, n_points) array}, optional
            Rudder position [rad].
        '''
        self.set_controls(U, throttle, aileron, elevator, rudder)

    def set_controls(
            self, U=None,
            throttle=None, aileron=None, elevator=None, rudder=None
        ):
        '''
        Set (some parts of) the control inputs by an array or by name.

        Parameters
        ----------
        U : {(4,) array, (4, n_points) array}, optional
            Control(s) as a numpy array. If provided, other inputs are ignored.
        throttle : {float, (n_points,) array, (1, n_points) array}, optional
            Throttle setting (increases motor speed).
        aileron : {float, (n_points,) array, (1, n_points) array}, optional
            Aileron position [rad].
        elevator : {float, (n_points,) array, (1, n_points) array}, optional
            Elevator position [rad].
        rudder : {float, (n_points,) array, (1, n_points) array}, optional
            Rudder position [rad].
        '''

        # Initialize from numpy array
        if U is not None:
            throttle, aileron, elevator, rudder = U

        if throttle is not None:
            self.throttle = np.atleast_2d(throttle)
        if aileron is not None:
            self.aileron = np.atleast_2d(aileron)
        if elevator is not None:
            self.elevator = np.atleast_2d(elevator)
        if rudder is not None:
            self.rudder = np.atleast_2d(rudder)

        _fix_shapes(self, CONTROLS_ORDER, 'control')

    def as_array(self):
        U = np.vstack([getattr(self, var) for var in CONTROLS_ORDER])
        return np.squeeze(U)

def _fix_shapes(container, order, attr_name):
    '''
    Utility function which makes sure that all the states or controls are the
    same size, broadcasting if necessary.

    Raises
    ------
    ValueError
        If the shapes of each attribute are not mutually broadcastable.
    '''
    shapes = [getattr(container, var).shape[1] for var in order]

    shapes, idx = np.unique(shapes, return_inverse=True)

    err_msg = 'Attempted to set ' + attr_name + 's using different size arrays'
    err_msg += ' which are not mutually broadcastable.'

    if len(shapes) > 1:
        # Too many unique shapes
        if len(shapes) > 2:
            raise ValueError(err_msg)

        # Only two shapes, but neither is 1 so cannot broadcast
        if 1 not in shapes:
            raise ValueError(err_msg)

        # This uses the fact that numpy.unique returns the unique values sorted
        # from lowest (1) to highest, so shapes[0] = 1 and idx will contain 0 in
        # every spot corresponding to such an attribute.
        for i, var_name in zip(idx, order):
            if i == 0:
                var = np.tile(getattr(container, var_name), (1, shapes[1]))
                setattr(container, var_name, var)

def _make_indices(classdef, ordering):
    '''
    Make a dictionary of variable names (for VehicleState or Controls) and
    slices to retrieve their corresponding rows from the as_array() method.

    Parameters
    ----------
    classdef : {VehicleState, Controls} class definition
        Which class to make the index slicing for.
    ordering : list of strings
        List of variables for which to get slices. Normally this is STATES_ORDER
        or CONTROLS_ORDER, but could also be any subset.

    Returns
    -------
    indices : dict
        Keys are the entries of ordering and values are slices. For a string var
        in ordering, we get
        classdef().as_array()[indices[var]] == getattr(classdef(), var)
    '''
    dummy_class = classdef()
    name_length_pairs = [
        [var, getattr(dummy_class, var).shape[0]] for var in ordering
    ]
    indices = {}
    i = 0
    for pair in name_length_pairs:
        var, length = pair
        indices[var] = slice(i, i + length)
        i = i + length

    return indices

STATES_IDX = _make_indices(VehicleState, STATES_ORDER)
CONTROLS_IDX = _make_indices(Controls, CONTROLS_ORDER)
