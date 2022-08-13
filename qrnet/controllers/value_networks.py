import numpy as np
import tensorflow as tf

from .base import BaseNN
from .utilities import make_dense_graph, tf_jacobian
from .optimize import optimize

class ValueNN(BaseNN):
    '''
    A NN for approximating the value function of an infinite horizon OCP.
    '''
    def __init__(
            self, LQR, n_hidden=None, n_neurons=None, n_out=None,
            activation='tanh', U_star_fun=None, scaling={}, parameters={}
        ):
        '''
        Build the computational graph for the NN and its loss functions. If
        scaling has been pre-computed elsewhere and supplied, uses this.
        Otherwise if data is supplied, computes scaling parameters
        appropriately. Initializes NN parameters.

        Parameters
        ----------
        LQR : object
            Instance of controllers.linear_quadratic_regulator.LQR.
        n_hidden : int
            Number of hidden layers. Required if weights is None.
        n_neurons : int
            Number of neurons per hidden layer. Required if weights is None.
        n_out : int
            Number of output neurons. Required if weights is None.
        activation : str, default='tanh'
            Activation function to use for hidden layers.
        U_star_fun : callable
            Function which evaluates the optimal control based on the state and
            value gradient. Takes two tensor arguments and outputs a tensor.
            Required for value_networks and gradient_networks.
        scaling : dict, optional
            Dict of arrays specifying scaling for inputs and outputs containing
            X_lb : (n_states, 1) array
                Lower bound of input data
            X_ub : (n_states, 1) array
                Upper bound of input data
            U_lb : (n_controls, 1) array
                Lower bound of control data
            U_ub : (n_controls, 1) array
                Upper bound of control data
            dVdX_lb : (n_states, 1) array
                Lower bound of gradient/costate data
            dVdX_ub : (n_states, 1) array
                Upper bound of gradient/costate data
            V_ub : float
                Upper bound of value function data
        parameters : dict, optional
            Dict containing a list of pre-trained weights and biases for each
            layer under the key 'weights', arranged as
            [weights(layer1), biases(layer1), weights(layer2), biases(layer2), ...]
        '''
        super().__init__(
            LQR,
            n_hidden=n_hidden,
            n_neurons=n_neurons,
            n_out=1,
            activation=activation,
            U_star_fun=U_star_fun,
            scaling=scaling,
            parameters=parameters
        )

    def _setup_scaling(self, scaling=None, data=None):
        '''
        Setup input and output scaling parameters for the network. If scaling
        has been pre-computed elsewhere and supplied, uses this. Otherwise if
        data is supplied, computes scaling parameters appropriately.

        Parameters
        ----------
        scaling : dict, optional, containing (a subset of)
            X_lb : (n_states, 1) array
                Lower bound of input data
            X_ub : (n_states, 1) array
                Upper bound of input data
            U_lb : (n_controls, 1) array
                Lower bound of control data
            U_ub : (n_controls, 1) array
                Upper bound of control data
            dVdX_lb : (n_states, 1) array
                Lower bound of gradient/costate data
            dVdX_ub : (n_states, 1) array
                Upper bound of gradient/costate data
            V_ub : float
                Upper bound of value function data
        data : dict, optional, containing
            X : (n_states, n_data) array
                Input state data
            U : (n_controls, n_data) array
                Optimal control data
            dVdX : (n_states, n_data) array
                Value function gradient data
            V : (1, n_data) array
                Value function data

        Returns
        ----------
        scaling : dict
            Dictionary of arrays specifying scaling for inputs and outputs. May
            not have meaningful contents if success_flag is False
        success_flag : bool
            True if scaling was initialized successfully
        '''
        return super()._setup_scaling(
            ['X', 'U', 'dVdX', 'V'], scaling=scaling, data=data
        )

    def _build_graph(self):
        '''
        Build the computational graph for the NN and its loss functions.
        '''
        n_states = self.LQR.n_states
        n_controls = self.LQR.n_controls

        # Builds computational graph
        self.X_tf = tf.placeholder(tf.float32, shape=(n_states, None))
        self.V_tf = tf.placeholder(tf.float32, shape=(1, None))
        self.dVdX_tf = tf.placeholder(tf.float32, shape=(n_states, None))
        self.U_tf = tf.placeholder(tf.float32, shape=(n_controls, None))

        self.V_scaled_tf = tf.placeholder(tf.float32, shape=(1, None))
        self.dVdX_scaled_tf = tf.placeholder(tf.float32, shape=(n_states, None))
        self.U_scaled_tf = tf.placeholder(tf.float32, shape=(n_controls, None))

        self.gradient_loss_weight = tf.placeholder(tf.float32)
        self.value_loss_weight = tf.placeholder(tf.float32)

        V_scaled_pred, self.V_pred = self._make_eval_graph(self.X_tf)

        self.dVdX_pred = tf.gradients(self.V_pred, self.X_tf)[0]
        dVdX_scaled_pred = self.dVdX_scale*(self.dVdX_pred - self.dVdX_lb) - 1.

        self.U_pred = self.U_star_fun(self.X_tf, self.dVdX_pred)
        self.dUdX = tf_jacobian(self.U_pred, self.X_tf)
        U_scaled_pred = self.U_scale*(self.U_pred - self.U_lb) - 1.

        # Value function loss using scaled data
        self.loss_V = tf.reduce_mean((V_scaled_pred - self.V_scaled_tf)**2)

        # Value gradient loss using scaled data
        self.loss_dVdX = tf.reduce_mean(
            tf.reduce_sum((dVdX_scaled_pred - self.dVdX_scaled_tf)**2, axis=0)
        )

        # Control loss using scaled data
        self.loss_U = tf.reduce_mean(
            tf.reduce_sum((U_scaled_pred - self.U_scaled_tf)**2, axis=0)
        )

        self.loss = (
            self.loss_U
            + self.value_loss_weight * self.loss_V
            + self.gradient_loss_weight * self.loss_dVdX
        )

    def _make_eval_graph(self, X):
        '''
        Helper function which builds a dense NN and transforms the output to
        make the prediction tensor operations.

        Arguments
        ----------
        X : (n_states, n_data) tensor
            State locations to make predictions for

        Returns
        ----------
        V_scaled : (1, n_data) tensor
            Linearly scaled value function predictions for each state
        V : (1, n_data) tensor
            Value function predictions for each state in original domain
        '''

        # Raw NN prediction in the scaled domain
        V_scaled = make_dense_graph(
            X - self.LQR.X_bar,
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        V = self.V_ub*(V_scaled + 1.)/2.

        return V_scaled, V

    def bvp_guess(self, X):
        '''
        Predicts the value function V(X), its gradient dVdX(X), and the optimal
        control U(X) for each sample state in X. If the network does not make
        predictions for a quantity, then return the LQR approximation.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to make predictions for.

        Returns
        ----------
        V : (1, n_data) or (1,) array
            Value function prediction for each column in X.
        dVdX : (n_states, n_data) or (n_states,) array
            Value gradient prediction for each column in X.
        U : (n_controls, n_data) or (n_controls,) array
            NN feedback control for each column in X.
        '''
        V, dVdX, U = self.sess.run(
            (self.V_pred, self.dVdX_pred, self.U_pred),
            {self.X_tf: X.reshape(X.shape[0], -1)}
        )
        if X.ndim < 2:
            V = V.flatten()
            U = U.flatten()
        return V, dVdX.reshape(X.shape), U.astype(np.float64)

    def train(
            self, data, optimizer='L-BFGS-B', optimizer_opts={},
            batch_size=None, n_epochs=1, callback_epoch=1,
            gradient_loss_weight=1., value_loss_weight=1.,
            var_to_bounds={}, **kwargs
        ):
        '''
        Optimize the training loss using limited memory BFGS.

        Parameters
        ----------
        data : dict
            Dict of open loop optimal control data containing
            X : (n_states, n_data) array
                Input state data
            U : (n_controls, n_data) array
                Optimal control data
            dVdX : (n_states, n_data) array
                Value function gradient data
            V : (1, n_data) array
                Value function data
        optimizer_opts : dict, optional
            Dict of options to pass to the optimizer.
        batch_size : int, optional
            Number of data points (not trajectories) to use for SGD optimizers.
            If used in conjunction with 'L-BFGS-B', sets the maximum number of
            data to use.
        n_epochs : int, default=1
            How many times to iterate through the dataset (for SGD optimizers).
        callback_epoch : int, default=1
            Specifies after how many epochs to print loss functions (for SGD
            optimizers).
        gradient_loss_weight : float, default=1.
            Scalar multiplier in front of the value gradient mean square loss
            term. Not used by control_networks.
        value_loss_weight : float, default=1.
            Scalar multiplier in front of the value function mean square loss
            term. Not used by control_networks and gradient_networks.
        var_to_bounds : dict, optional, for use by subclasses
            Maps variables to lower and upper bounds: keys are tensors and
            values are tuples of arrays
        '''
        self._build(data=data)

        U_scaled = self.U_scale*(data['U'] - self.U_lb) - 1.
        dVdX_scaled = self.dVdX_scale*(data['dVdX'] - self.dVdX_lb) - 1.
        V_scaled = 2.*data['V']/self.V_ub - 1.

        print(
            '\nGradient loss weight = %1.1e, value loss weight %1.1e'
            % (gradient_loss_weight, value_loss_weight)
        )

        def callback(feed_dict, session):
            losses = (self.loss_U, self.loss_dVdX, self.loss_V)
            print(
                '\nloss_U = %1.2e, loss_dVdX = %1.2e, loss_V = %1.2e'
                % session.run(losses, feed_dict)
            )

        optimize(
            self.loss, self.sess,
            [self.X_tf, self.U_scaled_tf, self.dVdX_scaled_tf, self.V_scaled_tf],
            [data['X'], U_scaled, dVdX_scaled, V_scaled],
            parameter_keys=[self.gradient_loss_weight, self.value_loss_weight],
            parameter_vals=[gradient_loss_weight, value_loss_weight],
            batch_size=batch_size,
            n_epochs=n_epochs,
            optimizer=optimizer,
            options=optimizer_opts,
            callback=callback,
            callback_epoch=callback_epoch
        )

# ---------------------------------------------------------------------------- #

class ValueQRnet(ValueNN):
    def __init__(
            self, LQR, n_hidden=None, n_neurons=None, n_out=None,
            activation='tanh', U_star_fun=None, scaling={}, parameters={}
        ):
        '''
        Build the computational graph for the NN and its loss functions. If
        scaling has been pre-computed elsewhere and supplied, uses this.
        Otherwise if data is supplied, computes scaling parameters
        appropriately. Initializes NN parameters.

        Parameters
        ----------
        LQR : object
            Instance of controllers.linear_quadratic_regulator.LQR.
        n_hidden : int
            Number of hidden layers. Required if weights is None.
        n_neurons : int
            Number of neurons per hidden layer. Required if weights is None.
        n_out : int
            Number of output neurons. Required if weights is None.
        activation : str, default='tanh'
            Activation function to use for hidden layers.
        U_star_fun : callable
            Function which evaluates the optimal control based on the state and
            value gradient. Takes two tensor arguments and outputs a tensor.
            Required for value_networks and gradient_networks.
        scaling : dict, optional
            Dict of arrays specifying scaling for inputs and outputs containing
            X_lb : (n_states, 1) array
                Lower bound of input data
            X_ub : (n_states, 1) array
                Upper bound of input data
            U_lb : (n_controls, 1) array
                Lower bound of control data
            U_ub : (n_controls, 1) array
                Upper bound of control data
            dVdX_lb : (n_states, 1) array
                Lower bound of gradient/costate data
            dVdX_ub : (n_states, 1) array
                Upper bound of gradient/costate data
            V_ub : float
                Upper bound of value function data
        parameters : dict, optional
            Dict containing a list of pre-trained weights and biases for each
            layer under the key 'weights', arranged as
            [weights(layer1), biases(layer1), weights(layer2), biases(layer2), ...]
        '''
        gamma = parameters.get('gamma')
        self.initialized_gamma = gamma is not None
        if not self.initialized_gamma:
            gamma = 1.

        self.gamma = tf.Variable(gamma, dtype=tf.float32)

        super().__init__(
            LQR,
            n_hidden=n_hidden,
            n_neurons=n_neurons,
            n_out=1,
            activation=activation,
            U_star_fun=U_star_fun,
            scaling=scaling,
            parameters=parameters
        )

    def _get_learned_params(self):
        '''
        Get numpy values for the trainable tensorflow variables including NN
        weights and the scalar gamma.

        Parameters
        ----------
        parameters : dict
            Dictionary of (lists of) arrays giving trained parameter values
        '''
        return self.sess.run({'weights': self.weights, 'gamma': self.gamma})

    def _make_eval_graph(self, X):
        '''
        Helper function which builds a dense NN and transforms the output to
        make the prediction tensor operations.

        Parameters
        ----------
        X : (n_states, n_data) tensor
            State locations to make predictions for

        Returns
        -------
        V_scaled : (1, n_data) tensor
            Linearly scaled value function predictions for each state
        V : (1, n_data) tensor
            Value function predictions for each state in original domain
        '''
        if not self.initialized_gamma:
            gamma_init = tf.assign(self.gamma, 2./np.squeeze(self.V_ub))
            self.sess.run(gamma_init)
            self.initialized_gamma = True

        X_err = X - self.LQR.X_bar

        # Raw NN prediction in the scaled domain
        V_scaled = make_dense_graph(
            X - self.LQR.X_bar,
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        # LQR component
        XPX = X_err * tf.matmul(self.LQR.P.astype(np.float32), X_err)
        XPX = tf.reduce_sum(XPX, axis=0, keepdims=True)
        XPX = 2./(self.V_ub * self.gamma) * tf.log(1. + self.gamma * XPX)

        V_scaled = V_scaled + XPX

        V = self.V_ub*(V_scaled + 1.)/2.

        return V_scaled, V

    def train(
            self, data, optimizer='L-BFGS-B', optimizer_opts={},
            batch_size=None, n_epochs=1, callback_epoch=1,
            gradient_loss_weight=1., value_loss_weight=1.,
            var_to_bounds={}, **kwargs
        ):
        '''
        Optimize the training loss using limited memory BFGS.

        Parameters
        ----------
        data : dict
            Dict of open loop optimal control data containing
            X : (n_states, n_data) array
                Input state data
            U : (n_controls, n_data) array
                Optimal control data
            dVdX : (n_states, n_data) array
                Value function gradient data
            V : (1, n_data) array
                Value function data
        optimizer_opts : dict, optional
            Dict of options to pass to the optimizer.
        batch_size : int, optional
            Number of data points (not trajectories) to use for SGD optimizers.
            If used in conjunction with 'L-BFGS-B', sets the maximum number of
            data to use.
        n_epochs : int, default=1
            How many times to iterate through the dataset (for SGD optimizers).
        callback_epoch : int, default=1
            Specifies after how many epochs to print loss functions (for SGD
            optimizers).
        gradient_loss_weight : float, default=1.
            Scalar multiplier in front of the value gradient mean square loss
            term. Not used by control_networks.
        value_loss_weight : float, default=1.
            Scalar multiplier in front of the value function mean square loss
            term. Not used by control_networks and gradient_networks.
        var_to_bounds : dict, optional, for use by subclasses
            Maps variables to lower and upper bounds: keys are tensors and
            values are tuples of arrays
        '''
        super().train(
            data,
            optimizer=optimizer,
            optimizer_opts=optimizer_opts,
            batch_size=batch_size,
            n_epochs=n_epochs,
            callback_epoch=callback_epoch,
            gradient_loss_weight=gradient_loss_weight,
            value_loss_weight=value_loss_weight,
            var_to_bounds={self.gamma: (1e-06, 1e+06)},
            **kwargs
        )
